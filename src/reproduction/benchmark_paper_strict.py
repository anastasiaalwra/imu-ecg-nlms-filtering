import os
import glob
import re
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import pearsonr
import neurokit2 as nk

FS = 500.0  # PTT-PPG sampling rate 
DT = 1.0 / FS
NYQ = 0.5 * FS

# NLMS hyperparameters. Paper does not specify. We use standard NLMS values.
MU = 0.05
EPSILON = 1e-6  # ε for ||X||^2; avoid division with zero. 

# Filter design — paper Section 2.3
B_BP, A_BP   = signal.butter(4, [0.16 / NYQ, 40.0 / NYQ], btype='bandpass')
B_NOTCH, A_NOTCH = signal.iirnotch(w0=50.0, Q=10.0, fs=FS)   # bandwidth 5 Hz
B_LP, A_LP   = signal.butter(2, 0.5 / NYQ, btype='lowpass')  # baseline extractor (0–0.5 Hz)

# R-peak retention tolerance: 500 Hz / 25 samples = 50 ms 
R_PEAK_TOL = 25

# Dataset path
DATASET_DIR = os.path.join('..', '..', 'Physionet_PTT_Dataset', 'physionet.org', 'files',
                           'pulse-transit-time-ppg', '1.1.0')



def load_ecg_calibration(hea_path):
    with open(hea_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 9 and parts[-1] == 'ecg':
                # parts[2] is e.g. "0.06989430617115582(-2457)/mV"
                m = re.match(r'([-+0-9.eE]+)\(([-+0-9]+)\)/(\w+)', parts[2])
                if not m:
                    raise ValueError(f"Cannot parse gain spec in {hea_path}: {parts[2]}")
                gain= float(m.group(1))
                baseline = int(m.group(2))
                return gain, baseline
    raise ValueError(f"No ECG channel in {hea_path}")

# Function for trapezoidal integration; papers' eq (3).
def trapezoidal_integrate(x):
    v = np.zeros_like(x, dtype=float)
    # Vectorized trapezoidal integration via cumulative sum.
    v[1:] = np.cumsum((x[1:] + x[:-1]) * (DT / 2.0))
    return v

# Function for quaternion multiplication; paper eq (5)
def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

# Function to compute and extract real part of quaternion; paper eq (4), eq (6).
def compute_quaternion_real(wx_rad, wy_rad, wz_rad):
    N = len(wx_rad)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    q_real = np.empty(N, dtype=float)
    q_real[0] = 1.0
    for n in range(1, N):
        dq = np.array([1.0,
                       0.5 * wx_rad[n] * DT,
                       0.5 * wy_rad[n] * DT,
                       0.5 * wz_rad[n] * DT])
        q = quat_mul(dq, q)
        q /= np.linalg.norm(q)
        q_real[n] = q[0]
    return q_real


# Function for cross-correlation; paper eq (7), eq (8).
def cross_corr_lag(ref, target, max_lag=500):
    # normalization Z-score
    r = (ref - np.mean(ref)) / (np.std(ref) + 1e-12)
    t = (target - np.mean(target)) / (np.std(target) + 1e-12)
    
    # Checking where the 2 signals match better. 
    corr = signal.correlate(t, r, mode='full')
    lags = signal.correlation_lags(len(t), len(r), mode='full')

    # limit maximum lag in 500 samples. 
    mask = (lags >= -max_lag) & (lags <= max_lag)
    return int(lags[mask][np.argmax(corr[mask])])


def shift(sig, k):
    if k > 0:
        return np.pad(sig, (k, 0), mode='constant')[:-k]
    elif k < 0:
        return np.pad(sig, (0, -k), mode='constant')[-k:]
    return sig

# NLMS implemenatation; paper eq (10), eq (11), eq (12).
def nlms(s_obs, X):
    N, M = X.shape
    W = np.zeros(M, dtype=float)
    e = np.empty(N, dtype=float)
    for n in range(N):
        x = X[n]
        e_n = s_obs[n] - W @ x
        e[n] = e_n
        W += MU * (e_n * x) / (x @ x + EPSILON)
    return e


def sensitivity(detected, true_peaks, tol=R_PEAK_TOL):
    """R-peak retention rate"""
    if len(true_peaks) == 0:
        return 100.0 # IMPROV: maybe np.nan would work as well better. 
    if len(detected) == 0:
        return 0.0
    tp = 0
    for tp_idx in true_peaks:   # for each true peak
        distances = np.abs(detected - tp_idx) # distance from all the detected
        if np.any(distances <= tol):  # True: within the range of 50ms; False: otherwise. 
            tp += 1  
    return 100.0 * tp / len(true_peaks) # 100 * tp / tp + fn; fn: how many tp got lost.


def delta_rr_ms(detected, true_peaks, tol=R_PEAK_TOL):
    """
    ΔRR: mean absolute deviation (in ms) of RR intervals.
    Detected peaks are matched to nearest true peak within tolerance, then
    consecutive matched-pair differences are compared.
    """
    if len(detected) < 2 or len(true_peaks) < 2: # need at least 2 peaks. 
        return np.nan
    matched_det = []
    matched_true = []
    # For each true peak, find the nearest detected. 
    for tp_idx in true_peaks:
        d = np.abs(detected - tp_idx)
        # If it is within tolerance 
        if d.min() <= tol:
            matched_det.append(detected[d.argmin()]) # store in array the index of detected.
            matched_true.append(tp_idx) # store in array the index of true peak.
    if len(matched_det) < 2:
        return np.nan
    # remove duplicates (set) and sort chronically (sorted).
    md = np.array(sorted(set(matched_det))) 
    mt = np.array(sorted(set(matched_true)))
    L = min(len(md), len(mt))
    # Calculate differences of consecutive samples. 
    rr_det  = np.diff(md[:L]) / FS * 1000.0
    rr_true = np.diff(mt[:L]) / FS * 1000.0
    return float(np.mean(np.abs(rr_det - rr_true)))


def evaluate_record(csv_path, hea_path):
    df = pd.read_csv(csv_path)

    gain, baseline = load_ecg_calibration(hea_path)
    s_raw_adc = df['ecg'].values.astype(float)
    s_raw     = (s_raw_adc - baseline) * gain / 1e6     # V (true ECG scale)
    s_raw    -= np.mean(s_raw)                          # DC removal

    ax = df['a_x'].values.astype(float)
    ay = df['a_y'].values.astype(float)
    az = df['a_z'].values.astype(float)
    wx = np.deg2rad(df['g_x'].values.astype(float))
    wy = np.deg2rad(df['g_y'].values.astype(float))
    wz = np.deg2rad(df['g_z'].values.astype(float))

    # Preprocessing
    s_bp  = signal.filtfilt(B_BP,    A_BP,    s_raw)
    s_obs = signal.filtfilt(B_NOTCH, A_NOTCH, s_bp)

    # Velocity from X-axis only
    ax_c  = ax - np.mean(ax)
    ax_bp = signal.filtfilt(B_BP, A_BP, ax_c)
    v_ref = trapezoidal_integrate(ax_bp)

    # Quaternion real part 
    q_ref = compute_quaternion_real(wx, wy, wz)

    # Oscillation Index (OI) / Δείκτης Ταλάντωσης
    # OI = total angular path length / net rotation
    # High OI → motion is highly oscillatory → quaternion (net rotation) is a
    # poor reference for motion artifacts (it captures only the "ending" but
    # not the "going back and forth" that actually causes electrode displacement).
    wx_dps = df['g_x'].values.astype(float)
    wy_dps = df['g_y'].values.astype(float)
    wz_dps = df['g_z'].values.astype(float)
    omega_mag_dps   = np.sqrt(wx_dps**2 + wy_dps**2 + wz_dps**2)
    angle_traveled_deg = float(np.sum(omega_mag_dps) * DT)
    theta_net_end_deg  = float(np.rad2deg(2.0 * np.arccos(np.clip(q_ref[-1], -1.0, 1.0))))
    oscillation_index  = angle_traveled_deg / max(theta_net_end_deg, 0.01)

    # Alignment; Call functions
    tau1 = cross_corr_lag(v_ref, s_obs)
    tau2 = cross_corr_lag(q_ref, s_obs)
    v_al = shift(v_ref, tau1)
    q_al = shift(q_ref, tau2)

    # NLMS: single-input (v only) and dual-input (v, q)
    s_clean_single = nlms(s_obs, v_al.reshape(-1, 1))
    s_clean_dual   = nlms(s_obs, np.column_stack([v_al, q_al]))

    # Metrics + BPR decomposition 
    baseline_raw = signal.filtfilt(B_LP, A_LP, s_raw)
    baseline_obs = signal.filtfilt(B_LP, A_LP, s_obs)
    gt_peaks = np.where(df['peaks'].values == 1)[0] # Find the true peaks from dataset PhysioNet.

    # BPR_preproc: how much baseline power the bandpass+notch removed before NLMS.
    # This is what the preprocessing already does without any adaptive filter.
    BPR_preproc = 10.0 * np.log10(
        np.mean(baseline_raw ** 2) / (np.mean(baseline_obs ** 2) + 1e-30)
    )

    def metric_block(s_clean):
        bl_clean = signal.filtfilt(B_LP, A_LP, s_clean)
        # BPR_total: paper's headline metric (s_raw vs s_clean)
        BPR_total = 10.0 * np.log10(
            np.mean(baseline_raw ** 2) / (np.mean(bl_clean ** 2) + 1e-30)
        )
        # BPR_NLMS_only: what the adaptive filter ACTUALLY contributed
        # (s_obs vs s_clean). This isolates the NLMS effect from preprocessing.
        BPR_NLMS = 10.0 * np.log10(
            np.mean(baseline_obs ** 2) / (np.mean(bl_clean ** 2) + 1e-30)
        )
        MSE_BL = float(np.mean(bl_clean ** 2))
        R_BL   = float(np.max(bl_clean) - np.min(bl_clean))
        # PCC vs s_obs (post-preproc, pre-NLMS) — paper does not specify reference;
        # this measures morphological similarity to the pre-NLMS signal.
        pcc, _ = pearsonr(s_obs, s_clean)
        try:
            # Dictionary named 'info' with the detected peaks that Neurokit2 found.
            _, info = nk.ecg_peaks(s_clean, sampling_rate=FS)
            det = np.array(info['ECG_R_Peaks'])
        except Exception:
            det = np.array([], dtype=int)
        ret = sensitivity(det, gt_peaks)
        drr = delta_rr_ms(det, gt_peaks)
        return BPR_total, BPR_NLMS, MSE_BL, R_BL, float(pcc), ret, drr

    bpr_tot_s, bpr_nlms_s, mse_s, rbl_s, pcc_s, ret_s, drr_s = metric_block(s_clean_single)
    bpr_tot_d, bpr_nlms_d, mse_d, rbl_d, pcc_d, ret_d, drr_d = metric_block(s_clean_dual)

    return {
        'File': os.path.basename(csv_path),
        # Per-record diagnostics
        'tau1_samples': int(tau1),
        'tau2_samples': int(tau2),
        'Q_mean': float(np.mean(q_ref)),
        'Q_std':  float(np.std(q_ref)),
        'Q_min':  float(np.min(q_ref)),
        'Q_max':  float(np.max(q_ref)),
        # Oscillation Index diagnostics (OUR metric)
        'omega_mag_mean_dps':  float(np.mean(omega_mag_dps)),
        'omega_mag_max_dps':   float(np.max(omega_mag_dps)),
        'angle_traveled_deg':  angle_traveled_deg,
        'theta_net_end_deg':   theta_net_end_deg,
        'OscillationIndex':    oscillation_index,
        # BPR decomposition: preprocessing contribution (same for all NLMS configs)
        'BPR_preproc_dB':   float(BPR_preproc),
        # Single-input NLMS
        'BPR_total_Single_dB':  bpr_tot_s,
        'BPR_NLMS_only_Single_dB': bpr_nlms_s,
        'MSE_BL_Single_V2':     mse_s,
        'R_BL_Single_V':        rbl_s,
        'PCC_Single':           pcc_s,
        'Retention_Single_%':   ret_s,
        'dRR_Single_ms':        drr_s,
        # Dual-input NLMS
        'BPR_total_Dual_dB':    bpr_tot_d,
        'BPR_NLMS_only_Dual_dB': bpr_nlms_d,
        'MSE_BL_Dual_V2':       mse_d,
        'R_BL_Dual_V':          rbl_d,
        'PCC_Dual':             pcc_d,
        'Retention_Dual_%':     ret_d,
        'dRR_Dual_ms':          drr_d,
        # Dual - Single deltas
        'BPR_total_gain_dB':    bpr_tot_d - bpr_tot_s,
        'BPR_NLMS_gain_dB':     bpr_nlms_d - bpr_nlms_s,
        'MSE_reduction_%':      100.0 * (mse_s - mse_d) / mse_s if mse_s > 0 else np.nan,
        'PCC_gain_%':           100.0 * (pcc_d - pcc_s) / pcc_s if pcc_s != 0 else np.nan,
        'Retention_gain_pp':    ret_d - ret_s,
    }



def main():
    """
    Iterate over all records in DATASET_DIR/csv/, compute per-record
    metrics, write a single per-record CSV.
    """
    csv_dir = os.path.join(DATASET_DIR, 'csv')
    csv_files = sorted(glob.glob(os.path.join(csv_dir, '*.csv')))
    if not csv_files:
        print(f"[WARN] No CSV files in {csv_dir}. Check DATASET_DIR.")
        return

    print(f"\nFound {len(csv_files)} records.\n" + "=" * 60)

    results = []
    for i, csv_path in enumerate(csv_files, start=1):
        rec_name = os.path.basename(csv_path).replace('.csv', '')
        hea_path = os.path.join(DATASET_DIR, rec_name + '.hea')
        if not os.path.exists(hea_path):
            print(f"[{i:2d}/{len(csv_files)}] SKIP (no .hea): {rec_name}")
            continue
        print(f"[{i:2d}/{len(csv_files)}] {rec_name} ...", end=' ', flush=True)
        try:
            m = evaluate_record(csv_path, hea_path)
            results.append(m)
            # Live progress: shows that the script is working.
            print(f"BPR_pre={m['BPR_preproc_dB']:5.1f} | "
                  f"BPR_NLMS={m['BPR_NLMS_only_Dual_dB']:+5.2f} | "
                  f"OI={m['OscillationIndex']:6.1f} | "
                  f"PCC={m['PCC_Dual']:.3f} | "
                  f"Ret={m['Retention_Dual_%']:5.1f}%")
        except Exception as ex:
            print(f"ERROR: {ex}")

    if not results:
        print("\nNo records evaluated.")
        return
    df_out = pd.DataFrame(results)
    out_path = os.path.join('..', '..', 'results', 'benchmark_results_paper_strict.csv')
    df_out.to_csv(out_path, index=False)

    print("\n" + "=" * 60)
    print(f"Done. {len(results)}/{len(csv_files)} records processed.")
    print(f"Wrote: {out_path}")
    print("Next File to Run:  python table1_paper.py ")


if __name__ == '__main__':
    main()
