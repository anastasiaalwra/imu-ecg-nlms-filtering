"""
STRICT REPRODUCTION of the paper's pipeline applied to **PPG** instead of ECG.

The paper targets ECG. This script swaps the target signal to PPG
(pleth_2 = MAX30101 infrared, distal phalanx, 500 Hz) but keeps
EVERYTHING ELSE of the paper's pipeline identical:

    s_obs <-- bandpass + notch on PPG (instead of ECG)
    v     <-- trapezoidal integration of accel-X
    q_w   <-- real part of quaternion (small-angle integration of ω)
    NLMS_single = NLMS(s_obs, [v])
    NLMS_dual   = NLMS(s_obs, [v, q_w])

WHY pleth_2?
    - IR has the deepest tissue penetration → least motion-sensitive of
      the 6 PPG channels. Plotted the raw signal and chose this specific 
      channel because it showed the biggest motion.

The paper itself does not specify PPG. Used the same bandpass
(0.16-40 Hz), same notch (50 Hz Q=10), same baseline-extractor
(0-0.5 Hz LP) so that any difference vs ECG is attributable to the
    target signal, not to retuned preprocessing.

A normal PPG-specific bandpass would be ~0.5-8 Hz. It is not used that
way here because the goal is a strict transplant of the paper. 
"""

# --- Path resolution: enables this script to be run from any cwd ---
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..'))
_os.chdir(_PROJECT_ROOT)
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

import os
import glob
import re
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import pearsonr
import neurokit2 as nk

# CONFIG
FS = 500.0
DT = 1.0 / FS
NYQ = 0.5 * FS

MU = 0.05
EPSILON = 1e-6

B_BP, A_BP       = signal.butter(4, [0.16 / NYQ, 40.0 / NYQ], btype='bandpass')
B_NOTCH, A_NOTCH = signal.iirnotch(w0=50.0, Q=10.0, fs=FS)
B_LP, A_LP       = signal.butter(2, 0.5 / NYQ, btype='lowpass')   # baseline (0-0.5 Hz)

# PPG peak retention tolerance: 75 ms @ 500 Hz = 37 samples.
# Wider than ECG's 50 ms because PPG systolic peaks are broader and
# the "ground truth" detector itself is fuzzier.
P_PEAK_TOL = 37

PPG_COL = 'pleth_2'

DATASET_DIR = os.path.join('Physionet_PTT_Dataset', 'physionet.org', 'files',
                           'pulse-transit-time-ppg', '1.1.0')


# HELPERS
def load_ppg_calibration(hea_path, col_name=PPG_COL):
    with open(hea_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 9 and parts[-1] == col_name:
                m = re.match(r'([-+0-9.eE]+)\(([-+0-9]+)\)', parts[2])
                if m:
                    return float(m.group(1)), int(m.group(2))
    return 1.0, 0


def trapezoidal_integrate(x):
    v = np.zeros_like(x, dtype=float)
    v[1:] = np.cumsum((x[1:] + x[:-1]) * (DT / 2.0))
    return v


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def compute_quaternion_real(wx_rad, wy_rad, wz_rad):
    """Paper Eqs. (4)-(6). Inputs MUST be rad/s."""
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


def cross_corr_lag(ref, target, max_lag=500):
    r = (ref - np.mean(ref)) / (np.std(ref) + 1e-12)
    t = (target - np.mean(target)) / (np.std(target) + 1e-12)
    corr = signal.correlate(t, r, mode='full')
    lags = signal.correlation_lags(len(t), len(r), mode='full')
    mask = (lags >= -max_lag) & (lags <= max_lag)
    return int(lags[mask][np.argmax(corr[mask])])


def shift(sig, k):
    if k > 0:
        return np.pad(sig, (k, 0), mode='constant')[:-k]
    elif k < 0:
        return np.pad(sig, (0, -k), mode='constant')[-k:]
    return sig


def nlms(s_obs, X):
    """Paper Eqs. (10)-(12). Same code as ECG version."""
    N, M = X.shape
    W = np.zeros(M, dtype=float)
    e = np.empty(N, dtype=float)
    for n in range(N):
        x = X[n]
        e_n = s_obs[n] - W @ x
        e[n] = e_n
        W += MU * (e_n * x) / (x @ x + EPSILON)
    return e


def retention_rate(detected, true_peaks, tol=P_PEAK_TOL):
    if len(true_peaks) == 0:
        return 100.0
    if len(detected) == 0:
        return 0.0
    tp = sum(np.any(np.abs(detected - tp_idx) <= tol) for tp_idx in true_peaks)
    return 100.0 * tp / len(true_peaks)


def delta_rr_ms(detected, true_peaks, tol=P_PEAK_TOL):
    if len(detected) < 2 or len(true_peaks) < 2:
        return np.nan
    matched_det = []
    matched_true = []
    for tp_idx in true_peaks:
        d = np.abs(detected - tp_idx)
        if d.min() <= tol:
            matched_det.append(detected[d.argmin()])
            matched_true.append(tp_idx)
    if len(matched_det) < 2:
        return np.nan
    md = np.array(sorted(set(matched_det)))
    mt = np.array(sorted(set(matched_true)))
    L = min(len(md), len(mt))
    rr_det  = np.diff(md[:L]) / FS * 1000.0
    rr_true = np.diff(mt[:L]) / FS * 1000.0
    return float(np.mean(np.abs(rr_det - rr_true)))


def detect_ppg_peaks(x):
    """
    Reference peak detector for PPG. Returns sample indices of systolic peaks.
    Uses NeuroKit2 ppg_clean → ppg_findpeaks. The 'truth' for this strict
    reproduction is the detector applied to s_obs (post bandpass+notch,
    pre-NLMS), so we can ask: does NLMS DESTROY peaks that preprocessing
    already kept?
    """
    try:
        x_cl = nk.ppg_clean(x, sampling_rate=FS)
        info = nk.ppg_findpeaks(x_cl, sampling_rate=FS)
        return np.array(info['PPG_Peaks'], dtype=int)
    except Exception:
        return np.array([], dtype=int)


# MAIN EVAL
def evaluate_record(csv_path, hea_path):
    df = pd.read_csv(csv_path)

    if PPG_COL not in df.columns:
        raise KeyError(f"{PPG_COL} not in {csv_path}")

    # PPG (arbitrary units, but apply WFDB convention for consistency)
    gain, baseline = load_ppg_calibration(hea_path, PPG_COL)
    p_raw = (df[PPG_COL].values.astype(float) - baseline) * gain
    p_raw -= np.mean(p_raw) # DC removal

    # IMU
    ax = df['a_x'].values.astype(float)
    wx = np.deg2rad(df['g_x'].values.astype(float))
    wy = np.deg2rad(df['g_y'].values.astype(float))
    wz = np.deg2rad(df['g_z'].values.astype(float))

    # Preprocessing (paper Section 2.3, applied to PPG)
    p_bp  = signal.filtfilt(B_BP,    A_BP,    p_raw)
    s_obs = signal.filtfilt(B_NOTCH, A_NOTCH, p_bp)

    # Velocity reference (paper Section 2.1, X-axis only)
    ax_c  = ax - np.mean(ax)
    ax_bp = signal.filtfilt(B_BP, A_BP, ax_c)
    v_ref = trapezoidal_integrate(ax_bp)

    # Quaternion real part (paper Section 2.2)
    q_ref = compute_quaternion_real(wx, wy, wz)

    # Oscillation Index (diagnostic, same as ECG file)
    wx_dps = df['g_x'].values.astype(float)
    wy_dps = df['g_y'].values.astype(float)
    wz_dps = df['g_z'].values.astype(float)
    omega_mag_dps = np.sqrt(wx_dps**2 + wy_dps**2 + wz_dps**2)
    angle_traveled_deg = float(np.sum(omega_mag_dps) * DT)
    theta_net_end_deg  = float(np.rad2deg(2.0 * np.arccos(np.clip(q_ref[-1], -1.0, 1.0))))
    oscillation_index  = angle_traveled_deg / max(theta_net_end_deg, 0.01)

    # Alignment (paper Eqs. 7-8)
    tau1 = cross_corr_lag(v_ref, s_obs)
    tau2 = cross_corr_lag(q_ref, s_obs)
    v_al = shift(v_ref, tau1)
    q_al = shift(q_ref, tau2)

    # NLMS single + dual
    s_clean_single = nlms(s_obs, v_al.reshape(-1, 1))
    s_clean_dual   = nlms(s_obs, np.column_stack([v_al, q_al]))

    # Metrics
    baseline_raw = signal.filtfilt(B_LP, A_LP, p_raw)
    baseline_obs = signal.filtfilt(B_LP, A_LP, s_obs)
    # Ground truth for retention: PPG peaks detected on s_obs
    # (post-preproc, pre-NLMS). This isolates the NLMS contribution.
    gt_peaks = detect_ppg_peaks(s_obs)

    BPR_preproc = 10.0 * np.log10(
        np.mean(baseline_raw ** 2) / (np.mean(baseline_obs ** 2) + 1e-30)
    )

    def metric_block(s_clean):
        bl_clean = signal.filtfilt(B_LP, A_LP, s_clean)
        BPR_total = 10.0 * np.log10(
            np.mean(baseline_raw ** 2) / (np.mean(bl_clean ** 2) + 1e-30)
        )
        BPR_NLMS = 10.0 * np.log10(
            np.mean(baseline_obs ** 2) / (np.mean(bl_clean ** 2) + 1e-30)
        )
        MSE_BL = float(np.mean(bl_clean ** 2))
        R_BL   = float(np.max(bl_clean) - np.min(bl_clean))
        pcc, _ = pearsonr(s_obs, s_clean)
        det = detect_ppg_peaks(s_clean)
        ret = retention_rate(det, gt_peaks)
        drr = delta_rr_ms(det, gt_peaks)
        return BPR_total, BPR_NLMS, MSE_BL, R_BL, float(pcc), ret, drr

    bpr_t_s, bpr_n_s, mse_s, rbl_s, pcc_s, ret_s, drr_s = metric_block(s_clean_single)
    bpr_t_d, bpr_n_d, mse_d, rbl_d, pcc_d, ret_d, drr_d = metric_block(s_clean_dual)

    return {
        'File': os.path.basename(csv_path),
        'PPG_channel': PPG_COL,
        'tau1_samples': int(tau1),
        'tau2_samples': int(tau2),
        'Q_mean': float(np.mean(q_ref)),
        'Q_std':  float(np.std(q_ref)),
        'Q_min':  float(np.min(q_ref)),
        'Q_max':  float(np.max(q_ref)),
        'omega_mag_mean_dps': float(np.mean(omega_mag_dps)),
        'omega_mag_max_dps':  float(np.max(omega_mag_dps)),
        'angle_traveled_deg': angle_traveled_deg,
        'theta_net_end_deg':  theta_net_end_deg,
        'OscillationIndex':   oscillation_index,
        'N_truth_peaks':      int(len(gt_peaks)),
        # Preprocessing contribution
        'BPR_preproc_dB': float(BPR_preproc),
        # Single-input
        'BPR_total_Single_dB':     bpr_t_s,
        'BPR_NLMS_only_Single_dB': bpr_n_s,
        'MSE_BL_Single':           mse_s,
        'R_BL_Single':             rbl_s,
        'PCC_Single':              pcc_s,
        'Retention_Single_%':      ret_s,
        'dRR_Single_ms':           drr_s,
        # Dual-input
        'BPR_total_Dual_dB':       bpr_t_d,
        'BPR_NLMS_only_Dual_dB':   bpr_n_d,
        'MSE_BL_Dual':             mse_d,
        'R_BL_Dual':               rbl_d,
        'PCC_Dual':                pcc_d,
        'Retention_Dual_%':        ret_d,
        'dRR_Dual_ms':             drr_d,
        # Deltas
        'BPR_total_gain_dB':       bpr_t_d - bpr_t_s,
        'BPR_NLMS_gain_dB':        bpr_n_d - bpr_n_s,
        'MSE_reduction_%':         100.0 * (mse_s - mse_d) / mse_s if mse_s > 0 else np.nan,
        'PCC_gain_%':              100.0 * (pcc_d - pcc_s) / pcc_s if pcc_s != 0 else np.nan,
        'Retention_gain_pp':       ret_d - ret_s,
    }


# DRIVER
def main():
    csv_dir = os.path.join(DATASET_DIR, 'csv')
    csv_files = sorted(glob.glob(os.path.join(csv_dir, '*.csv')))
    if not csv_files:
        print(f"[WARN] No CSV files in {csv_dir}.")
        return

    print(f"\nStrict PPG reproduction — target column: {PPG_COL}")
    print(f"Found {len(csv_files)} records.\n" + "=" * 60)

    results = []
    for i, csv_path in enumerate(csv_files, start=1):
        rec = os.path.basename(csv_path).replace('.csv', '')
        hea_path = os.path.join(DATASET_DIR, rec + '.hea')
        if not os.path.exists(hea_path):
            print(f"[{i:2d}/{len(csv_files)}] SKIP (no .hea): {rec}")
            continue
        print(f"[{i:2d}/{len(csv_files)}] {rec} ...", end=' ', flush=True)
        try:
            m = evaluate_record(csv_path, hea_path)
            results.append(m)
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
    

    out = pd.DataFrame(results)
    out_path = os.path.join('results', 'benchmark_results_paper_strict_ppg.csv')
    out.to_csv(out_path, index=False)
    print("\n" + "=" * 60)
    print(f"Done. {len(results)}/{len(csv_files)} records processed.")
    print(f"Wrote: {out_path}")


if __name__ == '__main__':
    main()
