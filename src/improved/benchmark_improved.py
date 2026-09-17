"""
benchmark_improved.py
=====================

Improved version of the paper's NLMS-based ECG filtering pipeline.

Architecture (unchanged from the paper): two-channel NLMS with linear
velocity + real-part quaternion as references.

Refinements over the strict reproduction (each one documented in
REPORT.docx §3.2):

  (a) WEIGHTED MULTI-AXIS VELOCITY. v_ref = w_x*vx + w_y*vy + w_z*vz,
      where per-axis weights come from a min-max-normalised average of
      |Pearson| and MI between each integrated-axis velocity and the
      ECG noise residual (s_raw - s_obs). Faithful to the paper text.

  (b) BANDPASS-PREFILTERED CROSS-CORRELATION FOR DELAY. Both target and
      references are bandpassed to 1-10 Hz before computing the
      maximum-correlation lag. Restricts the alignment problem to the
      motion band and avoids QRS-spike-driven false alignments.

  (c) L-TAP NLMS WITH LEAKAGE. Default L = 128 (256 ms of memory at
      500 Hz). The paper's L = 1 cannot model the body's motion-to-
      electrode transfer; longer L can. L = 128 is the safe default —
      large enough for body-motion propagation, short enough to avoid
      learning cardiac periodicity (RR interval ~ 600-1000 ms).
      A small leakage term (lambda = 1e-4) prevents weight drift on
      flat-input segments. A systematic sweep is in sensitivity_sweep_L.py.

  (d) QUATERNION REFERENCE UNCHANGED. Kept as the paper defines it,
      despite the §4.2 ablation suggesting it can be matched by a constant.
      That finding is empirical on the finger-IMU dataset and the
      geometric explanation is a hypothesis. Replacing q with a
      placement-robust alternative (e.g. |omega|) is future work (§7.6).

Output: results/benchmark_results_improved.csv (one row per record).
"""

import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..'))
_os.chdir(_PROJECT_ROOT)
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

import os
import re
import glob
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression
import neurokit2 as nk


FS         = 500.0
DT         = 1.0 / FS
NYQ        = 0.5 * FS

L_TAPS     = 256        
MU         = 0.5  # NLMS learninh rate
EPSILON    = 1e-2       
LEAK       = 1e-4       

MAX_LAG    = 150  # cross-correlation lag search window (samples: ~0.3 s)
SAFE_IDX_S = 20.0 # seconds skipped at the start when computing z-score statistics

# Filters
B_BP,    A_BP    = signal.butter(4, [0.16 / NYQ, 40.0 / NYQ], btype='bandpass')
B_NOTCH, A_NOTCH = signal.iirnotch(w0=50.0, Q=10.0, fs=FS)
B_HP_REF, A_HP_REF = signal.butter(2, 0.16 / NYQ, btype='highpass')
B_LP_BL,  A_LP_BL  = signal.butter(2, 0.5  / NYQ, btype='lowpass') 
B_BP_AL,  A_BP_AL  = signal.butter(4, [1.0 / NYQ, 10.0 / NYQ], btype='bandpass')  # alignment band

DATASET_DIR = os.path.join('Physionet_PTT_Dataset', 'physionet.org', 'files',
                           'pulse-transit-time-ppg', '1.1.0')


def load_ecg_calibration(hea_path):
    with open(hea_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 9 and parts[-1] == 'ecg':
                m = re.match(r'([-+0-9.eE]+)\(([-+0-9]+)\)', parts[2])
                if m:
                    return float(m.group(1)), int(m.group(2))
    raise ValueError(f"no ecg calibration in {hea_path}")


def calc_velocity(acc, dt=DT):
    acc_c = acc - np.mean(acc)
    acc_bp = signal.filtfilt(B_BP, A_BP, acc_c)
    v = np.zeros_like(acc_bp)
    v[1:]   = np.cumsum((acc_bp[1:] + acc_bp[:-1]) * (dt / 2.0))
    return v


def quaternion_real_part(wx, wy, wz, dt=DT):
    """Real part q_0(n) of the integrated quaternion from gyroscope (rad/s)."""
    N = len(wx)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    out = np.zeros(N)
    out[0] = 1.0
    for n in range(1, N):
        dq = np.array([1.0, 0.5*wx[n]*dt, 0.5*wy[n]*dt, 0.5*wz[n]*dt])
        # Small-angle quaternion update + strict normalisation
        w1, x1, y1, z1 = dq; w2, x2, y2, z2 = q
        q = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])
        q /= np.linalg.norm(q)
        out[n] = q[0]
    return out

# need a 'strict' normalization , because MI and pearson correlation
#  have different ranges when they are being computed. 
def normalize_dict(d):
    """Min-max normalise the values of a dict to [0, 1]."""
    vals = np.array(list(d.values()))
    mn, mx = vals.min(), vals.max()
    if mx - mn < 1e-12:
        return {k: 1.0 for k in d}
    return {k: (v - mn) / (mx - mn) for k, v in d.items()}


# computing the velocity using all 3-axis of gyroscope and taking the
# percentage of the biggest correlation with noise for every step and every axis.
def axis_weights(vx, vy, vz, noise_target, random_state=42):
    """Combined Pearson + MI weighting of the three velocity axes."""
    corrs = {
        'X': abs(pearsonr(vx, noise_target)[0]),
        'Y': abs(pearsonr(vy, noise_target)[0]),
        'Z': abs(pearsonr(vz, noise_target)[0]),
    }
    V = np.column_stack([vx, vy, vz])
    mi = mutual_info_regression(V, noise_target, random_state=random_state)
    mi_dict = {'X': mi[0], 'Y': mi[1], 'Z': mi[2]}

    p_norm  = normalize_dict(corrs)
    mi_norm = normalize_dict(mi_dict)
    combined = {axis: (p_norm[axis] + mi_norm[axis]) / 2.0 for axis in ['X', 'Y', 'Z']}
    total = sum(combined.values())
    if total < 1e-12:
        return {'X': 1.0/3, 'Y': 1.0/3, 'Z': 1.0/3}
    return {axis: combined[axis] / total for axis in ['X', 'Y', 'Z']}


def find_delay(target, ref, max_lag=MAX_LAG):
    """Cross-correlation lag with bandpass prefiltering on both signals."""
    t = signal.filtfilt(B_BP_AL, A_BP_AL, target)
    r = signal.filtfilt(B_BP_AL, A_BP_AL, ref)
    t = (t - np.mean(t)) / (np.std(t) + 1e-8)
    r = (r - np.mean(r)) / (np.std(r) + 1e-8)
    corr = signal.correlate(t, r, mode='full')
    lags = signal.correlation_lags(len(t), len(r), mode='full')
    mask = (lags >= -max_lag) & (lags <= max_lag)
    return int(lags[mask][np.argmax(corr[mask])])


def shift_signal(sig, k):
    if k > 0:   return np.pad(sig, (k, 0), mode='constant')[:-k]
    elif k < 0: return np.pad(sig, (0, -k), mode='constant')[-k:]
    return sig


def leaky_nlms(s_obs, X_cols, L, mu=MU, eps=EPSILON, leak=LEAK):
    N = len(s_obs)
    K = len(X_cols)
    W = np.zeros(L * K)
    e = np.zeros(N)
    # Pad each reference so the FIR window starts at sample 0
    pads = [np.pad(x, (L - 1, 0), mode='constant') for x in X_cols]
    for n in range(N):
        Xn = np.concatenate([p[n : n + L][::-1] for p in pads])
        yhat = float(W @ Xn)
        en   = s_obs[n] - yhat
        e[n] = en
        norm = float(Xn @ Xn) + eps
        W = (1.0 - mu * leak) * W + (mu * en / norm) * Xn
    return e


def compute_metrics(s_raw, s_obs, s_clean, peaks_gt, fs=FS):
    """BPR (dB), MSE_BL, PCC (vs s_raw), Retention (% wrt ground-truth peaks)."""
    baseline_raw   = signal.filtfilt(B_LP_BL, A_LP_BL, s_raw)
    baseline_clean = signal.filtfilt(B_LP_BL, A_LP_BL, s_clean)
    bpr = 10.0 * np.log10(np.mean(baseline_raw**2) / (np.mean(baseline_clean**2) + 1e-12))
    mse_bl = float(np.mean(baseline_clean**2))
    pcc, _ = pearsonr(s_raw, s_clean)
    if len(peaks_gt) == 0:
        retention = 100.0
    else:
        try:
            _, info = nk.ecg_peaks(s_clean, sampling_rate=fs)
            detected = np.asarray(info['ECG_R_Peaks'])
        except Exception:
            detected = np.array([], dtype=int)
        tol = 25  # 50 ms at 500 Hz
        tp = sum(int(np.any(np.abs(detected - p) <= tol)) for p in peaks_gt)
        retention = 100.0 * tp / len(peaks_gt)
    return bpr, mse_bl, pcc, retention



def evaluate_record(csv_path, hea_path, L=L_TAPS, mu=MU):
    df = pd.read_csv(csv_path)
    s_raw_adc = df['ecg'].values.astype(float)
    ax = df['a_x'].values.astype(float)
    ay = df['a_y'].values.astype(float)
    az = df['a_z'].values.astype(float)
    wx = np.deg2rad(df['g_x'].values.astype(float))
    wy = np.deg2rad(df['g_y'].values.astype(float))
    wz = np.deg2rad(df['g_z'].values.astype(float))
    peaks_gt = np.where(df['peaks'].values == 1)[0]

    # 1. Per-subject ECG calibration from .hea 
    gain, baseline = load_ecg_calibration(hea_path)
    s_raw = ((s_raw_adc - baseline) * gain) / 1000.0   # to mV conversion
    s_raw = s_raw - np.mean(s_raw)

    # 2. Preprocessing (bandpass + notch)
    s_bp  = signal.filtfilt(B_BP, A_BP, s_raw)
    s_obs = signal.filtfilt(B_NOTCH, A_NOTCH, s_bp)

    # 3. Noise residual (target for the weighting)
    noise = s_raw - s_obs

    # 4. Weighted velocity reference (a)
    vx, vy, vz = calc_velocity(ax), calc_velocity(ay), calc_velocity(az)
    w = axis_weights(vx, vy, vz, noise)
    v_ref = w['X']*vx + w['Y']*vy + w['Z']*vz

    # 5. Quaternion reference 
    q_ref = quaternion_real_part(wx, wy, wz)

    # 6. Highpass both references (remove DC + slow trends from integration)
    v_ref_hp = signal.filtfilt(B_HP_REF, A_HP_REF, v_ref - np.mean(v_ref))
    q_ref_hp = signal.filtfilt(B_HP_REF, A_HP_REF, q_ref - np.mean(q_ref))

    safe = int(SAFE_IDX_S * FS) if len(v_ref_hp) > int(SAFE_IDX_S * FS) else 0
    v_mu, v_sd = float(np.mean(v_ref_hp[safe:])), float(np.std(v_ref_hp[safe:]) + 1e-8)
    q_mu, q_sd = float(np.mean(q_ref_hp[safe:])), float(np.std(q_ref_hp[safe:]) + 1e-8)
    v_ref_z = np.clip((v_ref_hp - v_mu) / v_sd, -3.0, 3.0)
    q_ref_z = np.clip((q_ref_hp - q_mu) / q_sd, -3.0, 3.0)

    # 8. Bandpass-prefiltered cross-correlation alignment (b)
    tau_v = find_delay(s_obs, v_ref_z)
    tau_q = find_delay(s_obs, q_ref_z)
    v_al = shift_signal(v_ref_z, tau_v)
    q_al = shift_signal(q_ref_z, tau_q)

    # 9. L-tap leaky NLMS — single (v only) and dual (v + q) (c)
    s_clean_single = leaky_nlms(s_obs, [v_al],        L, mu=mu)
    s_clean_dual   = leaky_nlms(s_obs, [v_al, q_al],  L, mu=mu)

    # 10. Calculate Metrics
    bpr_s, mse_s, pcc_s, ret_s = compute_metrics(s_raw, s_obs, s_clean_single, peaks_gt)
    bpr_d, mse_d, pcc_d, ret_d = compute_metrics(s_raw, s_obs, s_clean_dual,   peaks_gt)

    return {
        'File': os.path.basename(csv_path),
        'L_taps': L,
        'mu': mu,
        'tau_v_samples': tau_v, 'tau_q_samples': tau_q,
        'W_X': w['X'], 'W_Y': w['Y'], 'W_Z': w['Z'],
        'BPR_Single_dB': bpr_s, 'BPR_Dual_dB': bpr_d, 'BPR_gain_dB': bpr_d - bpr_s,
        'MSE_BL_Single': mse_s, 'MSE_BL_Dual': mse_d,
        'PCC_Single': pcc_s, 'PCC_Dual': pcc_d,
        'Retention_Single_%': ret_s, 'Retention_Dual_%': ret_d,
        'Retention_gain_pp': ret_d - ret_s,
    }



def main():
    csv_dir = os.path.join(DATASET_DIR, 'csv')
    csv_files = sorted(glob.glob(os.path.join(csv_dir, '*.csv')))
    print(f"Found {len(csv_files)} CSV files. L = {L_TAPS}.")

    rows = []
    for i, csv_path in enumerate(csv_files, 1):
        rec_name = os.path.basename(csv_path).replace('.csv', '')
        hea_path = os.path.join(DATASET_DIR, rec_name + '.hea')
        if not os.path.exists(hea_path):
            print(f"[{i:2d}/{len(csv_files)}] SKIP (no .hea): {rec_name}")
            continue
        try:
            row = evaluate_record(csv_path, hea_path, L=L_TAPS)
            rows.append(row)
            print(f"[{i:2d}/{len(csv_files)}] {rec_name}  "
                  f"BPR_dual={row['BPR_Dual_dB']:6.2f} dB  "
                  f"Ret_dual={row['Retention_Dual_%']:5.2f}%")
        except Exception as e:
            print(f"[{i:2d}/{len(csv_files)}] ERROR {rec_name}: {e}")

    df = pd.DataFrame(rows)
    out_dir = 'results'
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, 'benchmark_results_improved.csv')
    df.to_csv(out_csv, index=False)
    print(f"\nWrote: {out_csv}  ({len(df)} rows)")

    if len(df):
        print("\nPer-activity means (Dual):")
        df['activity'] = df['File'].str.extract(r'_(sit|walk|run)\.csv', expand=False)
        agg = df.groupby('activity')[['BPR_Dual_dB', 'PCC_Dual', 'Retention_Dual_%']].mean().round(3)
        print(agg.to_string())


if __name__ == '__main__':
    main()
