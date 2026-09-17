"""
diagnostic_peak_fp.py

Aim : Quantify how aggressive NeuroKit2's ecg_peaks() is with FALSE
POSITIVES on noisy ECG. The hypothesis is that the high retention rate
is partly an artifact of NeuroKit2 over-detecting peaks.

Testing 4 signal versions for each record:
  1. s_raw         : raw uncalibrated CSV column (very noisy)
  2. s_obs         : after bandpass + notch (paper's preprocessing)
  3. s_clean_single: after single-input NLMS
  4. s_clean_dual  : after dual-input NLMS

For each, compute detection metrics:
  TP, FP, FN, Sensitivity, PPV
relative to the ground-truth peaks in df['peaks'].
"""

# --- Path resolution: enables this script to be run from any cwd ---
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..'))
_os.chdir(_PROJECT_ROOT)
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)


import os
import re  # regex parsing of .hea files
import numpy as np
import pandas as pd
from scipy import signal
import neurokit2 as nk

FS = 500.0
DT = 1.0 / FS
NYQ = 0.5 * FS
DATASET_DIR = os.path.join('Physionet_PTT_Dataset', 'physionet.org', 'files',
                           'pulse-transit-time-ppg', '1.1.0')
TOL = 25  # ±50 ms tolerance at 500 Hz

B_BP, A_BP = signal.butter(4, [0.16 / NYQ, 40.0 / NYQ], btype='bandpass')
B_NOTCH, A_NOTCH = signal.iirnotch(50.0, 10.0, FS)


def load_calib(hea_path):
    with open(hea_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 9 and parts[-1] == 'ecg':
                m = re.match(r'([-+0-9.eE]+)\(([-+0-9]+)\)', parts[2])
                return float(m.group(1)), int(m.group(2))
    raise ValueError(f"no ecg in {hea_path}")


def trap_int(x):
    v = np.zeros_like(x, dtype=float)
    v[1:] = np.cumsum((x[1:] + x[:-1]) * (DT / 2.0))
    return v


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1; w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_real(wx, wy, wz):
    N = len(wx); q = np.array([1.0, 0, 0, 0]); out = np.empty(N); out[0] = 1
    for n in range(1, N):
        dq = np.array([1.0, 0.5*wx[n]*DT, 0.5*wy[n]*DT, 0.5*wz[n]*DT])
        q = quat_mul(dq, q); q /= np.linalg.norm(q); out[n] = q[0]
    return out


def cross_corr_lag(ref, target, max_lag=500):
    r = (ref - np.mean(ref)) / (np.std(ref) + 1e-12)
    t = (target - np.mean(target)) / (np.std(target) + 1e-12)
    corr = signal.correlate(t, r, mode='full')
    lags = signal.correlation_lags(len(t), len(r), mode='full')
    mask = (lags >= -max_lag) & (lags <= max_lag)
    return int(lags[mask][np.argmax(corr[mask])])


def shift(s, k):
    if k > 0:   return np.pad(s, (k, 0), mode='constant')[:-k]
    elif k < 0: return np.pad(s, (0, -k), mode='constant')[-k:]
    return s


def nlms(s_obs, X, mu=0.05, eps=1e-6):
    N, M = X.shape
    W = np.zeros(M); e = np.empty(N)
    for n in range(N):
        x = X[n]; e_n = s_obs[n] - W @ x; e[n] = e_n
        W += mu * (e_n * x) / (x @ x + eps)
    return e


def detection_metrics(detected, gt, tol=TOL):
    """Return TP, FP, FN, Sens, PPV with tolerance-based matching."""
    if len(gt) == 0:
        return 0, len(detected), 0, np.nan, np.nan
    if len(detected) == 0:
        return 0, 0, len(gt), 0.0, np.nan
    # Greedy matching: each detected peak matches at most one GT peak.
    used_gt = np.zeros(len(gt), dtype=bool)
    tp = 0
    for d in detected:
        diffs = np.abs(gt - d)
        # candidates: GT peaks within tolerance and not yet used
        candidates = np.where((diffs <= tol) & ~used_gt)[0]
        if len(candidates):
            best = candidates[np.argmin(diffs[candidates])]
            used_gt[best] = True
            tp += 1
    fp = len(detected) - tp
    fn = int((~used_gt).sum())
    sens = 100.0 * tp / (tp + fn) if (tp + fn) else np.nan
    ppv  = 100.0 * tp / (tp + fp) if (tp + fp) else np.nan
    return tp, fp, fn, sens, ppv


def detect(sig):
    try:
        _, info = nk.ecg_peaks(sig, sampling_rate=FS)
        return np.array(info['ECG_R_Peaks'])
    except Exception:
        return np.array([], dtype=int)


def analyze(csv_path, hea_path):
    df = pd.read_csv(csv_path)
    gain, baseline = load_calib(hea_path)
    s_raw_adc = df['ecg'].values.astype(float)
    s_raw = (s_raw_adc - baseline) * gain / 1e6
    s_raw -= np.mean(s_raw)

    ax = df['a_x'].values.astype(float)
    wx = np.deg2rad(df['g_x'].values.astype(float))
    wy = np.deg2rad(df['g_y'].values.astype(float))
    wz = np.deg2rad(df['g_z'].values.astype(float))

    s_bp = signal.filtfilt(B_BP, A_BP, s_raw)
    s_obs = signal.filtfilt(B_NOTCH, A_NOTCH, s_bp)

    ax_c = ax - np.mean(ax); ax_bp = signal.filtfilt(B_BP, A_BP, ax_c)
    v_ref = trap_int(ax_bp)
    q_ref = quat_real(wx, wy, wz)
    v_al = shift(v_ref, cross_corr_lag(v_ref, s_obs))
    q_al = shift(q_ref, cross_corr_lag(q_ref, s_obs))

    s_clean_single = nlms(s_obs, v_al.reshape(-1, 1))
    s_clean_dual   = nlms(s_obs, np.column_stack([v_al, q_al]))

    gt = np.where(df['peaks'].values == 1)[0]
    rows = []
    for label, sig in [('s_raw_uncal', df['ecg'].values.astype(float)),
                       ('s_obs',          s_obs),
                       ('s_clean_single', s_clean_single),
                       ('s_clean_dual',   s_clean_dual)]:
        det = detect(sig)
        tp, fp, fn, sens, ppv = detection_metrics(det, gt)
        rows.append({
            'signal': label,
            'GT': len(gt),
            'Det': len(det),
            'TP': tp, 'FP': fp, 'FN': fn,
            'Sens%': sens, 'PPV%': ppv,
        })
    return rows


def main():
    import glob, sys
    # CLI flag: --full to run on entire dataset, otherwise sample
    full_mode = '--full' in sys.argv
    if full_mode:
        csv_dir = os.path.join(DATASET_DIR, 'csv')
        records = [os.path.basename(p).replace('.csv', '')
                   for p in sorted(glob.glob(os.path.join(csv_dir, '*.csv')))]
        print(f"FULL MODE: running on all {len(records)} records.")
    else:
        records = ['s1_sit', 's1_walk', 's1_run', 's5_run', 's10_run']
        print(f"SAMPLE MODE: running on {len(records)} records "
              f"(use --full to run on entire dataset).")

    all_rows = []
    for i, rec in enumerate(records, 1):
        csv_path = os.path.join(DATASET_DIR, 'csv', rec + '.csv')
        hea_path = os.path.join(DATASET_DIR, rec + '.hea')
        if not (os.path.exists(csv_path) and os.path.exists(hea_path)):
            print(f"[skip] {rec}: missing"); continue
        if full_mode:
            print(f"[{i:2d}/{len(records)}] {rec} ...", end=' ', flush=True)
        rows = analyze(csv_path, hea_path)
        for r in rows: r['record'] = rec
        all_rows.extend(rows)
        if full_mode:
            # Show only the dual_NLMS result per record in live progress
            dr = next(r for r in rows if r['signal'] == 's_clean_dual')
            print(f"Sens={dr['Sens%']:5.2f}%  PPV={dr['PPV%']:5.2f}%")

    df = pd.DataFrame(all_rows)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', None)

    if not full_mode:
        # SAMPLE MODE: show per-record breakdown (legacy behavior)
        df_disp = df[['record', 'signal', 'GT', 'Det', 'TP', 'FP', 'FN',
                      'Sens%', 'PPV%']]
        print("\n" + "=" * 100)
        print("R-PEAK DETECTION METRICS — NeuroKit2 vs Ground Truth (SAMPLE)")
        print("=" * 100)
        print(df_disp.to_string(index=False, float_format=lambda x: f"{x:6.2f}"))
    else:
        # FULL MODE: aggregate by activity and signal stage
        df['activity'] = df['record'].str.extract(r'_(sit|walk|run)$', expand=False)
        agg_cols = ['Sens%', 'PPV%', 'FP', 'FN']
        summary = (df.groupby(['activity', 'signal'])[agg_cols]
                   .agg(['mean', 'std']).round(3))
        out_path = os.path.join('results', 'diagnostic_peak_fp_results.csv')
        df.to_csv(out_path, index=False)
        # Pretty per-activity printout
        for act in ['sit', 'walk', 'run']:
            sub = df[df['activity'] == act]
            if sub.empty: continue
            print(f"\n{'=' * 100}")
            print(f"[{act.upper()}]  n = {len(sub) // 4} records")
            print('=' * 100)
            tbl = (sub.groupby('signal')[agg_cols]
                   .agg(['mean', 'std']).round(3))
            print(tbl.to_string())
        print(f"\nWrote: {out_path}")

    print("\n" + "-" * 100)
    print("Understanding the Output:")
    print("  Sens% high + PPV% LOW   -->  NeuroKit2 over-detects --> many FPs")
    print("  Sens% high + PPV% high  --> NeuroKit2 is ok --> paper's metric is fair")
    print("  FP/record drops vs s_obs --> NLMS reduces spurious peaks")
    print("  FP/record rises vs s_obs --> NLMS introduces spurious peaks")


if __name__ == '__main__':
    main()
