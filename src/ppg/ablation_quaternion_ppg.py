"""
Compare 5 NLMS configurationzs on every record:

  1. SINGLE    : v_ref only
  2. DUAL_REAL : v_ref + real q_ref
  3. DUAL_CONST: v_ref + constant 1.0    (zero information)
  4. DUAL_RAMP : v_ref + linear 0 → 1    (slow monotonic, no motion info)
  5. DUAL_RANDOM: v_ref + low-pass Gaussian noise (slow random, no motion info)
"""

# Path resolution: enables this script to be run from any cwd 
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
import neurokit2 as nk

FS = 500.0
DT = 1.0 / FS
NYQ = 0.5 * FS
MU = 0.05
EPSILON = 1e-6
P_PEAK_TOL = 37
PPG_COL = 'pleth_2'

DATASET_DIR = os.path.join('Physionet_PTT_Dataset', 'physionet.org', 'files',
                           'pulse-transit-time-ppg', '1.1.0')

B_BP, A_BP       = signal.butter(4, [0.16 / NYQ, 40.0 / NYQ], btype='bandpass')
B_NOTCH, A_NOTCH = signal.iirnotch(50.0, 10.0, FS)
B_LP, A_LP       = signal.butter(2, 0.5 / NYQ, btype='lowpass')
# Very-low-pass for the random surrogate (~0.01 Hz, matches the quaternion's super-slow drift timescale in the PTT-PPG data).
B_VLP, A_VLP     = signal.butter(2, 0.01 / NYQ, btype='lowpass')

def load_calib(hea_path, col=PPG_COL):
    with open(hea_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 9 and parts[-1] == col:
                m = re.match(r'([-+0-9.eE]+)\(([-+0-9]+)\)', parts[2])
                if m:
                    return float(m.group(1)), int(m.group(2))
    return 1.0, 0


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
        q = quat_mul(dq, q); 
        q /= np.linalg.norm(q); 
        out[n] = q[0]
    return out


def cross_corr_lag(ref, target, max_lag=500):
    if np.std(ref) < 1e-12:
        return 0
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


def nlms(s_obs, X, mu=MU, eps=EPSILON):
    N, M = X.shape; W = np.zeros(M); e = np.empty(N)
    for n in range(N):
        x = X[n]; e_n = s_obs[n] - W @ x; e[n] = e_n
        W += mu * (e_n * x) / (x @ x + eps)
    return e


def retention(detected, true_peaks, tol=P_PEAK_TOL):
    if len(true_peaks) == 0: return 100.0
    if len(detected) == 0:   return 0.0
    tp = sum(np.any(np.abs(detected - t) <= tol) for t in true_peaks)
    return 100.0 * tp / len(true_peaks)


def make_slow_random(N, seed):
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(N)
    y = signal.filtfilt(B_VLP, A_VLP, raw)
    return (y - y.min()) / (y.max() - y.min() + 1e-12)


def detect_ppg_peaks(x):
    try:
        x_cl = nk.ppg_clean(x, sampling_rate=FS)
        info = nk.ppg_findpeaks(x_cl, sampling_rate=FS)
        return np.array(info['PPG_Peaks'], dtype=int)
    except Exception:
        return np.array([], dtype=int)


def metric_block(s_clean, s_obs, baseline_obs, gt_peaks):
    bl_clean = signal.filtfilt(B_LP, A_LP, s_clean)
    BPR_NLMS = 10.0 * np.log10(
        np.mean(baseline_obs ** 2) / (np.mean(bl_clean ** 2) + 1e-30)
    )
    MSE_BL = float(np.mean(bl_clean ** 2))
    pcc, _ = pearsonr(s_obs, s_clean)
    det = detect_ppg_peaks(s_clean)
    ret = retention(det, gt_peaks)
    return BPR_NLMS, MSE_BL, float(pcc), ret


def evaluate_record(csv_path, hea_path):
    df = pd.read_csv(csv_path)
    if PPG_COL not in df.columns:
        raise KeyError(f"{PPG_COL} missing in {csv_path}")

    gain, base = load_calib(hea_path, PPG_COL)
    p_raw = (df[PPG_COL].values.astype(float) - base) * gain
    p_raw -= np.mean(p_raw)

    ax = df['a_x'].values.astype(float)
    wx = np.deg2rad(df['g_x'].values.astype(float))
    wy = np.deg2rad(df['g_y'].values.astype(float))
    wz = np.deg2rad(df['g_z'].values.astype(float))

    p_bp  = signal.filtfilt(B_BP,    A_BP,    p_raw)
    s_obs = signal.filtfilt(B_NOTCH, A_NOTCH, p_bp)

    ax_c  = ax - np.mean(ax)
    ax_bp = signal.filtfilt(B_BP, A_BP, ax_c)
    v_ref = trap_int(ax_bp)
    q_r   = quat_real(wx, wy, wz)

    N = len(p_raw)
    q_const  = np.ones(N)
    q_ramp   = np.linspace(0.0, 1.0, N)
    seed     = abs(hash(os.path.basename(csv_path))) % (2**32)
    q_rand   = make_slow_random(N, seed)

    v_al  = shift(v_ref, cross_corr_lag(v_ref, s_obs))
    qR_al = shift(q_r,   cross_corr_lag(q_r,   s_obs))
    qP_al = shift(q_ramp,cross_corr_lag(q_ramp,s_obs))
    qN_al = shift(q_rand,cross_corr_lag(q_rand,s_obs))

    baseline_obs = signal.filtfilt(B_LP, A_LP, s_obs)
    gt_peaks = detect_ppg_peaks(s_obs)

    configs = {
        'single':      v_al.reshape(-1, 1),
        'dual_real':   np.column_stack([v_al, qR_al]),
        'dual_const':  np.column_stack([v_al, q_const]),
        'dual_ramp':   np.column_stack([v_al, qP_al]),
        'dual_random': np.column_stack([v_al, qN_al]),
    }
    out = {'File': os.path.basename(csv_path)}
    for label, X in configs.items():
        s_clean = nlms(s_obs, X)
        bpr, mse, pcc, ret = metric_block(s_clean, s_obs, baseline_obs, gt_peaks)
        out[f'BPR_{label}'] = bpr
        out[f'MSE_{label}'] = mse
        out[f'PCC_{label}'] = pcc
        out[f'Ret_{label}'] = ret
    return out


def main():
    csv_dir = os.path.join(DATASET_DIR, 'csv')
    csv_files = sorted(glob.glob(os.path.join(csv_dir, '*.csv')))
    print(f"\nPPG ablation — target: {PPG_COL}")
    print(f"Found {len(csv_files)} records.\n" + "=" * 60)

    results = []
    for i, csv_path in enumerate(csv_files, 1):
        rec = os.path.basename(csv_path).replace('.csv', '')
        hea = os.path.join(DATASET_DIR, rec + '.hea')
        if not os.path.exists(hea):
            print(f"[{i:2d}/{len(csv_files)}] SKIP (no .hea): {rec}"); continue
        print(f"[{i:2d}/{len(csv_files)}] {rec} ...", end=' ', flush=True)
        try:
            m = evaluate_record(csv_path, hea)
            results.append(m)
            print(f"BPR: single={m['BPR_single']:5.2f}  "
                  f"real={m['BPR_dual_real']:5.2f}  "
                  f"const={m['BPR_dual_const']:5.2f}  "
                  f"ramp={m['BPR_dual_ramp']:5.2f}  "
                  f"rand={m['BPR_dual_random']:5.2f}")
        except Exception as ex:
            print(f"ERROR: {ex}")

    if not results:
        print("Nothing to write."); return

    df = pd.DataFrame(results)
    df['activity'] = df['File'].str.extract(r'_(sit|walk|run)\.csv', expand=False)
    out_path = os.path.join('results', 'ablation_quaternion_ppg_results.csv')
    df.to_csv(out_path, index=False)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', None)

    for metric in ['BPR', 'PCC', 'MSE', 'Ret']:
        cols = [f'{metric}_single', f'{metric}_dual_real',
                f'{metric}_dual_const', f'{metric}_dual_ramp',
                f'{metric}_dual_random']
        agg = df.groupby('activity')[cols].mean().round(4)
        agg = agg.reindex(['sit', 'walk', 'run'])
        print(f"\n{'=' * 100}")
        print(f"{metric}  — mean per activity (5 NLMS configs)  [PPG, pleth_2]")
        print('=' * 100)
        print(agg.to_string())

        print(f"\n  Δ from REAL (smaller |Δ| → surrogate matches real → q has no specific value):")
        for surr in ['const', 'ramp', 'random']:
            real_col = f'{metric}_dual_real'
            surr_col = f'{metric}_dual_{surr}'
            diffs = (df[real_col] - df[surr_col])
            print(f"    real − {surr:6s}:  ", end='')
            for act in ['sit', 'walk', 'run']:
                mask = df['activity'] == act
                if mask.sum() == 0: continue
                mean_d = diffs[mask].mean()
                print(f"{act}={mean_d:+7.3f}  ", end='')
            print()

    print("\nWrote: {out_path}")


if __name__ == '__main__':
    main()
