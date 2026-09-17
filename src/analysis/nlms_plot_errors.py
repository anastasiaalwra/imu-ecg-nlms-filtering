# Path resolution: enables this script to be run from any cwd
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..'))
_os.chdir(_PROJECT_ROOT)
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

import os
import re
import sys
import argparse
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

try:
    import padasip as pa
except ImportError:
    raise ImportError("padasip not installed. Run:  pip install padasip")

FS = 500.0
DT = 1.0 / FS
NYQ = 0.5 * FS
MU = 0.05
EPSILON = 1e-6

DATASET_DIR = os.path.join('Physionet_PTT_Dataset', 'physionet.org', 'files',
                           'pulse-transit-time-ppg', '1.1.0')

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
    N = len(wx)
    q = np.array([1.0, 0, 0, 0])
    out = np.empty(N)
    out[0] = 1.0
    for n in range(1, N):
        dq = np.array([1.0, 0.5*wx[n]*DT, 0.5*wy[n]*DT, 0.5*wz[n]*DT])
        q = quat_mul(dq, q)
        q /= np.linalg.norm(q)
        out[n] = q[0]
    return out


def cross_corr_lag(ref, target, max_lag=500):
    r = (ref - np.mean(ref)) / (np.std(ref) + 1e-12)
    t = (target - np.mean(target)) / (np.std(target) + 1e-12)
    corr = signal.correlate(t, r, mode='full')
    lags = signal.correlation_lags(len(t), len(r), mode='full')
    mask = (lags >= -max_lag) & (lags <= max_lag)
    return int(lags[mask][np.argmax(corr[mask])])


def shift_signal(s, k):
    if k > 0:   return np.pad(s, (k, 0), mode='constant')[:-k]
    elif k < 0: return np.pad(s, (0, -k), mode='constant')[-k:]
    return s

# Printing error (e) with function MANUALLY. 
def nlms_manual(s_obs, X, mu=MU, eps=EPSILON):
    """Paper Eq. 10-12. Returns (e, y, W_hist)."""
    N, M = X.shape
    W = np.zeros(M, dtype=float)
    e = np.empty(N, dtype=float)
    y = np.empty(N, dtype=float)
    W_hist = np.empty((N, M), dtype=float)
    for n in range(N):
        x = X[n]
        W_hist[n] = W.copy()
        y_n = W @ x
        e_n = s_obs[n] - y_n
        y[n] = y_n
        e[n] = e_n
        W += mu * (e_n * x) / (x @ x + eps)
    return e, y, W_hist


def prepare(csv_path, hea_path):
    df = pd.read_csv(csv_path)
    gain, baseline = load_calib(hea_path)
    s_raw = (df['ecg'].values.astype(float) - baseline) * gain / 1e6
    s_raw -= np.mean(s_raw)

    ax = df['a_x'].values.astype(float)
    wx = np.deg2rad(df['g_x'].values.astype(float))
    wy = np.deg2rad(df['g_y'].values.astype(float))
    wz = np.deg2rad(df['g_z'].values.astype(float))

    s_bp = signal.filtfilt(B_BP, A_BP, s_raw)
    s_obs = signal.filtfilt(B_NOTCH, A_NOTCH, s_bp)

    ax_c = ax - np.mean(ax)
    ax_bp = signal.filtfilt(B_BP, A_BP, ax_c)
    v_ref = trap_int(ax_bp)
    q_ref = quat_real(wx, wy, wz)

    v_al = shift_signal(v_ref, cross_corr_lag(v_ref, s_obs))
    q_al = shift_signal(q_ref, cross_corr_lag(q_ref, s_obs))
    return s_obs, v_al, q_al


def plot_error_figure(time_s, e_pa, e_outpa, e_man, record_name, mode):
    """
    Build one figure with 4 subplots:
      0: e_padasip
      1: e_outof_padasip
      2: e_manual
      3: all three overlaid
    The first 3 subplots share a common y-axis range so that the visual
    comparison is honest (otherwise independent autoscaling can make
    numerically identical signals look different).
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"NLMS errors ({mode}-input) — {record_name}")

    axes[0].plot(time_s, e_pa, linewidth=0.8, color="tab:blue")
    axes[0].set_ylabel(f"e_{mode}_padasip")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_s, e_outpa, linewidth=0.8, color="tab:orange")
    axes[1].set_ylabel(f"e_{mode}_outof_padasip")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_s, e_man, linewidth=0.8, color="tab:green")
    axes[2].set_ylabel(f"e_{mode}_manual")
    axes[2].grid(True, alpha=0.3)

    # Force common y-limits across the 3 individual plots --> fair visual comparison
    all_y = np.concatenate([e_pa, e_outpa, e_man])
    ymin, ymax = float(np.min(all_y)), float(np.max(all_y))
    pad = 0.05 * (ymax - ymin) if ymax > ymin else 1e-6
    for ax in axes[:3]:
        ax.set_ylim(ymin - pad, ymax + pad)

    axes[3].plot(time_s, e_pa,    linewidth=0.8, label=f"e_{mode}_padasip",        color="tab:blue")
    axes[3].plot(time_s, e_outpa, linewidth=0.8, label=f"e_{mode}_outof_padasip",  color="tab:orange")
    axes[3].plot(time_s, e_man,   linewidth=0.8, label=f"e_{mode}_manual",         color="tab:green")
    axes[3].set_ylabel("All three (overlay)")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend(loc="upper right")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()


def main():
    parser = argparse.ArgumentParser(description='Plot NLMS error signals for one record.')
    parser.add_argument('record', help='Record name (e.g. s1_walk) OR full path to .csv')
    parser.add_argument('--start',  type=int, default=0,
                        help='Starting sample index for the plotted/printed slice (default 0)')
    parser.add_argument('--count',  type=int, default=None,
                        help='Number of samples to PLOT (default: ALL from --start to end)')
    parser.add_argument('--print',  type=int, default=0, dest='print_rows',
                        help='Number of samples to PRINT to terminal, starting at --start (default 0 = no printout)')
    parser.add_argument('--save',   type=str, default=None,
                        help='Optional: filename to save FULL comparison CSV')
    args = parser.parse_args()

    # Resolve csv/hea paths
    if args.record.endswith('.csv') and os.path.exists(args.record):
        csv_path = args.record
        rec_name = os.path.basename(args.record).replace('.csv', '')
        hea_path = os.path.join(DATASET_DIR, rec_name + '.hea')
    else:
        rec_name = args.record.replace('.csv', '')
        csv_path = os.path.join(DATASET_DIR, 'csv', rec_name + '.csv')
        hea_path = os.path.join(DATASET_DIR, rec_name + '.hea')

    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV not found: {csv_path}")
        sys.exit(1)
    if not os.path.exists(hea_path):
        print(f"[ERROR] HEA not found: {hea_path}")
        sys.exit(1)

    print(f"\nLoading: {rec_name}")
    print(f"  csv:  {csv_path}")
    print(f"  hea:  {hea_path}")

    s_obs, v_al, q_al = prepare(csv_path, hea_path)
    N = len(s_obs)
    print(f"  N samples = {N},  duration = {N/FS:.2f} s\n")

    X_single = v_al.reshape(-1, 1)
    X_dual   = np.column_stack([v_al, q_al])

    # METHOD 1: PADASIP
    print("Running padasip NLMS (single)...")
    f1 = pa.filters.FilterNLMS(n=1, mu=MU, eps=EPSILON, w="zeros")
    y_single_pa, e_single_pa, _ = f1.run(s_obs, X_single)

    print("Running padasip NLMS (dual)...")
    f2 = pa.filters.FilterNLMS(n=2, mu=MU, eps=EPSILON, w="zeros")
    y_dual_pa, e_dual_pa, _ = f2.run(s_obs, X_dual)

    # METHOD 3: Manual
    print("Running manual NLMS (single)...")
    e_single_man, _, _ = nlms_manual(s_obs, X_single)
    print("Running manual NLMS (dual)...")
    e_dual_man, _, _ = nlms_manual(s_obs, X_dual)

    # METHOD 2: OUT OF PADASIP (s_obs - y)
    e_single_outpa = s_obs - np.asarray(y_single_pa, dtype=float)
    e_dual_outpa   = s_obs - np.asarray(y_dual_pa,   dtype=float)

    # Assemble full DataFrame
    df = pd.DataFrame({
        'n':                       np.arange(N),
        't_sec':                   np.arange(N) / FS,
        's_obs':                   s_obs,
        'e_single_padasip':        np.asarray(e_single_pa, dtype=float),
        'e_single_outof_padasip':  e_single_outpa,
        'e_single_manual':         e_single_man,
        'e_dual_padasip':          np.asarray(e_dual_pa, dtype=float),
        'e_dual_outof_padasip':    e_dual_outpa,
        'e_dual_manual':           e_dual_man,
    })

    # Resolve slice
    n_start = max(0, args.start)
    if args.count is None:
        n_end = N
    else:
        n_end = min(N, n_start + args.count)
    if n_end <= n_start:
        print(f"[ERROR] Empty slice: start={n_start}, end={n_end}")
        sys.exit(1)
    n_plot = n_end - n_start

    # Optional per-sample printout (controlled by --print N)
    if args.print_rows > 0:
        n_print_end = min(N, n_start + args.print_rows)
        n_print_rows = n_print_end - n_start
        print("\n" + "=" * 130)
        print(f"SAMPLE-LEVEL PRINTOUT  (record: {rec_name}, samples {n_start}..{n_print_end-1}, {n_print_rows} rows)")
        print("=" * 130)
        pd.set_option('display.width', 300)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.float_format', lambda v: f'{v: .6e}')
        print(df.iloc[n_start:n_print_end].to_string(index=False))

    # Optional save
    if args.save:
        df.to_csv(args.save, index=False)
        print(f"\nFull per-sample errors saved to: {args.save}")

    # Always: numerical sanity check on the FULL signal
    def _cmp(label, a, b):
        d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
        print(f"  {label:55s}  max|diff|={float(np.max(np.abs(d))):.3e}")

    print("\n" + "=" * 130)
    print(f"FULL-SIGNAL SANITY CHECK  (N = {N} samples)")
    print("=" * 130)
    print("[SINGLE-INPUT]")
    _cmp('e_padasip   vs  e_outof_padasip',
         df['e_single_padasip'].values, df['e_single_outof_padasip'].values)
    _cmp('e_padasip   vs  e_manual',
         df['e_single_padasip'].values, df['e_single_manual'].values)
    print("[DUAL-INPUT]")
    _cmp('e_padasip   vs  e_outof_padasip',
         df['e_dual_padasip'].values, df['e_dual_outof_padasip'].values)
    _cmp('e_padasip   vs  e_manual',
         df['e_dual_padasip'].values, df['e_dual_manual'].values)

    # Build the two figures over the chosen slice
    sl = slice(n_start, n_end)
    time_s = df['t_sec'].values[sl]

    plot_error_figure(
        time_s,
        df['e_single_padasip'].values[sl],
        df['e_single_outof_padasip'].values[sl],
        df['e_single_manual'].values[sl],
        rec_name, 'single',
    )

    plot_error_figure(
        time_s,
        df['e_dual_padasip'].values[sl],
        df['e_dual_outof_padasip'].values[sl],
        df['e_dual_manual'].values[sl],
        rec_name, 'dual',
    )

    print(f"\nSamples plotted: {n_plot}  (window {n_start/FS:.2f}s .. {n_end/FS:.2f}s)")
    plt.show()


if __name__ == '__main__':
    main()
