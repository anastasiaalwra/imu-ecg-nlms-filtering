"""
Diagnostic plotting for the NLMS adaptive ECG filter.

For each mode (single-input, dual-input) we draw 4 stacked subplots, all on a
COMMON y-axis so amplitudes are directly comparable:

    1) s_obs              the observed ECG fed to the NLMS (after BP + notch)
    2) y = W * x          the component that NLMS estimates and SUBTRACTS
                          (this is what the filter thinks is motion noise)
    3) e = s_obs - y      the CLEANED output (what the paper calls e[n])
    4) overlay            s_obs (grey, semi-transparent) + e (blue)
                          so you can see exactly what changed

The diagnostic question this answers:
    - Does y look like motion drift / wandering low-frequency content?
      -> filter is doing its job
    - Does y look like miniature QRS complexes (sharp spikes at beat positions)?
      -> filter is eating signal, not just noise -> red flag
    - Does the overlay show s_obs and e nearly identical?
      -> filter is barely active for this segment
"""

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


def plot_diagnostic_figure(time_s, s_obs, y_rem, e, record_name, mode):
    """
    4-row diagnostic figure with SHARED y-axis across rows.
      0: s_obs (observed input to NLMS)
      1: y = W·x (component subtracted by NLMS)
      2: e = s_obs - y (cleaned output)
      3: overlay s_obs vs e (semi-transparent grey + blue)
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, sharey=True)
    fig.suptitle(f"NLMS diagnostic ({mode}-input) — {record_name}")

    axes[0].plot(time_s, s_obs, linewidth=0.8, color="tab:cyan")
    axes[0].set_ylabel("s_obs (input)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_s, y_rem, linewidth=0.8, color="tab:red")
    axes[1].set_ylabel("y = W·x  (removed)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_s, e, linewidth=0.8, color="tab:orange")
    axes[2].set_ylabel(f"e_{mode}  (cleaned)")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(time_s, s_obs, linewidth=0.8, color="tab:cyan",
                 alpha=0.55, label="s_obs")
    axes[3].plot(time_s, e,     linewidth=0.8, color="tab:orange",
                 alpha=0.9,  label=f"e_{mode}")
    axes[3].set_ylabel("Overlay  s_obs / e")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend(loc="upper right")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()


def energy_ratio(s_obs_win, y_win):
    """Fraction of input energy that NLMS removed in this window."""
    num = float(np.sum(y_win ** 2))
    den = float(np.sum(s_obs_win ** 2)) + 1e-20
    return num / den


def main():
    parser = argparse.ArgumentParser(
        description='Diagnostic: s_obs vs NLMS-removed component vs cleaned output.')
    parser.add_argument('record', help='Record name (e.g. s20_run) OR full path to .csv')
    parser.add_argument('--start', type=int, default=0,
                        help='Starting sample index for the plotted slice (default 0)')
    parser.add_argument('--count', type=int, default=None,
                        help='Number of samples to plot (default: ALL from --start to end)')
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

    print("Running padasip NLMS (single)...")
    f1 = pa.filters.FilterNLMS(n=1, mu=MU, eps=EPSILON, w="zeros")
    y_single, e_single, _ = f1.run(s_obs, X_single)

    print("Running padasip NLMS (dual)...")
    f2 = pa.filters.FilterNLMS(n=2, mu=MU, eps=EPSILON, w="zeros")
    y_dual, e_dual, _ = f2.run(s_obs, X_dual)

    y_single = np.asarray(y_single, dtype=float)
    e_single = np.asarray(e_single, dtype=float)
    y_dual   = np.asarray(y_dual,   dtype=float)
    e_dual   = np.asarray(e_dual,   dtype=float)

    # Resolve slice
    n_start = max(0, args.start)
    n_end   = N if args.count is None else min(N, n_start + args.count)
    if n_end <= n_start:
        print(f"[ERROR] Empty slice: start={n_start}, end={n_end}")
        sys.exit(1)
    n_plot = n_end - n_start
    sl = slice(n_start, n_end)
    time_s = (np.arange(N) / FS)[sl]

        


    # Quantitative summary on the plotted window
    er_single = energy_ratio(s_obs[sl], y_single[sl])
    er_dual   = energy_ratio(s_obs[sl], y_dual[sl])

    print("\n" + "=" * 90)
    print(f"NLMS ENERGY REMOVED  (window: {n_start/FS:.2f}s .. {n_end/FS:.2f}s, {n_plot} samples)")
    print("=" * 90)
    print(f"  single-input:  ||y||^2 / ||s_obs||^2 = {er_single*100:.3f}%")
    print(f"  dual-input:    ||y||^2 / ||s_obs||^2 = {er_dual*100:.3f}%")
    print()

    # Plot
    plot_diagnostic_figure(time_s, s_obs[sl], y_single[sl], e_single[sl], rec_name, 'single')
    plot_diagnostic_figure(time_s, s_obs[sl], y_dual[sl],   e_dual[sl],   rec_name, 'dual')

    print(f"\nSamples plotted: {n_plot}  (window {n_start/FS:.2f}s .. {n_end/FS:.2f}s)")
    plt.show()


if __name__ == '__main__':
    main()
