"""
figures_file1.py

Generates the 3 key figures for File 1 presentation/paper:

  Figure 1: Time-domain overlay — s_raw / s_obs / s_clean_dual on a walking record
            Shows VISUALLY how much of the change is preprocessing vs NLMS.

  Figure 2: PSD before/after — confirms (or refutes) that NLMS removes 0–0.5 Hz
            power beyond what the bandpass already does.

  Figure 3: Oscillation Index by activity — bar chart showing the OI bombshell.
            High OI in walk/run means quaternion captures ~1% of real motion.

Run AFTER benchmark_paper_strict.py.
Outputs: figures/fig1_timedomain.png, fig2_psd.png, fig3_oi_bars.png
"""

# Path resolution: enables this script to be run from any cwd
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..'))
_os.chdir(_PROJECT_ROOT)
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import pearsonr
import neurokit2 as nk

# Reuse pipeline from benchmark for consistency
FS = 500.0
DT = 1.0 / FS
NYQ = 0.5 * FS
DATASET_DIR = os.path.join('Physionet_PTT_Dataset', 'physionet.org', 'files',
                           'pulse-transit-time-ppg', '1.1.0')
FIG_DIR = 'figures'
os.makedirs(FIG_DIR, exist_ok=True)

B_BP, A_BP = signal.butter(4, [0.16 / NYQ, 40.0 / NYQ], btype='bandpass')
B_NOTCH, A_NOTCH = signal.iirnotch(50.0, 10.0, FS)
B_LP, A_LP = signal.butter(2, 0.5 / NYQ, btype='lowpass')


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
    N, M = X.shape; W = np.zeros(M); e = np.empty(N)
    for n in range(N):
        x = X[n]; e_n = s_obs[n] - W @ x; e[n] = e_n
        W += mu * (e_n * x) / (x @ x + eps)
    return e


def run_pipeline(rec_name):
    """Run the full File 1 pipeline on a single record and return signals."""
    csv_path = os.path.join(DATASET_DIR, 'csv', rec_name + '.csv')
    hea_path = os.path.join(DATASET_DIR, rec_name + '.hea')
    df = pd.read_csv(csv_path)
    gain, baseline = load_calib(hea_path)
    s_raw = (df['ecg'].values.astype(float) - baseline) * gain / 1e6
    s_raw -= np.mean(s_raw)
    ax = df['a_x'].values.astype(float)
    wx = np.deg2rad(df['g_x'].values.astype(float))
    wy = np.deg2rad(df['g_y'].values.astype(float))
    wz = np.deg2rad(df['g_z'].values.astype(float))
    s_bp  = signal.filtfilt(B_BP, A_BP, s_raw)
    s_obs = signal.filtfilt(B_NOTCH, A_NOTCH, s_bp)
    ax_c = ax - np.mean(ax)
    ax_bp = signal.filtfilt(B_BP, A_BP, ax_c)
    v_ref = trap_int(ax_bp)
    q_ref = quat_real(wx, wy, wz)
    v_al = shift(v_ref, cross_corr_lag(v_ref, s_obs))
    q_al = shift(q_ref, cross_corr_lag(q_ref, s_obs))
    s_clean_single = nlms(s_obs, v_al.reshape(-1, 1))
    s_clean_dual   = nlms(s_obs, np.column_stack([v_al, q_al]))
    gt_peaks = np.where(df['peaks'].values == 1)[0]
    return {
        's_raw': s_raw, 's_obs': s_obs,
        's_clean_single': s_clean_single, 's_clean_dual': s_clean_dual,
        'q_ref': q_ref, 'gt_peaks': gt_peaks,
    }


def figure1_timedomain(rec_name='s1_walk', t_start=10.0, duration=10.0):
    """Time-domain comparison of raw / preprocessed / NLMS-filtered ECG."""
    print(f"[Fig 1] Time-domain overlay on {rec_name} ...")
    sigs = run_pipeline(rec_name)
    n0 = int(t_start * FS); n1 = n0 + int(duration * FS)
    t = np.arange(n0, n1) / FS

    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(t, sigs['s_raw'][n0:n1] * 1000, color='#888', lw=0.8)
    axes[0].set_ylabel('s_raw (mV)')
    axes[0].set_title(f'{rec_name}: raw calibrated ECG (after DC removal only)')
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, sigs['s_obs'][n0:n1] * 1000, color='#1f77b4', lw=0.8)
    axes[1].set_ylabel('s_obs (mV)')
    axes[1].set_title('After bandpass 0.16–40 Hz + notch 50 Hz (preprocessing)')
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, sigs['s_clean_dual'][n0:n1] * 1000, color='#d62728', lw=0.8)
    # Mark ground-truth R-peaks in this window
    gt = sigs['gt_peaks']
    gt_in = gt[(gt >= n0) & (gt < n1)]
    axes[2].plot(gt_in / FS, sigs['s_clean_dual'][gt_in] * 1000,
                 'kv', markersize=6, label='GT R-peaks')
    axes[2].set_ylabel('s_clean_dual (mV)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_title('After dual-input NLMS (acceleration + quaternion refs)')
    axes[2].grid(alpha=0.3); axes[2].legend(loc='upper right')

    fig.suptitle('Figure 1 — Visual comparison of pipeline stages', fontsize=13, y=1.00)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'fig1_timedomain.png')
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"    wrote {out}")


def figure2_psd(rec_name='s1_walk'):
    """PSD before vs after preprocessing vs after NLMS. Shows where reduction happens."""
    print(f"[Fig 2] PSD comparison on {rec_name} ...")
    sigs = run_pipeline(rec_name)
    f_raw,   p_raw   = signal.welch(sigs['s_raw'],          fs=FS, nperseg=4096)
    f_obs,   p_obs   = signal.welch(sigs['s_obs'],          fs=FS, nperseg=4096)
    f_clean, p_clean = signal.welch(sigs['s_clean_dual'],   fs=FS, nperseg=4096)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(f_raw,   p_raw,   color='#888',    label='raw (calibrated)',     lw=1.0)
    ax.semilogy(f_obs,   p_obs,   color='#1f77b4', label='after preprocessing',  lw=1.0)
    ax.semilogy(f_clean, p_clean, color='#d62728', label='after dual-NLMS',      lw=1.0)
    ax.axvspan(0, 0.5, color='orange', alpha=0.12, label='baseline band (0–0.5 Hz)')
    ax.set_xlim(0, 60)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (V² / Hz)')
    ax.set_title(f'Figure 2 — Power spectral density at each pipeline stage ({rec_name})')
    ax.grid(alpha=0.3, which='both')
    ax.legend()
    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'fig2_psd.png')
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"    wrote {out}")


def main():
    figure1_timedomain(rec_name='s1_walk', t_start=10.0, duration=10.0)
    figure2_psd(rec_name='s1_walk')
    print(f"\nAll figures written to ./{FIG_DIR}/")

if __name__ == '__main__':
    main()
