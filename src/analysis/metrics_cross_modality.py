"""
Cross-modality validation of the NLMS-based motion artifact cancellation,
where:
    HR derived from PPG (raw / NLMS-single / NLMS-dual) is compared against
    HR derived from a clean ECG of the same subject in the same time window.

WHY THIS METHODOLOGY:
    The PTT-PPG dataset records ECG and PPG simultaneously on the same body.
    ECG provides a clean, motion-resilient cardiac reference. We treat
    ECG-derived HR as the ground truth and ask: does the NLMS-cleaned PPG
    track that ground truth better than the raw PPG?
    This avoids two known traps:
      1) Circular metrics where the "truth" is the same signal being judged
         (e.g. comparing detected peaks of e against detected peaks of s_obs).
      2) Metrics manipulable by signal destruction (BPR, MSE) where zeroing
         the signal "improves" the score.

REFERENCES THAT VALIDATE THIS APPROACH:
    - Reiss, A. et al. (2019). "Deep PPG: Large-Scale Heart Rate Estimation
      with Convolutional Neural Networks." Sensors 19(14):3079.
        --> Establishes 8 sec sliding window + ECG-derived HR as PPG benchmark.
    - Schaeck, T. et al. (2017). "Computationally efficient heart rate
      estimation during physical exercise using PPG signals." EUSIPCO 2017.
        --> Same methodology applied to exercise (running) data.
    - IEEE Std 1708a-2019. Wearable, Cuffless Blood Pressure Measuring Devices.
        --> Official tolerance thresholds; <5 bpm is the industry coverage point.

GROUND-TRUTH SOURCE:
    The PTT-PPG dataset includes a binary `peaks` column in each CSV with
    R-peaks of the ECG (1 = peak sample, 0 = no peak).

    Columns sorted by activity:
        HR_MAE_bpm           : mean absolute error in bpm (lower is better)
        Coverage_5bpm_%      : fraction of windows where |error| < 5 bpm
        N_windows            : number of valid windows that entered the average
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
import sys                                            
import argparse                                      
import numpy as np                                   
import pandas as pd                                  
from scipy import signal                           
import neurokit2 as nk  # ECG and PPG peak detection

try:
    import padasip as pa  # NLMS filter implementation
except ImportError:
    raise ImportError("padasip not installed. Run: pip install padasip")


# Sampling rate of the PTT-PPG dataset (Hz). All recordings are 500 Hz.
FS = 500.0
DT = 1.0 / FS  # sample period in seconds
NYQ = 0.5 * FS                                      
# NLMS hyperparameters.
MU = 0.05 # NLMS learning rate
EPSILON = 1e-6                                       

# Path to the PhysioNet PTT-PPG dataset root (csv + hea files inside).
DATASET_DIR = os.path.join('Physionet_PTT_Dataset', 'physionet.org', 'files',
                           'pulse-transit-time-ppg', '1.1.0')

PPG_COL = 'pleth_2'

# Both ECG and PPG go through the same preprocessing as the strict reproduction
# (bandpass + 50 Hz notch).
B_BP, A_BP       = signal.butter(4, [0.16 / NYQ, 40.0 / NYQ], btype='bandpass')
B_NOTCH, A_NOTCH = signal.iirnotch(w0=50.0, Q=10.0, fs=FS)

# Windowing for HR comparison
'''
 Reiss et al. 2019 ("Deep PPG") established 8 sec windows as the de facto
 benchmark window for HR validation against ECG. Step 2 sec gives 75% overlap.
'''
HR_WIN_SEC  = 8.0  # window duration in seconds
HR_STEP_SEC = 2.0  # window step in seconds
HR_WIN_LEN  = int(HR_WIN_SEC * FS) # window length in samples (4000)
HR_STEP_LEN = int(HR_STEP_SEC * FS) # window step in samples (1000)

'''
Minimum beats per window to compute a meaningful HR.
Need at least 2 RR intervals to be averaged.
Verified visually as well while zooming in and
found 3 peaks in the range of two seconds for both 
ECG and PPG.  
'''
HR_MIN_BEATS = 3

# Physiologically plausible HR range. Anything outside is flagged as invalid
# and the window is excluded from the average. Adult resting min ~40 bpm,
# exercise max ~220 bpm.
HR_MIN_BPM = 40.0
HR_MAX_BPM = 220.0

ACTIVITY_KEYS = ('sit', 'walk', 'run')

def load_calib_for_column(hea_path, col_name):
    with open(hea_path, 'r') as f:                   
        for line in f:                               
            parts = line.strip().split()         
            if len(parts) >= 9 and parts[-1] == col_name:
                # parts[2] is e.g. "200(1024)/mV" — gain in parens contains baseline.
                m = re.match(r'([-+0-9.eE]+)\(([-+0-9]+)\)', parts[2])
                if m:  # if regex matched, return them
                    return float(m.group(1)), int(m.group(2))
    # Fall back to identity calibration if column missing (PPG sometimes lacks it).
    return 1.0, 0


def trap_int(x):
    """Trapezoidal numerical integration with dt = 1/FS. Used to derive
    velocity from acceleration (paper Sec. 2.1)."""
    v = np.zeros_like(x, dtype=float) # output buffer, same shape as x
    # cumulative trapezoid: each step adds the average of consecutive samples * dt
    v[1:] = np.cumsum((x[1:] + x[:-1]) * (DT / 2.0))
    return v  # return the integrated signal


def quat_mul(q1, q2):
    """ Quaternion Multiplication q1*q2. Used in the quaternion
    integration step of the IMU reference computation."""
    w1, x1, y1, z1 = q1                              
    w2, x2, y2, z2 = q2                              
    return np.array([                                
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_real(wx, wy, wz):
    """Real-part trajectory of orientation quaternion from gyroscope rates
    (rad/s). Paper Eqs. (4)-(6). Returns q[0][n] for each sample n."""
    N = len(wx)   # number of samples
    q = np.array([1.0, 0.0, 0.0, 0.0])               
    out = np.empty(N, dtype=float) # store real part of quaternion at each step.
    out[0] = 1.0                                     
    for n in range(1, N): # integrate over time
        # small-angle quaternion increment dq for one sample of body-rate
        dq = np.array([1.0,
                       0.5 * wx[n] * DT,
                       0.5 * wy[n] * DT,
                       0.5 * wz[n] * DT])
        q = quat_mul(dq, q)                          
        q /= np.linalg.norm(q)                       
        out[n] = q[0]                                 
    return out


def cross_corr_lag(ref, target, max_lag=500):
    """Find integer lag (in samples) that maximises Pearson cross-correlation
    of ref vs target. Used to align IMU references with the cardiac signal
    (paper Sec. 2.3)."""
    # z-normalise both signals so correlation magnitude is comparable
    r = (ref    - np.mean(ref))    / (np.std(ref)    + 1e-12)
    t = (target - np.mean(target)) / (np.std(target) + 1e-12)
    corr = signal.correlate(t, r, mode='full')  # all-lags cross-correlation
    lags = signal.correlation_lags(len(t), len(r), mode='full')
    mask = (lags >= -max_lag) & (lags <= max_lag)     # restrict to range of max_lag samples
    return int(lags[mask][np.argmax(corr[mask])])     # return lag of maximum correlation


def shift_signal(s, k):
    if k > 0: # shift right: pad zeros at front
        return np.pad(s, (k, 0), mode='constant')[:-k]
    elif k < 0: # shift left: pad zeros at end
        return np.pad(s, (0, -k), mode='constant')[-k:]
    return s # no shift


def preprocess_ecg(df, hea_path):
    """Convert raw ECG counts to mV-scale and apply BP + notch. Returns the
    s_obs(ECG) signal used both as ground-truth source AND as the input to
    the ECG-side NLMS."""
    gain, baseline = load_calib_for_column(hea_path, 'ecg')  # WFDB calibration
    s_raw = (df['ecg'].values.astype(float) - baseline) * gain / 1e6
    s_raw -= np.mean(s_raw)                           
    s_bp  = signal.filtfilt(B_BP,    A_BP,    s_raw)  # bandpass filter
    s_obs = signal.filtfilt(B_NOTCH, A_NOTCH, s_bp)   # notchfilter 
    return s_obs


def preprocess_ppg(df, hea_path):
    """Same preprocessing chain as ECG, applied to pleth_2. Mirrors the
    strict-paper transplant."""
    gain, baseline = load_calib_for_column(hea_path, PPG_COL)
    p_raw = (df[PPG_COL].values.astype(float) - baseline) * gain 
    p_raw -= np.mean(p_raw)                           
    p_bp  = signal.filtfilt(B_BP,    A_BP,    p_raw)  # same bandpass
    s_obs = signal.filtfilt(B_NOTCH, A_NOTCH, p_bp)   # same notch 
    return s_obs


def build_imu_references(df, target_signal):
    """Build aligned (v_al, q_al) references that the NLMS will use.
    target_signal is what the alignment is timed against (PPG s_obs here)."""
    ax = df['a_x'].values.astype(float)               
    # convert deg to rad/sec.
    wx = np.deg2rad(df['g_x'].values.astype(float))   
    wy = np.deg2rad(df['g_y'].values.astype(float))   
    wz = np.deg2rad(df['g_z'].values.astype(float))  

    ax_c  = ax - np.mean(ax) # remove gravity DC component as paper mentions.
    ax_bp = signal.filtfilt(B_BP, A_BP, ax_c) # same bandpass as the signal side
    v_ref = trap_int(ax_bp)                           
    q_ref = quat_real(wx, wy, wz)                     

    tau_v = cross_corr_lag(v_ref, target_signal)
    tau_q = cross_corr_lag(q_ref, target_signal)
    v_al  = shift_signal(v_ref, tau_v) # aligned velocity reference
    q_al  = shift_signal(q_ref, tau_q) # aligned quaternion reference
    return v_al, q_al


def run_nlms_single_dual(s_obs, v_al, q_al):
    """Run padasip NLMS twice on the same s_obs: once with v_al only (single),
    once with [v_al, q_al] (dual). Returns the two cleaned outputs."""
    X_single = v_al.reshape(-1, 1)  # (N,1) reference for single
    X_dual   = np.column_stack([v_al, q_al]) # (N,2) reference for dual

    # FilterNLMS with 1 tap and zero initial weights — matches paper setup.
    f1 = pa.filters.FilterNLMS(n=1, mu=MU, eps=EPSILON, w="zeros")
    # padasip returns (y, e, w_history); we only need e (the cleaned signal).
    _, e_single, _ = f1.run(s_obs, X_single)

    f2 = pa.filters.FilterNLMS(n=2, mu=MU, eps=EPSILON, w="zeros")
    _, e_dual,   _ = f2.run(s_obs, X_dual)

    # Cast padasip outputs to plain float arrays for downstream detectors.
    return np.asarray(e_single, dtype=float), np.asarray(e_dual, dtype=float)


# PEAK DETECTION

def detect_ppg_peaks(ppg_sig):
    """Systolic-peak detection on PPG using NeuroKit2. Returns sample indices."""
    try:
        ppg_cl = nk.ppg_clean(ppg_sig, sampling_rate=FS)
        info   = nk.ppg_findpeaks(ppg_cl, sampling_rate=FS)
        return np.array(info['PPG_Peaks'], dtype=int)
    except Exception:
        return np.array([], dtype=int)


# HR FROM PEAKS

def hr_in_window(peak_indices, win_start_sample, win_end_sample):
    """
    Compute one HR value (bpm) from the peaks that fall inside a window.
    Returns NaN if too few peaks or if HR is implausible.

    Method: HR = 60 / median(RR_intervals_in_window_seconds).
    Using median (not mean) makes it robust to a single missed/extra beat.
    """
    # Select only the peaks that lie inside this window.
    in_win = peak_indices[(peak_indices >= win_start_sample) &
                          (peak_indices <  win_end_sample)]
    # Need at least HR_MIN_BEATS peaks to produce ≥2 RR intervals.
    if len(in_win) < HR_MIN_BEATS:
        return np.nan
    # RR intervals in seconds: consecutive differences / FS.
    rr_sec = np.diff(in_win) / FS
    # Guard against zero/negative (shouldn't happen but be safe).
    rr_sec = rr_sec[rr_sec > 0]
    if len(rr_sec) < 1:
        return np.nan
    hr = 60.0 / float(np.median(rr_sec))              # median for robustness
    # Reject windows where the inferred HR is non-physiological.
    if hr < HR_MIN_BPM or hr > HR_MAX_BPM:
        return np.nan
    return hr


def hr_series_over_windows(peaks, n_samples):
    """
    Slide a HR_WIN_SEC window with HR_STEP_SEC step over a recording.
    Returns (window_centers_sec, hr_bpm_array). Both same length.
    """
    centers = []                                      # midpoints of each window (sec)
    hrs     = []                                      # HR per window (bpm, may be NaN)
    # Iterate window start in samples; last window must end within n_samples.
    start = 0
    while start + HR_WIN_LEN <= n_samples:
        end = start + HR_WIN_LEN                      # window end (exclusive)
        hr = hr_in_window(peaks, start, end)          # one HR for this window
        centers.append((start + end) / 2.0 / FS)      # center time in seconds
        hrs.append(hr)
        start += HR_STEP_LEN                          # advance by HR_STEP_LEN samples
    return np.asarray(centers), np.asarray(hrs)



def compare_hr_series(hr_truth, hr_estimate):
    # Pair each window: keep only those where BOTH series produced a valid HR.
    valid = ~np.isnan(hr_truth) & ~np.isnan(hr_estimate)
    n_valid = int(np.sum(valid))  # how many windows survived
    if n_valid == 0:
        # If nothing matches, return NaNs and zero count.
        return {'HR_MAE_bpm': np.nan,
                'Coverage_5bpm_%': np.nan,
                'N_windows': 0}
    abs_err = np.abs(hr_estimate[valid] - hr_truth[valid])  # |error| per window
    return {
        'HR_MAE_bpm':      float(np.mean(abs_err)),        
        'Coverage_5bpm_%': float(100.0 * np.mean(abs_err < 5.0)), 
        'N_windows':       n_valid,
    }


def infer_activity(rec_name):
    """Return 'sit'/'walk'/'run' from a record name like 's12_walk', else
    'unknown'. Used to aggregate per activity in the summary table."""
    rec_low = rec_name.lower()
    for key in ACTIVITY_KEYS:                       
        if key in rec_low:
            return key
    return 'unknown'


def evaluate_record(csv_path, hea_path):
    """
    Full pipeline for one recording:
      1. read WFDB-annotated R-peaks from the `peaks` column (= truth)
      2. preprocess PPG (same chain as the strict paper transplant)
      3. build IMU references aligned to PPG s_obs
      4. run NLMS single and dual on the PPG side
      5. detect peaks on each PPG variant (raw/single/dual) with NeuroKit2
      6. compute HR per window for each, compare against ECG truth
      7. return one dict of metrics for this record
    """
    df = pd.read_csv(csv_path) # load the full CSV once
    n_samples = len(df)                               

    if 'peaks' not in df.columns:
        raise RuntimeError(f"`peaks` column missing in {csv_path}")
    ecg_peaks = np.where(df['peaks'].values == 1)[0]  # sample indices where peak=1

    # Sanity check: need enough annotated peaks to compute any HR.
    if len(ecg_peaks) < HR_MIN_BEATS:
        raise RuntimeError(f"Too few annotated R-peaks in {csv_path}")

    centers, hr_truth = hr_series_over_windows(ecg_peaks, n_samples)

    # PPG path: build the three variants we want to compare
    ppg_sobs       = preprocess_ppg(df, hea_path)     # raw PPG after BP+notch
    v_al, q_al     = build_imu_references(df, ppg_sobs)
    ppg_e_single, ppg_e_dual = run_nlms_single_dual(ppg_sobs, v_al, q_al)

    # Detect peaks on each PPG variant
    pks_raw    = detect_ppg_peaks(ppg_sobs)           # baseline: no NLMS
    pks_single = detect_ppg_peaks(ppg_e_single)       # after single NLMS
    pks_dual   = detect_ppg_peaks(ppg_e_dual)         # after dual NLMS

    # HR series for each variant on the same window grid
    _, hr_raw    = hr_series_over_windows(pks_raw,    n_samples)
    _, hr_single = hr_series_over_windows(pks_single, n_samples)
    _, hr_dual   = hr_series_over_windows(pks_dual,   n_samples)

    # Compare each PPG variant against the ECG ground truth
    m_raw    = compare_hr_series(hr_truth, hr_raw)
    m_single = compare_hr_series(hr_truth, hr_single)
    m_dual   = compare_hr_series(hr_truth, hr_dual)

    rec_name = os.path.basename(csv_path).replace('.csv', '')
    activity = infer_activity(rec_name)

    return {
        'File':      os.path.basename(csv_path),
        'Record':    rec_name,
        'Activity':  activity,
        'N_total_windows': len(hr_truth), # how many windows we scanned
        # Raw PPG (no NLMS): two metrics + window count
        'MAE_raw_bpm':    m_raw['HR_MAE_bpm'],
        'Cov5_raw_%':     m_raw['Coverage_5bpm_%'],
        'Nwin_raw':       m_raw['N_windows'],
        # Single-input NLMS
        'MAE_single_bpm': m_single['HR_MAE_bpm'],
        'Cov5_single_%':  m_single['Coverage_5bpm_%'],
        'Nwin_single':    m_single['N_windows'],
        # Dual-input NLMS
        'MAE_dual_bpm':   m_dual['HR_MAE_bpm'],
        'Cov5_dual_%':    m_dual['Coverage_5bpm_%'],
        'Nwin_dual':      m_dual['N_windows'],
        # Deltas: positive = NLMS improvement over raw (raw_MAE − variant_MAE)
        'd_MAE_raw_to_single':  m_raw['HR_MAE_bpm']    - m_single['HR_MAE_bpm'],
        'd_MAE_raw_to_dual':    m_raw['HR_MAE_bpm']    - m_dual['HR_MAE_bpm'],
        'd_MAE_single_to_dual': m_single['HR_MAE_bpm'] - m_dual['HR_MAE_bpm'],
    }


def per_activity_summary(df_results):
    """
    Aggregate per-record metrics to per-activity rows (sit / walk / run).
    Uses N_windows-weighted means so that long recordings count more.
    """
    rows = []                                        
    for act in list(ACTIVITY_KEYS) + ['unknown', 'ALL']:
        if act == 'ALL':
            sub = df_results                        
        else:
            sub = df_results[df_results['Activity'] == act]
        if len(sub) == 0:
            continue # skip activities not present

        def wmean(value_col, weight_col):
            v = sub[value_col].values.astype(float)
            w = sub[weight_col].values.astype(float)
            mask = ~np.isnan(v) & (w > 0) # ignore NaNs and zero-weight rows
            if not np.any(mask):
                return np.nan
            return float(np.sum(v[mask] * w[mask]) / np.sum(w[mask]))

        row = {
            'Activity':  act,
            'N_records': len(sub),
            # Each variant: weighted MAE + weighted Cov5% + total window count.
            'MAE_raw_bpm':       wmean('MAE_raw_bpm',    'Nwin_raw'),
            'Cov5_raw_%':        wmean('Cov5_raw_%',     'Nwin_raw'),
            'Nwin_raw_total':    int(sub['Nwin_raw'].sum()),

            'MAE_single_bpm':    wmean('MAE_single_bpm', 'Nwin_single'),
            'Cov5_single_%':     wmean('Cov5_single_%',  'Nwin_single'),
            'Nwin_single_total': int(sub['Nwin_single'].sum()),

            'MAE_dual_bpm':      wmean('MAE_dual_bpm',   'Nwin_dual'),
            'Cov5_dual_%':       wmean('Cov5_dual_%',    'Nwin_dual'),
            'Nwin_dual_total':   int(sub['Nwin_dual'].sum()),
        }
        rows.append(row)
    return pd.DataFrame(rows)


# MAIN function
def main():
    parser = argparse.ArgumentParser(
        description='Cross-modality NLMS evaluation: PPG-HR vs ECG-HR.')
    parser.add_argument('--record', type=str, default=None,
                        help='Process a single record name (e.g. s12_walk). '
                             'If omitted, process every CSV in the dataset.')
    parser.add_argument('--out_records', type=str,
                        default=os.path.join('results', 'metrics_cross_modality_results.csv'),
                        help='Per-record CSV output filename.')
    parser.add_argument('--out_summary', type=str,
                        default=os.path.join('results', 'metrics_cross_modality_summary.csv'),
                        help='Per-activity summary CSV output filename.')
    args = parser.parse_args()

    csv_dir = os.path.join(DATASET_DIR, 'csv')        
    if args.record:
        # Single-record path: derive csv/hea from the given record name.
        csv_path = os.path.join(csv_dir, args.record + '.csv')
        hea_path = os.path.join(DATASET_DIR, args.record + '.hea')
        if not os.path.exists(csv_path):
            print(f"[ERROR] CSV not found: {csv_path}")
            sys.exit(1)
        csv_files = [csv_path]
    else:
        csv_files = sorted(glob.glob(os.path.join(csv_dir, '*.csv')))
        if not csv_files:
            print(f"[ERROR] No CSV files found under {csv_dir}")
            sys.exit(1)

    print(f"\nCross-modality NLMS evaluation (ECG-truth vs PPG-{PPG_COL})")
    print(f"Records to process: {len(csv_files)}")
    print(f"Truth: WFDB `peaks` column (R-peak annotations) — no detector on truth side")
    print(f"Window: {HR_WIN_SEC}s, step: {HR_STEP_SEC}s, tolerance: 5 bpm")
    print("=" * 70)

    results = []                                      
    for i, csv_path in enumerate(csv_files, start=1):
        rec = os.path.basename(csv_path).replace('.csv', '')
        hea_path = os.path.join(DATASET_DIR, rec + '.hea')
        if not os.path.exists(hea_path):
            print(f"[{i:2d}/{len(csv_files)}] SKIP (no .hea): {rec}")
            continue
        print(f"[{i:2d}/{len(csv_files)}] {rec} ...", end=' ', flush=True)
        try:
            row = evaluate_record(csv_path, hea_path)
            results.append(row)
            print(f"act={row['Activity']:<5} "
                  f"MAE raw/single/dual = "
                  f"{row['MAE_raw_bpm']:5.2f} / "
                  f"{row['MAE_single_bpm']:5.2f} / "
                  f"{row['MAE_dual_bpm']:5.2f} bpm  "
                  f"(N={row['N_total_windows']})")
        except Exception as ex:
            print(f"ERROR: {ex}")

    if not results:
        print("\nNo records evaluated successfully.")
        sys.exit(1)

    # Per-record table to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(args.out_records, index=False)
    print(f"\nPer-record metrics written to: {args.out_records}")

    # Per-activity summary table to CSV
    df_summary = per_activity_summary(df_results)
    df_summary.to_csv(args.out_summary, index=False)
    print(f"Per-activity summary written to: {args.out_summary}")

    # Pretty-print the summary so user can see it immediately
    print("\n" + "=" * 70)
    print("PER-ACTIVITY SUMMARY (weighted by window count)")
    print("=" * 70)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.float_format', lambda v: f'{v:7.2f}')
    print(df_summary.to_string(index=False))


if __name__ == '__main__':
    main()
