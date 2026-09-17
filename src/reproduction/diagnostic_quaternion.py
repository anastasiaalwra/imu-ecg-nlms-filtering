"""
quaternion_analysis.py

Determine whether the near-constant quaternion real-part (Q ≈ 1.0)
observed in benchmark_paper_strict.py is:
  (a) A bug in the quaternion computation, OR
  (b) A feature of the dataset (small/cancelling rotations at the finger IMU)

This assumption is made due to the quaternion's comparison with the 3 surrogates. 
The output showed that a literal constant scores higher performance in the metrics. 

What this script does for a sample of records per activity:
  1. Reports raw gyroscope statistics |ω| in deg/s — is there motion at all?
  2. Reports cumulative rotation angle θ_total(N) in degrees — net rotation
     over the record (high |ω| can still give low net θ if rotations cancel)
  3. Reports quaternion real-part statistics over time
  4. Cross-checks: cos(θ_total/2) should match q_real(end)
"""

import os
import glob
import numpy as np
import pandas as pd

FS = 500.0
DT = 1.0 / FS
DATASET_DIR = os.path.join('..','..','Physionet_PTT_Dataset', 'physionet.org', 'files',
                           'pulse-transit-time-ppg', '1.1.0')


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1; w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def analyze(csv_path):
    df = pd.read_csv(csv_path)
    # Gyroscope: deg/s
    wx_dps = df['g_x'].values.astype(float)
    wy_dps = df['g_y'].values.astype(float)
    wz_dps = df['g_z'].values.astype(float)
    # In rad/s for the quaternion math
    wx = np.deg2rad(wx_dps); 
    wy = np.deg2rad(wy_dps); 
    wz = np.deg2rad(wz_dps)

    # |ω| in deg/s -  motion intensity - metro dianysmatos gwniakhs taxythtas. 
    omega_mag_dps = np.sqrt(wx_dps**2 + wy_dps**2 + wz_dps**2)

    # sum of all movements. It is the angular distance; how much the angle moved in total/ arc length. 
    angle_traveled_deg = np.sum(omega_mag_dps) * DT  # = ∫|ω|dt

    # NET rotation: compute via quaternion (real part = cos(θ_net/2))
    N = len(wx)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    q_real = np.empty(N, dtype=float)
    q_real[0] = 1.0
    for n in range(1, N):
        dq = np.array([1.0, 0.5*wx[n]*DT, 0.5*wy[n]*DT, 0.5*wz[n]*DT])
        q = quat_mul(dq, q); q /= np.linalg.norm(q)
        q_real[n] = q[0]

    # Recover θ_net at end from q_real (paper's "real part captures rotation")
    # q_real = cos(θ_net/2)  =>  θ_net = 2·arccos(q_real); net displacement/ Gwniakh Metatopish. 
    theta_net_end_deg = float(np.rad2deg(2.0 * np.arccos(np.clip(q_real[-1], -1, 1))))

    return {
        'File': os.path.basename(csv_path),
        # Gyroscope (motion intensity)
        '|w|_mean_dps':   float(np.mean(omega_mag_dps)),
        '|w|_max_dps':    float(np.max(omega_mag_dps)),
        '|w|_std_dps':    float(np.std(omega_mag_dps)),
        # Total angular distance traveled (sum of |ω|·dt, NOT net rotation)
        'angle_traveled_deg': angle_traveled_deg,
        # Net rotation (from quaternion)
        'theta_net_end_deg':  theta_net_end_deg,
        # Quaternion real part stats
        'Q_mean': float(np.mean(q_real)),
        'Q_min':  float(np.min(q_real)),
        'Q_max':  float(np.max(q_real)),
        'Q_std':  float(np.std(q_real)),
        # Net rotation derived from Q_min (max deviation point)
        'theta_max_dev_deg': float(np.rad2deg(2.0 * np.arccos(np.clip(np.min(q_real), -1, 1)))),
    }


def main():
    # Pick 2 records per activity for a quick scan
    sample = ['s1_sit', 's5_sit','s12_sit', 's20_sit', 
              's1_walk', 's5_walk','s12_walk','s20_walk',
              's1_run', 's5_run', 's12_run','s20_run']

    results = []
    for rec in sample:
        csv_path = os.path.join(DATASET_DIR, 'csv', rec + '.csv')
        if not os.path.exists(csv_path):
            print(f"[skip] {rec}: file not found")
            continue
        results.append(analyze(csv_path))

    df = pd.DataFrame(results)
    pd.set_option('display.width', 200)
    print(df.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))

    print("\nINTERPRETATION KEY:")
    print("  |w|_mean_dps         : average rotation speed (high → lots of rotation)")
    print("  angle_traveled_deg   : ∫|ω|·dt  (high → lots of motion, regardless of direction)")
    print("  theta_net_end_deg    : NET rotation (cumulative direction-aware)")
    print("  Q_min                : if ≈ 1.0 → quaternion barely deviates → near-constant ref")
    print("  theta_max_dev_deg    : maximum NET rotation reached (from q_real_min)")
    print()


if __name__ == '__main__':
    main()