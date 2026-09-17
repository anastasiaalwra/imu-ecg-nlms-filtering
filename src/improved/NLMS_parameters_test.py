import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..'))
_os.chdir(_PROJECT_ROOT)
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

import os
import glob
import pandas as pd

# Re-use everything from the improved pipeline.
from src.improved import benchmark_improved as bi

L_GRID = [1, 16, 64, 128, 256]

# value multiplies total runtime by the number of records × L_GRID size.
MU_GRID =  [ 0.05, 0.1, 0.3, 0.5]                   

FULL = False                      # True for ALL 66 records (slow run).  False for subset (ideal to test quick).
SUBSET_PER_ACTIVITY = 1  # This variable can change, depending on how many subjects you wanna run.
ACTIVITIES = ['sit', 'walk', 'run']

OUT_CSV = os.path.join('results', 'NLMS_parameters_test.csv')


def select_records():
    """Choose the records to sweep over."""
    csv_dir = os.path.join(bi.DATASET_DIR, 'csv')
    all_csvs = sorted(glob.glob(os.path.join(csv_dir, '*.csv')))
    if FULL:
        return all_csvs
    chosen = []
    for act in ACTIVITIES:
        per_act = [p for p in all_csvs if p.endswith(f'_{act}.csv')]
        chosen.extend(per_act[:SUBSET_PER_ACTIVITY])
    return chosen


def main():
    csv_files = select_records()
    mu_values = MU_GRID if MU_GRID is not None else [bi.MU]
    total_runs = len(csv_files) * len(L_GRID) * len(mu_values)

    print(f"Records: {len(csv_files)}  |  L grid: {L_GRID}  |  "
          f"μ grid: {mu_values}  |  FULL mode: {FULL}")
    print(f"Total per-record runs: {total_runs}\n")

    rows = []
    for mu in mu_values:
        for L in L_GRID:
            print(f"\n===== L = {L}  |  μ = {mu} =====")
            for i, csv_path in enumerate(csv_files, 1):
                rec_name = os.path.basename(csv_path).replace('.csv', '')
                hea_path = os.path.join(bi.DATASET_DIR, rec_name + '.hea')
                if not os.path.exists(hea_path):
                    print(f"  [{i:2d}] SKIP {rec_name}: no .hea")
                    continue
                try:
                    row = bi.evaluate_record(csv_path, hea_path, L=L, mu=mu)
                    rows.append(row)
                    print(f"  [{i:2d}] {rec_name:14s}  "
                          f"BPR_dual={row['BPR_Dual_dB']:6.2f}  "
                          f"Ret_dual={row['Retention_Dual_%']:5.2f}")
                except Exception as e:
                    print(f"  [{i:2d}] ERROR {rec_name}: {e}")

    df = pd.DataFrame(rows)
    os.makedirs('results', exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote: {OUT_CSV}  ({len(df)} rows across {len(L_GRID)} L values)")

    if len(df):
        df['activity'] = df['File'].str.extract(r'_(sit|walk|run)\.csv', expand=False)
        group_cols = ['L_taps', 'mu', 'activity'] if MU_GRID is not None else ['L_taps', 'activity']
        agg = (df.groupby(group_cols)
                 [['BPR_Dual_dB', 'PCC_Dual', 'Retention_Dual_%']]
                 .mean()
                 .round(3))
        print(f"\n===== Per-({'/'.join(group_cols)}) means (Dual) =====")
        print(agg.to_string())


if __name__ == '__main__':
    main()
