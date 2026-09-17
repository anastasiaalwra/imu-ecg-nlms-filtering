"""
table1_paper.py

Generates the formatted "Table 1" (mean ± std per activity) from the
per-record results of benchmark_paper_strict.py.

Run AFTER benchmark_paper_strict.py has produced benchmark_results_paper_strict.csv.
"""

import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

INPUT_CSV = '../../results/benchmark_results_paper_strict.csv'
OUTPUT_CSV = '../../results/table1_paper.csv'

# Load per-record results
df = pd.read_csv(INPUT_CSV)

# Extract activity from filename: sXX_{sit|walk|run}.csv
df['Activity'] = df['File'].str.extract(r'_(sit|walk|run)\.csv', expand=False)
df['Activity'] = df['Activity'].str.capitalize()  # 'Sit' / 'Walk' / 'Run'

# Scale MSE for display convenience (V² → mV²)
# Multiplying by 1e6 converts V² to mV². (Paper reports in V², τάξης 1e-10 – 1e-12.)
for col in ['MSE_BL_Single_V2', 'MSE_BL_Dual_V2']:
    if col in df.columns:
        df[col + '_mV2'] = df[col] * 1e6

# Columns to summarize
cols = [
    # Quaternion diagnostics
    'Q_mean', 'Q_std', 'Q_min', 'Q_max',
    # Oscillation Index (our diagnostic)
    'omega_mag_mean_dps', 'omega_mag_max_dps',
    'angle_traveled_deg', 'theta_net_end_deg', 'OscillationIndex',
    # BPR decomposition (THE big diagnostic — separates preproc from NLMS)
    'BPR_preproc_dB',
    'BPR_total_Single_dB', 'BPR_NLMS_only_Single_dB',
    'BPR_total_Dual_dB',   'BPR_NLMS_only_Dual_dB',
    # Other single-input metrics
    'MSE_BL_Single_V2_mV2', 'R_BL_Single_V',
    'PCC_Single', 'Retention_Single_%', 'dRR_Single_ms',
    # Other dual-input metrics
    'MSE_BL_Dual_V2_mV2', 'R_BL_Dual_V',
    'PCC_Dual', 'Retention_Dual_%', 'dRR_Dual_ms',
    # Gains from dual over single
    'BPR_total_gain_dB', 'BPR_NLMS_gain_dB',
    'MSE_reduction_%', 'PCC_gain_%', 'Retention_gain_pp',
]
cols = [c for c in cols if c in df.columns]

# Aggregate
summary = df.groupby('Activity')[cols].agg(['mean', 'std']).round(4)

# Format as "mean ± std" strings
summary_fmt = pd.DataFrame(index=summary.index)
for col in cols:
    mean_vals = summary[col]['mean'].apply(lambda x: f"{x:.4f}")
    std_vals  = summary[col]['std'].apply(lambda x: f"{x:.4f}")
    summary_fmt[col] = mean_vals + " ± " + std_vals

# Reorder rows: Sit, Walk, Run (low → high motion intensity)
activity_order = [a for a in ['Sit', 'Walk', 'Run'] if a in summary_fmt.index]
summary_fmt = summary_fmt.reindex(activity_order)

# Display
print("\n" + "=" * 100)
print("  TABLE 1 — PAPER STRICT REPRODUCTION (mean ± std per activity)")
print("=" * 100)
print(summary_fmt.T.to_string())   # Transpose: metrics as rows, activities as columns
print("=" * 100)

# Note for MSE scaling
print("\n[NOTE] MSE values are in mV² (V² × 1e6). Paper reports in V² (× 1e-10 to × 1e-12).")
print("       Divide displayed MSE by 1e6 to recover paper units.\n")


summary.columns = ['_'.join(c).strip() for c in summary.columns]
summary.to_csv(OUTPUT_CSV)
summary_fmt.to_csv(OUTPUT_CSV.replace('.csv', '_formatted2.csv'))
print(f"Wrote: {OUTPUT_CSV} (raw mean/std)")
print(f"Wrote: {OUTPUT_CSV.replace('.csv', '_formatted2.csv')} (mean ± std strings)\n")
