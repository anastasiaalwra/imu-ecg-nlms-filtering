"""
Statistical significance testing results.

For each activity (sit, walk, run) and each headline metric, runs a
paired Wilcoxon test comparing single-input NLMS vs
dual-input NLMS, using the per-record values.

Interpretation:
  p < 0.05  → dual is STATISTICALLY DIFFERENT from single (real effect)
  p >= 0.05 → cannot reject H0 (difference is consistent with noise)

If the paper claims a benefit (BPR +8 dB, retention +10 pp) but the
Wilcoxon yields p > 0.05 → reproducibility failure with statistical backing.
"""

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

INPUT_CSV = '../../results/benchmark_results_paper_strict.csv'
# Metric pairs to test: (single_column, dual_column, friendly_name, direction)
# direction = 'higher_is_better' or 'lower_is_better'
METRIC_PAIRS = [
    ('BPR_total_Single_dB',     'BPR_total_Dual_dB',     'BPR (total)',           'higher'),
    ('BPR_NLMS_only_Single_dB', 'BPR_NLMS_only_Dual_dB', 'BPR (NLMS only)',       'higher'),
    ('MSE_BL_Single_V2',        'MSE_BL_Dual_V2',        'MSE_BL',                'lower'),
    ('R_BL_Single_V',           'R_BL_Dual_V',           'R_BL',                  'lower'),
    ('PCC_Single',              'PCC_Dual',              'PCC (vs s_obs)',        'higher'),
    ('Retention_Single_%',      'Retention_Dual_%',      'Retention',             'higher'),
    ('dRR_Single_ms',           'dRR_Dual_ms',           'ΔRR (ms)',              'lower'),
]


def stars(p):
    if np.isnan(p):     return ''
    if p < 0.001:       return '***'
    if p < 0.01:        return '**'
    if p < 0.05:        return '*'
    return 'ns'   # not significant


def run_test(single_vals, dual_vals, direction):
    """
    Wilcoxon paired test. Returns (p_value, median_diff, conclusion_string).
    """
    # Drop pairs where either value is NaN
    mask = np.isfinite(single_vals) & np.isfinite(dual_vals)
    s = np.asarray(single_vals)[mask]
    d = np.asarray(dual_vals)[mask]
    if len(s) < 5:
        return np.nan, np.nan, 'too few samples'
    diff = d - s
    if np.all(diff == 0):
        return 1.0, 0.0, 'all zero differences'
    try:
        stat, p = wilcoxon(s, d, zero_method='wilcox', alternative='two-sided')
    except ValueError as e:
        return np.nan, float(np.median(diff)), f'wilcoxon failed: {e}'
    median_diff = float(np.median(diff))
    # Interpretation
    if p >= 0.05:
        verdict = 'NO significant difference'
    else:
        if direction == 'higher':
            verdict = 'DUAL better' if median_diff > 0 else 'DUAL worse'
        else:  # lower is better
            verdict = 'DUAL better' if median_diff < 0 else 'DUAL worse'
    return p, median_diff, verdict


def main():
    df = pd.read_csv(INPUT_CSV)
    df['activity'] = df['File'].str.extract(r'_(sit|walk|run)\.csv', expand=False)

    print("=" * 110)
    print("PAIRED WILCOXON SIGNED-RANK TESTS  —  Single-input NLMS vs Dual-input NLMS")
    print(f"Per-activity (n records per activity in parentheses).")
    print(f"Significance: *** p<0.001  ** p<0.01  * p<0.05  ns p>=0.05")
    print("=" * 110)

    rows = []
    for activity in ['sit', 'walk', 'run']:
        sub = df[df['activity'] == activity]
        if sub.empty:
            continue
        print(f"\n[{activity.upper()}]  n = {len(sub)}")
        print(f"  {'Metric':<22} {'Median Δ (D-S)':>16} {'p-value':>12}  {'sig':>5}  Verdict")
        print("  " + "-" * 100)
        for s_col, d_col, name, direction in METRIC_PAIRS:
            if s_col not in sub.columns or d_col not in sub.columns:
                continue
            p, med_diff, verdict = run_test(sub[s_col].values, sub[d_col].values, direction)
            star = stars(p)
            p_str = f"{p:.4f}" if np.isfinite(p) else "n/a"
            print(f"  {name:<22} {med_diff:>16.4g} {p_str:>12}  {star:>5}  {verdict}")
            rows.append({
                'activity': activity, 'metric': name,
                'median_diff_DminusS': med_diff,
                'p_value': p, 'significance': star, 'verdict': verdict,
            })
    out = pd.DataFrame(rows)
    out_path = '../../results/statistical_tests_results.csv'
    out.to_csv(out_path, index=False)
    print("\n" + "=" * 110)
    print(f"Wrote: {out_path}")
    print()


if __name__ == '__main__':
    main()