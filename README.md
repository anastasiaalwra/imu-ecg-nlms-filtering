# IMU-Assisted Adaptive ECG Filtering — Reproduction, Critique & Improvement

A reproduction and critical evaluation of an NLMS-based adaptive ECG-filtering method that uses dual-channel IMU signals to remove baseline drift and motion artifacts in real time, together with a DSP-corrected, placement-robust variant. Internship project in biosignal analysis (2026).

Reproduction target: Ou, Z., Ma, S., Zhao, Z., & Wang, H. (2025). A Study on NLMS-Based Adaptive ECG Filtering Algorithm Assisted by Dual-Channel IMU Signals. ISAIMS 2025.

# Key findings
Reproduced the paper's NLMS pipeline on the public PhysioNet PTT-PPG dataset (22 subjects × 3 activities: sit / walk / run, 500 Hz, finger IMU, 3-lead chest ECG).

The NLMS algorithm does removes baseline-drift (16–19 dB BPR across activities); the 0.16–40 Hz bandpass contributes ≈ 0 dB.

The **quaternion** reference carries no motion-specific information: a literal constant q ≡ 1.0 matches or beats the real quaternion in BPR across every activity (paired Wilcoxon, p < 0.001).

The paper's headline +10 pp R-peak-retention gain (single → dual) does not reproduce: the actual improvement is +0.02 to +0.15 pp — roughly 100× smaller.

Proposed a DSP-corrected variant: weighted multi-axis velocity reference, bandpass-prefiltered cross-correlation for delay estimation, and a 256-tap NLMS (instead of a single tap).

PPG transplant — negative result: the ECG-tuned pipeline is harmful on PPG-derived heart rate; the report gives the mechanistic explanation (possible leakage).


# Folder Overview
Strict reproduction of the paper pipeline: per-subject ECG calibration from WFDB headers, 0.16–40 Hz bandpass + 50 Hz notch, trapezoidal-rule velocity from IMU acceleration, small-angle quaternion update, cross-correlation alignment, single-tap dual-reference NLMS.

Ablations replacing the quaternion with motion-free surrogates (constant, linear ramp, slow noise).

Proper R-peak analysis (TP/FP/FN) against the dataset's annotated peaks with NeuroKit2 (±50 ms).

Statistical testing: paired Wilcoxon signed-rank tests per metric per activity.

Improved DSP variant and parameter sweeps over filter length L and step size μ.

Cross-modality validation: ECG-derived HR as ground truth for the PPG experiments (8s sliding windows, IEEE 1708a-2019 coverage).


# Repository structure

```

.
├── README.md
├── requirements.txt
├── .gitignore
├── report/
│   └── Internship_Report.pdf            # full write-up (methods, results, limitations)
├── src/
│   ├── reproduction/                    # strict reproduction of the paper
│   │   ├── benchmark_paper_strict.py    # main pipeline → results/benchmark_results_paper_strict.csv
│   │   ├── ablation_quaternion.py       # quaternion vs constant/ramp/noise surrogates
│   │   ├── diagnostic_quaternion.py     # gyroscope/quaternion statistics 
│   │   ├── diagnostic_peak_fp.py        # TP/FP/FN R-peak analysis (NeuroKit2)
│   │   └── statistical_tests.py         # paired Wilcoxon single-vs-dual
│   ├── improved/                        # DSP-corrected variant
│   │   ├── benchmark_improved.py
│   │   └── NLMS_parameters_test.py      # sweep over L and μ
│   ├── ppg/                             # PPG transplant (negative result)
│   │   ├── benchmark_paper_strict_ppg.py
│   │   ├── ablation_quaternion_ppg.py
│   │   └── nlms_plot_diagnostic_ppg.py
│   └── analysis/                        # aggregation, plots, cross-modality checks
│       ├── figures_file1.py
│       ├── table1_paper.py
│       ├── table2_paper_ppg.py
│       ├── metrics_cross_modality.py
│       ├── nlms_plot_errors.py
│       └── nlms_plot_diagnostic.py
├── results/                             # per-record metric CSVs
└── figures/                             # generated plots
```

# Data

The dataset is not included ( it is public). Download the PhysioNet Pulse Transit Time PPG dataset (v1.1.0) and place it in the project root as Physionet_PTT_Dataset/:

https://physionet.org/content/pulse-transit-time-ppg/
Setup
bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
How to run

The scripts use paths relative to their own folder (e.g. ../../results, ../../Physionet_PTT_Dataset), so run each one from inside its own directory.

bash
# Reproduction
cd src/reproduction
python benchmark_paper_strict.py
python ablation_quaternion.py
python diagnostic_quaternion.py
python diagnostic_peak_fp.py --full
python statistical_tests.py

# Analysis / figures
cd ../analysis
python table1_paper.py
python figures_file1.py
python nlms_plot_errors.py -h
python nlms_plot_diagnostic.py -h

# Improved variant
cd ../improved
python benchmark_improved.py
python NLMS_parameters_test.py

# PPG transplant (negative result)
cd ../ppg
python benchmark_paper_strict_ppg.py
python ablation_quaternion_ppg.py
python nlms_plot_diagnostic_ppg.py -h
cd ../analysis
python table2_paper_ppg.py
python metrics_cross_modality.py -h

# Tech stack

Python · NumPy · SciPy · pandas · NeuroKit2 · padasip · WFDB · matplotlib


Author
Anastasia Lora — BSc in Computer Science, University of Crete. 
GitHub: <https://github.com/anastasiaalwra> · LinkedIn: <www.linkedin.com/in/anastasia-lora-61b5822bb>