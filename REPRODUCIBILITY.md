# Reproducibility Guide

**Synthetic Sleep Environment Dataset Generator**
TECHIN 513 Final Project — Team 7 (Rushav Dash, Lisa Li)

Every result in our report was produced by running `demo.py` with the
parameters documented below.  Starting from a fresh environment, the
following steps reproduce all figures, metrics, and dataset files exactly.

---

## Environment

| Parameter | Value |
|---|---|
| Python version | 3.12.3 |
| OS tested on | Ubuntu 24.04 (Linux 6.17) |
| Global random seed | **42** |
| Total wall time | ~31 seconds |

---

## Step-by-Step Reproduction

### 1. Clone the repository

```bash
git clone https://github.com/rushavsd/Techin513_Final.git
cd Techin513_Final
```

### 2. Create and activate the virtual environment

```bash
python3 -m venv venv
source venv/bin/activate    # Linux/macOS
# venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Dependencies installed:
- numpy >= 1.26
- scipy >= 1.11
- matplotlib >= 3.7
- pandas >= 2.0
- scikit-learn >= 1.3
- seaborn >= 0.12
- jupyter >= 1.0

### 4. Run the full pipeline

```bash
python demo.py
```

This single command generates all results.  Progress is logged to stdout.

---

## Expected Outputs

After running `python demo.py`, the following files are created:

### Dataset
| File | Description |
|---|---|
| `data/synthetic_sleep_dataset.csv` | 2,500 × 38 CSV (34 features + 4 labels + 3 metadata columns) |

### Figures (all at 300 DPI)
| File | Description |
|---|---|
| `results/figures/F02_example_session.png` | 4-panel time-series for one representative session |
| `results/figures/F03_temperature_spectrum.png` | Averaged FFT spectrum with HVAC frequency marker |
| `results/figures/F04_filter_effect.png` | Before/after Butterworth filter comparison |
| `results/figures/F05_poisson_events.png` | Light + noise Poisson event visualization |
| `results/figures/F06_sleep_label_distributions.png` | Histograms of all 4 sleep quality targets |
| `results/figures/F07_feature_correlation.png` | Feature–label Pearson correlation heatmap |
| `results/figures/F08_feature_importance.png` | Random Forest feature importance ranking |
| `results/figures/F09_model_performance.png` | Predicted vs. actual scatter (4 targets) |
| `results/figures/F10_ablation_study.png` | Ablation ΔR² bar chart |
| `results/figures/F11_ks_test_comparison.png` | KS-test CDF comparisons (6 variables) |
| `results/figures/F12_seasonal_comparison.png` | Sleep efficiency by season and quality class |

### Metrics
| File | Description |
|---|---|
| `results/metrics/dataset_summary.json` | Dataset statistics (mean, std, min, max per label) |
| `results/metrics/ml_results.json` | Train/val/test metrics, CV results, top features |
| `results/metrics/validation_report.json` | Full validation results (KS, discriminability, sanity) |
| `results/metrics/ks_test_results.csv` | Per-variable KS statistics and p-values |
| `results/metrics/sanity_checks.csv` | 6 sleep science sanity check results |
| `results/metrics/ablation_results.csv` | Ablation study: ΔR² per SP component |

---

## Expected Numerical Results (seed=42)

The following numbers appear verbatim in the report.  After running
`python demo.py`, they can be verified in `results/metrics/`:

### Dataset statistics (`dataset_summary.json`)
```
sleep_efficiency:  mean=0.7668, std=0.1299
sleep_duration_h:  mean=6.1459, std=1.0626
awakenings:        mean=2.9084, std=1.3951
sleep_score:       mean=77.8841, std=11.5599
```

### ML results (`ml_results.json`)
```
Random Forest test mean R²:  0.7949
Mean baseline test R²:       -0.0048
Ridge regression test R²:    0.7514
CV sleep_efficiency R²:      0.8964 ± 0.0047
CV awakenings R²:            0.7653 ± 0.0141
```

### Ablation results (`ablation_results.csv`)
```
full_pipeline:  R²=0.7949  ΔR²= 0.0000
no_lpf:         R²=0.7882  ΔR²=-0.0067
no_pink:        R²=0.7954  ΔR²=+0.0005
no_poisson:     R²=0.2236  ΔR²=-0.5713
no_hvac:        R²=0.7975  ΔR²=+0.0026
```

### Sanity checks (`sanity_checks.csv`)
All 6 checks pass:
1. Optimal temperature (18–21°C) → higher sleep efficiency ✓
2. Light events negatively correlated with efficiency ✓
3. Mean awakenings in published range [1.5, 4.0] ✓
4. Noise events positively correlated with awakenings ✓
5. ≥50% of sessions in healthy efficiency range [0.75, 0.95] ✓
6. Good quality class has higher efficiency than poor class ✓

---

## Seed Documentation

We use a hierarchical seeding strategy:
- **Global seed**: 42 (set via `seed_everything(42)` in `demo.py`)
- **Per-session seed**: `global_seed + session_id` for each of the 2,500 sessions
- **Per-signal seed**: `session_seed + signal_offset` (1000, 2000, 3000 for light, humidity, noise)

This means any individual session can be reproduced independently.
The global seed controls the train/val/test split and all ML operations.

To reproduce with a different seed:
```bash
python demo.py --seed 123
```
Note: different seeds will produce numerically different results but
the same qualitative conclusions should hold.

---

## Quick Verification (< 10 seconds)

```bash
python demo.py --quick --skip-ablation
```

Expected output: `PIPELINE COMPLETE` with 500 sessions and 10 figures saved.
