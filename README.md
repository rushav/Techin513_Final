# Synthetic Sleep Environment Dataset Generator

**TECHIN 513 Final Project — Team 7**
Rushav Dash · Lisa Li
University of Washington, 2026

---

## Overview

We generate a realistic synthetic dataset of **2,500 sleep sessions** that pair
bedroom environmental time-series (temperature, illuminance, relative humidity,
ambient noise) with validated sleep quality labels.  The dataset enables sleep
researchers to train predictive models without deploying expensive sensor
infrastructure.

No public dataset links bedroom environmental conditions to sleep quality metrics.
Existing sleep datasets contain physiological signals without environmental data;
IoT sensor datasets contain readings without sleep labels.  Our pipeline bridges
this gap using rigorous signal processing and machine learning.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     GENERATION LAYER                            │
│                                                                 │
│  Session Profile   ──►  Temperature  ──►  Butterworth LPF      │
│  (season, quality)       (spectral        (order 4, 0.02 cpm)  │
│                          synthesis +                            │
│                          HVAC sinusoidal +                      │
│  Poisson events   ──►   pink 1/f noise)                        │
│  (light, noise)                                                 │
│                    ──►  Humidity     ──►  LPF (0.02 cpm)       │
│                          (anti-corr with temperature)           │
└────────────────────────────┬────────────────────────────────────┘
                             │ 4 signals × 96 time steps
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  FEATURE EXTRACTION LAYER                       │
│                                                                 │
│  Statistical: mean, std, min, max, range, skewness             │
│  Temporal:    slope, lag-1 ACF, threshold crossings            │
│  Spectral:    dominant frequency, band power, spectral entropy  │
│  Events:      Poisson event counts, above-threshold fractions   │
│  Cross-signal: temperature–humidity correlation                 │
│                                                                 │
│                  34 scalar features per session                 │
└────────────────────────────┬────────────────────────────────────┘
                             │ X: (2500, 34)  y: (2500, 4)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ML PIPELINE                                 │
│                                                                 │
│  70/15/15 stratified split (by quality class)                  │
│  Random Forest (grid search: n_est, max_features, min_split)   │
│  Baseline 1: Mean predictor (DummyRegressor)                   │
│  Baseline 2: Ridge Regression                                  │
│  5-fold cross-validation                                       │
│  Ablation study: disable each SP component individually         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  THREE-TIER VALIDATION                          │
│                                                                 │
│  Tier 1: KS-tests vs. published reference distributions         │
│  Tier 2: Discriminability (structured signal detection)         │
│  Tier 3: Sleep science sanity checks (6/6 pass)                │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
              2,500 session CSV  +  12 figures  +  6 metrics files
```

---

## Key Results (seed=42, N=2,500)

| Metric | Value |
|---|---|
| Sleep efficiency (mean ± std) | 0.805 ± 0.129 |
| Sleep duration (mean ± std) | 6.85 ± 1.13 h |
| Awakenings (mean ± std) | 2.44 ± 1.15 |
| Sleep score (mean ± std) | 82.8 ± 11.0 |
| RF Test R² (mean over 4 targets) | **0.744** |
| Mean baseline R² | −0.001 |
| Ridge Regression R² | 0.727 |
| RF vs. Ridge advantage | +0.017 (confirms non-linearity) |
| Sanity checks passed | **6 / 6** |
| Ablation: no Poisson events | ΔR² = **−0.355** |
| Wall time | ~30 s |

---

## Signals Generated

| Signal | Unit | Range | Key Components |
|---|---|---|---|
| Temperature | °C | 15–30 | Circadian sinusoid + HVAC sinusoidal model + pink noise + Butterworth LPF |
| Illuminance | lux | 0–200 | Dark baseline + Poisson light-intrusion events |
| Relative Humidity | % | 20–80 | Anti-correlated with temperature + pink noise + LPF |
| Ambient Noise | dB SPL | 20–65 | Quiet baseline + Poisson noise events |

---

## Signal Processing Techniques

| Technique | Where Applied | Justification |
|---|---|---|
| Butterworth LPF (order 4) | Temperature, humidity | Maximally flat passband; preserves circadian/HVAC dynamics |
| FFT / spectral synthesis | Pink noise generation; PSD analysis | Efficient 1/f noise; frequency-domain verification |
| Poisson event injection | Light events, noise events | Memoryless model for rare, independent disturbances |
| Autocorrelation analysis | Feature extraction; ablation | Quantifies temporal persistence — key LP filter diagnostic |
| Welch PSD estimation | Spectral features | Reduced-variance spectral estimate for short (96-sample) signals |

---

## Installation

```bash
git clone https://github.com/rushavsd/Techin513_Final.git
cd Techin513_Final
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## How to Run

```bash
# Full pipeline (2,500 sessions, ~30 seconds)
python demo.py

# Quick demo (500 sessions, ~10 seconds)
python demo.py --quick

# Custom seed
python demo.py --seed 123

# Skip ablation study
python demo.py --skip-ablation
```

All outputs land in:
- `data/synthetic_sleep_dataset.csv` — the dataset
- `results/figures/` — 12 publication-quality PNG figures
- `results/metrics/` — 6 JSON/CSV metrics files

---

## Repository Structure

```
Techin513_Final/
├── src/
│   ├── __init__.py
│   ├── utils.py              # Seeding, logging, I/O helpers
│   ├── signal_processing.py  # Butterworth, FFT, pink noise, Poisson
│   ├── data_generation.py    # Signal + label synthesis
│   ├── feature_extraction.py # 34 scalar features from time-series
│   ├── ml_pipeline.py        # RF training, ablation, baselines
│   ├── validation.py         # KS tests, sanity checks
│   └── visualisation.py      # 12 publication figures
├── notebooks/
│   ├── exploration.ipynb
│   └── ablation_study.ipynb
├── data/
│   └── synthetic_sleep_dataset.csv
├── results/
│   ├── figures/              # F01–F12 PNG files
│   └── metrics/              # 6 JSON/CSV metric files
├── docs/
│   ├── report.tex
│   ├── poster_guide.md
│   ├── explainer.md
│   └── faq.md
├── demo.py                   # Single entry point
├── README.md
├── REPRODUCIBILITY.md
└── requirements.txt
```

---

## Team

| Name | Contribution |
|---|---|
| Rushav Dash | Signal processing pipeline, FFT analysis, Butterworth filter design, validation framework |
| Lisa Li | ML pipeline, feature extraction, ablation study, documentation |

---

## Citation

If you use this dataset or code, please cite:

```
Dash, R. & Li, L. (2026). Synthetic Sleep Environment Dataset Generator.
TECHIN 513 Final Project, University of Washington.
GitHub: https://github.com/rushavsd/Techin513_Final
```
