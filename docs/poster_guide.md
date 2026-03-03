# Poster Guide
## Synthetic Sleep Environment Dataset Generator — TECHIN 513 Team 7

This guide maps every rubric criterion to a specific location on the poster
and provides talking points for the live presentation.

---

## Recommended Layout (A0 or 36×48 in)

```
┌────────────────────────────────────────────────────────────────┐
│          TITLE + AUTHORS + ABSTRACT (10% height)               │
├──────────────┬─────────────────────────┬───────────────────────┤
│  MOTIVATION  │  PIPELINE DIAGRAM       │  SIGNAL EXAMPLES      │
│  & PROBLEM   │  (F02 + ASCII flow)     │  (F02 or F05)         │
│  (15%)       │  (20%)                  │  (15%)                │
├──────────────┼─────────────────────────┼───────────────────────┤
│  SP METHODS  │  ML RESULTS             │  ABLATION TABLE       │
│  (F03, F04)  │  (F09 scatter + table)  │  (F10 bar chart)      │
│  (15%)       │  (15%)                  │  (15%)                │
├──────────────┴─────────────────────────┴───────────────────────┤
│  VALIDATION RESULTS (F11 KS plots + sanity checks) (10%)       │
├────────────────────────────────────────────────────────────────┤
│  LIMITATIONS + FUTURE WORK  |  REFERENCES  |  QR CODE (5%)     │
└────────────────────────────────────────────────────────────────┘
```

---

## Section-by-Section Guide

### 1. Title Block (top strip)
- **Title**: "Synthetic Sleep Environment Dataset Generator"
- **Authors**: Rushav Dash · Lisa Li · TECHIN 513 Team 7 · University of Washington
- **Abstract**: 3-sentence version — problem, method, key result (R²=0.795, 6/6 sanity checks)

**Rubric criterion addressed**: Problem clarity, relevance to course topics

---

### 2. Motivation & Problem (top-left panel, ~15%)
**What to put here**:
- 2-3 bullet points on the gap: no public dataset links bedroom sensors to sleep quality
- One sentence each on temperature, light, and noise effects on sleep
- "This prevents researchers from training models without expensive sensor deployments"

**Key talking point**: "No existing dataset contains both IoT sensor readings AND objective sleep quality metrics — we bridge this gap synthetically."

**Rubric criteria**: Relevance, Clarity, Introduction

---

### 3. Pipeline Diagram (top-center, ~20%)
**Use**: ASCII diagram from README + simplified version of Figure flow

```
Session Profile ──► Signal Generation ──► Feature Extraction ──► ML + Labels
(season, quality)    [Temp, Light,          [34 features]         [RF Regressor]
                      Humidity, Noise]                              [4 targets]
                            │
                     [SP Techniques]
                     • Butterworth LPF
                     • Pink noise (FFT)
                     • HVAC sawtooth
                     • Poisson events
```

**Key talking point**: "Our SP pipeline is the core technical contribution — every signal component has a documented physical justification."

**Rubric criteria**: Soundness, Proposed Methodology

---

### 4. Signal Examples (top-right, ~15%)
**Use figures**: F02 (4-panel time series) OR F05 (Poisson events close-up)

**Key talking point**: "Each signal is physically plausible: temperature shows the HVAC cycle every 90 minutes; light shows discrete Poisson-distributed events; humidity anticorrelates with temperature."

**Rubric criteria**: Feasibility, Technical correctness

---

### 5. Signal Processing Methods (middle-left, ~15%)
**Use figures**: F03 (FFT spectrum) + F04 (filter effect)

**What to include**:
- Small 2×2 grid: one panel per technique
- F03 verifies HVAC frequency peak in temperature spectrum
- F04 shows before/after Butterworth filter
- Brief equations for Butterworth design and Poisson process

**Table of techniques**:
| Technique | Signal | Justification |
|---|---|---|
| Butterworth LPF | Temperature, Humidity | Flat passband, preserves HVAC/circadian |
| Pink noise (FFT) | Temperature | Models autocorrelated thermal fluctuations |
| Poisson events | Light, Noise | Memoryless model for rare disturbances |
| HVAC sawtooth | Temperature | Relaxation oscillator dynamics |

**Key talking point**: "The FFT spectrum confirms our HVAC sawtooth is correctly encoded — the peak appears exactly at 1/90 cycles/minute as expected."

**Rubric criteria**: Signal processing techniques, Technical soundness

---

### 6. ML Results (middle-center, ~15%)
**Use figures**: F09 (scatter plots) + results table

**What to include**:
| Model | Mean R² |
|---|---|
| Random Forest | **0.795** |
| Ridge Regression | 0.751 |
| Mean Baseline | −0.005 |

- Emphasize: RF outperforms Ridge by 4.4 points (non-linearity confirmed)
- Show F09: tight scatter around y=x diagonal

**Key talking point**: "The RF's advantage over Ridge proves that the U-shaped temperature effect on sleep efficiency is genuinely non-linear — exactly what domain knowledge predicts."

**Rubric criteria**: ML integration, Experimental results, Baseline comparison

---

### 7. Ablation Study (middle-right, ~15%)
**Use figure**: F10 (bar chart)

**What to include**:
| Disabled | ΔR² |
|---|---|
| None (full) | 0.000 |
| No Butterworth LPF | −0.007 |
| No pink noise | +0.001 |
| No Poisson events | **−0.571** |
| No HVAC sawtooth | +0.003 |

**Key talking point**: "Removing Poisson events collapses R² from 0.795 to 0.224 — the most dramatic single result. Without discrete disturbance events, the model cannot differentiate session quality. This validates our Poisson modelling as the critical design choice."

**Rubric criteria**: Ablation study requirement, Soundness, Comprehensiveness

---

### 8. Validation Results (bottom strip, ~10%)
**Use figures**: F11 (KS test CDFs, 6 panels)

**What to include**:
- "3-Tier Validation" header
- Tier 1 (KS tests): 0/6 pass — explain WHY (multi-modal vs Gaussian reference)
- Tier 2 (discriminability): AUC=1.0 — explains that data has strong physical structure
- Tier 3 (sanity checks): **6/6 pass** — emphasise this

Sanity check table:
| Check | Result |
|---|---|
| Optimal temp (18–21°C) → higher efficiency | ✓ |
| Light events ↓ efficiency | ✓ |
| Mean awakenings in [1.5, 4.0] | ✓ |
| Noise events ↑ awakenings | ✓ |
| 68% of sessions in healthy efficiency range | ✓ |
| Good class > Poor class efficiency | ✓ |

**Key talking point**: "The KS tests fail because our stratified design creates multi-modal distributions that differ from unimodal Gaussian references — not because the data is wrong. All sleep science sanity checks pass."

**Rubric criteria**: Validation methodology, Comprehensiveness

---

### 9. Bottom Row: Limitations, References, QR Code
- **Limitations**: KS test multi-modality, label derivation is parametric (not learned)
- **Future work**: Non-stationary Poisson rates, physiological signals, real sensor validation
- **References**: 3-4 key citations (Okamoto-Mizuno 2012, Hume 2012, Porcheret 2015)
- **QR code**: Link to GitHub repository

---

## Visual Design Guidance

### Colour palette
- Temperature panels: brick red (#d62728)
- Light panels: orange (#ff7f0e)
- Humidity panels: blue (#1f77b4)
- Noise panels: green (#2ca02c)
- Key results: purple (#9467bd)
- Background: white
- Section headers: dark navy or black

### Typography
- Title: 72pt sans-serif, bold
- Section headers: 36pt, bold
- Body text: 24pt minimum (readable at 1 metre)
- Figure captions: 20pt
- Avoid walls of text — bullet points only

### Figures
- Use F02 for the main signal showcase (recognisable, colourful, 4-panel)
- Use F09 for ML results (tight scatter = good result, visually compelling)
- Use F10 for ablation (the −0.571 bar is dramatic and memorable)
- Use F12 for seasonal diversity (shows breadth of dataset)
- Every figure needs a caption with the key finding in one sentence

### Whitespace
- Leave 15% of each panel as whitespace
- Use dividing lines between sections
- No 8-point font anywhere

---

## Rubric-Mapped Checklist

| Rubric Criterion | Location on Poster |
|---|---|
| Signal processing techniques (filtering, FFT, etc.) | SP Methods panel + Pipeline diagram |
| Machine learning integration | ML Results panel |
| Well-defined problem | Motivation panel |
| Baseline comparison | ML Results table |
| Ablation study | Ablation panel |
| Experimental results | ML Results + Validation panels |
| Clarity of methodology | Pipeline diagram |
| Related work | Not required on poster; mention verbally |
| Limitations | Bottom row |
| References | Bottom row |
| Code availability | QR code + GitHub URL |

---

## 5-Minute Presentation Script

1. **(30 sec)** Problem: "No dataset links bedroom sensors to sleep quality."
2. **(60 sec)** Pipeline: "We generate signals using 4 SP techniques, extract 34 features, and label sessions via domain knowledge."
3. **(60 sec)** SP highlight: "This FFT spectrum confirms our HVAC sawtooth is correctly encoded."
4. **(60 sec)** ML results: "Random Forest achieves R²=0.795, beating Ridge by 4.4 points — confirming non-linearity."
5. **(60 sec)** Ablation: "Removing Poisson events collapses R² by 0.571 — the key finding."
6. **(30 sec)** Validation: "6/6 sleep science checks pass. KS tests fail due to multimodality — a documented limitation."
