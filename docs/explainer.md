# Project Explainer
## Synthetic Sleep Environment Dataset Generator

**Team 7 — Rushav Dash, Lisa Li | TECHIN 513 Final Project**

This document explains what we built, why every decision was made, and
how all components connect.  Written for a technically literate reader
who may not specialise in sleep science or machine learning.

---

## The Core Problem

Sleep quality is measurable (through wrist-worn actigraphy, clinical sleep
studies, or questionnaires) and so are bedroom environmental conditions
(temperature sensors, light sensors, noise meters).  But these two streams
of data have never been collected together in a public dataset.

The consequence: sleep researchers who want to train a model that predicts
"how well will someone sleep given their bedroom conditions tonight?" have
no training data.  They would need to recruit subjects, install sensors,
run multi-night studies, and obtain ethics approval.  This is expensive,
slow, and hard to scale.

Our solution: generate the data synthetically.

---

## What We Built

A Python pipeline that produces 2,500 synthetic 8-hour sleep sessions.
Each session contains:

1. **Four environmental time-series** at 5-minute resolution:
   - Temperature (°C) — the dominant sleep quality driver
   - Illuminance (lux) — light disturbances
   - Relative humidity (%) — comfort factor
   - Ambient noise (dB SPL) — the second major disruptor

2. **Four sleep quality labels**:
   - Sleep efficiency (0–1): fraction of the 8-hour window actually asleep
   - Sleep duration (hours): total sleep time
   - Awakenings (count): how many times the sleeper woke up
   - Sleep score (0–100): composite quality rating

The result is a CSV with 2,500 rows and 38 columns, directly usable as
training data for downstream sleep optimisation models.

---

## Why Each Technical Decision Was Made

### Decision 1: Butterworth filter for temperature

Temperature from a real bedroom sensor is not perfectly smooth — it
contains the thermostat cycling (on/off every 60–120 minutes), the slow
daily circadian drift, AND sensor noise.  When we add these components
together in simulation, we get a noisy composite that would be physically
implausible as a single sensor reading.

We apply a 4th-order Butterworth low-pass filter with a cutoff at 0.02
cycles/minute.  This preserves the circadian drift and HVAC cycling (both
slower than the cutoff) while removing the high-frequency sensor noise.
We chose Butterworth specifically because it has a maximally flat passband
— no ripple or ringing that might mimic real thermal events.

*Alternative considered: Zero-phase filtfilt.*
We chose causal (forward-only) filtering because it better models what
a real IoT gateway would report.  A real sensor cannot look into the future.

### Decision 2: Pink noise for thermal fluctuations

Real temperature measurements don't fluctuate with white noise (where each
sample is independent of the previous).  Real sensors show "autocorrelated"
noise — if the temperature is slightly above the setpoint at 2:00 AM, it's
likely still above at 2:05 AM because temperature changes slowly.

Pink noise (power spectral density ∝ 1/f) captures this autocorrelation.
We generate it by manipulating the FFT coefficients directly — this is
mathematically exact and computationally efficient (O(N log N) via the FFT).

### Decision 3: Poisson process for disturbance events

When does a bedroom light turn on at 3 AM?  It might be a partner visiting
the bathroom, a notification light, or a car headlight.  These events happen
at random times with no "memory" of previous events.  The correct
mathematical model is a Poisson process: events arrive with a constant
average rate (λ), and inter-arrival times are exponentially distributed.

This is not just a stylistic choice — it's the same model used in queueing
theory, radioactive decay, and call center traffic modelling.  Our
implementation generates arrival times analytically from the exponential
distribution, giving exactly Poisson-distributed event counts.

*Why this matters for ML*: The ablation study shows that removing Poisson
events collapses model R² from 0.795 to 0.224 — a 57% drop.  Without
discrete disturbance events, the model cannot differentiate sleep quality.

### Decision 4: HVAC sawtooth wave

Real thermostats overshoot their setpoint and then drift back — a
"relaxation oscillator" dynamics.  When the heater turns on, temperature
rises linearly; when it turns off (setpoint reached), it cools slowly.
This is well-approximated by a sawtooth wave with period 60–120 minutes.

We include this because it creates the spectral peak at ~1/90 cycles/minute
that is visible in the FFT of real bedroom temperature data.  Our FFT
analysis (Figure F03) confirms this peak appears correctly.

### Decision 5: Random Forest for regression

Our task is multi-output regression: given 34 environmental features,
predict 4 continuous sleep quality values.  Random Forest is the right
choice because:
- Non-linear relationships exist (temperature has a U-shaped effect:
  too cold and too hot are both bad; only 18–21°C is optimal)
- We have 2,500 samples — enough for RF, not enough to justify deep learning
- We need feature importances for the ablation study interpretation
- RF trains in ~2 seconds; neural networks would require much more tuning

### Decision 6: Parametric label derivation

We derive sleep quality labels from environmental conditions using
domain knowledge (published sleep science thresholds) rather than a
learned model, because we don't have a paired sensor/polysomnography
dataset.  This is a deliberate design choice, not a limitation:
- Every parameter is documented with a literature reference
- The resulting labels satisfy all 6 sleep science sanity checks
- The derivation is transparent and auditable

The trade-off is that our labels are "theoretically correct" rather than
empirically learned.  A future version would train a label-assignment model
on real paired data and use our synthesised signals as inputs.

---

## How All Components Connect

```
Session profile (seed, season, quality class)
    │
    ├─► Temperature signal:
    │   • Circadian sinusoid
    │   • + HVAC sawtooth        ─► composite ─► Butterworth LPF (0.02 cpm) ─► T[t]
    │   • + pink noise (1/f)
    │
    ├─► Light signal:
    │   • Near-dark baseline
    │   • + Poisson arrivals (λ depends on quality class)              ─► L[t]
    │   • + random amplitudes, durations
    │
    ├─► Humidity signal:
    │   • Anti-correlated with T: H = H₀ - β(T - T̄) + pink noise
    │   • Butterworth LPF (0.01 cpm)                                   ─► H[t]
    │
    └─► Noise signal:
        • Quiet baseline + Poisson arrivals with exponential decay      ─► N[t]

[T[t], L[t], H[t], N[t]] × 2,500 sessions
    │
    ▼ Feature extraction (34 scalar features per session)
    │   • Statistical: mean, std, skew, kurtosis, range
    │   • Temporal: ACF at lag-1, slope
    │   • Spectral: FFT dominant frequency, Welch PSD, band power
    │   • Event-based: Poisson event counts, threshold fractions
    │   • Cross-signal: temperature–humidity Pearson r
    │
    ▼ Label derivation (domain knowledge + calibration)
    │   • Sleep efficiency = base(quality class) - Σ penalties
    │   • Awakenings = base + event contributions
    │   • Duration = efficiency × 8h + Gaussian residual
    │   • Sleep score = weighted combination of above
    │
    ▼ Machine learning pipeline
    │   • 70/15/15 stratified split
    │   • Grid search: Random Forest (n_est, max_features, min_split)
    │   • Comparison: Mean baseline, Ridge Regression
    │   • 5-fold CV for stability assessment
    │   • Ablation: disable each SP component, measure ΔR²
    │
    ▼ Three-tier validation
        • KS-tests vs. published statistics
        • Discriminability (AUC confirms physical structure)
        • 6 sleep science sanity checks (all pass)
```

---

## Results Explained

### R² = 0.795 — what does this mean?

R² = 0.795 means our Random Forest explains 79.5% of the variance in the
four sleep quality targets.  The remaining 20.5% is "residual" — variance
in the labels that cannot be predicted from the environmental features alone.
This residual comes from the Gaussian noise we intentionally added to labels
to prevent a perfectly deterministic (unrealistic) dataset.

### Why Ridge R² = 0.751 but RF = 0.795?

The 4.4-point gap confirms that at least some feature–label relationships
are genuinely non-linear.  The most likely non-linearity is the temperature
"sweet spot": sessions at both 14°C and 28°C get penalised for sleep quality,
while sessions at 18–21°C do not.  This is a U-shape that Ridge (a linear model)
cannot fit but Random Forest handles naturally via its binary splitting structure.

### Ablation ΔR² = −0.571 for no-Poisson — why so dramatic?

Removing Poisson events means light_n_events and noise_n_events become zero
for every session.  These two features are among the highest-ranked by feature
importance.  Without them, the model loses its primary mechanism for
distinguishing poor sessions (many disturbances) from good sessions (few).
The remaining features (temperature statistics, humidity) have much weaker
relationships with the labels in our formulation.

### Why do KS tests fail?

Our dataset is stratified into three quality classes, each with a different
mean efficiency, awakening count, etc.  The resulting distribution is
multi-modal (three peaks).  The reference distributions are unimodal
Gaussians.  The KS test is sensitive to shape differences, so multi-modal
vs. unimodal always fails.  The correct response is to use a Gaussian
mixture reference — but this requires knowing the mixing proportions of
real bedroom quality, which we don't have.  We document this as a
known limitation.

### Why do all 6 sanity checks pass?

Because we designed the label derivation to encode the correct directional
relationships.  If the data passed KS tests but failed sanity checks, that
would be far more concerning — it would mean we had the right marginal
distributions but the wrong causal structure.  Passing 6/6 sanity checks
confirms that the fundamental sleep science relationships are correctly
represented.

---

## Honest Limitations

1. **Multi-modal labels**: Hard quality-class stratification creates
   non-Gaussian label distributions.  Softer stratification would help.

2. **Parametric label derivation**: Labels are derived from a formula,
   not from real physiological data.  The formula is well-justified but
   not empirically validated.

3. **Homogeneous Poisson process**: We assume a constant disturbance rate
   throughout the night.  In reality, disturbances are more common during
   light sleep (NREM stage 1/2) than deep sleep.

4. **Individual differences not modelled**: The optimal temperature of
   18–21°C is a population average; individuals vary by ±2–3°C.

5. **Discriminability AUC = 1.0**: Our structured synthetic data is
   completely distinguishable from unstructured Gaussian noise (AUC=1).
   This validates physical structure but doesn't test discriminability
   from real bedroom data (which we don't have access to).

---

## What We Would Do With More Time

- Learn the label assignment model from real paired sensor/sleep data
- Add physiological signals (estimated from environmental conditions)
- Non-stationary Poisson rates (varying by sleep phase)
- Validate KS tests against a Gaussian mixture reference
- Extend to 10,000 sessions with more diverse populations
- Release as a Kaggle dataset for the broader research community
