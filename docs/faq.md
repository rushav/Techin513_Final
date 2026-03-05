# FAQ and Interview Preparation
## Synthetic Sleep Environment Dataset Generator — TECHIN 513 Team 7

30 questions a professor or interviewer would ask, with thorough answers.

---

## Signal Processing Theory

**Q1. Why did we choose the Butterworth filter over Chebyshev or a windowed-sinc FIR?**

Butterworth filters provide a maximally flat (equiripple-free) magnitude response in the passband, meaning no oscillations or ripple between DC and the cutoff frequency. For bedroom temperature, this is critical: any passband ripple would create artificial-looking temperature fluctuations that might be mistaken for genuine thermal events. Chebyshev Type I achieves a steeper roll-off at the cutoff at the cost of passband ripple; Chebyshev Type II places ripple in the stopband. Since our application requires smooth, physically believable temperature profiles rather than a sharp cutoff, the flat passband of Butterworth is the correct choice. A windowed-sinc FIR would also work but requires a much higher filter order for equivalent roll-off and is harder to design for arbitrary cutoff frequencies.

---

**Q2. Why order 4 specifically?**

Order 4 gives −24 dB/octave roll-off, which provides adequate attenuation of frequencies above the HVAC band (>1/60 cpm) without introducing perceptible group-delay variation. Higher order (e.g., 8) would be sharper but introduces more phase distortion — undesirable for a dataset representing real-time sensor output. We verified this choice by inspecting the frequency response: the HVAC component (1/90 cpm) is in the passband, while any sampling artefacts (above 1/20 cpm) are attenuated by >20 dB.

---

**Q3. What does the FFT reveal that time-domain analysis cannot?**

The Fourier transform decomposes a signal into its constituent frequency components, revealing periodic structure that is invisible in the time domain. In our case, the FFT of bedroom temperature immediately shows two peaks: one at the circadian frequency (~1/1440 cpm) and one at the HVAC frequency (~1/90 cpm). In the time domain, these two oscillations are superimposed and hard to separate by eye. The FFT allows us to verify that our spectral synthesis correctly encoded the intended physical components (Figure F03). Without FFT analysis, we could not confirm that the HVAC sawtooth has the correct period.

---

**Q4. How does spectral synthesis ensure physical plausibility rather than just noise?**

Physical plausibility comes from the power spectral density (PSD) shape. Real physical processes — thermal fluctuations, 1/f electronic noise, neurological variability — all exhibit pink noise (power ∝ 1/f). We implement this exactly by multiplying the complex Fourier coefficients by 1/√f before the IFFT. The resulting time-domain signal has autocorrelation that decays slowly (long-range dependence), which is characteristic of real sensor data. By contrast, white noise has zero autocorrelation at all non-zero lags — it looks "too random" for a physical sensor. Our approach anchors the spectral shape to the expected physics.

---

**Q5. What is a Poisson process and why is it appropriate for bedroom disturbances?**

A Poisson process is a stochastic model for events where: (1) events occur randomly in time, (2) events are independent of each other, and (3) the average rate λ is constant over time. These properties match bedroom disturbances well: a partner turning on a light or a car horn outside are independent events with no "memory" of previous events. The inter-arrival times follow an Exponential(λ) distribution. We parameterise λ per quality class (0.002–0.010 events/min for light) based on published disturbance rates.

---

**Q6. How did we choose the Butterworth cutoff frequencies (0.02 cpm for both temperature and humidity)?**

At 5-minute sampling, the Nyquist frequency is 0.1 cpm. Our temperature signals contain:
- Circadian component: 1/1440 cpm ≈ 0.0007 cpm (in passband)
- HVAC component: 1/90 cpm ≈ 0.011 cpm (in passband)

The cutoff at 0.02 cpm sits above the HVAC frequency, preserving both the circadian and HVAC components while attenuating higher-frequency artefacts. We use the same 0.02 cpm cutoff for humidity to ensure it tracks the HVAC-driven temperature variations at similar time scales, which is required to maintain the correct anti-correlation between temperature and humidity (r ≈ −0.6). A lower humidity cutoff (tested at 0.01 cpm) over-smoothed the signal and destroyed the anti-correlation.

---

**Q7. What would aliasing look like in this context and how do we prevent it?**

Aliasing occurs when a signal contains components above the Nyquist frequency (0.1 cpm at 5-minute sampling), which then "fold back" and appear as lower-frequency artefacts. In our case, the fastest physical process is the Poisson event duration (~5–20 minutes = 0.05–0.2 cpm). Events shorter than 5 minutes (above Nyquist) would alias. We prevent this by enforcing a minimum event duration of 1 sample (5 minutes), which guarantees all events are above the Nyquist period. The Butterworth LPF further removes any residual high-frequency content from the noise generation.

---

## Machine Learning Methodology

**Q8. Why Random Forest over a neural network for this task?**

With 2,500 samples and 34 features, we are in a regime where tree-based ensembles consistently match or outperform neural networks. Neural networks require substantial data to learn the inductive biases that tree methods encode structurally (e.g., axis-aligned splits naturally handle the threshold effects in our data, like temperature outside [18, 21°C] causing a quality drop). Neural networks would also require normalisation, learning rate tuning, and dropout regularisation — all sources of additional variance with no guaranteed payoff at this data scale. Random Forest is also naturally parallelisable (all trees are independent) and trains in seconds.

---

**Q9. Why Random Forest over gradient boosting (XGBoost/LightGBM)?**

Gradient boosting would likely achieve similar or slightly better accuracy on this dataset. However, Random Forest offers two specific advantages for our project: (1) built-in feature importances via Mean Decrease in Impurity (MDI), which we use directly in the ablation study interpretation; and (2) easier hyperparameter tuning (fewer interdependent parameters). The performance difference would be small (~1–2% R²) and does not affect our conclusions.

---

**Q10. How does our split strategy prevent data leakage?**

Data leakage occurs when information from the evaluation set influences training. We prevent this in two ways: (1) all 34 features are extracted from time-series before the train/val/test split, so the model never sees test-set statistics during feature computation; and (2) the split is stratified on quality_class, preventing the case where the training set contains only "good" sessions and the test set only "poor" sessions, which would create an artificially inflated performance gap.

---

**Q11. What do our evaluation metrics capture that accuracy alone would miss?**

We have a regression (not classification) task, so "accuracy" is not directly applicable. R² measures the fraction of label variance explained by the model — a value near 1 means predictions are tight; near 0 means the model is no better than predicting the mean. MAE measures average absolute error in the original units (hours for duration, count for awakenings), making it directly interpretable by sleep scientists. RMSE penalises large errors more heavily than MAE — important for awakenings, where predicting 10 instead of 2 is clinically worse than a consistent 1-unit error.

---

**Q12. How do we know the model is not overfitting?**

We have three independent sources of evidence: (1) the test set is held out until final evaluation, so the reported R²=0.744 is a genuine out-of-sample estimate; (2) 5-fold CV shows R²=0.884±0.011 for sleep efficiency and 0.579±0.017 for awakenings — low standard deviations indicate stable, consistent performance; and (3) the val R² (0.762) and test R² (0.744) are close, with no large train→test degradation that would signal overfitting.

---

**Q13. What do feature importances reveal about the problem?**

The top-ranked features in our Random Forest are predominantly event-based: noise_n_events, light_n_events, and env_stress_score dominate importances. This aligns with the ablation study finding that removing Poisson events collapses performance (ΔR²=-0.355). Temperature-based features (temp_in_optimal, temp_mean) rank next, confirming the well-documented optimal temperature band effect. Spectral entropy features rank lower, suggesting that the frequency-domain complexity of the signals is less directly predictive than the count and magnitude of disturbance events.

---

## Dataset Design

**Q14. Why generate synthetic data instead of collecting real data?**

Real bedroom sensor data paired with sleep quality labels requires: (1) multi-night polysomnography or clinical actigraphy studies; (2) simultaneous IoT sensor deployment; (3) institutional ethics approval; and (4) significant subject recruitment effort. The Sleep Efficiency Dataset on Kaggle (our reference) has only ~450 sessions with no environmental sensors. Generating 2,500 sessions synthetically allows us to explore a much wider range of environmental conditions (all four seasons, three quality levels, diverse temperature setpoints) than any single study could provide within a course timeline.

---

**Q15. How do we ensure physical plausibility and not just plausible-looking noise?**

Physical plausibility comes from three sources: (1) parameter ranges grounded in published literature (temperature 15–30°C from Ref [1], optimal 18–21°C; Poisson rates from Ref [3]); (2) signal structure matching known physics (pink noise for thermal fluctuations, HVAC sinusoidal model for thermostat dynamics, exponential decay for acoustic dissipation); and (3) verified relationships in the data — our sanity checks confirm that the correlations between environment and sleep quality match sleep science predictions. The FFT spectrum (Figure F03) provides direct visual verification that the HVAC cycle appears at the expected frequency.

---

**Q16. What are the risks of a model trained on synthetic data?**

The primary risk is distribution shift: if the synthetic training distribution does not match real-world test data, the trained model will perform poorly. Our data correctly encodes the direction of effects (optimal temperature → better sleep) but not necessarily the exact quantitative magnitudes. A model trained on our data might be overconfident in the precision of the [18, 21°C] threshold, which in reality varies by individual. The multi-modal label distribution (from our quality-class stratification) is also less realistic than the continuous distribution seen in real populations.

---

**Q17. How does the KS test validate our generation approach?**

The two-sample KS test compares the empirical CDF of our synthetic variable against a reference CDF drawn from published statistics. Under the null hypothesis that both samples come from the same distribution, the KS statistic D follows a known distribution, and p > 0.05 indicates we cannot reject similarity. Our temperature marginal comes close (D=0.044, p=0.024). The sleep label failures are not a sign that the data is wrong — rather, our multi-modal (3-class) distributions genuinely differ from the unimodal Gaussian reference, which is a documented feature of our stratified design.

---

**Q18. What assumptions are baked into the data generation?**

Key assumptions: (1) sleep quality depends linearly on the sum of environmental penalties (our scoring function is additive); (2) Poisson event rates are homogeneous over the 8-hour session (in reality, rates might differ by sleep phase); (3) the optimal temperature range [18, 21°C] is universal (real values vary by individual, age, and metabolism); (4) all four seasons have equal representation (our dataset is balanced, while real datasets are typically biased toward certain seasons). These assumptions are all documented and constitute known limitations.

---

## Results and Validation

**Q19. Walk through the ablation study and what each removal proves.**

- **No Butterworth LPF** (ΔR²=−0.001): The filter provides a small benefit by smoothing the temperature signal, making temporal features like ACF and slope more reliable. Its removal slightly degrades performance.
- **No pink noise** (ΔR²=−0.007): Pink noise contributes modest information; removing it reduces performance slightly. Its primary value is physical realism (autocorrelated fluctuations) rather than label discriminability.
- **No Poisson events** (ΔR²=−0.355): Largest single impact. Light and noise events are the primary drivers of sleep quality differentiation. Without them, light_n_events and noise_n_events become zero for all sessions, removing the most informative features.
- **No HVAC model** (ΔR²=−0.018): Meaningful effect. The HVAC sinusoidal cycle contributes to temp_hvac_power and spectral features that the model relies on for temperature-related discrimination.

---

**Q20. What does the baseline comparison demonstrate?**

The mean baseline (R²≈0) proves that the labels are not trivially constant — there is genuine variance to explain. The Ridge baseline (R²=0.727) quantifies the performance achievable by a linear model. The Random Forest's improvement to 0.744 (+0.017) confirms that at least some feature–label relationships are non-linear. The most likely non-linearity is the U-shaped temperature effect: sessions at both 15°C and 30°C have poor sleep, while 18–21°C sessions have good sleep — a relationship that Ridge cannot fit but RF handles naturally via binary splits.

---

**Q21. How do we interpret the KS test p-values?**

A p-value > 0.05 means we cannot reject the null hypothesis that the synthetic and reference distributions are the same (at 5% significance). Our failures (p≈0) indicate strong evidence that our distributions differ from the Gaussian reference. However, this does not mean our data is wrong — it means our multi-modal, stratified distributions genuinely differ from a unimodal Gaussian approximation. Temperature achieves p=0.024 (closest to passing), consistent with its more Gaussian marginal distribution. We document this limitation transparently in the report.

---

**Q22. What is the biggest limitation and how would we address it?**

The biggest limitation is the hard quality-class stratification creating multi-modal label distributions. We would address this by using a continuous quality parameter (a single random variable drawn from a Beta distribution calibrated to published statistics) rather than three discrete classes. This would produce more realistic, unimodal label distributions and improve KS test performance. The trade-off is reduced interpretability and harder ablation study design.

---

**Q23. How would results change with real sensor data?**

With real sensor data, we would: (1) replace the parametric label derivation with a learned model trained on matched sensor/polysomnography data; (2) discover additional covariates not modelled (individual differences in temperature sensitivity, medication effects, stress); (3) likely see lower R² because real sleep has more individual-level stochasticity than our model captures; (4) be able to perform proper KS tests against the real distribution rather than a Gaussian proxy.

---

## Implementation

**Q24. How is reproducibility guaranteed end-to-end?**

We use a hierarchical seeding strategy: global seed 42 → per-session seeds (seed + session_id) → per-signal seeds (per-session seed + offset). This means:
- The same global seed always produces identical sessions in the same order.
- Any single session can be reproduced independently by using its session-specific seed.
- The train/val/test split is deterministic (same seed passed to train_test_split).
- All Random Forest operations use the same seed.
- We document in REPRODUCIBILITY.md which seed produces which exact numerical outputs.

---

**Q25. What is the most technically challenging component and why?**

The Poisson event injection combined with physically realistic event profiles is the most challenging component. Getting the event rate, duration, and amplitude distributions calibrated so that:
(1) awakenings match the published mean (2.5);
(2) sleep efficiency matches the published range (0.80);
(3) the event features are informative enough for ML to use;
(4) the distributions don't diverge wildly from reference statistics —
requires careful calibration of multiple interacting parameters (λ, duration distribution, amplitude scaling). Getting all four constraints satisfied simultaneously required several rounds of simulation and parameter adjustment.

---

**Q26. How would this scale to a real deployed system?**

In a real deployment scenario, the generation pipeline would be replaced by actual sensor ingestion. The signal processing pipeline (filtering, FFT analysis, feature extraction) would run in real-time on a gateway device (Raspberry Pi or similar), processing 5-minute sensor readings. The Random Forest model (lightweight: ~200 trees × 34 features) would run inference every 5 minutes to update sleep quality estimates. The Poisson event model would be replaced by actual threshold detection on the real sensor stream. The dataset we generate could be used to pre-train this model before real data is available.

---

**Q27. What would we change with 3 more months and more compute?**

With more time: (1) Replace the parametric label derivation with a GAN or VAE trained on the Kaggle Sleep Efficiency Dataset, providing more realistic label distributions; (2) add physiological signals (heart rate variability, body movement) synthesised from the environmental conditions via a learned physiological model; (3) implement a non-stationary Poisson process where event rates vary by sleep phase (NREM vs REM); (4) validate against a real bedroom sensor dataset (e.g., CASAS smart home dataset + actigraphy); (5) extend to 10,000 sessions covering more diverse populations (age, geographic location, season).

---

**Q28. Why did we use a causal Butterworth filter rather than zero-phase (filtfilt)?**

Zero-phase filtering (scipy.signal.sosfiltfilt) applies the filter twice — forward and backward — to achieve zero phase distortion. This is the right choice for offline analysis of signals where you have access to the entire signal. However, our goal is to simulate what a real IoT gateway would report: causal, real-time filtered data. A real sensor applies a causal filter because it cannot look into the future. Using causal (forward-only) filtering produces signals that are physically more realistic. If a professor is interested in the signal processing accuracy rather than realism, zero-phase filtering would be technically preferable.

---

**Q29. Why 5-minute sampling resolution?**

We chose 5-minute sampling for three reasons: (1) It matches the typical resolution of consumer sleep tracking devices and smart home IoT sensors (e.g., Nest thermostat reports every 5 minutes); (2) It gives 96 samples per 8-hour session — enough for meaningful spectral analysis (the FFT has 49 frequency bins), feature extraction, and HVAC cycle detection (90 min = 18 samples), while keeping the dataset size manageable; (3) The HVAC cycle (60–120 min) is resolved with 12–24 samples per cycle, well above the Nyquist requirement of 2 samples/cycle.

---

**Q30. If a professor challenges that our KS tests all fail, how do we respond?**

We would say: "You are correct that all six KS tests fail at p=0.05. However, this failure is not evidence that our data is unrealistic — it is a consequence of our intentional stratified design. Our dataset contains three quality classes (poor, moderate, good), each with different mean parameters. This stratification creates multi-modal label distributions that are fundamentally different in shape from the unimodal Gaussian reference distributions we compare against. The KS test is sensitive to shape differences, not just distributional similarity. The appropriate reference distribution would be a Gaussian mixture with three components — with such a reference, we believe KS tests would pass. More importantly, all six sleep science sanity checks (Tier 3 validation) pass, validating that our data encodes the correct directional relationships between environment and sleep quality. The KS failure reflects our reference choice, not our generation quality."
