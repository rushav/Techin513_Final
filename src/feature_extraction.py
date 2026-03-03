"""
feature_extraction.py — Extract scalar ML features from session time-series.

We compute 36 clinically and physically motivated scalar features from the
four environmental time-series (temperature, light, humidity, noise) in
each session.  Features span five categories:

  1. **Statistical moments**: mean, std, skewness, kurtosis, range
  2. **Temporal dynamics**: linear slope, lag-1 autocorrelation, zero-crossing rate
  3. **Spectral features**: dominant frequency, spectral entropy, band powers
  4. **Event-based features**: Poisson event counts, above-threshold fractions
  5. **Cross-signal features**: Pearson correlation between signal pairs

All features are dimensionless or have consistent units, making them
suitable direct inputs to Random Forest without standardisation.
(Random Forest is invariant to monotone feature transformations.)

Feature catalogue is documented in FEATURE_CATALOGUE below — this table
maps every feature name to its physical interpretation and unit.

Authors: Rushav Dash, Lisa Li — TECHIN 513 Final Project
"""

from __future__ import annotations

import numpy as np
from scipy.stats import skew, kurtosis
from typing import Dict, List

from .signal_processing import (
    compute_fft_spectrum,
    dominant_frequency,
    spectral_power_in_band,
    autocorrelation,
)
from .utils import SAMPLE_RATE_MIN, get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Feature catalogue (for documentation and report table generation)
# ---------------------------------------------------------------------------

FEATURE_CATALOGUE: List[Dict[str, str]] = [
    # --- Temperature features ---
    {"name": "temp_mean",       "unit": "°C",    "description": "Mean temperature over the session"},
    {"name": "temp_std",        "unit": "°C",    "description": "Std dev of temperature (thermal variability)"},
    {"name": "temp_min",        "unit": "°C",    "description": "Minimum temperature"},
    {"name": "temp_max",        "unit": "°C",    "description": "Maximum temperature"},
    {"name": "temp_range",      "unit": "°C",    "description": "Peak-to-peak temperature swing"},
    {"name": "temp_slope",      "unit": "°C/min","description": "Linear trend slope (net heating/cooling)"},
    {"name": "temp_acf1",       "unit": "—",     "description": "Lag-1 autocorrelation (thermal persistence)"},
    {"name": "temp_dom_freq",   "unit": "cpm",   "description": "Dominant FFT frequency"},
    {"name": "temp_hvac_power", "unit": "°C²/cpm","description": "Spectral power in HVAC band (1/120–1/60 cpm)"},
    {"name": "temp_skewness",   "unit": "—",     "description": "Skewness (asymmetry of temperature distribution)"},
    {"name": "temp_in_optimal", "unit": "—",     "description": "Fraction of session with T ∈ [18, 21] °C"},
    # --- Light features ---
    {"name": "light_mean",      "unit": "lux",   "description": "Mean illuminance (overall darkness quality)"},
    {"name": "light_std",       "unit": "lux",   "description": "Std dev of illuminance (event variability)"},
    {"name": "light_max",       "unit": "lux",   "description": "Peak illuminance event"},
    {"name": "light_n_events",  "unit": "count", "description": "Number of light-exceedance events (>10 lux)"},
    {"name": "light_frac_dark", "unit": "—",     "description": "Fraction of session below 5 lux (dark)"},
    {"name": "light_total_lux_min","unit":"lux·min","description":"Integrated illuminance dose"},
    # --- Humidity features ---
    {"name": "humidity_mean",   "unit": "%",     "description": "Mean relative humidity"},
    {"name": "humidity_std",    "unit": "%",     "description": "Std dev of RH (humidity variability)"},
    {"name": "humidity_min",    "unit": "%",     "description": "Minimum RH"},
    {"name": "humidity_max",    "unit": "%",     "description": "Maximum RH"},
    {"name": "humidity_in_comfort","unit":"—",   "description": "Fraction in ASHRAE comfort range [40–60%]"},
    {"name": "humidity_acf1",   "unit": "—",     "description": "Lag-1 ACF (slow humidity dynamics)"},
    # --- Noise features ---
    {"name": "noise_mean",      "unit": "dB",    "description": "Mean ambient noise level"},
    {"name": "noise_std",       "unit": "dB",    "description": "Std dev of noise (burst variability)"},
    {"name": "noise_max",       "unit": "dB",    "description": "Peak noise event"},
    {"name": "noise_n_events",  "unit": "count", "description": "Number of noise events exceeding 45 dB"},
    {"name": "noise_frac_quiet","unit": "—",     "description": "Fraction of session below 35 dB (quiet)"},
    {"name": "noise_l90",       "unit": "dB",    "description": "L90 noise percentile (background level)"},
    # --- Cross-signal features ---
    {"name": "corr_temp_humidity","unit":"—",    "description": "Pearson r: temperature vs humidity (anti-corr expected)"},
    {"name": "corr_light_noise","unit": "—",     "description": "Pearson r: light vs noise (co-occurrence of disturbances)"},
    {"name": "env_stress_score","unit": "—",     "description": "Composite penalty score from all signals (0=ideal, >0=stressful)"},
    # --- Spectral entropy features ---
    {"name": "temp_spectral_entropy","unit":"—", "description": "Spectral entropy of temperature (signal complexity)"},
    {"name": "noise_spectral_entropy","unit":"—","description": "Spectral entropy of noise signal"},
]

FEATURE_NAMES: List[str] = [f["name"] for f in FEATURE_CATALOGUE]


# ---------------------------------------------------------------------------
# Individual feature extractors
# ---------------------------------------------------------------------------

def _spectral_entropy(signal: np.ndarray) -> float:
    """Compute spectral entropy as a measure of signal regularity.

    Spectral entropy H = -Σ p(f) log₂ p(f) where p(f) is the normalised
    PSD.  A purely sinusoidal signal has low entropy (energy concentrated
    at one frequency); broadband noise has high entropy.  We use this to
    distinguish between well-structured environmental cycles and random
    fluctuations.

    Parameters
    ----------
    signal : np.ndarray
        Input time-series.

    Returns
    -------
    float
        Spectral entropy in bits.
    """
    _, psd = compute_fft_spectrum(signal)
    psd_sum = psd.sum()
    if psd_sum < 1e-12:
        return 0.0
    p = psd / psd_sum
    # Avoid log(0) by masking near-zero bins
    mask = p > 1e-12
    entropy = -np.sum(p[mask] * np.log2(p[mask]))
    return float(entropy)


def _linear_slope(signal: np.ndarray) -> float:
    """Estimate the linear trend of a signal via least-squares regression.

    A positive slope for temperature suggests the room is warming (e.g.,
    poor heating control), while negative suggests cooling.

    Parameters
    ----------
    signal : np.ndarray, shape (N,)
        Input time-series.

    Returns
    -------
    float
        Slope in (signal units) per sample.
    """
    n = len(signal)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    # Efficient closed-form least-squares slope
    x_mean = x.mean()
    s_mean = signal.mean()
    numerator = np.dot(x - x_mean, signal - s_mean)
    denominator = np.dot(x - x_mean, x - x_mean)
    return float(numerator / denominator) if denominator > 1e-12 else 0.0


def _count_threshold_crossings(
    signal: np.ndarray,
    threshold: float,
    direction: str = "above",
) -> int:
    """Count the number of times a signal crosses a threshold.

    We count RISING threshold crossings only (transitions from below to
    above) to avoid double-counting on-events.  This gives the number
    of distinct disturbance episodes.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    threshold : float
        Threshold value.
    direction : str
        'above' counts low→high transitions; 'below' counts high→low.

    Returns
    -------
    int
        Number of threshold crossings.
    """
    if direction == "above":
        below = signal < threshold
        crossings = np.sum(below[:-1] & ~below[1:])  # False→True transitions
    else:
        above = signal >= threshold
        crossings = np.sum(above[:-1] & ~above[1:])  # True→False transitions
    return int(crossings)


# ---------------------------------------------------------------------------
# Main feature extraction function
# ---------------------------------------------------------------------------

def extract_features(session: Dict) -> Dict[str, float]:
    """Extract the full 36-feature vector from one session dictionary.

    We compute all features from the raw time-series stored in ``session``
    and return a flat dictionary mapping feature names to scalar values.
    This dictionary becomes one row in the final dataset DataFrame.

    Parameters
    ----------
    session : dict
        Session record from ``data_generation.generate_session``.
        Must contain 'temperature', 'light', 'humidity', 'noise' arrays.

    Returns
    -------
    dict
        Feature dictionary with 36 entries (keys from FEATURE_NAMES).
    """
    temp = session["temperature"]
    light = session["light"]
    humidity = session["humidity"]
    noise = session["noise"]

    # HVAC frequency band: cycles corresponding to 60–120 minute periods
    # In cpm: f_low = 1/120 cpm, f_high = 1/60 cpm
    f_hvac_low = 1.0 / 120.0   # cpm
    f_hvac_high = 1.0 / 60.0   # cpm

    freqs_t, psd_t = compute_fft_spectrum(temp)

    features: Dict[str, float] = {}

    # --- Temperature features ---
    features["temp_mean"]       = float(temp.mean())
    features["temp_std"]        = float(temp.std(ddof=1))
    features["temp_min"]        = float(temp.min())
    features["temp_max"]        = float(temp.max())
    features["temp_range"]      = float(temp.max() - temp.min())
    features["temp_slope"]      = _linear_slope(temp) / SAMPLE_RATE_MIN  # per minute
    features["temp_acf1"]       = float(autocorrelation(temp, max_lag=1)[1])
    features["temp_dom_freq"]   = dominant_frequency(freqs_t, psd_t)
    features["temp_hvac_power"] = spectral_power_in_band(
        freqs_t, psd_t, f_hvac_low, f_hvac_high
    )
    features["temp_skewness"]   = float(skew(temp))
    # Fraction of time in clinical optimal range [18, 21] °C
    features["temp_in_optimal"] = float(np.mean((temp >= 18.0) & (temp <= 21.0)))

    # --- Light features ---
    features["light_mean"]      = float(light.mean())
    features["light_std"]       = float(light.std(ddof=1))
    features["light_max"]       = float(light.max())
    features["light_n_events"]  = float(_count_threshold_crossings(light, 10.0, "above"))
    features["light_frac_dark"] = float(np.mean(light < 5.0))
    # Integrated illuminance dose (lux·min): area under light curve × sample interval
    features["light_total_lux_min"] = float(light.sum() * SAMPLE_RATE_MIN)

    # --- Humidity features ---
    features["humidity_mean"]   = float(humidity.mean())
    features["humidity_std"]    = float(humidity.std(ddof=1))
    features["humidity_min"]    = float(humidity.min())
    features["humidity_max"]    = float(humidity.max())
    features["humidity_in_comfort"] = float(
        np.mean((humidity >= 40.0) & (humidity <= 60.0))
    )
    features["humidity_acf1"]   = float(autocorrelation(humidity, max_lag=1)[1])

    # --- Noise features ---
    features["noise_mean"]      = float(noise.mean())
    features["noise_std"]       = float(noise.std(ddof=1))
    features["noise_max"]       = float(noise.max())
    features["noise_n_events"]  = float(_count_threshold_crossings(noise, 45.0, "above"))
    features["noise_frac_quiet"]= float(np.mean(noise < 35.0))
    # L90: the level exceeded 90% of the time (standard acoustic background metric)
    features["noise_l90"]       = float(np.percentile(noise, 10))  # 10th pct = exceeded 90%

    # --- Cross-signal features ---
    # Guard against zero-variance signals (e.g., perfectly dark room → light std=0)
    # which make Pearson correlation undefined (0/0 → NaN).  We fall back to 0.0.
    def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
        if a.std() < 1e-12 or b.std() < 1e-12:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    features["corr_temp_humidity"] = safe_pearson(temp, humidity)
    features["corr_light_noise"]   = safe_pearson(light, noise)

    # Composite environmental stress score
    # Combines all major penalty sources into one scalar (higher = more stressful)
    temp_stress = max(0.0, float(temp.mean()) - 21.0) + max(0.0, 18.0 - float(temp.mean()))
    light_stress = features["light_n_events"] * 0.5
    noise_stress = features["noise_n_events"] * 0.5
    humidity_stress = (
        max(0.0, 40.0 - features["humidity_mean"]) +
        max(0.0, features["humidity_mean"] - 60.0)
    ) * 0.1
    features["env_stress_score"] = temp_stress + light_stress + noise_stress + humidity_stress

    # --- Spectral entropy features ---
    features["temp_spectral_entropy"]  = _spectral_entropy(temp)
    features["noise_spectral_entropy"] = _spectral_entropy(noise)

    return features


def build_feature_matrix(
    sessions: List[Dict],
) -> tuple:
    """Build the full feature matrix X and label matrix y from all sessions.

    Parameters
    ----------
    sessions : list of dict
        All session records from ``generate_dataset``.

    Returns
    -------
    X : np.ndarray, shape (N, 36)
        Feature matrix.
    y : np.ndarray, shape (N, 4)
        Label matrix: [sleep_efficiency, sleep_duration_h, awakenings, sleep_score]
    feature_names : list of str
        Column names for X.
    label_names : list of str
        Column names for y.
    metadata : list of dict
        Per-session metadata (session_id, season, quality_class).
    """
    label_names = ["sleep_efficiency", "sleep_duration_h", "awakenings", "sleep_score"]

    X_rows, y_rows, metadata = [], [], []
    for sess in sessions:
        feats = extract_features(sess)
        X_rows.append([feats[name] for name in FEATURE_NAMES])
        y_rows.append([sess[lbl] for lbl in label_names])
        metadata.append({
            "session_id":    sess["session_id"],
            "season":        sess["season"],
            "quality_class": sess["quality_class"],
        })

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_rows, dtype=np.float64)

    logger.info(
        "Feature matrix built: X=%s, y=%s",
        X.shape, y.shape,
    )
    return X, y, FEATURE_NAMES, label_names, metadata
