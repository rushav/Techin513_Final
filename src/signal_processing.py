"""
signal_processing.py — Core signal-processing toolkit.

We implement the four principal SP techniques used throughout the pipeline:

1. **Butterworth low-pass filtering** — removes unphysical high-frequency
   fluctuations while preserving circadian and HVAC-scale dynamics.

2. **Spectral synthesis (IFFT-based)** — constructs coloured noise with
   a prescribed power spectral density (1/f for pink noise) directly in
   the frequency domain, then transforms back via IFFT.

3. **FFT analysis** — decomposes each generated signal into its spectral
   components so we can verify that the expected frequency peaks (circadian
   ~1/1440 min⁻¹, HVAC ~1/90 min⁻¹) are present.

4. **Poisson event injection** — models discrete, randomly timed disturbances
   (bathroom light flicks, noise spikes) whose inter-arrival times follow an
   exponential distribution, consistent with a homogeneous Poisson process.

Authors: Rushav Dash, Lisa Li — TECHIN 513 Final Project
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi, welch
from typing import Tuple

from .utils import N_SAMPLES, SAMPLE_RATE_MIN, NYQUIST_CPM, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 1.  Butterworth low-pass filter
# ---------------------------------------------------------------------------

def design_butterworth_lpf(
    cutoff_cpm: float,
    order: int = 4,
    fs_cpm: float = 1.0 / SAMPLE_RATE_MIN,
) -> np.ndarray:
    """Design a Butterworth low-pass filter and return second-order sections.

    We choose the Butterworth design because it provides a maximally flat
    (Chebyshev-equiripple-free) magnitude response in the passband.  For
    bedroom temperature this is important: we do not want ringing artefacts
    introduced by sharper filters (e.g., Chebyshev Type I) to mimic the
    look of genuine thermal events.

    We express all frequencies in cycles per minute (cpm) to match our
    5-minute sampling rate.  The normalised cutoff W_n = cutoff / Nyquist.

    Parameters
    ----------
    cutoff_cpm : float
        Cutoff frequency in cycles per minute.  Signals above this frequency
        are attenuated.
    order : int
        Filter order.  Higher order → steeper roll-off but more phase
        distortion.  We default to 4 (−24 dB/octave), which gives adequate
        attenuation without introducing perceptible group-delay variation
        across the passband.
    fs_cpm : float
        Sampling frequency in cycles per minute.  Defaults to 1/5 cpm.

    Returns
    -------
    sos : np.ndarray, shape (order//2, 6)
        Second-order sections representation for numerically stable
        filtering (preferred over transfer-function coefficients for
        higher-order filters).
    """
    # W_n is the normalised frequency in (0, 1), where 1 = Nyquist.
    nyquist = fs_cpm / 2.0
    W_n = cutoff_cpm / nyquist
    # Clamp to a safe range to prevent degenerate filter designs.
    W_n = float(np.clip(W_n, 1e-4, 0.9999))
    sos = butter(order, W_n, btype="low", output="sos")
    return sos


def apply_butterworth_lpf(
    signal: np.ndarray,
    cutoff_cpm: float,
    order: int = 4,
) -> np.ndarray:
    """Filter a 1-D signal with a Butterworth LPF using SOS cascade.

    We use ``sosfilt`` (forward-only) rather than ``sosfiltfilt``
    (zero-phase) because the causal filter better represents real-time
    sensor smoothing.  For offline post-processing the zero-phase variant
    is preferable, but our goal is to mimic what an IoT gateway would
    report, so causal filtering is the right choice.

    Parameters
    ----------
    signal : np.ndarray, shape (N,)
        Raw input signal.
    cutoff_cpm : float
        Cutoff frequency in cycles per minute.
    order : int
        Butterworth filter order (default 4).

    Returns
    -------
    filtered : np.ndarray, shape (N,)
        Low-pass filtered signal.
    """
    sos = design_butterworth_lpf(cutoff_cpm, order)
    # Initialise the filter state to the first sample value so the output
    # tracks the signal from sample 0 without a startup transient.
    # A zero initial condition would cause the filter to ramp up from 0
    # over ~(1/cutoff) samples, producing physically impossible fast drifts
    # at the beginning of each session (e.g., 3–6 °C/sample for temperature).
    zi = sosfilt_zi(sos) * signal[0]
    filtered, _ = sosfilt(sos, signal, zi=zi)
    return filtered


# ---------------------------------------------------------------------------
# 2.  Spectral synthesis — pink (1/f) noise
# ---------------------------------------------------------------------------

def generate_pink_noise(
    n: int = N_SAMPLES,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Synthesise unit-variance pink noise via FFT-based spectral shaping.

    Pink noise (power ∝ 1/f) arises naturally in many physical systems
    including thermal fluctuations and neurological activity.  We construct
    it by:

      1. Drawing white-noise coefficients in the frequency domain
         (complex Gaussian with unit variance).
      2. Multiplying the amplitude spectrum by 1/√f to impose the
         1/f power relationship (power = amplitude²).
      3. Applying conjugate symmetry to guarantee a real-valued output.
      4. Taking the IFFT to obtain the time-domain signal.
      5. Normalising to zero mean and unit variance.

    This approach is exact in distribution (unlike the Voss–McCartney
    algorithm) and efficient—O(N log N) via the FFT.

    Parameters
    ----------
    n : int
        Number of samples to generate.
    rng : np.random.Generator, optional
        Reproducible random number generator.  If None, we create one
        seeded with the global seed.

    Returns
    -------
    pink : np.ndarray, shape (n,)
        Zero-mean, unit-variance pink noise array.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Step 1 — white noise in frequency domain
    f_coeffs = rng.standard_normal(n) + 1j * rng.standard_normal(n)

    # Step 2 — build the 1/f amplitude weighting
    freqs = np.fft.rfftfreq(n)          # frequencies for real FFT, shape (n//2+1,)
    freqs[0] = 1.0                       # avoid divide-by-zero at DC; DC → no colouring

    # We work with the real-FFT representation for efficiency
    f_real = rng.standard_normal(n // 2 + 1) + 1j * rng.standard_normal(n // 2 + 1)
    # Amplitude weighting: multiply by 1/sqrt(f) so that power ~ 1/f
    f_real[1:] /= np.sqrt(freqs[1:])
    f_real[0] = 0.0                      # zero DC component → zero mean output

    # Step 3–4 — inverse FFT back to time domain
    pink = np.fft.irfft(f_real, n=n)

    # Step 5 — normalise to zero mean and unit standard deviation
    pink -= pink.mean()
    std = pink.std()
    if std > 1e-12:
        pink /= std

    return pink


# ---------------------------------------------------------------------------
# 3.  FFT analysis
# ---------------------------------------------------------------------------

def compute_fft_spectrum(
    signal: np.ndarray,
    fs_cpm: float = 1.0 / SAMPLE_RATE_MIN,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the single-sided power spectral density via FFT.

    We use the real FFT (``rfft``) because our signals are real-valued,
    which halves the computation and avoids the redundant negative-frequency
    mirror.  The PSD (power per unit frequency) is obtained by squaring the
    magnitude of the FFT coefficients and applying the Parseval-consistent
    normalisation factor 2 / (N * fs).

    Parameters
    ----------
    signal : np.ndarray, shape (N,)
        Input signal (zero-padded internally if needed).
    fs_cpm : float
        Sampling frequency in cycles per minute (default 1/5 cpm).

    Returns
    -------
    freqs : np.ndarray, shape (N//2+1,)
        Frequency axis in cycles per minute.
    psd : np.ndarray, shape (N//2+1,)
        One-sided power spectral density (unit²/cpm).
    """
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_cpm)  # cpm axis

    # Normalised FFT coefficients
    spectrum = np.fft.rfft(signal - signal.mean())  # remove DC before transform

    # Single-sided PSD: factor 2 accounts for the energy in the negative
    # frequencies being folded into the positive half.
    psd = (2.0 / (n * fs_cpm)) * np.abs(spectrum) ** 2
    # DC and Nyquist bins are not doubled (they have no negative-freq twin)
    psd[0] /= 2.0
    if n % 2 == 0:
        psd[-1] /= 2.0

    return freqs, psd


def welch_psd(
    signal: np.ndarray,
    fs_cpm: float = 1.0 / SAMPLE_RATE_MIN,
    nperseg: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate PSD via Welch's method (averaged periodogram).

    Welch's method reduces spectral variance by averaging over overlapping
    windowed segments.  We use it as a more robust alternative to the raw
    FFT periodogram when the signal is short (96 samples).

    Parameters
    ----------
    signal : np.ndarray, shape (N,)
        Input signal.
    fs_cpm : float
        Sampling frequency in cpm.
    nperseg : int
        Samples per Welch segment.  We default to 32 (≈ 160 min per segment)
        which gives 3 segments with 50% overlap for our 96-sample signals.

    Returns
    -------
    freqs : np.ndarray
        Frequency axis (cpm).
    psd : np.ndarray
        Power spectral density estimate.
    """
    freqs, psd = welch(signal, fs=fs_cpm, nperseg=min(nperseg, len(signal)))
    return freqs, psd


def dominant_frequency(freqs: np.ndarray, psd: np.ndarray) -> float:
    """Return the frequency bin with maximum spectral power (excluding DC).

    Parameters
    ----------
    freqs : np.ndarray
        Frequency axis from ``compute_fft_spectrum`` or ``welch_psd``.
    psd : np.ndarray
        Corresponding power spectral density.

    Returns
    -------
    float
        Dominant frequency in cycles per minute.
    """
    # Skip DC bin (index 0) to avoid the mean offset dominating
    idx = np.argmax(psd[1:]) + 1
    return float(freqs[idx])


def spectral_power_in_band(
    freqs: np.ndarray,
    psd: np.ndarray,
    f_low: float,
    f_high: float,
) -> float:
    """Integrate PSD over a specified frequency band.

    We use the trapezoid rule for numerical integration over the discrete
    frequency grid.  This gives a physically meaningful power quantity
    (unit²) rather than a raw sum of bins.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency axis in cpm.
    psd : np.ndarray
        Power spectral density.
    f_low, f_high : float
        Lower and upper band edges (cpm).

    Returns
    -------
    float
        Integrated band power.
    """
    mask = (freqs >= f_low) & (freqs <= f_high)
    if mask.sum() < 2:
        return 0.0
    # np.trapezoid is the NumPy 2.x name; np.trapz was deprecated in 2.0
    return float(np.trapezoid(psd[mask], freqs[mask]))


# ---------------------------------------------------------------------------
# 4.  Poisson event injection
# ---------------------------------------------------------------------------

def generate_poisson_events(
    rate_per_min: float,
    duration_min: float,
    min_gap_min: float = 5.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate event arrival times via a homogeneous Poisson process.

    A Poisson process with rate λ produces inter-arrival times that are
    independent and identically distributed according to Exp(λ).  This is
    the correct model for rare, memoryless events like a partner switching
    on a bedroom light or a car horn waking a sleeper.

    We optionally enforce a minimum gap between events to prevent
    overlapping pulses in the discrete time grid.

    Parameters
    ----------
    rate_per_min : float
        Mean number of events per minute (λ).
    duration_min : float
        Total time window in minutes.
    min_gap_min : float
        Minimum gap between consecutive events in minutes (default 5 min =
        one sample interval, to prevent double-events in the same bin).
    rng : np.random.Generator, optional
        Random generator for reproducibility.

    Returns
    -------
    arrival_times : np.ndarray
        Array of event arrival times in minutes, sorted ascending.
        May be empty if no events fall within ``duration_min``.
    """
    if rng is None:
        rng = np.random.default_rng()

    if rate_per_min <= 0:
        return np.array([])

    # Draw the correct number of events within [0, duration_min] by:
    #   1. Sampling n ~ Poisson(λ·T) — the exact count for this window.
    #   2. Placing the n events UNIFORMLY at random in [0, T].
    # This is mathematically equivalent to a homogeneous Poisson process
    # (by the order-statistics characterisation of the Poisson process) and
    # avoids the systematic undercounting bug in the previous implementation:
    #   The old code drew n Exp(λ) inter-arrivals and then truncated those
    #   whose cumulative sum exceeded T.  For low-rate processes (λ·T ≈ 1–3),
    #   many drawn events fell outside the window, so the actual event count
    #   was only 25–65 % of the Poisson-distributed expected value.
    expected_n = rate_per_min * duration_min
    n_events = rng.poisson(expected_n)
    if n_events == 0:
        return np.array([])

    # Place n_events uniformly in [0, duration_min) and sort ascending.
    arrivals = np.sort(rng.uniform(0.0, duration_min, size=n_events))

    # Enforce minimum gap: drop events that are too close to the preceding one
    if min_gap_min > 0 and len(arrivals) > 1:
        keep = [True] * len(arrivals)
        for i in range(1, len(arrivals)):
            if arrivals[i] - arrivals[i - 1] < min_gap_min:
                keep[i] = False
        arrivals = arrivals[keep]

    return arrivals


def inject_events_into_signal(
    signal: np.ndarray,
    arrival_times_min: np.ndarray,
    amplitudes: np.ndarray,
    duration_samples: int,
    sample_rate_min: float = SAMPLE_RATE_MIN,
) -> np.ndarray:
    """Add rectangular pulse events at the specified arrival times.

    Each event is a rectangular pulse of the given amplitude and duration.
    We chose rectangular pulses (rather than Gaussian or exponential decay)
    because they accurately model sharp-onset disturbances: a light switch
    that turns on and stays on, or a noise source that abruptly starts and
    stops.

    Parameters
    ----------
    signal : np.ndarray, shape (N,)
        Base signal to add events to (not modified in-place).
    arrival_times_min : np.ndarray
        Event start times in minutes.
    amplitudes : np.ndarray
        Amplitude of each event pulse (same length as ``arrival_times_min``).
    duration_samples : int
        Duration of each pulse in samples.  A value of 1 corresponds to
        a single-sample impulse (delta function approximation).
    sample_rate_min : float
        Minutes per sample.

    Returns
    -------
    result : np.ndarray, shape (N,)
        Signal with events injected.
    """
    result = signal.copy()
    n = len(signal)
    for t_min, amp in zip(arrival_times_min, amplitudes):
        # Convert arrival time to sample index
        start_idx = int(t_min / sample_rate_min)
        end_idx = min(start_idx + duration_samples, n)
        if start_idx < n:
            result[start_idx:end_idx] += amp
    return result


# ---------------------------------------------------------------------------
# 5.  Autocorrelation (validation helper)
# ---------------------------------------------------------------------------

def autocorrelation(signal: np.ndarray, max_lag: int = 20) -> np.ndarray:
    """Compute normalised autocorrelation function up to ``max_lag`` lags.

    We use this to verify that our filtered signals retain the temporal
    structure expected from real bedroom sensors: filtered temperature
    should show high ACF at lag-1 (strong persistence), while raw white
    noise would not.

    Parameters
    ----------
    signal : np.ndarray, shape (N,)
        Zero-mean signal (we demean internally).
    max_lag : int
        Maximum lag to compute.

    Returns
    -------
    acf : np.ndarray, shape (max_lag + 1,)
        ACF at lags 0, 1, …, max_lag.  ACF[0] = 1.0 by definition.
    """
    x = signal - signal.mean()
    n = len(x)
    # Full normalised autocorrelation via FFT (O(N log N))
    full = np.fft.irfft(np.abs(np.fft.rfft(x, n=2 * n)) ** 2)
    acf = full[:max_lag + 1] / full[0]
    return acf
