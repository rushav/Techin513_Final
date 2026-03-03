"""
data_generation.py — Synthetic bedroom environment signal generation.

We generate four physically plausible time-series per sleep session:

  • Temperature (°C)   — slow sinusoidal drift + HVAC sawtooth + pink noise
  • Illuminance (lux)  — near-dark baseline + Poisson light-intrusion events
  • Relative Humidity (%) — anti-correlated with temperature + pink noise
  • Ambient Noise (dB SPL) — quiet baseline + Poisson disturbance spikes

Every design decision below is grounded in sleep science and IoT sensor
literature.  Key references used in parameter choice:

  [1] Okamoto-Mizuno & Mizuno (2012). Effects of thermal environment on
      sleep and circadian rhythm. J Physiol Anthropol 31(1):14.
      → Optimal sleep temperature 18–21 °C; HVAC cycle 60–90 min.

  [2] Porcheret et al. (2015). The effect of two types of light on sleep
      quality in healthy adults. J Sleep Res 24(5):535–545.
      → Even dim light (10 lux) can suppress melatonin; bedroom target < 5 lux.

  [3] Hume et al. (2012). Objective measurement of sleep disturbance.
      J Sleep Res 21(5):539–546.
      → Noise events > 45 dB cause micro-arousals at 0.3–0.7/hour rate.

Authors: Rushav Dash, Lisa Li — TECHIN 513 Final Project
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .signal_processing import (
    apply_butterworth_lpf,
    generate_pink_noise,
    generate_poisson_events,
    inject_events_into_signal,
)
from .utils import (
    N_SAMPLES,
    SAMPLE_RATE_MIN,
    SESSION_DURATION_MIN,
    get_logger,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Session profile: encapsulates per-session randomised parameters
# ---------------------------------------------------------------------------

@dataclass
class SessionProfile:
    """Random draw of environmental parameters for one 8-hour sleep session.

    We stratify sessions across four seasons and three quality classes to
    ensure diversity in the generated dataset.  All physical parameter
    ranges are drawn from published IoT and sleep science literature.

    Attributes
    ----------
    session_id : int
        Unique integer identifier for this session.
    season : str
        One of {'winter', 'spring', 'summer', 'autumn'}.  Controls the
        base temperature offset (winter cooler, summer warmer).
    quality_class : str
        One of {'poor', 'moderate', 'good'}.  Influences noise/light
        disturbance rates and optimal-temperature proximity.
    base_temp_c : float
        Thermostat setpoint in °C.  Range 16–26 °C across seasons.
    hvac_period_min : float
        HVAC duty cycle in minutes.  Range 60–120 min (Ref [1]).
    hvac_amplitude_c : float
        Peak-to-peak temperature variation due to HVAC sawtooth (°C).
    temp_noise_scale : float
        Standard deviation of the pink-noise temperature component (°C).
    light_base_lux : float
        Persistent ambient light level (street lights, standby LEDs).
        Range 0–5 lux for a dark bedroom (Ref [2]).
    light_event_rate : float
        Poisson rate of light-intrusion events (events/minute).
    light_event_lux : float
        Mean illuminance of each light event (lux).
    humidity_base_pct : float
        Mean relative humidity (%).  Range 35–65 %.
    noise_base_db : float
        Ambient noise floor (dB SPL).  Range 25–40 dB (quiet bedroom).
    noise_event_rate : float
        Poisson rate of noise disturbance events (events/minute).
    noise_event_db : float
        Mean amplitude of each noise event above the floor (dB SPL).
    rng_seed : int
        Per-session RNG seed derived from session_id + global seed.
    """

    session_id: int
    season: str
    quality_class: str
    base_temp_c: float
    hvac_period_min: float
    hvac_amplitude_c: float
    temp_noise_scale: float
    light_base_lux: float
    light_event_rate: float
    light_event_lux: float
    humidity_base_pct: float
    noise_base_db: float
    noise_event_rate: float
    noise_event_db: float
    rng_seed: int


# Seasonal temperature offsets in °C relative to a reference of 20 °C
_SEASON_TEMP_OFFSET: Dict[str, tuple] = {
    # (mean_offset, std_offset) — real bedroom thermostat distributions
    "winter":  (-2.0, 1.5),   # 16–20 °C typical
    "spring":  ( 0.0, 1.5),   # 18–22 °C typical
    "summer":  ( 3.0, 1.5),   # 20–26 °C typical
    "autumn":  ( 0.5, 1.5),   # 17–22 °C typical
}

# Disturbance rates scale with quality class (events per minute).
# We calibrate these so that mean awakenings across classes is ~2.5,
# matching the published Sleep Efficiency Dataset mean.
_QUALITY_LIGHT_RATE: Dict[str, float] = {
    "good":     0.002,   # ~0.1 events/hour — very dark room
    "moderate": 0.005,   # ~0.3 events/hour — occasional phone/bathroom light
    "poor":     0.010,   # ~0.6 events/hour — frequent disturbances
}
_QUALITY_NOISE_RATE: Dict[str, float] = {
    "good":     0.002,   # ~0.1 events/hour — quiet suburban bedroom
    "moderate": 0.004,   # ~0.24 events/hour
    "poor":     0.008,   # ~0.48 events/hour — noisy urban environment
}


def sample_session_profile(
    session_id: int,
    global_seed: int = 42,
    seasons: Optional[List[str]] = None,
    quality_classes: Optional[List[str]] = None,
) -> SessionProfile:
    """Draw a random session profile, stratified by season and quality class.

    We use a deterministic per-session seed (global_seed + session_id) so
    that any individual session can be reproduced independently.  The
    stratification ensures the dataset covers all four seasons and three
    quality levels proportionally.

    Parameters
    ----------
    session_id : int
        Session index (0-based).
    global_seed : int
        Master seed from which per-session seeds are derived.
    seasons : list of str, optional
        Override the default season pool.
    quality_classes : list of str, optional
        Override the default quality pool.

    Returns
    -------
    SessionProfile
        Fully-specified parameter set for this session.
    """
    if seasons is None:
        seasons = ["winter", "spring", "summer", "autumn"]
    if quality_classes is None:
        quality_classes = ["poor", "moderate", "good"]

    rng_seed = global_seed + session_id
    rng = np.random.default_rng(rng_seed)

    # Stratified assignment: distribute sessions evenly across groups
    # We use modulo so that every 12th session completes one full cycle
    # over all 4 seasons × 3 quality classes.
    n_seasons = len(seasons)
    n_qualities = len(quality_classes)
    season = seasons[session_id % n_seasons]
    quality_class = quality_classes[(session_id // n_seasons) % n_qualities]

    # --- Temperature parameters ---
    temp_mean, temp_std = _SEASON_TEMP_OFFSET[season]
    base_temp = 20.0 + rng.normal(temp_mean, temp_std)
    base_temp = float(np.clip(base_temp, 14.0, 28.0))  # physical range

    # HVAC cycle: 60–120 minutes (from thermostat dead-band dynamics)
    hvac_period = float(rng.uniform(60.0, 120.0))
    # HVAC amplitude: 0.5–2.0 °C (modern smart thermostats are tight)
    hvac_amp = float(rng.uniform(0.5, 2.0))
    # Thermal noise level: 0.1–0.5 °C (sensor noise + air stratification)
    temp_noise = float(rng.uniform(0.1, 0.5))

    # --- Light parameters ---
    # Ambient light base: 0–5 lux (dark bedroom, small LED indicators)
    light_base = float(rng.uniform(0.0, 4.0))
    # Light event rate: depends on quality class, with ±20% session variation
    light_rate = _QUALITY_LIGHT_RATE[quality_class] * float(rng.uniform(0.8, 1.2))
    # Event intensity: 20–200 lux (bathroom light, phone screen, streetlight)
    light_lux = float(rng.uniform(20.0, 200.0))

    # --- Humidity parameters ---
    # Comfortable indoor humidity: 35–65% (ASHRAE 55 standard)
    humidity_base = float(rng.uniform(35.0, 65.0))

    # --- Noise parameters ---
    # Quiet bedroom baseline: 25–40 dB SPL (WHO recommendation < 35 dB)
    noise_base = float(rng.uniform(25.0, 40.0))
    noise_rate = _QUALITY_NOISE_RATE[quality_class] * float(rng.uniform(0.8, 1.2))
    # Noise event magnitude: 10–30 dB above floor (snoring ~55 dB, traffic ~70 dB)
    noise_event = float(rng.uniform(10.0, 30.0))

    return SessionProfile(
        session_id=session_id,
        season=season,
        quality_class=quality_class,
        base_temp_c=base_temp,
        hvac_period_min=hvac_period,
        hvac_amplitude_c=hvac_amp,
        temp_noise_scale=temp_noise,
        light_base_lux=light_base,
        light_event_rate=light_rate,
        light_event_lux=light_lux,
        humidity_base_pct=humidity_base,
        noise_base_db=noise_base,
        noise_event_rate=noise_rate,
        noise_event_db=noise_event,
        rng_seed=rng_seed,
    )


# ---------------------------------------------------------------------------
# Individual signal generators
# ---------------------------------------------------------------------------

def generate_temperature(
    profile: SessionProfile,
    t_min: np.ndarray,
) -> np.ndarray:
    """Generate an 8-hour bedroom temperature time-series.

    We compose three physically motivated components:

    1. **Sinusoidal circadian drift**: bedroom temperature follows a daily
       cycle driven by outdoor temperature and human body heat.  We use
       a 24-hour sinusoid (ω = 2π/1440 rad/min) with amplitude 1–2 °C.

    2. **HVAC sawtooth wave**: a heating/cooling system overshoots its
       setpoint and then coasts back — a classic relaxation oscillator.
       We model this as a sawtooth with period 60–120 min and clip the
       amplitude to ±hvac_amplitude_c.

    3. **Pink (1/f) thermal noise**: real thermal sensors exhibit
       autocorrelated noise (not white) due to heat diffusion.  We add
       pink noise scaled to ~0.1–0.5 °C before Butterworth filtering.

    The composite is then low-pass filtered (cutoff = 0.02 cpm, ≈ 1/50 min,
    period ≈ 50 min) to remove any residual artefacts above the HVAC
    timescale while preserving both the circadian and HVAC components.

    Parameters
    ----------
    profile : SessionProfile
        Session-specific parameters.
    t_min : np.ndarray, shape (N,)
        Time axis in minutes from session start (0, 5, 10, …, 475).

    Returns
    -------
    temperature : np.ndarray, shape (N,)
        Temperature in °C, clipped to [10, 35] for physical realism.
    """
    rng = np.random.default_rng(profile.rng_seed)

    # Component 1 — circadian sinusoidal drift
    # We start at a random phase within the 24-hour cycle to represent
    # sessions starting at different clock times (typically 22:00–00:00).
    circadian_phase = rng.uniform(0.0, 2.0 * np.pi)
    circadian_amplitude = rng.uniform(0.5, 1.5)  # °C
    # Angular frequency: 2π radians per 1440 minutes
    omega_circadian = 2.0 * np.pi / 1440.0
    circadian = circadian_amplitude * np.sin(omega_circadian * t_min + circadian_phase)

    # Component 2 — HVAC sawtooth
    # A sawtooth wave with period T_hvac in minutes: value rises linearly
    # from -A to +A over one period (the heat-on phase), then drops (heat-off).
    # ``t_min % T`` gives the phase within each HVAC cycle.
    T_hvac = profile.hvac_period_min
    phase_hvac = (t_min % T_hvac) / T_hvac          # normalised phase ∈ [0, 1)
    sawtooth = 2.0 * phase_hvac - 1.0               # linear ramp from -1 to +1
    sawtooth *= profile.hvac_amplitude_c             # scale to physical amplitude

    # Component 3 — pink (1/f) thermal noise
    pink = generate_pink_noise(n=len(t_min), rng=rng)
    pink *= profile.temp_noise_scale                 # scale to desired std dev (°C)

    # Composite raw signal
    raw = profile.base_temp_c + circadian + sawtooth + pink

    # Low-pass filter: cutoff at 0.02 cpm (period ~50 min)
    # This retains HVAC (period ≥ 60 min) and circadian (1440 min) but
    # removes any high-frequency artefacts introduced by noise or HVAC edges.
    cutoff_cpm = 0.02  # cycles per minute
    filtered = apply_butterworth_lpf(raw, cutoff_cpm=cutoff_cpm, order=4)

    # Physical range clip: no realistic bedroom falls below 10 °C or above 35 °C
    return np.clip(filtered, 10.0, 35.0)


def generate_light(
    profile: SessionProfile,
    t_min: np.ndarray,
) -> np.ndarray:
    """Generate an 8-hour bedroom illuminance time-series.

    Bedroom light during sleep should be near zero.  We model:

    1. **Dark baseline**: constant ambient level (0–5 lux) from streetlights,
       power-indicator LEDs, and moonlight through curtains.

    2. **Poisson light events**: discrete disturbances (partner using phone,
       bathroom visit) modelled as a Poisson arrival process.  Each event
       has a randomly drawn duration (1–4 samples) and amplitude (20–200 lux).
       Inter-arrivals are exponentially distributed, consistent with the
       memoryless property of a Poisson process.

    We do NOT filter the light signal — abrupt transitions (switch on/off)
    are physically accurate and we want to preserve the sharp edges for
    feature extraction (event counting).

    Parameters
    ----------
    profile : SessionProfile
        Session-specific parameters.
    t_min : np.ndarray, shape (N,)
        Time axis in minutes.

    Returns
    -------
    light : np.ndarray, shape (N,)
        Illuminance in lux, non-negative.
    """
    rng = np.random.default_rng(profile.rng_seed + 1000)  # offset to decorrelate

    n = len(t_min)
    # Step 1 — dark baseline
    light = np.full(n, profile.light_base_lux)

    # Step 2 — Poisson light events
    arrivals = generate_poisson_events(
        rate_per_min=profile.light_event_rate,
        duration_min=SESSION_DURATION_MIN,
        min_gap_min=SAMPLE_RATE_MIN,
        rng=rng,
    )

    if len(arrivals) > 0:
        # Each event has an independent amplitude draw
        amplitudes = rng.uniform(
            0.5 * profile.light_event_lux,
            1.5 * profile.light_event_lux,
            size=len(arrivals),
        )
        # Duration: 1–4 samples (5–20 minutes), exponentially distributed
        durations = np.maximum(1, rng.exponential(1.5, size=len(arrivals)).astype(int))
        durations = np.minimum(durations, 4)  # cap at 4 samples = 20 minutes

        for t, amp, dur in zip(arrivals, amplitudes, durations):
            start = int(t / SAMPLE_RATE_MIN)
            end = min(start + dur, n)
            if start < n:
                light[start:end] += amp

    return np.clip(light, 0.0, 5000.0)  # lux upper bound (bright indoor scene)


def generate_humidity(
    profile: SessionProfile,
    t_min: np.ndarray,
    temperature: np.ndarray,
) -> np.ndarray:
    """Generate an 8-hour relative humidity time-series.

    Relative humidity (RH) is negatively correlated with temperature at
    constant absolute humidity — as air temperature rises, its water-vapour
    capacity increases, so RH falls.  We exploit this physical relationship:

      RH(t) = base_RH − β × [T(t) − mean(T)] + pink_noise

    where β ≈ 2 %/°C is a typical indoor value (derived from the Magnus
    formula for typical indoor conditions).  We then apply a gentle
    Butterworth LPF (cutoff 0.01 cpm, period ~100 min) because humidity
    changes even more slowly than temperature due to the thermal inertia
    of walls and furniture.

    Parameters
    ----------
    profile : SessionProfile
        Session parameters.
    t_min : np.ndarray, shape (N,)
        Time axis in minutes.
    temperature : np.ndarray, shape (N,)
        Already-generated temperature signal (°C), used for anti-correlation.

    Returns
    -------
    humidity : np.ndarray, shape (N,)
        Relative humidity (%), clipped to [20, 80].
    """
    rng = np.random.default_rng(profile.rng_seed + 2000)

    # β: change in RH per degree Celsius of temperature variation
    # A value of 2 %/°C is conservative for indoor sealed rooms.
    beta = rng.uniform(1.5, 2.5)
    temp_anomaly = temperature - temperature.mean()

    # Pink noise for humidity: slower dynamics than temperature
    pink = generate_pink_noise(n=len(t_min), rng=rng)
    noise_scale = rng.uniform(0.5, 2.0)  # %RH standard deviation
    pink *= noise_scale

    raw = profile.humidity_base_pct - beta * temp_anomaly + pink

    # Very gentle LPF for humidity (cutoff 0.01 cpm, period 100 min)
    filtered = apply_butterworth_lpf(raw, cutoff_cpm=0.01, order=2)

    return np.clip(filtered, 20.0, 80.0)


def generate_noise_level(
    profile: SessionProfile,
    t_min: np.ndarray,
) -> np.ndarray:
    """Generate an 8-hour ambient noise level time-series.

    Noise is modelled similarly to light: a quiet baseline with Poisson
    disturbance events.  However, unlike light (binary on/off), noise
    events decay exponentially after onset because acoustic energy
    dissipates over time.

    We model each noise event as a one-sided exponential decay:
      amplitude(t) = A × exp(−t / τ)  for t ≥ t_event
    where τ ~ Uniform(2, 8) minutes for typical transient noise sources
    (car horn, door slam, snoring bout).

    Parameters
    ----------
    profile : SessionProfile
        Session parameters.
    t_min : np.ndarray, shape (N,)
        Time axis in minutes.

    Returns
    -------
    noise : np.ndarray, shape (N,)
        Ambient noise level in dB SPL, clipped to [20, 90].
    """
    rng = np.random.default_rng(profile.rng_seed + 3000)

    n = len(t_min)
    # Quiet baseline with ±1 dB white noise (sensor quantisation)
    baseline_noise = rng.normal(0.0, 1.0, size=n)
    noise = np.full(n, profile.noise_base_db) + baseline_noise

    # Poisson noise events
    arrivals = generate_poisson_events(
        rate_per_min=profile.noise_event_rate,
        duration_min=SESSION_DURATION_MIN,
        min_gap_min=SAMPLE_RATE_MIN,
        rng=rng,
    )

    for t in arrivals:
        start = int(t / SAMPLE_RATE_MIN)
        if start >= n:
            continue
        amp = rng.uniform(0.5, 1.5) * profile.noise_event_db
        # Exponential decay time constant: 2–8 minutes = 0.4–1.6 samples
        tau_min = rng.uniform(2.0, 8.0)
        tau_samples = tau_min / SAMPLE_RATE_MIN
        for i, idx in enumerate(range(start, min(start + 8, n))):
            noise[idx] += amp * np.exp(-i / tau_samples)

    return np.clip(noise, 20.0, 90.0)


# ---------------------------------------------------------------------------
# Sleep quality label derivation
# ---------------------------------------------------------------------------

def derive_sleep_labels(
    profile: SessionProfile,
    temperature: np.ndarray,
    light: np.ndarray,
    humidity: np.ndarray,
    noise_level: np.ndarray,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Derive sleep quality labels from environmental conditions.

    We use a transparent, domain-knowledge-driven scoring approach based
    on published sleep physiology literature (see module docstring refs).

    The four targets are:
      • sleep_efficiency (0–1): fraction of 8-hour window actually asleep
      • sleep_duration (h): total sleep time
      • awakenings (count): number of discrete wake episodes
      • sleep_score (0–100): composite subjective quality score

    Each environmental variable contributes a penalty or bonus according
    to its deviation from the clinically optimal range.  We then add
    session-specific Gaussian residual noise to prevent a perfectly
    deterministic (and therefore unrealistically clean) dataset.

    Parameters
    ----------
    profile : SessionProfile
        Session metadata (quality class, season, etc.).
    temperature : np.ndarray
        Temperature signal (°C).
    light : np.ndarray
        Light signal (lux).
    humidity : np.ndarray
        Humidity signal (%).
    noise_level : np.ndarray
        Noise signal (dB SPL).
    rng : np.random.Generator
        For adding residual noise.

    Returns
    -------
    dict with keys: sleep_efficiency, sleep_duration_h, awakenings, sleep_score
    """
    # --- Temperature penalty ---
    # Optimal bedroom temperature: 18–21 °C [Ref 1].
    # Penalty grows quadratically with distance from optimal range.
    mean_temp = float(temperature.mean())
    temp_optimal_low, temp_optimal_high = 18.0, 21.0
    if mean_temp < temp_optimal_low:
        temp_penalty = (temp_optimal_low - mean_temp) ** 2 * 0.05
    elif mean_temp > temp_optimal_high:
        temp_penalty = (mean_temp - temp_optimal_high) ** 2 * 0.05
    else:
        temp_penalty = 0.0

    # --- Light penalty ---
    # Any light above 10 lux during sleep is clinically disruptive [Ref 2].
    light_events_above_10lux = int(np.sum(light > 10.0))
    light_penalty = light_events_above_10lux * 0.008  # 0.8 % efficiency loss per event

    # --- Humidity penalty ---
    # Comfortable range: 40–60% RH (ASHRAE 55 standard)
    mean_humidity = float(humidity.mean())
    if mean_humidity < 40.0:
        humidity_penalty = (40.0 - mean_humidity) * 0.003
    elif mean_humidity > 60.0:
        humidity_penalty = (mean_humidity - 60.0) * 0.003
    else:
        humidity_penalty = 0.0

    # --- Noise penalty ---
    # Each noise exceedance above 45 dB can cause a micro-arousal [Ref 3].
    # We count threshold crossings (not individual samples) to reflect discrete events.
    noise_crossings = int(np.sum(
        (noise_level[:-1] < 45.0) & (noise_level[1:] >= 45.0)
    ))
    noise_penalty = noise_crossings * 0.012  # 1.2 % efficiency loss per event

    # --- Total penalty and base efficiency ---
    # Base efficiencies calibrated so the unpenalised dataset mean ≈ 0.87
    # which, after environmental penalties, converges to ~0.80–0.82, close
    # to the published Sleep Efficiency Dataset mean of 0.805 [Ref 1].
    base_efficiency = {"good": 0.94, "moderate": 0.86, "poor": 0.78}[profile.quality_class]
    total_penalty = temp_penalty + light_penalty + humidity_penalty + noise_penalty
    sleep_efficiency = float(np.clip(base_efficiency - total_penalty, 0.50, 0.99))

    # --- Sleep duration ---
    # Duration = efficiency × 8 hours, with ±20 min random residual
    sleep_duration_h = sleep_efficiency * 8.0 + rng.normal(0.0, 0.33)
    sleep_duration_h = float(np.clip(sleep_duration_h, 4.0, 8.5))

    # --- Awakenings ---
    # Base counts calibrated so the dataset mean ≈ 2.5 (published value).
    # Good/moderate/poor mean: (1 + 2 + 3) / 3 = 2.0; plus event contributions
    # and Poisson residual → target 2.3–2.7.
    base_awakenings = {"good": 1.0, "moderate": 2.0, "poor": 3.0}[profile.quality_class]
    event_contribution = noise_crossings * 0.25 + light_events_above_10lux * 0.15
    awakenings_float = base_awakenings + event_contribution + rng.exponential(0.25)
    awakenings = int(np.clip(round(awakenings_float), 0, 12))

    # --- Sleep score (0–100 composite) ---
    # Linear combination of efficiency (40%), duration (30%), inverse awakenings (30%)
    score_efficiency = sleep_efficiency * 40.0
    score_duration = min(sleep_duration_h / 8.0, 1.0) * 30.0
    score_awakenings = max(0.0, 1.0 - awakenings / 15.0) * 30.0
    sleep_score = score_efficiency + score_duration + score_awakenings
    sleep_score = float(np.clip(sleep_score + rng.normal(0.0, 3.0), 20.0, 100.0))

    return {
        "sleep_efficiency": round(sleep_efficiency, 4),
        "sleep_duration_h": round(sleep_duration_h, 3),
        "awakenings": awakenings,
        "sleep_score": round(sleep_score, 2),
    }


# ---------------------------------------------------------------------------
# Full session generator
# ---------------------------------------------------------------------------

def generate_session(
    profile: SessionProfile,
) -> Dict[str, np.ndarray | str | int | float]:
    """Generate all signals and labels for one sleep session.

    This is the atomic unit called by the dataset generator.  It returns a
    dictionary containing:
      - 'temperature', 'light', 'humidity', 'noise': np.ndarray (N,)
      - 't_min': np.ndarray (N,)  — time axis
      - 'session_id', 'season', 'quality_class': metadata
      - 'sleep_efficiency', 'sleep_duration_h', 'awakenings', 'sleep_score': labels

    Parameters
    ----------
    profile : SessionProfile
        Fully-specified parameter set for this session.

    Returns
    -------
    dict
        Complete session record ready for feature extraction.
    """
    rng = np.random.default_rng(profile.rng_seed + 9999)

    # Time axis: 0, 5, 10, …, 475 minutes
    t_min = np.arange(N_SAMPLES, dtype=float) * SAMPLE_RATE_MIN

    # Generate each signal using the session profile
    temperature = generate_temperature(profile, t_min)
    light = generate_light(profile, t_min)
    humidity = generate_humidity(profile, t_min, temperature)
    noise = generate_noise_level(profile, t_min)

    # Derive sleep quality labels from environmental conditions
    labels = derive_sleep_labels(profile, temperature, light, humidity, noise, rng)

    return {
        "session_id": profile.session_id,
        "season": profile.season,
        "quality_class": profile.quality_class,
        "t_min": t_min,
        "temperature": temperature,
        "light": light,
        "humidity": humidity,
        "noise": noise,
        **labels,
    }


# ---------------------------------------------------------------------------
# Dataset generator: produce N sessions
# ---------------------------------------------------------------------------

def generate_dataset(
    n_sessions: int = 2500,
    global_seed: int = 42,
) -> List[Dict]:
    """Generate the full synthetic dataset of ``n_sessions`` sleep sessions.

    We stratify sessions across four seasons and three quality classes.
    Each session is generated independently with a deterministic seed
    (global_seed + session_id) to guarantee reproducibility while
    maintaining inter-session independence.

    Parameters
    ----------
    n_sessions : int
        Number of sessions to generate.  Default 2500 as per the proposal.
    global_seed : int
        Master RNG seed.  We document this as 42 in REPRODUCIBILITY.md.

    Returns
    -------
    list of dict
        Each dict is one session record from ``generate_session``.
    """
    logger.info("Generating %d sessions (seed=%d) …", n_sessions, global_seed)
    sessions = []
    for i in range(n_sessions):
        profile = sample_session_profile(i, global_seed=global_seed)
        session = generate_session(profile)
        sessions.append(session)
        if (i + 1) % 500 == 0:
            logger.info("  Generated %d / %d sessions", i + 1, n_sessions)
    logger.info("Dataset generation complete.")
    return sessions
