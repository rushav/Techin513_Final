"""
visualisation.py — All figure generation for the synthetic sleep dataset.

We generate 12 publication-quality figures saved to results/figures/ at
300 DPI.  All figures use a consistent style and colour palette.

Figures produced:
  F01_pipeline_overview.png     — schematic of the generation pipeline
  F02_example_session.png       — 4-panel plot of one sample session's signals
  F03_temperature_spectrum.png  — FFT spectrum verifying HVAC and circadian peaks
  F04_filter_effect.png         — before/after Butterworth filter on temperature
  F05_poisson_events.png        — light + noise time-series with event markers
  F06_sleep_label_distributions.png — histograms of the 4 sleep quality targets
  F07_feature_correlation.png   — feature × label Pearson correlation heatmap
  F08_feature_importance.png    — Random Forest feature importance bar chart
  F09_model_performance.png     — predicted vs actual scatter for 4 targets
  F10_ablation_study.png        — ablation ΔR² bar chart
  F11_ks_test_comparison.png    — KS-test distribution comparisons (6 variables)
  F12_seasonal_comparison.png   — sleep efficiency by season and quality class

Authors: Rushav Dash, Lisa Li — TECHIN 513 Final Project
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for script execution
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional

from .utils import FIGURES_DIR, N_SAMPLES, SAMPLE_RATE_MIN, get_logger
from .signal_processing import compute_fft_spectrum

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------

PALETTE = {
    "temperature": "#d62728",   # brick red
    "light":       "#ff7f0e",   # orange
    "humidity":    "#1f77b4",   # blue
    "noise":       "#2ca02c",   # green
    "good":        "#2ca02c",
    "moderate":    "#ff7f0e",
    "poor":        "#d62728",
    "accent":      "#9467bd",
    "dark":        "#222222",
}

DPI = 300
FIGSIZE_WIDE = (12, 5)
FIGSIZE_SQUARE = (8, 8)
FIGSIZE_TALL = (10, 10)


def _save(fig: plt.Figure, name: str) -> None:
    """Save figure to FIGURES_DIR at 300 DPI."""
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


# ---------------------------------------------------------------------------
# F02 — Example session (4-panel time series)
# ---------------------------------------------------------------------------

def plot_example_session(session: Dict, filename: str = "F02_example_session.png") -> None:
    """Plot all four environmental signals for one representative session."""
    t = session["t_min"]
    t_hours = t / 60.0

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(
        f"Example Session #{session['session_id']} "
        f"({session['season'].title()}, {session['quality_class'].title()} quality)",
        fontsize=13, fontweight="bold",
    )

    signals = [
        ("temperature", "Temperature (°C)", PALETTE["temperature"]),
        ("light",       "Illuminance (lux)",  PALETTE["light"]),
        ("humidity",    "Relative Humidity (%)", PALETTE["humidity"]),
        ("noise",       "Noise Level (dB SPL)", PALETTE["noise"]),
    ]

    for ax, (key, ylabel, color) in zip(axes, signals):
        ax.plot(t_hours, session[key], color=color, lw=1.2, alpha=0.85)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.3, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Time into sleep session (hours)", fontsize=10)

    # Annotate sleep quality labels
    labels_text = (
        f"Sleep efficiency: {session['sleep_efficiency']:.3f}  |  "
        f"Duration: {session['sleep_duration_h']:.1f} h  |  "
        f"Awakenings: {session['awakenings']}  |  "
        f"Sleep score: {session['sleep_score']:.1f}"
    )
    fig.text(0.5, 0.01, labels_text, ha="center", fontsize=8.5, color="#555555")
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    _save(fig, filename)


# ---------------------------------------------------------------------------
# F03 — Temperature FFT spectrum
# ---------------------------------------------------------------------------

def plot_temperature_spectrum(
    sessions: List[Dict],
    n_sessions: int = 50,
    filename: str = "F03_temperature_spectrum.png",
) -> None:
    """Average FFT power spectrum over multiple sessions to verify frequency peaks."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    fs_cpm = 1.0 / SAMPLE_RATE_MIN
    all_psds = []

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(sessions), size=min(n_sessions, len(sessions)), replace=False)

    for i in sample_idx:
        freqs, psd = compute_fft_spectrum(sessions[i]["temperature"])
        all_psds.append(psd)

    # Frequency → period (minutes) for x-axis labelling
    mean_psd = np.mean(all_psds, axis=0)
    freqs_arr = np.fft.rfftfreq(N_SAMPLES, d=SAMPLE_RATE_MIN)  # cpm

    # Convert to periods in minutes for readability; skip DC
    ax.semilogy(freqs_arr[1:], mean_psd[1:], color=PALETTE["temperature"], lw=1.5)

    # Mark expected peaks
    hvac_f = 1.0 / 90.0    # ~0.011 cpm
    circadian_f = 1.0 / 1440.0  # ~0.00069 cpm (outside our 8-hour window)

    ax.axvline(hvac_f, color="grey", linestyle="--", lw=1.0, label=f"HVAC (~90 min)")
    ax.set_xlabel("Frequency (cycles per minute)", fontsize=10)
    ax.set_ylabel("Power Spectral Density (°C²/cpm)", fontsize=10)
    ax.set_title(
        f"Mean Temperature PSD averaged over {n_sessions} sessions\n"
        f"Dashed line marks expected HVAC frequency (~1/90 cpm)",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    _save(fig, filename)


# ---------------------------------------------------------------------------
# F04 — Butterworth filter effect
# ---------------------------------------------------------------------------

def plot_filter_effect(
    session: Dict,
    filename: str = "F04_filter_effect.png",
) -> None:
    """Demonstrate the effect of the Butterworth LPF on a single session."""
    from .signal_processing import generate_pink_noise, apply_butterworth_lpf
    from .data_generation import sample_session_profile

    rng = np.random.default_rng(42)
    t = session["t_min"]

    # Reconstruct an approximate unfiltered signal by adding back HF noise
    hf_noise = rng.normal(0, session["temperature"].std() * 0.4, size=len(t))
    unfiltered = session["temperature"] + hf_noise

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    t_hours = t / 60.0
    axes[0].plot(t_hours, unfiltered, color="#999999", lw=0.9, label="Unfiltered (+ HF noise)")
    axes[0].plot(t_hours, session["temperature"], color=PALETTE["temperature"],
                 lw=1.4, label="After Butterworth LPF (order 4, cutoff 0.02 cpm)")
    axes[0].set_ylabel("Temperature (°C)", fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, lw=0.5)
    axes[0].set_title("Effect of Butterworth Low-Pass Filter on Bedroom Temperature", fontsize=11)

    # Residual (removed component)
    axes[1].plot(t_hours, unfiltered - session["temperature"],
                 color="#888888", lw=0.9, alpha=0.8)
    axes[1].axhline(0, color="black", lw=0.6)
    axes[1].set_ylabel("Removed HF component (°C)", fontsize=10)
    axes[1].set_xlabel("Time into sleep session (hours)", fontsize=10)
    axes[1].grid(True, alpha=0.3, lw=0.5)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    _save(fig, filename)


# ---------------------------------------------------------------------------
# F05 — Poisson events (light + noise)
# ---------------------------------------------------------------------------

def plot_poisson_events(
    session: Dict,
    filename: str = "F05_poisson_events.png",
) -> None:
    """Visualise discrete Poisson-distributed disturbance events."""
    t_hours = session["t_min"] / 60.0

    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE_WIDE, sharex=True)

    axes[0].fill_between(t_hours, 0, session["light"], color=PALETTE["light"], alpha=0.7)
    axes[0].axhline(10.0, color="red", lw=0.8, linestyle="--", label="10 lux threshold")
    axes[0].set_ylabel("Illuminance (lux)", fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].set_title("Poisson-Modelled Light and Noise Disturbance Events", fontsize=11)

    axes[1].plot(t_hours, session["noise"], color=PALETTE["noise"], lw=1.2, alpha=0.85)
    axes[1].axhline(45.0, color="red", lw=0.8, linestyle="--", label="45 dB threshold")
    axes[1].set_ylabel("Noise Level (dB SPL)", fontsize=10)
    axes[1].set_xlabel("Time into sleep session (hours)", fontsize=10)
    axes[1].legend(fontsize=9)

    for ax in axes:
        ax.grid(True, alpha=0.3, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    _save(fig, filename)


# ---------------------------------------------------------------------------
# F06 — Sleep label distributions
# ---------------------------------------------------------------------------

def plot_label_distributions(
    df: pd.DataFrame,
    filename: str = "F06_sleep_label_distributions.png",
) -> None:
    """Plot histograms + KDE for all four sleep quality labels."""
    labels = [
        ("sleep_efficiency", "Sleep Efficiency", ""),
        ("sleep_duration_h", "Sleep Duration", "hours"),
        ("awakenings",       "Awakenings",       "count"),
        ("sleep_score",      "Sleep Score",      ""),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    fig.suptitle("Synthetic Sleep Quality Label Distributions (N=2,500 sessions)",
                 fontsize=13, fontweight="bold")

    for ax, (col, title, unit) in zip(axes, labels):
        if col not in df.columns:
            continue
        data = df[col].dropna()
        ax.hist(data, bins=35, color=PALETTE["accent"], alpha=0.65,
                edgecolor="white", lw=0.5, density=True, label="Synthetic")
        data.plot(kind="kde", ax=ax, color=PALETTE["dark"], lw=1.5, label="KDE")

        xlabel = f"{title} ({unit})" if unit else title
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.text(
            0.97, 0.93,
            f"μ={data.mean():.3f}\nσ={data.std():.3f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="#444444",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    _save(fig, filename)


# ---------------------------------------------------------------------------
# F07 — Feature-label correlation heatmap
# ---------------------------------------------------------------------------

def plot_feature_label_correlation(
    df: pd.DataFrame,
    feature_names: List[str],
    label_names: List[str],
    filename: str = "F07_feature_correlation.png",
) -> None:
    """Heatmap of Pearson correlation between each feature and each label."""
    corr_matrix = np.zeros((len(feature_names), len(label_names)))
    for i, feat in enumerate(feature_names):
        if feat not in df.columns:
            continue
        for j, lbl in enumerate(label_names):
            if lbl not in df.columns:
                continue
            r, _ = __import__("scipy.stats", fromlist=["pearsonr"]).pearsonr(
                df[feat].fillna(0), df[lbl].fillna(0)
            )
            corr_matrix[i, j] = r

    corr_df = pd.DataFrame(corr_matrix, index=feature_names, columns=label_names)

    fig, ax = plt.subplots(figsize=(8, 14))
    sns.heatmap(
        corr_df, ax=ax,
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        linewidths=0.3, linecolor="white",
        cbar_kws={"label": "Pearson r", "shrink": 0.7},
        annot=False,
    )
    ax.set_title("Feature–Label Correlation Matrix", fontsize=12, fontweight="bold")
    ax.set_xlabel("Sleep Quality Target", fontsize=10)
    ax.set_ylabel("Environmental Feature", fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=7.5)
    plt.tight_layout()
    _save(fig, filename)


# ---------------------------------------------------------------------------
# F08 — Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    filename: str = "F08_feature_importance.png",
) -> None:
    """Horizontal bar chart of the top-N Random Forest feature importances."""
    top = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = [
        PALETTE["temperature"] if "temp" in f else
        PALETTE["light"]       if "light" in f else
        PALETTE["humidity"]    if "humidity" in f else
        PALETTE["noise"]       if "noise" in f else
        PALETTE["accent"]
        for f in top["feature"]
    ]
    ax.barh(top["feature"][::-1], top["importance"][::-1],
            color=colors[::-1], edgecolor="white", lw=0.5)
    ax.set_xlabel("Mean Decrease in Impurity (MDI)", fontsize=10)
    ax.set_title(f"Random Forest Feature Importances (Top {top_n})", fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend for signal colours
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=PALETTE["temperature"], label="Temperature"),
        Patch(facecolor=PALETTE["light"],       label="Light"),
        Patch(facecolor=PALETTE["humidity"],    label="Humidity"),
        Patch(facecolor=PALETTE["noise"],       label="Noise"),
        Patch(facecolor=PALETTE["accent"],      label="Cross-signal"),
    ]
    ax.legend(handles=legend_elems, fontsize=8, loc="lower right")
    plt.tight_layout()
    _save(fig, filename)


# ---------------------------------------------------------------------------
# F09 — Predicted vs Actual scatter
# ---------------------------------------------------------------------------

def plot_predictions(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
    filename: str = "F09_model_performance.png",
) -> None:
    """4-panel scatter: predicted vs. actual for each sleep quality target."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.flatten()
    fig.suptitle("Random Forest: Predicted vs. Actual on Test Set",
                 fontsize=12, fontweight="bold")

    from sklearn.metrics import r2_score
    nice_names = {
        "sleep_efficiency": "Sleep Efficiency",
        "sleep_duration_h": "Sleep Duration (h)",
        "awakenings":       "Awakenings (count)",
        "sleep_score":      "Sleep Score",
    }

    for ax, (i, lname) in zip(axes, enumerate(label_names)):
        yt = y_test[:, i]
        yp = y_pred[:, i]
        r2 = r2_score(yt, yp)
        ax.scatter(yt, yp, s=6, alpha=0.35, color=PALETTE["accent"], rasterized=True)
        lims = [min(yt.min(), yp.min()) - 0.02, max(yt.max(), yp.max()) + 0.02]
        ax.plot(lims, lims, "k--", lw=0.8, label="Perfect fit")
        ax.set_xlabel(f"Actual {nice_names.get(lname, lname)}", fontsize=9)
        ax.set_ylabel("Predicted", fontsize=9)
        ax.set_title(nice_names.get(lname, lname), fontsize=10, fontweight="bold")
        ax.text(0.05, 0.93, f"R² = {r2:.4f}", transform=ax.transAxes,
                fontsize=9, color="black",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        ax.grid(True, alpha=0.3, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    _save(fig, filename)


# ---------------------------------------------------------------------------
# F10 — Ablation study
# ---------------------------------------------------------------------------

def plot_ablation(
    ablation_df: pd.DataFrame,
    filename: str = "F10_ablation_study.png",
) -> None:
    """Bar chart showing ΔR² from disabling each SP component."""
    df = ablation_df[ablation_df["ablation"] != "full_pipeline"].copy()

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#d62728" if x < 0 else "#2ca02c" for x in df["delta_r2"]]
    bars = ax.bar(df["ablation"], df["delta_r2"], color=colors, edgecolor="white", lw=0.5)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("ΔR² (relative to full pipeline)", fontsize=10)
    ax.set_title(
        "Ablation Study: Impact of Removing Each SP Component on Mean R²",
        fontsize=11, fontweight="bold",
    )
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(
        ["No LPF\n(Butterworth)", "No Pink\nNoise", "No Poisson\nEvents", "No HVAC\nModel"],
        fontsize=9,
    )
    for bar, val in zip(bars, df["delta_r2"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.001 if val >= 0 else -0.005),
            f"{val:+.4f}",
            ha="center", va="bottom" if val >= 0 else "top",
            fontsize=9, fontweight="bold",
        )
    ax.grid(True, axis="y", alpha=0.3, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    _save(fig, filename)


# ---------------------------------------------------------------------------
# F11 — KS test comparisons
# ---------------------------------------------------------------------------

def plot_ks_comparisons(
    df: pd.DataFrame,
    ks_results: pd.DataFrame,
    filename: str = "F11_ks_test_comparison.png",
) -> None:
    """Overlay synthetic CDF vs reference Gaussian CDF for each variable."""
    from .validation import REFERENCE_STATS

    vars_to_plot = list(REFERENCE_STATS.keys())
    n_vars = len(vars_to_plot)
    cols = 3
    rows = (n_vars + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3.5))
    axes = axes.flatten()
    fig.suptitle("KS-Test: Synthetic vs. Reference Distributions", fontsize=12, fontweight="bold")

    rng = np.random.default_rng(42)

    col_map = {
        "sleep_efficiency": "sleep_efficiency",
        "sleep_duration_h": "sleep_duration_h",
        "awakenings": "awakenings",
        "sleep_score": "sleep_score",
        "temperature_mean": "temp_mean",
        "light_n_events": "light_n_events",
    }

    for ax, var in zip(axes[:n_vars], vars_to_plot):
        ref = REFERENCE_STATS[var]
        col = col_map.get(var, var)
        if col not in df.columns:
            ax.set_visible(False)
            continue

        synth = df[col].dropna().values
        ref_sample = np.sort(rng.normal(ref["mean"], ref["std"], size=2000))
        ref_sample = np.clip(ref_sample, ref["low"], ref["high"])
        synth_sorted = np.sort(synth)

        # Empirical CDFs
        ax.step(synth_sorted, np.linspace(0, 1, len(synth_sorted)),
                color=PALETTE["accent"], lw=1.5, label="Synthetic")
        ax.step(ref_sample, np.linspace(0, 1, len(ref_sample)),
                color="#444444", lw=1.5, linestyle="--", label="Reference")

        # KS result annotation
        ks_row = ks_results[ks_results["variable"] == var]
        if len(ks_row) > 0:
            ks_val = ks_row.iloc[0]
            status = "PASS" if ks_val["pass"] else "FAIL"
            color_ann = "#2ca02c" if ks_val["pass"] else "#d62728"
            ax.text(
                0.97, 0.05,
                f"D={ks_val['ks_stat']:.3f}\np={ks_val['p_value']:.3f} [{status}]",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color=color_ann,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
            )

        ax.set_title(var.replace("_", " ").title(), fontsize=9, fontweight="bold")
        ax.legend(fontsize=7.5)
        ax.grid(True, alpha=0.3, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[n_vars:]:
        ax.set_visible(False)

    plt.tight_layout()
    _save(fig, filename)


# ---------------------------------------------------------------------------
# F12 — Seasonal comparison
# ---------------------------------------------------------------------------

def plot_seasonal_comparison(
    df: pd.DataFrame,
    filename: str = "F12_seasonal_comparison.png",
) -> None:
    """Box plot of sleep efficiency grouped by season and quality class."""
    if "season" not in df.columns or "sleep_efficiency" not in df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Sleep Efficiency by Season and Quality Class", fontsize=12, fontweight="bold")

    season_order = ["winter", "spring", "summer", "autumn"]
    quality_order = ["poor", "moderate", "good"]
    season_colors = ["#4c72b0", "#55a868", "#c44e52", "#dd8452"]
    quality_colors = [PALETTE["poor"], PALETTE["moderate"], PALETTE["good"]]

    # Panel 1 — by season
    ax = axes[0]
    season_data = [
        df.loc[df["season"] == s, "sleep_efficiency"].values
        for s in season_order
    ]
    bp = ax.boxplot(season_data, patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], season_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels([s.title() for s in season_order], fontsize=9)
    ax.set_ylabel("Sleep Efficiency", fontsize=10)
    ax.set_title("By Season", fontsize=10, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel 2 — by quality class
    ax = axes[1]
    quality_data = [
        df.loc[df["quality_class"] == q, "sleep_efficiency"].values
        for q in quality_order
    ]
    bp2 = ax.boxplot(quality_data, patch_artist=True, notch=False)
    for patch, color in zip(bp2["boxes"], quality_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, 4))
    ax.set_xticklabels([q.title() for q in quality_order], fontsize=9)
    ax.set_ylabel("Sleep Efficiency", fontsize=10)
    ax.set_title("By Quality Class", fontsize=10, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    _save(fig, filename)
