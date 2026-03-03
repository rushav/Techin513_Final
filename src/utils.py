"""
utils.py — Shared utilities for the Synthetic Sleep Environment pipeline.

We centralise seeding, directory management, logging, and I/O helpers
here so that every other module imports one consistent interface.
Global reproducibility is guaranteed by a single top-level call to
`seed_everything`; all downstream NumPy, SciPy, and scikit-learn
operations inherit this state.

Authors: Rushav Dash, Lisa Li — TECHIN 513 Final Project
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Project-wide constants
# ---------------------------------------------------------------------------

# We chose seed 42 because it is widely recognised in the community as a
# canonical reproducibility seed; any fixed integer would work equally well.
GLOBAL_SEED: int = 42

# Sampling resolution: one observation every 5 minutes over an 8-hour window
# gives 96 time steps per session, fine enough to capture HVAC cycles (~90 min)
# and circadian drift while keeping dataset size manageable.
SAMPLE_RATE_MIN: float = 5.0           # minutes per sample
SESSION_DURATION_MIN: float = 480.0    # 8 hours in minutes
N_SAMPLES: int = int(SESSION_DURATION_MIN / SAMPLE_RATE_MIN)  # 96 samples

# Nyquist frequency in cycles per minute at 5-minute sampling
NYQUIST_CPM: float = 1.0 / (2.0 * SAMPLE_RATE_MIN)  # 0.1 cycles/min

# Directory roots — resolved relative to this file so the project is portable
_SRC_DIR = Path(__file__).parent
PROJECT_ROOT = _SRC_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
DATA_DIR = PROJECT_ROOT / "data"


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def seed_everything(seed: int = GLOBAL_SEED) -> None:
    """Seed Python, NumPy, and the OS environment for full reproducibility.

    We call this once at programme entry so that every stochastic step—
    noise generation, Random Forest bootstrap draws, train/val/test splits—
    produces identical results across runs on the same platform.

    Parameters
    ----------
    seed : int
        Integer seed value.  Default is GLOBAL_SEED (42).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger with a consistent format.

    Parameters
    ----------
    name : str
        Logger name, conventionally ``__name__`` of the calling module.
    level : int
        Logging level (default ``logging.INFO``).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def ensure_dirs() -> None:
    """Create output directories if they do not already exist.

    We separate figures/ and metrics/ so that generated images and numerical
    results are easy to locate and reference from the report.
    """
    for d in (FIGURES_DIR, METRICS_DIR, DATA_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_json(obj: Any, path: Path | str, indent: int = 2) -> None:
    """Serialise a Python object to JSON on disk.

    Parameters
    ----------
    obj : Any
        JSON-serialisable object (dict, list, etc.).
    path : Path | str
        Destination file path.
    indent : int
        Pretty-print indentation width.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=indent, default=_json_default)


def _json_default(obj: Any) -> Any:
    """Fallback serialiser for non-standard types (NumPy scalars, etc.)."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


# ---------------------------------------------------------------------------
# Timing context manager
# ---------------------------------------------------------------------------

class Timer:
    """Simple wall-clock timer for profiling pipeline stages.

    Usage
    -----
    >>> with Timer("signal generation") as t:
    ...     run_generation()
    >>> print(t.elapsed)
    """

    def __init__(self, label: str = "") -> None:
        self.label = label
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed = time.perf_counter() - self._start
        logger = get_logger(__name__)
        label = f" [{self.label}]" if self.label else ""
        logger.info("Elapsed%s: %.2f s", label, self.elapsed)
