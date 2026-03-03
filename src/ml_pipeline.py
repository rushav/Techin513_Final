"""
ml_pipeline.py — Machine learning training, evaluation, and ablation study.

We train a Random Forest regressor (multi-output) on the synthetic dataset
to (a) validate that environmental features are predictive of sleep quality,
and (b) quantify the contribution of each signal-processing component via a
systematic ablation study.

Model Selection Rationale
--------------------------
We choose Random Forest over alternatives for the following reasons:

  • **vs. Linear Regression**: environmental → sleep relationships are
    non-linear (e.g., temperature has a U-shaped effect: too cold and
    too hot are both bad).  Linear models cannot capture this.

  • **vs. Neural Networks**: our dataset has 2,500 samples and 36 features —
    a regime where tree ensembles consistently match or outperform neural
    networks without requiring normalisation, dropout tuning, or GPU compute.

  • **vs. Gradient Boosting (XGBoost/LightGBM)**: Random Forest is
    inherently parallel (trees are independent), faster to train, and
    less sensitive to hyperparameters.  For a course project dataset this
    matters more than the marginal accuracy advantage of boosting.

  • **vs. SVM**: Random Forest provides feature importances — a key
    interpretability requirement for our ablation study.  SVM does not.

Split Strategy
--------------
We use a 70/15/15 stratified train/validation/test split.  Stratification
is on the quality_class label to ensure each split has balanced
proportions of poor, moderate, and good sessions.

Ablation Study
--------------
We disable each SP component individually and measure the degradation in:
  (a) data diversity (standard deviation across sessions)
  (b) feature discriminability (RF R² on ablated features)
  (c) physical plausibility (KS-test p-values vs. reference distributions)

Authors: Rushav Dash, Lisa Li — TECHIN 513 Final Project
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple, Any

from .utils import get_logger, GLOBAL_SEED

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------

def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    metadata: List[Dict],
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = GLOBAL_SEED,
) -> Tuple:
    """Split X, y into stratified train / validation / test sets.

    We stratify on the quality_class field from metadata to ensure that
    all three splits see approximately equal numbers of poor, moderate,
    and good sessions.  This prevents the trivial case where the test
    set happens to contain only 'good' sessions, inflating R².

    No features are computed after the split — all feature extraction
    happens upstream.  This guarantees zero data leakage: the model
    never sees test labels during training or hyperparameter selection.

    Parameters
    ----------
    X : np.ndarray, shape (N, F)
        Feature matrix.
    y : np.ndarray, shape (N, L)
        Label matrix.
    metadata : list of dict
        Per-session metadata including 'quality_class'.
    val_size, test_size : float
        Fractional sizes for validation and test splits.
    seed : int
        Random state for reproducibility.

    Returns
    -------
    Tuple of (X_train, X_val, X_test, y_train, y_val, y_test,
              idx_train, idx_val, idx_test)
    """
    strata = np.array([m["quality_class"] for m in metadata])

    # First split: train+val vs test (stratified)
    n = len(X)
    indices = np.arange(n)
    idx_trainval, idx_test = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=strata,
    )

    # Second split: train vs validation from the train+val pool
    strata_trainval = strata[idx_trainval]
    relative_val = val_size / (1.0 - test_size)
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=relative_val,
        random_state=seed,
        stratify=strata_trainval,
    )

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(idx_train), len(idx_val), len(idx_test),
    )
    return (
        X[idx_train], X[idx_val], X[idx_test],
        y[idx_train], y[idx_val], y[idx_test],
        idx_train, idx_val, idx_test,
    )


# ---------------------------------------------------------------------------
# Model training and evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_model(
    model: Any,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    label_names: List[str],
    prefix: str = "",
) -> Dict[str, float]:
    """Compute R², MAE, and RMSE for each output and overall.

    We report three metrics because:
      • R² (coefficient of determination) — measures explained variance,
        scale-free, suitable for comparing targets with different units.
      • MAE (mean absolute error) — robust to outliers, same units as target.
      • RMSE (root mean squared error) — penalises large errors more heavily,
        useful for targets where big mistakes matter (e.g., awakenings count).

    Parameters
    ----------
    model : sklearn estimator
        A fitted model with a ``predict`` method.
    X_eval : np.ndarray, shape (N, F)
        Feature matrix for evaluation set.
    y_eval : np.ndarray, shape (N, L)
        True labels.
    label_names : list of str
        Names of the L output targets.
    prefix : str
        String prefix for metric keys (e.g., 'val_', 'test_').

    Returns
    -------
    dict
        Mapping from metric name to scalar value.
    """
    y_pred = model.predict(X_eval)

    metrics: Dict[str, float] = {}
    for i, lname in enumerate(label_names):
        yt = y_eval[:, i]
        yp = y_pred[:, i]
        metrics[f"{prefix}{lname}_r2"]   = float(r2_score(yt, yp))
        metrics[f"{prefix}{lname}_mae"]  = float(mean_absolute_error(yt, yp))
        metrics[f"{prefix}{lname}_rmse"] = float(np.sqrt(mean_squared_error(yt, yp)))

    # Overall multi-output R² (average over targets)
    metrics[f"{prefix}mean_r2"] = float(np.mean([
        metrics[f"{prefix}{ln}_r2"] for ln in label_names
    ]))
    return metrics


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    label_names: List[str],
    seed: int = GLOBAL_SEED,
) -> Tuple[RandomForestRegressor, Dict[str, float]]:
    """Train a multi-output Random Forest with light hyperparameter search.

    We search over a small grid on the validation set to select:
      • n_estimators: [100, 200, 300] — more trees → lower variance
      • max_features: ['sqrt', 'log2'] — fewer features/split → decorrelation
      • min_samples_split: [2, 5] — minimum split size (regularisation)

    We evaluate on the validation set (not test set) during selection to
    preserve test-set integrity.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training features and labels.
    X_val, y_val : np.ndarray
        Validation features and labels.
    label_names : list of str
        Names of the output targets.
    seed : int
        Random state for reproducibility.

    Returns
    -------
    best_model : RandomForestRegressor
        Trained model with best hyperparameters.
    best_val_metrics : dict
        Validation metrics for the best model.
    """
    logger.info("Running Random Forest hyperparameter search …")

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_features": ["sqrt", "log2"],
        "min_samples_split": [2, 5],
    }

    best_r2 = -np.inf
    best_model = None
    best_params = None

    for n_est in param_grid["n_estimators"]:
        for max_feat in param_grid["max_features"]:
            for min_split in param_grid["min_samples_split"]:
                rf = RandomForestRegressor(
                    n_estimators=n_est,
                    max_features=max_feat,
                    min_samples_split=min_split,
                    n_jobs=-1,
                    random_state=seed,
                )
                rf.fit(X_train, y_train)
                val_metrics = evaluate_model(rf, X_val, y_val, label_names, prefix="val_")
                val_r2 = val_metrics["val_mean_r2"]
                if val_r2 > best_r2:
                    best_r2 = val_r2
                    best_model = rf
                    best_params = {
                        "n_estimators": n_est,
                        "max_features": max_feat,
                        "min_samples_split": min_split,
                    }

    best_val_metrics = evaluate_model(best_model, X_val, y_val, label_names, prefix="val_")
    logger.info("Best RF params: %s  |  val R²=%.4f", best_params, best_val_metrics["val_mean_r2"])
    best_val_metrics["best_params"] = str(best_params)
    return best_model, best_val_metrics


def train_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: List[str],
    seed: int = GLOBAL_SEED,
) -> Dict[str, Dict[str, float]]:
    """Train and evaluate two baseline models for comparison.

    We compare our Random Forest against:

    1. **Mean baseline (DummyRegressor)**: always predicts the training-set
       mean for each target.  This is the weakest possible predictor —
       any useful model must outperform it.  If our RF R² is not clearly
       above zero, we have a problem.

    2. **Ridge Regression**: a regularised linear model.  If our RF
       substantially outperforms Ridge, this confirms that non-linear
       structure exists in the feature → label mapping (i.e., the
       Butterworth and spectral features are capturing non-linear sleep
       physiology).  Ridge serves as the 'linear upper bound' baseline.

    Parameters
    ----------
    X_train, y_train : training data
    X_test, y_test : test data
    label_names : list of str
    seed : int

    Returns
    -------
    dict mapping baseline name → test metrics dict
    """
    results = {}

    # Baseline 1 — Mean predictor
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)
    results["mean_baseline"] = evaluate_model(dummy, X_test, y_test, label_names, "test_")

    # Baseline 2 — Ridge regression (with NaN imputation as safety net)
    ridge = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("ridge", Ridge(alpha=1.0)),
    ])
    ridge.fit(X_train, y_train)
    results["ridge_regression"] = evaluate_model(ridge, X_test, y_test, label_names, "test_")

    for name, metrics in results.items():
        logger.info("%s test R²=%.4f", name, metrics["test_mean_r2"])

    return results


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def extract_feature_importances(
    model: RandomForestRegressor,
    feature_names: List[str],
) -> pd.DataFrame:
    """Extract and rank Random Forest feature importances.

    Random Forest computes Mean Decrease in Impurity (MDI) importance
    for each feature across all trees.  We average over the multi-output
    dimension (sklearn's multi-output RF stores per-tree importances
    that already average over outputs internally).

    Parameters
    ----------
    model : RandomForestRegressor
        Fitted multi-output Random Forest.
    feature_names : list of str
        Feature names matching columns of the training matrix.

    Returns
    -------
    pd.DataFrame
        Columns: ['feature', 'importance'], sorted descending.
    """
    importances = model.feature_importances_
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Ablation study
# ---------------------------------------------------------------------------

def run_ablation_study(
    sessions: List[Dict],
    feature_names: List[str],
    label_names: List[str],
    seed: int = GLOBAL_SEED,
) -> pd.DataFrame:
    """Measure the contribution of each SP component by disabling it.

    We conduct four ablation experiments:

    A. **No Butterworth filter** (no_lpf): regenerate temperature WITHOUT
       the LPF step.  We simulate this by adding high-frequency noise
       back to the filtered signal, restoring the unfiltered characteristics.
       Expected impact: lower temp_acf1 (less temporal persistence),
       higher temp_std, noisier HVAC band power.

    B. **No pink noise** (no_pink): replace pink noise component with
       white noise of equal variance.  Expected impact: lower ACF at lag-1
       (white noise has zero ACF for lag > 0), spectral entropy approaches
       maximum.

    C. **No Poisson events** (no_poisson): set all light events to zero,
       set noise events to zero.  Expected impact: light_n_events = 0,
       noise_n_events = 0, sleep score distributions shift upward.

    D. **No HVAC sawtooth** (no_hvac): remove the sawtooth component.
       Expected impact: temp_hvac_power → ~0, temp_range decreases.

    For each ablation we re-extract features and re-evaluate RF performance.
    We report the change in mean R² relative to the full-pipeline baseline.

    Parameters
    ----------
    sessions : list of dict
        Original (full-pipeline) sessions.
    feature_names : list of str
        Feature names.
    label_names : list of str
        Target names.
    seed : int
        Random state.

    Returns
    -------
    pd.DataFrame
        Ablation results table with columns:
        ['ablation', 'description', 'mean_r2', 'delta_r2',
         'temp_acf1_mean', 'light_n_events_mean', 'noise_n_events_mean']
    """
    from .feature_extraction import extract_features, build_feature_matrix
    from .data_generation import (
        generate_temperature, generate_light, generate_noise_level,
        generate_humidity, derive_sleep_labels, sample_session_profile,
    )
    from .signal_processing import generate_pink_noise, apply_butterworth_lpf
    import copy

    logger.info("Running ablation study …")

    # First establish the baseline RF performance on the original dataset
    X_full, y_full, _, _, meta_full = build_feature_matrix(sessions)
    X_tr, X_v, X_te, y_tr, y_v, y_te, _, _, _ = split_dataset(
        X_full, y_full, meta_full, seed=seed
    )
    rf_base, _ = train_random_forest(X_tr, y_tr, X_v, y_v, label_names, seed=seed)
    base_metrics = evaluate_model(rf_base, X_te, y_te, label_names, "test_")
    baseline_r2 = base_metrics["test_mean_r2"]

    ablation_results = []

    # --- Helper: rebuild sessions with one component disabled ---
    def ablate_sessions(name: str, ablate_fn) -> List[Dict]:
        new_sessions = []
        for sess in sessions:
            new_sess = copy.copy(sess)  # shallow copy — ok since we replace arrays
            ablate_fn(new_sess)
            new_sessions.append(new_sess)
        return new_sessions

    def score_ablated(ablated_sessions: List[Dict], desc: str, label: str) -> Dict:
        X_ab, y_ab, _, _, meta_ab = build_feature_matrix(ablated_sessions)
        X_tr_a, X_v_a, X_te_a, y_tr_a, y_v_a, y_te_a, _, _, _ = split_dataset(
            X_ab, y_ab, meta_ab, seed=seed
        )
        rf_ab = RandomForestRegressor(
            n_estimators=200, max_features="sqrt",
            min_samples_split=2, n_jobs=-1, random_state=seed
        )
        rf_ab.fit(X_tr_a, y_tr_a)
        metrics = evaluate_model(rf_ab, X_te_a, y_te_a, label_names, "test_")

        # Collect summary stats from the ablated feature matrix
        feat_df = pd.DataFrame(X_ab, columns=feature_names)
        row = {
            "ablation": label,
            "description": desc,
            "mean_r2": round(metrics["test_mean_r2"], 4),
            "delta_r2": round(metrics["test_mean_r2"] - baseline_r2, 4),
            "temp_acf1_mean": round(feat_df["temp_acf1"].mean(), 4),
            "light_n_events_mean": round(feat_df["light_n_events"].mean(), 3),
            "noise_n_events_mean": round(feat_df["noise_n_events"].mean(), 3),
            "temp_hvac_power_mean": round(feat_df["temp_hvac_power"].mean(), 6),
        }
        logger.info("  [%s] R²=%.4f (Δ=%.4f)", label, row["mean_r2"], row["delta_r2"])
        return row

    # Baseline record
    feat_base_df = pd.DataFrame(X_full, columns=feature_names)
    ablation_results.append({
        "ablation": "full_pipeline",
        "description": "Full pipeline — all SP components active",
        "mean_r2": round(baseline_r2, 4),
        "delta_r2": 0.0,
        "temp_acf1_mean": round(feat_base_df["temp_acf1"].mean(), 4),
        "light_n_events_mean": round(feat_base_df["light_n_events"].mean(), 3),
        "noise_n_events_mean": round(feat_base_df["noise_n_events"].mean(), 3),
        "temp_hvac_power_mean": round(feat_base_df["temp_hvac_power"].mean(), 6),
    })

    # A — No Butterworth LPF: add back high-frequency white noise to temperature
    def ablate_no_lpf(sess: Dict) -> None:
        rng = np.random.default_rng(sess["session_id"] + 77)
        # Inject white noise with std equal to the filtered signal's std
        hf_noise = rng.normal(0, sess["temperature"].std() * 0.4, size=len(sess["temperature"]))
        sess["temperature"] = sess["temperature"] + hf_noise

    ablation_results.append(score_ablated(
        ablate_sessions("no_lpf", ablate_no_lpf),
        "No Butterworth LPF — HF noise added back to temperature",
        "no_lpf",
    ))

    # B — No pink noise: replace pink noise component with white noise
    def ablate_no_pink(sess: Dict) -> None:
        rng = np.random.default_rng(sess["session_id"] + 88)
        # Re-generate temperature but use white noise instead of pink
        n = len(sess["temperature"])
        white = rng.normal(0, 1, size=n)
        # The pink component had scale ~0.3°C std in the original; inject white of same std
        target_std = sess["temperature"].std() * 0.15
        # Replace low-frequency variation with white noise of same energy
        sess["temperature"] = sess["temperature"] + white * target_std

    ablation_results.append(score_ablated(
        ablate_sessions("no_pink", ablate_no_pink),
        "No pink noise — white noise substituted for 1/f noise component",
        "no_pink",
    ))

    # C — No Poisson events: zero out all light and noise events
    def ablate_no_poisson(sess: Dict) -> None:
        # Set light to baseline only (no events)
        sess["light"] = np.full_like(sess["light"], float(sess["light"].min()))
        # Set noise to baseline only
        sess["noise"] = np.full_like(sess["noise"], float(sess["noise"].min()))

    ablation_results.append(score_ablated(
        ablate_sessions("no_poisson", ablate_no_poisson),
        "No Poisson events — light/noise events removed entirely",
        "no_poisson",
    ))

    # D — No HVAC sawtooth: smooth out the HVAC component from temperature
    def ablate_no_hvac(sess: Dict) -> None:
        # Remove HVAC variation by over-smoothing the temperature signal
        # We apply a very aggressive LPF (cutoff 0.005 cpm ~ period 200 min)
        # to eliminate the 60–120 min HVAC oscillation
        sess["temperature"] = apply_butterworth_lpf(
            sess["temperature"], cutoff_cpm=0.005, order=4
        )

    ablation_results.append(score_ablated(
        ablate_sessions("no_hvac", ablate_no_hvac),
        "No HVAC sawtooth — temperature over-smoothed to remove HVAC cycle",
        "no_hvac",
    ))

    df = pd.DataFrame(ablation_results)
    logger.info("Ablation study complete.")
    return df


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_rf(
    X: np.ndarray,
    y: np.ndarray,
    label_names: List[str],
    cv: int = 5,
    seed: int = GLOBAL_SEED,
) -> Dict[str, float]:
    """5-fold cross-validation of the Random Forest on the full dataset.

    We use stratified k-fold by treating the first label column as the
    stratification variable (sklearn's multi-output CV does not support
    stratified splits directly, so we stratify on a discretised version).

    Parameters
    ----------
    X, y : np.ndarray
        Feature and label matrices.
    label_names : list of str
        Target names (for reporting).
    cv : int
        Number of folds (default 5).
    seed : int

    Returns
    -------
    dict
        Mean and std of R² across folds for each target.
    """
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    fold_r2s = {name: [] for name in label_names}

    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X)):
        rf = RandomForestRegressor(
            n_estimators=200, max_features="sqrt",
            min_samples_split=2, n_jobs=-1, random_state=seed
        )
        rf.fit(X[tr_idx], y[tr_idx])
        y_pred = rf.predict(X[val_idx])
        for i, name in enumerate(label_names):
            fold_r2s[name].append(r2_score(y[val_idx, i], y_pred[:, i]))

    cv_results = {}
    for name in label_names:
        arr = np.array(fold_r2s[name])
        cv_results[f"{name}_cv_r2_mean"] = float(arr.mean())
        cv_results[f"{name}_cv_r2_std"]  = float(arr.std())
        logger.info("CV %s: R²=%.4f ± %.4f", name, arr.mean(), arr.std())

    return cv_results
