#!/usr/bin/env python3
"""
export_data.py — Pre-compute all dashboard data as static JSON files.

Run from the project root (with venv active):
  python web/scripts/export_data.py

Outputs go to: web/public/data/
"""

import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_DIR = PROJECT_ROOT / "web" / "public" / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from src.utils import seed_everything, N_SAMPLES, SAMPLE_RATE_MIN, SESSION_DURATION_MIN
from src.data_generation import (generate_dataset, sample_session_profile,
                                  generate_temperature, generate_light,
                                  generate_humidity, generate_noise_level)
from src.feature_extraction import build_feature_matrix, FEATURE_NAMES, FEATURE_CATALOGUE
from src.signal_processing import (compute_fft_spectrum, apply_butterworth_lpf,
                                    generate_pink_noise, generate_poisson_events)
from src.ml_pipeline import (split_dataset, train_random_forest, train_baselines,
                               extract_feature_importances, run_ablation_study,
                               evaluate_model, cross_validate_rf)
from src.validation import run_ks_tests, REFERENCE_STATS

seed_everything(42)

def j(obj):
    """Round-trip safe JSON encoder for numpy types."""
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    if isinstance(obj, bool):           return bool(obj)
    raise TypeError(f"Not serialisable: {type(obj)}")

def save(name, data):
    path = OUTPUT_DIR / name
    with open(path, "w") as f:
        json.dump(data, f, default=j, separators=(",", ":"))
    kb = path.stat().st_size / 1024
    print(f"  ✓ {name:40s}  {kb:6.1f} KB")

def r2(v, decimals=4):
    """Round array or scalar."""
    if isinstance(v, (list, np.ndarray)):
        return [round(float(x), decimals) for x in v]
    return round(float(v), decimals)

# ══════════════════════════════════════════════════════════════════════════════
print("▶ Generating dataset (500 sessions, seed=42)...")
sessions = generate_dataset(n_sessions=500, global_seed=42)

print("▶ Extracting features...")
X, y, feature_names, label_names, metadata = build_feature_matrix(sessions)
df = pd.DataFrame(X, columns=feature_names)
for i, lbl in enumerate(label_names):
    df[lbl] = y[:, i]
for k in ("session_id", "season", "quality_class"):
    df[k] = [m[k] for m in metadata]

print("▶ Splitting dataset...")
X_train, X_val, X_test, y_train, y_val, y_test, idx_tr, idx_v, idx_te = \
    split_dataset(X, y, metadata, seed=42)

print("▶ Training Random Forest...")
rf_model, val_metrics = train_random_forest(X_train, y_train, X_val, y_val,
                                             label_names, seed=42)
test_metrics = evaluate_model(rf_model, X_test, y_test, label_names, "test_")
y_pred_test  = rf_model.predict(X_test)

print("▶ Training baselines...")
baseline_metrics = train_baselines(X_train, y_train, X_test, y_test, label_names, seed=42)

print("▶ Cross-validation...")
cv_metrics = cross_validate_rf(X, y, label_names, cv=5, seed=42)

print("▶ Feature importances...")
importance_df = extract_feature_importances(rf_model, feature_names)

print("▶ KS tests...")
ks_results = run_ks_tests(df, seed=42)

print("▶ Training RF classifier (for confusion matrix / ROC)...")
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                              f1_score, roc_curve, auc, precision_recall_curve,
                              average_precision_score)
from sklearn.preprocessing import label_binarize

quality_enc  = {"good": 0, "moderate": 1, "poor": 2}
quality_all  = np.array([quality_enc[m["quality_class"]] for m in metadata])
quality_train = quality_all[idx_tr]
quality_test  = quality_all[idx_te]

clf = RandomForestClassifier(n_estimators=200, max_features="sqrt",
                              min_samples_split=2, n_jobs=-1, random_state=42)
clf.fit(X_train, quality_train)
quality_pred = clf.predict(X_test)
quality_prob = clf.predict_proba(X_test)
clf_accuracy = float((quality_pred == quality_test).mean())

print("▶ Ablation study...")
ablation_df = run_ablation_study(sessions, feature_names, label_names, seed=42)

print("\n▶ Writing JSON files to", OUTPUT_DIR)

# ══════════════════════════════════════════════════════════════════════════════
# 1. dataset_stats.json
# ══════════════════════════════════════════════════════════════════════════════
quality_counts = df["quality_class"].value_counts().to_dict()
season_counts  = df["season"].value_counts().to_dict()

save("dataset_stats.json", {
    "n_sessions":    500,
    "n_features":    len(feature_names),
    "n_labels":      len(label_names),
    "seed":          42,
    "sample_rate_min": SAMPLE_RATE_MIN,
    "n_samples_per_session": N_SAMPLES,
    "session_duration_h": 8,
    "quality_distribution": quality_counts,
    "season_distribution":  season_counts,
    "label_stats": {
        lbl: {
            "mean": r2(df[lbl].mean()),
            "std":  r2(df[lbl].std()),
            "min":  r2(df[lbl].min()),
            "max":  r2(df[lbl].max()),
        } for lbl in label_names
    },
    "signal_info": [
        {"name": "temperature", "unit": "°C",     "description": "Bedroom temperature with circadian drift and HVAC sawtooth"},
        {"name": "light",       "unit": "lux",    "description": "Illuminance: dark baseline + Poisson light-intrusion events"},
        {"name": "humidity",    "unit": "%",      "description": "Relative humidity, anti-correlated with temperature"},
        {"name": "noise",       "unit": "dB SPL", "description": "Ambient noise: quiet baseline + exponentially-decaying Poisson events"},
    ],
    "rf_test_mean_r2":   r2(test_metrics["test_mean_r2"]),
    "clf_accuracy":      r2(clf_accuracy),
})

# ══════════════════════════════════════════════════════════════════════════════
# 2. raw_signals.json  — one representative session per quality class
# ══════════════════════════════════════════════════════════════════════════════
sessions_by_class = {
    q: [s for s in sessions if s["quality_class"] == q]
    for q in ["good", "moderate", "poor"]
}
example = {q: sessions_by_class[q][5] for q in ["good", "moderate", "poor"]}

t_hours = [round(float(t / 60.0), 4) for t in example["good"]["t_min"]]

def round_sig(arr, dec=3):
    return [round(float(v), dec) for v in arr]

raw_data = {"t_hours": t_hours, "sessions": {}}
for q, sess in example.items():
    raw_data["sessions"][q] = {
        "temperature": round_sig(sess["temperature"]),
        "light":       round_sig(sess["light"]),
        "humidity":    round_sig(sess["humidity"]),
        "noise":       round_sig(sess["noise"]),
        "sleep_score":      r2(sess["sleep_score"]),
        "sleep_efficiency": r2(sess["sleep_efficiency"]),
        "awakenings":       int(sess["awakenings"]),
        "sleep_duration_h": r2(sess["sleep_duration_h"]),
        "season":       sess["season"],
    }
save("raw_signals.json", raw_data)

# ══════════════════════════════════════════════════════════════════════════════
# 3. filtered_signals.json — raw vs filtered temperature for before/after
# ══════════════════════════════════════════════════════════════════════════════
def reconstruct_raw_temp(session, global_seed=42):
    profile = sample_session_profile(session["session_id"], global_seed=global_seed)
    rng = np.random.default_rng(profile.rng_seed)
    t = session["t_min"]
    n = len(t)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    amp   = rng.uniform(0.5, 1.5)
    circ  = amp * np.sin(2 * np.pi / 1440.0 * t + phase)
    T_h   = profile.hvac_period_min
    saw   = (2 * ((t % T_h) / T_h) - 1.0) * profile.hvac_amplitude_c
    _ = rng.standard_normal(n); _ = rng.standard_normal(n)
    freqs_pk = np.fft.rfftfreq(n); freqs_pk[0] = 1.0
    f_r = rng.standard_normal(n // 2 + 1) + 1j * rng.standard_normal(n // 2 + 1)
    f_r[1:] /= np.sqrt(freqs_pk[1:]); f_r[0] = 0.0
    pink = np.fft.irfft(f_r, n=n)
    pink -= pink.mean()
    if pink.std() > 1e-12: pink /= pink.std()
    pink *= profile.temp_noise_scale
    return profile.base_temp_c + circ + saw + pink

filt_data = {"t_hours": t_hours, "signals": {}}
for q, sess in example.items():
    raw_t  = reconstruct_raw_temp(sess)
    filt_t = sess["temperature"]
    filt_data["signals"][q] = {
        "raw_temperature":      round_sig(raw_t),
        "filtered_temperature": round_sig(filt_t),
        "residual":             round_sig(raw_t - filt_t),
        "raw_std":    r2(float(raw_t.std())),
        "filt_std":   r2(float(filt_t.std())),
    }
save("filtered_signals.json", filt_data)

# ══════════════════════════════════════════════════════════════════════════════
# 4. fft_data.json — FFT spectra for each signal (averaged over sessions)
# ══════════════════════════════════════════════════════════════════════════════
rng_fft = np.random.default_rng(42)
sample50 = rng_fft.choice(len(sessions), size=50, replace=False)

fft_data_out = {}
for sig in ["temperature", "light", "humidity", "noise"]:
    all_psds = []
    freqs_ref = None
    for i in sample50:
        f, p = compute_fft_spectrum(sessions[i][sig])
        all_psds.append(p)
        freqs_ref = f
    mean_psd = np.mean(all_psds, axis=0)
    # Convert cpm → mHz  (1 cpm = 1000/60 mHz)
    freqs_mhz = freqs_ref * (1000.0 / 60.0)
    # Skip DC (index 0)
    fft_data_out[sig] = {
        "freqs_mhz": [round(float(v), 5) for v in freqs_mhz[1:]],
        "psd":       [round(float(v), 8) for v in mean_psd[1:]],
    }
    if sig == "temperature":
        fft_data_out[sig]["hvac_band_mhz"] = {
            "low":  round((1/120.0) * (1000.0/60.0), 5),
            "high": round((1/60.0)  * (1000.0/60.0), 5),
            "mid":  round((1/90.0)  * (1000.0/60.0), 5),
        }
save("fft_data.json", fft_data_out)

# ══════════════════════════════════════════════════════════════════════════════
# 5. poisson_events.json — events on light + noise for the moderate session
# ══════════════════════════════════════════════════════════════════════════════
sess_m = example["moderate"]
t_h_m  = [round(float(v / 60.0), 4) for v in sess_m["t_min"]]
light_s = sess_m["light"]
noise_s = sess_m["noise"]

light_onset = [int(i + 1) for i in range(len(light_s) - 1)
               if light_s[i] < 10.0 and light_s[i + 1] >= 10.0]
noise_onset = [int(i + 1) for i in range(len(noise_s) - 1)
               if noise_s[i] < 45.0 and noise_s[i + 1] >= 45.0]

save("poisson_events.json", {
    "t_hours": t_h_m,
    "light": {
        "signal":             round_sig(light_s),
        "event_indices":      light_onset,
        "event_times_hours":  [round(float(t_h_m[i]), 4) for i in light_onset],
        "threshold":          10.0,
        "n_events":           len(light_onset),
    },
    "noise": {
        "signal":             round_sig(noise_s),
        "event_indices":      noise_onset,
        "event_times_hours":  [round(float(t_h_m[i]), 4) for i in noise_onset],
        "threshold":          45.0,
        "n_events":           len(noise_onset),
    },
})

# ══════════════════════════════════════════════════════════════════════════════
# 6. ks_test_results.json
# ══════════════════════════════════════════════════════════════════════════════
ks_out = []
for _, row in ks_results.iterrows():
    ks_out.append({
        "variable":       row["variable"],
        "ks_stat":        float(row["ks_stat"]),
        "p_value":        float(row["p_value"]),
        "pass":           bool(row["pass"]),
        "synthetic_mean": float(row["synthetic_mean"]),
        "synthetic_std":  float(row["synthetic_std"]),
        "reference_mean": float(row["reference_mean"]),
        "reference_std":  float(row["reference_std"]),
    })
save("ks_test_results.json", {
    "results": ks_out,
    "n_pass":  int(ks_results["pass"].sum()),
    "n_total": len(ks_results),
    "alpha":   0.05,
})

# ══════════════════════════════════════════════════════════════════════════════
# 7. distribution_data.json — histogram bins for synthetic vs reference
# ══════════════════════════════════════════════════════════════════════════════
COL_MAP = {
    "sleep_efficiency": "sleep_efficiency",
    "sleep_duration_h": "sleep_duration_h",
    "awakenings":       "awakenings",
    "sleep_score":      "sleep_score",
    "temperature_mean": "temp_mean",
    "light_n_events":   "light_n_events",
}
rng_ks = np.random.default_rng(42)
dist_data = {}
for var, ref in REFERENCE_STATS.items():
    col = COL_MAP.get(var, var)
    if col not in df.columns: continue
    synth = df[col].dropna().values
    ref_smp = np.clip(rng_ks.normal(ref["mean"], ref["std"], 2000), ref["low"], ref["high"])
    # Build KDE curves using linspace
    lo = min(synth.min(), ref_smp.min())
    hi = max(synth.max(), ref_smp.max())
    xs = np.linspace(lo, hi, 200)
    from scipy.stats import gaussian_kde
    kde_synth = gaussian_kde(synth)(xs)
    kde_ref   = gaussian_kde(ref_smp)(xs)
    # Normalize
    kde_synth /= kde_synth.max()
    kde_ref   /= kde_ref.max()
    dist_data[var] = {
        "xs":          [round(float(v), 4) for v in xs],
        "kde_synthetic": [round(float(v), 6) for v in kde_synth],
        "kde_reference": [round(float(v), 6) for v in kde_ref],
        "synthetic_mean": r2(float(synth.mean())),
        "reference_mean": ref["mean"],
        "synthetic_std":  r2(float(synth.std())),
        "reference_std":  ref["std"],
    }
save("distribution_data.json", dist_data)

# ══════════════════════════════════════════════════════════════════════════════
# 8. feature_importance.json
# ══════════════════════════════════════════════════════════════════════════════
feat_cat_map = {f["name"]: f for f in FEATURE_CATALOGUE}
feat_imp_out = []
for _, row in importance_df.head(20).iterrows():
    cat = feat_cat_map.get(row["feature"], {})
    feat_imp_out.append({
        "feature":     row["feature"],
        "importance":  round(float(row["importance"]), 6),
        "unit":        cat.get("unit", ""),
        "description": cat.get("description", ""),
        "group": (
            "temperature" if "temp" in row["feature"] else
            "light"       if "light" in row["feature"] else
            "humidity"    if "humidity" in row["feature"] else
            "noise"       if "noise" in row["feature"] else
            "composite"
        ),
    })
save("feature_importance.json", {"features": feat_imp_out})

# ══════════════════════════════════════════════════════════════════════════════
# 9. confusion_matrix.json
# ══════════════════════════════════════════════════════════════════════════════
cm      = confusion_matrix(quality_test, quality_pred, labels=[0, 1, 2])
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
class_names = ["Good", "Moderate", "Poor"]

save("confusion_matrix.json", {
    "classes":   class_names,
    "matrix_norm": [[round(float(v), 4) for v in row] for row in cm_norm],
    "matrix_raw":  [[int(v) for v in row] for row in cm],
    "accuracy":    r2(clf_accuracy),
})

# ══════════════════════════════════════════════════════════════════════════════
# 10. classification_report.json
# ══════════════════════════════════════════════════════════════════════════════
clf_report = {}
for i, name in enumerate(class_names):
    y_true_bin = (quality_test == i).astype(int)
    y_pred_bin = (quality_pred == i).astype(int)
    clf_report[name] = {
        "precision": r2(float(precision_score(quality_test, quality_pred, labels=[i], average="macro"))),
        "recall":    r2(float(recall_score(quality_test, quality_pred, labels=[i], average="macro"))),
        "f1":        r2(float(f1_score(quality_test, quality_pred, labels=[i], average="macro"))),
        "support":   int((quality_test == i).sum()),
    }
clf_report["macro_avg"] = {
    "precision": r2(float(precision_score(quality_test, quality_pred, average="macro"))),
    "recall":    r2(float(recall_score(quality_test, quality_pred, average="macro"))),
    "f1":        r2(float(f1_score(quality_test, quality_pred, average="macro"))),
    "support":   len(quality_test),
}
clf_report["accuracy"] = r2(clf_accuracy)
save("classification_report.json", clf_report)

# ══════════════════════════════════════════════════════════════════════════════
# 11. roc_curves.json
# ══════════════════════════════════════════════════════════════════════════════
y_test_bin = label_binarize(quality_test, classes=[0, 1, 2])
roc_colors = ["#4ade80", "#fb923c", "#f87171"]
roc_out = {"classes": class_names, "colors": roc_colors, "curves": []}
for i, (name, color) in enumerate(zip(class_names, roc_colors)):
    from sklearn.metrics import roc_curve, auc as sk_auc
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], quality_prob[:, i])
    roc_auc = sk_auc(fpr, tpr)
    # Downsample for smaller JSON (keep ~100 points)
    step = max(1, len(fpr) // 100)
    roc_out["curves"].append({
        "class": name,
        "color": color,
        "fpr": [round(float(v), 4) for v in fpr[::step]],
        "tpr": [round(float(v), 4) for v in tpr[::step]],
        "auc": r2(roc_auc),
    })
save("roc_curves.json", roc_out)

# ══════════════════════════════════════════════════════════════════════════════
# 12. ablation_results.json
# ══════════════════════════════════════════════════════════════════════════════
ABLATION_LABELS = {
    "full_pipeline": "Full Pipeline",
    "no_lpf":        "No Butterworth LPF",
    "no_pink":       "No Pink Noise",
    "no_poisson":    "No Poisson Events",
    "no_hvac":       "No HVAC Sawtooth",
}
ab_out = []
for _, row in ablation_df.iterrows():
    ab_out.append({
        "ablation":    row["ablation"],
        "label":       ABLATION_LABELS.get(row["ablation"], row["ablation"]),
        "description": row["description"],
        "mean_r2":     round(float(row["mean_r2"]), 4),
        "delta_r2":    round(float(row["delta_r2"]), 4),
        "temp_acf1_mean":         round(float(row["temp_acf1_mean"]), 4),
        "light_n_events_mean":    round(float(row["light_n_events_mean"]), 3),
        "noise_n_events_mean":    round(float(row["noise_n_events_mean"]), 3),
        "temp_hvac_power_mean":   round(float(row["temp_hvac_power_mean"]), 6),
        "is_baseline":            row["ablation"] == "full_pipeline",
    })
save("ablation_results.json", {"conditions": ab_out})

# ══════════════════════════════════════════════════════════════════════════════
# 13. baseline_comparison.json
# ══════════════════════════════════════════════════════════════════════════════
model_labels = ["Random Forest", "Ridge Regression", "Mean Baseline"]
model_keys   = ["rf", "ridge", "mean"]
bl_out = {
    "models": model_labels,
    "metrics": {
        "mean_r2": [
            r2(test_metrics["test_mean_r2"]),
            r2(baseline_metrics["ridge_regression"]["test_mean_r2"]),
            r2(baseline_metrics["mean_baseline"]["test_mean_r2"]),
        ],
    },
    "per_label": {},
    "colors": ["#6366f1", "#22d3ee", "#94a3b8"],
}
for lbl in label_names:
    bl_out["per_label"][lbl] = {
        "rf":    r2(test_metrics[f"test_{lbl}_r2"]),
        "ridge": r2(baseline_metrics["ridge_regression"][f"test_{lbl}_r2"]),
        "mean":  r2(baseline_metrics["mean_baseline"][f"test_{lbl}_r2"]),
    }
    bl_out["per_label"][lbl]["rf_mae"] = r2(test_metrics[f"test_{lbl}_mae"])

# RF improvement over ridge
bl_out["rf_vs_ridge_delta"] = r2(test_metrics["test_mean_r2"] -
                                  baseline_metrics["ridge_regression"]["test_mean_r2"])
bl_out["rf_vs_mean_delta"]  = r2(test_metrics["test_mean_r2"] -
                                  baseline_metrics["mean_baseline"]["test_mean_r2"])
save("baseline_comparison.json", bl_out)

# ══════════════════════════════════════════════════════════════════════════════
# 14. parameter_explorer.json — precomputed grid for the parameter sliders
# ══════════════════════════════════════════════════════════════════════════════
# Grid: quality_class × temperature_setting
# quality: poor(0), moderate(1), good(2)
# temp: cold(0,−2°C offset), optimal(1,0°C), warm(2,+3°C)
print("▶ Building parameter explorer grid (9 combinations)...")
TEMP_OFFSETS = {"cold": -2.0, "optimal": 0.0, "warm": 3.0}
QUALITY_ORDER = ["poor", "moderate", "good"]
TEMP_ORDER    = ["cold", "optimal", "warm"]

# Find example sessions close to each target
from src.data_generation import sample_session_profile as ssp

explorer_grid = []
for q_idx, quality in enumerate(QUALITY_ORDER):
    for t_idx, temp_name in enumerate(TEMP_ORDER):
        # Pick a session of this quality class
        q_sessions = sessions_by_class[quality]
        target_offset = TEMP_OFFSETS[temp_name]
        # Find session whose mean temp is closest to (20 + offset)
        target_temp = 20.0 + target_offset
        best_sess = min(q_sessions, key=lambda s: abs(s["temperature"].mean() - target_temp))
        explorer_grid.append({
            "quality":      quality,
            "quality_idx":  q_idx,
            "temp_setting": temp_name,
            "temp_idx":     t_idx,
            "t_hours":      [round(float(v/60),3) for v in best_sess["t_min"]],
            "temperature":  round_sig(best_sess["temperature"]),
            "light":        round_sig(best_sess["light"]),
            "humidity":     round_sig(best_sess["humidity"]),
            "noise":        round_sig(best_sess["noise"]),
            "sleep_score":      r2(best_sess["sleep_score"]),
            "sleep_efficiency": r2(best_sess["sleep_efficiency"]),
            "awakenings":       int(best_sess["awakenings"]),
        })

save("parameter_explorer.json", {"grid": explorer_grid})

# ══════════════════════════════════════════════════════════════════════════════
# 15. ml_summary.json — comprehensive ML metrics for hero + results sections
# ══════════════════════════════════════════════════════════════════════════════
label_display = {
    "sleep_efficiency": "Sleep Efficiency",
    "sleep_duration_h": "Sleep Duration",
    "awakenings":       "Awakenings",
    "sleep_score":      "Sleep Score",
}
ml_summary = {
    "rf_test_mean_r2":  r2(test_metrics["test_mean_r2"]),
    "rf_val_mean_r2":   r2(val_metrics["val_mean_r2"]),
    "clf_accuracy":     r2(clf_accuracy),
    "clf_macro_f1":     r2(float(f1_score(quality_test, quality_pred, average="macro"))),
    "clf_macro_precision": r2(float(precision_score(quality_test, quality_pred, average="macro"))),
    "clf_macro_recall":    r2(float(recall_score(quality_test, quality_pred, average="macro"))),
    "clf_roc_auc_macro": r2(float(np.mean([
        sk_auc(*roc_curve(y_test_bin[:, i], quality_prob[:, i])[:2])
        for i in range(3)
    ]))),
    "best_params": val_metrics.get("best_params", "n_estimators=200, max_features=sqrt"),
    "n_cv_folds": 5,
    "per_label": {
        lbl: {
            "display": label_display[lbl],
            "test_r2":   r2(test_metrics[f"test_{lbl}_r2"]),
            "test_mae":  r2(test_metrics[f"test_{lbl}_mae"]),
            "test_rmse": r2(test_metrics[f"test_{lbl}_rmse"]),
            "cv_r2_mean": r2(cv_metrics[f"{lbl}_cv_r2_mean"]),
            "cv_r2_std":  r2(cv_metrics[f"{lbl}_cv_r2_std"]),
        } for lbl in label_names
    },
    "ridge_mean_r2": r2(baseline_metrics["ridge_regression"]["test_mean_r2"]),
    "mean_baseline_r2": r2(baseline_metrics["mean_baseline"]["test_mean_r2"]),
}
save("ml_summary.json", ml_summary)

print("\n✓ All data files written successfully.")
print(f"  Output dir: {OUTPUT_DIR}")
print(f"  Files: {len(list(OUTPUT_DIR.glob('*.json')))}")
