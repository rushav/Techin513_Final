#!/usr/bin/env python3
"""
demo.py — Single entry point for the Synthetic Sleep Environment pipeline.

Running this script end-to-end produces:
  • The full 2,500-session synthetic dataset (data/synthetic_sleep_dataset.csv)
  • 12 publication-quality figures (results/figures/)
  • 6 metrics files (results/metrics/)

Usage
-----
    python demo.py               # Full run (2,500 sessions, ~3–5 minutes)
    python demo.py --quick       # Quick run (500 sessions, ~45 seconds)
    python demo.py --seed 123    # Custom random seed

All outputs are deterministic given a fixed seed.  The canonical seed
used in the report is 42 (the default).

Authors: Rushav Dash, Lisa Li — TECHIN 513 Final Project
"""

import argparse
import sys
import time
import json
from pathlib import Path

# Ensure the project root is in the Python path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from src.utils import (
    seed_everything, ensure_dirs, save_json,
    FIGURES_DIR, METRICS_DIR, DATA_DIR, get_logger, Timer,
    GLOBAL_SEED,
)
from src.data_generation import generate_dataset
from src.feature_extraction import build_feature_matrix, FEATURE_NAMES
from src.signal_processing import compute_fft_spectrum, autocorrelation
from src.ml_pipeline import (
    split_dataset, train_random_forest, train_baselines,
    extract_feature_importances, run_ablation_study, cross_validate_rf,
    evaluate_model,
)
from src.validation import (
    run_ks_tests, run_discriminability_test, run_sanity_checks,
    generate_validation_report, REFERENCE_STATS,
)
from src.visualisation import (
    plot_example_session, plot_temperature_spectrum, plot_filter_effect,
    plot_poisson_events, plot_label_distributions, plot_feature_label_correlation,
    plot_feature_importance, plot_predictions, plot_ablation,
    plot_ks_comparisons, plot_seasonal_comparison,
)

logger = get_logger("demo")

LABEL_NAMES = ["sleep_efficiency", "sleep_duration_h", "awakenings", "sleep_score"]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthetic Sleep Environment Dataset Generator")
    p.add_argument(
        "--n-sessions", type=int, default=2500,
        help="Number of sessions to generate (default 2500)",
    )
    p.add_argument(
        "--quick", action="store_true",
        help="Run a quick demonstration with 500 sessions",
    )
    p.add_argument(
        "--seed", type=int, default=GLOBAL_SEED,
        help=f"Global random seed (default {GLOBAL_SEED})",
    )
    p.add_argument(
        "--skip-ablation", action="store_true",
        help="Skip the ablation study (saves time)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    n_sessions = 500 if args.quick else args.n_sessions
    seed = args.seed

    print(
        "\n" + "=" * 65 + "\n"
        "  Synthetic Sleep Environment Dataset Generator\n"
        "  Team 7 — Rushav Dash, Lisa Li — TECHIN 513 Final Project\n"
        + "=" * 65
    )
    print(f"\n  Sessions : {n_sessions}")
    print(f"  Seed     : {seed}")
    print(f"  Quick    : {args.quick}\n")

    # -----------------------------------------------------------------------
    # Step 0 — Initialise
    # -----------------------------------------------------------------------
    seed_everything(seed)
    ensure_dirs()
    start_wall = time.perf_counter()

    # -----------------------------------------------------------------------
    # Step 1 — Generate dataset
    # -----------------------------------------------------------------------
    logger.info("=== STEP 1: Generating synthetic dataset ===")
    with Timer("data generation"):
        sessions = generate_dataset(n_sessions=n_sessions, global_seed=seed)

    logger.info("  %d sessions generated, each with %d time steps.",
                len(sessions), len(sessions[0]["t_min"]))

    # -----------------------------------------------------------------------
    # Step 2 — Extract features
    # -----------------------------------------------------------------------
    logger.info("=== STEP 2: Extracting features ===")
    with Timer("feature extraction"):
        X, y, feature_names, label_names, metadata = build_feature_matrix(sessions)

    # Assemble flat DataFrame for analysis and export
    df = pd.DataFrame(X, columns=feature_names)
    for i, lname in enumerate(label_names):
        df[lname] = y[:, i]
    for key in ("session_id", "season", "quality_class"):
        df[key] = [m[key] for m in metadata]

    # Save dataset
    dataset_path = DATA_DIR / "synthetic_sleep_dataset.csv"
    df.to_csv(dataset_path, index=False)
    logger.info("  Dataset saved to %s", dataset_path)

    # -----------------------------------------------------------------------
    # Step 3 — Figures: signals and spectra
    # -----------------------------------------------------------------------
    logger.info("=== STEP 3: Generating signal figures ===")
    # Pick a representative 'moderate' session for illustration
    moderate_sessions = [s for s in sessions if s["quality_class"] == "moderate"]
    example = moderate_sessions[42 % len(moderate_sessions)]

    plot_example_session(example)
    plot_temperature_spectrum(sessions, n_sessions=min(50, n_sessions))
    plot_filter_effect(example)
    plot_poisson_events(example)

    # -----------------------------------------------------------------------
    # Step 4 — ML pipeline
    # -----------------------------------------------------------------------
    logger.info("=== STEP 4: Training ML models ===")
    X_train, X_val, X_test, y_train, y_val, y_test, idx_tr, idx_v, idx_te = \
        split_dataset(X, y, metadata, seed=seed)

    with Timer("Random Forest training"):
        rf_model, val_metrics = train_random_forest(
            X_train, y_train, X_val, y_val, label_names, seed=seed
        )

    test_metrics = evaluate_model(rf_model, X_test, y_test, label_names, "test_")
    logger.info("  Test mean R² = %.4f", test_metrics["test_mean_r2"])

    # Baseline comparison
    baseline_metrics = train_baselines(X_train, y_train, X_test, y_test, label_names, seed)

    # Cross-validation
    cv_metrics = cross_validate_rf(X, y, label_names, cv=5, seed=seed)

    # Feature importances
    importance_df = extract_feature_importances(rf_model, feature_names)

    # Predictions for plotting
    y_pred_test = rf_model.predict(X_test)

    # -----------------------------------------------------------------------
    # Step 5 — ML figures
    # -----------------------------------------------------------------------
    logger.info("=== STEP 5: Generating ML figures ===")
    plot_label_distributions(df)
    plot_feature_label_correlation(df, feature_names, label_names)
    plot_feature_importance(importance_df)
    plot_predictions(y_test, y_pred_test, label_names)

    # -----------------------------------------------------------------------
    # Step 6 — Validation
    # -----------------------------------------------------------------------
    logger.info("=== STEP 6: Running validation ===")
    ks_results = run_ks_tests(df, seed=seed)
    discriminability = run_discriminability_test(X, feature_names, seed=seed)
    sanity_checks = run_sanity_checks(df)
    validation_report = generate_validation_report(ks_results, discriminability, sanity_checks)

    # -----------------------------------------------------------------------
    # Step 7 — Ablation study
    # -----------------------------------------------------------------------
    if not args.skip_ablation:
        logger.info("=== STEP 7: Running ablation study ===")
        with Timer("ablation study"):
            ablation_df = run_ablation_study(sessions, feature_names, label_names, seed=seed)
    else:
        logger.info("=== STEP 7: Ablation study skipped ===")
        ablation_df = pd.DataFrame([
            {"ablation": "full_pipeline", "description": "Skipped", "mean_r2": 0, "delta_r2": 0,
             "temp_acf1_mean": 0, "light_n_events_mean": 0, "noise_n_events_mean": 0,
             "temp_hvac_power_mean": 0}
        ])

    # -----------------------------------------------------------------------
    # Step 8 — Remaining figures
    # -----------------------------------------------------------------------
    logger.info("=== STEP 8: Generating validation and ablation figures ===")
    plot_ks_comparisons(df, ks_results)
    plot_seasonal_comparison(df)
    if not args.skip_ablation:
        plot_ablation(ablation_df)

    # -----------------------------------------------------------------------
    # Step 9 — Save all metrics
    # -----------------------------------------------------------------------
    logger.info("=== STEP 9: Saving metrics ===")

    # 9a — Dataset summary statistics
    summary = {
        "n_sessions": len(df),
        "n_features": len(feature_names),
        "n_labels": len(label_names),
        "seed": seed,
        "dataset_path": str(dataset_path),
        "label_stats": {
            lname: {
                "mean": round(float(df[lname].mean()), 4),
                "std": round(float(df[lname].std()), 4),
                "min": round(float(df[lname].min()), 4),
                "max": round(float(df[lname].max()), 4),
            }
            for lname in label_names
        },
        "season_counts": df["season"].value_counts().to_dict(),
        "quality_counts": df["quality_class"].value_counts().to_dict(),
    }
    save_json(summary, METRICS_DIR / "dataset_summary.json")

    # 9b — ML results
    ml_results = {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "cv_metrics": cv_metrics,
        "baselines": baseline_metrics,
        "top10_features": importance_df.head(10).to_dict(orient="records"),
    }
    save_json(ml_results, METRICS_DIR / "ml_results.json")

    # 9c — Validation report
    save_json(validation_report, METRICS_DIR / "validation_report.json")

    # 9d — KS test results CSV
    ks_results.to_csv(METRICS_DIR / "ks_test_results.csv", index=False)

    # 9e — Sanity check results CSV
    sanity_checks.to_csv(METRICS_DIR / "sanity_checks.csv", index=False)

    # 9f — Ablation results CSV
    ablation_df.to_csv(METRICS_DIR / "ablation_results.csv", index=False)

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    wall_time = time.perf_counter() - start_wall

    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE")
    print("=" * 65)
    print(f"\n  Wall time          : {wall_time:.1f} s")
    print(f"  Sessions generated : {n_sessions}")
    print(f"  Features           : {len(feature_names)}")
    print(f"\n  --- Dataset Statistics ---")
    for lname in label_names:
        row = summary["label_stats"][lname]
        print(f"  {lname:<25} mean={row['mean']:.4f}  std={row['std']:.4f}")

    print(f"\n  --- ML Performance (Test Set) ---")
    print(f"  Random Forest mean R²  : {test_metrics['test_mean_r2']:.4f}")
    print(f"  Mean baseline mean R²  : {baseline_metrics['mean_baseline']['test_mean_r2']:.4f}")
    print(f"  Ridge regression R²    : {baseline_metrics['ridge_regression']['test_mean_r2']:.4f}")
    for lname in label_names:
        print(f"  {lname:<25} R²={test_metrics[f'test_{lname}_r2']:.4f}  "
              f"MAE={test_metrics[f'test_{lname}_mae']:.4f}")

    print(f"\n  --- Validation ---")
    n_ks_pass = int(ks_results["pass"].sum())
    n_sanity_pass = int(sanity_checks["pass"].sum())
    print(f"  KS tests passed   : {n_ks_pass}/{len(ks_results)}")
    print(f"  Sanity checks     : {n_sanity_pass}/{len(sanity_checks)}")
    print(f"  Discriminability AUC : {discriminability['auc_roc']:.4f}")

    if not args.skip_ablation:
        print(f"\n  --- Ablation Study ---")
        for _, row in ablation_df[ablation_df["ablation"] != "full_pipeline"].iterrows():
            print(f"  {row['ablation']:<20} ΔR²={row['delta_r2']:+.4f}")

    print(f"\n  Figures : {FIGURES_DIR}")
    print(f"  Metrics : {METRICS_DIR}")
    print(f"  Dataset : {dataset_path}")
    print("\n" + "=" * 65 + "\n")


if __name__ == "__main__":
    main()
