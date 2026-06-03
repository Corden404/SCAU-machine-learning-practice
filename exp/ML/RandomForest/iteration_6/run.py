from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.ensemble import RandomForestClassifier

from exp.ML.common import load_ionosphere, run_experiment


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

BEST_PARAMS: dict[str, Any] = {
    "n_estimators": 100,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_samples": 1.0,
    "max_leaf_nodes": 16,
    "max_features": 0.5,
    "criterion": "entropy",
    "ccp_alpha": 0.0,
}

WEIGHT_VARIANTS = {
    "None": None,
    "balanced": "balanced",
    "balanced_subsample": "balanced_subsample",
}


def build_model(seed: int, class_weight: Any) -> RandomForestClassifier:
    return RandomForestClassifier(
        random_state=seed,
        n_jobs=-1,
        class_weight=class_weight,
        **BEST_PARAMS,
    )


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    comparisons: dict[str, dict[str, Any]] = {}

    for label, cw in WEIGHT_VARIANTS.items():
        iteration = f"iteration_6_class_weight_{label}"
        note = (
            f"RandomForest with best params from iteration_4, "
            f"class_weight={cw!r}. "
            f"Comparison: None vs 'balanced' vs 'balanced_subsample' "
            f"(b:g ≈ 36:64, moderate imbalance)."
        )

        print(f"\n{'=' * 60}")
        print(f"RF class_weight={cw!r}")
        print(f"{'=' * 60}")
        run_experiment(
            model_name="RandomForest",
            iteration_name=iteration,
            build_model=lambda seed, cw=cw: build_model(seed, cw),
            output_dir=OUTPUT_DIR,
            experiment_note=note,
            run_cv=True,
            run_repeated=True,
        )
        # Rename outputs to include class_weight label
        for suffix in [
            "single_split_metrics.csv",
            "cv_metrics.csv",
            "cv_metrics_summary.csv",
            "repeated_split_metrics.csv",
            "repeated_split_summary.csv",
            "metrics.json",
        ]:
            src = OUTPUT_DIR / suffix
            if src.exists():
                dst = OUTPUT_DIR / f"{label}_{suffix}"
                src.replace(dst)

        # Read metrics for comparison
        metrics_path = OUTPUT_DIR / f"{label}_metrics.json"
        if metrics_path.exists():
            comparisons[label] = json.loads(metrics_path.read_text(encoding="utf-8"))

    # Build comparison table
    keys = [
        "test_accuracy", "test_precision", "test_recall",
        "test_f1", "test_roc_auc",
    ]
    print(f"\n{'=' * 60}")
    print("RF class_weight 对比 (single split)")
    print(f"{'=' * 60}")
    header = f"{'Metric':<20}"
    for label in WEIGHT_VARIANTS:
        header += f" {label:>12}"
    print(header)
    print("-" * (20 + 13 * 3))
    for key in keys:
        line = f"{key:<20}"
        for label in WEIGHT_VARIANTS:
            sm = comparisons.get(label, {}).get("single_split_metrics", {})
            val = sm.get(key, float("nan"))
            line += f" {val:>12.4f}" if isinstance(val, float) else f" {'N/A':>12}"
        print(line)

    # Also print repeated split summary if available
    print(f"\nRepeated split (mean ± std, n={5}):")
    for key in keys:
        line = f"{key:<20}"
        for label in WEIGHT_VARIANTS:
            # Read from repeated split summary
            summary_path = OUTPUT_DIR / f"{label}_repeated_split_summary.csv"
            if summary_path.exists():
                df = pd.read_csv(summary_path, encoding="utf-8-sig")
                row = df[df["metric"] == key]
                if not row.empty:
                    m = float(row[f"repeated_mean"].iloc[0])
                    s = float(row[f"repeated_std"].iloc[0])
                    line += f" {m:.4f}±{s:.3f}"
                    continue
            line += f" {'N/A':>15}"
        print(line)

    comparison_path = OUTPUT_DIR / "class_weight_comparison.json"
    comparison_path.write_text(json.dumps(comparisons, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nComparison saved to {comparison_path}")
