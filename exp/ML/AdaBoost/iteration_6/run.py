from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from exp.ML.common import load_ionosphere, run_experiment


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

BEST_PARAMS: dict[str, Any] = {
    "n_estimators": 50,
    "learning_rate": 0.5,
    "estimator__max_depth": 2,
    "estimator__min_samples_leaf": 2,
    "estimator__criterion": "entropy",
}

WEIGHT_VARIANTS = {
    "None": None,
    "balanced": "balanced",
}


def build_model(seed: int, class_weight: Any) -> AdaBoostClassifier:
    model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(
            random_state=seed,
            class_weight=class_weight,
        ),
        random_state=seed,
    )
    model.set_params(**BEST_PARAMS)
    return model


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    comparisons: dict[str, dict[str, Any]] = {}

    for label, cw in WEIGHT_VARIANTS.items():
        iteration = f"iteration_6_class_weight_{label}"
        note = (
            f"AdaBoost with best params from iteration_4, "
            f"base estimator class_weight={cw!r}. "
            f"Comparison: None vs 'balanced'. "
            f"(b:g ≈ 36:64, moderate imbalance). "
            f"Note: iteration_4 GridSearchCV found class_weight=None performed best; "
            f"this iteration re-runs both for a direct single-split + CV + repeated comparison."
        )

        print(f"\n{'=' * 60}")
        print(f"AdaBoost class_weight={cw!r}")
        print(f"{'=' * 60}")
        run_experiment(
            model_name="AdaBoost",
            iteration_name=iteration,
            build_model=lambda seed, cw=cw: build_model(seed, cw),
            output_dir=OUTPUT_DIR,
            experiment_note=note,
            run_cv=True,
            run_repeated=True,
        )
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

        metrics_path = OUTPUT_DIR / f"{label}_metrics.json"
        if metrics_path.exists():
            comparisons[label] = json.loads(metrics_path.read_text(encoding="utf-8"))

    keys = [
        "test_accuracy", "test_precision", "test_recall",
        "test_f1", "test_roc_auc",
    ]
    print(f"\n{'=' * 60}")
    print("AdaBoost class_weight 对比 (single split)")
    print(f"{'=' * 60}")
    header = f"{'Metric':<20}"
    for label in WEIGHT_VARIANTS:
        header += f" {label:>12}"
    print(header)
    print("-" * (20 + 13 * 2))
    for key in keys:
        line = f"{key:<20}"
        for label in WEIGHT_VARIANTS:
            sm = comparisons.get(label, {}).get("single_split_metrics", {})
            val = sm.get(key, float("nan"))
            line += f" {val:>12.4f}" if isinstance(val, float) else f" {'N/A':>12}"
        print(line)

    print(f"\nRepeated split (mean ± std, n={5}):")
    for key in keys:
        line = f"{key:<20}"
        for label in WEIGHT_VARIANTS:
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
