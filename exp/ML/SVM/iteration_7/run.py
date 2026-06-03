from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from exp.ML.common import (
    CV_FOLDS,
    LABELS,
    POS_LABEL,
    RANDOM_SEED,
    TEST_SIZE,
    binary_target,
    load_ionosphere,
    summarize_table,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
FIXED_C = 3.0
FIXED_GAMMA = 0.03
FIXED_CLASS_WEIGHT = "balanced"
INNER_CV_FOLDS = 5
EXPANDED_SEEDS = list(range(30))


def take_rows(data: pd.DataFrame | pd.Series, indices: np.ndarray) -> pd.DataFrame | pd.Series:
    if hasattr(data, "iloc"):
        return data.iloc[indices]
    return data[indices]


def base_pipeline(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("standardize", StandardScaler()),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    C=FIXED_C,
                    gamma=FIXED_GAMMA,
                    class_weight=FIXED_CLASS_WEIGHT,
                    random_state=seed,
                ),
            ),
        ]
    )


def positive_decision_scores(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    values = np.asarray(model.decision_function(x), dtype=float)
    classes = list(getattr(model, "classes_", LABELS))
    if values.ndim != 1:
        values = values[:, classes.index(POS_LABEL)]
    elif len(classes) == 2 and classes[1] != POS_LABEL:
        values = -values
    return values


def threshold_candidates(scores: np.ndarray) -> np.ndarray:
    unique_scores = np.unique(np.asarray(scores, dtype=float))
    if len(unique_scores) == 0:
        return np.asarray([0.0])
    if len(unique_scores) == 1:
        value = float(unique_scores[0])
        return np.asarray([value - 1e-6, 0.0, value + 1e-6])
    midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0
    score_range = float(unique_scores[-1] - unique_scores[0])
    epsilon = max(score_range * 1e-6, 1e-6)
    candidates = np.concatenate(
        [
            [unique_scores[0] - epsilon],
            midpoints,
            [unique_scores[-1] + epsilon],
            [0.0],
        ]
    )
    return np.asarray(sorted(set(float(value) for value in candidates)))


def metrics_at_threshold(
    scores: np.ndarray, y: pd.Series, threshold: float
) -> dict[str, float]:
    prediction = pd.Series(
        np.where(scores >= threshold, POS_LABEL, "b"), index=y.index
    )
    return {
        "threshold": float(threshold),
        "macro_f1": f1_score(
            y, prediction, average="macro", pos_label=None, zero_division=0
        ),
        "accuracy": accuracy_score(y, prediction),
        "b_recall": recall_score(y, prediction, pos_label="b", zero_division=0),
        "b_f1": f1_score(y, prediction, pos_label="b", zero_division=0),
        "g_recall": recall_score(
            y, prediction, pos_label=POS_LABEL, zero_division=0
        ),
        "g_f1": f1_score(
            y, prediction, pos_label=POS_LABEL, zero_division=0
        ),
    }


def find_best_threshold(scores: np.ndarray, y: pd.Series) -> tuple[float, float]:
    rows = [
        metrics_at_threshold(scores, y, t)
        for t in threshold_candidates(scores)
    ]
    results = pd.DataFrame(rows).sort_values(
        by=["macro_f1", "b_recall", "g_f1", "accuracy"],
        ascending=[False, False, False, False],
    )
    best = results.iloc[0]
    default = metrics_at_threshold(scores, y, 0.0)
    return float(best["threshold"]), default["macro_f1"]


def detailed_metrics(
    model: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
    threshold: float | None,
) -> dict[str, float]:
    """Compute all metrics. If threshold is None, use model's default predict."""
    if threshold is not None:
        scores = pd.Series(positive_decision_scores(model, x), index=x.index)
        prediction = pd.Series(
            np.where(scores.values >= threshold, POS_LABEL, "b"), index=y.index
        )
        roc_scores = scores
    else:
        prediction = pd.Series(model.predict(x), index=y.index)
        roc_scores = pd.Series(positive_decision_scores(model, x), index=x.index)

    return {
        "accuracy": accuracy_score(y, prediction),
        "macro_f1": f1_score(
            y, prediction, average="macro", pos_label=None, zero_division=0
        ),
        "b_recall": recall_score(y, prediction, pos_label="b", zero_division=0),
        "b_f1": f1_score(y, prediction, pos_label="b", zero_division=0),
        "g_recall": recall_score(
            y, prediction, pos_label=POS_LABEL, zero_division=0
        ),
        "g_f1": f1_score(
            y, prediction, pos_label=POS_LABEL, zero_division=0
        ),
        "roc_auc": roc_auc_score(binary_target(y), roc_scores),
    }


def run_one_split(
    x: pd.DataFrame,
    y: pd.Series,
    seed: int,
) -> dict[str, Any]:
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, stratify=y, random_state=seed
    )

    # Model A: iter_4/5 — fixed threshold = 0.0 (SVC default)
    model_a = base_pipeline(seed)
    model_a.fit(x_train, y_train)

    # Model B: iter_6 — threshold tuned on inner 5-fold OOF
    model_b = base_pipeline(seed)
    model_b.fit(x_train, y_train)

    oof_scores = np.empty(len(y_train), dtype=float)
    inner_splitter = StratifiedKFold(
        n_splits=INNER_CV_FOLDS, shuffle=True, random_state=seed
    )
    for train_idx, valid_idx in inner_splitter.split(x_train, y_train):
        inner_model = base_pipeline(seed)
        inner_model.fit(
            take_rows(x_train, train_idx), take_rows(y_train, train_idx)
        )
        oof_scores[valid_idx] = positive_decision_scores(
            inner_model, take_rows(x_train, valid_idx)
        )

    best_threshold, default_macro_f1 = find_best_threshold(oof_scores, y_train)

    train_a = detailed_metrics(model_a, x_train, y_train, threshold=None)
    test_a = detailed_metrics(model_a, x_test, y_test, threshold=None)
    train_b = detailed_metrics(model_b, x_train, y_train, threshold=best_threshold)
    test_b = detailed_metrics(model_b, x_test, y_test, threshold=best_threshold)

    row: dict[str, Any] = {"seed": seed, "best_threshold": best_threshold}

    for key, value in train_a.items():
        row[f"iter45_train_{key}"] = value
    for key, value in test_a.items():
        row[f"iter45_test_{key}"] = value
    for key, value in train_b.items():
        row[f"iter6_train_{key}"] = value
    for key, value in test_b.items():
        row[f"iter6_test_{key}"] = value

    return row


def build_comparison_summary(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "accuracy",
        "macro_f1",
        "b_recall",
        "b_f1",
        "g_recall",
        "g_f1",
        "roc_auc",
    ]
    rows = []
    for model_label, prefix in [("iter_4/5", "iter45"), ("iter_6", "iter6")]:
        for metric in metrics:
            train_col = f"{prefix}_train_{metric}"
            test_col = f"{prefix}_test_{metric}"

            train_mean = results[train_col].mean()
            train_std = results[train_col].std()
            test_mean = results[test_col].mean()
            test_std = results[test_col].std()

            rows.append(
                {
                    "model": model_label,
                    "metric": metric,
                    "train_mean": train_mean,
                    "train_std": train_std,
                    "test_mean": test_mean,
                    "test_std": test_std,
                    "train_test_gap": train_mean - test_mean,
                }
            )
    return pd.DataFrame(rows)


def build_head_to_head(results: pd.DataFrame) -> pd.DataFrame:
    """For each metric, show iter_6 - iter_4/5 difference per seed, then summarize."""
    metrics = [
        "accuracy",
        "macro_f1",
        "b_recall",
        "b_f1",
        "g_recall",
        "g_f1",
        "roc_auc",
    ]
    rows = []
    for metric in metrics:
        train_diffs = (
            results[f"iter6_train_{metric}"] - results[f"iter45_train_{metric}"]
        )
        test_diffs = (
            results[f"iter6_test_{metric}"] - results[f"iter45_test_{metric}"]
        )
        rows.append(
            {
                "metric": metric,
                "train_diff_mean": train_diffs.mean(),
                "train_diff_std": train_diffs.std(),
                "test_diff_mean": test_diffs.mean(),
                "test_diff_std": test_diffs.std(),
                "test_diff_wins": int((test_diffs > 0).sum()),
                "test_diff_losses": int((test_diffs < 0).sum()),
                "test_diff_ties": int((test_diffs == 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    x, y = load_ionosphere()

    print("=" * 60)
    print(f"SVM iteration_7: 30-seed stability comparison")
    print(f"  seeds: {EXPANDED_SEEDS[0]}–{EXPANDED_SEEDS[-1]} ({len(EXPANDED_SEEDS)} total)")
    print("=" * 60)

    # ── 30-seed repeated split ──
    repeated_rows = []
    for seed in EXPANDED_SEEDS:
        row = run_one_split(x, y, seed)
        repeated_rows.append(row)
        if (seed + 1) % 10 == 0:
            print(f"  completed seed {seed + 1}/{len(EXPANDED_SEEDS)}")

    repeated = pd.DataFrame(repeated_rows)
    repeated.to_csv(
        OUTPUT_DIR / "expanded_repeated_split.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ── Summaries ──
    comparison = build_comparison_summary(repeated)
    comparison.to_csv(
        OUTPUT_DIR / "model_comparison_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    head_to_head = build_head_to_head(repeated)
    head_to_head.to_csv(
        OUTPUT_DIR / "head_to_head_diff.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ── Also run legacy 5-seed repeated split (seeds 0-4) for both models ──
    # This gives us standard common.py output for backward compatibility
    from exp.ML.common import (
        REPEATED_SEEDS,
        evaluate_split,
        metric_block,
    )

    # Run with standard common.py evaluate_split (iter_4/5 model)
    legacy_a_rows = []
    for seed in REPEATED_SEEDS:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=TEST_SIZE, stratify=y, random_state=seed
        )
        model = base_pipeline(seed)
        model.fit(x_train, y_train)
        row = {
            "model": "SVM",
            "iteration": "iteration_7_iter45",
            "seed": seed,
        }
        row.update(metric_block(model, x_train, y_train, "train"))
        row.update(metric_block(model, x_test, y_test, "test"))
        legacy_a_rows.append(row)

    legacy_a = pd.DataFrame(legacy_a_rows)
    legacy_a.to_csv(
        OUTPUT_DIR / "repeated_split_metrics_iter45.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summarize_table(legacy_a, "repeated").to_csv(
        OUTPUT_DIR / "repeated_split_summary_iter45.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Run with iter_6-style threshold tuning
    legacy_b_rows = []
    for seed in REPEATED_SEEDS:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=TEST_SIZE, stratify=y, random_state=seed
        )
        model = base_pipeline(seed)
        model.fit(x_train, y_train)

        oof_scores = np.empty(len(y_train), dtype=float)
        inner_splitter = StratifiedKFold(
            n_splits=INNER_CV_FOLDS, shuffle=True, random_state=seed
        )
        for train_idx, valid_idx in inner_splitter.split(x_train, y_train):
            inner_model = base_pipeline(seed)
            inner_model.fit(
                take_rows(x_train, train_idx), take_rows(y_train, train_idx)
            )
            oof_scores[valid_idx] = positive_decision_scores(
                inner_model, take_rows(x_train, valid_idx)
            )
        best_threshold, _ = find_best_threshold(oof_scores, y_train)

        train_metrics = detailed_metrics(
            model, x_train, y_train, threshold=best_threshold
        )
        test_metrics = detailed_metrics(
            model, x_test, y_test, threshold=best_threshold
        )
        row = {
            "model": "SVM",
            "iteration": "iteration_7_iter6",
            "seed": seed,
            "best_threshold": best_threshold,
        }
        for k, v in train_metrics.items():
            row[f"train_{k}"] = v
        for k, v in test_metrics.items():
            row[f"test_{k}"] = v
        legacy_b_rows.append(row)

    legacy_b = pd.DataFrame(legacy_b_rows)
    legacy_b.to_csv(
        OUTPUT_DIR / "repeated_split_metrics_iter6.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summarize_table(legacy_b, "repeated").to_csv(
        OUTPUT_DIR / "repeated_split_summary_iter6.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ── 5-fold CV for both models ──
    cv_splitter = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED
    )

    cv_rows_a = []
    cv_rows_b = []
    for fold_idx, (train_idx, test_idx) in enumerate(
        cv_splitter.split(x, y), start=1
    ):
        x_train_cv = take_rows(x, train_idx)
        x_test_cv = take_rows(x, test_idx)
        y_train_cv = take_rows(y, train_idx)
        y_test_cv = take_rows(y, test_idx)

        # iter_4/5 model
        model_a = base_pipeline(RANDOM_SEED)
        model_a.fit(x_train_cv, y_train_cv)
        metrics_a = detailed_metrics(
            model_a, x_test_cv, y_test_cv, threshold=None
        )
        row_a = {"fold": fold_idx}
        for k, v in metrics_a.items():
            row_a[f"test_{k}"] = v
        cv_rows_a.append(row_a)

        # iter_6 model
        model_b = base_pipeline(RANDOM_SEED)
        model_b.fit(x_train_cv, y_train_cv)
        oof_scores = np.empty(len(y_train_cv), dtype=float)
        inner_splitter = StratifiedKFold(
            n_splits=INNER_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED
        )
        for inner_train_idx, inner_valid_idx in inner_splitter.split(
            x_train_cv, y_train_cv
        ):
            inner_model = base_pipeline(RANDOM_SEED)
            inner_model.fit(
                take_rows(x_train_cv, inner_train_idx),
                take_rows(y_train_cv, inner_train_idx),
            )
            oof_scores[inner_valid_idx] = positive_decision_scores(
                inner_model, take_rows(x_train_cv, inner_valid_idx)
            )
        best_threshold, _ = find_best_threshold(oof_scores, y_train_cv)
        metrics_b = detailed_metrics(
            model_b, x_test_cv, y_test_cv, threshold=best_threshold
        )
        row_b = {"fold": fold_idx, "best_threshold": best_threshold}
        for k, v in metrics_b.items():
            row_b[f"test_{k}"] = v
        cv_rows_b.append(row_b)

    cv_a = pd.DataFrame(cv_rows_a)
    cv_b = pd.DataFrame(cv_rows_b)
    cv_a.to_csv(
        OUTPUT_DIR / "cv_metrics_iter45.csv", index=False, encoding="utf-8-sig"
    )
    cv_b.to_csv(
        OUTPUT_DIR / "cv_metrics_iter6.csv", index=False, encoding="utf-8-sig"
    )
    summarize_table(cv_a, "cv").to_csv(
        OUTPUT_DIR / "cv_summary_iter45.csv", index=False, encoding="utf-8-sig"
    )
    summarize_table(cv_b, "cv").to_csv(
        OUTPUT_DIR / "cv_summary_iter6.csv", index=False, encoding="utf-8-sig"
    )

    # ── Print key results ──
    print()
    print("=" * 60)
    print("30-SEED REPEATED SPLIT RESULTS")
    print("=" * 60)

    test_metrics_names = [
        "accuracy",
        "macro_f1",
        "b_recall",
        "g_f1",
        "roc_auc",
    ]

    print()
    print(f"{'Metric':<14} {'iter_4/5 mean':>14} {'iter_4/5 std':>14} {'iter_6 mean':>14} {'iter_6 std':>14} {'Δ (i6-i45)':>14}")
    print("-" * 74)
    for metric in test_metrics_names:
        a_mean = repeated[f"iter45_test_{metric}"].mean()
        a_std = repeated[f"iter45_test_{metric}"].std()
        b_mean = repeated[f"iter6_test_{metric}"].mean()
        b_std = repeated[f"iter6_test_{metric}"].std()
        delta = b_mean - a_mean
        print(
            f"{metric:<14} {a_mean:>14.4f} {a_std:>14.4f} "
            f"{b_mean:>14.4f} {b_std:>14.4f} {delta:>+14.4f}"
        )

    print()
    print(f"iter_6 win count (test macro_f1): "
          f"{(repeated['iter6_test_macro_f1'] > repeated['iter45_test_macro_f1']).sum()}/30")
    print(f"iter_6 win count (test b_recall):  "
          f"{(repeated['iter6_test_b_recall'] > repeated['iter45_test_b_recall']).sum()}/30")

    # ── Save full metrics.json ──
    result_metadata = {
        "iteration": "iteration_7",
        "description": (
            "30-seed expanded stability comparison between "
            "iter_4/5 model (default threshold) and iter_6 model (threshold-tuned)"
        ),
        "fixed_params": {
            "kernel": "rbf",
            "C": FIXED_C,
            "gamma": FIXED_GAMMA,
            "class_weight": FIXED_CLASS_WEIGHT,
        },
        "expanded_seeds": EXPANDED_SEEDS,
        "n_expanded_seeds": len(EXPANDED_SEEDS),
        "iter45_30seed_summary": {
            "test_accuracy_mean": float(repeated["iter45_test_accuracy"].mean()),
            "test_accuracy_std": float(repeated["iter45_test_accuracy"].std()),
            "test_macro_f1_mean": float(repeated["iter45_test_macro_f1"].mean()),
            "test_macro_f1_std": float(repeated["iter45_test_macro_f1"].std()),
            "test_b_recall_mean": float(repeated["iter45_test_b_recall"].mean()),
            "test_b_recall_std": float(repeated["iter45_test_b_recall"].std()),
            "test_g_f1_mean": float(repeated["iter45_test_g_f1"].mean()),
            "test_g_f1_std": float(repeated["iter45_test_g_f1"].std()),
            "test_roc_auc_mean": float(repeated["iter45_test_roc_auc"].mean()),
            "test_roc_auc_std": float(repeated["iter45_test_roc_auc"].std()),
        },
        "iter6_30seed_summary": {
            "test_accuracy_mean": float(repeated["iter6_test_accuracy"].mean()),
            "test_accuracy_std": float(repeated["iter6_test_accuracy"].std()),
            "test_macro_f1_mean": float(repeated["iter6_test_macro_f1"].mean()),
            "test_macro_f1_std": float(repeated["iter6_test_macro_f1"].std()),
            "test_b_recall_mean": float(repeated["iter6_test_b_recall"].mean()),
            "test_b_recall_std": float(repeated["iter6_test_b_recall"].std()),
            "test_g_f1_mean": float(repeated["iter6_test_g_f1"].mean()),
            "test_g_f1_std": float(repeated["iter6_test_g_f1"].std()),
            "test_roc_auc_mean": float(repeated["iter6_test_roc_auc"].mean()),
            "test_roc_auc_std": float(repeated["iter6_test_roc_auc"].std()),
        },
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(result_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print()
    print("Output files written to:", str(OUTPUT_DIR))
    print("Done.")
