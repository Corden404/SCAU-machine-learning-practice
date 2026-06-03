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

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from exp.ML.common import (
    LABELS,
    POS_LABEL,
    REPEATED_SEEDS,
    TEST_SIZE,
    binary_target,
    load_ionosphere,
    positive_scores,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

BEST_PARAMS: dict[str, Any] = {
    "n_estimators": 50,
    "learning_rate": 0.5,
    "estimator__max_depth": 2,
    "estimator__min_samples_leaf": 2,
    "estimator__criterion": "entropy",
    "estimator__class_weight": None,
}

THRESHOLDS = np.linspace(0.10, 0.90, 41)


def build_iter4(seed: int) -> AdaBoostClassifier:
    model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(random_state=seed),
        random_state=seed,
    )
    model.set_params(**BEST_PARAMS)
    return model


def build_default(seed: int) -> AdaBoostClassifier:
    return AdaBoostClassifier(random_state=seed)


def evaluate_thresholds(
    model_builder: Any,
    x: pd.DataFrame,
    y: pd.Series,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in REPEATED_SEEDS:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=TEST_SIZE, stratify=y, random_state=seed,
        )
        model = model_builder(seed)
        model.fit(x_train, y_train)
        classes = list(model.classes_)
        proba_g = model.predict_proba(x_test)[:, classes.index(POS_LABEL)]

        for threshold in thresholds:
            pred = pd.Series(
                np.where(proba_g >= threshold, POS_LABEL, "b"),
                index=y_test.index,
            )
            rows.append({
                "seed": seed,
                "threshold": round(float(threshold), 4),
                "accuracy": float(accuracy_score(y_test, pred)),
                "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
                "f1_g": float(f1_score(y_test, pred, pos_label=POS_LABEL, zero_division=0)),
                "f1_b": float(f1_score(y_test, pred, pos_label="b", zero_division=0)),
                "recall_g": float(recall_score(y_test, pred, pos_label=POS_LABEL, zero_division=0)),
                "recall_b": float(recall_score(y_test, pred, pos_label="b", zero_division=0)),
                "precision_g": float(precision_score(y_test, pred, pos_label=POS_LABEL, zero_division=0)),
                "precision_b": float(precision_score(y_test, pred, pos_label="b", zero_division=0)),
            })
    return pd.DataFrame(rows)


def summarize_by_threshold(frame: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "accuracy", "macro_f1", "f1_g", "f1_b",
        "recall_g", "recall_b", "precision_g", "precision_b",
    ]
    grouped = frame.groupby("threshold")[metric_cols]
    mean = grouped.mean()
    std = grouped.std()
    result = mean.reset_index()
    for col in metric_cols:
        result[f"{col}_std"] = std[col].values
    return result


def find_best_thresholds(summary: pd.DataFrame) -> dict[str, Any]:
    best: dict[str, Any] = {}
    for metric in ["macro_f1", "accuracy", "f1_g", "recall_b"]:
        idx = int(summary[metric].idxmax())
        best[metric] = {
            "threshold": float(summary.loc[idx, "threshold"]),
            "value": float(summary.loc[idx, metric]),
            "accuracy": float(summary.loc[idx, "accuracy"]),
            "macro_f1": float(summary.loc[idx, "macro_f1"]),
            "f1_g": float(summary.loc[idx, "f1_g"]),
            "f1_b": float(summary.loc[idx, "f1_b"]),
            "recall_b": float(summary.loc[idx, "recall_b"]),
            "recall_g": float(summary.loc[idx, "recall_g"]),
        }

    default_row = summary[summary["threshold"] == 0.50]
    if not default_row.empty:
        idx = int(default_row.index[0])
        best["default_0.5"] = {
            "threshold": 0.5,
            "accuracy": float(summary.loc[idx, "accuracy"]),
            "macro_f1": float(summary.loc[idx, "macro_f1"]),
            "f1_g": float(summary.loc[idx, "f1_g"]),
            "f1_b": float(summary.loc[idx, "f1_b"]),
            "recall_b": float(summary.loc[idx, "recall_b"]),
            "recall_g": float(summary.loc[idx, "recall_g"]),
        }
    return best


if __name__ == "__main__":
    x, y = load_ionosphere()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, Any] = {}

    for model_key, model_builder, model_label in [
        ("iter4_search", build_iter4, "AdaBoost iter4 (全参数搜索)"),
        ("iter3_default", build_default, "AdaBoost iter3 (默认参数)"),
    ]:
        print(f"\n{'=' * 60}")
        print(f"阈值分析: {model_label}")
        print(f"{'=' * 60}")

        raw = evaluate_thresholds(model_builder, x, y, THRESHOLDS)
        raw.to_csv(OUTPUT_DIR / f"threshold_raw_{model_key}.csv", index=False, encoding="utf-8-sig")

        summary = summarize_by_threshold(raw)
        summary.to_csv(OUTPUT_DIR / f"threshold_summary_{model_key}.csv", index=False, encoding="utf-8-sig")

        best = find_best_thresholds(summary)

        print(f"\n默认阈值 (0.5):")
        for k, v in best["default_0.5"].items():
            if k != "threshold" and isinstance(v, float):
                print(f"  {k}: {v:.4f}")

        print(f"\n各指标最佳阈值:")
        for metric in ["macro_f1", "accuracy", "f1_g", "recall_b"]:
            info = best[metric]
            print(f"  最优 {metric}: 阈值={info['threshold']:.2f}, 值={info['value']:.4f} "
                  f"(此时 accuracy={info['accuracy']:.4f}, macro_f1={info['macro_f1']:.4f})")

        all_results[model_key] = {"label": model_label, "best_thresholds": best}

    (OUTPUT_DIR / "threshold_analysis.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"\n结果已保存到 {OUTPUT_DIR}")
