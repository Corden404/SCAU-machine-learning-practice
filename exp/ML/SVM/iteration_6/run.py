from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp.ML.common import (
    CV_FOLDS,
    LABELS,
    POS_LABEL,
    RANDOM_SEED,
    REPEATED_SEEDS,
    TEST_SIZE,
    binary_target,
    load_ionosphere,
    run_experiment,
    summarize_table,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
FIXED_C = 3.0
FIXED_GAMMA = 0.03
FIXED_CLASS_WEIGHT = "balanced"
INNER_CV_FOLDS = 5


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


def metrics_for_threshold(scores: np.ndarray, y: pd.Series, threshold: float) -> dict[str, float]:
    prediction = pd.Series(np.where(scores >= threshold, POS_LABEL, "b"), index=y.index)
    return {
        "threshold": float(threshold),
        "macro_f1": f1_score(y, prediction, average="macro", pos_label=None, zero_division=0),
        "accuracy": accuracy_score(y, prediction),
        "b_recall": recall_score(y, prediction, pos_label="b", zero_division=0),
        "b_f1": f1_score(y, prediction, pos_label="b", zero_division=0),
        "g_recall": recall_score(y, prediction, pos_label=POS_LABEL, zero_division=0),
        "g_f1": f1_score(y, prediction, pos_label=POS_LABEL, zero_division=0),
    }


def threshold_search(scores: np.ndarray, y: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = [metrics_for_threshold(scores, y, threshold) for threshold in threshold_candidates(scores)]
    results = pd.DataFrame(rows).sort_values(
        by=["macro_f1", "b_recall", "g_f1", "accuracy"],
        ascending=[False, False, False, False],
    )
    return results.reset_index(drop=True), results.iloc[0].to_dict()


def oof_scores(x: pd.DataFrame, y: pd.Series, seed: int, cv_folds: int = INNER_CV_FOLDS) -> np.ndarray:
    scores = np.empty(len(y), dtype=float)
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    for train_index, valid_index in splitter.split(x, y):
        model = base_pipeline(seed)
        model.fit(take_rows(x, train_index), take_rows(y, train_index))
        scores[valid_index] = positive_decision_scores(model, take_rows(x, valid_index))
    return scores


class ThresholdTunedSVC(BaseEstimator, ClassifierMixin):
    def __init__(self, random_state: int = RANDOM_SEED, inner_cv_folds: int = INNER_CV_FOLDS):
        self.random_state = random_state
        self.inner_cv_folds = inner_cv_folds

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "ThresholdTunedSVC":
        y_series = pd.Series(y, index=getattr(y, "index", None)).astype(str)
        self.classes_ = np.asarray(LABELS)
        training_scores = oof_scores(x, y_series, self.random_state, self.inner_cv_folds)
        threshold_results, best = threshold_search(training_scores, y_series)
        default_metrics = metrics_for_threshold(training_scores, y_series, 0.0)

        self.threshold_ = float(best["threshold"])
        self.threshold_selection_metrics_ = {str(key): clean_json_value(value) for key, value in best.items()}
        self.default_threshold_metrics_ = {str(key): clean_json_value(value) for key, value in default_metrics.items()}
        self.threshold_candidate_count_ = int(len(threshold_results))
        self.model_ = base_pipeline(self.random_state)
        self.model_.fit(x, y_series)
        return self

    def decision_function(self, x: pd.DataFrame) -> np.ndarray:
        return positive_decision_scores(self.model_, x)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        scores = self.decision_function(x)
        return np.where(scores >= self.threshold_, POS_LABEL, "b")


def build_model(seed: int) -> ThresholdTunedSVC:
    return ThresholdTunedSVC(random_state=seed)


def clean_json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def detailed_metric_block(model: ThresholdTunedSVC, x: pd.DataFrame, y: pd.Series, prefix: str) -> dict[str, float]:
    prediction = pd.Series(model.predict(x), index=y.index)
    scores = pd.Series(model.decision_function(x), index=y.index)
    return {
        f"{prefix}_accuracy": accuracy_score(y, prediction),
        f"{prefix}_macro_f1": f1_score(y, prediction, average="macro", pos_label=None, zero_division=0),
        f"{prefix}_b_recall": recall_score(y, prediction, pos_label="b", zero_division=0),
        f"{prefix}_b_f1": f1_score(y, prediction, pos_label="b", zero_division=0),
        f"{prefix}_g_recall": recall_score(y, prediction, pos_label=POS_LABEL, zero_division=0),
        f"{prefix}_g_f1": f1_score(y, prediction, pos_label=POS_LABEL, zero_division=0),
        f"{prefix}_roc_auc": roc_auc_score(binary_target(y), scores),
    }


def evaluate_detailed_split(
    iteration_name: str,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    seed: int,
) -> dict[str, Any]:
    model = build_model(seed)
    model.fit(x_train, y_train)
    row: dict[str, Any] = {
        "model": "SVM",
        "iteration": iteration_name,
        "seed": seed,
        "selected_threshold": model.threshold_,
        "threshold_train_macro_f1": model.threshold_selection_metrics_["macro_f1"],
        "threshold_default_macro_f1": model.default_threshold_metrics_["macro_f1"],
    }
    row.update(detailed_metric_block(model, x_train, y_train, "train"))
    row.update(detailed_metric_block(model, x_test, y_test, "test"))
    return row


def run_detailed_outputs(x: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )
    single = pd.DataFrame([evaluate_detailed_split("iteration_6", x_train, x_test, y_train, y_test, RANDOM_SEED)])
    single.to_csv(OUTPUT_DIR / "detailed_single_split_metrics.csv", index=False, encoding="utf-8-sig")

    cv_rows = []
    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    for fold_index, (train_index, test_index) in enumerate(splitter.split(x, y), start=1):
        row = evaluate_detailed_split(
            "iteration_6",
            take_rows(x, train_index),
            take_rows(x, test_index),
            take_rows(y, train_index),
            take_rows(y, test_index),
            RANDOM_SEED,
        )
        row["fold"] = fold_index
        cv_rows.append(row)
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(OUTPUT_DIR / "detailed_cv_metrics.csv", index=False, encoding="utf-8-sig")
    summarize_table(cv.drop(columns=["fold"], errors="ignore"), "cv").to_csv(
        OUTPUT_DIR / "detailed_cv_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    repeated_rows = []
    for seed in REPEATED_SEEDS:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=seed,
        )
        repeated_rows.append(evaluate_detailed_split("iteration_6", x_train, x_test, y_train, y_test, seed))
    repeated = pd.DataFrame(repeated_rows)
    repeated.to_csv(OUTPUT_DIR / "detailed_repeated_split_metrics.csv", index=False, encoding="utf-8-sig")
    summarize_table(repeated, "repeated").to_csv(
        OUTPUT_DIR / "detailed_repeated_split_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return {
        "single_split": {key: clean_json_value(value) for key, value in single.iloc[0].to_dict().items()},
        "cv_summary": summarize_table(cv.drop(columns=["fold"], errors="ignore"), "cv").to_dict(orient="records"),
        "repeated_summary": summarize_table(repeated, "repeated").to_dict(orient="records"),
    }


def save_reference_threshold_search(x: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    scores = oof_scores(x, y, RANDOM_SEED)
    results, best = threshold_search(scores, y)
    default_metrics = metrics_for_threshold(scores, y, 0.0)
    results.to_csv(OUTPUT_DIR / "threshold_search_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([best, default_metrics]).to_csv(
        OUTPUT_DIR / "threshold_search_best_vs_default.csv",
        index=False,
        encoding="utf-8-sig",
    )
    info = {
        "fixed_model": {
            "pipeline": "StandardScaler + RBF SVC",
            "C": FIXED_C,
            "gamma": FIXED_GAMMA,
            "class_weight": FIXED_CLASS_WEIGHT,
        },
        "threshold_selection": "5-fold out-of-fold decision_function scores on the training data of each fit",
        "reference_full_data_oof": {
            "candidate_count": int(len(results)),
            "best_threshold_metrics": {str(key): clean_json_value(value) for key, value in best.items()},
            "default_threshold_metrics": {str(key): clean_json_value(value) for key, value in default_metrics.items()},
        },
    }
    (OUTPUT_DIR / "threshold_selection.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    x, y = load_ionosphere()
    threshold_info = save_reference_threshold_search(x, y)
    run_experiment(
        model_name="SVM",
        iteration_name="iteration_6",
        build_model=build_model,
        output_dir=OUTPUT_DIR,
        experiment_note=(
            "Threshold-tuned SVM. The model is fixed to StandardScaler + RBF SVC "
            "(C=3.0, gamma=0.03, class_weight='balanced'), and the decision threshold "
            "is selected by inner 5-fold out-of-fold macro F1."
        ),
        run_cv=True,
        run_repeated=True,
    )
    detailed_info = run_detailed_outputs(x, y)

    metrics_path = OUTPUT_DIR / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["threshold_selection"] = threshold_info
    metrics["detailed_metric_outputs"] = {
        "single_split": "detailed_single_split_metrics.csv",
        "cv_metrics": "detailed_cv_metrics.csv",
        "cv_summary": "detailed_cv_summary.csv",
        "repeated_split_metrics": "detailed_repeated_split_metrics.csv",
        "repeated_split_summary": "detailed_repeated_split_summary.csv",
    }
    metrics["detailed_metric_summary"] = detailed_info
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics["threshold_selection"], ensure_ascii=False, indent=2))
