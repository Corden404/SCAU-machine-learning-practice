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
from sklearn.metrics import f1_score, make_scorer, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from exp.ML.common import CV_FOLDS, RANDOM_SEED, binary_target, load_ionosphere, positive_scores, run_experiment


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
N_ESTIMATORS_VALUES = [25, 50, 100, 200]
LEARNING_RATE_VALUES = [0.1, 0.5, 1.0, 1.5]
BASE_MAX_DEPTH_VALUES = [1, 2, 3]
BASE_MIN_SAMPLES_LEAF_VALUES = [1, 2, 5]
BASE_CRITERION_VALUES = ["gini", "entropy"]
BASE_CLASS_WEIGHT_VALUES = [None, "balanced"]


def base_model(seed: int) -> AdaBoostClassifier:
    return AdaBoostClassifier(
        estimator=DecisionTreeClassifier(random_state=seed),
        random_state=seed,
    )


def build_model(seed: int, best_params: dict[str, Any] | None = None) -> AdaBoostClassifier:
    model = base_model(seed)
    if best_params:
        model.set_params(**best_params)
    return model


def score_roc_auc(estimator: AdaBoostClassifier, x: pd.DataFrame, y: pd.Series) -> float:
    scores = positive_scores(estimator, x)
    if scores is None:
        return float("nan")
    return float(roc_auc_score(binary_target(pd.Series(y)), scores))


def clean_json_value(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def clean_record(row: dict[str, Any]) -> dict[str, Any]:
    return {str(key): clean_json_value(value) for key, value in row.items()}


def save_feature_importance(model: AdaBoostClassifier, x: pd.DataFrame) -> dict[str, Any]:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        summary = {"top_features": []}
    else:
        feature_importances = sorted(zip(x.columns, importances), key=lambda item: item[1], reverse=True)
        summary = {
            "top_features": [
                {"feature": str(feature), "importance": float(importance)}
                for feature, importance in feature_importances[:10]
                if float(importance) > 0
            ]
        }
    (OUTPUT_DIR / "feature_importance.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def grid_search() -> tuple[dict[str, Any], dict[str, Any]]:
    x, y = load_ionosphere()
    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    scoring = {
        "macro_f1": make_scorer(f1_score, average="macro", pos_label=None, zero_division=0),
        "f1_g": make_scorer(f1_score, pos_label="g", zero_division=0),
        "f1_b": make_scorer(f1_score, pos_label="b", zero_division=0),
        "recall_b": make_scorer(recall_score, pos_label="b", zero_division=0),
        "accuracy": "accuracy",
        "roc_auc": score_roc_auc,
    }
    param_grid = {
        "n_estimators": N_ESTIMATORS_VALUES,
        "learning_rate": LEARNING_RATE_VALUES,
        "estimator__max_depth": BASE_MAX_DEPTH_VALUES,
        "estimator__min_samples_leaf": BASE_MIN_SAMPLES_LEAF_VALUES,
        "estimator__criterion": BASE_CRITERION_VALUES,
        "estimator__class_weight": BASE_CLASS_WEIGHT_VALUES,
    }
    search = GridSearchCV(
        estimator=base_model(RANDOM_SEED),
        param_grid=param_grid,
        scoring=scoring,
        refit="macro_f1",
        cv=splitter,
        n_jobs=4,
        return_train_score=True,
    )
    search.fit(x, y)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(search.cv_results_).sort_values(
        by=["rank_test_macro_f1", "mean_test_recall_b", "mean_test_f1_g", "mean_test_accuracy"],
        ascending=[True, False, False, False],
    )
    results.to_csv(OUTPUT_DIR / "grid_search_results.csv", index=False, encoding="utf-8-sig")

    useful_columns = [
        "rank_test_macro_f1",
        "param_n_estimators",
        "param_learning_rate",
        "param_estimator__max_depth",
        "param_estimator__min_samples_leaf",
        "param_estimator__criterion",
        "param_estimator__class_weight",
        "mean_test_macro_f1",
        "std_test_macro_f1",
        "mean_test_f1_g",
        "std_test_f1_g",
        "mean_test_f1_b",
        "std_test_f1_b",
        "mean_test_recall_b",
        "std_test_recall_b",
        "mean_test_accuracy",
        "std_test_accuracy",
        "mean_test_roc_auc",
        "std_test_roc_auc",
        "mean_train_macro_f1",
        "std_train_macro_f1",
    ]
    results.loc[:, useful_columns].to_csv(OUTPUT_DIR / "grid_search_summary.csv", index=False, encoding="utf-8-sig")

    best_params = {key: clean_json_value(value) for key, value in search.best_params_.items()}
    feature_importance = save_feature_importance(search.best_estimator_, x)
    best_info = {
        "candidate_n_estimators_values": N_ESTIMATORS_VALUES,
        "candidate_learning_rate_values": LEARNING_RATE_VALUES,
        "candidate_base_max_depth_values": BASE_MAX_DEPTH_VALUES,
        "candidate_base_min_samples_leaf_values": BASE_MIN_SAMPLES_LEAF_VALUES,
        "candidate_base_criterion_values": BASE_CRITERION_VALUES,
        "candidate_base_class_weight_values": BASE_CLASS_WEIGHT_VALUES,
        "selection_metric": "mean_test_macro_f1",
        "candidate_count": int(len(results)),
        "best_score_macro_f1": float(search.best_score_),
        "best_params": best_params,
        "best_feature_importance": feature_importance,
        "best_row": clean_record(results.iloc[0].to_dict()),
    }
    (OUTPUT_DIR / "model_selection.json").write_text(json.dumps(best_info, ensure_ascii=False, indent=2), encoding="utf-8")
    return best_params, best_info


if __name__ == "__main__":
    best_params, best_info = grid_search()
    run_experiment(
        model_name="AdaBoost",
        iteration_name="iteration_4",
        build_model=lambda seed: build_model(seed, best_params),
        output_dir=OUTPUT_DIR,
        experiment_note=(
            "AdaBoost full GridSearchCV over n_estimators, learning_rate, and "
            "DecisionTree base-estimator depth, min_samples_leaf, criterion, class_weight. "
            f"Best params: {best_info['best_params']}."
        ),
        run_cv=True,
        run_repeated=True,
    )

    metrics_path = OUTPUT_DIR / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["model_selection"] = best_info
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(best_info, ensure_ascii=False, indent=2))
