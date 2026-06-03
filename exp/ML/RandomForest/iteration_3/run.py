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
from sklearn.metrics import f1_score, make_scorer, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from exp.ML.common import CV_FOLDS, RANDOM_SEED, binary_target, load_ionosphere, positive_scores, run_experiment


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
RANDOM_SEARCH_ITERATIONS = 160

N_ESTIMATORS_VALUES = [100, 200, 300, 500]
CRITERION_VALUES = ["gini", "entropy", "log_loss"]
MAX_DEPTH_VALUES = [None, 3, 4, 5, 6, 8, 10, 12, 16]
MIN_SAMPLES_SPLIT_VALUES = [2, 4, 8, 12, 20]
MIN_SAMPLES_LEAF_VALUES = [1, 2, 4, 8, 12]
MAX_FEATURES_VALUES = ["sqrt", "log2", None, 0.3, 0.5, 0.7]
MAX_LEAF_NODES_VALUES = [None, 8, 16, 32, 64]
MIN_IMPURITY_DECREASE_VALUES = [0.0, 0.001, 0.005, 0.01]
CCP_ALPHA_VALUES = [0.0, 0.001, 0.005, 0.01]
CLASS_WEIGHT_VALUES = [None, "balanced", "balanced_subsample"]
MAX_SAMPLES_VALUES = [None, 0.6, 0.8, 1.0]


def base_model(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        random_state=seed,
        n_jobs=1,
    )


def build_model(seed: int, best_params: dict[str, Any] | None = None) -> RandomForestClassifier:
    model = RandomForestClassifier(
        random_state=seed,
        n_jobs=-1,
    )
    if best_params:
        model.set_params(**best_params)
    return model


def score_roc_auc(estimator: RandomForestClassifier, x: pd.DataFrame, y: pd.Series) -> float:
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


def save_feature_importance(model: RandomForestClassifier, x: pd.DataFrame) -> dict[str, Any]:
    feature_importances = sorted(zip(x.columns, model.feature_importances_), key=lambda item: item[1], reverse=True)
    summary = {
        "top_features": [
            {"feature": str(feature), "importance": float(importance)}
            for feature, importance in feature_importances[:10]
            if float(importance) > 0
        ]
    }
    (OUTPUT_DIR / "feature_importance.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parameter_distributions() -> list[dict[str, list[Any]]]:
    common = {
        "n_estimators": N_ESTIMATORS_VALUES,
        "criterion": CRITERION_VALUES,
        "max_depth": MAX_DEPTH_VALUES,
        "min_samples_split": MIN_SAMPLES_SPLIT_VALUES,
        "min_samples_leaf": MIN_SAMPLES_LEAF_VALUES,
        "max_features": MAX_FEATURES_VALUES,
        "max_leaf_nodes": MAX_LEAF_NODES_VALUES,
        "min_impurity_decrease": MIN_IMPURITY_DECREASE_VALUES,
        "ccp_alpha": CCP_ALPHA_VALUES,
        "class_weight": CLASS_WEIGHT_VALUES,
    }
    return [
        {
            **common,
            "bootstrap": [True],
            "max_samples": MAX_SAMPLES_VALUES,
        },
        {
            **common,
            "bootstrap": [False],
        },
    ]


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
    search = RandomizedSearchCV(
        estimator=base_model(RANDOM_SEED),
        param_distributions=parameter_distributions(),
        n_iter=RANDOM_SEARCH_ITERATIONS,
        scoring=scoring,
        refit="macro_f1",
        cv=splitter,
        n_jobs=4,
        random_state=RANDOM_SEED,
        return_train_score=True,
    )
    search.fit(x, y)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(search.cv_results_).sort_values(
        by=["rank_test_macro_f1", "mean_test_recall_b", "mean_test_f1_g", "mean_test_accuracy"],
        ascending=[True, False, False, False],
    )
    results.to_csv(OUTPUT_DIR / "random_search_results.csv", index=False, encoding="utf-8-sig")

    useful_columns = [
        "rank_test_macro_f1",
        "param_n_estimators",
        "param_criterion",
        "param_max_depth",
        "param_min_samples_split",
        "param_min_samples_leaf",
        "param_max_features",
        "param_bootstrap",
        "param_max_samples",
        "param_max_leaf_nodes",
        "param_min_impurity_decrease",
        "param_ccp_alpha",
        "param_class_weight",
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
    results.loc[:, useful_columns].to_csv(OUTPUT_DIR / "random_search_summary.csv", index=False, encoding="utf-8-sig")

    best_params = {key: clean_json_value(value) for key, value in search.best_params_.items()}
    feature_importance = save_feature_importance(search.best_estimator_, x)
    best_info = {
        "search_method": "RandomizedSearchCV",
        "n_iter": RANDOM_SEARCH_ITERATIONS,
        "cv_folds": CV_FOLDS,
        "candidate_n_estimators_values": N_ESTIMATORS_VALUES,
        "candidate_criterion_values": CRITERION_VALUES,
        "candidate_max_depth_values": MAX_DEPTH_VALUES,
        "candidate_min_samples_split_values": MIN_SAMPLES_SPLIT_VALUES,
        "candidate_min_samples_leaf_values": MIN_SAMPLES_LEAF_VALUES,
        "candidate_max_features_values": MAX_FEATURES_VALUES,
        "candidate_bootstrap_values": [True, False],
        "candidate_max_samples_values": MAX_SAMPLES_VALUES,
        "candidate_max_leaf_nodes_values": MAX_LEAF_NODES_VALUES,
        "candidate_min_impurity_decrease_values": MIN_IMPURITY_DECREASE_VALUES,
        "candidate_ccp_alpha_values": CCP_ALPHA_VALUES,
        "candidate_class_weight_values": CLASS_WEIGHT_VALUES,
        "selection_metric": "mean_test_macro_f1",
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
        model_name="RandomForest",
        iteration_name="iteration_3",
        build_model=lambda seed: build_model(seed, best_params),
        output_dir=OUTPUT_DIR,
        experiment_note=(
            "RandomForest broad RandomizedSearchCV over major hyperparameters, "
            "including tree count, split criterion, depth, leaf/split size, feature sampling, "
            "bootstrap/max_samples, max_leaf_nodes, impurity threshold, ccp_alpha, and class_weight. "
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
