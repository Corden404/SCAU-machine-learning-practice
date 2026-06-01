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
from sklearn.metrics import accuracy_score, f1_score, make_scorer, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

from exp.ML.common import (
    CV_FOLDS,
    POS_LABEL,
    RANDOM_SEED,
    REPEATED_SEEDS,
    TEST_SIZE,
    binary_target,
    load_ionosphere,
    positive_scores,
    run_experiment,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
RANDOM_SEARCH_ITERATIONS = 220
TOP_CANDIDATES_FOR_REPEATED = 24

N_ESTIMATORS_VALUES = [100, 200, 300]
CRITERION_VALUES = ["gini", "entropy", "log_loss"]
MAX_FEATURES_VALUES = ["sqrt", 0.3, 0.5]
MAX_LEAF_NODES_VALUES = [None, 16, 32]
MIN_SAMPLES_LEAF_VALUES = [1, 2, 4]
MIN_SAMPLES_SPLIT_VALUES = [2, 4, 8]
CCP_ALPHA_VALUES = [0.0, 0.001]
MAX_SAMPLES_VALUES = [None, 0.8, 1.0]


def make_model(seed: int, params: dict[str, Any] | None = None, *, n_jobs: int = 1) -> RandomForestClassifier:
    model = RandomForestClassifier(
        bootstrap=True,
        class_weight=None,
        random_state=seed,
        n_jobs=n_jobs,
    )
    if params:
        model.set_params(**params)
    return model


def build_model(seed: int, best_params: dict[str, Any] | None = None) -> RandomForestClassifier:
    return make_model(seed, best_params, n_jobs=-1)


def score_roc_auc(estimator: RandomForestClassifier, x: pd.DataFrame, y: pd.Series) -> float:
    scores = positive_scores(estimator, x)
    if scores is None:
        return float("nan")
    return float(roc_auc_score(binary_target(pd.Series(y)), scores))


def clean_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json_value(item) for item in value]
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


def parameter_distributions() -> dict[str, list[Any]]:
    return {
        "n_estimators": N_ESTIMATORS_VALUES,
        "criterion": CRITERION_VALUES,
        "max_features": MAX_FEATURES_VALUES,
        "max_leaf_nodes": MAX_LEAF_NODES_VALUES,
        "min_samples_leaf": MIN_SAMPLES_LEAF_VALUES,
        "min_samples_split": MIN_SAMPLES_SPLIT_VALUES,
        "ccp_alpha": CCP_ALPHA_VALUES,
        "max_samples": MAX_SAMPLES_VALUES,
    }


def summarize_repeated_candidate(candidate_id: int, params: dict[str, Any], x: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed in REPEATED_SEEDS:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=seed,
        )
        model = make_model(seed, params, n_jobs=1)
        model.fit(x_train, y_train)
        prediction = pd.Series(model.predict(x_test), index=y_test.index)
        scores = positive_scores(model, x_test)
        row = {
            "candidate_id": candidate_id,
            "seed": seed,
            "test_accuracy": accuracy_score(y_test, prediction),
            "test_macro_f1": f1_score(y_test, prediction, average="macro", zero_division=0),
            "test_f1_g": f1_score(y_test, prediction, pos_label=POS_LABEL, zero_division=0),
            "test_f1_b": f1_score(y_test, prediction, pos_label="b", zero_division=0),
            "test_recall_b": recall_score(y_test, prediction, pos_label="b", zero_division=0),
        }
        if scores is not None:
            row["test_roc_auc"] = roc_auc_score(binary_target(y_test), scores)
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_DIR / f"repeated_candidate_{candidate_id:02d}.csv", index=False, encoding="utf-8-sig")
    summary: dict[str, Any] = {"candidate_id": candidate_id, "params": params}
    for column in [
        "test_accuracy",
        "test_macro_f1",
        "test_f1_g",
        "test_f1_b",
        "test_recall_b",
        "test_roc_auc",
    ]:
        if column in frame:
            summary[f"{column}_mean"] = float(frame[column].mean())
            summary[f"{column}_std"] = float(frame[column].std())
    return summary


def search_and_select() -> tuple[dict[str, Any], dict[str, Any]]:
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
        estimator=make_model(RANDOM_SEED, n_jobs=1),
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
    results.to_csv(OUTPUT_DIR / "local_random_search_results.csv", index=False, encoding="utf-8-sig")

    useful_columns = [
        "rank_test_macro_f1",
        "param_n_estimators",
        "param_criterion",
        "param_max_features",
        "param_max_leaf_nodes",
        "param_min_samples_leaf",
        "param_min_samples_split",
        "param_ccp_alpha",
        "param_max_samples",
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
    results.loc[:, useful_columns].to_csv(OUTPUT_DIR / "local_random_search_summary.csv", index=False, encoding="utf-8-sig")

    repeated_rows: list[dict[str, Any]] = []
    top_candidates = results.head(TOP_CANDIDATES_FOR_REPEATED).reset_index(drop=True)
    for candidate_id, (_, row) in enumerate(top_candidates.iterrows(), start=1):
        params = clean_json_value(row["params"])
        repeated_summary = summarize_repeated_candidate(candidate_id, params, x, y)
        cv_summary = {
            "cv_rank_macro_f1": clean_json_value(row["rank_test_macro_f1"]),
            "cv_mean_macro_f1": clean_json_value(row["mean_test_macro_f1"]),
            "cv_mean_accuracy": clean_json_value(row["mean_test_accuracy"]),
            "cv_mean_f1_g": clean_json_value(row["mean_test_f1_g"]),
            "cv_mean_recall_b": clean_json_value(row["mean_test_recall_b"]),
            "cv_mean_roc_auc": clean_json_value(row["mean_test_roc_auc"]),
            "cv_mean_train_macro_f1": clean_json_value(row["mean_train_macro_f1"]),
        }
        repeated_rows.append({**repeated_summary, **cv_summary})

    repeated = pd.DataFrame(repeated_rows).sort_values(
        by=[
            "test_macro_f1_mean",
            "test_accuracy_mean",
            "test_recall_b_mean",
            "test_f1_g_mean",
            "test_roc_auc_mean",
        ],
        ascending=[False, False, False, False, False],
    )
    repeated.to_csv(OUTPUT_DIR / "repeated_candidate_summary.csv", index=False, encoding="utf-8-sig")
    selected = repeated.iloc[0].to_dict()
    selected_params = clean_json_value(selected["params"])

    selected_model = make_model(RANDOM_SEED, selected_params, n_jobs=-1)
    selected_model.fit(x, y)
    feature_importance = save_feature_importance(selected_model, x)

    cv_best_params = {key: clean_json_value(value) for key, value in search.best_params_.items()}
    best_info = {
        "search_method": "RandomizedSearchCV plus repeated-split reranking",
        "n_iter": RANDOM_SEARCH_ITERATIONS,
        "top_candidates_for_repeated": TOP_CANDIDATES_FOR_REPEATED,
        "cv_folds": CV_FOLDS,
        "repeated_seeds": REPEATED_SEEDS,
        "candidate_n_estimators_values": N_ESTIMATORS_VALUES,
        "candidate_criterion_values": CRITERION_VALUES,
        "candidate_max_features_values": MAX_FEATURES_VALUES,
        "candidate_max_leaf_nodes_values": MAX_LEAF_NODES_VALUES,
        "candidate_min_samples_leaf_values": MIN_SAMPLES_LEAF_VALUES,
        "candidate_min_samples_split_values": MIN_SAMPLES_SPLIT_VALUES,
        "candidate_ccp_alpha_values": CCP_ALPHA_VALUES,
        "candidate_max_samples_values": MAX_SAMPLES_VALUES,
        "fixed_bootstrap": True,
        "fixed_class_weight": None,
        "cv_selection_metric": "mean_test_macro_f1",
        "final_selection_metric": "mean repeated test_macro_f1 among top CV candidates",
        "cv_best_score_macro_f1": float(search.best_score_),
        "cv_best_params": cv_best_params,
        "selected_params": selected_params,
        "selected_repeated_summary": clean_record(selected),
        "selected_feature_importance": feature_importance,
        "cv_best_row": clean_record(results.iloc[0].to_dict()),
    }
    (OUTPUT_DIR / "model_selection.json").write_text(json.dumps(best_info, ensure_ascii=False, indent=2), encoding="utf-8")
    return selected_params, best_info


if __name__ == "__main__":
    best_params, best_info = search_and_select()
    run_experiment(
        model_name="RandomForest",
        iteration_name="iteration_4",
        build_model=lambda seed: build_model(seed, best_params),
        output_dir=OUTPUT_DIR,
        experiment_note=(
            "RandomForest local RandomizedSearchCV around iteration_2/3, followed by "
            "repeated-split reranking of the top CV candidates. "
            f"Selected params: {best_info['selected_params']}."
        ),
        run_cv=True,
        run_repeated=True,
    )

    metrics_path = OUTPUT_DIR / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["model_selection"] = best_info
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(best_info, ensure_ascii=False, indent=2))
