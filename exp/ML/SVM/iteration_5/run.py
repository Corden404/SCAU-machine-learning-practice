from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import f1_score, make_scorer, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from exp.ML.common import CV_FOLDS, RANDOM_SEED, binary_target, load_ionosphere, positive_scores, run_experiment


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
C_VALUES = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
GAMMA_VALUES = ["scale", 0.01, 0.03, 0.1, 0.3, 1.0]
CLASS_WEIGHT_VALUES = [None, "balanced"]
POLY_DEGREES = [2, 3, 4]
POLY_COEF0_VALUES = [0.0, 0.5, 1.0]


def base_pipeline(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("standardize", StandardScaler()),
            ("classifier", SVC(random_state=seed)),
        ]
    )


def build_model(seed: int, best_params: dict[str, Any] | None = None) -> Pipeline:
    model = base_pipeline(seed)
    if best_params:
        model.set_params(**best_params)
    return model


def score_roc_auc(estimator: Pipeline, x: pd.DataFrame, y: pd.Series) -> float:
    scores = positive_scores(estimator, x)
    if scores is None:
        return float("nan")
    return float(roc_auc_score(binary_target(pd.Series(y)), scores))


def clean_json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def clean_record(row: dict[str, Any]) -> dict[str, Any]:
    return {str(key): clean_json_value(value) for key, value in row.items()}


def save_best_by_kernel(results: pd.DataFrame) -> None:
    rows = []
    for kernel, group in results.groupby("param_classifier__kernel", dropna=False):
        rows.append(group.sort_values(["rank_test_macro_f1", "mean_test_accuracy"]).iloc[0])
    pd.DataFrame(rows).sort_values("rank_test_macro_f1").to_csv(
        OUTPUT_DIR / "best_by_kernel.csv",
        index=False,
        encoding="utf-8-sig",
    )


def grid_search() -> tuple[dict[str, Any], dict[str, Any]]:
    x, y = load_ionosphere()
    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    scoring = {
        "macro_f1": make_scorer(f1_score, average="macro", pos_label=None, zero_division=0),
        "f1_b": make_scorer(f1_score, pos_label="b", zero_division=0),
        "recall_b": make_scorer(recall_score, pos_label="b", zero_division=0),
        "accuracy": "accuracy",
        "roc_auc": score_roc_auc,
    }
    param_grid = [
        {
            "classifier__kernel": ["linear"],
            "classifier__C": C_VALUES,
            "classifier__class_weight": CLASS_WEIGHT_VALUES,
        },
        {
            "classifier__kernel": ["rbf"],
            "classifier__C": C_VALUES,
            "classifier__gamma": GAMMA_VALUES,
            "classifier__class_weight": CLASS_WEIGHT_VALUES,
        },
        {
            "classifier__kernel": ["poly"],
            "classifier__C": C_VALUES,
            "classifier__gamma": GAMMA_VALUES,
            "classifier__degree": POLY_DEGREES,
            "classifier__coef0": POLY_COEF0_VALUES,
            "classifier__class_weight": CLASS_WEIGHT_VALUES,
        },
    ]
    search = GridSearchCV(
        estimator=base_pipeline(RANDOM_SEED),
        param_grid=param_grid,
        scoring=scoring,
        refit="macro_f1",
        cv=splitter,
        n_jobs=-1,
        return_train_score=True,
    )
    search.fit(x, y)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(search.cv_results_).sort_values(
        by=["rank_test_macro_f1", "mean_test_recall_b", "mean_test_accuracy"],
        ascending=[True, False, False],
    )
    results.to_csv(OUTPUT_DIR / "grid_search_results.csv", index=False, encoding="utf-8-sig")
    save_best_by_kernel(results)

    useful_columns = [
        "rank_test_macro_f1",
        "param_classifier__kernel",
        "param_classifier__C",
        "param_classifier__gamma",
        "param_classifier__degree",
        "param_classifier__coef0",
        "param_classifier__class_weight",
        "mean_test_macro_f1",
        "std_test_macro_f1",
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
    best_info = {
        "candidate_C_values": C_VALUES,
        "candidate_gamma_values": GAMMA_VALUES,
        "candidate_class_weight_values": CLASS_WEIGHT_VALUES,
        "candidate_poly_degrees": POLY_DEGREES,
        "candidate_poly_coef0_values": POLY_COEF0_VALUES,
        "candidate_kernels": ["linear", "rbf", "poly"],
        "selection_metric": "mean_test_macro_f1",
        "candidate_count": int(len(results)),
        "best_score_macro_f1": float(search.best_score_),
        "best_params": best_params,
        "best_row": clean_record(results.iloc[0].to_dict()),
    }
    (OUTPUT_DIR / "model_selection.json").write_text(json.dumps(best_info, ensure_ascii=False, indent=2), encoding="utf-8")
    return best_params, best_info


if __name__ == "__main__":
    best_params, best_info = grid_search()
    run_experiment(
        model_name="SVM",
        iteration_name="iteration_5",
        build_model=lambda seed: build_model(seed, best_params),
        output_dir=OUTPUT_DIR,
        experiment_note=(
            "Exploratory full SVM search with StandardScaler, linear/RBF/poly kernels, "
            "C, gamma, class_weight, poly degree, and poly coef0. "
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
