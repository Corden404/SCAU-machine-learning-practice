"""SVM iteration_8 — RobustScaler + PCA preprocessing.

Pipeline variants:
  A: RobustScaler → SVC           (isolate scaler change)
  B: RobustScaler → PCA → SVC     (scaler + dimensionality reduction)

Baseline: iteration_5 = StandardScaler → SVC (C=3.0, gamma=0.03, balanced)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, make_scorer, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

from exp.ML.common import (
    CV_FOLDS,
    POS_LABEL,
    RANDOM_SEED,
    binary_target,
    load_ionosphere,
    positive_scores,
    run_experiment,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

C_VALUES = [1.0, 3.0, 10.0]
GAMMA_VALUES = [0.01, 0.03, 0.1]
CLASS_WEIGHT_VALUES = ["balanced"]
PCA_VARIANCES = [0.90, 0.95, 0.99]


def pipeline_robust(seed: int) -> Pipeline:
    return Pipeline([
        ("robust", RobustScaler()),
        ("classifier", SVC(random_state=seed)),
    ])


def pipeline_robust_pca(seed: int) -> Pipeline:
    return Pipeline([
        ("robust", RobustScaler()),
        ("pca", PCA(random_state=seed)),
        ("classifier", SVC(random_state=seed)),
    ])


def build_model(seed: int, best_params: dict[str, Any] | None = None) -> Pipeline:
    if best_params and "pca__n_components" in best_params:
        model = pipeline_robust_pca(seed)
    else:
        model = pipeline_robust(seed)
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
    if hasattr(value, "get_params"):
        return str(value)
    return value


def clean_record(row: dict[str, Any]) -> dict[str, Any]:
    return {str(key): clean_json_value(value) for key, value in row.items()}


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
        # variant A: RobustScaler only
        {
            "robust": [RobustScaler()],
            "classifier__kernel": ["rbf"],
            "classifier__C": C_VALUES,
            "classifier__gamma": GAMMA_VALUES,
            "classifier__class_weight": CLASS_WEIGHT_VALUES,
        },
        # variant B: RobustScaler + PCA
        {
            "robust": [RobustScaler()],
            "pca__n_components": PCA_VARIANCES,
            "classifier__kernel": ["rbf"],
            "classifier__C": C_VALUES,
            "classifier__gamma": GAMMA_VALUES,
            "classifier__class_weight": CLASS_WEIGHT_VALUES,
        },
    ]

    search = GridSearchCV(
        estimator=pipeline_robust_pca(RANDOM_SEED),
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

    useful_columns = [
        "rank_test_macro_f1",
        "param_classifier__C",
        "param_classifier__gamma",
        "param_classifier__class_weight",
        "param_pca__n_components",
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

    best_params = {key: clean_json_value(value) for key, value in search.best_params_.items()
                   if key != "robust"}

    # determine pipeline variant
    has_pca = "pca__n_components" in best_params
    variant = "RobustScaler + PCA" if has_pca else "RobustScaler only"
    pca_n = best_params.get("pca__n_components", None)

    if has_pca:
        fitted_pca = search.best_estimator_.named_steps["pca"]
        n_components_actual = int(fitted_pca.n_components_)
    else:
        n_components_actual = None

    # isolate RobustScaler-only vs RobustScaler+PCA best rows
    # extract top results per variant (no raw params to avoid serialization issues)
    pca_rows = results[results["param_pca__n_components"].notna()]
    no_pca_rows = results[results["param_pca__n_components"].isna()]

    variant_summary = {}
    if not no_pca_rows.empty:
        r = no_pca_rows.iloc[0]
        variant_summary["RobustScaler only"] = {
            "cv_macro_f1": float(r["mean_test_macro_f1"]),
            "C": clean_json_value(r["param_classifier__C"]),
            "gamma": clean_json_value(r["param_classifier__gamma"]),
        }
    if not pca_rows.empty:
        r = pca_rows.iloc[0]
        variant_summary["RobustScaler + PCA"] = {
            "cv_macro_f1": float(r["mean_test_macro_f1"]),
            "pca_n_components": clean_json_value(r["param_pca__n_components"]),
            "C": clean_json_value(r["param_classifier__C"]),
            "gamma": clean_json_value(r["param_classifier__gamma"]),
        }

    best_row_first = results.iloc[0]
    best_info = {
        "candidate_C_values": C_VALUES,
        "candidate_gamma_values": GAMMA_VALUES,
        "candidate_class_weight_values": CLASS_WEIGHT_VALUES,
        "pca_variance_values": PCA_VARIANCES,
        "selection_metric": "mean_test_macro_f1",
        "candidate_count": int(len(results)),
        "best_score_macro_f1": float(search.best_score_),
        "best_params": best_params,
        "selected_variant": variant,
        "pca_n_components_param": pca_n,
        "pca_n_components_actual": n_components_actual,
        "best_cv_accuracy": float(best_row_first["mean_test_accuracy"]),
        "best_cv_recall_b": float(best_row_first["mean_test_recall_b"]),
        "best_cv_roc_auc": float(best_row_first["mean_test_roc_auc"]),
        "variant_summary": variant_summary,
    }
    (OUTPUT_DIR / "model_selection.json").write_text(
        json.dumps(best_info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nSearch complete. Best variant: {variant}")
    print(f"Best CV macro_f1: {search.best_score_:.4f}")
    print(f"Best params: {best_params}")
    if n_components_actual:
        print(f"PCA components: {n_components_actual} (param: {pca_n})")

    return best_params, best_info


if __name__ == "__main__":
    best_params, best_info = grid_search()

    run_experiment(
        model_name="SVM",
        iteration_name="iteration_8",
        build_model=lambda seed: build_model(seed, best_params),
        output_dir=OUTPUT_DIR,
        experiment_note=(
            f"SVM with RobustScaler + PCA preprocessing. "
            f"Selected variant: {best_info['selected_variant']}. "
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
