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
from sklearn.tree import DecisionTreeClassifier

from exp.ML.common import CV_FOLDS, RANDOM_SEED, binary_target, load_ionosphere, positive_scores, run_experiment


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
FIXED_CCP_ALPHA = 0.00852480852480852
CRITERION_VALUES = ["gini", "entropy", "log_loss"]


def build_model(seed: int, best_params: dict[str, Any] | None = None) -> DecisionTreeClassifier:
    model = DecisionTreeClassifier(
        ccp_alpha=FIXED_CCP_ALPHA,
        random_state=seed,
    )
    if best_params:
        model.set_params(**best_params)
    return model


def score_roc_auc(estimator: DecisionTreeClassifier, x: pd.DataFrame, y: pd.Series) -> float:
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


def save_tree_summary(model: DecisionTreeClassifier, x: pd.DataFrame) -> dict[str, Any]:
    feature_importances = sorted(zip(x.columns, model.feature_importances_), key=lambda item: item[1], reverse=True)
    top_features = [
        {"feature": str(feature), "importance": float(importance)}
        for feature, importance in feature_importances[:10]
        if float(importance) > 0
    ]
    summary = {
        "depth": int(model.get_depth()),
        "leaf_count": int(model.get_n_leaves()),
        "top_features": top_features,
    }
    (OUTPUT_DIR / "tree_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


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
    search = GridSearchCV(
        estimator=DecisionTreeClassifier(ccp_alpha=FIXED_CCP_ALPHA, random_state=RANDOM_SEED),
        param_grid={"criterion": CRITERION_VALUES},
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
        "param_criterion",
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
    best_estimator = search.best_estimator_
    tree_summary = save_tree_summary(best_estimator, x)
    best_info = {
        "fixed_ccp_alpha": FIXED_CCP_ALPHA,
        "candidate_criterion_values": CRITERION_VALUES,
        "selection_metric": "mean_test_macro_f1",
        "best_score_macro_f1": float(search.best_score_),
        "best_params": best_params,
        "best_tree_summary": tree_summary,
        "best_row": clean_record(results.iloc[0].to_dict()),
    }
    (OUTPUT_DIR / "model_selection.json").write_text(json.dumps(best_info, ensure_ascii=False, indent=2), encoding="utf-8")
    return best_params, best_info


if __name__ == "__main__":
    best_params, best_info = grid_search()
    run_experiment(
        model_name="DecisionTree",
        iteration_name="iteration_5",
        build_model=lambda seed: build_model(seed, best_params),
        output_dir=OUTPUT_DIR,
        experiment_note=(
            "DecisionTree criterion comparison (gini vs entropy vs log_loss) "
            f"with fixed ccp_alpha={FIXED_CCP_ALPHA} from iteration_3. "
            f"Best criterion: {best_info['best_params']}."
        ),
        run_cv=True,
        run_repeated=True,
    )

    metrics_path = OUTPUT_DIR / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["model_selection"] = best_info
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(best_info, ensure_ascii=False, indent=2))
