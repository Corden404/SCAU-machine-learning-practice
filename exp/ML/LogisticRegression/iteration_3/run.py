from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from exp.ML.common import CV_FOLDS, POS_LABEL, RANDOM_SEED, binary_target, load_ionosphere, positive_scores, run_experiment


C_VALUES = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
NEG_LABEL = "b"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def build_model(seed: int, c_value: float = 1.0) -> Pipeline:
    return Pipeline(
        steps=[
            ("drop_constant_features", VarianceThreshold()),
            ("standardize", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=seed,
                ),
            ),
        ]
    )


def evaluate_c_values() -> tuple[float, pd.DataFrame]:
    x, y = load_ionosphere()
    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    rows = []

    for c_value in C_VALUES:
        for fold, (train_index, test_index) in enumerate(splitter.split(x, y), start=1):
            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model = build_model(RANDOM_SEED, c_value)
            model.fit(x_train, y_train)
            prediction = model.predict(x_test)
            scores = positive_scores(model, x_test)

            row = {
                "C": c_value,
                "fold": fold,
                "accuracy": accuracy_score(y_test, prediction),
                "precision_b": precision_score(y_test, prediction, pos_label=NEG_LABEL, zero_division=0),
                "recall_b": recall_score(y_test, prediction, pos_label=NEG_LABEL, zero_division=0),
                "f1_b": f1_score(y_test, prediction, pos_label=NEG_LABEL, zero_division=0),
                "precision_g": precision_score(y_test, prediction, pos_label=POS_LABEL, zero_division=0),
                "recall_g": recall_score(y_test, prediction, pos_label=POS_LABEL, zero_division=0),
                "f1_g": f1_score(y_test, prediction, pos_label=POS_LABEL, zero_division=0),
                "macro_f1": f1_score(y_test, prediction, average="macro", zero_division=0),
            }
            if scores is not None:
                row["roc_auc"] = roc_auc_score(binary_target(y_test), scores)
            rows.append(row)

    metrics = pd.DataFrame(rows)
    summary = metrics.groupby("C", as_index=False).agg(["mean", "std"])
    summary.columns = ["_".join(part for part in column if part) for column in summary.columns.to_flat_index()]
    summary = summary.reset_index()
    summary = summary.sort_values(
        by=["macro_f1_mean", "recall_b_mean", "accuracy_mean"],
        ascending=[False, False, False],
    )
    best_c = float(summary.iloc[0]["C"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_DIR / "c_search_cv_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "c_search_summary.csv", index=False, encoding="utf-8-sig")
    return best_c, summary


if __name__ == "__main__":
    best_c, c_summary = evaluate_c_values()

    run_experiment(
        model_name="LogisticRegression",
        iteration_name="iteration_3",
        build_model=lambda seed: build_model(seed, best_c),
        output_dir=OUTPUT_DIR,
        experiment_note=(
            "LogisticRegression with constant-feature removal, StandardScaler, class_weight='balanced', "
            f"and C selected by 5-fold CV macro F1 from {C_VALUES}. Best C={best_c}."
        ),
        run_cv=True,
        run_repeated=True,
    )

    model_selection = {
        "candidate_C_values": C_VALUES,
        "selection_metric": "macro_f1_mean, then recall_b_mean, then accuracy_mean",
        "best_C": best_c,
        "best_row": c_summary.iloc[0].to_dict(),
    }
    (OUTPUT_DIR / "model_selection.json").write_text(json.dumps(model_selection, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics_path = OUTPUT_DIR / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["model_selection"] = model_selection
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(model_selection, ensure_ascii=False, indent=2))
