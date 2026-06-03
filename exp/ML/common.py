from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preanalysis import ensure_extracted, find_ionosphere_table


RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
REPEATED_SEEDS = [0, 1, 2, 3, 4]
LABELS = ["b", "g"]
POS_LABEL = "g"

ModelBuilder = Callable[[int], ClassifierMixin]


def load_ionosphere() -> tuple[pd.DataFrame, pd.Series]:
    extracted_root, _ = ensure_extracted("ionosphere")
    table_path = find_ionosphere_table(extracted_root)
    frame = pd.read_csv(table_path, header=None).dropna(how="all")
    feature_names = [f"f{i + 1}" for i in range(frame.shape[1] - 1)]
    frame.columns = feature_names + ["label"]
    x = frame[feature_names].apply(pd.to_numeric, errors="coerce")
    y = frame["label"].astype(str).str.strip()
    return x, y


def positive_scores(model: ClassifierMixin, x: pd.DataFrame) -> pd.Series | None:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)
        classes = list(getattr(model, "classes_", LABELS))
        if POS_LABEL in classes:
            return pd.Series(probabilities[:, classes.index(POS_LABEL)], index=x.index)

    if hasattr(model, "decision_function"):
        values = model.decision_function(x)
        if getattr(values, "ndim", 1) != 1:
            classes = list(getattr(model, "classes_", LABELS))
            if POS_LABEL in classes:
                values = values[:, classes.index(POS_LABEL)]
        else:
            classes = list(getattr(model, "classes_", LABELS))
            if len(classes) == 2 and classes[1] != POS_LABEL:
                values = -values
        return pd.Series(values, index=x.index)

    return None


def binary_target(y: pd.Series) -> pd.Series:
    return y.eq(POS_LABEL).astype(int)


def metric_block(model: ClassifierMixin, x: pd.DataFrame, y: pd.Series, prefix: str) -> dict[str, float]:
    prediction = pd.Series(model.predict(x), index=y.index)
    scores = positive_scores(model, x)
    result = {
        f"{prefix}_accuracy": accuracy_score(y, prediction),
        f"{prefix}_precision": precision_score(y, prediction, pos_label=POS_LABEL, zero_division=0),
        f"{prefix}_recall": recall_score(y, prediction, pos_label=POS_LABEL, zero_division=0),
        f"{prefix}_f1": f1_score(y, prediction, pos_label=POS_LABEL, zero_division=0),
    }
    if scores is not None and y.nunique() == 2:
        result[f"{prefix}_roc_auc"] = roc_auc_score(binary_target(y), scores)
    return result


def evaluate_split(
    model_name: str,
    iteration_name: str,
    model: ClassifierMixin,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    seed: int,
) -> dict[str, float | int | str]:
    model.fit(x_train, y_train)
    row: dict[str, float | int | str] = {
        "model": model_name,
        "iteration": iteration_name,
        "seed": seed,
    }
    row.update(metric_block(model, x_train, y_train, "train"))
    row.update(metric_block(model, x_test, y_test, "test"))
    return row


def run_cross_validation(
    model_name: str,
    iteration_name: str,
    build_model: ModelBuilder,
    x: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    rows = []
    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    for fold_index, (train_index, test_index) in enumerate(splitter.split(x, y), start=1):
        model = build_model(RANDOM_SEED)
        row = evaluate_split(
            model_name,
            iteration_name,
            model,
            x.iloc[train_index],
            x.iloc[test_index],
            y.iloc[train_index],
            y.iloc[test_index],
            RANDOM_SEED,
        )
        row["fold"] = fold_index
        rows.append(row)
    return pd.DataFrame(rows)


def run_repeated_splits(
    model_name: str,
    iteration_name: str,
    build_model: ModelBuilder,
    x: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    rows = []
    for seed in REPEATED_SEEDS:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=seed,
        )
        rows.append(evaluate_split(model_name, iteration_name, build_model(seed), x_train, x_test, y_train, y_test, seed))
    return pd.DataFrame(rows)


def save_confusion_matrix(model: ClassifierMixin, x_test: pd.DataFrame, y_test: pd.Series, output_dir: Path) -> None:
    prediction = model.predict(x_test)
    matrix = confusion_matrix(y_test, prediction, labels=LABELS)
    pd.DataFrame(matrix, index=[f"actual_{label}" for label in LABELS], columns=[f"pred_{label}" for label in LABELS]).to_csv(
        output_dir / "confusion_matrix.csv",
        encoding="utf-8-sig",
    )

    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=LABELS)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def save_roc_curve(model: ClassifierMixin, x_test: pd.DataFrame, y_test: pd.Series, output_dir: Path) -> None:
    scores = positive_scores(model, x_test)
    if scores is None or y_test.nunique() != 2:
        return
    fpr, tpr, _ = roc_curve(binary_target(y_test), scores)
    auc = roc_auc_score(binary_target(y_test), scores)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}", color="#4E79A7")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=160)
    plt.close(fig)


def summarize_table(frame: pd.DataFrame, group_name: str) -> pd.DataFrame:
    numeric = frame.select_dtypes(include="number")
    summary = numeric.agg(["mean", "std"]).T.reset_index()
    summary.columns = ["metric", f"{group_name}_mean", f"{group_name}_std"]
    return summary


def save_summary_markdown(
    output_dir: Path,
    model_name: str,
    iteration_name: str,
    experiment_note: str,
    single_metrics: dict[str, float | int | str],
    data_summary: dict[str, object],
    has_cv: bool,
    has_repeated: bool,
) -> None:
    lines = [
        f"# {model_name} {iteration_name}",
        "",
        "## Experiment Note",
        "",
        experiment_note,
        "",
        "## Data",
        "",
        f"- Samples: {data_summary['sample_count']}",
        f"- Features: {data_summary['feature_count']}",
        f"- Labels: {data_summary['label_counts']}",
        f"- Positive label for binary metrics: `{POS_LABEL}`",
        "",
        "## Single Split Test Metrics",
        "",
    ]
    for key in ["test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"]:
        value = single_metrics.get(key)
        if isinstance(value, float):
            lines.append(f"- {key}: {value:.4f}")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `single_split_metrics.csv`",
            "- `classification_report.json`",
            "- `confusion_matrix.csv`",
            "- `confusion_matrix.png`",
            "- `roc_curve.png`",
        ]
    )
    if has_cv:
        lines.append("- `cv_metrics.csv`")
        lines.append("- `cv_metrics_summary.csv`")
    if has_repeated:
        lines.append("- `repeated_split_metrics.csv`")
        lines.append("- `repeated_split_summary.csv`")
    (output_dir / "experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(
    *,
    model_name: str,
    iteration_name: str,
    build_model: ModelBuilder,
    output_dir: Path,
    experiment_note: str,
    run_cv: bool = False,
    run_repeated: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    x, y = load_ionosphere()
    data_summary = {
        "sample_count": int(len(y)),
        "feature_count": int(x.shape[1]),
        "label_counts": {str(key): int(value) for key, value in y.value_counts().sort_index().items()},
    }

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )
    model = build_model(RANDOM_SEED)
    single_metrics = evaluate_split(model_name, iteration_name, model, x_train, x_test, y_train, y_test, RANDOM_SEED)

    pd.DataFrame([single_metrics]).to_csv(output_dir / "single_split_metrics.csv", index=False, encoding="utf-8-sig")
    save_confusion_matrix(model, x_test, y_test, output_dir)
    save_roc_curve(model, x_test, y_test, output_dir)
    report = classification_report(y_test, model.predict(x_test), labels=LABELS, output_dict=True, zero_division=0)
    (output_dir / "classification_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    metadata = {
        "model": model_name,
        "iteration": iteration_name,
        "note": experiment_note,
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "labels": LABELS,
        "positive_label": POS_LABEL,
        "data": data_summary,
        "single_split_metrics": single_metrics,
    }

    if run_cv:
        cv_metrics = run_cross_validation(model_name, iteration_name, build_model, x, y)
        cv_metrics.to_csv(output_dir / "cv_metrics.csv", index=False, encoding="utf-8-sig")
        summarize_table(cv_metrics.drop(columns=["fold"], errors="ignore"), "cv").to_csv(
            output_dir / "cv_metrics_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        metadata["cv_folds"] = CV_FOLDS

    if run_repeated:
        repeated_metrics = run_repeated_splits(model_name, iteration_name, build_model, x, y)
        repeated_metrics.to_csv(output_dir / "repeated_split_metrics.csv", index=False, encoding="utf-8-sig")
        summarize_table(repeated_metrics, "repeated").to_csv(
            output_dir / "repeated_split_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        metadata["repeated_seeds"] = REPEATED_SEEDS

    (output_dir / "metrics.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    save_summary_markdown(
        output_dir,
        model_name,
        iteration_name,
        experiment_note,
        single_metrics,
        data_summary,
        run_cv,
        run_repeated,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
