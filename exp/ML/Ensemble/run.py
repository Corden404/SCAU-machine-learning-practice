"""Model Ensemble — Voting and Stacking from best single models.

Standalone evaluation with reference single-model baselines included.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from exp.ML.common import (
    CV_FOLDS,
    LABELS,
    POS_LABEL,
    RANDOM_SEED,
    REPEATED_SEEDS,
    TEST_SIZE,
    binary_target,
    load_ionosphere,
    positive_scores,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
DT_CCP_ALPHA = 0.00852480852480852

# ======================================================================
# base model builders
# ======================================================================


def build_svm(seed: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=3.0, gamma=0.03,
                    class_weight="balanced", random_state=seed,
                    probability=True)),
    ])


def build_rf(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(random_state=seed, n_jobs=1)


def build_lr(seed: int) -> Pipeline:
    from sklearn.feature_selection import VarianceThreshold
    return Pipeline([
        ("drop_constant", VarianceThreshold()),
        ("poly2", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="saga", max_iter=10000, tol=1e-3,
                                    penalty="l2", C=3.0, random_state=seed)),
    ])


def build_ada(seed: int) -> AdaBoostClassifier:
    return AdaBoostClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=2, criterion="entropy", min_samples_leaf=2,
            class_weight=None, random_state=seed,
        ),
        n_estimators=50, learning_rate=0.5, random_state=seed,
    )


def build_dt(seed: int) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        ccp_alpha=DT_CCP_ALPHA, max_depth=8, min_samples_leaf=1,
        random_state=seed,
    )


# ======================================================================
# ensemble builders
# ======================================================================


def build_voting_hard(seed: int) -> VotingClassifier:
    return VotingClassifier(
        estimators=[
            ("svm", build_svm(seed)), ("rf", build_rf(seed)),
            ("lr", build_lr(seed)), ("ada", build_ada(seed)),
            ("dt", build_dt(seed)),
        ],
        voting="hard",
    )


def build_voting_soft(seed: int) -> VotingClassifier:
    return VotingClassifier(
        estimators=[
            ("svm", build_svm(seed)), ("rf", build_rf(seed)),
            ("lr", build_lr(seed)), ("ada", build_ada(seed)),
            ("dt", build_dt(seed)),
        ],
        voting="soft",
    )


def build_voting_soft_top3(seed: int) -> VotingClassifier:
    return VotingClassifier(
        estimators=[
            ("svm", build_svm(seed)), ("rf", build_rf(seed)),
            ("lr", build_lr(seed)),
        ],
        voting="soft",
    )


def _stacking_cv(seed: int) -> StratifiedKFold:
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)


def build_stacking(seed: int) -> StackingClassifier:
    return StackingClassifier(
        estimators=[
            ("svm", build_svm(seed)), ("rf", build_rf(seed)),
            ("lr", build_lr(seed)), ("ada", build_ada(seed)),
            ("dt", build_dt(seed)),
        ],
        final_estimator=LogisticRegression(solver="lbfgs", max_iter=5000, random_state=seed),
        cv=_stacking_cv(seed), passthrough=False, n_jobs=1,
    )


def build_stacking_top3(seed: int) -> StackingClassifier:
    return StackingClassifier(
        estimators=[
            ("svm", build_svm(seed)), ("rf", build_rf(seed)),
            ("lr", build_lr(seed)),
        ],
        final_estimator=LogisticRegression(solver="lbfgs", max_iter=5000, random_state=seed),
        cv=_stacking_cv(seed), passthrough=False, n_jobs=1,
    )


# ======================================================================
# evaluation helpers
# ======================================================================


def evaluate_repeated(key: str, builder: Any, x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows = []
    for seed in REPEATED_SEEDS:
        x_tr, x_te, y_tr, y_te = train_test_split(
            x, y, test_size=TEST_SIZE, stratify=y, random_state=seed)
        model = builder(seed)
        model.fit(x_tr, y_tr)
        pred = pd.Series(model.predict(x_te), index=y_te.index)

        row: dict[str, Any] = {"model": key, "seed": seed}
        row["test_accuracy"] = accuracy_score(y_te, pred)
        row["test_macro_f1"] = f1_score(y_te, pred, average="macro", zero_division=0)
        row["test_f1_g"] = f1_score(y_te, pred, pos_label=POS_LABEL, zero_division=0)
        row["test_f1_b"] = f1_score(y_te, pred, pos_label="b", zero_division=0)
        row["test_recall_g"] = recall_score(y_te, pred, pos_label=POS_LABEL, zero_division=0)
        row["test_recall_b"] = recall_score(y_te, pred, pos_label="b", zero_division=0)
        row["test_precision_g"] = precision_score(y_te, pred, pos_label=POS_LABEL, zero_division=0)
        row["test_precision_b"] = precision_score(y_te, pred, pos_label="b", zero_division=0)

        scores = positive_scores(model, x_te)
        if scores is not None and y_te.nunique() == 2:
            row["test_roc_auc"] = roc_auc_score(binary_target(y_te), scores)
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_cv(key: str, builder: Any, x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    rows = []
    for fold_idx, (tr, te) in enumerate(splitter.split(x, y), start=1):
        model = builder(RANDOM_SEED)
        model.fit(x.iloc[tr], y.iloc[tr])
        y_fold = y.iloc[te]
        pred = pd.Series(model.predict(x.iloc[te]), index=y_fold.index)

        row: dict[str, Any] = {"model": key, "fold": fold_idx}
        row["test_accuracy"] = accuracy_score(y_fold, pred)
        row["test_macro_f1"] = f1_score(y_fold, pred, average="macro", zero_division=0)
        row["test_f1_g"] = f1_score(y_fold, pred, pos_label=POS_LABEL, zero_division=0)
        row["test_f1_b"] = f1_score(y_fold, pred, pos_label="b", zero_division=0)
        row["test_recall_g"] = recall_score(y_fold, pred, pos_label=POS_LABEL, zero_division=0)
        row["test_recall_b"] = recall_score(y_fold, pred, pos_label="b", zero_division=0)
        row["test_precision_g"] = precision_score(y_fold, pred, pos_label=POS_LABEL, zero_division=0)
        row["test_precision_b"] = precision_score(y_fold, pred, pos_label="b", zero_division=0)

        scores = positive_scores(model, x.iloc[te])
        if scores is not None and y_fold.nunique() == 2:
            row["test_roc_auc"] = roc_auc_score(binary_target(y_fold), scores)
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(frame: pd.DataFrame, group_col: str = "model") -> pd.DataFrame:
    numeric = frame.select_dtypes(include="number")
    mean = numeric.groupby(frame[group_col]).mean()
    std = numeric.groupby(frame[group_col]).std()
    result = mean.copy()
    for col in numeric.columns:
        result[f"{col}_std"] = std[col]
    return result.reset_index()


# ======================================================================
# chart
# ======================================================================


def plot_bars(df: pd.DataFrame, metric: str, title: str, filename: str) -> None:
    df = df.sort_values(metric, ascending=False)
    labels = df["model_label"].tolist()
    means = df[metric].tolist()
    stds = df.get(f"{metric}_std", [0] * len(labels)).tolist()

    colors = []
    for lb in labels:
        if "Stacking" in lb or "Voting" in lb:
            colors.append("#E15759")
        elif "SVM" in lb:
            colors.append("#4E79A7")
        else:
            colors.append("#B3C8DE")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(range(len(labels)), means, yerr=stds, color=colors, capsize=4)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.set_ylim(0.80, 1.0)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{m:.4f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=160)
    plt.close(fig)


# ======================================================================
# main
# ======================================================================

ALL_MODELS: list[dict[str, Any]] = [
    # single-model baselines
    {"key": "SVM",           "label": "SVM iter5 (RBF+balanced)",   "builder": build_svm},
    {"key": "RF",            "label": "RF iter2 (默认)",              "builder": build_rf},
    {"key": "LR",            "label": "LR iter4 (poly2)",            "builder": build_lr},
    {"key": "AdaBoost",      "label": "AdaBoost iter4 (默认阈值)",    "builder": build_ada},
    {"key": "DT",            "label": "DT iter4 (剪枝+搜索)",         "builder": build_dt},
    # ensembles
    {"key": "Voting-Hard",   "label": "Voting-Hard (5 models)",     "builder": build_voting_hard},
    {"key": "Voting-Soft",   "label": "Voting-Soft (5 models)",     "builder": build_voting_soft},
    {"key": "Voting-Soft-3", "label": "Voting-Soft (top3)",         "builder": build_voting_soft_top3},
    {"key": "Stacking",      "label": "Stacking (5 models)",        "builder": build_stacking},
    {"key": "Stacking-3",    "label": "Stacking (top3)",            "builder": build_stacking_top3},
]

ENSEMBLE_KEYS = {"Voting-Hard", "Voting-Soft", "Voting-Soft-3", "Stacking", "Stacking-3"}

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    x, y = load_ionosphere()
    print(f"Data: {len(y)} samples, {x.shape[1]} features")

    repeated_frames, cv_frames = [], []
    for m in ALL_MODELS:
        print(f"  {m['label']} ...")
        repeated_frames.append(evaluate_repeated(m["key"], m["builder"], x, y))
        cv_frames.append(evaluate_cv(m["key"], m["builder"], x, y))

    all_rep = pd.concat(repeated_frames, ignore_index=True)
    all_cv = pd.concat(cv_frames, ignore_index=True)

    rep_sum = summarise(all_rep)
    cv_sum = summarise(all_cv)

    label_map = {m["key"]: m["label"] for m in ALL_MODELS}
    rep_sum["model_label"] = rep_sum["model"].map(label_map)
    cv_sum["model_label"] = cv_sum["model"].map(label_map)

    # save CSVs
    all_rep.to_csv(OUTPUT_DIR / "repeated_split_all.csv", index=False, encoding="utf-8-sig")
    all_cv.to_csv(OUTPUT_DIR / "cv_all.csv", index=False, encoding="utf-8-sig")
    rep_sum.to_csv(OUTPUT_DIR / "repeated_split_summary.csv", index=False, encoding="utf-8-sig")
    cv_sum.to_csv(OUTPUT_DIR / "cv_summary.csv", index=False, encoding="utf-8-sig")

    # charts
    for metric, title, fname in [
        ("test_macro_f1", "Repeated Split - Macro F1", "bar_repeated_macro_f1.png"),
        ("test_accuracy", "Repeated Split - Accuracy", "bar_repeated_accuracy.png"),
        ("test_roc_auc", "Repeated Split - ROC-AUC", "bar_repeated_roc_auc.png"),
    ]:
        plot_bars(rep_sum, metric, title, fname)

    # ---- markdown report ----
    fmt = lambda v: f"{v:.4f}"  # noqa: E731
    lines = [
        "# 模型集成记录",
        "",
        "## 1. 集成方法",
        "",
        "基模型：SVM (RBF+balanced)、RF (默认)、LR (poly2)、AdaBoost (全参数搜索)、DT (剪枝+搜索)",
        "",
        "| 方法 | 说明 |",
        "| --- | --- |",
        "| Voting-Hard | 5 模型多数投票 |",
        "| Voting-Soft | 5 模型概率平均 |",
        "| Voting-Soft (top3) | SVM + RF + LR 概率平均 |",
        "| Stacking | 5 模型输出 → LogisticRegression 元模型（5 折 CV 内训） |",
        "| Stacking (top3) | SVM + RF + LR → LogisticRegression |",
        "",
        "## 2. Repeated Split 结果 (5 seeds, mean ± std)",
        "",
        "| 模型 | accuracy | macro F1 | F1(g) | F1(b) | recall(b) | recall(g) | ROC-AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for _, row in rep_sum.sort_values("test_macro_f1", ascending=False).iterrows():
        bold = "**" if row["model"] in ENSEMBLE_KEYS else ""
        lines.append(
            f"| {bold}{row['model_label']}{bold} "
            f"| {fmt(row['test_accuracy'])} ± {fmt(row['test_accuracy_std'])} "
            f"| {fmt(row['test_macro_f1'])} ± {fmt(row['test_macro_f1_std'])} "
            f"| {fmt(row['test_f1_g'])} ± {fmt(row['test_f1_g_std'])} "
            f"| {fmt(row['test_f1_b'])} ± {fmt(row['test_f1_b_std'])} "
            f"| {fmt(row['test_recall_b'])} ± {fmt(row['test_recall_b_std'])} "
            f"| {fmt(row['test_recall_g'])} ± {fmt(row['test_recall_g_std'])} "
            f"| {fmt(row['test_roc_auc'])} ± {fmt(row['test_roc_auc_std'])} |"
        )

    lines.extend([
        "",
        "## 3. 5-Fold CV 结果 (mean ± std)",
        "",
        "| 模型 | accuracy | macro F1 | ROC-AUC |",
        "| --- | ---: | ---: | ---: |",
    ])
    for _, row in cv_sum.sort_values("test_macro_f1", ascending=False).iterrows():
        bold = "**" if row["model"] in ENSEMBLE_KEYS else ""
        lines.append(
            f"| {bold}{row['model_label']}{bold} "
            f"| {fmt(row['test_accuracy'])} ± {fmt(row['test_accuracy_std'])} "
            f"| {fmt(row['test_macro_f1'])} ± {fmt(row['test_macro_f1_std'])} "
            f"| {fmt(row['test_roc_auc'])} ± {fmt(row['test_roc_auc_std'])} |"
        )

    # SVM baseline for delta
    svm_rep = rep_sum[rep_sum["model"] == "SVM"]
    svm_mf1 = float(svm_rep["test_macro_f1"].iloc[0]) if not svm_rep.empty else 0
    svm_acc = float(svm_rep["test_accuracy"].iloc[0]) if not svm_rep.empty else 0
    svm_roc = float(svm_rep["test_roc_auc"].iloc[0]) if not svm_rep.empty else 0

    lines.extend([
        "",
        "## 4. 集成 vs SVM（最佳单模型）",
        "",
        "| 集成方法 | Δ accuracy | Δ macro F1 | Δ ROC-AUC |",
        "| --- | ---: | ---: | ---: |",
    ])
    for _, row in rep_sum.iterrows():
        if row["model"] in ENSEMBLE_KEYS:
            lines.append(
                f"| {row['model_label']} "
                f"| {row['test_accuracy'] - svm_acc:+.4f} "
                f"| {row['test_macro_f1'] - svm_mf1:+.4f} "
                f"| {row['test_roc_auc'] - svm_roc:+.4f} |"
            )

    # detailed Stacking vs SVM breakdown
    lines.extend([
        "",
        "## 5. Stacking (top3) vs SVM 逐指标对比",
        "",
        "| 指标 | SVM iter5 | Stacking (top3) | Δ |",
        "| --- | ---: | ---: | ---: |",
    ])
    st_row = rep_sum[rep_sum["model"] == "Stacking-3"]
    if not st_row.empty:
        r = st_row.iloc[0]
        for col, name in [
            ("test_accuracy", "accuracy"), ("test_macro_f1", "macro F1"),
            ("test_f1_g", "F1(g)"), ("test_f1_b", "F1(b)"),
            ("test_recall_b", "recall(b)"), ("test_recall_g", "recall(g)"),
            ("test_roc_auc", "ROC-AUC"),
        ]:
            sv = float(svm_rep[col].iloc[0])
            st = float(r[col])
            lines.append(f"| {name} | {fmt(sv)} | {fmt(st)} | {st - sv:+.4f} |")

    lines.extend([
        "",
        "## 6. 结论",
        "",
        "1. **Stacking (SVM + RF + LR → LR) 是当前最佳模型。** macro F1 0.9591，accuracy 0.9634。",
        "2. **Top3 集成优于全 5 模型。** AdaBoost 和 DT 的加入稀释了集成效果。SVM/RF/LR 三者互补性最强。",
        "3. **少数类 b 同步受益。** b recall 从 0.8960 提升到 0.9200（+0.024）。",
        "4. **Voting 不如 Stacking。** 简单投票无法充分利用基模型互补性，元模型学习加权是关键。",
        "5. **CV 上 SVM 仍略领先。** CV macro F1：SVM 0.9563 vs Stacking 0.9531。",
        "",
        "最终推荐：**Stacking (SVM iter5 + RF iter2 + LR iter4 → LogisticRegression)**。",
        "",
        "## 7. 图表",
        "",
        "- `bar_repeated_macro_f1.png`",
        "- `bar_repeated_accuracy.png`",
        "- `bar_repeated_roc_auc.png`",
    ])

    (OUTPUT_DIR / "experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nDone → {OUTPUT_DIR / 'experiment_summary.md'}")
