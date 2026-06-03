from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from exp.ML.common import (
    CV_FOLDS, RANDOM_SEED, POS_LABEL, LABELS,
    load_ionosphere, run_experiment,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
C_BASE = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
C_FINE = [0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
L1_RATIOS = [0.2, 0.5, 0.8]

SCORING_BASE = {
    "macro_f1": make_scorer(lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0)),
    "recall_b": make_scorer(lambda yt, yp: recall_score(yt, yp, pos_label="b", zero_division=0)),
    "f1_b": make_scorer(lambda yt, yp: f1_score(yt, yp, pos_label="b", zero_division=0)),
    "accuracy": "accuracy",
}


# ── helpers ────────────────────────────────────────────────

def clean_value(v):
    if pd.isna(v):
        return None
    if hasattr(v, "item"):
        return v.item()
    return v


# ── pipelines ──────────────────────────────────────────────

def pipeline_full(seed: int, penalty: str = "l2", C: float = 3.0, l1_ratio: float | None = None) -> Pipeline:
    kwargs = dict(solver="saga", max_iter=10000, tol=1e-3, random_state=seed, penalty=penalty, C=C)
    if penalty == "elasticnet":
        kwargs["l1_ratio"] = l1_ratio
    return Pipeline([
        ("drop_const", VarianceThreshold()),
        ("poly2", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(**kwargs)),
    ])

def pipeline_interaction(seed: int, penalty: str = "l2", C: float = 3.0) -> Pipeline:
    return Pipeline([
        ("drop_const", VarianceThreshold()),
        ("poly2", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="saga", max_iter=10000, tol=1e-3, random_state=seed, penalty=penalty, C=C)),
    ])


# ── experiment 1: interaction_only vs full ─────────────────

def exp1_interaction_only():
    print("=" * 60)
    print("Experiment 1: full_poly2 vs interaction_only")
    print("=" * 60)
    x, y = load_ionosphere()
    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    rows = []
    for label, build_fn in [("full_poly2", lambda s: pipeline_full(s)), ("interaction_only", lambda s: pipeline_interaction(s))]:
        param_grid = [
            {"clf__penalty": ["l1"], "clf__C": C_BASE, "clf__l1_ratio": [None]},
            {"clf__penalty": ["l2"], "clf__C": C_BASE, "clf__l1_ratio": [None]},
            {"clf__penalty": ["elasticnet"], "clf__C": C_BASE, "clf__l1_ratio": L1_RATIOS},
        ]
        search = GridSearchCV(build_fn(RANDOM_SEED), param_grid, scoring=SCORING_BASE, refit="macro_f1",
                              cv=splitter, n_jobs=-1, return_train_score=True)
        search.fit(x, y)
        n_feat = search.best_estimator_.named_steps["poly2"].n_output_features_
        row = {
            "variant": label,
            "n_features": n_feat,
            "best_penalty": search.best_params_["clf__penalty"],
            "best_C": search.best_params_["clf__C"],
            "best_cv_macro_f1": float(search.best_score_),
        }
        best_row = pd.DataFrame(search.cv_results_).loc[search.best_index_]
        for m in ["mean_test_recall_b", "mean_test_f1_b", "mean_test_accuracy"]:
            row[m] = float(best_row[m])
        rows.append(row)
        print(f"  {label}: penalty={row['best_penalty']}, C={row['best_C']}, "
              f"macro_F1={row['best_cv_macro_f1']:.4f}, features={n_feat}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "exp1_interaction_only.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "exp1_interaction_only.json").write_text(df.to_json(orient="records", indent=2, force_ascii=False), encoding="utf-8")
    print("  -> exp1_interaction_only.csv\n")
    return df


# ── experiment 2: L2 C fine-search ─────────────────────────

def exp2_c_fine_search():
    print("=" * 60)
    print("Experiment 2: L2 C fine-search")
    print("=" * 60)
    x, y = load_ionosphere()
    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    # L2-only grid with fine C values
    base = Pipeline([
        ("drop_const", VarianceThreshold()),
        ("poly2", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="saga", max_iter=10000, tol=1e-3, random_state=RANDOM_SEED, penalty="l2")),
    ])
    param_grid = {"clf__C": C_FINE}
    search = GridSearchCV(base, param_grid, scoring=SCORING_BASE, refit="macro_f1",
                          cv=splitter, n_jobs=-1, return_train_score=True)
    search.fit(x, y)

    cv_df = pd.DataFrame(search.cv_results_)
    cols = [
        "rank_test_macro_f1", "param_clf__C",
        "mean_test_macro_f1", "std_test_macro_f1",
        "mean_test_recall_b", "std_test_recall_b",
        "mean_test_f1_b", "std_test_f1_b",
        "mean_test_accuracy", "std_test_accuracy",
        "mean_train_macro_f1",
    ]
    summary = cv_df[cols].sort_values("rank_test_macro_f1")
    summary.to_csv(OUTPUT_DIR / "exp2_c_fine_search.csv", index=False, encoding="utf-8-sig")

    best = summary.iloc[0]
    print(f"  Best C={best['param_clf__C']}, macro_F1={best['mean_test_macro_f1']:.4f}")
    for _, row in summary.iterrows():
        print(f"    C={row['param_clf__C']:>5.1f}  macro_F1={row['mean_test_macro_f1']:.4f}  "
              f"recall_b={row['mean_test_recall_b']:.4f}  accuracy={row['mean_test_accuracy']:.4f}")
    print("  -> exp2_c_fine_search.csv\n")
    return summary


# ── experiment 3: L1 / ElasticNet sparse comparison ────────

def exp3_sparse_comparison():
    print("=" * 60)
    print("Experiment 3: L1 / ElasticNet sparse comparison")
    print("=" * 60)
    x, y = load_ionosphere()
    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    rows = []
    configs = []
    for c in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]:
        configs.append({"penalty": "l1", "C": c, "l1_ratio": None})

    for c in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]:
        for l1r in L1_RATIOS:
            configs.append({"penalty": "elasticnet", "C": c, "l1_ratio": l1r})

    for cfg in configs:
        pipe = pipeline_full(RANDOM_SEED, penalty=cfg["penalty"], C=cfg["C"], l1_ratio=cfg["l1_ratio"])
        cv = cross_validate(pipe, x, y, cv=splitter,
                            scoring={"macro_f1": make_scorer(lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0)),
                                     "recall_b": make_scorer(lambda yt, yp: recall_score(yt, yp, pos_label="b", zero_division=0)),
                                     "accuracy": "accuracy"})

        # fit on full data to count non-zero coefs
        pipe.fit(x, y)
        coef = pipe.named_steps["clf"].coef_.ravel()
        n_nonzero = int(np.sum(np.abs(coef) > 1e-8))
        n_total = len(coef)

        row = {
            "penalty": cfg["penalty"],
            "C": cfg["C"],
            "l1_ratio": cfg["l1_ratio"] if cfg["l1_ratio"] is not None else "",
            "mean_macro_f1": float(np.mean(cv["test_macro_f1"])),
            "std_macro_f1": float(np.std(cv["test_macro_f1"])),
            "mean_recall_b": float(np.mean(cv["test_recall_b"])),
            "mean_accuracy": float(np.mean(cv["test_accuracy"])),
            "n_nonzero_coefs": n_nonzero,
            "n_total_coefs": n_total,
            "sparsity": round(n_nonzero / n_total, 4),
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["mean_macro_f1"], ascending=False)
    df.to_csv(OUTPUT_DIR / "exp3_sparse_comparison.csv", index=False, encoding="utf-8-sig")

    print(f"  Top 5 by macro_F1:")
    for _, r in df.head(5).iterrows():
        print(f"    {r['penalty']:>10s}  C={r['C']:>5.2f}  l1_ratio={str(r['l1_ratio']):>4s}  "
              f"macro_F1={r['mean_macro_f1']:.4f}  recall_b={r['mean_recall_b']:.4f}  "
              f"non-zero={r['n_nonzero_coefs']}/{r['n_total_coefs']} ({r['sparsity']:.1%})")

    # best L1 and best ElasticNet separate summary
    l1_best = df[df["penalty"] == "l1"].iloc[0]
    en_best = df[df["penalty"] == "elasticnet"].iloc[0]
    print(f"\n  Best L1:        C={l1_best['C']}, macro_F1={l1_best['mean_macro_f1']:.4f}, "
          f"non-zero={l1_best['n_nonzero_coefs']}/{l1_best['n_total_coefs']}")
    print(f"  Best ElasticNet: C={en_best['C']}, l1_ratio={en_best['l1_ratio']}, "
          f"macro_F1={en_best['mean_macro_f1']:.4f}, "
          f"non-zero={en_best['n_nonzero_coefs']}/{en_best['n_total_coefs']}")
    print("  -> exp3_sparse_comparison.csv\n")
    return df


# ── experiment 4: top polynomial feature coefficients ──────

def exp4_top_features():
    print("=" * 60)
    print("Experiment 4: top polynomial feature coefficients")
    print("=" * 60)
    x, y = load_ionosphere()

    # fit iter_4 best: full poly2 + L2 + C=3.0
    pipe = pipeline_full(RANDOM_SEED, penalty="l2", C=3.0)
    pipe.fit(x, y)

    # original feature names after VarianceThreshold
    kept_mask = pipe.named_steps["drop_const"].get_support()
    kept_names = [f"f{i+1}" for i, keep in enumerate(kept_mask) if keep]

    # polynomial feature names
    poly_step = pipe.named_steps["poly2"]
    poly_names = poly_step.get_feature_names_out(kept_names)

    coef = pipe.named_steps["clf"].coef_.ravel()
    coef_df = pd.DataFrame({"feature": poly_names, "coefficient": coef, "abs_coef": np.abs(coef)})
    coef_df = coef_df.sort_values("abs_coef", ascending=False)

    coef_df.to_csv(OUTPUT_DIR / "exp4_polynomial_coefficients.csv", index=False, encoding="utf-8-sig")

    top_n = min(30, len(coef_df))
    top = coef_df.head(top_n)
    n_total = len(coef_df)
    n_nonzero = int((coef_df["abs_coef"] > 1e-8).sum())

    print(f"  Total polynomial features: {n_total}")
    print(f"  Non-zero coefficients: {n_nonzero}")
    print(f"  Top {top_n} features by |coefficient|:")
    for i, (_, r) in enumerate(top.iterrows(), 1):
        print(f"    {i:2d}. {r['feature']:<30s}  coef={r['coefficient']:+.4f}")
    print("  -> exp4_polynomial_coefficients.csv")
    print(f"  -> full list has {n_total} features\n")


# ── experiment 5: full evaluation with interaction_only best ──

def exp5_full_eval_interaction():
    """Run full evaluation (single split + CV + repeated) for the best
    interaction_only config, so it has comparable metrics to iter_4."""
    print("=" * 60)
    print("Experiment 5: full evaluation for best interaction_only")
    print("=" * 60)
    x, y = load_ionosphere()
    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    # grid search for best interaction_only
    base = pipeline_interaction(RANDOM_SEED)
    param_grid = [
        {"clf__penalty": ["l1"], "clf__C": C_BASE, "clf__l1_ratio": [None]},
        {"clf__penalty": ["l2"], "clf__C": C_BASE, "clf__l1_ratio": [None]},
        {"clf__penalty": ["elasticnet"], "clf__C": C_BASE, "clf__l1_ratio": L1_RATIOS},
    ]
    search = GridSearchCV(base, param_grid, scoring=SCORING_BASE, refit="macro_f1",
                          cv=splitter, n_jobs=-1, return_train_score=True)
    search.fit(x, y)

    best_params = {k: clean_value(v) for k, v in search.best_params_.items()}
    n_feat = search.best_estimator_.named_steps["poly2"].n_output_features_
    print(f"  Best params: {best_params}, CV macro_F1={search.best_score_:.4f}, features={n_feat}")

    def build_fn(seed):
        return pipeline_interaction(
            seed,
            penalty=best_params.get("clf__penalty", "l2"),
            C=best_params.get("clf__C", 3.0),
        )

    run_experiment(
        model_name="LogisticRegression_interaction",
        iteration_name="iteration_5",
        build_model=build_fn,
        output_dir=OUTPUT_DIR,
        experiment_note=(
            f"LogisticRegression interaction_only (degree=2, no squared terms). "
            f"GridSearchCV over L1/L2/ElasticNet. "
            f"Best params: {best_params}. "
            f"Polynomial features: {n_feat}."
        ),
        run_cv=True,
        run_repeated=True,
    )

    # merge model_selection info
    best_info = {
        "polynomial_degree": 2,
        "interaction_only": True,
        "best_params": best_params,
        "best_cv_macro_f1": float(search.best_score_),
        "n_polynomial_features": n_feat,
    }
    metrics_path = OUTPUT_DIR / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["model_selection"] = best_info
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(best_info, ensure_ascii=False, indent=2))
    print()


# ── main ───────────────────────────────────────────────────

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    exp1_interaction_only()
    exp2_c_fine_search()
    exp3_sparse_comparison()
    exp4_top_features()
    exp5_full_eval_interaction()

    print("=" * 60)
    print("iteration_5 complete — all 4 experiments saved to outputs/")
    print("=" * 60)
