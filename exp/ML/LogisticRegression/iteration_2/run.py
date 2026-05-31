from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from exp.ML.common import run_experiment


def build_model(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("drop_constant_features", VarianceThreshold()),
            ("standardize", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )


if __name__ == "__main__":
    run_experiment(
        model_name="LogisticRegression",
        iteration_name="iteration_2",
        build_model=build_model,
        output_dir=Path(__file__).resolve().parent / "outputs",
        experiment_note="Enhanced LogisticRegression with constant-feature removal, standardization, 5-fold CV, and repeated stratified splits.",
        run_cv=True,
        run_repeated=True,
    )
