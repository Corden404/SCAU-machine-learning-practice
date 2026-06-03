from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from exp.ML.common import run_experiment


def build_model(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("standardize", StandardScaler()),
            ("classifier", SVC(random_state=seed)),
        ]
    )


if __name__ == "__main__":
    run_experiment(
        model_name="SVM",
        iteration_name="iteration_2",
        build_model=build_model,
        output_dir=Path(__file__).resolve().parent / "outputs",
        experiment_note="Enhanced SVM with StandardScaler, 5-fold CV, and repeated stratified splits.",
        run_cv=True,
        run_repeated=True,
    )
