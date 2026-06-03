from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.ensemble import RandomForestClassifier

from exp.ML.common import run_experiment


def build_model(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(random_state=seed, n_jobs=-1)


if __name__ == "__main__":
    run_experiment(
        model_name="RandomForest",
        iteration_name="iteration_2",
        build_model=build_model,
        output_dir=Path(__file__).resolve().parent / "outputs",
        experiment_note="Enhanced RandomForest evaluation with 5-fold CV and repeated stratified splits to examine ensemble stability.",
        run_cv=True,
        run_repeated=True,
    )
