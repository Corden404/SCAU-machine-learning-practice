from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.tree import DecisionTreeClassifier

from exp.ML.common import run_experiment


def build_model(seed: int) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=seed)


if __name__ == "__main__":
    run_experiment(
        model_name="DecisionTree",
        iteration_name="iteration_2",
        build_model=build_model,
        output_dir=Path(__file__).resolve().parent / "outputs",
        experiment_note="Enhanced DecisionTree with max_depth=4 and min_samples_leaf=5 to reduce overfitting on the small Ionosphere dataset.",
        run_cv=True,
        run_repeated=True,
    )
