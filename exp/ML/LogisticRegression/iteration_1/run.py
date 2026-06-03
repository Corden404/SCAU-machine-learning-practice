from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.linear_model import LogisticRegression

from exp.ML.common import run_experiment


def build_model(seed: int) -> LogisticRegression:
    return LogisticRegression(random_state=seed)


if __name__ == "__main__":
    run_experiment(
        model_name="LogisticRegression",
        iteration_name="iteration_1",
        build_model=build_model,
        output_dir=Path(__file__).resolve().parent / "outputs",
        experiment_note="Default-parameter LogisticRegression baseline on the original Ionosphere features.",
    )
