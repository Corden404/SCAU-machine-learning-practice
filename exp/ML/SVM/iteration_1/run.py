from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.svm import SVC

from exp.ML.common import run_experiment


def build_model(seed: int) -> SVC:
    return SVC(random_state=seed)


if __name__ == "__main__":
    run_experiment(
        model_name="SVM",
        iteration_name="iteration_1",
        build_model=build_model,
        output_dir=Path(__file__).resolve().parent / "outputs",
        experiment_note="Default-parameter SVM baseline with the original feature scale.",
    )
