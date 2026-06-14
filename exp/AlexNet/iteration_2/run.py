from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp.AlexNet.common import build_arg_parser, run_training_experiment


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "iteration_2"
EXPERIMENT_NOTE = (
    "Same AlexNet baseline and hyperparameters as iteration_1, but run on the "
    "manually pre-filtered Cats vs. Dogs data version. Legacy train2 logs show "
    "19968 train / 4993 val images and best val_acc=0.9263."
)


if __name__ == "__main__":
    parser = build_arg_parser(OUTPUT_DIR, "AlexNet iteration_2 manually filtered data training")
    args = parser.parse_args()
    run_training_experiment("iteration_2", EXPERIMENT_NOTE, args)
