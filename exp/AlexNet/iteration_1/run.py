from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp.AlexNet.common import build_arg_parser, run_training_experiment


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "iteration_1"
EXPERIMENT_NOTE = (
    "Baseline hand-written AlexNet. Uses 224x224 resize, random horizontal flip, "
    "ImageNet normalization, Adam(lr=1e-4), dropout=0.5, and 20 epochs. "
    "Legacy train1 logs show 20000 train / 5002 val images and best val_acc=0.9256."
)


if __name__ == "__main__":
    parser = build_arg_parser(OUTPUT_DIR, "AlexNet iteration_1 baseline training")
    args = parser.parse_args()
    run_training_experiment("iteration_1", EXPERIMENT_NOTE, args)
