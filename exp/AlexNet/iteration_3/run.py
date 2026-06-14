from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp.AlexNet.common import build_arg_parser, resolve_image_root_from_config, run_training_experiment


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "iteration_3"
EXPERIMENT_NOTE = (
    "Train3 baseline on the final cleaned Cats vs. Dogs dataset. "
    "This iteration keeps the original hand-written AlexNet architecture and baseline "
    "training hyperparameters, then changes the experiment protocol: use the final "
    "cleaned data, generate a fixed stratified manifest split, add an independent "
    "test set, and save full metric artifacts."
)


if __name__ == "__main__":
    parser = build_arg_parser(OUTPUT_DIR, "AlexNet iteration_3 final-cleaned-data baseline")
    args = parser.parse_args()
    args.use_manifest = True
    args.image_root = args.image_root or resolve_image_root_from_config()
    args.manifest_path = args.manifest_path or (OUTPUT_DIR / "manifest.csv")
    args.force_manifest = args.force_manifest or not args.manifest_path.exists()
    run_training_experiment("iteration_3", EXPERIMENT_NOTE, args)
