from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp.AlexNet.common import build_arg_parser, resolve_image_root_from_config, run_training_experiment


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "iteration_4"
EXPERIMENT_NOTE = (
    "Train4 regularization baseline. Uses the same final cleaned dataset and fixed "
    "8:1:1 stratified manifest protocol as iteration_3, keeps the hand-written "
    "AlexNet architecture unchanged, and changes only the training recipe: "
    "RandomResizedCrop + light ColorJitter for train transforms, deterministic "
    "Resize + CenterCrop for validation/test, AdamW optimizer, and weight_decay=1e-4."
)


if __name__ == "__main__":
    parser = build_arg_parser(OUTPUT_DIR, "AlexNet iteration_4 augmentation and AdamW baseline")
    args = parser.parse_args()
    args.use_manifest = True
    args.image_root = args.image_root or resolve_image_root_from_config()
    args.manifest_path = args.manifest_path or (OUTPUT_DIR / "manifest.csv")
    args.force_manifest = args.force_manifest or not args.manifest_path.exists()
    args.transform_preset = "augmented"
    args.optimizer = "adamw"
    args.weight_decay = 1e-4
    run_training_experiment("iteration_4", EXPERIMENT_NOTE, args)
