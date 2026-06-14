from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp.AlexNet.common import build_arg_parser, resolve_image_root_from_config, run_training_experiment


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "iteration_6"
EXPERIMENT_NOTE = (
    "Train6 AlexNet-BN structure baseline. Uses the same final cleaned dataset, "
    "8:1:1 stratified manifest protocol, baseline transforms, Adam optimizer, "
    "lr=1e-4, dropout=0.5, batch_size=32, and 20 epochs as iteration_3. "
    "This iteration changes only the model structure: add BatchNorm after every "
    "convolution layer and after the hidden fully connected layers."
)


if __name__ == "__main__":
    parser = build_arg_parser(OUTPUT_DIR, "AlexNet iteration_6 BatchNorm structure baseline")
    args = parser.parse_args()
    args.use_manifest = True
    args.image_root = args.image_root or resolve_image_root_from_config()
    args.manifest_path = args.manifest_path or (OUTPUT_DIR / "manifest.csv")
    args.force_manifest = args.force_manifest or not args.manifest_path.exists()
    args.model_variant = "alexnet_bn"
    run_training_experiment("iteration_6", EXPERIMENT_NOTE, args)
