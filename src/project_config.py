from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "paths.local.json"
CONFIG_ENV_VAR = "ML_PRACTICE_CONFIG"


def get_config_path() -> Path:
    """Return the local config path, allowing an environment variable override."""
    env_path = os.environ.get(CONFIG_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser().resolve()
    return DEFAULT_CONFIG_PATH


def load_config() -> dict[str, Any]:
    """Load private local paths without hard-coding them in experiment scripts."""
    config_path = get_config_path()
    if not config_path.exists():
        example_path = PROJECT_ROOT / "config" / "paths.example.json"
        raise FileNotFoundError(
            f"Local config not found: {config_path}\n"
            f"Create it from the template: {example_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_data_root(*, must_exist: bool = True) -> Path:
    config = load_config()
    data_root = Path(config["data_root"]).expanduser()
    if must_exist and not data_root.exists():
        raise FileNotFoundError(f"Configured data_root does not exist: {data_root}")
    return data_root


def get_dataset_path(name: str, *, must_exist: bool = True) -> Path:
    config = load_config()
    datasets = config.get("datasets", {})
    if name not in datasets:
        available = ", ".join(sorted(datasets)) or "none"
        raise KeyError(f"Unknown dataset: {name}. Available datasets: {available}")

    dataset_path = Path(datasets[name]).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = get_data_root(must_exist=must_exist) / dataset_path

    if must_exist and not dataset_path.exists():
        raise FileNotFoundError(f"Configured dataset does not exist: {dataset_path}")
    return dataset_path
