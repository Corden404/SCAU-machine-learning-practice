from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import IMG_EXTENSIONS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp.AlexNet.model import MODEL_VARIANTS, build_model
from src.project_config import get_dataset_path


RANDOM_SEED = 42
IMAGE_SIZE = 224
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


def public_path(path: str | Path | None) -> str | None:
    """Return a report-safe path without exposing user-specific directories."""
    if path is None:
        return None
    path_obj = Path(path)
    try:
        resolved = path_obj.resolve()
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        pass
    if path_obj.is_absolute():
        return f"<local_path>/{path_obj.name}"
    return path_obj.as_posix()


def _candidate_image_roots(dataset_path: Path) -> list[Path]:
    base = dataset_path.with_suffix("") if dataset_path.is_file() and dataset_path.suffix.lower() == ".zip" else dataset_path
    return [
        base,
        base / "PetImages",
        base / "images" / "PetImages",
    ]


def resolve_image_root_from_config() -> Path:
    """Resolve Cats vs. Dogs image root from local config without hard-coded paths."""
    errors = []
    for dataset_name in ["cats_vs_dogs_cleaned", "cats_vs_dogs"]:
        try:
            dataset_path = get_dataset_path(dataset_name)
        except (KeyError, FileNotFoundError) as error:
            errors.append(f"{dataset_name}: {error}")
            continue
        for candidate in _candidate_image_roots(dataset_path):
            if candidate.exists() and candidate.is_dir():
                return candidate
        errors.append(
            f"{dataset_name}: configured path exists, but no image root was found. "
            "Expected the configured directory itself, PetImages/, or images/PetImages/."
        )
    raise FileNotFoundError(
        "Could not resolve Cats vs. Dogs image root from config/paths.local.json.\n"
        "Add datasets.cats_vs_dogs_cleaned pointing to the cleaned PetImages directory, "
        "or ensure datasets.cats_vs_dogs is extracted next to its zip.\n"
        + "\n".join(errors)
    )


def seed_everything(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def pil_loader(path: str | Path) -> Image.Image | None:
    try:
        image = Image.open(path)
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None


class SafeImageFolder(VisionDataset):
    """ImageFolder-like dataset that skips unreadable images during batching."""

    def __init__(self, root: str | Path, transform: Any = None) -> None:
        root_path = Path(root)
        super().__init__(str(root_path), transform=transform)
        self.root_path = root_path
        self.samples: list[tuple[Path, int]] = []
        self.classes = [
            path.name
            for path in sorted(root_path.iterdir(), key=lambda item: item.name.lower())
            if path.is_dir()
        ]
        self.class_to_idx = {class_name: index for index, class_name in enumerate(self.classes)}

        valid_extensions = {extension.lower() for extension in IMG_EXTENSIONS}
        for class_name in self.classes:
            class_dir = root_path / class_name
            for image_path in sorted(class_dir.iterdir(), key=lambda item: item.name.lower()):
                if image_path.is_file() and image_path.suffix.lower() in valid_extensions:
                    self.samples.append((image_path, self.class_to_idx[class_name]))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int] | None:
        path, target = self.samples[index]
        image = pil_loader(path)
        if image is None:
            return None
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def __len__(self) -> int:
        return len(self.samples)


class ManifestImageDataset(VisionDataset):
    """Dataset backed by a fixed manifest split."""

    def __init__(
        self,
        image_root: str | Path,
        rows: list[dict[str, str]],
        split: str,
        classes: list[str],
        transform: Any = None,
    ) -> None:
        root_path = Path(image_root)
        super().__init__(str(root_path), transform=transform)
        self.root_path = root_path
        self.split = split
        self.classes = classes
        self.class_to_idx = {class_name: index for index, class_name in enumerate(classes)}
        self.samples: list[tuple[Path, int]] = []
        for row in rows:
            if row["split"] != split:
                continue
            self.samples.append((root_path / row["relative_path"], int(row["class_index"])))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int] | None:
        path, target = self.samples[index]
        image = pil_loader(path)
        if image is None:
            return None
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def __len__(self) -> int:
        return len(self.samples)


def safe_collate(batch: list[Any]) -> Any:
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return torch.utils.data.default_collate(batch)


def list_image_rows(image_root: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    root_path = Path(image_root)
    valid_extensions = {extension.lower() for extension in IMG_EXTENSIONS}
    classes = [
        path.name
        for path in sorted(root_path.iterdir(), key=lambda item: item.name.lower())
        if path.is_dir()
    ]
    class_to_idx = {class_name: index for index, class_name in enumerate(classes)}
    rows: list[dict[str, str]] = []
    for class_name in classes:
        class_dir = root_path / class_name
        for image_path in sorted(class_dir.iterdir(), key=lambda item: item.name.lower()):
            if not image_path.is_file() or image_path.suffix.lower() not in valid_extensions:
                continue
            rows.append({
                "relative_path": image_path.relative_to(root_path).as_posix(),
                "label": class_name,
                "class_index": str(class_to_idx[class_name]),
                "split": "",
            })
    return rows, classes


def assign_stratified_splits(
    rows: list[dict[str, str]],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> list[dict[str, str]]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum:.6f}")

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["label"], []).append(row)

    rng = random.Random(seed)
    split_rows: list[dict[str, str]] = []
    for label in sorted(grouped):
        label_rows = list(grouped[label])
        rng.shuffle(label_rows)
        total = len(label_rows)
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)
        for index, row in enumerate(label_rows):
            item = dict(row)
            if index < train_count:
                item["split"] = "train"
            elif index < train_count + val_count:
                item["split"] = "val"
            else:
                item["split"] = "test"
            split_rows.append(item)

    return sorted(split_rows, key=lambda row: (row["split"], row["label"], row["relative_path"]))


def summarize_manifest(rows: list[dict[str, str]], classes: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total": len(rows),
        "classes": classes,
        "splits": {},
    }
    for split in ["train", "val", "test"]:
        split_rows = [row for row in rows if row["split"] == split]
        label_counts = {
            class_name: sum(1 for row in split_rows if row["label"] == class_name)
            for class_name in classes
        }
        summary["splits"][split] = {
            "total": len(split_rows),
            "label_counts": label_counts,
        }
    return summary


def write_manifest(rows: list[dict[str, str]], manifest_path: str | Path) -> None:
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["relative_path", "label", "class_index", "split"])
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(manifest_path: str | Path) -> list[dict[str, str]]:
    with Path(manifest_path).open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def prepare_manifest(
    *,
    image_root: str | Path,
    manifest_path: str | Path,
    summary_path: str | Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    force: bool,
) -> tuple[list[dict[str, str]], list[str], dict[str, Any]]:
    manifest = Path(manifest_path)
    if manifest.exists() and not force:
        rows = read_manifest(manifest)
        classes = sorted({row["label"] for row in rows})
        summary = summarize_manifest(rows, classes)
        summary.update({
            "image_root": public_path(image_root),
            "manifest_path": public_path(manifest),
            "random_seed": seed,
            "split_ratio": {
                "train": train_ratio,
                "val": val_ratio,
                "test": test_ratio,
            },
        })
        Path(summary_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return rows, classes, summary

    rows, classes = list_image_rows(image_root)
    rows = assign_stratified_splits(
        rows,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    summary = summarize_manifest(rows, classes)
    summary.update({
        "image_root": public_path(image_root),
        "manifest_path": public_path(manifest),
        "random_seed": seed,
        "split_ratio": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
    })
    write_manifest(rows, manifest)
    Path(summary_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows, classes, summary


def build_transforms(preset: str) -> tuple[Any, Any]:
    normalize = transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD)
    if preset == "augmented":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            transforms.ToTensor(),
            normalize,
        ])
        eval_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            normalize,
        ])
        return train_transform, eval_transform

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, eval_transform


def get_dataloaders(
    data_dir: str | Path,
    batch_size: int,
    num_workers: int,
    transform_preset: str,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader, SafeImageFolder, SafeImageFolder]:
    data_path = Path(data_dir)
    train_transform, val_transform = build_transforms(transform_preset)
    train_dataset = SafeImageFolder(data_path / "train", transform=train_transform)
    val_dataset = SafeImageFolder(data_path / "val", transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=safe_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=safe_collate,
    )
    return train_loader, val_loader, train_dataset, val_dataset


def get_manifest_dataloaders(
    image_root: str | Path,
    rows: list[dict[str, str]],
    classes: list[str],
    batch_size: int,
    num_workers: int,
    transform_preset: str,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader, DataLoader, ManifestImageDataset, ManifestImageDataset, ManifestImageDataset]:
    train_transform, eval_transform = build_transforms(transform_preset)
    train_dataset = ManifestImageDataset(image_root, rows, "train", classes, transform=train_transform)
    val_dataset = ManifestImageDataset(image_root, rows, "val", classes, transform=eval_transform)
    test_dataset = ManifestImageDataset(image_root, rows, "test", classes, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=safe_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=safe_collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=safe_collate,
    )
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        if batch is None:
            continue
        images, labels = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predicted = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    if total == 0:
        return 0.0, 0.0
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        if batch is None:
            continue
        images, labels = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        predicted = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    if total == 0:
        return 0.0, 0.0
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate_with_predictions(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    for batch in loader:
        if batch is None:
            continue
        images, labels = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        predicted = outputs.argmax(dim=1)

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(predicted.detach().cpu().tolist())

    if total == 0:
        return 0.0, 0.0, y_true, y_pred
    return running_loss / total, correct / total, y_true, y_pred


def confusion_matrix_from_predictions(y_true: list[int], y_pred: list[int], class_count: int) -> list[list[int]]:
    matrix = [[0 for _ in range(class_count)] for _ in range(class_count)]
    for actual, predicted in zip(y_true, y_pred):
        matrix[actual][predicted] += 1
    return matrix


def classification_report_from_matrix(matrix: list[list[int]], classes: list[str]) -> dict[str, Any]:
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[index][index] for index in range(len(classes)))
    report: dict[str, Any] = {
        "accuracy": correct / total if total else 0.0,
        "classes": {},
    }
    precision_values = []
    recall_values = []
    f1_values = []
    for index, class_name in enumerate(classes):
        tp = matrix[index][index]
        fp = sum(matrix[row][index] for row in range(len(classes))) - tp
        fn = sum(matrix[index]) - tp
        support = sum(matrix[index])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        report["classes"][class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    report["macro_avg"] = {
        "precision": sum(precision_values) / len(precision_values) if precision_values else 0.0,
        "recall": sum(recall_values) / len(recall_values) if recall_values else 0.0,
        "f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "support": total,
    }
    return report


def save_confusion_matrix(matrix: list[list[int]], classes: list[str], output_dir: Path) -> None:
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["actual\\predicted", *classes])
        for class_name, row in zip(classes, matrix):
            writer.writerow([class_name, *row])

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(classes)), labels=classes)
    ax.set_yticks(range(len(classes)), labels=classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            ax.text(col_index, row_index, str(value), ha="center", va="center", color="#222222")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def save_test_outputs(
    *,
    model: nn.Module,
    test_loader: DataLoader | None,
    criterion: nn.Module,
    device: torch.device,
    classes: list[str],
    output_dir: Path,
) -> dict[str, Any] | None:
    if test_loader is None:
        return None
    test_loss, test_acc, y_true, y_pred = evaluate_with_predictions(model, test_loader, criterion, device)
    matrix = confusion_matrix_from_predictions(y_true, y_pred, len(classes))
    report = classification_report_from_matrix(matrix, classes)
    report["test_loss"] = test_loss
    report["test_accuracy"] = test_acc
    (output_dir / "classification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_confusion_matrix(matrix, classes, output_dir)
    with (output_dir / "test_metrics.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["test_loss", "test_accuracy", "macro_precision", "macro_recall", "macro_f1"])
        writer.writeheader()
        writer.writerow({
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "macro_precision": report["macro_avg"]["precision"],
            "macro_recall": report["macro_avg"]["recall"],
            "macro_f1": report["macro_avg"]["f1"],
        })
    return report


def build_optimizer(name: str, model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(
    name: str,
    optimizer: torch.optim.Optimizer,
    *,
    factor: float,
    patience: int,
    min_lr: float,
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    if name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )
    return None


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def save_history_csv(history: list[dict[str, float | int]], output_dir: Path) -> None:
    if not history:
        return
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def plot_curves(history: list[dict[str, float | int]], output_dir: Path) -> None:
    if not history:
        return
    epochs = [int(row["epoch"]) for row in history]
    train_losses = [float(row["train_loss"]) for row in history]
    val_losses = [float(row["val_loss"]) for row in history]
    train_accs = [float(row["train_acc"]) for row in history]
    val_accs = [float(row["val_acc"]) for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_losses, label="Train Loss")
    axes[0].plot(epochs, val_losses, label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].legend()

    axes[1].plot(epochs, train_accs, label="Train Acc")
    axes[1].plot(epochs, val_accs, label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curve")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close(fig)


def build_arg_parser(default_output_dir: Path, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data_dir", type=Path, default=Path("data"), help="Directory containing train/ and val/.")
    parser.add_argument("--use_manifest", action="store_true", help="Use a fixed manifest with train/val/test splits.")
    parser.add_argument("--image_root", type=Path, default=None, help="Image root containing class folders, e.g. PetImages/.")
    parser.add_argument("--manifest_path", type=Path, default=None, help="Path to manifest.csv. Created if missing.")
    parser.add_argument("--force_manifest", action="store_true", help="Regenerate manifest even if it already exists.")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default="adam")
    parser.add_argument("--lr_scheduler", choices=["none", "reduce_on_plateau"], default="none")
    parser.add_argument("--lr_scheduler_factor", type=float, default=0.5)
    parser.add_argument("--lr_scheduler_patience", type=int, default=2)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--transform_preset", choices=["baseline", "augmented"], default="baseline")
    parser.add_argument("--model_variant", choices=sorted(MODEL_VARIANTS), default="alexnet")
    parser.add_argument("--output_dir", type=Path, default=default_output_dir)
    return parser


def run_training_experiment(iteration_name: str, experiment_note: str, args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader: DataLoader | None = None
    manifest_summary: dict[str, Any] | None = None

    if args.use_manifest:
        if args.image_root is None:
            raise ValueError("--image_root is required when --use_manifest is set")
        manifest_path = Path(args.manifest_path) if args.manifest_path else output_dir / "manifest.csv"
        rows, classes, manifest_summary = prepare_manifest(
            image_root=args.image_root,
            manifest_path=manifest_path,
            summary_path=output_dir / "split_summary.json",
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            force=args.force_manifest,
        )
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = get_manifest_dataloaders(
            args.image_root,
            rows,
            classes,
            args.batch_size,
            args.num_workers,
            args.transform_preset,
            pin_memory=device.type == "cuda",
        )
    else:
        train_loader, val_loader, train_dataset, val_dataset = get_dataloaders(
            args.data_dir,
            args.batch_size,
            args.num_workers,
            args.transform_preset,
            pin_memory=device.type == "cuda",
        )
        test_dataset = None

    model = build_model(args.model_variant, num_classes=len(train_dataset.classes), dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(args.optimizer, model, args.lr, args.weight_decay)
    scheduler = build_scheduler(
        args.lr_scheduler,
        optimizer,
        factor=args.lr_scheduler_factor,
        patience=args.lr_scheduler_patience,
        min_lr=args.min_lr,
    )

    history: list[dict[str, float | int]] = []
    best_val_acc = 0.0
    best_epoch = 0
    start_time = time.time()

    print(f"Device: {device}")
    print(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")
    if test_dataset is not None:
        print(f"Test: {len(test_dataset)} images")
    print(f"Classes: {train_dataset.class_to_idx}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": get_current_lr(optimizer),
        })

        print(
            f"Epoch {epoch:2d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
            f"LR: {get_current_lr(optimizer):.6g}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": train_dataset.class_to_idx,
                    "iteration": iteration_name,
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "args": {key: public_path(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                },
                output_dir / "best_alexnet.pth",
            )
            print(f"  -> saved best model (val_acc={best_val_acc:.4f})")

    elapsed_seconds = time.time() - start_time
    save_history_csv(history, output_dir)
    plot_curves(history, output_dir)

    test_report = None
    best_model_path = output_dir / "best_alexnet.pth"
    if best_model_path.exists() and test_loader is not None:
        checkpoint = torch.load(best_model_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        test_report = save_test_outputs(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
            classes=train_dataset.classes,
            output_dir=output_dir,
        )

    metadata: dict[str, Any] = {
        "model": "AlexNet",
        "model_variant": args.model_variant,
        "iteration": iteration_name,
        "note": experiment_note,
        "random_seed": args.seed,
        "data_dir": public_path(args.data_dir),
        "use_manifest": args.use_manifest,
        "image_root": public_path(args.image_root) if args.image_root else None,
        "manifest_path": public_path(Path(args.manifest_path) if args.manifest_path else output_dir / "manifest.csv")
        if args.use_manifest else None,
        "split_ratio": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        } if args.use_manifest else None,
        "manifest_summary": manifest_summary,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset) if test_dataset is not None else None,
        "classes": train_dataset.classes,
        "class_to_idx": train_dataset.class_to_idx,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "lr_scheduler": args.lr_scheduler,
        "lr_scheduler_factor": args.lr_scheduler_factor,
        "lr_scheduler_patience": args.lr_scheduler_patience,
        "min_lr": args.min_lr,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "transform_preset": args.transform_preset,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_report": test_report,
        "elapsed_seconds": elapsed_seconds,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "experiment_summary.md").write_text(
        "\n".join([
            f"# AlexNet {iteration_name}",
            "",
            "## Experiment Note",
            "",
            experiment_note,
            "",
            "## Model",
            "",
            f"- model_variant: `{args.model_variant}`",
            "",
            "## Data",
            "",
            f"- Train images: {len(train_dataset)}",
            f"- Val images: {len(val_dataset)}",
            f"- Test images: {len(test_dataset) if test_dataset is not None else 'N/A'}",
            f"- Classes: `{train_dataset.class_to_idx}`",
            f"- Manifest: `{metadata['manifest_path']}`",
            "",
            "## Best Validation Result",
            "",
            f"- best_epoch: {best_epoch}",
            f"- best_val_acc: {best_val_acc:.4f}",
            "",
            "## Test Result",
            "",
            f"- test_accuracy: {test_report['test_accuracy']:.4f}" if test_report else "- test_accuracy: N/A",
            f"- test_macro_f1: {test_report['macro_avg']['f1']:.4f}" if test_report else "- test_macro_f1: N/A",
            "",
            "## Outputs",
            "",
            "- `best_alexnet.pth`",
            "- `history.csv`",
            "- `training_curves.png`",
            "- `classification_report.json`",
            "- `confusion_matrix.csv`",
            "- `confusion_matrix.png`",
            "- `test_metrics.csv`",
            "- `manifest.csv`",
            "- `split_summary.json`",
            "- `metrics.json`",
        ]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
