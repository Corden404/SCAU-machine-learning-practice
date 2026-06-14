from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp.AlexNet.common import build_transforms, pil_loader, read_manifest, resolve_image_root_from_config
from exp.AlexNet.model import AlexNet


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "iteration_5"
ITERATION_3_DIR = Path(__file__).resolve().parents[1] / "results" / "iteration_3"
ITERATION_4_DIR = Path(__file__).resolve().parents[1] / "results" / "iteration_4"


class TestManifestDataset(Dataset):
    def __init__(self, image_root: Path, rows: list[dict[str, str]], transform: Any) -> None:
        self.image_root = image_root
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        row = self.rows[index]
        relative_path = row["relative_path"]
        image = pil_loader(self.image_root / relative_path)
        if image is None:
            raise RuntimeError(f"Cannot read image: {self.image_root / relative_path}")
        return self.transform(image), int(row["class_index"]), relative_path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clean_dir(path: Path) -> None:
    if path.exists():
        trash_root = PROJECT_ROOT / ".trash" / "alexnet_iteration_5_rerun"
        trash_root.mkdir(parents=True, exist_ok=True)
        target = trash_root / f"{path.name}_{len(list(trash_root.glob(path.name + '*'))):03d}"
        shutil.move(str(path), str(target))
    path.mkdir(parents=True, exist_ok=True)


def safe_name(relative_path: str) -> str:
    return relative_path.replace("/", "_").replace("\\", "_").replace(" ", "_")


def load_model(checkpoint_path: Path, class_count: int, device: torch.device) -> AlexNet:
    model = AlexNet(num_classes=class_count)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_model(
    *,
    label: str,
    result_dir: Path,
    test_rows: list[dict[str, str]],
    image_root: Path,
    classes: list[str],
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 2,
) -> list[dict[str, Any]]:
    metrics = load_json(result_dir / "metrics.json")
    _, eval_transform = build_transforms(metrics["transform_preset"])
    dataset = TestManifestDataset(image_root, test_rows, eval_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model = load_model(result_dir / "best_alexnet.pth", len(classes), device)

    rows: list[dict[str, Any]] = []
    for images, targets, relative_paths in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1).detach().cpu()
        predictions = probabilities.argmax(dim=1)
        targets_cpu = targets.detach().cpu()

        for idx, relative_path in enumerate(relative_paths):
            actual_index = int(targets_cpu[idx].item())
            predicted_index = int(predictions[idx].item())
            probability_values = [float(value) for value in probabilities[idx].tolist()]
            predicted_prob = probability_values[predicted_index]
            actual_prob = probability_values[actual_index]
            sorted_probs = sorted(probability_values, reverse=True)
            margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
            rows.append({
                "model": label,
                "relative_path": str(relative_path),
                "actual": classes[actual_index],
                "predicted": classes[predicted_index],
                "correct": actual_index == predicted_index,
                "confidence": predicted_prob,
                "actual_prob": actual_prob,
                "margin": margin,
                **{f"prob_{classes[class_index]}": probability_values[class_index] for class_index in range(len(classes))},
            })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_error_thumbnails(model_dir: Path, image_root: Path, errors: list[dict[str, Any]], limit: int = 9999) -> None:
    thumbs_dir = model_dir / "error_samples"
    clean_dir(thumbs_dir)
    for index, row in enumerate(errors[:limit], start=1):
        image = Image.open(image_root / row["relative_path"]).convert("RGB")
        image.thumbnail((224, 224))
        canvas = Image.new("RGB", (224, 254), "white")
        canvas.paste(image, ((224 - image.width) // 2, 0))
        draw = ImageDraw.Draw(canvas)
        text = f"{row['actual']} -> {row['predicted']}  p={row['confidence']:.2f}"
        draw.text((6, 230), text, fill=(180, 0, 0))
        filename = f"{index:04d}_{row['actual']}_as_{row['predicted']}_{safe_name(row['relative_path'])}"
        canvas.save(thumbs_dir / filename, quality=90)


def save_contact_sheet(path: Path, image_root: Path, errors: list[dict[str, Any]], title: str, limit: int = 30) -> None:
    selected = errors[:limit]
    if not selected:
        return
    cell_w, cell_h = 180, 220
    cols = 5
    rows = math.ceil(len(selected) / cols)
    header_h = 34
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h + header_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill=(0, 0, 0))

    for index, row in enumerate(selected):
        image = Image.open(image_root / row["relative_path"]).convert("RGB")
        image.thumbnail((cell_w, cell_w))
        col = index % cols
        row_idx = index // cols
        x = col * cell_w
        y = row_idx * cell_h + header_h
        sheet.paste(image, (x + (cell_w - image.width) // 2, y))
        text = f"{row['actual']}->{row['predicted']} p={row['confidence']:.2f}"
        draw.text((x + 6, y + 184), text, fill=(180, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=90)


def summarize_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [row for row in rows if not row["correct"]]
    total = len(rows)
    by_actual: dict[str, int] = {}
    by_pair: dict[str, int] = {}
    for row in errors:
        by_actual[row["actual"]] = by_actual.get(row["actual"], 0) + 1
        pair = f"{row['actual']}->{row['predicted']}"
        by_pair[pair] = by_pair.get(pair, 0) + 1
    return {
        "total": total,
        "correct": total - len(errors),
        "errors": len(errors),
        "accuracy": (total - len(errors)) / total if total else 0.0,
        "errors_by_actual": by_actual,
        "errors_by_pair": by_pair,
        "mean_error_confidence": sum(float(row["confidence"]) for row in errors) / len(errors) if errors else 0.0,
    }


def build_comparison(
    iteration_3_rows: list[dict[str, Any]],
    iteration_4_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    i3_by_path = {row["relative_path"]: row for row in iteration_3_rows}
    i4_by_path = {row["relative_path"]: row for row in iteration_4_rows}
    i3_errors = {path for path, row in i3_by_path.items() if not row["correct"]}
    i4_errors = {path for path, row in i4_by_path.items() if not row["correct"]}
    common_errors = sorted(i3_errors & i4_errors)
    fixed_by_i4 = sorted(i3_errors - i4_errors)
    new_errors_i4 = sorted(i4_errors - i3_errors)
    return {
        "iteration_3": summarize_predictions(iteration_3_rows),
        "iteration_4": summarize_predictions(iteration_4_rows),
        "common_error_count": len(common_errors),
        "fixed_by_iteration_4_count": len(fixed_by_i4),
        "new_error_in_iteration_4_count": len(new_errors_i4),
        "common_errors": common_errors,
        "fixed_by_iteration_4": fixed_by_i4,
        "new_error_in_iteration_4": new_errors_i4,
    }


def write_path_rows(path: Path, paths: list[str], lookup: dict[str, dict[str, Any]]) -> None:
    rows = [lookup[item] for item in paths]
    write_csv(path, rows)


def write_report(summary: dict[str, Any]) -> None:
    i3 = summary["iteration_3"]
    i4 = summary["iteration_4"]
    lines = [
        "# AlexNet iteration_5 错误样本分析",
        "",
        "## 目的",
        "",
        "本轮不重新训练模型，只加载 iteration_3 与 iteration_4 的最佳权重，在同一份固定 test split 上导出预测与错分样本，用于分析模型错误来源。",
        "",
        "## 总体对比",
        "",
        "| model | accuracy | errors | Cat errors | Dog errors | mean error confidence |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| iteration_3 | {i3['accuracy']:.4f} | {i3['errors']} "
            f"| {i3['errors_by_actual'].get('Cat', 0)} | {i3['errors_by_actual'].get('Dog', 0)} "
            f"| {i3['mean_error_confidence']:.4f} |"
        ),
        (
            f"| iteration_4 | {i4['accuracy']:.4f} | {i4['errors']} "
            f"| {i4['errors_by_actual'].get('Cat', 0)} | {i4['errors_by_actual'].get('Dog', 0)} "
            f"| {i4['mean_error_confidence']:.4f} |"
        ),
        "",
        "## 错分重叠",
        "",
        f"- 两个模型共同错分：{summary['common_error_count']} 张",
        f"- iteration_4 修正了 iteration_3 的错分：{summary['fixed_by_iteration_4_count']} 张",
        f"- iteration_4 新增错分：{summary['new_error_in_iteration_4_count']} 张",
        "",
        "## 初步结论",
        "",
        "- iteration_4 的验证集表现更高，但 test 错分数比 iteration_3 略多，说明增强与 AdamW 的收益没有稳定转化到独立测试集。",
        "- 如果共同错分样本占比较高，说明当前主要瓶颈可能来自数据本身的难样本，例如主体小、遮挡、背景复杂、图像质量差或标签争议。",
        "- 后续应优先人工查看 `common_errors.csv` 和两个 `top_confident_errors.jpg`，再决定是继续改训练策略还是改模型结构。",
        "",
        "## 输出文件",
        "",
        "- `iteration_3/predictions.csv`",
        "- `iteration_3/error_samples.csv`",
        "- `iteration_3/error_samples/`",
        "- `iteration_3/top_confident_errors.jpg`",
        "- `iteration_4/predictions.csv`",
        "- `iteration_4/error_samples.csv`",
        "- `iteration_4/error_samples/`",
        "- `iteration_4/top_confident_errors.jpg`",
        "- `common_errors.csv`",
        "- `fixed_by_iteration_4.csv`",
        "- `new_error_in_iteration_4.csv`",
        "- `comparison_summary.json`",
    ]
    (OUTPUT_DIR / "error_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    i3_metrics = load_json(ITERATION_3_DIR / "metrics.json")
    image_root = resolve_image_root_from_config()
    classes = i3_metrics["classes"]
    manifest_rows = [row for row in read_manifest(ITERATION_3_DIR / "manifest.csv") if row["split"] == "test"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Test samples: {len(manifest_rows)}")

    all_results: dict[str, list[dict[str, Any]]] = {}
    for label, result_dir in [("iteration_3", ITERATION_3_DIR), ("iteration_4", ITERATION_4_DIR)]:
        print(f"Evaluating {label}...")
        rows = predict_model(
            label=label,
            result_dir=result_dir,
            test_rows=manifest_rows,
            image_root=image_root,
            classes=classes,
            device=device,
        )
        model_dir = OUTPUT_DIR / label
        model_dir.mkdir(parents=True, exist_ok=True)
        errors = sorted(
            [row for row in rows if not row["correct"]],
            key=lambda row: float(row["confidence"]),
            reverse=True,
        )
        write_csv(model_dir / "predictions.csv", rows)
        write_csv(model_dir / "error_samples.csv", errors)
        save_error_thumbnails(model_dir, image_root, errors)
        save_contact_sheet(model_dir / "top_confident_errors.jpg", image_root, errors, f"{label} top confident errors")
        all_results[label] = rows

    summary = build_comparison(all_results["iteration_3"], all_results["iteration_4"])
    i3_lookup = {row["relative_path"]: row for row in all_results["iteration_3"]}
    i4_lookup = {row["relative_path"]: row for row in all_results["iteration_4"]}
    write_path_rows(OUTPUT_DIR / "common_errors.csv", summary["common_errors"], i3_lookup)
    write_path_rows(OUTPUT_DIR / "fixed_by_iteration_4.csv", summary["fixed_by_iteration_4"], i3_lookup)
    write_path_rows(OUTPUT_DIR / "new_error_in_iteration_4.csv", summary["new_error_in_iteration_4"], i4_lookup)
    (OUTPUT_DIR / "comparison_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary)
    print(json.dumps({
        key: value
        for key, value in summary.items()
        if key not in {"common_errors", "fixed_by_iteration_4", "new_error_in_iteration_4"}
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
