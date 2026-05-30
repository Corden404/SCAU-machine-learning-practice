from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.project_config import PROJECT_ROOT, get_data_root, get_dataset_path


RANDOM_SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_FULL_ANALYSIS_SECONDS = 20 * 60


@dataclass
class ExtractionResult:
    zip_name: str
    destination_name: str
    extracted: bool
    file_count: int
    elapsed_seconds: float


def set_plot_style() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_zip_extract(zip_path: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if not str(target).startswith(str(destination_resolved)):
                raise ValueError(f"Unsafe zip member path: {member.filename}")
            archive.extract(member, destination)


def ensure_extracted(dataset_name: str) -> tuple[Path, ExtractionResult]:
    zip_path = get_dataset_path(dataset_name)
    destination = zip_path.with_suffix("")
    ensure_dir(destination)

    started = time.perf_counter()
    existing_files = [path for path in destination.rglob("*") if path.is_file()]
    extracted = False
    if not existing_files:
        safe_zip_extract(zip_path, destination)
        extracted = True
        existing_files = [path for path in destination.rglob("*") if path.is_file()]

    elapsed = time.perf_counter() - started
    return destination, ExtractionResult(
        zip_name=zip_path.name,
        destination_name=destination.name,
        extracted=extracted,
        file_count=len(existing_files),
        elapsed_seconds=elapsed,
    )


def as_relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def save_json(data: dict[str, Any], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def save_markdown(text: str, output_path: Path) -> None:
    output_path.write_text(text, encoding="utf-8")


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f} 秒"
    return f"{seconds / 60:.2f} 分钟"


def quantile_summary(series: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {}
    quantiles = clean.quantile([0, 0.25, 0.5, 0.75, 1]).to_dict()
    return {
        "min": float(quantiles[0]),
        "q1": float(quantiles[0.25]),
        "median": float(quantiles[0.5]),
        "q3": float(quantiles[0.75]),
        "max": float(quantiles[1]),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=0)),
    }


def fmt_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "无"
    if isinstance(value, float):
        if math.isnan(value):
            return "无"
        return f"{value:.{digits}f}"
    return str(value)


def fmt_summary(summary: dict[str, float], digits: int = 2) -> str:
    if not summary:
        return "无可用统计"
    return (
        f"min={summary['min']:.{digits}f}, "
        f"Q1={summary['q1']:.{digits}f}, "
        f"median={summary['median']:.{digits}f}, "
        f"Q3={summary['q3']:.{digits}f}, "
        f"max={summary['max']:.{digits}f}, "
        f"mean={summary['mean']:.{digits}f}"
    )


def find_ionosphere_table(extracted_root: Path) -> Path:
    candidates = sorted(
        [
            path
            for path in extracted_root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".data", ".csv", ".txt"}
        ]
    )
    if not candidates:
        raise FileNotFoundError("No tabular data file found in Ionosphere extraction.")

    scored: list[tuple[int, int, Path]] = []
    for path in candidates:
        try:
            frame = pd.read_csv(path, header=None)
        except Exception:
            continue
        scored.append((frame.shape[0], frame.shape[1], path))

    if not scored:
        raise ValueError("No readable table file found in Ionosphere extraction.")

    scored.sort(reverse=True)
    return scored[0][2]


def plot_bar(counter: dict[str, int], title: str, output_path: Path, xlabel: str = "", ylabel: str = "数量") -> None:
    labels = list(counter.keys())
    values = [counter[label] for label in labels]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    bars = ax.bar(labels, values, color=["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#76B7B2"][: len(labels)])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f"{int(height)}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def analyze_ionosphere(output_dir: Path) -> dict[str, Any]:
    ensure_dir(output_dir)
    extracted_root, extraction = ensure_extracted("ionosphere")
    table_path = find_ionosphere_table(extracted_root)

    df = pd.read_csv(table_path, header=None).dropna(how="all")
    feature_names = [f"f{i + 1}" for i in range(df.shape[1] - 1)]
    label_name = "label"
    df.columns = feature_names + [label_name]

    features = df[feature_names].apply(pd.to_numeric, errors="coerce")
    labels = df[label_name].astype(str).str.strip()

    sample_count = int(df.shape[0])
    feature_count = int(features.shape[1])
    label_counts = labels.value_counts().sort_index().to_dict()
    label_ratios = (labels.value_counts(normalize=True).sort_index() * 100).round(2).to_dict()
    majority_ratio = float(labels.value_counts(normalize=True).max())

    missing_by_column = df.isna().sum()
    missing_cells = int(missing_by_column.sum())
    duplicate_rows = int(df.duplicated().sum())
    duplicate_features = int(features.duplicated().sum())

    numeric_missing_cells = int(features.isna().sum().sum())
    feature_unique_counts = features.nunique(dropna=False)
    constant_features = [str(name) for name, value in feature_unique_counts.items() if value <= 1]

    variances = features.var(ddof=0)
    stds = features.std(ddof=0)
    ranges = features.max() - features.min()
    nonzero_ranges = ranges[ranges > 0]
    range_ratio = float(nonzero_ranges.max() / nonzero_ranges.min()) if not nonzero_ranges.empty else None
    nonzero_stds = stds[stds > 0]
    std_ratio = float(nonzero_stds.max() / nonzero_stds.min()) if not nonzero_stds.empty else None

    outlier_counts: dict[str, int] = {}
    outlier_mask = pd.DataFrame(False, index=features.index, columns=features.columns)
    for column in features.columns:
        series = features[column].dropna()
        if series.empty:
            outlier_counts[column] = 0
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            outlier_counts[column] = 0
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        column_mask = (features[column] < lower) | (features[column] > upper)
        outlier_mask[column] = column_mask.fillna(False)
        outlier_counts[column] = int(column_mask.sum())
    outlier_rows = int(outlier_mask.any(axis=1).sum())
    top_outlier_features = dict(sorted(outlier_counts.items(), key=lambda item: item[1], reverse=True)[:10])

    corr = features.corr(numeric_only=True)
    high_corr_pairs: list[dict[str, Any]] = []
    max_corr_pair: dict[str, Any] | None = None
    for i, col_a in enumerate(corr.columns):
        for col_b in corr.columns[i + 1 :]:
            value = corr.loc[col_a, col_b]
            if pd.isna(value):
                continue
            abs_value = abs(float(value))
            pair = {"feature_a": str(col_a), "feature_b": str(col_b), "corr": float(value), "abs_corr": abs_value}
            if max_corr_pair is None or abs_value > max_corr_pair["abs_corr"]:
                max_corr_pair = pair
            if abs_value >= 0.9:
                high_corr_pairs.append(pair)
    high_corr_pairs.sort(key=lambda item: item["abs_corr"], reverse=True)

    plot_bar({str(k): int(v) for k, v in label_counts.items()}, "Ionosphere 类别分布", output_dir / "class_distribution.png", xlabel="类别")

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.boxplot([features[column].dropna().values for column in features.columns], tick_labels=features.columns, showfliers=False)
    ax.set_title("Ionosphere 特征分布箱线图（隐藏离群点）")
    ax.set_xlabel("特征")
    ax.set_ylabel("取值")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(output_dir / "feature_boxplot.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7.2))
    im = ax.imshow(corr.fillna(0).values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Ionosphere 特征相关性热力图")
    ax.set_xticks(range(feature_count))
    ax.set_yticks(range(feature_count))
    ax.set_xticklabels(feature_names, rotation=90, fontsize=6)
    ax.set_yticklabels(feature_names, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_heatmap.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    variances.sort_values(ascending=False).plot(kind="bar", ax=ax, color="#4E79A7")
    ax.set_title("Ionosphere 特征方差")
    ax.set_xlabel("特征")
    ax.set_ylabel("方差")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(output_dir / "feature_variance.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    top_series = pd.Series(top_outlier_features)
    top_series.plot(kind="bar", ax=ax, color="#E15759")
    ax.set_title("Ionosphere 离群值数量最多的特征")
    ax.set_xlabel("特征")
    ax.set_ylabel("IQR 离群值数量")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / "outlier_features.png")
    plt.close(fig)

    result = {
        "extraction": extraction.__dict__,
        "source_file": as_relative(table_path, extracted_root),
        "sample_count": sample_count,
        "feature_count": feature_count,
        "class_count": len(label_counts),
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "label_ratios_percent": {str(k): float(v) for k, v in label_ratios.items()},
        "majority_ratio": majority_ratio,
        "missing_cells": missing_cells,
        "numeric_missing_cells": numeric_missing_cells,
        "duplicate_rows": duplicate_rows,
        "duplicate_features": duplicate_features,
        "constant_features": constant_features,
        "range_ratio": range_ratio,
        "std_ratio": std_ratio,
        "top_outlier_features": top_outlier_features,
        "outlier_rows": outlier_rows,
        "outlier_row_ratio": outlier_rows / sample_count if sample_count else 0,
        "high_corr_pair_count": len(high_corr_pairs),
        "top_high_corr_pairs": high_corr_pairs[:15],
        "max_corr_pair": max_corr_pair,
        "feature_range_summary": {str(k): float(v) for k, v in ranges.describe().to_dict().items()},
        "feature_std_summary": {str(k): float(v) for k, v in stds.describe().to_dict().items()},
        "plots": [
            "class_distribution.png",
            "feature_boxplot.png",
            "correlation_heatmap.png",
            "feature_variance.png",
            "outlier_features.png",
        ],
    }
    save_json(result, output_dir / "analysis_results.json")
    save_markdown(render_ionosphere_report(result), output_dir / "数据解读.md")
    return result


def render_ionosphere_report(result: dict[str, Any]) -> str:
    labels_text = ", ".join(
        f"{label}: {count} ({result['label_ratios_percent'][label]:.2f}%)"
        for label, count in result["label_counts"].items()
    )
    constant_features = result["constant_features"] or []
    constant_text = "、".join(constant_features) if constant_features else "未发现"
    top_outliers = ", ".join(f"{feature}: {count}" for feature, count in result["top_outlier_features"].items())
    high_corr_text = "未发现 |r| >= 0.90 的高度相关特征对"
    if result["top_high_corr_pairs"]:
        high_corr_text = "；".join(
            f"{pair['feature_a']}-{pair['feature_b']}: r={pair['corr']:.3f}"
            for pair in result["top_high_corr_pairs"][:8]
        )
    max_corr = result["max_corr_pair"] or {}

    return f"""# Ionosphere 数据解读

## 1. 分析范围

本次预分析对象为 Ionosphere 电离层雷达回波二分类数据集。数据从本地数据目录中的压缩包读取，并解压到本地数据目录下与压缩包同名的文件夹中，报告不记录本地绝对路径。

- 解压状态：{"本次新解压" if result["extraction"]["extracted"] else "已存在解压文件，本次复用"}
- 解压后文件数：{result["extraction"]["file_count"]}
- 解压耗时：{format_seconds(result["extraction"]["elapsed_seconds"])}
- 表格文件：`{result["source_file"]}`

## 2. 数据基本信息

- 样本数量：{result["sample_count"]}
- 特征数量：{result["feature_count"]}
- 类别数量：{result["class_count"]}
- 类别分布：{labels_text}
- 多数类占比：{result["majority_ratio"] * 100:.2f}%

![类别分布](class_distribution.png)

数据集规模很小，只有 {result["sample_count"]} 条样本，但有 {result["feature_count"]} 个特征，属于“小样本 + 相对高维特征”的结构化分类任务。这个特点会直接影响后续建模：单次训练集/测试集划分的结果可能波动较大，因此不能只看一次划分下的准确率，应加入分层划分、K 折交叉验证和多次随机划分对比。

## 3. 类别比例是否平衡

类别比例不是完全均衡，但多数类占比为 {result["majority_ratio"] * 100:.2f}%，没有达到极端不平衡程度。后续训练和评估时建议：

1. 使用 `stratify` 进行分层划分，保持训练集、验证集和测试集类别比例一致；
2. 除 accuracy 外，同时报告 precision、recall、F1 和 ROC-AUC；
3. 对 SVM、逻辑回归等模型关注正负类召回率差异，避免模型只偏向多数类。

## 4. 缺失值与重复样本检查

- 总缺失单元格数量：{result["missing_cells"]}
- 数值特征缺失单元格数量：{result["numeric_missing_cells"]}
- 完全重复样本数量：{result["duplicate_rows"]}
- 特征完全重复但标签可能不同的样本数量：{result["duplicate_features"]}

从缺失值角度看，该数据集可以直接进入传统机器学习流程，不需要复杂填补。重复样本数量也没有对建模造成明显风险。后续代码仍建议保留缺失值和重复值检查步骤，作为报告中“数据质量控制”的组成部分。

## 5. 特征分布与尺度

![特征分布箱线图](feature_boxplot.png)

![特征方差](feature_variance.png)

- 常量或近似无信息特征：{constant_text}
- 非零特征取值范围最大/最小比例：{fmt_number(result["range_ratio"])}
- 非零特征标准差最大/最小比例：{fmt_number(result["std_ratio"])}

Ionosphere 的多数特征位于相近数值范围内，但仍建议对逻辑回归和 SVM 使用标准化。原因是这两类模型依赖距离、间隔或线性系数尺度，标准化能够减少某些特征因尺度差异而被放大的风险。决策树、随机森林和 AdaBoost 对特征尺度不敏感，可以不标准化；但为了实验流程统一，可以通过 `Pipeline` 对需要标准化的模型单独处理。

## 6. 异常值检查

![离群值数量最多的特征](outlier_features.png)

- 至少在一个特征上被 IQR 规则标记为离群的样本数：{result["outlier_rows"]}
- 离群样本占比：{result["outlier_row_ratio"] * 100:.2f}%
- 离群值较多的特征：{top_outliers if top_outliers else "未发现明显离群集中"}

这里的“离群值”只代表统计分布上的异常点，不等于错误数据。Ionosphere 是雷达信号数据，极端数值可能对应真实信号差异。后续不建议在预处理阶段直接删除这些样本，而应保留原始数据，并通过模型对比观察异常值对 SVM、逻辑回归和树模型的影响。

## 7. 特征相关性

![特征相关性热力图](correlation_heatmap.png)

- 高相关特征对数量（|r| >= 0.90）：{result["high_corr_pair_count"]}
- 相关性最高的特征对：{max_corr.get("feature_a", "无")}-{max_corr.get("feature_b", "无")}，r={fmt_number(max_corr.get("corr"))}
- 主要高度相关特征对：{high_corr_text}

相关性分析说明部分特征之间可能存在冗余信息。对于决策树和随机森林，这类冗余通常不会造成严重问题；对于逻辑回归和 SVM，相关特征可能影响模型系数解释和决策边界稳定性。后续可以在改进实验中加入特征选择或 PCA 作为可选方向，但第一轮建模建议先保留全部特征，保证流程简单、结果可解释。

## 8. 数据是否适合当前任务

该数据集适合本项目的传统机器学习任务，原因如下：

1. 特征已经是结构化数值形式，适合逻辑回归、SVM、决策树、随机森林和 AdaBoost；
2. 样本数量小，训练速度快，便于进行多次划分、交叉验证和调参；
3. 特征数量相对样本数较多，适合讨论小样本场景下的过拟合和稳定性问题；
4. 二分类评价指标清晰，适合展示混淆矩阵、F1、ROC-AUC 等结果。

## 9. 对后续建模的影响

后续建模建议如下：

1. 数据划分使用分层抽样，避免小样本下类别比例波动；
2. SVM 和逻辑回归使用 `StandardScaler`；
3. 决策树限制 `max_depth`，观察模型复杂度对过拟合的影响；
4. 随机森林和 AdaBoost 作为集成学习模型，与单棵决策树进行稳定性对比；
5. 评价方式使用单次划分 + K 折交叉验证 + 多次随机划分，重点分析结果波动；
6. 不在预分析阶段删除离群样本，先保留原始数据进行基准实验。

## 10. 小结

Ionosphere 数据质量较好，缺失值和重复样本问题不突出，主要挑战不是清洗，而是小样本条件下模型评估是否稳定。因此本数据集的报告重点应放在“标准化是否影响 SVM/逻辑回归”“单棵树是否过拟合”“集成学习是否更稳定”“交叉验证是否比单次划分更可靠”这些问题上。
"""


def infer_catdog_label(path: Path) -> str:
    for part in path.parts:
        normalized = part.lower().replace("_", " ").replace("-", " ").strip()
        if normalized in {"cat", "cats"}:
            return "Cat"
        if normalized in {"dog", "dogs"}:
            return "Dog"
    parent = path.parent.name.lower()
    if parent.startswith("cat"):
        return "Cat"
    if parent.startswith("dog"):
        return "Dog"
    return "Unknown"


def find_image_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def stratified_sample(paths: list[Path], fraction: float) -> list[Path]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        grouped[infer_catdog_label(path)].append(path)
    rng = random.Random(RANDOM_SEED)
    sample: list[Path] = []
    for label_paths in grouped.values():
        count = max(1, int(round(len(label_paths) * fraction))) if label_paths else 0
        sample.extend(rng.sample(label_paths, min(count, len(label_paths))))
    return sorted(sample)


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average_hash(image: Image.Image) -> str:
    small = image.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
    arr = np.asarray(small, dtype=np.float32)
    bits = arr > arr.mean()
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def channel_count(mode: str) -> int:
    return {
        "1": 1,
        "L": 1,
        "P": 1,
        "RGB": 3,
        "RGBA": 4,
        "CMYK": 4,
        "YCbCr": 3,
        "I": 1,
        "F": 1,
    }.get(mode, 0)


def analyze_one_image(path: Path, root: Path) -> dict[str, Any]:
    label = infer_catdog_label(path)
    record: dict[str, Any] = {
        "relative_path": as_relative(path, root),
        "absolute_path": str(path),
        "label": label,
        "valid": False,
        "error": "",
    }
    try:
        record["file_size_kb"] = path.stat().st_size / 1024
        record["exact_hash"] = file_md5(path)
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            mode = image.mode
            rgb = image.convert("RGB")
            gray = rgb.convert("L").resize((128, 128), Image.Resampling.BILINEAR)
            gray_arr = np.asarray(gray, dtype=np.float32)
            grad_y, grad_x = np.gradient(gray_arr)
            sharpness = float(np.var(grad_x) + np.var(grad_y))

            rgb_small = rgb.resize((64, 64), Image.Resampling.BILINEAR)
            rgb_arr = np.asarray(rgb_small, dtype=np.float32) / 255.0
            channel_mean = rgb_arr.reshape(-1, 3).mean(axis=0)
            channel_std = rgb_arr.reshape(-1, 3).std(axis=0)

            record.update(
                {
                    "valid": True,
                    "width": int(width),
                    "height": int(height),
                    "aspect_ratio": float(width / height) if height else None,
                    "mode": mode,
                    "channels": channel_count(mode),
                    "brightness": float(gray_arr.mean()),
                    "contrast": float(gray_arr.std(ddof=0)),
                    "sharpness": sharpness,
                    "average_hash": average_hash(rgb),
                    "r_mean": float(channel_mean[0]),
                    "g_mean": float(channel_mean[1]),
                    "b_mean": float(channel_mean[2]),
                    "r_std": float(channel_std[0]),
                    "g_std": float(channel_std[1]),
                    "b_std": float(channel_std[2]),
                }
            )
    except Exception as exc:
        record["error"] = type(exc).__name__
    return record


def analyze_images(paths: list[Path], root: Path) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    records = [analyze_one_image(path, root) for path in paths]
    return records, time.perf_counter() - started


def duplicate_summary(records: list[dict[str, Any]], hash_key: str) -> dict[str, Any]:
    groups: dict[str, list[str]] = defaultdict(list)
    group_labels: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if not record.get("valid") or not record.get(hash_key):
            continue
        groups[str(record[hash_key])].append(record["relative_path"])
        group_labels[str(record[hash_key])].add(record["label"])

    duplicate_groups = [paths for paths in groups.values() if len(paths) > 1]
    duplicate_groups.sort(key=len, reverse=True)
    cross_label_groups = [
        groups[hash_value]
        for hash_value, labels in group_labels.items()
        if len(groups[hash_value]) > 1 and len(labels - {"Unknown"}) > 1
    ]
    cross_label_groups.sort(key=len, reverse=True)
    return {
        "group_count": len(duplicate_groups),
        "duplicate_image_count": int(sum(len(paths) - 1 for paths in duplicate_groups)),
        "top_groups": duplicate_groups[:10],
        "cross_label_group_count": len(cross_label_groups),
        "cross_label_duplicate_image_count": int(sum(len(paths) for paths in cross_label_groups)),
        "cross_label_top_groups": cross_label_groups[:10],
    }


def make_contact_sheet(records: list[dict[str, Any]], output_path: Path, title: str, max_images: int = 20, columns: int = 5) -> None:
    valid_records = [record for record in records if record.get("valid")]
    if not valid_records:
        return

    selected = valid_records[:max_images]
    thumb_w, thumb_h = 170, 150
    label_h = 28
    title_h = 42
    rows = math.ceil(len(selected) / columns)
    sheet = Image.new("RGB", (columns * thumb_w, title_h + rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 12), title, fill=(20, 20, 20))

    for index, record in enumerate(selected):
        row = index // columns
        column = index % columns
        x = column * thumb_w
        y = title_h + row * (thumb_h + label_h)
        try:
            with Image.open(record["absolute_path"]) as image:
                image = image.convert("RGB")
                thumb = ImageOps.contain(image, (thumb_w - 10, thumb_h - 10), Image.Resampling.LANCZOS)
                paste_x = x + (thumb_w - thumb.width) // 2
                paste_y = y + (thumb_h - thumb.height) // 2
                sheet.paste(thumb, (paste_x, paste_y))
        except Exception:
            continue
        name = Path(record["relative_path"]).name
        text = f"{record['label']} | {name}"
        draw.text((x + 6, y + thumb_h + 4), text[:28], fill=(30, 30, 30))

    sheet.save(output_path)


def render_quality_extremes(records: list[dict[str, Any]], output_path: Path) -> None:
    valid = [record for record in records if record.get("valid")]
    if not valid:
        return

    darkest = sorted(valid, key=lambda item: item.get("brightness", 0))[:4]
    brightest = sorted(valid, key=lambda item: item.get("brightness", 0), reverse=True)[:4]
    least_sharp = sorted(valid, key=lambda item: item.get("sharpness", 0))[:4]
    selected = darkest + brightest + least_sharp
    make_contact_sheet(selected, output_path, "Darkest / brightest / lowest-sharpness examples", max_images=12, columns=4)


def plot_cats_results(records: list[dict[str, Any]], all_label_counts: dict[str, int], output_dir: Path) -> None:
    valid_df = pd.DataFrame([record for record in records if record.get("valid")])
    plot_bar(all_label_counts, "Cats vs. Dogs 类别分布（全部文件）", output_dir / "class_distribution.png", xlabel="类别")

    if valid_df.empty:
        return

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for label, group in valid_df.groupby("label"):
        ax.scatter(group["width"], group["height"], s=12, alpha=0.45, label=label)
    ax.set_title("图片宽度与高度分布")
    ax.set_xlabel("宽度（px）")
    ax.set_ylabel("高度（px）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "image_dimensions.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.4))
    valid_df["aspect_ratio"].plot(kind="hist", bins=40, ax=ax, color="#59A14F", edgecolor="white")
    ax.set_title("图片宽高比分布")
    ax.set_xlabel("宽高比")
    fig.tight_layout()
    fig.savefig(output_dir / "aspect_ratio_hist.png")
    plt.close(fig)

    mode_counts = valid_df["mode"].value_counts().to_dict()
    plot_bar({str(k): int(v) for k, v in mode_counts.items()}, "图片颜色模式分布", output_dir / "mode_distribution.png", xlabel="颜色模式")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, column, title, color in [
        (axes[0], "brightness", "亮度分布", "#4E79A7"),
        (axes[1], "contrast", "对比度分布", "#F28E2B"),
        (axes[2], "sharpness", "清晰度指标分布", "#E15759"),
    ]:
        valid_df[column].plot(kind="hist", bins=40, ax=ax, color=color, edgecolor="white")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_dir / "quality_histograms.png")
    plt.close(fig)


def split_counts(label_counts: dict[str, int]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for label, count in label_counts.items():
        train = int(round(count * 0.8))
        val = int(round(count * 0.1))
        test = count - train - val
        result[label] = {"train": train, "val": val, "test": test}
    return result


def analyze_cats_vs_dogs(output_dir: Path) -> dict[str, Any]:
    ensure_dir(output_dir)
    extracted_root, extraction = ensure_extracted("cats_vs_dogs")
    image_files = find_image_files(extracted_root)
    all_label_counts = Counter(infer_catdog_label(path) for path in image_files)

    sample_paths = stratified_sample(image_files, 0.05)
    sample_records, sample_elapsed = analyze_images(sample_paths, extracted_root)
    estimated_full_seconds = sample_elapsed / max(len(sample_paths), 1) * len(image_files)
    run_full = estimated_full_seconds <= MAX_FULL_ANALYSIS_SECONDS

    if run_full:
        records, analysis_elapsed = analyze_images(image_files, extracted_root)
        analysis_mode = "full"
        analyzed_count = len(image_files)
    else:
        records = sample_records
        analysis_elapsed = sample_elapsed
        analysis_mode = "sample"
        analyzed_count = len(sample_paths)

    valid_records = [record for record in records if record.get("valid")]
    invalid_records = [record for record in records if not record.get("valid")]
    valid_df = pd.DataFrame(valid_records)

    exact_duplicates = duplicate_summary(records, "exact_hash")
    perceptual_duplicates = duplicate_summary(records, "average_hash")

    if valid_df.empty:
        raise ValueError("No valid images could be analyzed.")

    low_sharpness_threshold = float(valid_df["sharpness"].quantile(0.05))
    quality_flags = {
        "dark_images": int((valid_df["brightness"] < 40).sum()),
        "bright_images": int((valid_df["brightness"] > 215).sum()),
        "low_contrast_images": int((valid_df["contrast"] < 20).sum()),
        "relative_low_sharpness_images": int((valid_df["sharpness"] <= low_sharpness_threshold).sum()),
        "extreme_aspect_ratio_images": int(((valid_df["aspect_ratio"] < 0.5) | (valid_df["aspect_ratio"] > 2.0)).sum()),
        "small_images": int(((valid_df["width"] < 100) | (valid_df["height"] < 100)).sum()),
    }

    rng = random.Random(RANDOM_SEED)
    random_records = valid_records.copy()
    rng.shuffle(random_records)

    plot_cats_results(records, {str(k): int(v) for k, v in all_label_counts.items()}, output_dir)
    make_contact_sheet(random_records, output_dir / "sample_grid.png", "Random analyzed samples", max_images=20, columns=5)
    render_quality_extremes(valid_records, output_dir / "quality_extremes.png")

    label_counts = {str(k): int(v) for k, v in all_label_counts.items()}
    valid_label_counts = {str(k): int(v) for k, v in valid_df["label"].value_counts().sort_index().to_dict().items()}

    result = {
        "extraction": extraction.__dict__,
        "total_image_files": len(image_files),
        "all_label_counts": label_counts,
        "sample_fraction": 0.05,
        "sample_count": len(sample_paths),
        "sample_elapsed_seconds": sample_elapsed,
        "estimated_full_seconds": estimated_full_seconds,
        "full_threshold_seconds": MAX_FULL_ANALYSIS_SECONDS,
        "analysis_mode": analysis_mode,
        "analyzed_count": analyzed_count,
        "analysis_elapsed_seconds": analysis_elapsed,
        "valid_count": len(valid_records),
        "invalid_count": len(invalid_records),
        "invalid_examples": [
            {"relative_path": item["relative_path"], "error": item.get("error", "")}
            for item in invalid_records[:50]
        ],
        "valid_label_counts": valid_label_counts,
        "width_summary": quantile_summary(valid_df["width"]),
        "height_summary": quantile_summary(valid_df["height"]),
        "aspect_ratio_summary": quantile_summary(valid_df["aspect_ratio"]),
        "file_size_kb_summary": quantile_summary(valid_df["file_size_kb"]),
        "brightness_summary": quantile_summary(valid_df["brightness"]),
        "contrast_summary": quantile_summary(valid_df["contrast"]),
        "sharpness_summary": quantile_summary(valid_df["sharpness"]),
        "mode_counts": {str(k): int(v) for k, v in valid_df["mode"].value_counts().to_dict().items()},
        "channel_counts": {str(k): int(v) for k, v in valid_df["channels"].value_counts().sort_index().to_dict().items()},
        "quality_flags": quality_flags,
        "low_sharpness_threshold": low_sharpness_threshold,
        "exact_duplicates": exact_duplicates,
        "perceptual_hash_duplicates": perceptual_duplicates,
        "rgb_mean": {
            "r": float(valid_df["r_mean"].mean()),
            "g": float(valid_df["g_mean"].mean()),
            "b": float(valid_df["b_mean"].mean()),
        },
        "rgb_std": {
            "r": float(valid_df["r_std"].mean()),
            "g": float(valid_df["g_std"].mean()),
            "b": float(valid_df["b_std"].mean()),
        },
        "split_counts_8_1_1": split_counts(label_counts),
        "plots": [
            "class_distribution.png",
            "image_dimensions.png",
            "aspect_ratio_hist.png",
            "mode_distribution.png",
            "quality_histograms.png",
            "sample_grid.png",
            "quality_extremes.png",
        ],
    }
    save_json(result, output_dir / "analysis_results.json")
    save_markdown(render_cats_report(result), output_dir / "数据解读.md")
    return result


def render_cats_report(result: dict[str, Any]) -> str:
    label_total = sum(result["all_label_counts"].values())
    label_lines = []
    for label, count in sorted(result["all_label_counts"].items()):
        ratio = count / label_total * 100 if label_total else 0
        label_lines.append(f"{label}: {count} ({ratio:.2f}%)")
    label_text = ", ".join(label_lines)

    invalid_text = "未发现损坏图片"
    if result["invalid_count"]:
        examples = "；".join(f"`{item['relative_path']}` ({item['error']})" for item in result["invalid_examples"][:10])
        invalid_text = f"发现 {result['invalid_count']} 张无法正常读取的图片，示例：{examples}"

    exact_dup = result["exact_duplicates"]
    perceptual_dup = result["perceptual_hash_duplicates"]
    exact_dup_text = f"{exact_dup['group_count']} 组，重复图片数 {exact_dup['duplicate_image_count']}"
    perceptual_dup_text = f"{perceptual_dup['group_count']} 组，可能重复图片数 {perceptual_dup['duplicate_image_count']}"
    cross_label_text = "未发现跨标签精确重复"
    if exact_dup.get("cross_label_group_count", 0):
        examples = "；".join(
            " / ".join(f"`{path}`" for path in group[:4])
            for group in exact_dup.get("cross_label_top_groups", [])[:3]
        )
        cross_label_text = (
            f"发现 {exact_dup['cross_label_group_count']} 组跨标签精确重复，"
            f"涉及 {exact_dup['cross_label_duplicate_image_count']} 个文件，示例：{examples}"
        )

    split_text = "；".join(
        f"{label}: train={counts['train']}, val={counts['val']}, test={counts['test']}"
        for label, counts in sorted(result["split_counts_8_1_1"].items())
    )
    mode = "全量分析" if result["analysis_mode"] == "full" else "抽样分析"

    rgb_mean = result["rgb_mean"]
    rgb_std = result["rgb_std"]

    return f"""# Cats vs. Dogs 数据解读

## 1. 分析范围与执行策略

本次预分析对象为 Cats vs. Dogs 猫狗二分类图像数据集。数据从本地数据目录中的压缩包读取，并解压到本地数据目录下与压缩包同名的文件夹中，报告不记录本地绝对路径。

- 解压状态：{"本次新解压" if result["extraction"]["extracted"] else "已存在解压文件，本次复用"}
- 解压后文件数：{result["extraction"]["file_count"]}
- 解压耗时：{format_seconds(result["extraction"]["elapsed_seconds"])}
- 图片文件总数：{result["total_image_files"]}

按照预设流程，先抽取 5% 样本进行图片读取、尺寸、通道、质量和重复信息检测，再按耗时估算全量分析时间。

- 5% 样本数：{result["sample_count"]}
- 5% 样本分析耗时：{format_seconds(result["sample_elapsed_seconds"])}
- 估算全量分析耗时：{format_seconds(result["estimated_full_seconds"])}
- 20 分钟阈值判断：{"估算可在 20 分钟内完成，因此执行全量分析" if result["analysis_mode"] == "full" else "估算超过 20 分钟，因此保留抽样分析"}
- 最终分析方式：{mode}
- 最终分析图片数：{result["analyzed_count"]}
- 最终分析耗时：{format_seconds(result["analysis_elapsed_seconds"])}

## 2. 类别数量与类别比例

- 类别分布：{label_text}

![类别分布](class_distribution.png)

类别比例非常接近均衡，适合做二分类实验。后续划分训练集、验证集和测试集时仍建议使用分层划分，避免因为随机性导致某一集合中猫狗比例偏移。

建议按 8:1:1 划分时，各类别数量为：{split_text}。

## 3. 图片是否损坏

- 可正常读取图片数：{result["valid_count"]}
- 无法正常读取图片数：{result["invalid_count"]}
- 损坏图片记录：{invalid_text}

预处理和训练代码中不应删除损坏图片，建议在自定义 `Dataset` 中捕获读取异常，将异常文件记录到日志并跳过，保证原始数据可追溯。

## 4. 图片尺寸与宽高比

![图片宽高分布](image_dimensions.png)

![宽高比分布](aspect_ratio_hist.png)

- 宽度统计：{fmt_summary(result["width_summary"])}
- 高度统计：{fmt_summary(result["height_summary"])}
- 宽高比统计：{fmt_summary(result["aspect_ratio_summary"], digits=3)}
- 极端宽高比图片数量：{result["quality_flags"]["extreme_aspect_ratio_images"]}
- 小尺寸图片数量（任一边 < 100 px）：{result["quality_flags"]["small_images"]}

图片尺寸并不统一，这是自然图像数据集的典型特点。后续 AlexNet 训练必须统一输入尺寸。建议采用 `Resize + CenterCrop` 或训练阶段 `RandomResizedCrop` 的方式得到 224×224 输入。训练阶段可以使用随机裁剪增强泛化能力，验证集和测试集应使用确定性的 resize/crop，保证评估可复现。

## 5. 图片颜色通道

![颜色模式分布](mode_distribution.png)

- 颜色模式分布：{result["mode_counts"]}
- 通道数分布：{result["channel_counts"]}
- 估算 RGB 均值：R={rgb_mean["r"]:.3f}, G={rgb_mean["g"]:.3f}, B={rgb_mean["b"]:.3f}
- 估算 RGB 标准差：R={rgb_std["r"]:.3f}, G={rgb_std["g"]:.3f}, B={rgb_std["b"]:.3f}

AlexNet 输入应统一为 3 通道 RGB。即使大部分图片本身是 RGB，训练代码仍建议在读取阶段显式执行 `convert("RGB")`，避免灰度图、调色板图或其他模式导致张量维度不一致。

## 6. 图片质量

![质量指标分布](quality_histograms.png)

![质量极端样本](quality_extremes.png)

- 文件大小统计：{fmt_summary(result["file_size_kb_summary"])}
- 亮度统计：{fmt_summary(result["brightness_summary"])}
- 对比度统计：{fmt_summary(result["contrast_summary"])}
- 清晰度指标统计：{fmt_summary(result["sharpness_summary"])}
- 偏暗图片数量：{result["quality_flags"]["dark_images"]}
- 偏亮图片数量：{result["quality_flags"]["bright_images"]}
- 低对比度图片数量：{result["quality_flags"]["low_contrast_images"]}
- 相对低清晰度图片数量（最低 5% 阈值 {result["low_sharpness_threshold"]:.2f}）：{result["quality_flags"]["relative_low_sharpness_images"]}

质量指标显示数据集中存在亮度、清晰度和构图差异。对猫狗分类来说，这些差异不一定是错误数据，反而能提升模型对真实场景的适应能力。后续不建议按亮度或清晰度直接删除图片，而应通过数据增强、归一化和足够的训练样本提高鲁棒性。

## 7. 重复图片检查

- 精确重复检查：{exact_dup_text}
- 感知哈希相同检查：{perceptual_dup_text}
- 跨标签精确重复：{cross_label_text}

精确重复代表文件内容完全一致，感知哈希相同则只能说明图像外观高度相近，不能直接判定为重复。当前阶段建议只记录重复情况，不删除原文件。若后续模型表现异常偏高，或训练集与测试集之间存在重复图，再考虑在划分前进行去重或保证重复图片不会跨集合出现。

跨标签精确重复尤其需要注意，因为同一张图片如果同时出现在 Cat 和 Dog 目录中，会形成标签冲突。后续不建议删除原文件，但可以在训练清单中记录并排除冲突组，或保证同一重复组不会跨训练集、验证集和测试集。

## 8. 样本多样性与背景干扰

![随机样本](sample_grid.png)

从随机样本和质量极端样本可以看出，猫狗图片包含不同姿态、尺度、拍摄距离、光照条件和背景环境。背景往往不是纯色，可能包含室内家具、草地、人物、笼子等干扰信息。这个特点使任务更接近真实自然图像分类，但也意味着模型可能学习到背景偏差。

质量极端样本中还可以看到少量非典型图片，例如极小图、纯文字图、主体严重模糊或主体占比很小的图片。这类样本不应在预分析阶段直接删除，但训练代码应记录它们，后续错误样本分析时重点检查这些图片是否造成误判。

后续建议在 AlexNet 训练中使用：

1. 随机水平翻转，提高左右姿态鲁棒性；
2. 随机裁剪或随机缩放裁剪，减少模型对主体位置的依赖；
3. 轻微颜色扰动，缓解光照差异；
4. Dropout 和权重衰减，降低 AlexNet 参数量较大带来的过拟合风险；
5. 错误样本可视化，分析模型是否受背景、遮挡、主体过小等因素影响。

## 9. 数据划分是否合理

当前原始数据更适合由代码统一划分训练集、验证集和测试集。建议先收集全部有效图片路径，再按标签分层随机划分为 8:1:1。数据增强只能应用于训练集，验证集和测试集只能进行确定性预处理，避免数据泄漏。

如果检测到重复或近似重复图片，严格做法是在划分前先按重复组处理，避免同一张或高度相似图片同时出现在训练集和测试集中。当前阶段可以先记录重复组，在后续实验反思中说明其影响。

## 10. 数据是否适合 AlexNet

Cats vs. Dogs 适合本项目手动实现 AlexNet，原因如下：

1. 数据量较大，能够支撑卷积神经网络训练；
2. 图片为自然场景 RGB 图像，与 AlexNet 原始应用场景接近；
3. 类别二分类且比例均衡，评价指标清晰；
4. 图片尺寸、背景、光照和质量差异明显，适合讨论数据预处理和数据增强；
5. 训练结果可以通过 loss/accuracy 曲线、混淆矩阵和错误样本图直观展示。

## 11. 对后续建模的影响

后续 AlexNet 建模建议如下：

1. 所有图片读取后统一转换为 RGB；
2. 输入尺寸统一为 224×224；
3. 训练集使用随机裁剪、随机水平翻转和轻微颜色扰动；
4. 验证集和测试集只使用固定 resize/crop 与归一化；
5. 训练代码记录损坏图片并跳过，不删除原始文件；
6. 先用小样本调通手写 AlexNet，再扩大到全量训练；
7. 保留训练曲线、验证曲线、混淆矩阵、预测正确样本和预测错误样本；
8. 改进实验可比较基础 AlexNet、加入 Batch Normalization、调整 Dropout 或缩小全连接层规模。

## 12. 小结

Cats vs. Dogs 数据总体适合 AlexNet 图像分类实验。其主要问题不是类别不平衡，而是自然图像带来的尺寸不统一、背景复杂、图片质量差异和少量异常图片。因此报告重点应放在“图像预处理如何统一输入”“数据增强如何提升泛化”“AlexNet 是否过拟合”“错误样本反映了哪些数据难点”这些问题上。
"""


def run() -> None:
    parser = argparse.ArgumentParser(description="Run data pre-analysis for the ML practice project.")
    parser.add_argument("--only", choices=["all", "ionosphere", "cats"], default="all")
    args = parser.parse_args()

    set_plot_style()
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    get_data_root(must_exist=True)

    outputs: dict[str, Any] = {}
    if args.only in {"all", "ionosphere"}:
        outputs["ionosphere"] = analyze_ionosphere(PROJECT_ROOT / "exp" / "DataPreAnalyze" / "Ionosphere")
    if args.only in {"all", "cats"}:
        outputs["cats_vs_dogs"] = analyze_cats_vs_dogs(PROJECT_ROOT / "exp" / "DataPreAnalyze" / "Cats vs Dogs")

    summary_path = PROJECT_ROOT / "exp" / "DataPreAnalyze" / "analysis_summary.json"
    ensure_dir(summary_path.parent)
    save_json(outputs, summary_path)

    compact = {
        key: {
            "analysis_mode": value.get("analysis_mode"),
            "sample_count": value.get("sample_count"),
            "estimated_full_seconds": value.get("estimated_full_seconds"),
            "analysis_elapsed_seconds": value.get("analysis_elapsed_seconds"),
            "sample_count_or_rows": value.get("sample_count") or value.get("sample_count"),
        }
        for key, value in outputs.items()
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
