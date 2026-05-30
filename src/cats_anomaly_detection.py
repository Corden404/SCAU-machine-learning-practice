from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_preanalysis import (
    RANDOM_SEED,
    analyze_one_image,
    ensure_dir,
    ensure_extracted,
    find_image_files,
)
from src.project_config import PROJECT_ROOT


OUTPUT_DIR = PROJECT_ROOT / "data" / "AlexNet" / "Anomaly"
THUMB_DIR = OUTPUT_DIR / "thumbnails"
LOG_PATH = OUTPUT_DIR / "anomaly_detection.log"
DEFAULT_TOP_FRACTION = 0.03
FEATURE_COLUMNS = [
    "width",
    "height",
    "aspect_ratio",
    "log_file_size_kb",
    "brightness",
    "contrast",
    "log_sharpness",
    "entropy",
    "edge_density",
    "r_mean",
    "g_mean",
    "b_mean",
    "r_std",
    "g_std",
    "b_std",
]


def log(message: str) -> None:
    stamped = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(stamped, flush=True)
    try:
        ensure_dir(LOG_PATH.parent)
        with LOG_PATH.open("a", encoding="utf-8") as file:
            file.write(stamped + "\n")
    except Exception:
        pass


def is_nonempty_text(value: Any) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip()
    return bool(text) and text.lower() not in {"nan", "none", "null"}


def split_nonempty(value: Any) -> list[str]:
    if not is_nonempty_text(value):
        return []
    return [part.strip() for part in str(value).split(";") if is_nonempty_text(part)]


def entropy_from_gray(gray: Image.Image) -> float:
    hist = np.asarray(gray.histogram(), dtype=np.float64)
    total = hist.sum()
    if total == 0:
        return 0.0
    probs = hist[hist > 0] / total
    return float(-(probs * np.log2(probs)).sum())


def edge_density_from_gray(gray: Image.Image) -> float:
    arr = np.asarray(gray.resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float32)
    grad_y, grad_x = np.gradient(arr)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return float((magnitude > 18).mean())


def extra_image_features(path: Path) -> dict[str, float]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        gray = rgb.convert("L").resize((128, 128), Image.Resampling.BILINEAR)
        return {
            "entropy": entropy_from_gray(gray),
            "edge_density": edge_density_from_gray(gray),
        }


def duplicate_groups(records: list[dict[str, Any]], key: str) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        value = record.get(key)
        if record.get("valid") and value:
            groups[str(value)].append(str(record["relative_path"]))
    return {hash_value: paths for hash_value, paths in groups.items() if len(paths) > 1}


def duplicate_metadata(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    exact_groups = duplicate_groups(records, "exact_hash")
    phash_groups = duplicate_groups(records, "average_hash")
    metadata: dict[str, dict[str, Any]] = defaultdict(lambda: {"duplicate_group_ids": [], "cross_label_duplicate": False})

    for prefix, groups in [("exact", exact_groups), ("phash", phash_groups)]:
        for index, paths in enumerate(groups.values(), start=1):
            group_id = f"{prefix}_{index:04d}"
            labels = {path.split("/")[1] for path in paths if path.startswith("PetImages/")}
            cross_label = len(labels) > 1
            for path in paths:
                metadata[path]["duplicate_group_ids"].append(group_id)
                metadata[path]["cross_label_duplicate"] = metadata[path]["cross_label_duplicate"] or cross_label

    return metadata


def rule_flags(row: pd.Series) -> list[str]:
    flags: list[str] = []
    if not bool(row["valid"]):
        return ["corrupt"]
    if row["width"] < 100 or row["height"] < 100:
        flags.append("too_small")
    if row["aspect_ratio"] < 0.5 or row["aspect_ratio"] > 2.0:
        flags.append("extreme_aspect_ratio")
    if row["brightness"] < 40:
        flags.append("too_dark")
    if row["brightness"] > 215:
        flags.append("too_bright")
    if row["contrast"] < 8 or row["entropy"] < 1.2 or row["edge_density"] < 0.004:
        flags.append("blank_or_low_information")
    return flags


def thumbnail_name(relative_path: str) -> str:
    digest = hashlib.md5(relative_path.encode("utf-8")).hexdigest()[:12]
    stem = Path(relative_path).stem.replace(" ", "_")
    return f"{digest}_{stem}.jpg"


def create_thumbnail(source_path: Path, relative_path: str) -> str:
    ensure_dir(THUMB_DIR)
    thumb_name = thumbnail_name(relative_path)
    output_path = THUMB_DIR / thumb_name
    if output_path.exists():
        return f"thumbnails/{thumb_name}"

    try:
        with Image.open(source_path) as image:
            image = image.convert("RGB")
            thumb = ImageOps.contain(image, (280, 220), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (280, 220), "white")
            canvas.paste(thumb, ((280 - thumb.width) // 2, (220 - thumb.height) // 2))
    except Exception:
        canvas = Image.new("RGB", (280, 220), "#f2f2f2")
        draw = ImageDraw.Draw(canvas)
        draw.text((82, 98), "Unreadable image", fill="#444444")
    canvas.save(output_path, quality=88)
    return f"thumbnails/{thumb_name}"


def extract_records(root: Path) -> list[dict[str, Any]]:
    image_files = find_image_files(root)
    records: list[dict[str, Any]] = []
    for index, path in enumerate(image_files, start=1):
        record = analyze_one_image(path, root)
        if record.get("valid"):
            record.update(extra_image_features(path))
            record["log_file_size_kb"] = math.log1p(float(record["file_size_kb"]))
            record["log_sharpness"] = math.log1p(float(record["sharpness"]))
        else:
            for column in FEATURE_COLUMNS:
                record.setdefault(column, np.nan)
        records.append(record)
        if index % 2000 == 0:
            log(f"[features] processed {index}/{len(image_files)} images")
    return records


def score_models(frame: pd.DataFrame, top_fraction: float, n_clusters: int) -> pd.DataFrame:
    valid_mask = frame["valid"].astype(bool)
    valid = frame.loc[valid_mask].copy()
    features = valid[FEATURE_COLUMNS].astype(float).replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True))
    scaled = StandardScaler().fit_transform(features)

    top_n = max(1, int(math.ceil(len(valid) * top_fraction)))

    isolation = IsolationForest(
        n_estimators=300,
        contamination=top_fraction,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    isolation.fit(scaled)
    iforest_score = -isolation.decision_function(scaled)

    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled)
    distances = np.linalg.norm(scaled - kmeans.cluster_centers_[cluster_labels], axis=1)

    valid["iforest_score"] = iforest_score
    valid["kmeans_distance"] = distances
    valid["iforest_rank"] = valid["iforest_score"].rank(method="first", ascending=False).astype(int)
    valid["kmeans_rank"] = valid["kmeans_distance"].rank(method="first", ascending=False).astype(int)
    valid["iforest_top"] = valid["iforest_rank"] <= top_n
    valid["kmeans_top"] = valid["kmeans_rank"] <= top_n

    for column in ["iforest_score", "kmeans_distance", "iforest_rank", "kmeans_rank", "iforest_top", "kmeans_top"]:
        frame.loc[valid.index, column] = valid[column]

    frame["iforest_top"] = frame["iforest_top"].fillna(False).astype(bool)
    frame["kmeans_top"] = frame["kmeans_top"].fillna(False).astype(bool)
    return frame


def model_source_name(prefix: str, top_fraction: float) -> str:
    percent = top_fraction * 100
    if percent.is_integer():
        percent_text = str(int(percent))
    else:
        percent_text = f"{percent:.1f}".rstrip("0").rstrip(".")
    return f"{prefix}_top{percent_text}"


def build_candidates(frame: pd.DataFrame, root: Path, top_fraction: float) -> pd.DataFrame:
    candidate_rows = []
    for _, row in frame.iterrows():
        sources: list[str] = []
        reasons = []

        flags = split_nonempty(row.get("rule_flags", ""))
        if flags:
            sources.append("rule")
            reasons.extend(flags)
        duplicate_ids = split_nonempty(row.get("duplicate_group_ids", ""))
        if duplicate_ids:
            sources.append("duplicate")
            reasons.extend(duplicate_ids)
        if parse_bool(row["cross_label_duplicate"]):
            sources.append("cross_label_duplicate")
        if parse_bool(row["iforest_top"]):
            sources.append(model_source_name("iforest", top_fraction))
        if parse_bool(row["kmeans_top"]):
            sources.append(model_source_name("kmeans", top_fraction))

        if not sources:
            continue

        relative_path = str(row["relative_path"])
        absolute_path = root / relative_path
        record = row.to_dict()
        record["candidate_id"] = f"C{len(candidate_rows) + 1:05d}"
        record["status"] = "review"
        record["manual_note"] = ""
        record["detected_by"] = "+".join(sorted(set(sources)))
        record["detected_reason"] = ";".join(sorted(set(str(reason) for reason in reasons)))
        record["source_path_for_thumbnail"] = absolute_path
        record.pop("absolute_path", None)
        candidate_rows.append(record)

    candidates = pd.DataFrame(candidate_rows)
    if candidates.empty:
        return candidates

    candidates["max_model_rank"] = candidates[["iforest_rank", "kmeans_rank"]].min(axis=1, skipna=True)
    candidates = candidates.sort_values(
        by=["cross_label_duplicate", "valid", "max_model_rank", "relative_path"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    candidates["candidate_id"] = [f"C{index + 1:05d}" for index in range(len(candidates))]
    log(f"[candidates] union count: {len(candidates)}")
    thumbnails = []
    for index, row in candidates.iterrows():
        thumbnails.append(create_thumbnail(Path(row["source_path_for_thumbnail"]), str(row["relative_path"])))
        if (index + 1) % 200 == 0 or index + 1 == len(candidates):
            log(f"[thumbnails] generated {index + 1}/{len(candidates)}")
    candidates["thumbnail"] = thumbnails
    candidates = candidates.drop(columns=["source_path_for_thumbnail"], errors="ignore")
    return candidates


def build_candidate_frame_without_thumbnails(frame: pd.DataFrame, top_fraction: float) -> pd.DataFrame:
    rows = []
    for _, row in frame.iterrows():
        sources: list[str] = []
        reasons = []
        flags = split_nonempty(row.get("rule_flags", ""))
        if flags:
            sources.append("rule")
            reasons.extend(flags)
        duplicate_ids = split_nonempty(row.get("duplicate_group_ids", ""))
        if duplicate_ids:
            sources.append("duplicate")
            reasons.extend(duplicate_ids)
        if parse_bool(row["cross_label_duplicate"]):
            sources.append("cross_label_duplicate")
        if parse_bool(row["iforest_top"]):
            sources.append(model_source_name("iforest", top_fraction))
        if parse_bool(row["kmeans_top"]):
            sources.append(model_source_name("kmeans", top_fraction))
        if not sources:
            continue
        record = row.to_dict()
        record["status"] = "review"
        record["manual_note"] = ""
        record["detected_by"] = "+".join(sorted(set(sources)))
        record["detected_reason"] = ";".join(sorted(set(str(reason) for reason in reasons)))
        rows.append(record)

    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return candidates
    candidates["max_model_rank"] = candidates[["iforest_rank", "kmeans_rank"]].min(axis=1, skipna=True)
    candidates = candidates.sort_values(
        by=["cross_label_duplicate", "valid", "max_model_rank", "relative_path"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    candidates["candidate_id"] = [f"C{index + 1:05d}" for index in range(len(candidates))]
    return candidates


def add_thumbnails_in_batches(candidates: pd.DataFrame, root: Path, batch_size: int, max_batches: int | None) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    thumb_values = list(candidates["thumbnail"]) if "thumbnail" in candidates.columns else [""] * len(candidates)
    missing_indices = [index for index, value in enumerate(thumb_values) if not is_nonempty_text(value)]
    if max_batches is not None:
        missing_indices = missing_indices[: batch_size * max_batches]
    total_to_process = len(missing_indices)
    log(f"[thumbnails] missing in this run: {total_to_process}")
    for done, row_index in enumerate(missing_indices, start=1):
        row = candidates.iloc[row_index]
        relative_path = str(row["relative_path"])
        thumb_values[row_index] = create_thumbnail(root / relative_path, relative_path)
        if done % batch_size == 0 or done == total_to_process:
            log(f"[thumbnails] generated {done}/{total_to_process} in current run")
    candidates["thumbnail"] = thumb_values
    return candidates


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def write_review_template(candidates: pd.DataFrame, path: Path) -> None:
    columns = ["candidate_id", "relative_path", "label", "status", "manual_note", "detected_by", "detected_reason", "reviewed_at"]
    for column in columns:
        if column not in candidates.columns:
            candidates[column] = ""
    candidates.loc[:, columns].to_csv(path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)


def write_review_json(candidates: pd.DataFrame, path: Path) -> None:
    items = candidates.replace({np.nan: None}).to_dict(orient="records")
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def recompute_top_flags_from_ranks(frame: pd.DataFrame, top_fraction: float) -> pd.DataFrame:
    valid_mask = frame["valid"].map(parse_bool)
    top_n = max(1, int(math.ceil(int(valid_mask.sum()) * top_fraction)))
    frame["iforest_top"] = valid_mask & (pd.to_numeric(frame["iforest_rank"], errors="coerce") <= top_n)
    frame["kmeans_top"] = valid_mask & (pd.to_numeric(frame["kmeans_rank"], errors="coerce") <= top_n)
    return frame


def apply_existing_reviews(candidates: pd.DataFrame, review_path: Path) -> pd.DataFrame:
    candidates["reviewed_at"] = ""
    if not review_path.exists() or candidates.empty:
        return candidates

    old = pd.read_csv(review_path, encoding="utf-8-sig").fillna("")
    if "relative_path" not in old.columns:
        return candidates

    old_by_path = {}
    for _, row in old.iterrows():
        status = str(row.get("status", "")).strip()
        note = str(row.get("manual_note", "")).strip()
        if status and (status != "review" or note):
            old_by_path[str(row["relative_path"])] = {
                "status": status,
                "manual_note": note,
                "reviewed_at": str(row.get("reviewed_at", "")),
            }

    if not old_by_path:
        return candidates

    preserved = 0
    for index, row in candidates.iterrows():
        review = old_by_path.get(str(row["relative_path"]))
        if not review:
            continue
        candidates.at[index, "status"] = review["status"]
        candidates.at[index, "manual_note"] = review["manual_note"]
        candidates.at[index, "reviewed_at"] = review["reviewed_at"]
        preserved += 1
    log(f"[reviews] preserved existing labels: {preserved}")
    return candidates


def run() -> None:
    parser = argparse.ArgumentParser(description="Find Cats vs. Dogs anomaly candidates for manual review.")
    parser.add_argument("--top-fraction", type=float, default=DEFAULT_TOP_FRACTION, help="Top fraction for Isolation Forest and KMeans.")
    parser.add_argument("--clusters", type=int, default=8, help="KMeans cluster count.")
    parser.add_argument("--reuse-features", action="store_true", help="Rebuild candidates from all_image_anomaly_features.csv.")
    parser.add_argument("--thumbnail-batch-size", type=int, default=200, help="Progress interval for thumbnail generation.")
    parser.add_argument("--max-thumbnail-batches", type=int, default=None, help="Limit thumbnail batches in one run.")
    parser.add_argument("--skip-thumbnails", action="store_true", help="Do not generate missing thumbnails.")
    args = parser.parse_args()

    ensure_dir(OUTPUT_DIR)
    ensure_dir(THUMB_DIR)

    started = time.perf_counter()
    log("[start] cats anomaly detection")
    root, extraction = ensure_extracted("cats_vs_dogs")
    features_path = OUTPUT_DIR / "all_image_anomaly_features.csv"
    if args.reuse_features:
        log(f"[features] reusing {features_path}")
        frame = pd.read_csv(features_path, encoding="utf-8-sig")
        for column in ["valid", "cross_label_duplicate", "iforest_top", "kmeans_top"]:
            frame[column] = frame[column].map(parse_bool)
        frame = recompute_top_flags_from_ranks(frame, args.top_fraction)
    else:
        log("[features] extracting image features")
        records = extract_records(root)
        duplicate_info = duplicate_metadata(records)

        for record in records:
            duplicate = duplicate_info.get(record["relative_path"], {})
            record["duplicate_group_ids"] = ";".join(duplicate.get("duplicate_group_ids", []))
            record["cross_label_duplicate"] = bool(duplicate.get("cross_label_duplicate", False))

        frame = pd.DataFrame(records)
        frame["rule_flags"] = frame.apply(rule_flags, axis=1).map(lambda flags: ";".join(flags))
        log("[models] scoring Isolation Forest and KMeans")
        frame = score_models(frame, args.top_fraction, args.clusters)
    log("[candidates] building manual review candidates")
    candidates = build_candidate_frame_without_thumbnails(frame, args.top_fraction)
    log(f"[candidates] union count: {len(candidates)}")

    old_candidates_path = OUTPUT_DIR / "anomaly_candidates.csv"
    if old_candidates_path.exists() and not candidates.empty:
        old = pd.read_csv(old_candidates_path, encoding="utf-8-sig")
        if "relative_path" in old.columns and "thumbnail" in old.columns:
            thumbnail_map = {
                str(row["relative_path"]): row["thumbnail"]
                for _, row in old.iterrows()
                if is_nonempty_text(row.get("thumbnail"))
            }
            candidates["thumbnail"] = candidates["relative_path"].map(thumbnail_map).fillna("")

    if not args.skip_thumbnails:
        candidates = add_thumbnails_in_batches(candidates, root, args.thumbnail_batch_size, args.max_thumbnail_batches)
    elif "thumbnail" not in candidates.columns:
        candidates["thumbnail"] = ""
    candidates = apply_existing_reviews(candidates, OUTPUT_DIR / "manual_review.csv")

    public_feature_columns = [
        "relative_path",
        "label",
        "valid",
        "error",
        "width",
        "height",
        "aspect_ratio",
        "file_size_kb",
        "brightness",
        "contrast",
        "sharpness",
        "entropy",
        "edge_density",
        "mode",
        "channels",
        "duplicate_group_ids",
        "cross_label_duplicate",
        "rule_flags",
        "iforest_score",
        "iforest_rank",
        "iforest_top",
        "kmeans_distance",
        "kmeans_rank",
        "kmeans_top",
    ]
    write_csv(frame.loc[:, public_feature_columns], OUTPUT_DIR / "all_image_anomaly_features.csv")
    if not candidates.empty:
        write_csv(candidates.drop(columns=["max_model_rank"], errors="ignore"), OUTPUT_DIR / "anomaly_candidates.csv")
        write_review_template(candidates, OUTPUT_DIR / "manual_review.csv")
        write_review_json(candidates, OUTPUT_DIR / "review_items.json")

    summary = {
        "output_dir": "data/AlexNet/Anomaly",
        "extraction": extraction.__dict__,
        "total_records": int(len(frame)),
        "valid_records": int(frame["valid"].sum()),
        "invalid_records": int((~frame["valid"].astype(bool)).sum()),
        "top_fraction": args.top_fraction,
        "kmeans_clusters": args.clusters,
        "rule_candidates": int(frame["rule_flags"].map(is_nonempty_text).sum()),
        "duplicate_candidates": int(frame["duplicate_group_ids"].map(is_nonempty_text).sum()),
        "cross_label_duplicate_candidates": int(frame["cross_label_duplicate"].sum()),
        "iforest_candidates": int(frame["iforest_top"].sum()),
        "kmeans_candidates": int(frame["kmeans_top"].sum()),
        "union_candidates": int(len(candidates)),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (OUTPUT_DIR / "anomaly_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
