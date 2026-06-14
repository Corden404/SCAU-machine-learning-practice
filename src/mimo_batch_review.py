from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.project_config import PROJECT_ROOT


DEFAULT_REVIEW_PACKAGE = PROJECT_ROOT / "data" / "AlexNet" / "Anomaly_review_package"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "AlexNet" / "MiMo_anomaly_review"
DEFAULT_MODEL = "mimo-v2.5"
DEFAULT_BASE_URL = "https://api.xiaomimimo.com/v1/chat/completions"
DEFAULT_PROMPT = (
    "Review the 2x2 contact sheet. Each visible panel is labeled with a number and an expected class "
    "(Cat or Dog). Decide whether each panel is suitable for training a Cats vs. Dogs classifier.\n"
    "Return Y if it is a real photo of the expected animal class, even if it is cropped, overexposed, "
    "underexposed, small, blurry, or partially occluded, as long as the animal can be recognized or "
    "reasonably inferred from details.\n"
    "Return N for cartoons, drawings, logos, text-only images, blank/unreadable images, non-cat/dog images, "
    "images where the animal cannot be recognized, or images that clearly show the wrong class.\n"
    "If a panel is marked EMPTY, return NA for that panel.\n"
    "Do not explain your reasoning. Output exactly one line per panel in this format:\n"
    "1:Y\n2:N\n3:Y\n4:NA"
)


@dataclass
class ReviewItem:
    candidate_id: str
    relative_path: str
    label: str
    thumbnail: str


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_items(package_dir: Path) -> list[ReviewItem]:
    items_path = package_dir / "pending_items.json"
    if not items_path.exists():
        raise FileNotFoundError(f"pending_items.json not found: {items_path}")

    raw_items = json.loads(items_path.read_text(encoding="utf-8"))
    items: list[ReviewItem] = []
    for raw in raw_items:
        items.append(
            ReviewItem(
                candidate_id=str(raw.get("candidate_id", "")),
                relative_path=str(raw["relative_path"]),
                label=str(raw.get("label", "")),
                thumbnail=str(raw["thumbnail"]),
            )
        )
    return items


def chunks(items: list[ReviewItem], size: int) -> list[list[ReviewItem]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arialbd.ttf",
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arial.ttf",
    ]
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                pass
    return ImageFont.load_default()


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    padding_x = 8
    padding_y = 5
    rect = [
        bbox[0] - padding_x,
        bbox[1] - padding_y,
        bbox[2] + padding_x,
        bbox[3] + padding_y,
    ]
    draw.rounded_rectangle(rect, radius=7, fill=(0, 0, 0), outline=(255, 255, 255), width=2)
    draw.text((x, y), text, fill=(255, 255, 255), font=font)


def make_contact_sheet(
    batch_items: list[ReviewItem],
    package_dir: Path,
    output_path: Path,
    *,
    thumb_width: int,
    thumb_height: int,
    label_height: int,
    margin: int,
    quality: int,
) -> None:
    panel_height = thumb_height + label_height
    width = thumb_width * 2 + margin * 3
    height = panel_height * 2 + margin * 3
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    label_font = load_font(24)
    empty_font = load_font(24)

    positions = [
        (margin, margin),
        (thumb_width + margin * 2, margin),
        (margin, panel_height + margin * 2),
        (thumb_width + margin * 2, panel_height + margin * 2),
    ]

    for slot_index, xy in enumerate(positions, start=1):
        x, y = xy
        label_box = [x, y, x + thumb_width - 1, y + label_height - 1]
        image_box = [x, y + label_height, x + thumb_width - 1, y + label_height + thumb_height - 1]
        draw.rectangle(label_box, fill=(0, 0, 0))
        draw.rectangle(image_box, outline=(210, 210, 210), width=1)
        if slot_index > len(batch_items):
            draw.text((x + 10, y + 5), f"{slot_index} EMPTY", fill=(255, 255, 255), font=label_font)
            draw.rectangle(image_box, fill=(242, 242, 242))
            draw.text((x + 82, y + label_height + 92), "EMPTY", fill=(100, 100, 100), font=empty_font)
            continue

        item = batch_items[slot_index - 1]
        draw.text((x + 10, y + 5), f"{slot_index} {item.label or 'Unknown'}", fill=(255, 255, 255), font=label_font)
        source = package_dir / item.thumbnail
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = ImageOps.contain(image, (thumb_width, thumb_height), Image.Resampling.LANCZOS)
            panel = Image.new("RGB", (thumb_width, thumb_height), (245, 245, 245))
            panel.paste(image, ((thumb_width - image.width) // 2, (thumb_height - image.height) // 2))
            canvas.paste(panel, (x, y + label_height))

    canvas.save(output_path, format="JPEG", quality=quality, optimize=True)


def encode_data_uri(image_path: Path) -> str:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def request_payload(model: str, image_data_uri: str, prompt: str, max_completion_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are MiMo, an AI assistant developed by Xiaomi.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_uri,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
        ],
        "max_completion_tokens": max_completion_tokens,
    }


def build(args: argparse.Namespace) -> None:
    package_dir = args.package_dir.resolve()
    output_dir = args.output_dir.resolve()
    images_dir = output_dir / "images"
    ensure_dir(images_dir)

    items = load_items(package_dir)
    item_batches = chunks(items, 4)
    requests_path = output_dir / "mimo_requests.jsonl"
    manifest_path = output_dir / "manifest.json"
    prompt_path = output_dir / "prompt.txt"

    records = []
    with requests_path.open("w", encoding="utf-8", newline="\n") as requests_file:
        for batch_index, batch_items in enumerate(item_batches, start=1):
            image_name = f"batch_{batch_index:04d}.jpg"
            image_path = images_dir / image_name
            make_contact_sheet(
                batch_items,
                package_dir,
                image_path,
                thumb_width=args.thumb_width,
                thumb_height=args.thumb_height,
                label_height=args.label_height,
                margin=args.margin,
                quality=args.quality,
            )
            slots = []
            for slot_index in range(1, 5):
                if slot_index <= len(batch_items):
                    item = batch_items[slot_index - 1]
                    slots.append(
                        {
                            "slot": slot_index,
                            "candidate_id": item.candidate_id,
                            "relative_path": item.relative_path,
                            "label": item.label,
                            "thumbnail": item.thumbnail,
                        }
                    )
                else:
                    slots.append({"slot": slot_index, "empty": True})

            record = {
                "custom_id": f"cats_anomaly_{batch_index:04d}",
                "batch_index": batch_index,
                "image": str(image_path.relative_to(output_dir).as_posix()),
                "slots": slots,
                "request": request_payload(
                    args.model,
                    encode_data_uri(image_path),
                    args.prompt,
                    args.max_completion_tokens,
                ),
            }
            requests_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(
                {
                    "custom_id": record["custom_id"],
                    "batch_index": batch_index,
                    "image": record["image"],
                    "slots": slots,
                    "image_bytes": image_path.stat().st_size,
                }
            )

    manifest = {
        "model": args.model,
        "source_package_dir": str(package_dir),
        "output_dir": str(output_dir),
        "item_count": len(items),
        "batch_count": len(item_batches),
        "items_per_batch": 4,
        "contact_sheet": {
            "thumb_width": args.thumb_width,
            "thumb_height": args.thumb_height,
            "label_height": args.label_height,
            "margin": args.margin,
            "quality": args.quality,
        },
        "requests_jsonl": requests_path.name,
        "prompt_file": prompt_path.name,
        "records": records,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    prompt_path.write_text(args.prompt, encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "images_dir": str(images_dir),
                "requests_jsonl": str(requests_path),
                "manifest": str(manifest_path),
                "item_count": len(items),
                "batch_count": len(item_batches),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def iter_request_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def call_api(payload: dict[str, Any], api_key: str, base_url: str, timeout: int) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        base_url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_message_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts)
    return str(content)


RESULT_RE = re.compile(r"\b([1-4])\s*[:：.\-]\s*(Y|N|NA|YES|NO)\b", re.IGNORECASE)


def parse_panel_results(text: str) -> dict[int, str]:
    parsed: dict[int, str] = {}
    for match in RESULT_RE.finditer(text):
        slot = int(match.group(1))
        value = match.group(2).upper()
        if value == "YES":
            value = "Y"
        elif value == "NO":
            value = "N"
        parsed[slot] = value
    return parsed


def response_has_all_results(response_record: dict[str, Any]) -> bool:
    parsed = parse_panel_results(str(response_record.get("text", "")))
    expected_slots = [
        int(slot["slot"])
        for slot in response_record.get("slots", [])
        if not slot.get("empty")
    ]
    return bool(expected_slots) and all(slot in parsed for slot in expected_slots)


def write_review_csv(response_records: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "custom_id",
        "batch_index",
        "slot",
        "candidate_id",
        "relative_path",
        "label",
        "ai_decision",
        "ai_status",
        "raw_text",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for response_record in response_records:
            text = response_record.get("text", "")
            parsed = parse_panel_results(text)
            for slot in response_record["slots"]:
                if slot.get("empty"):
                    continue
                decision = parsed.get(int(slot["slot"]), "")
                status = "keep" if decision == "Y" else "exclude" if decision == "N" else "review"
                writer.writerow(
                    {
                        "custom_id": response_record["custom_id"],
                        "batch_index": response_record["batch_index"],
                        "slot": slot["slot"],
                        "candidate_id": slot.get("candidate_id", ""),
                        "relative_path": slot.get("relative_path", ""),
                        "label": slot.get("label", ""),
                        "ai_decision": decision,
                        "ai_status": status,
                        "raw_text": text,
                    }
                )


def run_api(args: argparse.Namespace) -> None:
    api_key = os.environ.get("MIMO_API_KEY")
    if not api_key:
        raise RuntimeError("Set MIMO_API_KEY in your local environment.")

    requests_path = args.requests_jsonl.resolve()
    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)
    responses_path = output_dir / "mimo_responses.jsonl"
    csv_path = output_dir / "mimo_review_result.csv"

    request_records = iter_request_records(requests_path)
    done_ids = set()
    response_records: list[dict[str, Any]] = []
    if responses_path.exists() and args.resume:
        with responses_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                record = json.loads(line)
                if response_has_all_results(record):
                    done_ids.add(record["custom_id"])
                    response_records.append(record)
                else:
                    print(f"[resume] will retry incomplete response: {record['custom_id']}", flush=True)

    pending_records = [
        (index, record)
        for index, record in enumerate(request_records, start=1)
        if record["custom_id"] not in done_ids
    ]

    def submit_one(index: int, record: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        last_error: BaseException | None = None
        for attempt in range(1, args.retries + 1):
            try:
                response = call_api(record["request"], api_key, args.base_url, args.timeout)
                text = extract_message_text(response)
                response_record = {
                    "custom_id": record["custom_id"],
                    "batch_index": record["batch_index"],
                    "image": record["image"],
                    "slots": record["slots"],
                    "text": text,
                    "response": response,
                }
                if response_has_all_results(response_record):
                    return index, response_record
                last_error = RuntimeError(f"incomplete response: {text!r}")
            except Exception as error:
                last_error = error
            if attempt < args.retries:
                sleep_seconds = args.retry_sleep * attempt
                print(f"[retry {attempt}] {record['custom_id']}: {last_error}; sleeping {sleep_seconds}s", flush=True)
                time.sleep(sleep_seconds)
        assert last_error is not None
        raise last_error

    failed_records = []
    with responses_path.open("a", encoding="utf-8", newline="\n") as responses_file:
        print(
            f"[start] total={len(request_records)} already_complete={len(done_ids)} "
            f"pending={len(pending_records)} parallel={args.parallel}",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as executor:
            future_map = {
                executor.submit(submit_one, index, record): (index, record)
                for index, record in pending_records
            }
            pending_futures = set(future_map)
            last_heartbeat = time.time()
            while pending_futures:
                done_futures, pending_futures = wait(
                    pending_futures,
                    timeout=args.heartbeat_seconds,
                    return_when=FIRST_COMPLETED,
                )
                if not done_futures:
                    print(
                        f"[heartbeat] complete_this_run={len(response_records)} "
                        f"remaining={len(pending_futures)}",
                        flush=True,
                    )
                    last_heartbeat = time.time()
                    continue
                for future in done_futures:
                    index, record = future_map[future]
                    try:
                        _, response_record = future.result()
                    except Exception as error:
                        failed_record = {
                            "custom_id": record["custom_id"],
                            "batch_index": record["batch_index"],
                            "image": record["image"],
                            "slots": record["slots"],
                            "error": repr(error),
                        }
                        failed_records.append(failed_record)
                        print(f"[failed] [{index}/{len(request_records)}] {record['custom_id']} {error!r}", flush=True)
                        continue
                    responses_file.write(json.dumps(response_record, ensure_ascii=False) + "\n")
                    responses_file.flush()
                    response_records.append(response_record)
                    print(f"[{index}/{len(request_records)}] {record['custom_id']} {response_record['text']!r}", flush=True)
                    if args.sleep:
                        time.sleep(args.sleep)
                if time.time() - last_heartbeat >= args.heartbeat_seconds:
                    print(
                        f"[heartbeat] complete_this_run={len(response_records)} "
                        f"remaining={len(pending_futures)}",
                        flush=True,
                    )
                    last_heartbeat = time.time()

    write_review_csv(response_records, csv_path)
    failed_path = output_dir / "mimo_failed.jsonl"
    if failed_records:
        with failed_path.open("a", encoding="utf-8", newline="\n") as file:
            for record in failed_records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "responses_jsonl": str(responses_path),
                "review_csv": str(csv_path),
                "response_count": len(response_records),
                "failed_jsonl": str(failed_path) if failed_records else "",
                "failed_count": len(failed_records),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and optionally run MiMo 4-image batch review requests.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Create 2x2 contact sheets and Base64 JSONL requests.")
    build_parser.add_argument("--package-dir", type=Path, default=DEFAULT_REVIEW_PACKAGE)
    build_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    build_parser.add_argument("--model", default=DEFAULT_MODEL)
    build_parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    build_parser.add_argument("--max-completion-tokens", type=int, default=4096)
    build_parser.add_argument("--thumb-width", type=int, default=280)
    build_parser.add_argument("--thumb-height", type=int, default=220)
    build_parser.add_argument("--label-height", type=int, default=34)
    build_parser.add_argument("--margin", type=int, default=8)
    build_parser.add_argument("--quality", type=int, default=88)
    build_parser.set_defaults(func=build)

    api_parser = subparsers.add_parser("run-api", help="Call MiMo API for a generated requests JSONL file.")
    api_parser.add_argument("--requests-jsonl", type=Path, default=DEFAULT_OUTPUT_DIR / "mimo_requests.jsonl")
    api_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    api_parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    api_parser.add_argument("--timeout", type=int, default=240)
    api_parser.add_argument("--retries", type=int, default=3)
    api_parser.add_argument("--retry-sleep", type=float, default=3.0)
    api_parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between successful requests.")
    api_parser.add_argument("--parallel", type=int, default=16, help="Number of concurrent API requests.")
    api_parser.add_argument("--heartbeat-seconds", type=float, default=15.0)
    api_parser.add_argument("--resume", action="store_true", help="Skip custom_id values already in mimo_responses.jsonl.")
    api_parser.set_defaults(func=run_api)

    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
