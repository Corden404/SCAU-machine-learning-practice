from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from src.data_preanalysis import ensure_extracted, find_image_files, infer_catdog_label
from src.project_config import PROJECT_ROOT


OUTPUT_DIR = PROJECT_ROOT / "data" / "AlexNet" / "Duplicates"
PACKAGE_DIR = PROJECT_ROOT / "data" / "AlexNet" / "Duplicate_review_package"
CACHE_PATH = OUTPUT_DIR / "image_hashes.csv"
IMAGE_SIZE = (280, 220)


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cats vs. Dogs 重复图片组审核</title>
  <style>
    :root { font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f7f9; color: #1f2933; }
    body { margin: 0; }
    header { position: sticky; top: 0; z-index: 10; background: #fff; border-bottom: 1px solid #d7dce2; padding: 10px 14px; }
    .bar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
    button, select, input { border: 1px solid #c4ccd5; border-radius: 6px; background: #fff; padding: 7px 10px; font-size: 14px; }
    button { cursor: pointer; }
    button.good { background: #1f7a4d; color: #fff; border-color: #1f7a4d; }
    button.bad { background: #b42318; color: #fff; border-color: #b42318; }
    button.warn { background: #9a6700; color: #fff; border-color: #9a6700; }
    button.primary { background: #1769aa; color: #fff; border-color: #1769aa; }
    main { padding: 14px; }
    .note { margin-top: 8px; color: #5f6b7a; font-size: 13px; line-height: 1.5; }
    .groups { display: grid; gap: 14px; }
    .group { background: #fff; border: 2px solid transparent; border-radius: 8px; padding: 10px; box-shadow: 0 1px 3px rgba(15, 23, 42, .12); }
    .group.selected { border-color: #1769aa; }
    .group-head { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 8px; font-size: 13px; }
    .status { display: inline-block; border-radius: 999px; padding: 2px 7px; font-weight: 700; background: #e4e7eb; }
    .status.duplicate { background: #ffe2dd; color: #8a1f11; }
    .status.not_duplicate { background: #d9f5e5; color: #116149; }
    .status.uncertain { background: #fff0c2; color: #7a4d00; }
    .imgs { display: flex; flex-wrap: wrap; gap: 8px; }
    .img-card { width: 180px; font-size: 11px; color: #3d4852; }
    .img-card img { width: 180px; height: 145px; object-fit: contain; background: #f0f2f4; border: 1px solid #d7dce2; border-radius: 4px; display: block; }
    .path { font-family: Consolas, monospace; word-break: break-all; }
    .actions { display: flex; gap: 6px; margin-top: 8px; }
  </style>
</head>
<body>
  <header>
    <div class="bar">
      <button class="bad" onclick="setStatusForSelected('duplicate')">D 真重复</button>
      <button class="good" onclick="setStatusForSelected('not_duplicate')">N 不是重复</button>
      <button class="warn" onclick="setStatusForSelected('uncertain')">U 不确定</button>
      <select id="filter" onchange="page=0; selected=0; render()">
        <option value="unreviewed">未审核</option>
        <option value="all">全部</option>
        <option value="duplicate">真重复</option>
        <option value="not_duplicate">不是重复</option>
        <option value="uncertain">不确定</option>
      </select>
      <input id="search" placeholder="搜索路径/原因/标签" oninput="page=0; selected=0; render()">
      <button onclick="prevPage()">上一页</button>
      <button onclick="nextPage()">下一页</button>
      <button class="primary" onclick="downloadCsv()">导出 CSV</button>
      <button onclick="downloadJson()">导出 JSON</button>
      <span id="pageInfo"></span>
    </div>
    <div class="note">审核对象是“图片组”。如果组内图片肉眼看起来是同一张或几乎同一张，选“真重复”；如果只是姿势、背景、颜色相似但不是同一张，选“不是重复”；拿不准选“不确定”。进度保存在当前浏览器，完成后导出 CSV。</div>
    <div class="note" id="stats"></div>
  </header>
  <main><div id="groups" class="groups"></div></main>
  <script>
    const PACKAGE_ID = "__PACKAGE_ID__";
    const GROUPS = __GROUPS_JSON__;
    let groups = GROUPS.map(group => ({...group, status: group.status || 'review'}));
    let page = 0;
    let pageSize = 30;
    let selected = 0;

    function saveLocal() {
      const reviewed = groups.filter(group => (group.status || 'review') !== 'review');
      localStorage.setItem(PACKAGE_ID, JSON.stringify(reviewed));
    }

    function loadLocal() {
      try {
        const raw = localStorage.getItem(PACKAGE_ID);
        if (!raw) return;
        const reviewed = JSON.parse(raw);
        const byId = Object.fromEntries(reviewed.map(group => [group.group_id, group]));
        groups.forEach(group => {
          const old = byId[group.group_id];
          if (old) {
            group.status = old.status || 'review';
            group.reviewed_at = old.reviewed_at || group.reviewed_at || '';
          }
        });
      } catch (error) { console.warn(error); }
    }

    function filteredGroups() {
      const filter = document.getElementById('filter').value;
      const query = document.getElementById('search').value.trim().toLowerCase();
      return groups.filter(group => {
        const status = group.status || 'review';
        const ok = filter === 'all' || (filter === 'unreviewed' ? status === 'review' : status === filter);
        const text = `${group.group_id} ${group.reasons} ${group.labels} ${group.items.map(x => x.relative_path).join(' ')}`.toLowerCase();
        return ok && (!query || text.includes(query));
      });
    }

    function render() {
      const list = filteredGroups();
      const pages = Math.max(1, Math.ceil(list.length / pageSize));
      if (page >= pages) page = pages - 1;
      const visible = list.slice(page * pageSize, page * pageSize + pageSize);
      const root = document.getElementById('groups');
      root.innerHTML = '';
      visible.forEach((group, index) => {
        const div = document.createElement('div');
        div.className = 'group' + (index === selected ? ' selected' : '');
        div.onclick = () => { selected = index; render(); };
        div.innerHTML = `
          <div class="group-head">
            <span class="status ${group.status || 'review'}">${group.status || 'review'}</span>
            <strong>${group.group_id}</strong>
            <span>size=${group.size}</span>
            <span>labels=${group.labels}</span>
            <span>${group.cross_label ? '跨标签' : '同标签/未知'}</span>
            <span>${group.reasons}</span>
          </div>
          <div class="imgs">
            ${group.items.map(item => `
              <div class="img-card">
                <img src="${item.thumbnail}" loading="lazy" alt="">
                <div>${item.label}</div>
                <div class="path">${item.relative_path}</div>
              </div>
            `).join('')}
          </div>
          <div class="actions">
            <button class="bad" onclick="event.stopPropagation(); setStatus('${group.group_id}', 'duplicate')">真重复</button>
            <button class="good" onclick="event.stopPropagation(); setStatus('${group.group_id}', 'not_duplicate')">不是重复</button>
            <button class="warn" onclick="event.stopPropagation(); setStatus('${group.group_id}', 'uncertain')">不确定</button>
          </div>`;
        root.appendChild(div);
      });
      const counts = groups.reduce((acc, group) => {
        const status = group.status || 'review';
        acc[status] = (acc[status] || 0) + 1;
        return acc;
      }, {});
      document.getElementById('stats').textContent =
        `组数 ${groups.length} | 未审核 ${counts.review || 0} | 真重复 ${counts.duplicate || 0} | 不是重复 ${counts.not_duplicate || 0} | 不确定 ${counts.uncertain || 0}`;
      document.getElementById('pageInfo').textContent = `${page + 1}/${pages} (${list.length})`;
    }

    function setStatus(groupId, status) {
      const group = groups.find(group => group.group_id === groupId);
      if (!group) return;
      group.status = status;
      group.reviewed_at = new Date().toISOString();
      saveLocal();
      const list = filteredGroups();
      let visibleCount = Math.min(pageSize, Math.max(0, list.length - page * pageSize));
      if (visibleCount === 0 && page > 0) {
        page -= 1;
        visibleCount = Math.min(pageSize, Math.max(0, list.length - page * pageSize));
      }
      selected = Math.min(selected, Math.max(0, visibleCount - 1));
      render();
    }

    function setStatusForSelected(status) {
      const list = filteredGroups();
      const group = list[page * pageSize + selected];
      if (group) setStatus(group.group_id, status);
    }

    function nextPage() { page += 1; selected = 0; render(); }
    function prevPage() { page = Math.max(0, page - 1); selected = 0; render(); }
    function csvEscape(value) {
      const text = String(value ?? '');
      return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
    }
    function downloadCsv() {
      const fields = ['group_id','status','size','labels','cross_label','reasons','paths','reviewed_at'];
      const rows = [fields.join(',')].concat(groups.map(group => {
        const row = {
          group_id: group.group_id,
          status: group.status || 'review',
          size: group.size,
          labels: group.labels,
          cross_label: group.cross_label,
          reasons: group.reasons,
          paths: group.items.map(x => x.relative_path).join(';'),
          reviewed_at: group.reviewed_at || ''
        };
        return fields.map(field => csvEscape(row[field])).join(',');
      }));
      downloadText(rows.join('\n'), `duplicate_review_${PACKAGE_ID}.csv`, 'text/csv;charset=utf-8');
    }
    function downloadJson() {
      downloadText(JSON.stringify(groups, null, 2), `duplicate_review_${PACKAGE_ID}.json`, 'application/json;charset=utf-8');
    }
    function downloadText(text, filename, type) {
      const blob = new Blob(['\ufeff' + text], { type });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = filename;
      a.click();
      URL.revokeObjectURL(a.href);
    }
    document.addEventListener('keydown', event => {
      if (event.target.tagName === 'INPUT') return;
      const key = event.key.toLowerCase();
      if (key === 'd') setStatusForSelected('duplicate');
      if (key === 'n') setStatusForSelected('not_duplicate');
      if (key === 'u') setStatusForSelected('uncertain');
      if (event.key === 'ArrowRight') { selected = Math.min(selected + 1, pageSize - 1); render(); }
      if (event.key === 'ArrowLeft') { selected = Math.max(selected - 1, 0); render(); }
    });
    loadLocal();
    render();
  </script>
</body>
</html>
"""


README_TEXT = """# Cats vs. Dogs 重复图片组审核包

## 使用方法

1. 解压 zip。
2. 双击打开 `index.html`。
3. 每一组图片一起判断：
   - `D`：真重复，同一张或几乎同一张；
   - `N`：不是重复，只是相似；
   - `U`：不确定。
4. 完成后点击“导出 CSV”，把导出的 CSV 发回。

本包只包含缩略图，不包含原始图片。
"""


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def move_to_trash(path: Path) -> None:
    if not path.exists():
        return
    trash = PROJECT_ROOT / ".trash"
    trash.mkdir(parents=True, exist_ok=True)
    target = trash / f"{path.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.move(str(path), str(target))


def ensure_clean_dir(path: Path) -> None:
    move_to_trash(path)
    path.mkdir(parents=True, exist_ok=True)


def exact_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bits_to_hex(bits: np.ndarray) -> str:
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bool(bit))
    return f"{value:016x}"


def average_hash(image: Image.Image) -> str:
    gray = image.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
    arr = np.asarray(gray, dtype=np.float32)
    return bits_to_hex(arr > arr.mean())


def difference_hash(image: Image.Image) -> str:
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    arr = np.asarray(gray, dtype=np.float32)
    return bits_to_hex(arr[:, 1:] > arr[:, :-1])


def dct_matrix(n: int) -> np.ndarray:
    matrix = np.zeros((n, n), dtype=np.float32)
    factor = math.pi / (2 * n)
    scale0 = math.sqrt(1 / n)
    scale = math.sqrt(2 / n)
    for k in range(n):
        alpha = scale0 if k == 0 else scale
        for i in range(n):
            matrix[k, i] = alpha * math.cos((2 * i + 1) * k * factor)
    return matrix


DCT32 = dct_matrix(32)


def perceptual_hash(image: Image.Image) -> str:
    gray = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    arr = np.asarray(gray, dtype=np.float32)
    dct = DCT32 @ arr @ DCT32.T
    block = dct[1:9, 1:9]
    return bits_to_hex(block > np.median(block))


def hamming_hex(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


class Dsu:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


class BkNode:
    def __init__(self, value: str, index: int) -> None:
        self.value = value
        self.indices = [index]
        self.children: dict[int, BkNode] = {}


class BkTree:
    def __init__(self) -> None:
        self.root: BkNode | None = None

    def add(self, value: str, index: int) -> None:
        if self.root is None:
            self.root = BkNode(value, index)
            return
        node = self.root
        while True:
            distance = hamming_hex(value, node.value)
            if distance == 0:
                node.indices.append(index)
                return
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = BkNode(value, index)
                return
            node = child

    def query(self, value: str, max_distance: int) -> list[tuple[int, int]]:
        if self.root is None:
            return []
        matches: list[tuple[int, int]] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = hamming_hex(value, node.value)
            if distance <= max_distance:
                matches.extend((index, distance) for index in node.indices)
            lower = distance - max_distance
            upper = distance + max_distance
            for child_distance, child in node.children.items():
                if lower <= child_distance <= upper:
                    stack.append(child)
        return matches


def compute_hash_cache(root: Path, force: bool) -> pd.DataFrame:
    if CACHE_PATH.exists() and not force:
        log(f"reuse hash cache: {CACHE_PATH}")
        return pd.read_csv(CACHE_PATH, encoding="utf-8-sig")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_files = find_image_files(root)
    rows = []
    for index, path in enumerate(image_files, start=1):
        row = {
            "relative_path": path.relative_to(root).as_posix(),
            "label": infer_catdog_label(path),
            "valid": False,
            "error": "",
            "exact_md5": "",
            "ahash": "",
            "dhash": "",
            "phash": "",
            "width": "",
            "height": "",
        }
        try:
            row["exact_md5"] = exact_md5(path)
            with Image.open(path) as image:
                image.load()
                row["width"], row["height"] = image.size
                rgb = image.convert("RGB")
                row["ahash"] = average_hash(rgb)
                row["dhash"] = difference_hash(rgb)
                row["phash"] = perceptual_hash(rgb)
                row["valid"] = True
        except Exception as exc:
            row["error"] = type(exc).__name__
        rows.append(row)
        if index % 2000 == 0 or index == len(image_files):
            log(f"hashed {index}/{len(image_files)} images")
    frame = pd.DataFrame(rows)
    frame.to_csv(CACHE_PATH, index=False, encoding="utf-8-sig")
    return frame


def add_exact_edges(valid: pd.DataFrame, dsu: Dsu, edge_reasons: list[tuple[int, int, str]]) -> None:
    for indices in valid.groupby("exact_md5").indices.values():
        indices = list(indices)
        if len(indices) <= 1:
            continue
        base = indices[0]
        for other in indices[1:]:
            dsu.union(base, other)
            edge_reasons.append((base, other, "exact_md5"))


def add_hash_edges(valid: pd.DataFrame, dsu: Dsu, edge_reasons: list[tuple[int, int, str]], hash_name: str, threshold: int) -> None:
    if threshold < 0:
        return
    tree = BkTree()
    for index, value in enumerate(valid[hash_name].astype(str)):
        if not value or value == "nan":
            continue
        for other, distance in tree.query(value, threshold):
            dsu.union(index, other)
            edge_reasons.append((index, other, f"{hash_name}<={threshold}:d{distance}"))
        tree.add(value, index)
    log(f"{hash_name} threshold {threshold} done")


def build_duplicate_groups(frame: pd.DataFrame, phash_threshold: int, dhash_threshold: int, ahash_threshold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = frame[frame["valid"].astype(bool)].reset_index(drop=True)
    dsu = Dsu(len(valid))
    edge_reasons: list[tuple[int, int, str]] = []

    add_exact_edges(valid, dsu, edge_reasons)
    add_hash_edges(valid, dsu, edge_reasons, "phash", phash_threshold)
    add_hash_edges(valid, dsu, edge_reasons, "dhash", dhash_threshold)
    add_hash_edges(valid, dsu, edge_reasons, "ahash", ahash_threshold)

    members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(valid)):
        members[dsu.find(index)].append(index)

    reason_map: dict[int, set[str]] = defaultdict(set)
    for a, b, reason in edge_reasons:
        root = dsu.find(a)
        if root == dsu.find(b):
            reason_map[root].add(reason)

    group_rows = []
    item_rows = []
    group_index = 1
    for root, indices in members.items():
        if len(indices) <= 1:
            continue
        labels = sorted(set(str(valid.loc[index, "label"]) for index in indices))
        group_id = f"D{group_index:05d}"
        group_index += 1
        reasons = sorted(reason_map.get(root, set()))
        group_rows.append(
            {
                "group_id": group_id,
                "size": len(indices),
                "labels": ";".join(labels),
                "cross_label": len([label for label in labels if label != "Unknown"]) > 1,
                "reasons": ";".join(reasons),
                "status": "review",
                "reviewed_at": "",
            }
        )
        for index in indices:
            item_rows.append(
                {
                    "group_id": group_id,
                    "relative_path": valid.loc[index, "relative_path"],
                    "label": valid.loc[index, "label"],
                    "width": valid.loc[index, "width"],
                    "height": valid.loc[index, "height"],
                    "exact_md5": valid.loc[index, "exact_md5"],
                    "ahash": valid.loc[index, "ahash"],
                    "dhash": valid.loc[index, "dhash"],
                    "phash": valid.loc[index, "phash"],
                }
            )

    groups = pd.DataFrame(group_rows)
    items = pd.DataFrame(item_rows)
    if not groups.empty:
        groups = groups.sort_values(["cross_label", "size", "group_id"], ascending=[False, False, True]).reset_index(drop=True)
        id_map = {old: f"D{idx + 1:05d}" for idx, old in enumerate(groups["group_id"])}
        groups["group_id"] = groups["group_id"].map(id_map)
        items["group_id"] = items["group_id"].map(id_map)
    return groups, items


def create_thumbnail(source: Path, target: Path) -> None:
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(source) as image:
            image = image.convert("RGB")
            thumb = ImageOps.contain(image, IMAGE_SIZE, Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", IMAGE_SIZE, "white")
            canvas.paste(thumb, ((IMAGE_SIZE[0] - thumb.width) // 2, (IMAGE_SIZE[1] - thumb.height) // 2))
    except Exception:
        canvas = Image.new("RGB", IMAGE_SIZE, "#f2f2f2")
        draw = ImageDraw.Draw(canvas)
        draw.text((75, 100), "Unreadable image", fill="#444")
    canvas.save(target, quality=88)


def package_groups(groups: pd.DataFrame, items: pd.DataFrame, root: Path, package_dir: Path, zip_path: Path, max_groups: int | None) -> None:
    ensure_clean_dir(package_dir)
    if max_groups is not None:
        groups = groups.head(max_groups).copy()
        items = items[items["group_id"].isin(set(groups["group_id"]))].copy()

    group_objects = []
    for _, group in groups.iterrows():
        group_items = []
        for _, item in items[items["group_id"].eq(group["group_id"])].iterrows():
            rel = str(item["relative_path"])
            thumb_name = f"{hashlib.md5(rel.encode('utf-8')).hexdigest()[:12]}_{Path(rel).stem}.jpg"
            thumb_rel = f"thumbnails/{thumb_name}"
            create_thumbnail(root / rel, package_dir / thumb_rel)
            group_items.append({"relative_path": rel, "label": item["label"], "thumbnail": thumb_rel})
        group_objects.append(
            {
                "group_id": group["group_id"],
                "size": int(group["size"]),
                "labels": group["labels"],
                "cross_label": bool(group["cross_label"]),
                "reasons": group["reasons"],
                "status": "review",
                "reviewed_at": "",
                "items": group_items,
            }
        )

    package_id = f"cats_duplicate_review_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    html = HTML_TEMPLATE.replace("__PACKAGE_ID__", package_id).replace("__GROUPS_JSON__", json.dumps(group_objects, ensure_ascii=False))
    (package_dir / "index.html").write_text(html, encoding="utf-8")
    (package_dir / "README.md").write_text(README_TEXT, encoding="utf-8")
    (package_dir / "duplicate_groups.json").write_text(json.dumps(group_objects, ensure_ascii=False, indent=2), encoding="utf-8")

    if zip_path.exists():
        move_to_trash(zip_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in package_dir.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(package_dir.parent))


def run() -> None:
    parser = argparse.ArgumentParser(description="Build duplicate image review package for Cats vs. Dogs.")
    parser.add_argument("--phash-threshold", type=int, default=4)
    parser.add_argument("--dhash-threshold", type=int, default=3)
    parser.add_argument("--ahash-threshold", type=int, default=-1)
    parser.add_argument("--force-hash", action="store_true")
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--zip-path", type=Path, required=True)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    root, _ = ensure_extracted("cats_vs_dogs")
    frame = compute_hash_cache(root, args.force_hash)
    groups, items = build_duplicate_groups(frame, args.phash_threshold, args.dhash_threshold, args.ahash_threshold)
    groups.to_csv(OUTPUT_DIR / "duplicate_groups.csv", index=False, encoding="utf-8-sig")
    items.to_csv(OUTPUT_DIR / "duplicate_group_items.csv", index=False, encoding="utf-8-sig")
    package_groups(groups, items, root, PACKAGE_DIR, args.zip_path, args.max_groups)

    summary = {
        "total_images": int(len(frame)),
        "valid_images": int(frame["valid"].astype(bool).sum()),
        "group_count": int(len(groups)),
        "candidate_image_count": int(len(items["relative_path"].unique())) if not items.empty else 0,
        "cross_label_group_count": int(groups["cross_label"].sum()) if not groups.empty else 0,
        "phash_threshold": args.phash_threshold,
        "dhash_threshold": args.dhash_threshold,
        "ahash_threshold": args.ahash_threshold,
        "zip_name": args.zip_path.name,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (OUTPUT_DIR / "duplicate_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
