from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.project_config import PROJECT_ROOT


ANOMALY_DIR = PROJECT_ROOT / "data" / "AlexNet" / "Anomaly"
DEFAULT_BUILD_DIR = PROJECT_ROOT / "data" / "AlexNet" / "Anomaly_review_package"


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cats vs. Dogs 异常候选审核包</title>
  <style>
    :root { font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #20242a; background: #f6f7f9; }
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
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 12px; }
    .card { background: #fff; border: 2px solid transparent; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(15, 23, 42, .12); }
    .card.selected { border-color: #1769aa; }
    .card img { display: block; width: 100%; height: 180px; object-fit: contain; background: #f0f2f4; }
    .meta { padding: 9px; font-size: 12px; line-height: 1.45; }
    .path { font-family: Consolas, monospace; word-break: break-all; color: #243b53; }
    .reasons { color: #6b4f00; min-height: 32px; margin: 4px 0; }
    .status { display: inline-block; border-radius: 999px; padding: 2px 7px; font-weight: 700; background: #e4e7eb; }
    .status.keep { background: #d9f5e5; color: #116149; }
    .status.exclude { background: #ffe2dd; color: #8a1f11; }
    .status.uncertain { background: #fff0c2; color: #7a4d00; }
    .actions { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin-top: 8px; }
  </style>
</head>
<body>
  <header>
    <div class="bar">
      <button class="good" onclick="setStatusForSelected('keep')">K 保留</button>
      <button class="bad" onclick="setStatusForSelected('exclude')">A 异常</button>
      <button class="warn" onclick="setStatusForSelected('uncertain')">U 不确定</button>
      <select id="filter" onchange="page=0; selected=0; render()">
        <option value="unreviewed">未审核</option>
        <option value="all">全部</option>
        <option value="keep">保留</option>
        <option value="exclude">异常</option>
        <option value="uncertain">不确定</option>
      </select>
      <input id="search" placeholder="搜索路径/来源" oninput="page=0; selected=0; render()">
      <button onclick="prevPage()">上一页</button>
      <button onclick="nextPage()">下一页</button>
      <button class="primary" onclick="downloadCsv()">导出 CSV</button>
      <button onclick="downloadJson()">导出 JSON</button>
      <span id="pageInfo"></span>
    </div>
    <div class="note">
      标准：只要肉眼能看出是真实猫/狗且标签正确，选“保留”。卡通、纯文字、空白、没有猫狗、无法判断主体、标签明显错误，选“异常”。不确定就选“不确定”。进度自动保存在当前浏览器，完成后请导出 CSV。
    </div>
    <div class="note" id="stats"></div>
  </header>
  <main><div id="grid" class="grid"></div></main>
  <script>
    const PACKAGE_ID = "__PACKAGE_ID__";
    const ITEMS = __ITEMS_JSON__;
    let items = ITEMS.map(item => ({...item, status: item.status || 'review', manual_note: item.manual_note || ''}));
    let page = 0;
    let pageSize = 80;
    let selected = 0;

    function saveLocal() {
      const reviewed = items.filter(item => (item.status || 'review') !== 'review' || item.manual_note);
      localStorage.setItem(PACKAGE_ID, JSON.stringify(reviewed));
    }

    function loadLocal() {
      try {
        const raw = localStorage.getItem(PACKAGE_ID);
        if (!raw) return;
        const reviewed = JSON.parse(raw);
        const byPath = Object.fromEntries(reviewed.map(item => [item.relative_path, item]));
        items.forEach(item => {
          const old = byPath[item.relative_path];
          if (old) {
            item.status = old.status || 'review';
            item.manual_note = old.manual_note || '';
            item.reviewed_at = old.reviewed_at || item.reviewed_at || '';
          }
        });
      } catch (error) {
        console.warn(error);
      }
    }

    function filteredItems() {
      const filter = document.getElementById('filter').value;
      const query = document.getElementById('search').value.trim().toLowerCase();
      return items.filter(item => {
        const status = item.status || 'review';
        const ok = filter === 'all' || (filter === 'unreviewed' ? status === 'review' : status === filter);
        const text = `${item.relative_path} ${item.detected_by} ${item.detected_reason}`.toLowerCase();
        return ok && (!query || text.includes(query));
      });
    }

    function render() {
      const list = filteredItems();
      const pages = Math.max(1, Math.ceil(list.length / pageSize));
      if (page >= pages) page = pages - 1;
      const visible = list.slice(page * pageSize, page * pageSize + pageSize);
      const grid = document.getElementById('grid');
      grid.innerHTML = '';
      visible.forEach((item, index) => {
        const card = document.createElement('div');
        card.className = 'card' + (index === selected ? ' selected' : '');
        card.onclick = () => { selected = index; render(); };
        card.innerHTML = `
          <img src="${item.thumbnail}" loading="lazy" alt="">
          <div class="meta">
            <span class="status ${item.status || 'review'}">${item.status || 'review'}</span>
            <div class="path">${item.relative_path}</div>
            <div>${item.label || ''} | ${item.detected_by || ''}</div>
            <div class="reasons">${item.detected_reason || ''}</div>
            <div class="actions">
              <button class="good" onclick="event.stopPropagation(); setStatus('${item.relative_path}', 'keep')">保留</button>
              <button class="bad" onclick="event.stopPropagation(); setStatus('${item.relative_path}', 'exclude')">异常</button>
              <button class="warn" onclick="event.stopPropagation(); setStatus('${item.relative_path}', 'uncertain')">不确定</button>
            </div>
          </div>`;
        grid.appendChild(card);
      });
      const counts = items.reduce((acc, item) => {
        const status = item.status || 'review';
        acc[status] = (acc[status] || 0) + 1;
        return acc;
      }, {});
      document.getElementById('stats').textContent =
        `总数 ${items.length} | 未审核 ${counts.review || 0} | 保留 ${counts.keep || 0} | 异常 ${counts.exclude || 0} | 不确定 ${counts.uncertain || 0}`;
      document.getElementById('pageInfo').textContent = `${page + 1}/${pages} (${list.length})`;
    }

    function setStatus(relativePath, status) {
      const item = items.find(item => item.relative_path === relativePath);
      if (!item) return;
      item.status = status;
      item.reviewed_at = new Date().toISOString();
      saveLocal();
      const list = filteredItems();
      let visibleCount = Math.min(pageSize, Math.max(0, list.length - page * pageSize));
      if (visibleCount === 0 && page > 0) {
        page -= 1;
        visibleCount = Math.min(pageSize, Math.max(0, list.length - page * pageSize));
      }
      selected = Math.min(selected, Math.max(0, visibleCount - 1));
      render();
    }

    function setStatusForSelected(status) {
      const list = filteredItems();
      const item = list[page * pageSize + selected];
      if (item) setStatus(item.relative_path, status);
    }

    function nextPage() { page += 1; selected = 0; render(); }
    function prevPage() { page = Math.max(0, page - 1); selected = 0; render(); }

    function csvEscape(value) {
      const text = String(value ?? '');
      return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
    }

    function downloadCsv() {
      const fields = ['candidate_id','relative_path','label','status','manual_note','detected_by','detected_reason','reviewed_at'];
      const rows = [fields.join(',')].concat(items.map(item => fields.map(field => csvEscape(item[field])).join(',')));
      downloadText(rows.join('\n'), `review_result_${PACKAGE_ID}.csv`, 'text/csv;charset=utf-8');
    }

    function downloadJson() {
      downloadText(JSON.stringify(items, null, 2), `review_result_${PACKAGE_ID}.json`, 'application/json;charset=utf-8');
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
      if (key === 'k') setStatusForSelected('keep');
      if (key === 'a') setStatusForSelected('exclude');
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


README_TEXT = """# Cats vs. Dogs 异常候选审核包

## 使用方法

1. 先解压整个 zip。
2. 双击打开 `index.html`。
3. 按 `K` 表示保留，按 `A` 表示异常，按 `U` 表示不确定。
4. 审核完成后点击页面顶部的“导出 CSV”，把下载得到的 CSV 发回。

## 审核标准

只要肉眼能看出是真实猫/狗，并且标签正确，就选“保留”。

以下情况选“异常”：

- 卡通图、插画、表情包；
- 纯文字、空白、几乎看不出有效图像；
- 没有猫/狗主体；
- 标签明显错误；
- 主体无法判断。

有白边、背景复杂、动物占比小、光照差、照片质量一般，只要仍能看出是真实猫/狗，就选“保留”。
"""


def clean_dir(path: Path) -> None:
    if path.exists():
        trash_dir = PROJECT_ROOT / ".trash"
        trash_dir.mkdir(parents=True, exist_ok=True)
        target = trash_dir / f"{path.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.move(str(path), str(target))
    path.mkdir(parents=True, exist_ok=True)


def load_pending_items() -> list[dict]:
    review_path = ANOMALY_DIR / "manual_review.csv"
    candidates_path = ANOMALY_DIR / "anomaly_candidates.csv"

    reviews = pd.read_csv(review_path, encoding="utf-8-sig").fillna("")
    candidates = pd.read_csv(candidates_path, encoding="utf-8-sig").fillna("")
    merged = candidates.merge(
        reviews[["relative_path", "status", "manual_note", "reviewed_at"]],
        on="relative_path",
        how="left",
        suffixes=("", "_review"),
    )
    merged["status"] = merged["status_review"].where(merged["status_review"].astype(bool), merged["status"])
    merged["manual_note"] = merged["manual_note_review"].where(
        merged["manual_note_review"].astype(bool), merged.get("manual_note", "")
    )
    merged["reviewed_at"] = merged["reviewed_at_review"].where(
        merged["reviewed_at_review"].astype(bool), merged.get("reviewed_at", "")
    )
    pending = merged[merged["status"].eq("review")].copy()

    fields = [
        "candidate_id",
        "relative_path",
        "label",
        "status",
        "manual_note",
        "detected_by",
        "detected_reason",
        "reviewed_at",
        "thumbnail",
    ]
    return pending.loc[:, fields].to_dict(orient="records")


def copy_thumbnails(items: list[dict], package_dir: Path) -> None:
    thumb_dir = package_dir / "thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        thumb_rel = Path(item["thumbnail"])
        source = ANOMALY_DIR / thumb_rel
        dest = package_dir / thumb_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)


def write_package(package_dir: Path, items: list[dict], package_id: str) -> None:
    clean_dir(package_dir)
    copy_thumbnails(items, package_dir)
    (package_dir / "pending_items.json").write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    html = HTML_TEMPLATE.replace("__PACKAGE_ID__", package_id).replace(
        "__ITEMS_JSON__",
        json.dumps(items, ensure_ascii=False),
    )
    (package_dir / "index.html").write_text(html, encoding="utf-8")
    (package_dir / "README.md").write_text(README_TEXT, encoding="utf-8")


def make_zip(package_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in package_dir.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(package_dir.parent))


def run() -> None:
    parser = argparse.ArgumentParser(description="Build a static manual-review package for collaborators.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--zip-path", type=Path, required=True)
    args = parser.parse_args()

    package_id = f"cats_anomaly_review_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    items = load_pending_items()
    write_package(args.output_dir, items, package_id)
    make_zip(args.output_dir, args.zip_path)
    print(json.dumps({
        "package_dir": str(args.output_dir),
        "zip_path": str(args.zip_path),
        "pending_count": len(items),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
