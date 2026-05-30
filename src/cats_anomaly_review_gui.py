from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import sys
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.project_config import PROJECT_ROOT


OUTPUT_DIR = PROJECT_ROOT / "data" / "AlexNet" / "Anomaly"
ITEMS_PATH = OUTPUT_DIR / "review_items.json"
REVIEW_PATH = OUTPUT_DIR / "manual_review.csv"
EVENT_LOG_PATH = OUTPUT_DIR / "manual_review_events.csv"
ALLOWED_STATUS = {"review", "keep", "exclude", "uncertain"}
FIELDNAMES = ["candidate_id", "relative_path", "label", "status", "manual_note", "detected_by", "detected_reason", "reviewed_at"]
REVIEW_LOCK = threading.Lock()


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cats vs. Dogs 异常图片审核</title>
  <style>
    :root { color-scheme: light; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; background: #f6f7f8; color: #1f2933; }
    header { position: sticky; top: 0; z-index: 10; background: #ffffff; border-bottom: 1px solid #d9dee3; padding: 10px 14px; }
    .bar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
    button, select, input { border: 1px solid #c5ccd3; border-radius: 6px; background: #fff; padding: 7px 10px; font-size: 14px; }
    button { cursor: pointer; }
    button.primary { background: #1769aa; color: #fff; border-color: #1769aa; }
    button.bad { background: #b42318; color: #fff; border-color: #b42318; }
    button.warn { background: #9a6700; color: #fff; border-color: #9a6700; }
    button.good { background: #1f7a4d; color: #fff; border-color: #1f7a4d; }
    main { padding: 14px; }
    .stats { font-size: 13px; color: #52606d; margin-top: 8px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 12px; }
    .card { background: #fff; border: 2px solid transparent; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(15, 23, 42, .12); }
    .card.selected { border-color: #1769aa; }
    .card img { display: block; width: 100%; height: 180px; object-fit: contain; background: #f0f2f4; }
    .meta { padding: 9px; font-size: 12px; line-height: 1.45; }
    .path { font-family: Consolas, monospace; word-break: break-all; color: #243b53; }
    .reasons { color: #6b4f00; margin: 4px 0; min-height: 34px; }
    .actions { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin-top: 8px; }
    .status { display: inline-block; border-radius: 999px; padding: 2px 7px; font-weight: 600; background: #e4e7eb; }
    .status.keep { background: #d9f5e5; color: #116149; }
    .status.exclude { background: #ffe2dd; color: #8a1f11; }
    .status.uncertain { background: #fff0c2; color: #7a4d00; }
    .pager { display: flex; gap: 8px; align-items: center; margin: 12px 0; }
  </style>
</head>
<body>
  <header>
    <div class="bar">
      <button onclick="setStatusForSelected('keep')" class="good">K 保留</button>
      <button onclick="setStatusForSelected('exclude')" class="bad">A 异常</button>
      <button onclick="setStatusForSelected('uncertain')" class="warn">U 不确定</button>
      <select id="filter" onchange="render()">
        <option value="unreviewed">未审核</option>
        <option value="all">全部</option>
        <option value="keep">保留</option>
        <option value="exclude">异常</option>
        <option value="uncertain">不确定</option>
      </select>
      <input id="search" placeholder="搜索路径/来源" oninput="render()">
      <button onclick="prevPage()">上一页</button>
      <button onclick="nextPage()">下一页</button>
      <span id="pageInfo"></span>
    </div>
    <div class="stats" id="stats"></div>
  </header>
  <main>
    <div class="grid" id="grid"></div>
  </main>
  <script>
    let items = [];
    let page = 0;
    let pageSize = 80;
    let selected = 0;
    let saveChain = Promise.resolve();

    async function loadItems() {
      const res = await fetch('/api/items');
      items = await res.json();
      restoreLocalBackup();
      render();
    }

    function backupReviewedState() {
      const reviewed = items
        .filter(item => (item.status || 'review') !== 'review' || item.manual_note)
        .map(item => ({
          candidate_id: item.candidate_id,
          relative_path: item.relative_path,
          status: item.status || 'review',
          manual_note: item.manual_note || ''
        }));
      localStorage.setItem('catsAnomalyReviewState', JSON.stringify(reviewed));
    }

    function restoreLocalBackup() {
      try {
        const raw = localStorage.getItem('catsAnomalyReviewState');
        if (!raw) return;
        const backup = JSON.parse(raw);
        const byPath = Object.fromEntries(backup.map(item => [item.relative_path, item]));
        items.forEach(item => {
          const old = byPath[item.relative_path];
          if (old && (item.status || 'review') === 'review') {
            item.status = old.status || 'review';
            item.manual_note = old.manual_note || '';
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
        const statusOk = filter === 'all' || (filter === 'unreviewed' ? status === 'review' : status === filter);
        const text = `${item.relative_path} ${item.detected_by} ${item.detected_reason}`.toLowerCase();
        return statusOk && (!query || text.includes(query));
      });
    }

    function render() {
      const list = filteredItems();
      const pages = Math.max(1, Math.ceil(list.length / pageSize));
      if (page >= pages) page = pages - 1;
      const start = page * pageSize;
      const visible = list.slice(start, start + pageSize);
      const grid = document.getElementById('grid');
      grid.innerHTML = '';
      visible.forEach((item, index) => {
        const card = document.createElement('div');
        card.className = 'card' + (index === selected ? ' selected' : '');
        card.onclick = () => { selected = index; render(); };
        card.innerHTML = `
          <img src="/${item.thumbnail}" loading="lazy">
          <div class="meta">
            <span class="status ${item.status || 'review'}">${item.status || 'review'}</span>
            <div class="path">${item.relative_path}</div>
            <div>${item.label || ''} | ${item.detected_by || ''}</div>
            <div class="reasons">${item.detected_reason || ''}</div>
            <div>IF rank: ${item.iforest_rank || '-'} | KM rank: ${item.kmeans_rank || '-'}</div>
            <div class="actions">
              <button class="good" onclick="event.stopPropagation(); setStatus('${item.candidate_id}', 'keep')">保留</button>
              <button class="bad" onclick="event.stopPropagation(); setStatus('${item.candidate_id}', 'exclude')">异常</button>
              <button class="warn" onclick="event.stopPropagation(); setStatus('${item.candidate_id}', 'uncertain')">不确定</button>
            </div>
          </div>`;
        grid.appendChild(card);
      });
      const counts = items.reduce((acc, item) => {
        const s = item.status || 'review';
        acc[s] = (acc[s] || 0) + 1;
        return acc;
      }, {});
      document.getElementById('stats').textContent =
        `总数 ${items.length} | 未审核 ${counts.review || 0} | 保留 ${counts.keep || 0} | 异常 ${counts.exclude || 0} | 不确定 ${counts.uncertain || 0}`;
      document.getElementById('pageInfo').textContent = `${page + 1}/${pages} (${list.length})`;
    }

    function setStatus(candidateId, status) {
      const item = items.find(x => x.candidate_id === candidateId);
      if (item) item.status = status;
      backupReviewedState();
      const list = filteredItems();
      let visibleCount = Math.min(pageSize, Math.max(0, list.length - page * pageSize));
      if (visibleCount === 0 && page > 0) {
        page -= 1;
        visibleCount = Math.min(pageSize, Math.max(0, list.length - page * pageSize));
      }
      selected = Math.min(selected, Math.max(0, visibleCount - 1));
      render();
      saveChain = saveChain.then(async () => {
        const res = await fetch('/api/review', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({candidate_id: candidateId, status})
        });
        if (!res.ok) throw new Error(await res.text());
      }).catch(error => {
        console.error(error);
        alert('保存失败，请先暂停审核并检查服务日志。');
      });
    }

    function setStatusForSelected(status) {
      const list = filteredItems();
      const item = list[page * pageSize + selected];
      if (item) setStatus(item.candidate_id, status);
    }

    function nextPage() { page += 1; selected = 0; render(); }
    function prevPage() { page = Math.max(0, page - 1); selected = 0; render(); }

    document.addEventListener('keydown', event => {
      if (event.target.tagName === 'INPUT') return;
      if (event.key.toLowerCase() === 'k') setStatusForSelected('keep');
      if (event.key.toLowerCase() === 'a') setStatusForSelected('exclude');
      if (event.key.toLowerCase() === 'u') setStatusForSelected('uncertain');
      if (event.key === 'ArrowRight') { selected = Math.min(selected + 1, pageSize - 1); render(); }
      if (event.key === 'ArrowLeft') { selected = Math.max(selected - 1, 0); render(); }
    });

    loadItems();
  </script>
</body>
</html>
"""


def load_items() -> list[dict]:
    if not ITEMS_PATH.exists():
        raise FileNotFoundError(f"Review items not found: {ITEMS_PATH}")
    items = json.loads(ITEMS_PATH.read_text(encoding="utf-8"))
    reviews = load_reviews()
    for item in items:
        review = reviews.get(item["candidate_id"])
        if review:
            item["status"] = review["status"]
            item["manual_note"] = review.get("manual_note", "")
    return items


def load_reviews() -> dict[str, dict[str, str]]:
    if not REVIEW_PATH.exists():
        return {}
    with REVIEW_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or "candidate_id" not in reader.fieldnames:
            return {}
        return {
            row["candidate_id"]: row
            for row in reader
            if row.get("candidate_id")
        }


def write_reviews_atomic(reviews: dict[str, dict[str, str]]) -> None:
    tmp_path = REVIEW_PATH.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(reviews.values())
    tmp_path.replace(REVIEW_PATH)


def append_event(row: dict[str, str]) -> None:
    exists = EVENT_LOG_PATH.exists()
    with EVENT_LOG_PATH.open("a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def base_review_rows() -> dict[str, dict[str, str]]:
    items = json.loads(ITEMS_PATH.read_text(encoding="utf-8"))
    rows: dict[str, dict[str, str]] = {}
    for item in items:
        rows[item["candidate_id"]] = {
            "candidate_id": item["candidate_id"],
            "relative_path": item["relative_path"],
            "label": item.get("label", ""),
            "status": "review",
            "manual_note": "",
            "detected_by": item.get("detected_by", ""),
            "detected_reason": item.get("detected_reason", ""),
            "reviewed_at": "",
        }
    return rows


def save_review(candidate_id: str, status: str, note: str = "") -> None:
    if status not in ALLOWED_STATUS:
        raise ValueError(f"Unsupported status: {status}")
    with REVIEW_LOCK:
        reviews = base_review_rows()
        existing = load_reviews()
        for old_id, old_row in existing.items():
            if old_id in reviews:
                reviews[old_id].update({key: old_row.get(key, "") for key in FIELDNAMES})

        if candidate_id not in reviews:
            raise KeyError(f"Unknown candidate_id: {candidate_id}")

        reviewed_at = datetime.now().isoformat(timespec="seconds")
        reviews[candidate_id]["status"] = status
        reviews[candidate_id]["manual_note"] = note
        reviews[candidate_id]["reviewed_at"] = reviewed_at
        append_event(reviews[candidate_id])
        write_reviews_atomic(reviews)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/items":
            self.send_json(load_items())
            return
        if parsed.path.startswith("/thumbnails/"):
            rel = unquote(parsed.path.lstrip("/"))
            target = (OUTPUT_DIR / rel).resolve()
            if not str(target).startswith(str(OUTPUT_DIR.resolve())) or not target.exists():
                self.send_error(404)
                return
            mime = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            self.send_bytes(target.read_bytes(), mime)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/review":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        save_review(str(payload["candidate_id"]), str(payload["status"]), str(payload.get("manual_note", "")))
        self.send_json({"ok": True})

    def send_json(self, data: object) -> None:
        self.send_bytes(json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def send_bytes(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def run() -> None:
    parser = argparse.ArgumentParser(description="Start a local manual review GUI for Cats vs. Dogs anomaly candidates.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    load_items()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Review GUI: http://{args.host}:{args.port}")
    print(f"Labels are saved to: {REVIEW_PATH}")
    server.serve_forever()


if __name__ == "__main__":
    run()
