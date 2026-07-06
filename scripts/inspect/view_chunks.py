"""HTML visualization of a document's chunks.

Downloads the JSONL from Cleyrop and generates an HTML file with one card per chunk.

Usage:
    uv run scripts/inspect/view_chunks.py --name "fragment"   # search by name
    uv run scripts/inspect/view_chunks.py --id <file_id>       # search by ID
"""

from __future__ import annotations

import argparse
import html as _html
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

from dotenv import load_dotenv

from cleyrop import CleyropClient, ClientConfig

load_dotenv(override=True)

_config_path = Path(__file__).parent.parent.parent / "config.toml"
with open(_config_path, "rb") as f:
    _config = tomllib.load(f)

ROOT = Path(__file__).parent.parent.parent


# ─────────────────────────── cleyrop ─────────────────────────────────────────


def make_cleyrop_config() -> ClientConfig:
    domain = os.environ.get("CLEYROP_DOMAIN") or _config["cleyrop"].get("domain")
    if domain:
        os.environ.setdefault("CLEYROP_DOMAIN", domain)
        return ClientConfig.from_env()
    return ClientConfig.internal(
        client_id=os.environ.get("CLEYROP_CLIENT_ID", "cleyrop-cli"),
        client_secret=os.environ.get("CLEYROP_CLIENT_SECRET"),
        token=os.environ.get("CLEYROP_TOKEN"),
    )


# ─────────────────────────── file resolution ─────────────────────────────────


def load_manifest(manifest_path: Path) -> list[dict]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)["files"]


def find_entry(files: list[dict], name_fragment: str | None, file_id: str | None) -> dict:
    if file_id:
        for f in files:
            if f["id"] == file_id:
                return f
        print(f"[ERROR] ID not found: {file_id}")
        sys.exit(1)

    assert name_fragment
    matches = [f for f in files if name_fragment.lower() in f["name"].lower()]
    if not matches:
        print(f"[ERROR] No file matching '{name_fragment}'")
        sys.exit(1)
    if len(matches) > 1:
        print(f"Multiple matches for '{name_fragment}':")
        for m in matches[:10]:
            print(f"  {m['id']}  {m['name']}")
        print("Specify the ID with --id or refine the name.")
        sys.exit(1)
    return matches[0]


# ─────────────────────────── HTML rendering ──────────────────────────────────

_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 1000px; margin: 40px auto; padding: 0 20px;
       background: #f5f5f5; color: #222; }
h1 { font-size: 1.4em; border-bottom: 2px solid #ccc; padding-bottom: 8px; }
.meta { background: #e8f4f8; border: 1px solid #b8dce8; border-radius: 6px;
        padding: 12px 16px; margin-bottom: 24px; font-size: 0.88em; }
.stats { font-size: 0.88em; color: #555; margin-bottom: 20px; }
.chunk { background: white; border: 1px solid #ddd; border-radius: 8px;
         margin-bottom: 16px; padding: 16px 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
.chunk-header { display: flex; gap: 12px; align-items: baseline;
                font-size: 0.82em; color: #666; margin-bottom: 10px; flex-wrap: wrap; }
.chunk-id { font-family: monospace; background: #f0f0f0; padding: 2px 6px;
            border-radius: 4px; color: #333; }
.badge { background: #e0e7ff; color: #3730a3; padding: 2px 7px;
         border-radius: 10px; font-size: 0.85em; }
.badge.pages { background: #dcfce7; color: #166534; }
.headings { font-size: 0.82em; color: #7c3aed; font-style: italic; margin-bottom: 8px; }
.chunk-text { white-space: pre-wrap; font-size: 0.9em; line-height: 1.55;
              border-top: 1px solid #eee; padding-top: 10px; color: #333; }
.context { color: #888; }
.body { color: #111; }
"""


def _split_text(text: str) -> tuple[str, str]:
    """Split the chunk into context and body."""
    if "\nCHUNK:\n" in text:
        parts = text.split("\nCHUNK:\n", 1)
        return parts[0].strip(), parts[1].strip()
    return "", text.strip()


def render_chunk(chunk: dict, idx: int) -> str:
    chunk_id = _html.escape(chunk.get("chunk_id", f"chunk_{idx}"))
    pages = chunk.get("page_numbers") or []
    pages_label = f"p. {', '.join(str(p) for p in pages)}" if pages else ""
    headings = chunk.get("docling_meta", {}).get("headings") or []
    heading_str = " › ".join(headings) if headings else ""

    context, body = _split_text(chunk.get("text", ""))

    pages_badge = f'<span class="badge pages">{_html.escape(pages_label)}</span>' if pages_label else ""
    heading_div = f'<div class="headings">📂 {_html.escape(heading_str)}</div>' if heading_str else ""

    context_html = f'<span class="context">{_html.escape(context)}</span>\n\n' if context else ""
    body_html = f'<span class="body">{_html.escape(body)}</span>'

    return f"""<div class="chunk">
  <div class="chunk-header">
    <span class="chunk-id">#{idx + 1} — {chunk_id}</span>
    {pages_badge}
    <span class="badge">{len(body)} chars</span>
  </div>
  {heading_div}
  <div class="chunk-text">{context_html}{body_html}</div>
</div>"""


def render_html(entry: dict, chunks: list[dict]) -> str:
    meta_html = (
        f'<div class="meta">'
        f"<strong>File:</strong> {_html.escape(entry['name'])}<br>"
        f"<strong>Path:</strong> {_html.escape(entry['path'])}<br>"
        f"<strong>ID:</strong> {entry['id']}"
        f"</div>"
    )
    stats_html = f'<div class="stats">{len(chunks)} chunks generated</div>'
    chunks_html = "\n".join(render_chunk(c, i) for i, c in enumerate(chunks))
    title = _html.escape(entry["name"])

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Chunks — {title}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Chunks — {title}</h1>
{meta_html}
{stats_html}
{chunks_html}
</body>
</html>"""


# ─────────────────────────── main ────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="HTML visualization of a document's chunks")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--name", help="Filename fragment")
    group.add_argument("--id", dest="file_id", help="Exact file ID")
    parser.add_argument("--manifest", default="output/manifest_fragus_clean.dedup.json")
    parser.add_argument("--open", action="store_true", help="Open in the browser")
    args = parser.parse_args()

    files = [f for f in load_manifest(ROOT / args.manifest) if f.get("mime_type") == "application/pdf"]
    entry = find_entry(files, args.name, args.file_id)
    file_id = entry["id"]

    # Fetch the cleyrop_jsonl_file_id from the chunk_manifest
    chunk_manifest_path = ROOT / "output" / "chunk_manifests" / f"{file_id}.json"
    if not chunk_manifest_path.exists():
        print(f"[ERROR] No chunk_manifest for {entry['name']} — run chunk_corpus.py first")
        sys.exit(1)

    with open(chunk_manifest_path, "r", encoding="utf-8") as f:
        cm = json.load(f)

    jsonl_file_id = cm.get("cleyrop_jsonl_file_id")
    if not jsonl_file_id:
        print(f"[ERROR] cleyrop_jsonl_file_id missing in the manifest of {entry['name']}")
        sys.exit(1)

    print(f"Downloading chunks: {entry['name']} …")
    cleyrop_cfg = make_cleyrop_config()
    with CleyropClient(cleyrop_cfg) as cleyrop:
        if cleyrop_cfg.client_secret:
            cleyrop.login_client_credentials()
        jsonl_bytes = cleyrop.download_file_bytes(jsonl_file_id)

    chunks = [
        json.loads(line)
        for line in jsonl_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]

    print(f"{len(chunks)} chunks loaded.")
    html = render_html(entry, chunks)

    out_dir = ROOT / "output" / "view" / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{file_id}.html"
    out_path.write_text(html, encoding="utf-8")

    print(f"HTML generated: {out_path}")
    if args.open:
        try:
            subprocess.run(["xdg-open", str(out_path)], check=False)
        except FileNotFoundError:
            print("(xdg-open missing — open the file manually)")


if __name__ == "__main__":
    main()
