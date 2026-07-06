"""Generate an HTML visualization of the chunks for N randomly sampled documents.

Downloads the JSONL files from Cleyrop (via chunk_manifests) and produces a
single self-contained HTML file with:
  - list of the selected docs (table of contents)
  - for each doc, each chunk collapsible with:
      • CONTEXT (Document / Path / Section / Caption)
      • CHUNK (raw body)
      • metadata (num_tokens, pages, chunk_id)

Usage:
    uv run scripts/inspect/visualize_chunks.py [--n 5] [--seed 42] [--out output/chunks_preview.html]
"""

from __future__ import annotations

import argparse
import html
import json
import os
import random
import tomllib
from pathlib import Path

from dotenv import load_dotenv

from cleyrop import CleyropClient, ClientConfig

load_dotenv(override=True)

_config_path = Path(__file__).parent.parent.parent / "config.toml"
with open(_config_path, "rb") as f:
    _config = tomllib.load(f)


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


def pick_random_docs(manifests_dir: Path, n: int, seed: int) -> list[dict]:
    done = []
    for p in manifests_dir.glob("*.json"):
        d = json.loads(p.read_text("utf-8"))
        if d.get("status") == "done" and d.get("cleyrop_jsonl_file_id"):
            done.append(d)
    rng = random.Random(seed)
    rng.shuffle(done)
    return done[:n]


def split_context_and_body(text: str) -> tuple[str, str]:
    """Split the CONTEXT preamble and the CHUNK body."""
    marker = "\nCHUNK:\n"
    if marker in text:
        head, body = text.split(marker, 1)
        return head.strip(), body.strip()
    return "", text.strip()


def render_html(docs_with_chunks: list[tuple[dict, list[dict]]]) -> str:
    parts = ["""<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8">
<title>Chunk preview — fragus</title>
<style>
  :root {
    --bg: #fafafa; --fg: #222; --muted: #666; --accent: #2b5fb3;
    --border: #ddd; --code-bg: #f4f4f4; --chunk-bg: #fff;
    --ctx-bg: #eef3fb;
  }
  body { font-family: -apple-system, system-ui, sans-serif; background: var(--bg); color: var(--fg); margin: 0; padding: 0; }
  header { background: #1f2a44; color: #fff; padding: 1rem 2rem; }
  header h1 { margin: 0; font-size: 1.4rem; }
  header .meta { color: #b9c4dc; font-size: 0.9rem; margin-top: 0.3rem; }
  nav.toc { padding: 1rem 2rem; background: #fff; border-bottom: 1px solid var(--border); }
  nav.toc ul { margin: 0.5rem 0 0; padding: 0; list-style: none; }
  nav.toc li { margin: 0.2rem 0; }
  nav.toc a { color: var(--accent); text-decoration: none; }
  nav.toc a:hover { text-decoration: underline; }
  main { max-width: 1100px; margin: 0 auto; padding: 1.5rem 2rem 4rem; }
  section.doc { margin-bottom: 3rem; }
  section.doc > h2 { border-bottom: 2px solid var(--accent); padding-bottom: 0.3rem; color: #1f2a44; }
  .doc-meta { color: var(--muted); font-size: 0.85rem; margin-bottom: 1rem; }
  .doc-meta code { background: var(--code-bg); padding: 0.1rem 0.3rem; border-radius: 3px; font-size: 0.8rem; }
  details.chunk { background: var(--chunk-bg); border: 1px solid var(--border); border-radius: 6px; margin: 0.6rem 0; padding: 0.3rem 0.8rem; }
  details.chunk[open] { box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
  details.chunk > summary { cursor: pointer; padding: 0.4rem 0; font-weight: 500; outline: none; }
  details.chunk > summary .tag { display: inline-block; background: #e8eef9; color: var(--accent); padding: 0.05rem 0.4rem; border-radius: 3px; font-size: 0.75rem; margin-right: 0.5rem; font-weight: normal; }
  details.chunk > summary .pages { color: var(--muted); font-size: 0.8rem; margin-left: 0.4rem; font-weight: normal; }
  .ctx { background: var(--ctx-bg); padding: 0.6rem 0.9rem; border-left: 3px solid var(--accent); border-radius: 3px; margin: 0.5rem 0; font-size: 0.85rem; color: #333; white-space: pre-wrap; font-family: ui-monospace, monospace; }
  .body { background: #fff; padding: 0.8rem; border: 1px dashed #ccc; border-radius: 3px; white-space: pre-wrap; font-size: 0.92rem; line-height: 1.5; }
  .stats { display: flex; gap: 1rem; flex-wrap: wrap; font-size: 0.8rem; color: var(--muted); margin: 0.4rem 0; }
  .stats span { background: #f0f0f0; padding: 0.1rem 0.5rem; border-radius: 3px; }
  .summary-bar { background: #fff; border: 1px solid var(--border); padding: 0.8rem; border-radius: 6px; margin-bottom: 1rem; display: flex; gap: 1.5rem; flex-wrap: wrap; font-size: 0.9rem; }
  .summary-bar b { color: var(--accent); }
</style></head><body>
"""]

    total_chunks = sum(len(chunks) for _, chunks in docs_with_chunks)
    parts.append(f"""<header>
<h1>Chunk preview — fragus</h1>
<div class="meta">{len(docs_with_chunks)} randomly sampled documents · {total_chunks} chunks total · HybridChunker bge-m3 (max_tokens=512, hard_cap=1024)</div>
</header>""")

    # TOC
    parts.append('<nav class="toc"><b>Documents:</b><ul>')
    for doc, chunks in docs_with_chunks:
        anchor = f"doc-{doc['file_id'][:8]}"
        parts.append(f'<li><a href="#{anchor}">{html.escape(doc["name"])}</a> <span style="color:#999;font-size:0.85em;">({len(chunks)} chunks · {doc.get("num_pages", "?")}p)</span></li>')
    parts.append("</ul></nav><main>")

    for doc, chunks in docs_with_chunks:
        anchor = f"doc-{doc['file_id'][:8]}"
        token_counts = [c.get("docling_meta", {}).get("num_tokens", 0) for c in chunks]
        avg_tok = sum(token_counts) // len(token_counts) if token_counts else 0
        max_tok = max(token_counts) if token_counts else 0
        min_tok = min(token_counts) if token_counts else 0

        parts.append(f"""<section class="doc" id="{anchor}">
<h2>{html.escape(doc['name'])}</h2>
<div class="doc-meta">
  <div><code>{html.escape(doc['path'])}</code></div>
  <div>file_id: <code>{doc['file_id']}</code></div>
</div>
<div class="summary-bar">
  <span><b>{len(chunks)}</b> chunks</span>
  <span><b>{doc.get('num_pages', '?')}</b> pages</span>
  <span>tokens / chunk: min <b>{min_tok}</b> · avg <b>{avg_tok}</b> · max <b>{max_tok}</b></span>
  <span>convert: {doc.get('elapsed_s', '?')}s</span>
</div>""")

        for idx, c in enumerate(chunks):
            ctx, body = split_context_and_body(c["text"])
            pages = c.get("page_numbers") or []
            pages_str = f"p.{pages[0]}" if len(pages) == 1 else (f"p.{pages[0]}–{pages[-1]}" if pages else "—")
            num_tok = c.get("docling_meta", {}).get("num_tokens", 0)
            headings = c.get("docling_meta", {}).get("headings") or []
            heading_str = " > ".join(headings) if headings else "(root)"
            preview = body[:80].replace("\n", " ")
            parts.append(f"""<details class="chunk">
<summary>
  <span class="tag">#{idx:03d}</span>
  <span>{html.escape(heading_str)}</span>
  <span class="pages">· {pages_str} · {num_tok} tok</span>
  <div style="color:#888;font-size:0.8rem;margin-top:0.2rem;margin-left:3rem;">{html.escape(preview)}…</div>
</summary>
<div class="stats">
  <span>chunk_id: {c['chunk_id']}</span>
  <span>pages: {pages or '—'}</span>
  <span>tokens: {num_tok}</span>
  <span>headings: {len(headings)}</span>
</div>
<div class="ctx">{html.escape(ctx)}</div>
<div class="body">{html.escape(body)}</div>
</details>""")

        parts.append("</section>")

    parts.append("</main></body></html>")
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="output/chunks_preview.html")
    args = parser.parse_args()

    root = Path(__file__).parent.parent.parent
    manifests_dir = root / "output" / "chunk_manifests"
    out_path = root / args.out

    docs = pick_random_docs(manifests_dir, args.n, args.seed)
    print(f"Sampling {args.n} docs (seed={args.seed}):")
    for d in docs:
        print(f"  - {d['name']} ({d['num_chunks']} chunks, {d.get('num_pages','?')}p)")

    cleyrop_cfg = make_cleyrop_config()
    docs_with_chunks: list[tuple[dict, list[dict]]] = []
    with CleyropClient(cleyrop_cfg) as cleyrop:
        if cleyrop_cfg.client_secret:
            cleyrop.login_client_credentials()
        for d in docs:
            jsonl_bytes = cleyrop.download_file_bytes(d["cleyrop_jsonl_file_id"])
            chunks = [json.loads(line) for line in jsonl_bytes.decode("utf-8").splitlines() if line.strip()]
            docs_with_chunks.append((d, chunks))
            print(f"  ✓ {d['name']} → {len(chunks)} chunks downloaded")

    html_out = render_html(docs_with_chunks)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_out, encoding="utf-8")
    print(f"\n→ HTML written: {out_path}")


if __name__ == "__main__":
    main()
