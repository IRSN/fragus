"""Side-by-side HTML visualization: original PDF (left) and Docling parsing (right).

Downloads the PDF and the DoclingDocument JSON from Cleyrop, generates an HTML
page with both panels in sync.

Usage:
    uv run scripts/inspect/view_parsed.py --name "fragment"   # search by name
    uv run scripts/inspect/view_parsed.py --id <file_id>       # search by ID
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

from dotenv import load_dotenv

from cleyrop import CleyropClient, ClientConfig
from docling_core.types.doc import DoclingDocument

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


# ─────────────────────────── Markdown → HTML rendering ───────────────────────


def md_to_html_body(md: str) -> str:
    import html
    import re

    lines = md.split("\n")
    out: list[str] = []
    in_pre = False
    in_table = False
    buffer: list[str] = []

    def flush_para() -> None:
        if buffer:
            text = " ".join(buffer).strip()
            if text:
                out.append(f"<p>{inline(text)}</p>")
            buffer.clear()

    def inline(text: str) -> str:
        text = html.escape(text, quote=False)
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
        text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
        return text

    for line in lines:
        if line.startswith("```"):
            flush_para()
            if in_pre:
                out.append("</pre>")
            else:
                out.append("<pre><code>")
            in_pre = not in_pre
            continue
        if in_pre:
            out.append(html.escape(line))
            continue

        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            flush_para()
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{inline(m.group(2))}</h{lvl}>")
            continue

        if re.match(r"^[-*_]{3,}$", line.strip()):
            flush_para()
            out.append("<hr>")
            continue

        if line.startswith("> "):
            flush_para()
            out.append(f"<blockquote><p>{inline(line[2:])}</p></blockquote>")
            continue

        if "|" in line and line.strip().startswith("|"):
            flush_para()
            if not in_table:
                out.append("<table>")
                in_table = True
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(re.match(r"^[-:]+$", c) for c in cells):
                continue
            tag = "th" if not any("<td>" in r for r in out[-5:] if r.startswith("<tr>")) else "td"
            out.append("<tr>" + "".join(f"<{tag}>{inline(c)}</{tag}>" for c in cells) + "</tr>")
            continue
        elif in_table:
            out.append("</table>")
            in_table = False

        if not line.strip():
            flush_para()
            continue

        buffer.append(line)

    flush_para()
    if in_table:
        out.append("</table>")
    if in_pre:
        out.append("</pre>")

    return "\n".join(out)


# ─────────────────────────── side-by-side HTML rendering ─────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #1e1e2e; color: #cdd6f4; height: 100vh;
       display: flex; flex-direction: column; overflow: hidden; }

.header { background: #181825; border-bottom: 1px solid #313244;
          padding: 10px 20px; display: flex; align-items: center; gap: 16px;
          flex-shrink: 0; }
.header h1 { font-size: 1em; font-weight: 600; color: #cba6f7;
             white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.meta-pill { font-size: 0.78em; background: #313244; border-radius: 12px;
             padding: 3px 10px; color: #a6adc8; white-space: nowrap; }

.split { display: flex; flex: 1; overflow: hidden; }

.panel { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
.panel + .panel { border-left: 2px solid #313244; }

.panel-title { background: #181825; padding: 8px 16px; font-size: 0.82em;
               font-weight: 600; color: #89b4fa; text-transform: uppercase;
               letter-spacing: 0.05em; flex-shrink: 0; border-bottom: 1px solid #313244; }

.panel-content { flex: 1; overflow: auto; }

/* PDF panel */
iframe { width: 100%; height: 100%; border: none; display: block; }

/* Parsed panel */
.parsed-body { padding: 24px 28px; font-family: Georgia, serif;
               font-size: 0.92em; line-height: 1.7; color: #cdd6f4; }
.parsed-body h1 { font-size: 1.5em; color: #cba6f7; border-bottom: 1px solid #313244;
                  padding-bottom: 8px; margin: 24px 0 12px; }
.parsed-body h2 { font-size: 1.25em; color: #89dceb; margin: 20px 0 8px; }
.parsed-body h3 { font-size: 1.1em; color: #a6e3a1; margin: 16px 0 6px; }
.parsed-body h4, .parsed-body h5, .parsed-body h6 { font-size: 1em; color: #f9e2af;
                  margin: 12px 0 4px; }
.parsed-body p { margin: 8px 0; }
.parsed-body table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 0.9em; }
.parsed-body th, .parsed-body td { border: 1px solid #45475a; padding: 6px 10px; text-align: left; }
.parsed-body th { background: #313244; color: #cba6f7; }
.parsed-body tr:nth-child(even) { background: #1e1e2e; }
.parsed-body tr:nth-child(odd) { background: #181825; }
.parsed-body code { background: #313244; padding: 2px 5px; border-radius: 4px;
                    font-family: monospace; font-size: 0.88em; color: #a6e3a1; }
.parsed-body pre { background: #181825; border: 1px solid #313244; border-radius: 6px;
                   padding: 12px; overflow-x: auto; margin: 12px 0; }
.parsed-body blockquote { border-left: 3px solid #cba6f7; padding-left: 14px;
                          color: #a6adc8; margin: 8px 0; font-style: italic; }
.parsed-body hr { border: none; border-top: 1px solid #313244; margin: 16px 0; }
"""


def render_html(entry: dict, doc: DoclingDocument, pdf_b64: str) -> str:
    import html as _html
    md = doc.export_to_markdown()
    body = md_to_html_body(md)
    num_pages = len(doc.pages) if doc.pages else "?"
    name_esc = _html.escape(entry["name"])
    path_esc = _html.escape(entry["path"])

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>{name_esc}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="header">
  <h1>{name_esc}</h1>
  <span class="meta-pill">{path_esc}</span>
  <span class="meta-pill">{num_pages} pages</span>
  <span class="meta-pill">ID: {entry['id']}</span>
</div>
<div class="split">
  <div class="panel">
    <div class="panel-title">Original PDF</div>
    <div class="panel-content">
      <iframe src="data:application/pdf;base64,{pdf_b64}" type="application/pdf"></iframe>
    </div>
  </div>
  <div class="panel">
    <div class="panel-title">Docling parsing</div>
    <div class="panel-content">
      <div class="parsed-body">{body}</div>
    </div>
  </div>
</div>
</body>
</html>"""


# ─────────────────────────── main ────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Side-by-side PDF + parsing visualization")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--name", help="Filename fragment")
    group.add_argument("--id", dest="file_id", help="Exact file ID")
    parser.add_argument("--manifest", default="output/manifest_fragus_clean.dedup.json")
    parser.add_argument("--open", action="store_true", help="Open in the browser")
    args = parser.parse_args()

    files = [f for f in load_manifest(ROOT / args.manifest) if f.get("mime_type") == "application/pdf"]
    entry = find_entry(files, args.name, args.file_id)
    file_id = entry["id"]

    convert_manifest = ROOT / "output" / "convert_manifests" / f"{file_id}.json"
    if not convert_manifest.exists():
        print(f"[ERROR] No convert_manifest for {entry['name']} — run convert_corpus.py first")
        sys.exit(1)

    with open(convert_manifest, "r", encoding="utf-8") as f:
        cm = json.load(f)

    json_file_id = cm.get("cleyrop_json_file_id")
    if not json_file_id:
        print(f"[ERROR] cleyrop_json_file_id missing in the manifest of {entry['name']}")
        sys.exit(1)

    print(f"Downloading PDF + JSON: {entry['name']} …")
    cleyrop_cfg = make_cleyrop_config()
    with CleyropClient(cleyrop_cfg) as cleyrop:
        if cleyrop_cfg.client_secret:
            cleyrop.login_client_credentials()
        pdf_bytes = cleyrop.download_file_bytes(file_id)
        json_bytes = cleyrop.download_file_bytes(json_file_id)

    pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")
    doc = DoclingDocument.model_validate_json(json_bytes)
    html = render_html(entry, doc, pdf_b64)

    out_dir = ROOT / "output" / "view" / "parsed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{file_id}.html"
    out_path.write_text(html, encoding="utf-8")

    pdf_mb = len(pdf_bytes) / 1024 / 1024
    print(f"HTML generated: {out_path}  ({pdf_mb:.1f} MB embedded PDF)")
    if args.open:
        try:
            subprocess.run(["xdg-open", str(out_path)], check=False)
        except FileNotFoundError:
            print("(xdg-open missing — open the file manually)")


if __name__ == "__main__":
    main()
