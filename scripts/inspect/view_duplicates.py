"""Docling parsing visualizations for the OCR ↔ "Documents IAGO" duplicates.

Uses manifest_fragus_clean.json (corpus re-uploaded to /fragus_corpus_clean/).
In this corpus version, the 5 "Documents IAGO/" versions (source files,
non-OCRed) were parsed successfully; the OCR versions failed on an expired
token (recoverable on the next run).

Generates one HTML file per parsed PDF plus a summary index.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import tomllib
from pathlib import Path

from dotenv import load_dotenv

from cleyrop import CleyropClient, ClientConfig
from docling_core.types.doc import DoclingDocument

sys.path.insert(0, str(Path(__file__).parent))
from view_parsed import md_to_html_body, render_html  # noqa: E402

load_dotenv(override=True)

ROOT = Path(__file__).parent.parent.parent
_config_path = ROOT / "config.toml"
with open(_config_path, "rb") as f:
    _config = tomllib.load(f)


# (name, ocr_id, iago_id) — IDs from manifest_fragus_clean.json
# NOTE: fictional placeholder examples — replace with the duplicate pairs
# (names and file IDs) from your own manifest.
PAIRS = [
    ("report_A.pdf",
     "00000000-0000-0000-0000-000000000a01",
     "00000000-0000-0000-0000-000000000a02"),
    ("report_B.pdf",
     "00000000-0000-0000-0000-000000000b01",
     "00000000-0000-0000-0000-000000000b02"),
    ("letter_C.pdf",
     "00000000-0000-0000-0000-000000000c01",
     "00000000-0000-0000-0000-000000000c02"),
    ("letter_D.pdf",
     "00000000-0000-0000-0000-000000000d01",
     "00000000-0000-0000-0000-000000000d02"),
    ("technical_guide_E.pdf",
     "00000000-0000-0000-0000-000000000e01",
     "00000000-0000-0000-0000-000000000e02"),
]

MANIFEST_FILE = "output/manifest_fragus_clean.json"
# Side of the pair to visualize: "iago" (source files, parsed) or "ocr" (OCRed, parsed)
PARSED_SIDE = "iago"


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


def render_index(rows: list[dict]) -> str:
    import html as _html
    body = []
    for r in rows:
        link = f'<a href="{_html.escape(r["file"])}">view</a>' if r["file"] else "—"
        ocr_md = r["ocr_chars"]
        ocr_md_s = f"{ocr_md:,}".replace(",", " ") if ocr_md is not None else "—"
        note = _html.escape(r.get("note") or "")
        body.append(f"""<tr>
  <td>{_html.escape(r['name'])}</td>
  <td>{r['ocr_pages']}</td>
  <td>{r['ocr_size_mb']:.2f} MB</td>
  <td>{ocr_md_s}</td>
  <td>{link}</td>
  <td class="status-{r['iago_status']}">{r['iago_status']}</td>
  <td class="note">{note}</td>
</tr>""")
    rows_html = "\n".join(body)
    side = PARSED_SIDE.upper()
    other = "OCR" if PARSED_SIDE == "iago" else "IAGO"
    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>OCR ↔ Documents IAGO duplicates</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #1e1e2e; color: #cdd6f4; padding: 32px; line-height: 1.5; }}
h1 {{ color: #cba6f7; margin-bottom: 8px; font-size: 1.4em; }}
p {{ color: #a6adc8; margin-bottom: 8px; max-width: 900px; }}
.warn {{ background: #382e2e; border-left: 4px solid #f38ba8; padding: 10px 16px;
         border-radius: 4px; margin: 16px 0; max-width: 900px; }}
table {{ border-collapse: collapse; max-width: 1100px; margin-top: 16px; }}
th, td {{ border: 1px solid #45475a; padding: 8px 14px; text-align: left; }}
th {{ background: #313244; color: #cba6f7; }}
tr:nth-child(even) {{ background: #181825; }}
a {{ color: #89b4fa; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
.status-done {{ color: #a6e3a1; }}
.status-error {{ color: #f38ba8; }}
</style>
</head>
<body>
<h1>Duplicates: documents_ocrises_20250929 ↔ Documents IAGO</h1>
<p>5 documents present as duplicates in both folders (manifest_fragus_clean).
Parsings visualized below: <strong>{side}</strong> versions.</p>
<table>
  <tr>
    <th>Document</th>
    <th>Pages</th>
    <th>PDF size</th>
    <th>Parsed chars ({side})</th>
    <th>Parsing {side}</th>
    <th>Status {other}</th>
    <th>Note</th>
  </tr>
  {rows_html}
</table>
</body>
</html>"""


def main() -> None:
    convert_dir = ROOT / "output" / "convert_manifests"
    out_dir = ROOT / "output" / "view" / "duplicates"
    out_dir.mkdir(parents=True, exist_ok=True)

    cleyrop_cfg = make_cleyrop_config()
    rows: list[dict] = []

    with open(ROOT / MANIFEST_FILE) as f:
        manifest = {fl["id"]: fl for fl in json.load(f)["files"]}

    with CleyropClient(cleyrop_cfg) as cleyrop:
        if cleyrop_cfg.client_secret:
            cleyrop.login_client_credentials()

        for name, ocr_id, iago_id in PAIRS:
            print(f"[{name}]")
            ocr_cm_path = convert_dir / f"{ocr_id}.json"
            iago_cm_path = convert_dir / f"{iago_id}.json"
            ocr_cm = json.loads(ocr_cm_path.read_text()) if ocr_cm_path.exists() else {}
            iago_cm = json.loads(iago_cm_path.read_text()) if iago_cm_path.exists() else {}

            ocr_status = ocr_cm.get("status", "no_manifest")
            iago_status = iago_cm.get("status", "no_manifest")

            target_id = iago_id if PARSED_SIDE == "iago" else ocr_id
            target_cm = iago_cm if PARSED_SIDE == "iago" else ocr_cm
            other_status = ocr_status if PARSED_SIDE == "iago" else iago_status

            if target_cm.get("status") != "done" or not target_cm.get("cleyrop_json_file_id"):
                print(f"  ⚠ no parsing available on the {PARSED_SIDE} side → skip")
                rows.append({
                    "name": name,
                    "file": None,
                    "ocr_pages": "—",
                    "ocr_size_mb": 0,
                    "ocr_chars": None,
                    "iago_status": other_status,
                    "note": f"{PARSED_SIDE}: no parsing available",
                })
                continue

            json_id = target_cm["cleyrop_json_file_id"]
            print(f"  - download PDF + JSON parsing ({PARSED_SIDE})")
            try:
                pdf_bytes = cleyrop.download_file_bytes(target_id)
                json_bytes = cleyrop.download_file_bytes(json_id)
            except Exception as e:
                msg = str(e)
                short = "virus scan pending" if "virus scan" in msg else msg[:80]
                print(f"  ⚠ download failed: {short}")
                rows.append({
                    "name": name,
                    "file": None,
                    "ocr_pages": "—",
                    "ocr_size_mb": 0,
                    "ocr_chars": None,
                    "iago_status": other_status,
                    "note": short,
                })
                continue

            doc = DoclingDocument.model_validate_json(json_bytes)
            entry = manifest[target_id]
            html = render_html(entry, doc, base64.b64encode(pdf_bytes).decode("ascii"))

            safe_name = name.replace("/", "_").replace(" ", "_")
            out_path = out_dir / f"{safe_name}.html"
            out_path.write_text(html, encoding="utf-8")
            print(f"  ✓ {out_path.name}  ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")

            md = doc.export_to_markdown()
            rows.append({
                "name": name,
                "file": out_path.name,
                "ocr_pages": len(doc.pages) if doc.pages else "?",
                "ocr_size_mb": len(pdf_bytes) / 1024 / 1024,
                "ocr_chars": len(md),
                "iago_status": other_status,
                "note": "",
            })

    index_path = out_dir / "index.html"
    index_path.write_text(render_index(rows), encoding="utf-8")
    print(f"\nIndex: {index_path}")


if __name__ == "__main__":
    main()
