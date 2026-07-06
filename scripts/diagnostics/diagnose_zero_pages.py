"""Diagnose a converted document that ended up with num_pages=0.

Downloads from Cleyrop the source PDF AND the produced DoclingDocument JSON, then:
  - inspects the raw PDF via pypdfium2 (page count, native text, images);
  - replays the pipeline's needs_ocr() logic;
  - inspects the Docling JSON (pages, texts, tables, pictures).

Goal: figure out why the conversion produced an empty document.

With --reconvert, additionally replays the conversion via docling-serve
(DOCLING_SERVICE_URL) with several configs (prod, forced OCR, alternative
backends) and compares num_pages.

Usage:
    uv run scripts/diagnostics/diagnose_zero_pages.py                  # targets the default doc
    uv run scripts/diagnostics/diagnose_zero_pages.py --id <file_id>   # another source doc
    uv run scripts/diagnostics/diagnose_zero_pages.py --name "report_A"
    uv run scripts/diagnostics/diagnose_zero_pages.py --reconvert      # + replay the conversion
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tomllib
from io import BytesIO
from pathlib import Path

import yaml
from dotenv import load_dotenv

from cleyrop import CleyropClient, ClientConfig

load_dotenv(override=True)

ROOT = Path(__file__).parent.parent.parent
with open(ROOT / "config.toml", "rb") as f:
    _config = tomllib.load(f)

# Problematic doc spotted in output/convert_manifests/ (num_pages=0)
# — replace with your own Cleyrop file_id
DEFAULT_FILE_ID = "00000000-0000-0000-0000-000000000000"

# Same thresholds as convert_corpus.py
_OCR_CHAR_THRESHOLD = 100
_OCR_SAMPLE_PAGES = 3


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


def resolve_file_id(name: str | None, file_id: str | None) -> str:
    if file_id:
        return file_id
    manifest = json.loads(
        (ROOT / "output/manifest_fragus_clean.dedup.json").read_text("utf-8")
    )["files"]
    matches = [f for f in manifest if name.lower() in f["name"].lower()]
    if not matches:
        sys.exit(f"[ERROR] No file matches \"{name}\"")
    if len(matches) > 1:
        print(f"Multiple matches for \"{name}\":")
        for m in matches[:10]:
            print(f"  {m['id']}  {m['name']}")
        sys.exit("Specify --id.")
    return matches[0]["id"]


def inspect_pdf(pdf_bytes: bytes) -> None:
    print(f"\n── Source PDF ── ({len(pdf_bytes) / 1024:.0f} KB)")
    try:
        import pypdfium2 as pdfium
    except ImportError:
        print("  pypdfium2 unavailable — PDF inspection skipped")
        return

    try:
        pdf = pdfium.PdfDocument(pdf_bytes)
    except Exception as e:
        print(f"  ✗ pypdfium2 CANNOT open the PDF: {e}")
        print("    → corrupted/non-standard PDF: explains the silent failure.")
        return

    n_pages = len(pdf)
    print(f"  Pages detected by pypdfium2: {n_pages}")
    if n_pages == 0:
        print("    → 0 pages on the pypdfium2 side: empty or unreadable PDF.")
        return

    sample = min(_OCR_SAMPLE_PAGES, n_pages)
    total_chars = 0
    for i in range(sample):
        page = pdf[i]
        txt = page.get_textpage().get_text_range()
        n_images = sum(1 for obj in page.get_objects() if obj.type == 3)  # 3 = image
        total_chars += len(txt)
        print(
            f"  p{i + 1}: {len(txt):>6d} native chars | {n_images} image object(s)"
        )
    avg = total_chars / sample
    ocr_decision = avg < _OCR_CHAR_THRESHOLD
    print(
        f"  Average {avg:.1f} chars/page over {sample} page(s) "
        f"(threshold={_OCR_CHAR_THRESHOLD}) → needs_ocr = {ocr_decision}"
    )


def inspect_docling_json(json_bytes: bytes) -> None:
    print(f"\n── DoclingDocument JSON ── ({len(json_bytes)} bytes)")
    try:
        doc = json.loads(json_bytes)
    except Exception as e:
        print(f"  ✗ unreadable JSON: {e}")
        return

    pages = doc.get("pages") or {}
    for key in ("texts", "tables", "pictures", "groups", "body"):
        val = doc.get(key)
        n = len(val) if isinstance(val, (list, dict)) else ("present" if val else 0)
        print(f"  {key:<9}: {n}")
    print(f"  pages    : {len(pages)}  ← this is what feeds num_pages")
    if not pages:
        print("    → pages empty: the conversion materialized no page at all.")
    # raw preview if tiny
    if len(json_bytes) < 2000:
        print("\n  Raw content (tiny doc):")
        print("  " + json.dumps(doc, ensure_ascii=False, indent=2).replace("\n", "\n  "))


def _needs_ocr(pdf_bytes: bytes) -> bool:
    """Replicates the pipeline's OCR detection (convert_corpus.needs_ocr)."""
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(pdf_bytes)
        n = min(_OCR_SAMPLE_PAGES, len(pdf))
        if n == 0:
            return False
        total = sum(len(pdf[i].get_textpage().get_text_range()) for i in range(n))
        return (total / n) < _OCR_CHAR_THRESHOLD
    except Exception:
        return True


def reconvert(pdf_bytes: bytes, name: str, url: str | None = None) -> None:
    """Replays the conversion via docling-serve with several configs and compares."""
    from docling.datamodel.base_models import OutputFormat
    from docling.datamodel.pipeline_options import PdfBackend
    from docling.datamodel.service.options import ConvertDocumentsOptions
    from docling_core.types.doc.document import ImageRefMode
    from docling_core.types.io import DocumentStream

    sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
    from _docling_client import PatchedDoclingServiceClient as DoclingServiceClient

    service_url = url or os.environ.get("DOCLING_SERVICE_URL")
    if not service_url:
        sys.exit("[ERROR] no docling-serve URL (neither --url nor DOCLING_SERVICE_URL)")
    api_key = os.environ.get("DOCLING_SERVICE_API_KEY")

    cfg = yaml.safe_load((ROOT / "configs/chunker.yaml").read_text("utf-8"))
    p = cfg["pipeline"]
    auto_ocr = _needs_ocr(pdf_bytes)

    def opts(*, do_ocr: bool, backend: str, table_mode: str,
             do_table: bool = True) -> ConvertDocumentsOptions:
        return ConvertDocumentsOptions(
            do_ocr=do_ocr,
            ocr_preset=p.get("ocr_preset", "rapidocr"),
            do_table_structure=do_table,
            table_mode=table_mode,
            include_images=False,
            image_export_mode=ImageRefMode.PLACEHOLDER,
            do_formula_enrichment=p.get("do_formula_enrichment", True),
            do_code_enrichment=p.get("do_code_enrichment", True),
            pdf_backend=PdfBackend(backend),
            to_formats=[OutputFormat.JSON],
        )

    # (label, options) — recovery candidates (the table stage crashes on this doc)
    matrix = [
        ("WITHOUT table structure (do_table_structure=False)",
         opts(do_ocr=auto_ocr, backend="docling_parse", table_mode="accurate", do_table=False)),
        ("table_mode=fast (TableFormer fast)",
         opts(do_ocr=auto_ocr, backend="docling_parse", table_mode="fast")),
    ]

    print(f"\n══ RECONVERSION via {service_url} ══")
    with DoclingServiceClient(url=service_url, api_key=api_key, job_timeout=1800.0) as dc:
        for label, options in matrix:
            print(f"\n── {label}")
            t0 = time.monotonic()
            try:
                stream = DocumentStream(name=name, stream=BytesIO(pdf_bytes))
                result = dc.convert(source=stream, options=options, raises_on_error=True)
                doc = result.document
                dt = time.monotonic() - t0
                npages = len(doc.pages)
                flag = "✓ RECOVERED" if npages > 0 else "✗ still empty"
                print(
                    f"  {flag} | {npages} pages | texts={len(doc.texts)} "
                    f"tables={len(doc.tables)} pictures={len(doc.pictures)} "
                    f"body.children={len(doc.body.children)} | {dt:.1f}s"
                )
            except Exception as e:
                dt = time.monotonic() - t0
                print(f"  ✗ EXCEPTION after {dt:.1f}s: {type(e).__name__}: {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", dest="file_id", default=None, help="Cleyrop source file_id")
    ap.add_argument("--name", default=None, help="name fragment (manifest)")
    ap.add_argument("--reconvert", action="store_true",
                    help="replay the conversion via docling-serve (several configs)")
    ap.add_argument("--url", default=None,
                    help="docling-serve URL (takes precedence over DOCLING_SERVICE_URL from .env)")
    args = ap.parse_args()

    src_id = resolve_file_id(args.name, args.file_id) if (args.name or args.file_id) else DEFAULT_FILE_ID

    # local conversion manifest → fetch the file_id of the produced Docling JSON
    cm_path = ROOT / "output/convert_manifests" / f"{src_id}.json"
    cm = json.loads(cm_path.read_text("utf-8")) if cm_path.exists() else {}
    json_file_id = cm.get("cleyrop_json_file_id")

    print(f"Doc       : {cm.get('name', '?')}")
    print(f"file_id   : {src_id}")
    print(f"manifest  : status={cm.get('status')} num_pages={cm.get('num_pages')} "
          f"json={cm.get('json_size_bytes')}B elapsed={cm.get('elapsed_s')}s")

    cfg = make_cleyrop_config()
    with CleyropClient(cfg) as cleyrop:
        if cfg.client_secret:
            cleyrop.login_client_credentials()
        pdf_bytes = cleyrop.download_file_bytes(src_id)
        inspect_pdf(pdf_bytes)
        if json_file_id:
            json_bytes = cleyrop.download_file_bytes(json_file_id)
            inspect_docling_json(json_bytes)
        else:
            print("\n── DoclingDocument JSON ── cleyrop_json_file_id missing from manifest")

    if args.reconvert:
        reconvert(pdf_bytes, cm.get("name") or "document.pdf", url=args.url)


if __name__ == "__main__":
    main()
