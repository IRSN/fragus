"""Probe: converts a page range (tables ON/OFF) and measures the content.

Helps understand the TableFormer crash: does it kill the whole doc or just one
batch, and which page range carries the table text.

Usage:
    uv run scripts/diagnostics/probe_convert.py --url http://localhost:5002 --range 1 8
    uv run scripts/diagnostics/probe_convert.py --url http://localhost:5002 --range 5 5 --no-tables
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import tomllib
from io import BytesIO
from pathlib import Path

from cleyrop import CleyropClient, ClientConfig
from docling.datamodel.base_models import OutputFormat
from docling.datamodel.pipeline_options import PdfBackend
from docling.datamodel.service.options import ConvertDocumentsOptions
from docling_core.types.doc.document import ImageRefMode
from docling_core.types.io import DocumentStream

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
from _docling_client import PatchedDoclingServiceClient as DoclingServiceClient

ROOT = Path(__file__).parent.parent.parent
# Cleyrop file_id of the problematic doc — replace with your own
FID = "00000000-0000-0000-0000-000000000000"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--range", nargs=2, type=int, metavar=("START", "END"), required=True)
    ap.add_argument("--no-tables", action="store_true")
    args = ap.parse_args()
    start, end = args.range
    do_table = not args.no_tables

    cache = Path("/tmp/probe_doc.pdf")
    if cache.exists():
        pdf = cache.read_bytes()
    else:
        c = tomllib.load(open(ROOT / "config.toml", "rb"))
        os.environ.setdefault("CLEYROP_DOMAIN", c["cleyrop"]["domain"])
        with CleyropClient(ClientConfig.from_env()) as cl:
            pdf = cl.download_file_bytes(FID)
        cache.write_bytes(pdf)

    options = ConvertDocumentsOptions(
        do_ocr=True, ocr_preset="rapidocr",
        do_table_structure=do_table, table_mode="accurate",
        include_images=False, image_export_mode=ImageRefMode.PLACEHOLDER,
        pdf_backend=PdfBackend("docling_parse"),
        page_range=(start, end),
        to_formats=[OutputFormat.JSON],
    )
    t0 = time.monotonic()
    with DoclingServiceClient(url=args.url, api_key=None, job_timeout=1800.0) as dc:
        result = dc.convert(
            source=DocumentStream(name="probe.pdf", stream=BytesIO(pdf)),
            options=options, raises_on_error=True,
        )
    doc = result.document
    dt = time.monotonic() - t0
    text_chars = sum(len(t.text or "") for t in doc.texts)
    cell_chars = sum(
        len(cell.text or "")
        for t in doc.tables for cell in (t.data.table_cells if t.data else [])
    )
    print(
        f"range=({start},{end}) tables={'on' if do_table else 'off'} | "
        f"pages={len(doc.pages)} keys={sorted(doc.pages)[:6]} | "
        f"texts={len(doc.texts)}({text_chars}c) tables={len(doc.tables)}(cells {cell_chars}c) | {dt:.1f}s"
    )


if __name__ == "__main__":
    main()
