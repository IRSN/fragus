"""Targeted reconversion of a doc whose docling-serve table stage crashes.

Context: on some PDFs, TableFormer (table_mode accurate/fast) crashes
(cv2.resize "!ssize.empty()") → docling-serve returns an empty document with
HTTP 200 → silent num_pages=0. The workaround: do_table_structure=False, which
recovers all pages (tables remain present as blocks, without cell structure
extraction).

This script ONLY performs the convert step (tables off): download PDF →
convert via docling-serve → upload JSON to docling_docs → patch the
convert_manifest. Chunking + embedding are then done via chunk_corpus.py /
embed_corpus.py (1-doc mini-manifest), which are unaffected by the table issue.

Usage:
    uv run scripts/diagnostics/repair_doc.py --id <file_id> --url http://localhost:5002
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

import yaml

# pipeline scripts (convert_corpus, _docling_client) available for import
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

# pipeline helpers (same config/cleyrop/manifest)
from convert_corpus import (
    get_or_create_folder,
    load_manifest,
    make_cleyrop_config,
    resolve_project_id,
    save_local_manifest,
)
from cleyrop import CleyropClient

from docling.datamodel.base_models import OutputFormat
from docling.datamodel.pipeline_options import PdfBackend
from docling.datamodel.service.options import ConvertDocumentsOptions
from docling_core.types.doc.document import ImageRefMode
from docling_core.types.io import DocumentStream

from _docling_client import PatchedDoclingServiceClient as DoclingServiceClient

ROOT = Path(__file__).parent.parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True, help="Cleyrop file_id of the doc to repair")
    ap.add_argument("--url", required=True, help="docling-serve URL (e.g. http://localhost:5002)")
    ap.add_argument("--manifest", default="output/manifest_fragus_clean.dedup.json")
    ap.add_argument("--config", default="configs/chunker.yaml")
    ap.add_argument("--delete-orphan", action="store_true",
                    help="delete the old empty JSON from Cleyrop after reconversion")
    args = ap.parse_args()

    file_id = args.id
    files = load_manifest(ROOT / args.manifest)
    entry = next((f for f in files if f["id"] == file_id), None)
    if entry is None:
        sys.exit(f"[ERROR] {file_id} missing from manifest")
    name = entry["name"]

    cfg = yaml.safe_load((ROOT / args.config).read_text("utf-8"))
    p = cfg["pipeline"]
    docling_docs_folder = cfg["convert"]["docling_docs_folder"]
    manifests_dir = ROOT / "output" / "convert_manifests"

    # old cleyrop_json_file_id (empty orphan to optionally clean up)
    old_manifest = json.loads((manifests_dir / f"{file_id}.json").read_text("utf-8")) \
        if (manifests_dir / f"{file_id}.json").exists() else {}
    old_json_file_id = old_manifest.get("cleyrop_json_file_id")

    options = ConvertDocumentsOptions(
        do_ocr=p.get("do_ocr", True),
        ocr_preset=p.get("ocr_preset", "rapidocr"),
        do_table_structure=False,                    # ← the TableFormer crash workaround
        table_mode=p.get("table_mode", "accurate"),
        include_images=False,
        image_export_mode=ImageRefMode.PLACEHOLDER,
        do_formula_enrichment=p.get("do_formula_enrichment", True),
        do_code_enrichment=p.get("do_code_enrichment", True),
        pdf_backend=PdfBackend(p.get("pdf_backend", "docling_parse")),
        to_formats=[OutputFormat.JSON],
    )

    cleyrop_cfg = make_cleyrop_config()
    project_id = resolve_project_id(cleyrop_cfg)

    print(f"Repairing: {name} ({file_id})")
    print(f"  docling-serve: {args.url} | do_table_structure=False")

    with CleyropClient(cleyrop_cfg) as cleyrop:
        if cleyrop_cfg.client_secret:
            cleyrop.login_client_credentials()
        folder_id = get_or_create_folder(cleyrop, project_id, docling_docs_folder)

        pdf_bytes = cleyrop.download_file_bytes(file_id)
        print(f"  PDF downloaded: {len(pdf_bytes) / 1024:.0f} KB")

        t0 = time.monotonic()
        with DoclingServiceClient(url=args.url, api_key=None, job_timeout=1800.0) as dc:
            result = dc.convert(
                source=DocumentStream(name=name, stream=BytesIO(pdf_bytes)),
                options=options,
                raises_on_error=True,
            )
        doc = result.document
        elapsed = time.monotonic() - t0
        num_pages = len(doc.pages)
        print(f"  Converted: {num_pages} pages | texts={len(doc.texts)} "
              f"tables={len(doc.tables)} | {elapsed:.1f}s")

        if num_pages == 0:
            sys.exit("[FAILURE] still 0 pages — writing nothing, keeping the old manifest")

        json_bytes = doc.model_dump_json().encode("utf-8")
        upload_resp = cleyrop.upload_file(
            BytesIO(json_bytes),
            project_id=project_id,
            folder_id=folder_id,
            filename=f"{file_id}.json",
        )
        print(f"  JSON uploaded: {len(json_bytes) / 1024:.0f} KB (cleyrop id={upload_resp.id})")

        save_local_manifest(file_id, {
            "file_id": file_id,
            "name": name,
            "path": entry["path"],
            "corpus_minimal": entry.get("corpus_minimal", False),
            "status": "done",
            "converted_at": datetime.now().isoformat(),
            "num_pages": num_pages,
            "json_size_bytes": len(json_bytes),
            "elapsed_s": round(elapsed, 1),
            "cleyrop_json_file_id": str(upload_resp.id),
            "note": "reconverted with do_table_structure=False (TableFormer cv2.resize crash)",
        }, manifests_dir)
        print(f"  convert_manifest patched: {manifests_dir / (file_id + '.json')}")

        if args.delete_orphan and old_json_file_id and old_json_file_id != str(upload_resp.id):
            cleyrop.delete_file(old_json_file_id)
            print(f"  Old empty JSON deleted from Cleyrop: {old_json_file_id}")
        elif old_json_file_id:
            print(f"  (old empty orphan JSON kept: {old_json_file_id} "
                  f"— rerun with --delete-orphan to remove it)")


if __name__ == "__main__":
    main()
