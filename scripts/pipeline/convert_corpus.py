"""Conversion pipeline PDF → DoclingDocument JSON.

Downloads the PDFs from Cleyrop, delegates the conversion to docling-serve
(RapidOCR OCR, fast tables, no images), and stores the resulting
DoclingDocument JSON in a dedicated folder on Cleyrop ("docling_docs/").

The chunking step is separate and will be done later from these JSON files.

Usage:
    uv run scripts/pipeline/convert_corpus.py [OPTIONS]

    --manifest        output/manifest_fragus_clean.dedup.json  (default)
    --config          configs/chunker.yaml  (default)
    --sample N        process the first N files not yet converted
    --workers N       parallel workers (default: 1)
    --force           re-convert even if already done
    --status          show the state without processing
    --corpus-minimal  process only the files flagged corpus_minimal
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import sys
import threading
import time
import tomllib
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from cleyrop import CleyropClient, ClientConfig
from docling.datamodel.base_models import OutputFormat
from docling.datamodel.pipeline_options import PdfBackend
from docling.datamodel.service.options import ConvertDocumentsOptions
from docling_core.types.doc.document import ImageRefMode
from docling_core.types.io import DocumentStream
from _docling_client import PatchedDoclingServiceClient as DoclingServiceClient

load_dotenv(override=True)

_config_path = Path(__file__).parent.parent.parent / "config.toml"
with open(_config_path, "rb") as f:
    _config = tomllib.load(f)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────── config / cleyrop ────────────────────────────────


def load_chunker_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def resolve_project_id(cfg: ClientConfig) -> str:
    client = CleyropClient(cfg)
    if cfg.client_secret:
        client.login_client_credentials()
    with client:
        if project_id := os.environ.get("PROJECT_ID"):
            return project_id
        if project_id := _config["project"].get("id"):
            return project_id
        project_slug = os.environ["PROJECT_SLUG"]
        return str(client.get_project_by_slug(project_slug).id)


def get_or_create_folder(client: CleyropClient, project_id: str, folder_name: str) -> str:
    contents = client.get_project_contents(project_id)
    for folder in contents.folders:
        if folder.name == folder_name:
            return str(folder.id)
    folder = client.create_folder(folder_name, project_id=project_id)
    logger.info(f"Folder created in Cleyrop: {folder_name} (id={folder.id})")
    return str(folder.id)


# ─────────────────────────── local manifest ──────────────────────────────────


def load_manifest(manifest_path: Path) -> list[dict]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)["files"]


def filter_files(files: list[dict]) -> list[dict]:
    return [f for f in files if f.get("mime_type") == "application/pdf"]


def load_local_manifest(file_id: str, manifests_dir: Path) -> dict:
    path = manifests_dir / f"{file_id}.json"
    return json.loads(path.read_text("utf-8")) if path.exists() else {}


def save_local_manifest(file_id: str, data: dict, manifests_dir: Path) -> None:
    manifests_dir.mkdir(parents=True, exist_ok=True)
    (manifests_dir / f"{file_id}.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), "utf-8"
    )


def is_already_converted(file_id: str, manifests_dir: Path) -> bool:
    return load_local_manifest(file_id, manifests_dir).get("status") == "done"


# ─────────────────────────── OCR-needed detection ────────────────────────────

# Threshold: if fewer than N native characters on average over the first pages,
# the PDF is considered scanned and requires OCR.
_OCR_CHAR_THRESHOLD = 100
_OCR_SAMPLE_PAGES = 3

# pypdfium2 is not thread-safe — concurrent calls → double free / segfault.
_pdfium_lock = threading.Lock()


def needs_ocr(pdf_bytes: bytes) -> bool:
    """Detects whether a PDF requires OCR by counting native characters.

    Samples the first pages via pypdfium2 (already available in the venv).
    Returns True if the PDF is probably scanned (little native text).
    """
    try:
        import pypdfium2 as pdfium
        with _pdfium_lock:
            pdf = pdfium.PdfDocument(pdf_bytes)
            n = min(_OCR_SAMPLE_PAGES, len(pdf))
            if n == 0:
                return False
            total_chars = sum(
                len(pdf[i].get_textpage().get_text_range())
                for i in range(n)
            )
        avg_chars = total_chars / n
        return avg_chars < _OCR_CHAR_THRESHOLD
    except Exception:
        return True  # on error, enable OCR to be safe


# ─────────────────────────── conversion options ──────────────────────────────


def build_convert_options(
    cfg: dict, do_ocr: bool
) -> ConvertDocumentsOptions:
    p = cfg.get("pipeline", {})
    timeout = cfg["convert"]["convert_timeout"]
    return ConvertDocumentsOptions(
        do_ocr=do_ocr,
        ocr_preset=p.get("ocr_preset", "rapidocr"),
        do_table_structure=p.get("do_table_structure", True),
        table_mode=p.get("table_mode", "accurate"),
        include_images=False,
        image_export_mode=ImageRefMode.PLACEHOLDER,  # avoids base64-encoded PDF pages
        do_formula_enrichment=p.get("do_formula_enrichment", True),
        do_code_enrichment=p.get("do_code_enrichment", True),
        pdf_backend=PdfBackend(p.get("pdf_backend", "docling_parse")),
        document_timeout=float(timeout),
        to_formats=[OutputFormat.JSON],
    )


# ─────────────────────────── per-thread clients ──────────────────────────────

_thread_local = threading.local()


def _get_thread_clients(
    cleyrop_cfg: ClientConfig, service_url: str, api_key: str | None
) -> tuple[CleyropClient, DoclingServiceClient]:
    if not hasattr(_thread_local, "cleyrop"):
        client = CleyropClient(cleyrop_cfg)
        if cleyrop_cfg.client_secret:
            client.login_client_credentials()
        client.__enter__()
        _thread_local.cleyrop = client

        docling = DoclingServiceClient(
            url=service_url,
            api_key=api_key,
            job_timeout=_DOCLING_JOB_TIMEOUT,
        )
        docling.__enter__()
        _thread_local.docling = docling

    return _thread_local.cleyrop, _thread_local.docling


_DOCLING_JOB_TIMEOUT: float = 300.0  # overridden in main() from configs/chunker.yaml


# ─────────────────────────── per-document pipeline ───────────────────────────


def process_one(
    entry: dict,
    cleyrop: CleyropClient,
    docling: DoclingServiceClient,
    cfg: dict,
    project_id: str,
    docling_docs_folder_id: str,
    manifests_dir: Path,
    force: bool,
) -> tuple[str, float]:
    file_id = entry["id"]
    name = entry["name"]

    if not force and is_already_converted(file_id, manifests_dir):
        logger.debug(f"Already converted, skip: {name}")
        return "skipped", 0.0

    t0 = time.monotonic()
    try:
        logger.info(f"Downloading: {name}")
        pdf_bytes = cleyrop.download_file_bytes(file_id)

        ocr = needs_ocr(pdf_bytes)
        convert_options = build_convert_options(cfg, do_ocr=ocr)
        logger.info(f"Converting via docling-serve (OCR={'on' if ocr else 'off'}): {name}")
        stream = DocumentStream(name=name, stream=BytesIO(pdf_bytes))
        result = docling.convert(source=stream, options=convert_options)

        doc = result.document
        json_bytes = doc.model_dump_json().encode("utf-8")

        upload_resp = cleyrop.upload_file(
            BytesIO(json_bytes),
            project_id=project_id,
            folder_id=docling_docs_folder_id,
            filename=f"{file_id}.json",
        )

        elapsed = time.monotonic() - t0
        num_pages = len(doc.pages)
        secs_per_page = (elapsed / num_pages) if num_pages else 0
        json_kb = len(json_bytes) / 1024
        logger.info(
            f"  → {num_pages}p | {json_kb:.0f} KB JSON | "
            f"{elapsed:.1f}s | {secs_per_page:.1f}s/page"
        )

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
        }, manifests_dir)
        return "done", elapsed

    except Exception as e:
        elapsed = time.monotonic() - t0
        logger.error(f"Error on {name}: {e}")
        save_local_manifest(file_id, {
            "file_id": file_id,
            "name": name,
            "path": entry["path"],
            "status": "error",
            "error": str(e),
            "converted_at": datetime.now().isoformat(),
        }, manifests_dir)
        return "error", elapsed


def _process_entry(
    entry: dict,
    i: int,
    total: int,
    cleyrop_cfg: ClientConfig,
    service_url: str,
    api_key: str | None,
    cfg: dict,
    project_id: str,
    docling_docs_folder_id: str,
    manifests_dir: Path,
    force: bool,
) -> tuple[str, float]:
    cleyrop, docling = _get_thread_clients(cleyrop_cfg, service_url, api_key)
    logger.info(f"[{i}/{total}] {entry['name']}")
    return process_one(
        entry=entry,
        cleyrop=cleyrop,
        docling=docling,
        cfg=cfg,
        project_id=project_id,
        docling_docs_folder_id=docling_docs_folder_id,
        manifests_dir=manifests_dir,
        force=force,
    )


# ─────────────────────────── status ──────────────────────────────────────────


def print_status(files: list[dict], manifests_dir: Path) -> None:
    counts: dict[str, int] = {"done": 0, "error": 0, "timeout": 0, "pending": 0}
    errors: list[tuple[str, str, str]] = []
    total_elapsed = 0.0
    total_pages = 0
    total_json_bytes = 0
    for entry in files:
        m = load_local_manifest(entry["id"], manifests_dir)
        status = m.get("status", "pending")
        counts[status if status in counts else "pending"] += 1
        if status == "done":
            total_elapsed += m.get("elapsed_s", 0)
            total_pages += m.get("num_pages", 0)
            total_json_bytes += m.get("json_size_bytes", 0)
        if status in ("error", "timeout"):
            errors.append((entry.get("path", entry["name"]), entry["name"], m.get("error", status)))

    total = len(files)
    print(f"\nConversion status ({total} PDF files):")
    print(f"  ✓ done    : {counts['done']}")
    print(f"  ⏳ pending : {counts['pending']}")
    print(f"  ✗ error   : {counts['error']}")
    print(f"  ⏱ timeout : {counts['timeout']}")
    if counts["done"] > 0:
        avg_per_doc = total_elapsed / counts["done"]
        avg_per_page = (total_elapsed / total_pages) if total_pages else 0
        print(f"\n  Total time   : {total_elapsed:.0f}s")
        print(f"  Avg/doc      : {avg_per_doc:.1f}s")
        print(f"  Avg/page     : {avg_per_page:.1f}s")
        print(f"  Total JSON   : {total_json_bytes / 1024 / 1024:.1f} MB")
    if counts["done"] > 0 and counts["pending"] > 0:
        avg = total_elapsed / counts["done"]
        remaining_s = counts["pending"] * avg
        print(f"\n  Remaining estimate ( 1 worker) : {remaining_s/3600:.1f}h")
        for w in [4, 8, 16]:
            print(f"  Remaining estimate ({w:2d} workers): {remaining_s/3600/w:.1f}h")
    if errors:
        errors_file = manifests_dir.parent / "conversion_errors.txt"
        lines = [f"{path}: {msg}\n" for path, _name, msg in sorted(errors)]
        errors_file.write_text("".join(lines), encoding="utf-8")
        print(f"\n  Errors/timeouts: see {errors_file}")


# ─────────────────────────── main ────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Conversion pipeline PDF → DoclingDocument JSON")
    parser.add_argument("--manifest", default="output/manifest_fragus_clean.dedup.json")
    parser.add_argument("--config", default="configs/chunker.yaml")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument(
        "--corpus-minimal",
        action="store_true",
        help="Process only the files flagged corpus_minimal",
    )
    args = parser.parse_args()

    root = Path(__file__).parent.parent.parent
    cfg = load_chunker_config(root / args.config)
    manifests_dir = root / "output" / "convert_manifests"

    files = filter_files(load_manifest(root / args.manifest))

    if args.corpus_minimal:
        files = [f for f in files if f.get("corpus_minimal")]
        logger.info(f"--corpus-minimal: {len(files)} PDFs targeted")

    if args.status:
        print_status(files, manifests_dir)
        return

    service_url = os.environ.get("DOCLING_SERVICE_URL")
    if not service_url:
        logger.error("DOCLING_SERVICE_URL not set in .env")
        sys.exit(1)

    if args.sample:
        pending = [f for f in files if not is_already_converted(f["id"], manifests_dir)]
        files = pending[: args.sample]
        logger.info(f"Sample mode: {len(files)} files to process")

    convert_timeout = cfg["convert"]["convert_timeout"]
    docling_docs_folder = cfg["convert"]["docling_docs_folder"]
    api_key = os.environ.get("DOCLING_SERVICE_API_KEY")

    global _DOCLING_JOB_TIMEOUT
    _DOCLING_JOB_TIMEOUT = float(convert_timeout)

    cleyrop_cfg = make_cleyrop_config()
    project_id = resolve_project_id(cleyrop_cfg)

    with CleyropClient(cleyrop_cfg) as cleyrop_main:
        if cleyrop_cfg.client_secret:
            cleyrop_main.login_client_credentials()
        docling_docs_folder_id = get_or_create_folder(cleyrop_main, project_id, docling_docs_folder)

    logger.info(
        f"OCR=auto (per-document detection) | "
        f"table_mode={cfg['pipeline'].get('table_mode', 'fast')} | "
        f"timeout={convert_timeout}s | workers={args.workers}"
    )

    stats: dict[str, int] = {"done": 0, "skipped": 0, "error": 0, "timeout": 0}
    total_elapsed = 0.0
    stats_lock = threading.Lock()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _process_entry,
                entry, i, len(files),
                cleyrop_cfg, service_url, api_key,
                cfg, project_id, docling_docs_folder_id,
                manifests_dir, args.force,
            ): entry
            for i, entry in enumerate(files, 1)
        }
        for future in concurrent.futures.as_completed(futures):
            entry = futures[future]
            try:
                result, elapsed = future.result(timeout=convert_timeout)
            except concurrent.futures.TimeoutError:
                logger.warning(f"Timeout ({convert_timeout}s) for: {entry['name']}")
                save_local_manifest(entry["id"], {
                    "file_id": entry["id"],
                    "name": entry["name"],
                    "path": entry["path"],
                    "status": "timeout",
                    "converted_at": datetime.now().isoformat(),
                }, manifests_dir)
                result, elapsed = "timeout", float(convert_timeout)
            with stats_lock:
                stats[result] = stats.get(result, 0) + 1
                if result == "done":
                    total_elapsed += elapsed

    # Write the errors file
    errors = []
    for entry in files:
        m = load_local_manifest(entry["id"], manifests_dir)
        if m.get("status") in ("error", "timeout"):
            errors.append(f"{entry.get('path', entry['name'])}: {m.get('error', m.get('status'))}\n")
    if errors:
        errors_file = root / "output" / "conversion_errors.txt"
        errors_file.parent.mkdir(parents=True, exist_ok=True)
        errors_file.write_text("".join(sorted(errors)), encoding="utf-8")
        logger.info(f"Errors written to {errors_file}")

    avg = (total_elapsed / stats["done"]) if stats["done"] else 0
    print(
        f"\nResults: {stats['done']} converted ({total_elapsed:.0f}s, avg {avg:.1f}s/doc), "
        f"{stats.get('skipped', 0)} skipped, "
        f"{stats['error']} errors, "
        f"{stats['timeout']} timeouts"
    )


if __name__ == "__main__":
    main()
