"""Chunking pipeline for DoclingDocument JSON → JSONL chunks.

Reads the DoclingDocument JSON files already parsed and stored in Cleyrop (by
convert_corpus.py), applies the HybridChunker locally, and uploads the JSONL
chunks to Cleyrop.

Usage:
    uv run scripts/pipeline/chunk_corpus.py [OPTIONS]

    --manifest        output/manifest_fragus_clean.dedup.json  (default)
    --config          configs/chunker.yaml  (default)
    --sample N        process the first N files not yet processed
    --workers N       parallel workers (default: 1)
    --force           re-chunk even if already done
    --status          show the state without processing
    --corpus-minimal  process only the files flagged corpus_minimal
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import threading
import time
import tomllib
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

import semchunk
from cleyrop import CleyropClient, ClientConfig
from docling.chunking import HybridChunker
from docling_core.types.doc import DoclingDocument
from transformers import AutoTokenizer

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


# ─────────────────────────── local manifests ─────────────────────────────────


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


def is_already_chunked(file_id: str, manifests_dir: Path) -> bool:
    return load_local_manifest(file_id, manifests_dir).get("status") == "done"


# ─────────────────────────── chunk formatting ────────────────────────────────


def _page_numbers(chunk) -> list[int]:
    return sorted({
        prov.page_no
        for item in (chunk.meta.doc_items or [])
        for prov in (item.prov or [])
    })


def build_chunk_text(entry: dict, chunk) -> str:
    # The context header is intentionally in French: the corpus and the
    # embedded chunks are French.
    headings = chunk.meta.headings or []
    lines = [
        "CONTEXTE DU CHUNK:",
        f"Document: {entry['name']}",
        f"Chemin: {entry['path']}",
    ]
    if headings:
        lines.append(f"Chapitre: {' > '.join(headings)}")
    captions = chunk.meta.captions or []
    if captions:
        lines.append(f"Légende: {captions[0]}")
    lines.extend(["", "CHUNK:", "", chunk.text or ""])
    return "\n".join(lines)


def make_chunk_dict(entry: dict, chunk, idx: int, num_tokens: int = 0) -> dict:
    file_id = entry["id"]
    return {
        "chunk_id": f"{file_id}_{idx:04d}",
        "text": build_chunk_text(entry, chunk),
        "file_id": file_id,
        "name": entry["name"],
        "path": entry["path"],
        "corpus_minimal": entry.get("corpus_minimal", False),
        "page_numbers": _page_numbers(chunk),
        "docling_meta": {
            "headings": chunk.meta.headings or [],
            "captions": chunk.meta.captions or [],
            "num_tokens": num_tokens,
        },
    }


def enforce_hard_cap(raw_chunks, tokenizer, hard_cap: int):
    """Re-split any chunk whose text exceeds `hard_cap` tokens via semchunk.

    Workaround for docling-core bug #119: HybridChunker can let through
    chunks > max_tokens when semchunk finds no usable separator
    (dense table, single-paragraph line). Without this safeguard, bge-m3
    would silently truncate at 8192 tokens during embedding.

    Returns `(fixed_chunks, token_count_per_chunk, n_split_events)`.
    """
    splitter = semchunk.chunkerify(tokenizer, chunk_size=hard_cap)
    safe = []
    token_counts = []
    n_split = 0
    for c in raw_chunks:
        text = c.text or ""
        n = len(tokenizer.encode(text, add_special_tokens=False))
        if n <= hard_cap:
            safe.append(c)
            token_counts.append(n)
            continue
        n_split += 1
        for sub_text in splitter(text):
            sub_chunk = c.model_copy()
            sub_chunk.text = sub_text
            safe.append(sub_chunk)
            token_counts.append(len(tokenizer.encode(sub_text, add_special_tokens=False)))
    return safe, token_counts, n_split


# ─────────────────────────── per-thread clients ──────────────────────────────

_thread_local = threading.local()


def _get_cleyrop_client(cleyrop_cfg: ClientConfig) -> CleyropClient:
    if not hasattr(_thread_local, "cleyrop"):
        client = CleyropClient(cleyrop_cfg)
        if cleyrop_cfg.client_secret:
            client.login_client_credentials()
        client.__enter__()
        _thread_local.cleyrop = client
    return _thread_local.cleyrop


# ─────────────────────────── per-document pipeline ───────────────────────────


def process_one(
    entry: dict,
    cleyrop: CleyropClient,
    chunker: HybridChunker,
    tokenizer,
    hard_cap_tokens: int,
    cfg: dict,
    project_id: str,
    chunks_folder_id: str,
    chunk_manifests_dir: Path,
    convert_manifests_dir: Path,
    force: bool,
) -> tuple[str, float]:
    file_id = entry["id"]
    name = entry["name"]
    min_len = cfg["processing"]["min_chunk_length"]

    if not force and is_already_chunked(file_id, chunk_manifests_dir):
        logger.debug(f"Already chunked, skip: {name}")
        return "skipped", 0.0

    t0 = time.monotonic()
    try:
        conv_m = load_local_manifest(file_id, convert_manifests_dir)
        json_file_id = conv_m.get("cleyrop_json_file_id")
        if not json_file_id:
            logger.warning(f"No converted JSON for: {name} — run convert_corpus.py first")
            return "skipped", 0.0

        logger.info(f"Downloading JSON: {name}")
        json_bytes = cleyrop.download_file_bytes(json_file_id)
        doc = DoclingDocument.model_validate_json(json_bytes)

        logger.info(f"Local chunking (HybridChunker): {name}")
        raw_chunks = list(chunker.chunk(doc))

        safe_chunks, token_counts, n_split = enforce_hard_cap(raw_chunks, tokenizer, hard_cap_tokens)
        if n_split:
            logger.info(f"  ↳ safeguard: {n_split} chunk(s) > {hard_cap_tokens} tok re-split")

        chunks = []
        for c, n_tok in zip(safe_chunks, token_counts):
            if len((c.text or "").strip()) < min_len:
                continue
            chunks.append(make_chunk_dict(entry, c, len(chunks), num_tokens=n_tok))

        elapsed = time.monotonic() - t0
        num_pages = max((max(c["page_numbers"]) for c in chunks if c["page_numbers"]), default=0)
        max_tok = max((c["docling_meta"]["num_tokens"] for c in chunks), default=0)
        logger.info(f"  → {len(chunks)} chunks | {num_pages}p | max {max_tok} tok | {elapsed:.1f}s")

        jsonl_bytes = "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks).encode("utf-8")
        upload_resp = cleyrop.upload_file(
            BytesIO(jsonl_bytes),
            project_id=project_id,
            folder_id=chunks_folder_id,
            filename=f"{file_id}.jsonl",
        )

        save_local_manifest(file_id, {
            "file_id": file_id,
            "name": name,
            "path": entry["path"],
            "corpus_minimal": entry.get("corpus_minimal", False),
            "status": "done",
            "chunked_at": datetime.now().isoformat(),
            "num_chunks": len(chunks),
            "num_pages": num_pages,
            "elapsed_s": round(elapsed, 1),
            "cleyrop_jsonl_file_id": str(upload_resp.id),
        }, chunk_manifests_dir)
        return "done", elapsed

    except Exception as e:
        elapsed = time.monotonic() - t0
        logger.error(f"Error on {name}: {e}")
        save_local_manifest(file_id, {
            "file_id": file_id, "name": name, "path": entry["path"],
            "status": "error", "error": str(e), "chunked_at": datetime.now().isoformat(),
        }, chunk_manifests_dir)
        return "error", elapsed


def _process_entry(
    entry: dict,
    i: int,
    total: int,
    cleyrop_cfg: ClientConfig,
    chunker: HybridChunker,
    tokenizer,
    hard_cap_tokens: int,
    cfg: dict,
    project_id: str,
    chunks_folder_id: str,
    chunk_manifests_dir: Path,
    convert_manifests_dir: Path,
    force: bool,
) -> tuple[str, float]:
    cleyrop = _get_cleyrop_client(cleyrop_cfg)
    logger.info(f"[{i}/{total}] {entry['name']}")
    return process_one(
        entry=entry,
        cleyrop=cleyrop,
        chunker=chunker,
        tokenizer=tokenizer,
        hard_cap_tokens=hard_cap_tokens,
        cfg=cfg,
        project_id=project_id,
        chunks_folder_id=chunks_folder_id,
        chunk_manifests_dir=chunk_manifests_dir,
        convert_manifests_dir=convert_manifests_dir,
        force=force,
    )


# ─────────────────────────── status ──────────────────────────────────────────


def print_status(files: list[dict], chunk_manifests_dir: Path, convert_manifests_dir: Path) -> None:
    counts: dict[str, int] = {"done": 0, "error": 0, "pending": 0, "not_converted": 0}
    total_elapsed = 0.0
    total_pages = 0
    errors: list[tuple[str, str]] = []

    for entry in files:
        cm = load_local_manifest(entry["id"], chunk_manifests_dir)
        status = cm.get("status", "pending")
        if status == "pending":
            conv_m = load_local_manifest(entry["id"], convert_manifests_dir)
            if not conv_m.get("cleyrop_json_file_id"):
                status = "not_converted"
        counts[status if status in counts else "pending"] += 1
        if status == "done":
            total_elapsed += cm.get("elapsed_s", 0)
            total_pages += cm.get("num_pages", 0)
        if status == "error":
            errors.append((entry.get("path", entry["name"]), cm.get("error", "")))

    total = len(files)
    print(f"\nChunking status ({total} PDF files):")
    print(f"  ✓ done          : {counts['done']}")
    print(f"  ⏳ pending       : {counts['pending']}")
    print(f"  ⚠ not_converted : {counts['not_converted']}")
    print(f"  ✗ error         : {counts['error']}")
    if counts["done"] > 0:
        print(f"\n  Total time: {total_elapsed:.0f}s | Avg/doc: {total_elapsed/counts['done']:.1f}s")
    if errors:
        for path, msg in sorted(errors):
            print(f"  ✗ {path}: {msg}")


# ─────────────────────────── main ────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunking pipeline (local HybridChunker)")
    parser.add_argument("--manifest", default="output/manifest_fragus_clean.dedup.json")
    parser.add_argument("--config", default="configs/chunker.yaml")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--corpus-minimal", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).parent.parent.parent
    cfg = load_chunker_config(root / args.config)
    chunk_manifests_dir = root / "output" / "chunk_manifests"
    convert_manifests_dir = root / "output" / "convert_manifests"

    files = filter_files(load_manifest(root / args.manifest))
    if args.corpus_minimal:
        files = [f for f in files if f.get("corpus_minimal")]
        logger.info(f"--corpus-minimal: {len(files)} PDFs targeted")

    if args.status:
        print_status(files, chunk_manifests_dir, convert_manifests_dir)
        return

    if args.sample:
        candidates = files if args.force else [
            f for f in files if not is_already_chunked(f["id"], chunk_manifests_dir)
        ]
        files = candidates[: args.sample]
        logger.info(f"Sample mode: {len(files)} files to process")

    chunker_cfg = cfg["chunker"]
    tokenizer_name = chunker_cfg.get("tokenizer", "BAAI/bge-m3")
    max_tokens = chunker_cfg.get("max_tokens", 512)
    hard_cap_tokens = chunker_cfg.get("hard_cap_tokens", 1024)
    chunker = HybridChunker(
        tokenizer=tokenizer_name,
        max_tokens=max_tokens,
        merge_peers=chunker_cfg.get("merge_peers", True),
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    logger.info(f"Chunker bge-m3: max_tokens={max_tokens}, hard_cap={hard_cap_tokens}")

    cleyrop_cfg = make_cleyrop_config()
    project_id = resolve_project_id(cleyrop_cfg)
    chunks_folder_name = cfg["storage"]["chunks_folder"]
    chunk_timeout = cfg["processing"]["chunk_timeout"]

    with CleyropClient(cleyrop_cfg) as cleyrop_main:
        if cleyrop_cfg.client_secret:
            cleyrop_main.login_client_credentials()
        chunks_folder_id = get_or_create_folder(cleyrop_main, project_id, chunks_folder_name)

    stats: dict[str, int] = {"done": 0, "skipped": 0, "error": 0}
    total_elapsed = 0.0
    stats_lock = threading.Lock()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _process_entry,
                entry, i, len(files),
                cleyrop_cfg, chunker, tokenizer, hard_cap_tokens, cfg,
                project_id, chunks_folder_id,
                chunk_manifests_dir, convert_manifests_dir,
                args.force,
            ): entry
            for i, entry in enumerate(files, 1)
        }
        for future in concurrent.futures.as_completed(futures):
            entry = futures[future]
            try:
                result, elapsed = future.result(timeout=chunk_timeout)
            except concurrent.futures.TimeoutError:
                logger.warning(f"Timeout ({chunk_timeout}s) for: {entry['name']}")
                save_local_manifest(entry["id"], {
                    "file_id": entry["id"], "name": entry["name"], "path": entry["path"],
                    "status": "error", "error": "timeout", "chunked_at": datetime.now().isoformat(),
                }, chunk_manifests_dir)
                result, elapsed = "error", float(chunk_timeout)
            with stats_lock:
                stats[result] = stats.get(result, 0) + 1
                if result == "done":
                    total_elapsed += elapsed

    avg = (total_elapsed / stats["done"]) if stats["done"] else 0
    print(
        f"\nResults: {stats['done']} processed ({total_elapsed:.0f}s, avg {avg:.1f}s/doc), "
        f"{stats.get('skipped', 0)} skipped, "
        f"{stats['error']} errors"
    )


if __name__ == "__main__":
    main()
