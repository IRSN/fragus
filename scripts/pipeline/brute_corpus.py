"""Raw pipeline: PDF → native pypdf text → token windows → Milvus.

Variant of the ingestion chain without docling and without OCR. For each PDF
of the corpus (same manifest as the docling chain):
  1. download the bytes from Cleyrop;
  2. extract the native text page by page via pypdf (no OCR);
  3. split into bge-m3 token windows (500 tokens, 100 overlap);
  4. embed with bge-m3 via the AI-gen proxy;
  5. upsert into a dedicated Milvus collection (config.toml → [milvus].collection_brut).

Scanned PDFs without a native text layer produce 0 chunks: this is expected
("empty" status), and it is precisely what this "raw" collection highlights
compared to the docling+OCR chain.

Usage:
    uv run scripts/pipeline/brute_corpus.py [OPTIONS]

    --manifest        output/manifest_fragus_clean.dedup.json  (default)
    --sample N        process the first N files not yet processed
    --force           re-process even if already done
    --status          show the state without processing
    --corpus-minimal  process only the files flagged corpus_minimal
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import tomllib
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from openai import OpenAI
from pymilvus import DataType, MilvusClient
from pypdf import PdfReader
from transformers import AutoTokenizer

from cleyrop import CleyropClient, ClientConfig

load_dotenv(override=True)

_config_path = Path(__file__).parent.parent.parent / "config.toml"
with open(_config_path, "rb") as f:
    _cfg_root = tomllib.load(f)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────── config / cleyrop ────────────────────────────────


def make_cleyrop_config() -> ClientConfig:
    domain = os.environ.get("CLEYROP_DOMAIN") or _cfg_root["cleyrop"].get("domain")
    if domain:
        os.environ.setdefault("CLEYROP_DOMAIN", domain)
        return ClientConfig.from_env()
    return ClientConfig.internal(
        client_id=os.environ.get("CLEYROP_CLIENT_ID", "cleyrop-cli"),
        client_secret=os.environ.get("CLEYROP_CLIENT_SECRET"),
        token=os.environ.get("CLEYROP_TOKEN"),
    )


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


def is_already_done(file_id: str, manifests_dir: Path) -> bool:
    return load_local_manifest(file_id, manifests_dir).get("status") in ("done", "empty")


# ─────────────────────────── embedding proxy resolution ──────────────────────


def _resolve_proxy_models(proxy_prefix: str) -> dict[str, dict]:
    url = f"{proxy_prefix}/emb/models"
    req = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except (URLError, HTTPError, TimeoutError) as exc:
        logger.warning(f"Could not reach the proxy: {exc}")
        return {}

    models = {}
    for item in data if isinstance(data, list) else []:
        name = item.get("model_name")
        base_url = item.get("base_url")
        if not name or not base_url:
            continue
        endpoint = base_url.rstrip("/") + "/v1"
        models[str(name)] = {"name": str(name), "endpoint": endpoint}
    return models


def build_openai_client(proxy_prefix: str, model_key: str) -> tuple[OpenAI, str]:
    models = _resolve_proxy_models(proxy_prefix)
    if model_key in models:
        m = models[model_key]
        logger.info(f"Embedding model resolved: {model_key} → {m['endpoint']}")
        return OpenAI(base_url=m["endpoint"], api_key="dummy"), m["name"]
    raise RuntimeError(
        f"Model '{model_key}' not found in the proxy's /emb/models. "
        f"Available models: {sorted(models)}"
    )


def embed_texts(oai: OpenAI, model_name: str, texts: list[str]) -> list[list[float]]:
    resp = oai.embeddings.create(model=model_name, input=texts, encoding_format="float")
    return [e.embedding for e in resp.data]


# ─────────────────────────── Milvus ──────────────────────────────────────────


def ensure_collection(client: MilvusClient, collection: str, dim: int) -> None:
    if collection in client.list_collections():
        logger.info(f"Existing Milvus collection: {collection}")
        return

    logger.info(f"Creating Milvus collection: {collection} (dim={dim})")
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("chunk_id", DataType.VARCHAR, max_length=512, is_primary=True)
    schema.add_field("file_id", DataType.VARCHAR, max_length=256)
    schema.add_field("name", DataType.VARCHAR, max_length=512)
    schema.add_field("path", DataType.VARCHAR, max_length=1024)
    schema.add_field("corpus_minimal", DataType.BOOL)
    schema.add_field("text", DataType.VARCHAR, max_length=65535)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 128},
    )
    client.create_collection(
        collection_name=collection, schema=schema, index_params=index_params
    )
    logger.info("Collection created.")


# ─────────────────────────── raw extraction + chunking ───────────────────────


def extract_pages(pdf_bytes: bytes) -> list[str]:
    """Native text page by page via pypdf, no OCR."""
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception as exc:  # corrupted page: do not abort the document
            logger.warning(f"  page extraction failed: {exc}")
            pages.append("")
    return pages


def token_windows(
    pages: list[str],
    tokenizer,
    chunk_tokens: int,
    overlap_tokens: int,
    min_chars: int,
) -> list[dict]:
    """Split the concatenated text into sliding windows of bge-m3 tokens.

    Each page is encoded separately while keeping the page number of each token,
    so we can record the pages covered by each window. Windows are
    `chunk_tokens` tokens long with a stride of `chunk_tokens - overlap_tokens`.
    """
    token_ids: list[int] = []
    token_pages: list[int] = []
    for page_no, text in enumerate(pages, start=1):
        text = (text or "").strip()
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        token_ids.extend(ids)
        token_pages.extend([page_no] * len(ids))

    if not token_ids:
        return []

    stride = max(1, chunk_tokens - overlap_tokens)
    windows: list[dict] = []
    for start in range(0, len(token_ids), stride):
        win_ids = token_ids[start : start + chunk_tokens]
        if not win_ids:
            break
        text = tokenizer.decode(win_ids, skip_special_tokens=True).strip()
        if len(text) >= min_chars:
            win_pages = sorted(set(token_pages[start : start + chunk_tokens]))
            windows.append({
                "text": text,
                "num_tokens": len(win_ids),
                "page_numbers": win_pages,
            })
        if start + chunk_tokens >= len(token_ids):
            break
    return windows


# ─────────────────────────── per-document pipeline ───────────────────────────


def process_one(
    entry: dict,
    cleyrop: CleyropClient,
    tokenizer,
    oai: OpenAI,
    model_name: str,
    milvus: MilvusClient,
    collection: str,
    chunk_tokens: int,
    overlap_tokens: int,
    min_chars: int,
    batch_size: int,
    manifests_dir: Path,
    force: bool,
) -> tuple[str, float]:
    file_id = entry["id"]
    name = entry["name"]

    if not force and is_already_done(file_id, manifests_dir):
        logger.debug(f"Already processed, skip: {name}")
        return "skipped", 0.0

    t0 = time.monotonic()
    try:
        logger.info(f"Downloading: {name}")
        pdf_bytes = cleyrop.download_file_bytes(file_id)

        pages = extract_pages(pdf_bytes)
        windows = token_windows(
            pages, tokenizer, chunk_tokens, overlap_tokens, min_chars
        )

        if not windows:
            elapsed = time.monotonic() - t0
            logger.warning(f"  → 0 chunks (PDF without native text?): {name}")
            save_local_manifest(file_id, {
                "file_id": file_id, "name": name, "path": entry["path"],
                "corpus_minimal": entry.get("corpus_minimal", False),
                "status": "empty", "processed_at": datetime.now().isoformat(),
                "num_pages": len(pages), "num_chunks": 0,
                "elapsed_s": round(elapsed, 1),
            }, manifests_dir)
            return "empty", elapsed

        rows: list[dict] = []
        for i in range(0, len(windows), batch_size):
            batch = windows[i : i + batch_size]
            vectors = embed_texts(oai, model_name, [w["text"] for w in batch])
            for j, (w, vec) in enumerate(zip(batch, vectors)):
                idx = i + j
                rows.append({
                    "chunk_id": f"{file_id}_{idx:04d}",
                    "file_id": file_id,
                    "name": name,
                    "path": entry["path"],
                    "corpus_minimal": entry.get("corpus_minimal", False),
                    "text": w["text"][:65535],
                    "embedding": vec,
                })

        milvus.upsert(collection_name=collection, data=rows)

        elapsed = time.monotonic() - t0
        max_tok = max(w["num_tokens"] for w in windows)
        logger.info(
            f"  → {len(rows)} chunks | {len(pages)}p | max {max_tok} tok | {elapsed:.1f}s"
        )
        save_local_manifest(file_id, {
            "file_id": file_id, "name": name, "path": entry["path"],
            "corpus_minimal": entry.get("corpus_minimal", False),
            "status": "done", "processed_at": datetime.now().isoformat(),
            "num_pages": len(pages), "num_chunks": len(rows),
            "max_tokens": max_tok, "elapsed_s": round(elapsed, 1),
        }, manifests_dir)
        return "done", elapsed

    except Exception as e:
        elapsed = time.monotonic() - t0
        logger.error(f"Error on {name}: {e}")
        save_local_manifest(file_id, {
            "file_id": file_id, "name": name, "path": entry["path"],
            "status": "error", "error": str(e),
            "processed_at": datetime.now().isoformat(),
        }, manifests_dir)
        return "error", elapsed


# ─────────────────────────── status ──────────────────────────────────────────


def print_status(files: list[dict], manifests_dir: Path) -> None:
    counts: dict[str, int] = {"done": 0, "empty": 0, "error": 0, "pending": 0}
    total_elapsed = total_chunks = 0
    errors: list[tuple[str, str]] = []

    for entry in files:
        m = load_local_manifest(entry["id"], manifests_dir)
        status = m.get("status", "pending")
        counts[status if status in counts else "pending"] += 1
        if status in ("done", "empty"):
            total_elapsed += m.get("elapsed_s", 0)
            total_chunks += m.get("num_chunks", 0)
        if status == "error":
            errors.append((entry.get("path", entry["name"]), m.get("error", "")))

    total = len(files)
    print(f"\nRaw pipeline status ({total} PDF files):")
    print(f"  ✓ done    : {counts['done']}")
    print(f"  ∅ empty   : {counts['empty']}  (PDF without native text)")
    print(f"  ⏳ pending : {counts['pending']}")
    print(f"  ✗ error   : {counts['error']}")
    done = counts["done"] + counts["empty"]
    if done:
        print(f"\n  Cumulative chunks: {total_chunks}")
        print(f"  Total time: {total_elapsed:.0f}s | Avg/doc: {total_elapsed/done:.1f}s")
    for path, msg in sorted(errors):
        print(f"  ✗ {path}: {msg}")


# ─────────────────────────── main ────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Raw pipeline PDF → Milvus (pypdf, no OCR)")
    parser.add_argument("--manifest", default="output/manifest_fragus_clean.dedup.json")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--corpus-minimal", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).parent.parent.parent
    manifests_dir = root / "output" / "brute_manifests"

    files = filter_files(load_manifest(root / args.manifest))
    if args.corpus_minimal:
        files = [f for f in files if f.get("corpus_minimal")]
        logger.info(f"--corpus-minimal: {len(files)} PDFs targeted")

    if args.status:
        print_status(files, manifests_dir)
        return

    if args.sample:
        pending = [f for f in files if not is_already_done(f["id"], manifests_dir)]
        files = pending[: args.sample]
        logger.info(f"Sample mode: {len(files)} files to process")

    emb_cfg = _cfg_root["embedding"]
    milvus_cfg = _cfg_root["milvus"]
    llm_cfg = _cfg_root["llm"]
    brut_cfg = _cfg_root["chunking_brut"]
    collection = milvus_cfg["collection_brut"]

    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    logger.info(
        f"Raw chunking: {brut_cfg['chunk_tokens']} tok / overlap "
        f"{brut_cfg['overlap_tokens']} | extractor={brut_cfg['extractor']} | no OCR"
    )

    oai, model_name = build_openai_client(llm_cfg["proxy_prefix"], emb_cfg["model_key"])
    milvus = MilvusClient(uri=milvus_cfg["uri"])
    ensure_collection(milvus, collection, emb_cfg["dim"])
    logger.info(f"Target collection: {collection}")

    cleyrop_cfg = make_cleyrop_config()
    stats: dict[str, int] = {"done": 0, "empty": 0, "skipped": 0, "error": 0}
    total_elapsed = 0.0

    with CleyropClient(cleyrop_cfg) as cleyrop:
        if cleyrop_cfg.client_secret:
            cleyrop.login_client_credentials()

        for i, entry in enumerate(files, 1):
            logger.info(f"[{i}/{len(files)}] {entry['name']}")
            result, elapsed = process_one(
                entry=entry,
                cleyrop=cleyrop,
                tokenizer=tokenizer,
                oai=oai,
                model_name=model_name,
                milvus=milvus,
                collection=collection,
                chunk_tokens=brut_cfg["chunk_tokens"],
                overlap_tokens=brut_cfg["overlap_tokens"],
                min_chars=brut_cfg["min_chunk_chars"],
                batch_size=emb_cfg["batch_size"],
                manifests_dir=manifests_dir,
                force=args.force,
            )
            stats[result] = stats.get(result, 0) + 1
            if result in ("done", "empty"):
                total_elapsed += elapsed

    milvus.close()

    processed = stats["done"] + stats["empty"]
    avg = (total_elapsed / processed) if processed else 0
    print(
        f"\nResults: {stats['done']} processed, {stats['empty']} empty "
        f"({total_elapsed:.0f}s, avg {avg:.1f}s/doc), "
        f"{stats['skipped']} skipped, {stats['error']} errors"
    )


if __name__ == "__main__":
    main()
