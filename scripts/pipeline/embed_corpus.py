"""Embedding pipeline for chunks → Milvus.

Downloads the chunk JSONL files from Cleyrop (produced by chunk_corpus.py),
generates the embeddings via the AI-gen proxy, and upserts into Milvus.

Usage:
    uv run scripts/pipeline/embed_corpus.py [OPTIONS]

    --manifest        output/manifest_fragus_clean.dedup.json  (default)
    --config          configs/chunker.yaml  (default)
    --sample N        process the first N files not yet processed
    --force           re-embed even if already done
    --status          show the state without processing
    --corpus-minimal  process only the files flagged corpus_minimal
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import tomllib
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen

import yaml
from dotenv import load_dotenv
from openai import OpenAI
from pymilvus import DataType, MilvusClient

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


def load_chunker_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def resolve_project_id(cfg: ClientConfig) -> str:
    client = CleyropClient(cfg)
    if cfg.client_secret:
        client.login_client_credentials()
    with client:
        if project_id := os.environ.get("PROJECT_ID"):
            return project_id
        if project_id := _cfg_root["project"].get("id"):
            return project_id
        project_slug = os.environ["PROJECT_SLUG"]
        return str(client.get_project_by_slug(project_slug).id)


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


def is_already_embedded(file_id: str, manifests_dir: Path) -> bool:
    return load_local_manifest(file_id, manifests_dir).get("status") == "done"


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
    """Returns (OpenAI client, model_name) for embedding."""
    models = _resolve_proxy_models(proxy_prefix)
    if model_key in models:
        m = models[model_key]
        logger.info(f"Embedding model resolved: {model_key} → {m['endpoint']}")
        return OpenAI(base_url=m["endpoint"], api_key="dummy"), m["name"]

    raise RuntimeError(
        f"Model '{model_key}' not found in the proxy's /emb/models. "
        f"Available models: {sorted(models)}"
    )


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
        collection_name=collection,
        schema=schema,
        index_params=index_params,
    )
    logger.info("Collection created.")


# ─────────────────────────── embedding ───────────────────────────────────────


def embed_texts(oai: OpenAI, model_name: str, texts: list[str]) -> list[list[float]]:
    resp = oai.embeddings.create(model=model_name, input=texts, encoding_format="float")
    return [e.embedding for e in resp.data]


# ─────────────────────────── per-document pipeline ───────────────────────────


def process_one(
    entry: dict,
    cleyrop: CleyropClient,
    oai: OpenAI,
    model_name: str,
    milvus: MilvusClient,
    collection: str,
    batch_size: int,
    chunk_manifests_dir: Path,
    embed_manifests_dir: Path,
    force: bool,
) -> tuple[str, float]:
    file_id = entry["id"]
    name = entry["name"]

    if not force and is_already_embedded(file_id, embed_manifests_dir):
        logger.debug(f"Already embedded, skip: {name}")
        return "skipped", 0.0

    t0 = time.monotonic()
    try:
        # Get the Cleyrop file_id of the JSONL from the chunk_manifest
        cm = load_local_manifest(file_id, chunk_manifests_dir)
        jsonl_file_id = cm.get("cleyrop_jsonl_file_id")
        if not jsonl_file_id:
            logger.warning(f"No JSONL for: {name} — run chunk_corpus.py first")
            return "skipped", 0.0

        logger.info(f"Downloading JSONL: {name}")
        jsonl_bytes = cleyrop.download_file_bytes(jsonl_file_id)
        chunks = [
            json.loads(line)
            for line in jsonl_bytes.decode("utf-8").splitlines()
            if line.strip()
        ]

        if not chunks:
            logger.warning(f"Empty JSONL for: {name}")
            return "skipped", 0.0

        logger.info(f"Embedding {len(chunks)} chunks: {name}")
        rows: list[dict] = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c["text"] for c in batch]
            vectors = embed_texts(oai, model_name, texts)
            for chunk, vec in zip(batch, vectors):
                rows.append({
                    "chunk_id": chunk["chunk_id"],
                    "file_id": chunk["file_id"],
                    "name": chunk["name"],
                    "path": chunk["path"],
                    "corpus_minimal": chunk.get("corpus_minimal", False),
                    "text": chunk["text"][:65535],
                    "embedding": vec,
                })

        milvus.upsert(collection_name=collection, data=rows)

        elapsed = time.monotonic() - t0
        logger.info(f"  → {len(rows)} entities upserted | {elapsed:.1f}s")

        save_local_manifest(file_id, {
            "file_id": file_id,
            "name": name,
            "path": entry["path"],
            "status": "done",
            "embedded_at": datetime.now().isoformat(),
            "num_chunks": len(rows),
            "elapsed_s": round(elapsed, 1),
        }, embed_manifests_dir)
        return "done", elapsed

    except Exception as e:
        elapsed = time.monotonic() - t0
        logger.error(f"Error on {name}: {e}")
        save_local_manifest(file_id, {
            "file_id": file_id, "name": name, "path": entry["path"],
            "status": "error", "error": str(e), "embedded_at": datetime.now().isoformat(),
        }, embed_manifests_dir)
        return "error", elapsed


# ─────────────────────────── status ──────────────────────────────────────────


def print_status(files: list[dict], embed_manifests_dir: Path, chunk_manifests_dir: Path) -> None:
    counts: dict[str, int] = {"done": 0, "error": 0, "pending": 0, "not_chunked": 0}
    total_elapsed = 0.0
    errors: list[tuple[str, str]] = []

    for entry in files:
        em = load_local_manifest(entry["id"], embed_manifests_dir)
        status = em.get("status", "pending")
        if status == "pending":
            cm = load_local_manifest(entry["id"], chunk_manifests_dir)
            if not cm.get("cleyrop_jsonl_file_id"):
                status = "not_chunked"
        counts[status if status in counts else "pending"] += 1
        if status == "done":
            total_elapsed += em.get("elapsed_s", 0)
        if status == "error":
            errors.append((entry.get("path", entry["name"]), em.get("error", "")))

    total = len(files)
    print(f"\nEmbedding status ({total} files):")
    print(f"  ✓ done        : {counts['done']}")
    print(f"  ⏳ pending     : {counts['pending']}")
    print(f"  ⚠ not_chunked : {counts['not_chunked']}")
    print(f"  ✗ error       : {counts['error']}")
    if counts["done"] > 0:
        print(f"\n  Total time: {total_elapsed:.0f}s | Avg/doc: {total_elapsed/counts['done']:.1f}s")
    for path, msg in sorted(errors):
        print(f"  ✗ {path}: {msg}")


# ─────────────────────────── main ────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Embedding pipeline chunks → Milvus")
    parser.add_argument("--manifest", default="output/manifest_fragus_clean.dedup.json")
    parser.add_argument("--config", default="configs/chunker.yaml")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--corpus-minimal", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).parent.parent.parent
    cfg = load_chunker_config(root / args.config)
    embed_manifests_dir = root / "output" / "embed_manifests"
    chunk_manifests_dir = root / "output" / "chunk_manifests"

    files = filter_files(load_manifest(root / args.manifest))
    if args.corpus_minimal:
        files = [f for f in files if f.get("corpus_minimal")]
        logger.info(f"--corpus-minimal: {len(files)} files targeted")

    if args.status:
        print_status(files, embed_manifests_dir, chunk_manifests_dir)
        return

    if args.sample:
        pending = [f for f in files if not is_already_embedded(f["id"], embed_manifests_dir)]
        files = pending[: args.sample]
        logger.info(f"Sample mode: {len(files)} files to process")

    emb_cfg = _cfg_root["embedding"]
    milvus_cfg = _cfg_root["milvus"]
    llm_cfg = _cfg_root["llm"]

    oai, model_name = build_openai_client(llm_cfg["proxy_prefix"], emb_cfg["model_key"])
    milvus = MilvusClient(uri=milvus_cfg["uri"])
    ensure_collection(milvus, milvus_cfg["collection"], emb_cfg["dim"])

    cleyrop_cfg = make_cleyrop_config()
    project_id = resolve_project_id(cleyrop_cfg)

    stats: dict[str, int] = {"done": 0, "skipped": 0, "error": 0}
    total_elapsed = 0.0

    with CleyropClient(cleyrop_cfg) as cleyrop:
        if cleyrop_cfg.client_secret:
            cleyrop.login_client_credentials()

        for i, entry in enumerate(files, 1):
            logger.info(f"[{i}/{len(files)}] {entry['name']}")
            result, elapsed = process_one(
                entry=entry,
                cleyrop=cleyrop,
                oai=oai,
                model_name=model_name,
                milvus=milvus,
                collection=milvus_cfg["collection"],
                batch_size=emb_cfg["batch_size"],
                chunk_manifests_dir=chunk_manifests_dir,
                embed_manifests_dir=embed_manifests_dir,
                force=args.force,
            )
            stats[result] = stats.get(result, 0) + 1
            if result == "done":
                total_elapsed += elapsed

    milvus.close()

    avg = (total_elapsed / stats["done"]) if stats["done"] else 0
    print(
        f"\nResults: {stats['done']} processed ({total_elapsed:.0f}s, avg {avg:.1f}s/doc), "
        f"{stats.get('skipped', 0)} skipped, "
        f"{stats['error']} errors"
    )


if __name__ == "__main__":
    main()
