"""Sparse embedding (fermi-1024) of the docling chunks → new Milvus collection.

fermi (`atomic-canyon/fermi-1024`) is a learned SPARSE model (SPLADE/LSR family),
English, fine-tuned on the nuclear domain — NOT a dense model like bge-m3. It is not
exposed by the ai-gen proxy: inference is done LOCALLY via `transformers` (CPU here).

We reuse as-is the chunks of the docling collection `documents_vectorises`
(read via query_iterator) and write their sparse vectors into a dedicated
collection `documents_vectorises_fermi` (SPARSE_FLOAT_VECTOR, IP metric).

The "raw" corpus (pypdf) reuses the same mechanics: the chunks are read from
the raw dense collection `documents_vectorises_brut` (no local cache) and written
into `documents_vectorises_brut_fermi` (see --corpus brut).

Usage:
    .venv/bin/python scripts/pipeline/embed_fermi.py --reset --limit 50   # docling smoke test
    .venv/bin/python scripts/pipeline/embed_fermi.py --reset              # full docling run
    # raw corpus via a remote Mac/MPS encoder (reverse SSH tunnel):
    .venv/bin/python scripts/pipeline/embed_fermi.py --corpus brut --reset \
        --encoder-url http://localhost:8000/encode --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import tomllib
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import torch
from dotenv import load_dotenv
from pymilvus import DataType, MilvusClient
from pymilvus.exceptions import MilvusException
from transformers import AutoModelForMaskedLM, AutoTokenizer

load_dotenv(override=True)  # allows providing HF_TOKEN via .env

_ROOT = Path(__file__).parent.parent.parent
_config_path = _ROOT / "config.toml"
with open(_config_path, "rb") as f:
    _cfg_root = tomllib.load(f)

CACHE = _ROOT / "output" / "all_chunks_cache.jsonl"
_NEEDED_FIELDS = ["chunk_id", "file_id", "name", "path", "corpus_minimal", "text"]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────── fermi model (local) ─────────────────────────────


def load_fermi(model_id: str):
    """Loads the fermi model + tokenizer locally (CPU). Also returns the
    special token ids (zeroed out in the sparse vector) and the device."""
    torch.set_num_threads(os.cpu_count() or 1)
    logger.info(f"Loading fermi: {model_id} (CPU, {os.cpu_count()} threads)")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id)
    model.eval()
    # Dynamic int8 quantization on the Linear layers: 2-3x on CPU for a BERT,
    # negligible impact on retrieval quality.
    try:
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        logger.info("Dynamic int8 quantization applied (Linear layers).")
    except Exception as e:  # pragma: no cover - depends on the torch build
        logger.warning(f"Quantization unavailable, float32 model: {e}")
    special_token_ids = tokenizer.all_special_ids
    return model, tokenizer, special_token_ids


def get_sparse_vector(feature, output, special_token_ids):
    """Official fermi recipe: masked max-pool over the sequence then
    log(1 + relu), special tokens zeroed out. output = logits [B, L, V]."""
    values, _ = torch.max(output * feature["attention_mask"].unsqueeze(-1), dim=1)
    values = torch.log(1 + torch.relu(values))
    values[:, special_token_ids] = 0
    return values  # [B, vocab]


def encode_sparse(
    texts: list[str], model, tokenizer, special_token_ids, max_seq_len: int
) -> list[dict[int, float]]:
    """Encodes a batch of texts into sparse vectors {token_id: weight}."""
    feature = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
        return_token_type_ids=False,
    )
    with torch.inference_mode():
        output = model(**feature)[0]
    values = get_sparse_vector(feature, output, special_token_ids)

    sparse: list[dict[int, float]] = []
    for i in range(values.shape[0]):
        row = values[i]
        idx = torch.nonzero(row, as_tuple=False).squeeze(1)
        weights = row[idx]
        sparse.append(
            {int(k): float(v) for k, v in zip(idx.tolist(), weights.tolist())}
        )
    return sparse


def encode_sparse_remote(
    texts: list[str], url: str, max_seq_len: int,
    timeout: float = 300.0, retries: int = 6, backoff: float = 3.0,
) -> list[dict[int, float]]:
    """Encodes via the remote inference server (Mac/MPS). Returns the same
    {token_id: weight} format (JSON string keys are converted back to int).

    Robust to transient hiccups of the reverse SSH tunnel (network drops,
    timeouts, reconnections): retries with exponential backoff — without this,
    a single lost request kills the whole run."""
    payload = json.dumps({"texts": texts, "max_seq_len": max_seq_len}).encode()
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            return [{int(k): float(v) for k, v in d.items()} for d in data["sparse"]]
        except (HTTPError, URLError, TimeoutError, ConnectionError) as e:
            last_err = e
            if attempt < retries:
                wait = backoff * (2 ** (attempt - 1))
                logger.warning(f"  remote encoding failed ({e}) — retry {attempt}/{retries - 1} in {wait:.0f}s")
                time.sleep(wait)
    raise RuntimeError(f"remote encoding failed after {retries} attempts: {last_err}")


# ─────────────────────────── Milvus ──────────────────────────────────────────


def ensure_sparse_collection(client: MilvusClient, collection: str, reset: bool) -> None:
    if collection in client.list_collections():
        if not reset:
            logger.info(f"Existing collection: {collection} (upsert without reset)")
            return
        logger.info(f"--reset: dropping collection {collection}")
        client.drop_collection(collection_name=collection)

    logger.info(f"Creating sparse collection: {collection}")
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("chunk_id", DataType.VARCHAR, max_length=512, is_primary=True)
    schema.add_field("file_id", DataType.VARCHAR, max_length=256)
    schema.add_field("name", DataType.VARCHAR, max_length=512)
    schema.add_field("path", DataType.VARCHAR, max_length=1024)
    schema.add_field("corpus_minimal", DataType.BOOL)
    schema.add_field("text", DataType.VARCHAR, max_length=65535)
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="sparse",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
    )

    client.create_collection(
        collection_name=collection,
        schema=schema,
        index_params=index_params,
    )
    logger.info("Collection created.")


def ensure_loaded(client: MilvusClient, name: str, retries: int = 30, delay: float = 5.0) -> None:
    """Loads the collection into memory (required for query_iterator). Tolerates
    a transient 'on recovering' state by retrying."""
    for attempt in range(1, retries + 1):
        try:
            client.load_collection(collection_name=name)
            logger.info(f"Collection loaded into memory: {name}")
            return
        except MilvusException as e:
            if "recover" in str(e).lower() and attempt < retries:
                logger.info(f"  {name} in recovery, retry {attempt}/{retries} in {delay:.0f}s…")
                time.sleep(delay)
                continue
            raise


def iter_source_chunks(client: MilvusClient, source: str, read_batch: int = 2000):
    """Iterates over the chunks of the source docling Milvus collection (without
    the dense vector). Requires the collection to be loaded (see ensure_loaded)."""
    it = client.query_iterator(
        collection_name=source,
        filter="",
        output_fields=_NEEDED_FIELDS,
        batch_size=read_batch,
    )
    try:
        while True:
            batch = it.next()
            if not batch:
                break
            yield batch
    finally:
        it.close()


def count_cache(path: Path) -> int:
    with path.open(encoding="utf-8") as f:
        return sum(1 for ln in f if ln.strip())


def iter_cache_chunks(path: Path, read_batch: int = 2000):
    """Iterates over the docling chunks from the local JSONL cache (offline) —
    independent of the loading state of the source Milvus collection."""
    buf: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            c = json.loads(ln)
            buf.append({k: c.get(k) for k in _NEEDED_FIELDS})
            if len(buf) >= read_batch:
                yield buf
                buf = []
    if buf:
        yield buf


# ─────────────────────────── main ────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Sparse fermi embedding → Milvus")
    parser.add_argument("--reset", action="store_true", help="recreate the target collection")
    parser.add_argument("--limit", type=int, default=None, help="process only N chunks (smoke test)")
    parser.add_argument(
        "--corpus", choices=["docling", "brut"], default="docling",
        help="corpus to encode: docling (default, JSONL cache) or brut (pypdf, read from Milvus). "
             "brut forces --source milvus (no local cache for the raw corpus).",
    )
    parser.add_argument(
        "--source", choices=["cache", "milvus"], default="cache",
        help="origin of the chunks: local JSONL cache (default) or Milvus collection",
    )
    parser.add_argument(
        "--encoder-url", default=None,
        help="URL of a remote fermi inference server (e.g. http://localhost:8000/encode). "
             "If absent, local inference on the VM (CPU).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="encoding batch size (otherwise config value); increase in remote mode",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="resume: do not re-encode chunk_ids already present in the target collection",
    )
    args = parser.parse_args()

    fcfg = _cfg_root["embedding_fermi"]
    milvus_cfg = _cfg_root["milvus"]
    if args.corpus == "brut":
        # pypdf corpus: dedicated sources/targets (see [fragus]). No local cache
        # for the raw corpus → we necessarily read from the raw dense Milvus collection.
        gcfg = _cfg_root["fragus"]
        collection = gcfg["brut_fermi"]      # documents_vectorises_brut_fermi
        source = gcfg["brut_dense"]          # documents_vectorises_brut
        if args.source == "cache":
            logger.info("raw corpus: forced switch to --source milvus (no raw cache)")
            args.source = "milvus"
    else:
        collection = fcfg["collection"]
        source = fcfg["source_collection"]
    enc_batch = int(args.batch_size or fcfg["batch_size"])
    max_seq_len = int(fcfg["max_seq_len"])
    upsert_batch = 500

    if args.encoder_url:
        logger.info(f"Remote encoding via {args.encoder_url} (batch={enc_batch})")
        def encode_batch(texts: list[str]) -> list[dict[int, float]]:
            return encode_sparse_remote(texts, args.encoder_url, max_seq_len)
    else:
        model, tokenizer, special_token_ids = load_fermi(fcfg["model_id"])
        def encode_batch(texts: list[str]) -> list[dict[int, float]]:
            return encode_sparse(texts, model, tokenizer, special_token_ids, max_seq_len)

    client = MilvusClient(uri=milvus_cfg["uri"])
    ensure_sparse_collection(client, collection, args.reset)

    # Choice of the docling chunk source
    if args.source == "cache":
        if not CACHE.exists():
            raise RuntimeError(f"Cache not found: {CACHE} (use --source milvus)")
        src_total = count_cache(CACHE)
        batches = iter_cache_chunks(CACHE)
        src_label = f"cache {CACHE.name}"
    else:
        if source not in client.list_collections():
            raise RuntimeError(f"Source collection not found: {source}")
        ensure_loaded(client, source)
        src_total = client.get_collection_stats(collection_name=source)["row_count"]
        batches = iter_source_chunks(client, source)
        src_label = f"collection {source}"

    # Materialize the chunks (respecting --limit) then sort by text length:
    # batches of homogeneous lengths minimize padding, and therefore the compute
    # wasted on the vocabulary projection (the dominant cost on CPU).
    all_chunks: list[dict] = []
    for batch in batches:
        all_chunks.extend(batch)
        if args.limit and len(all_chunks) >= args.limit:
            del all_chunks[args.limit :]
            break
    all_chunks.sort(key=lambda c: len(c["text"] or ""))

    # Resume: remove the chunk_ids already written to the target collection (the PK
    # makes the upsert idempotent, but this avoids re-encoding for nothing).
    if args.skip_existing and collection in client.list_collections():
        ensure_loaded(client, collection)
        done: set[str] = set()
        it = client.query_iterator(
            collection_name=collection, filter="", output_fields=["chunk_id"], batch_size=2000
        )
        try:
            while True:
                b = it.next()
                if not b:
                    break
                done.update(r["chunk_id"] for r in b)
        finally:
            it.close()
        before = len(all_chunks)
        all_chunks = [c for c in all_chunks if c["chunk_id"] not in done]
        logger.info(f"--skip-existing: {len(done)} already present · {before - len(all_chunks)} skipped")

    target = len(all_chunks)
    logger.info(f"Source: {src_label} ({src_total} chunks) · target to process: {target}")

    t0 = time.monotonic()
    processed = 0   # encoded chunks
    upserted = 0    # entities written to Milvus
    empty = 0       # chunks with an empty sparse vector (excluded)
    pending: list[dict] = []

    def flush() -> None:
        nonlocal upserted
        if pending:
            client.upsert(collection_name=collection, data=pending)
            upserted += len(pending)
            pending.clear()

    for i in range(0, target, enc_batch):
        sub = all_chunks[i : i + enc_batch]
        vectors = encode_batch([c["text"] or "" for c in sub])
        for chunk, vec in zip(sub, vectors):
            if not vec:
                empty += 1
                logger.warning(f"Empty sparse vector, excluded: {chunk['chunk_id']}")
                continue
            pending.append({
                "chunk_id": chunk["chunk_id"],
                "file_id": chunk["file_id"],
                "name": chunk["name"],
                "path": chunk["path"],
                "corpus_minimal": chunk.get("corpus_minimal", False),
                "text": (chunk["text"] or "")[:65535],
                "sparse": vec,
            })
        processed += len(sub)

        if len(pending) >= upsert_batch:
            flush()
        if processed % 1000 < enc_batch:
            rate = processed / (time.monotonic() - t0)
            eta = (target - processed) / rate if rate else 0
            logger.info(f"  … {processed}/{target} encoded ({rate:.1f}/s, {upserted} written, ETA {eta/60:.0f} min)")

    flush()
    elapsed = time.monotonic() - t0
    logger.info("─" * 60)
    logger.info(
        f"Done: {processed} chunks encoded · {upserted} entities written to "
        f"{collection} · {empty} excluded (empty sparse) · {elapsed:.0f}s"
    )
    client.close()


if __name__ == "__main__":
    main()
