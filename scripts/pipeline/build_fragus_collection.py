"""Builds a target `*_fragus` collection with 3 vector fields via a join.

Nothing is re-embedded: we gather into a single collection, joined on
`chunk_id`, the vectors that already exist in separate collections —
   • `embedding`     (dense bge-m3)   ← existing dense collection
   • `sparse_fermi`  (sparse SPLADE)  ← existing fermi collection
   • `sparse_bm25`   (sparse lexical) ← COMPUTED here via pymilvus.model (BM25, FR analyzer)

Since the 3 vectors then share the same chunk_id, the `hybrid_search` +
`RRFRanker` fusion is native to Milvus (multi-vector since 2.4) — no more Python-side RRF.

The BM25 state frozen at fit time (idf + avgdl + corpus_size) is PERSISTED (save()): it
must be reloaded as-is to encode queries (same FR analyzer, same idf), otherwise
the scores are wrong. The state file must be shared with the query pipeline.

Usage:
    .venv/bin/python scripts/pipeline/build_fragus_collection.py --corpus baseline --reset
    .venv/bin/python scripts/pipeline/build_fragus_collection.py --corpus brut --reset   # after fermi brut
"""

from __future__ import annotations

import argparse
import logging
import time
import tomllib
import warnings
from pathlib import Path

from pymilvus import DataType, MilvusClient
from pymilvus.exceptions import MilvusException

warnings.filterwarnings("ignore")  # nltk / pymilvus.model noise

_ROOT = Path(__file__).parent.parent.parent
with open(_ROOT / "config.toml", "rb") as f:
    _cfg = tomllib.load(f)

_SCALARS = ["chunk_id", "file_id", "name", "path", "corpus_minimal", "text"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────── reading sources ─────────────────────────────────


def ensure_loaded(client: MilvusClient, name: str, retries: int = 30, delay: float = 5.0) -> None:
    for attempt in range(1, retries + 1):
        try:
            client.load_collection(collection_name=name)
            return
        except MilvusException as e:
            if "recover" in str(e).lower() and attempt < retries:
                logger.info(f"  {name} in recovery, attempt {attempt}/{retries} in {delay:.0f}s…")
                time.sleep(delay)
                continue
            raise


def iter_rows(client: MilvusClient, collection: str, fields: list[str], batch: int = 2000):
    it = client.query_iterator(
        collection_name=collection, filter="", output_fields=fields, batch_size=batch
    )
    try:
        while True:
            rows = it.next()
            if not rows:
                break
            yield from rows
    finally:
        it.close()


def read_fermi_map(client: MilvusClient, collection: str) -> dict[str, dict]:
    """chunk_id -> fermi sparse vector (already stored as SPARSE_FLOAT_VECTOR)."""
    ensure_loaded(client, collection)
    out: dict[str, dict] = {}
    for r in iter_rows(client, collection, ["chunk_id", "sparse"]):
        out[r["chunk_id"]] = r["sparse"]
    logger.info(f"fermi: {len(out)} sparse vectors read from {collection}")
    return out


def read_dense_records(client: MilvusClient, collection: str, keep: set[str]) -> list[dict]:
    """Records (scalars + dense embedding) of the chunks whose chunk_id ∈ keep."""
    ensure_loaded(client, collection)
    records: list[dict] = []
    seen = 0
    for r in iter_rows(client, collection, _SCALARS + ["embedding"]):
        seen += 1
        if r["chunk_id"] in keep:
            records.append(r)
    logger.info(f"dense: {len(records)}/{seen} chunks kept from {collection} (∩ fermi)")
    return records


# ─────────────────────────── BM25 (pymilvus.model) ───────────────────────────


def fit_bm25(texts: list[str], language: str):
    """Fit BM25 on the corpus (df/avgdl) and encode the documents. Returns
    (bm25, list of dicts {token_id: weight}) aligned with `texts`."""
    from pymilvus.model.sparse.bm25 import BM25EmbeddingFunction
    from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer

    logger.info(f"BM25: fit on {len(texts)} documents (analyzer {language})…")
    t0 = time.monotonic()
    bm25 = BM25EmbeddingFunction(analyzer=build_default_analyzer(language=language))
    bm25.fit(texts)
    matrix = bm25.encode_documents(texts)  # scipy csr (n_docs × vocab)
    coo = matrix.tocoo()
    docs: list[dict[int, float]] = [{} for _ in range(matrix.shape[0])]
    for row, col, val in zip(coo.row, coo.col, coo.data):
        docs[int(row)][int(col)] = float(val)
    logger.info(
        f"BM25: fit+encode in {time.monotonic() - t0:.1f}s · vocab {matrix.shape[1]} · "
        f"avgdl {bm25.avgdl:.1f} · corpus_size {bm25.corpus_size}"
    )
    return bm25, docs


# ─────────────────────────── target collection ───────────────────────────────


def create_target(client: MilvusClient, collection: str, dim: int, reset: bool) -> None:
    if collection in client.list_collections():
        if not reset:
            raise RuntimeError(f"{collection} already exists (use --reset to recreate)")
        logger.info(f"--reset: dropping {collection}")
        client.drop_collection(collection_name=collection)

    logger.info(f"Creating {collection} (dense + sparse_bm25 + sparse_fermi)")
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("chunk_id", DataType.VARCHAR, max_length=512, is_primary=True)
    schema.add_field("file_id", DataType.VARCHAR, max_length=256)
    schema.add_field("name", DataType.VARCHAR, max_length=512)
    schema.add_field("path", DataType.VARCHAR, max_length=1024)
    schema.add_field("corpus_minimal", DataType.BOOL)
    schema.add_field("text", DataType.VARCHAR, max_length=65535)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("sparse_bm25", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("sparse_fermi", DataType.SPARSE_FLOAT_VECTOR)

    index_params = MilvusClient.prepare_index_params()
    # Exact FLAT for the retrieval benchmark (no ANN noise) — see target-pipeline.md
    index_params.add_index(field_name="embedding", index_type="FLAT", metric_type="COSINE")
    index_params.add_index(field_name="sparse_bm25", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
    index_params.add_index(field_name="sparse_fermi", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")

    client.create_collection(collection_name=collection, schema=schema, index_params=index_params)
    logger.info("Collection created.")


# ─────────────────────────── main ────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(description="Builds a 3-vector *_fragus collection")
    p.add_argument("--corpus", choices=["baseline", "brut"], default="baseline")
    p.add_argument("--reset", action="store_true", help="recreate the target collection if it exists")
    p.add_argument("--limit", type=int, default=None, help="process only N chunks (smoke test)")
    args = p.parse_args()

    fcfg = _cfg["fragus"]
    dim = int(_cfg["embedding"]["dim"])
    if args.corpus == "baseline":
        target, dense_src, fermi_src = fcfg["collection_baseline"], fcfg["baseline_dense"], fcfg["baseline_fermi"]
    else:
        target, dense_src, fermi_src = fcfg["collection_brut"], fcfg["brut_dense"], fcfg["brut_fermi"]

    client = MilvusClient(uri=_cfg["milvus"]["uri"])
    for src in (dense_src, fermi_src):
        if src not in client.list_collections():
            raise RuntimeError(f"Missing source collection: {src}")

    # 1) fermi (lightweight) → set of chunk_ids that will have all 3 vectors
    fermi_map = read_fermi_map(client, fermi_src)
    keep = set(fermi_map)

    # 2) dense (scalars + embedding) on the intersection
    records = read_dense_records(client, dense_src, keep)
    if args.limit:
        records = records[: args.limit]
    missing_fermi = [r["chunk_id"] for r in records if r["chunk_id"] not in fermi_map]
    assert not missing_fermi, f"inconsistency: {len(missing_fermi)} dense chunks without fermi"
    logger.info(f"Intersection to insert: {len(records)} chunks")

    # 3) BM25 computed on the texts of the intersection (order = records)
    texts = [(r["text"] or "") for r in records]
    bm25, bm25_docs = fit_bm25(texts, fcfg["bm25_language"])

    # 4) persist the BM25 state (idf+avgdl) — must be reloaded to encode queries
    state_dir = _ROOT / fcfg["bm25_state_dir"]
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / f"{target}.json"
    bm25.save(str(state_path))
    logger.info(f"BM25 state persisted: {state_path} (analyzer={fcfg['bm25_language']})")

    # 5) creation + batched insert (join of the 3 vectors on chunk_id)
    create_target(client, target, dim, args.reset)
    t0 = time.monotonic()
    inserted = 0
    batch_size = 500
    pending: list[dict] = []

    def flush() -> None:
        nonlocal inserted
        if pending:
            client.insert(collection_name=target, data=pending)
            inserted += len(pending)
            pending.clear()

    for r, bm in zip(records, bm25_docs):
        if not bm:
            logger.warning(f"Empty BM25 (text with no retained term): {r['chunk_id']}")
        pending.append({
            "chunk_id": r["chunk_id"],
            "file_id": r["file_id"],
            "name": r["name"],
            "path": r["path"],
            "corpus_minimal": r.get("corpus_minimal", False),
            "text": (r["text"] or "")[:65535],
            "embedding": r["embedding"],
            "sparse_bm25": bm,
            "sparse_fermi": fermi_map[r["chunk_id"]],
        })
        if len(pending) >= batch_size:
            flush()
            if inserted % 5000 < batch_size:
                logger.info(f"  … {inserted}/{len(records)} inserted")
    flush()

    client.flush(collection_name=target)
    final = client.get_collection_stats(collection_name=target)["row_count"]
    logger.info("─" * 60)
    logger.info(
        f"Done: {inserted} inserted into {target} · {final} rows confirmed · "
        f"{time.monotonic() - t0:.0f}s"
    )
    client.close()


if __name__ == "__main__":
    main()
