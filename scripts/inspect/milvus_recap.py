"""Statistical recap of a Milvus collection.

Authoritative source: Milvus (chunk count, doc count, chunk size in
characters, chunks/doc). Tokens (bge-m3) and pages/doc are not stored in
Milvus: by default they are read from a local chunk cache
(output/all_chunks_cache.jsonl), joined on chunk_id. Without a cache,
`--retokenize` recounts tokens by re-encoding the Milvus text with bge-m3.

Usage:
    .venv/bin/python scripts/inspect/milvus_recap.py                       # docling collection
    .venv/bin/python scripts/inspect/milvus_recap.py \\
        --collection documents_vectorises_brut --retokenize        # raw collection
"""

from __future__ import annotations

import argparse
import json
import statistics
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

from pymilvus import MilvusClient

ROOT = Path(__file__).parent.parent.parent
CFG = tomllib.load(open(ROOT / "config.toml", "rb"))["milvus"]
CACHE = ROOT / "output" / "all_chunks_cache.jsonl"


def q(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    i = int(round(p * (len(s) - 1)))
    return s[i]


def line(label: str, vals: list[float], fmt: str = ".0f") -> None:
    if not vals:
        print(f"│  {label:<22s}: (empty)")
        return
    mean = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    print(
        f"│  {label:<22s}: min {min(vals):{fmt}} · p25 {q(vals,.25):{fmt}} · "
        f"med {q(vals,.5):{fmt}} · mean {mean:{fmt}} · p75 {q(vals,.75):{fmt}} · "
        f"p95 {q(vals,.95):{fmt}} · max {max(vals):{fmt}} · σ {sd:{fmt}}"
    )


def histogram(values: list[float], bins: list[tuple[int, int | None]]) -> None:
    n = len(values) or 1
    for lo, hi in bins:
        if hi is None:
            count = sum(1 for v in values if v >= lo)
            label = f">= {lo}"
        else:
            count = sum(1 for v in values if lo <= v < hi)
            label = f"{lo}–{hi - 1}"
        pct = count / n * 100
        print(f"│  {label:>12s}  {count:>6d}  {pct:5.1f}%  {'█' * int(pct / 1.5)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Statistical recap of a Milvus collection")
    ap.add_argument("--collection", default=CFG["collection"])
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument(
        "--retokenize",
        action="store_true",
        help="recount tokens by re-encoding the Milvus text (bge-m3) instead of using the cache",
    )
    args = ap.parse_args()

    col = args.collection
    cache_path = Path(args.cache)
    client = MilvusClient(uri=CFG["uri"])

    row_count = client.get_collection_stats(collection_name=col)["row_count"]
    print(f"Collection: {col} @ {CFG['uri']}")
    print(f"Reading {row_count} entities (without the vector)…")

    # --- Pull everything from Milvus (excluding the embedding) ---
    chunk_chars: list[int] = []
    chunks_per_doc: Counter[str] = Counter()
    doc_name: dict[str, str] = {}
    corpus_minimal_docs: set[str] = set()
    texts: list[str] = []  # kept only if --retokenize
    n = 0
    it = client.query_iterator(
        collection_name=col,
        filter="",
        output_fields=["chunk_id", "file_id", "name", "text", "corpus_minimal"],
        batch_size=2000,
    )
    while True:
        batch = it.next()
        if not batch:
            break
        for r in batch:
            chunk_chars.append(len(r["text"]))
            if args.retokenize:
                texts.append(r["text"])
            fid = r["file_id"]
            chunks_per_doc[fid] += 1
            doc_name[fid] = r["name"]
            if r.get("corpus_minimal"):
                corpus_minimal_docs.add(fid)
        n += len(batch)
        print(f"  … {n}/{row_count}", end="\r")
    it.close()
    client.close()
    print(f"  ✓ {n} chunks read from Milvus" + " " * 20)

    n_docs = len(chunks_per_doc)
    cpd = list(chunks_per_doc.values())

    # --- Tokens + pages/doc ---
    tokens: list[int] = []
    pages_by_doc: dict[str, set[int]] = defaultdict(set)
    pages_per_chunk: list[int] = []
    matched = 0
    if args.retokenize:
        from transformers import AutoTokenizer
        print("  Re-encoding texts (bge-m3) for token counting…")
        tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        tokens = [len(tok.encode(t, add_special_tokens=False)) for t in texts]
        matched = len(tokens)
    elif cache_path.exists():
        with cache_path.open(encoding="utf-8") as f:
            for ln in f:
                if not ln.strip():
                    continue
                c = json.loads(ln)
                nt = (c.get("docling_meta") or {}).get("num_tokens")
                if nt is not None:
                    tokens.append(nt)
                pgs = c.get("page_numbers") or []
                pages_per_chunk.append(len(pgs))
                for p in pgs:
                    pages_by_doc[c["file_id"]].add(p)
                matched += 1

    pages_per_doc = [len(s) for s in pages_by_doc.values()]

    # ============================ DISPLAY ============================
    print("\n" + "═" * 70)
    print(f"MILVUS COLLECTION RECAP · {col}")
    print("═" * 70)
    print(f"  Chunks (entities)       : {row_count:,}")
    print(f"  Documents (file_id)     : {n_docs:,}")
    print(f"  incl. corpus_minimal    : {len(corpus_minimal_docs):,} docs")
    print(f"  Chunks / doc (mean)     : {statistics.mean(cpd):.1f}")
    if tokens:
        print(f"  Total tokens (bge-m3)   : {sum(tokens):,}")
    print(f"  Total characters        : {sum(chunk_chars):,}")

    print("\n┌─ Characters / chunk (Milvus) ─────────────────────────────────────")
    line("chars", [float(x) for x in chunk_chars])
    print("└" + "─" * 67)

    print("\n┌─ Chunks / document (Milvus) ──────────────────────────────────────")
    line("chunks/doc", [float(x) for x in cpd])
    print(f"│  docs with 1 chunk     : {sum(1 for v in cpd if v == 1)}")
    print(f"│  docs >= 200 chunks    : {sum(1 for v in cpd if v >= 200)}")
    print("└" + "─" * 67)
    top = sorted(chunks_per_doc.items(), key=lambda kv: -kv[1])[:5]
    print("  Top 5 docs by chunk count:")
    for fid, cnt in top:
        print(f"    {cnt:>5d}  {doc_name.get(fid, fid)[:60]}")

    if tokens:
        print("\n┌─ Tokens / chunk (cache, bge-m3 tokenizer) ────────────────────────")
        line("tokens", [float(x) for x in tokens])
        over = sum(1 for t in tokens if t > 512)
        print(f"│  > 512 tok (target max): {over} ({over / len(tokens) * 100:.1f}%)")
        print("└" + "─" * 67)
        print("┌─ Token distribution ──────────────────────────────────────────────")
        histogram([float(x) for x in tokens],
                  [(0, 50), (50, 100), (100, 256), (256, 512), (512, 768), (768, 1024), (1024, None)])
        print("└" + "─" * 67)

    if pages_per_doc:
        print("\n┌─ Pages / document (cache, distinct pages covered) ────────────────")
        line("pages/doc", [float(x) for x in pages_per_doc])
        print("└" + "─" * 67)
        print("┌─ Pages covered / chunk (cache) ───────────────────────────────────")
        line("pages/chunk", [float(x) for x in pages_per_chunk], ".2f")
        multi = sum(1 for x in pages_per_chunk if x > 1)
        print(f"│  multi-page chunks     : {multi} ({multi / len(pages_per_chunk) * 100:.1f}%)")
        print("└" + "─" * 67)

    if args.retokenize:
        print(f"\nNote: tokens recounted by re-encoding the Milvus text "
              f"({matched:,} chunks, bge-m3). Pages/doc unavailable "
              "(no cache, not stored in Milvus).")
    elif matched:
        print(f"\nNote: tokens/pages come from the local cache ({matched:,} chunks, "
              f"Milvus={row_count:,}). Gap = chunks not re-downloaded.")
    else:
        print("\nNote: local cache missing → tokens/pages unavailable "
              "(not stored in Milvus). Use --retokenize for tokens.")


if __name__ == "__main__":
    main()
