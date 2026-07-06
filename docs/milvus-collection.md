# Milvus collections

Two vector collections are built on the **same 223-PDF corpus**,
to compare two parsing/chunking strategies (`config.toml` → `[milvus]`):

| Collection | Parsing | Chunking | Chunks | Docs | Cumulative tokens |
|---|---|---|---|---|---|
| `documents_vectorises` | docling + OCR + tables | HybridChunker (≤512 tok, semantic) | **29,376** | 221 | 9.03 M |
| `documents_vectorises_brut` | pypdf, **no OCR** | 500-tok windows / 100 overlap | **15,520** | 219 | 7.62 M |

Both share the same `BAAI/bge-m3` embedding (1024-d) and the **schema below**.
The raw collection has ~2× fewer chunks: no tables and no over-segmentation,
and 4 scanned PDFs without native text are absent from it (see below).

## Schema (common to both collections)

| Field | Type | Notes |
|---|---|---|
| `chunk_id` | VARCHAR (PK) | `{file_id}_{sequential index}` — **not** a page number |
| `file_id` | VARCHAR | document identifier |
| `name` | VARCHAR | source PDF name |
| `path` | VARCHAR | path within the corpus |
| `corpus_minimal` | BOOL | membership in the minimal corpus |
| `text` | VARCHAR | chunk text |
| `embedding` | FLOAT_VECTOR | 1024-d, COSINE metric, IVF_FLAT index (bge-m3) |

> **Milvus server version: v2.4.13-hotfix** (will move to v2.6.15 with platform
> release 4.5.3). Consequences for hybrid retrieval:
> - multi-vector + `hybrid_search` + `RRFRanker`/`WeightedRanker`: ✅ since 2.4;
> - **native BM25** (`Function`/`FunctionType.BM25`, full-text search): ❌ requires
>   **≥ 2.5** → unavailable before 2.6.15. **But we do not wait for it**: BM25 is an
>   algorithm, computed client-side via `pymilvus.model` (`BM25EmbeddingFunction`,
>   **FR** analyzer) and stored as an ordinary `SPARSE_FLOAT_VECTOR` field;
> - the **lexical sparse output of bge-m3** is **not** stored and is **not** exposed
>   by the OVH/proxy embedding API (dense only) → out of scope.
>
> **Target collections: `baseline_fragus` (docling) and `brut_fragus` (pypdf)**, each with
> **3 vector fields** — `embedding` (dense bge-m3), `sparse_bm25`,
> `sparse_fermi` — for a native `hybrid_search` + `RRFRanker` fusion over all 3.
> Built by joining the existing collections on `chunk_id` (dense + fermi
> copied, BM25 computed). **`baseline_fragus` is built** (29,358 chunks);
> `brut_fragus` remains to be done (prerequisite: fermi on the raw chunks, not yet computed). See
> the dedicated section below.

---

## `documents_vectorises` — docling + OCR + HybridChunker

Pipeline: `convert_corpus.py` (docling-serve, OCR + tables) → `chunk_corpus.py`
(HybridChunker, semantic boundaries) → `embed_corpus.py`. The `text` field carries
a context header (Document / Path / Chapter). Generated on 2026-05-27 via:

```bash
.venv/bin/python scripts/inspect/milvus_recap.py
```

### Overview

| Metric | Value |
|---|---|
| Chunks (entities) | **29,376** |
| Documents (`file_id`) | **221** (including 92 `corpus_minimal`) |
| Cumulative tokens (bge-m3) | 9.03 M |
| Cumulative characters | 36.0 M |

### Distributions

**Characters / chunk** — med **1,300** · mean 1,224 · p25 695 · p75 1,627 · p95 2,134 · max 5,866 · σ 604

**Tokens / chunk** (bge-m3) — med **335** · mean 308 · p75 507 · p95 512 · **max 512** · σ 182
- 512 cap respected at 100% (0 chunks above). ~47% of chunks in 256–511, ~11% exactly at 512.

| Token bin | Chunks | % |
|---|---|---|
| 0–49 | 3,136 | 10.7% |
| 50–99 | 2,922 | 10.0% |
| 100–255 | 6,137 | 20.9% |
| 256–511 | 13,902 | 47.4% |
| 512–767 | 3,261 | 11.1% |
| ≥ 768 | 0 | 0.0% |

**Chunks / document** — med **59** · mean 133 · p25 24 · p75 143 · p95 423 · max 4,175 · σ 308
- 1 single-chunk doc · 39 docs ≥ 200 chunks · strong skew (mean ≫ median).

| Top docs | Chunks |
|---|---|
| PARKER O-Ring Handbook.pdf | 4,175 |
| SSG-26 rev.1_Advisory material.pdf | 977 |
| TS-R-1 amended 2003 English.pdf | 678 |
| TS-R-1 - ST-1 revised 2001 English.pdf | 668 |
| 1996 ST-1 US_OCR.pdf | 574 |

**Pages / document** — med **23** · mean 46 · p75 51 · p95 165 · max 514 · σ 63

**Pages covered / chunk** — med 1 · mean 1.15 · max 10; 14.2% of chunks span ≥ 2 pages.

### Methodology

- **Milvus** is the authoritative source: chunks, documents, characters, chunks/doc.
- **Tokens and pages are not stored in Milvus.** They are joined from the local cache
  `output/all_chunks_cache.jsonl` (29,358 chunks vs 29,376 in Milvus — a gap of
  18 chunks not re-downloaded, negligible for the distributions).

---

## `documents_vectorises_brut` — raw pypdf, no OCR

Pipeline: `scripts/pipeline/brute_corpus.py` in a single pass (download → pypdf extraction of the
native text **without OCR** → sliding windows of **500 bge-m3 tokens, overlap 100** → embedding
→ upsert). The `text` field contains the **raw text without a context header**. Generated on
2026-05-28 via:

```bash
.venv/bin/python scripts/inspect/milvus_recap.py --collection documents_vectorises_brut --retokenize
```

### Overview

| Metric | Value |
|---|---|
| Chunks (entities) | **15,520** |
| Documents (`file_id`) | **219** (including 91 `corpus_minimal`) |
| Cumulative tokens (bge-m3) | 7.62 M |
| Cumulative characters | 27.5 M |

> **223 PDFs processed, 0 errors.** 219 documents are indexed; **4 are empty** (scanned
> PDFs without a native text layer — not recoverable without OCR, hence absent from the
> collection): two scanned attachments (e.g. `attachment_A.pdf`, `attachment_B.pdf`)
> and two scanned source certificates.

### Distributions

**Characters / chunk** — med **1,835** · mean 1,772 · p25 1,585 · p75 1,995 · p95 2,328 · max 5,022 · σ 414
- Chunks are longer and more homogeneous than in the docling collection (fixed-size windows vs semantic boundaries).

**Tokens / chunk** (bge-m3) — med **500** · mean 491 · p25 499 · p75 500 · p95 501 · **max 505** · σ 47
- **98.6% of chunks in 256–511**, a direct consequence of the 500-token windowing. 0 chunks > 512.
- The max of 505 (> 500) comes from the decode→re-encode round trip: the decoded text of a 500-token window can re-tokenize into a few extra tokens. No impact (far below bge-m3's 8192 limit).

| Token bin | Chunks | % |
|---|---|---|
| 0–49 | 1 | 0.0% |
| 50–99 | 60 | 0.4% |
| 100–255 | 158 | 1.0% |
| 256–511 | 15,301 | 98.6% |
| 512–767 | 0 | 0.0% |
| ≥ 768 | 0 | 0.0% |

**Chunks / document** — med **37** · mean 71 · p25 15 · p75 91 · p95 232 · max 1,123 · σ 106
- 0 single-chunk docs · 17 docs ≥ 200 chunks.

| Top docs | Chunks |
|---|---|
| PARKER O-Ring Handbook.pdf | 1,123 |
| SSG-26 rev.1_Advisory material.pdf | 677 |
| SSR-6_Edition2018_Fr.pdf | 270 |
| SSG-33 rev.1_Shedules of provisions.pdf | 268 |
| report_A.pdf | 259 |

**Pages / document** — med **25** · mean 49 · p25 12 · p75 55 · p95 177 · max 520 · σ 65
- Total: 10,923 pages. Here "pages/doc" = total number of pages in the PDF (`len(reader.pages)`), including pages without native text — semantics slightly different from the "pages covered" of the docling collection.

### Methodology

- **Milvus** is the authoritative source: chunks, documents, characters, chunks/doc.
- **Tokens** recounted by re-encoding the Milvus `text` with bge-m3 (`--retokenize`) — no
  local chunk cache for this pipeline.
- **Pages/doc** read from the `output/brute_manifests/{file_id}.json` manifests.

---

## `baseline_fragus` — target 3-vector collection (docling)

**Hybrid retrieval** collection: gathers the three vectors into a single entity per chunk,
with no re-embedding at all. Built by joining `documents_vectorises` (dense) and
`documents_vectorises_fermi` (sparse fermi) on `chunk_id`,
with BM25 computed on the fly. Generated via:

```bash
.venv/bin/python scripts/pipeline/build_fragus_collection.py --corpus baseline --reset
```

### Schema

| Field | Type | Index / metric | Source |
|---|---|---|---|
| `chunk_id` | VARCHAR (PK) | — | join |
| `file_id`, `name`, `path`, `corpus_minimal`, `text` | (same as common schema) | — | `documents_vectorises` |
| `embedding` | FLOAT_VECTOR 1024-d | **FLAT / COSINE** | copied from `documents_vectorises` |
| `sparse_bm25` | SPARSE_FLOAT_VECTOR | SPARSE_INVERTED_INDEX / IP | **computed** (`pymilvus.model`, FR analyzer) |
| `sparse_fermi` | SPARSE_FLOAT_VECTOR | SPARSE_INVERTED_INDEX / IP | copied from `documents_vectorises_fermi` |

### Facts

- **29,358 chunks** = dense ∩ fermi intersection (18 dense chunks — a single doc —
  without fermi, excluded; 0 fermi orphans). Dense index in **FLAT** (exact, no ANN
  noise in the benchmark, cf. `target-pipeline.md`) ≠ the source's IVF_FLAT.
- **BM25**: `BM25EmbeddingFunction` (FR analyzer, `k1=1.5`, `b=0.75`), fit on the
  29,358 texts → vocab **85,057**, `avgdl` **142.5**. State persisted in
  `artifacts/bm25/baseline_fragus.json` (generated by `build_fragus_collection.py`;
  not versioned in the public repo) — **reload it as-is** (`load()` +
  `build_default_analyzer(language="fr")`) to encode queries on the pipeline side,
  otherwise scores are wrong. ⚠️ nltk downloads `punkt_tab`/`stopwords` into `~/nltk_data`
  on first call → pre-provision in SecNumCloud.
- **Native fusion validated**: `hybrid_search` over the 3 fields + `RRFRanker(60)`
  runs on the **2.4.13** server (multi-vector available since 2.4) → no more
  Python-side RRF. A French BM25 query ("colis de transport… type B") brings the
  ASN Guide No. 7 "Packages" documents to the top (lexical sanity check OK).
- The older collections (`documents_vectorises`, `_fermi`) are **kept** for now.
