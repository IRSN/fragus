# Target RAG pipeline — retrieval, fusion, rerank

Scoping note for the retrieval architecture targeted by the study (ASNR corpus,
French, on-prem / SecNumCloud). The reranking part is detailed in `reranking.md`.

## Diagram

```
                      retrieve              fuse              rerank          generate
  bge-m3 (dense)  ──top-100──┐
  BM25 (lexical)  ──top-100──┼──► RRF fusion ──top-N──► reranker ──top-k──► LLM
  fermi (sparse)  ──top-100──┘     (union)                                  (context)
                  └─── ① ───┘       └── ② ──┘           └── ③ ──┘        └── ④ ──┘
```

Three complementary retrievers → **rank-based** fusion (RRF) → neural re-scoring
(cross-encoder) → final context for the LLM.

## The three retrievers

| Component | Type | Language | Contribution |
|---|---|---|---|
| **bge-m3 (dense)** | dense semantic | multilingual / native FR | main baseline, robust to paraphrase |
| **BM25** | lexical statistical | language-agnostic | exact terms, regulatory references, codes |
| **fermi** (`atomic-canyon/fermi-1024`) | learned sparse | **EN**, nuclear domain | tests the *strong domain / weak language* tradeoff |

**Embedding stack locked in: dense bge-m3 + fermi + BM25 — nothing more,
nothing less.** In particular, the **lexical sparse output of bge-m3**
(`lexical_weights`) is **not** retained: it is not exposed by the OVH / AI-gen
proxy embedding API (OpenAI-style `/embeddings` format → dense only), so getting
it would require standing up a dedicated FlagEmbedding inference server and
**re-embedding the entire corpus** — **out of scope**. The lexical leg is
covered by **BM25** (see *Feasibility status* below).

> **Target: a single collection per corpus, with 3 vector fields.**
> `baseline_fragus` (docling) and `brut_fragus` (pypdf), each carrying `embedding`
> (dense bge-m3), `sparse_bm25` and `sparse_fermi`. The three vectors then share
> the **same set of chunks / same `chunk_id`** by construction → `hybrid_search`
> + `RRFRanker` fusion **native to Milvus** across all 3 (multi-vector available
> since 2.4, so available now). No more Python-side RRF.

## Feasibility status (Milvus versions & availability)

> **`baseline_fragus` (docling) is built** — 29,358 chunks, 3 vectors,
> `hybrid_search` + `RRFRanker` native fusion validated on the 2.4.13 server.
> Script: `scripts/pipeline/build_fragus_collection.py --corpus baseline --reset`.
> Remaining: `brut_fragus` (prerequisite: fermi on the raw corpus).

| Retriever | Existing | How it lands in the target collection |
|---|---|---|
| **dense bge-m3** | ✅ `documents_vectorises` (+ `_brut`) | copied as-is (no re-embedding) — `embedding` field, **FLAT/COSINE** index |
| **fermi** (sparse) | ✅ docling `documents_vectorises_fermi`; ❌ **raw corpus still to compute** | docling: copied → `sparse_fermi`; raw: run fermi on the 15,520 chunks (Mac/MPS server) |
| **BM25** (sparse) | ✅ **computed** (baseline) | `pymilvus.model` `BM25EmbeddingFunction`, **FR** analyzer (`k1=1.5`, `b=0.75`) → `sparse_bm25`. Vocab 85,057, `avgdl` 142.5; fit+encode ≈ 2 min on the VM |

- **We do not wait for Milvus native BM25.** BM25 is an **algorithm**, not a
  model: `pymilvus.model.sparse.BM25EmbeddingFunction` computes it client-side
  (fit = `df`/`avgdl` counting over the corpus, sparse output) and we store it
  as an ordinary vector field. Lightweight (single CPU core, ~MB of RAM, no GPU),
  aligned with what the native feature will do later.
- **Multi-vector + `hybrid_search` + `RRFRanker`**: available **since 2.4** → with
  the 3 vectors in **a single collection**, fusing all 3 (dense + bm25 +
  fermi) is **native Milvus, today**. No more Python-side RRF.
- **Query-time reproducibility = index-time:** the BM25 state frozen at fit time
  (vocab + idf + `avgdl`) is **persisted** in `artifacts/bm25/{collection}.json`
  (`save()`) and must be reloaded as-is (`load()` +
  `build_default_analyzer(language="fr")`) to encode queries — same analyzer,
  same idf, otherwise scores are wrong. **To be shared with the query pipeline.**
- **Dependencies:** `pymilvus-model` + `nltk` (pinned in the lock file). ⚠️ the FR
  analyzer downloads nltk data (`punkt_tab`, `stopwords`) into `~/nltk_data` on
  first call → must be **pre-provisioned** in a closed SecNumCloud environment.
- **Native BM25 (`Function`/`FunctionType.BM25`)** remains gated **≥ 2.5** (current
  server 2.4.13 → 2.6.15 with release **4.5.3**). It is a **future convenience**
  (auto-population at insert time from `text`), not a prerequisite. Migrating to
  the native feature = **rebuild** (internal analyzer ≠ ours, not bit-compatible).

## Fusion: RRF, no weighted average

The scores of the three methods live on **incompatible scales** (cosine
∈ [−1,1], BM25 ∈ [0,∞[, sparse inner-product ∈ [0,∞[). Averaging them — even
normalized — is fragile (the trap of the external production pipeline's
`0.7·a + 0.3·b`).

**Chosen default: RRF (Reciprocal Rank Fusion)** — fuses on *ranks*, giving
scale invariance for free:

```
RRF(d) = Σ_i  1 / (k + rank_i(d))        k ≈ 60
```

Natively supported by Milvus (`RRFRanker`) across multiple vector fields of a
single collection — available since 2.4 (our server runs 2.4.13). Since we bring
the **3 vectors (dense + `sparse_bm25` + `sparse_fermi`) together in a single
collection** (`baseline_fragus` / `brut_fragus`), fusing all 3 is **native
Milvus, today** — no more Python-side RRF. Score weighting (`WeightedRanker`)
remains a **variant to test**, not the default.

## Reranking

The fused top-N is re-scored by a cross-encoder, then the final top-k is kept.
Main reranker: **cross-encoder/mmarco-mMiniLMv2-L12-H384-v1** (see `reranking.md`
for the comparison triplet and libraries).

## Parameters to sweep

### Depths (keep them distinct — a single "top-k" is ambiguous)

| # | Parameter | Definition | Range to sweep | What it drives |
|---|---|---|---|---|
| ① | `retrieval_depth` | chunks brought back by **each** retriever | {20, 50, **100**} | **recall ceiling** (recall@k) |
| ② | `fusion_depth` | fused candidates kept for the rerank | {30, **40**, 50} | rerank **cost** |
| ③ | `top_k` | final chunks after rerank, injected into the LLM | {**5**, 10, 20} | context **precision / faithfulness** |

> Without a reranker, ① and ③ collapse into one. With a reranker, they are two
> distinct levers. Beyond ~50 chunks in context: dilution / *lost-in-the-middle*
> → degrading faithfulness.

### Retrieval composition (which components, which fusion)

| Axis | Values to compare |
|---|---|
| Active method(s) | dense only · BM25 only · fermi only · dense+BM25 · dense+fermi · **dense+BM25+fermi** |
| Fusion strategy | **RRF** (default) · WeightedRanker (variant) |
| RRF parameter | `k` ≈ 60 (low sensitivity, fixed except for a dedicated test) |

### Reranking

| Axis | Values |
|---|---|
| Reranker ON / OFF | both |
| Model | **mmarco-mMiniLMv2-L12-H384-v1** · gte-multilingual-base · ms-marco-MiniLM (EN control) |

### Upstream (impact all of retrieval)

| Axis | Values |
|---|---|
| Chunking | docling structural · raw pypdf |
| Chunk size | 512 (bge-m3 recommendation) · 500 (brute_corpus) · 2500 (external baseline) |

## Fixed parameters (out of scope for the study)

- Dense embedding model: **bge-m3** (1024 dims)
- Similarity: **cosine**
- Milvus index: **FLAT** (exact, no ANN noise in the benchmark)
- Generation LLM: **Mistral Large** (single — the study focuses on retrieval)
- Corpus, annotated test set, LLM judge: held constant

## Evaluation (reminder)

- **Sweep** (retrieval): **deterministic** IR metrics on annotated chunks —
  `recall@k`, `nDCG@k`, `MRR`. Fast, no LLM.
- **Short-list** (3–4 configs): RAGAS (faithfulness, factual correctness) +
  stability ×10 + expert review.

## Key takeaways

- Embedding stack **locked in: dense bge-m3 + fermi + BM25** (no bge-m3 lexical
  sparse, out of scope). 3 retrievers → **RRF fusion** → mmarco-mMiniLMv2
  rerank → top-k to the LLM.
- **Target: 1 collection per corpus with 3 vectors** (`baseline_fragus`,
  `brut_fragus`) → **native** `hybrid_search` + `RRFRanker` across all 3 (since 2.4).
- **BM25 computed ourselves** via `pymilvus.model` (FR analyzer), stored as
  `SPARSE_FLOAT_VECTOR` — **available now**, we don't wait for the native feature
  (≥2.5 / 2.6.15). BM25 state (vocab+idf) must be **persisted** so queries are
  encoded identically.
- **Never** use a weighted average of raw scores as the default.
- Three distinct depths: `retrieval_depth` / `fusion_depth` / `top_k`.
- **`baseline_fragus` built** (29,358, 3 vectors, native RRF validated). Remaining:
  fermi on the **raw corpus still to compute** (15,520, Mac/MPS server) then `brut_fragus`.
