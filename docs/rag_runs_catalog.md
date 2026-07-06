# RAG evaluation runs catalog — FRAGUS / ASNR

Each run corresponds to one configuration of the `HybridRAGPipeline` pipeline, evaluated on the set of 65 original questions (`data/rag_evaluation_dataset.xlsx`).

Result files follow this naming convention under `scripts/rag_evaluation/data/results/`:
- `eval_test_<run_id>.xlsx` — raw pipeline answers (question, answer, retrieved contexts)
- `score_test_<run_id>_per_question.xlsx` — per-question LLM scores
- `score_test_<run_id>_averages.xlsx` — averages per metric

> **Note**: the raw evaluation files are not published in this repository (they
> contain corpus-derived content). The aggregate tables in this catalog are the
> published record.

---

## Experiment-id naming convention

Pattern: **`{fields}_{ranker}[_top{n}][_comb][_thr{v}]`**

| Segment | Value | Meaning |
|---------|--------|---------------|
| fields | `db` | Dense + BM25 |
| | `df` | Dense + Fermi |
| | `dbf` | Dense + BM25 + Fermi |
| ranker | `w` | WeightedRanker |
| | `rrf` | RRFRanker |
| top-n | `_top25` | top-n=25 (absent = default 10) |
| ranking_mode | `_comb` | combined (absent = reranker only) |
| reranker threshold | `_thr-2` | threshold=-2.0 (absent = no filter) |

`baseline` (Dense+BM25, Weighted, top-10) is the reference run and keeps its name as-is.  
`dense_bm25_25_reranker_01` also keeps its original name (precursor run, not renamed).

---

## Architecture reminder

```
Milvus hybrid_search (multi-vector)
    ↓  fusion: WeightedRanker or RRFRanker
Candidates (top-100)
    ↓  cross-encoder reranker (mmarco-mMiniLMv2-L12-H384-v1, CPU)
Top-N chunks
    ↓  LLM (mistral-large-latest via the Cleyrop proxy)
Answer
```

**Collection**: `baseline_fragus` unless stated otherwise  
Corpus: 29,358 chunks produced by Docling + OCR (HybridChunker, target 512 tokens/chunk)  
3 vector fields available: `embedding` (BGE-M3 dense, COSINE), `sparse_bm25` (client-side BM25, IP), `sparse_fermi` (Fermi-1024 sparse, IP)

---

## Summary scores — all runs

| run | context_recall_llm | global_rag_score | faithfulness | factual_correctness_recall | citation_recall_files | paraphrase_robustness | diversity |
|-----|:-----------------:|:---------------:|:------------:|:--------------------------:|:--------------------:|:---------------------:|:---------:|
| baseline | 0.385 | 0.514 | 0.881 | 0.449 | 0.267 | 0.877 | 0.267 |
| baseline_raw | 0.417 | 0.485 | 0.775 | 0.457 | 0.183 | 0.874 | 0.333 |
| dbf_w | 0.403 | 0.506 | 0.835 | 0.457 | 0.242 | 0.882 | 0.261 |
| df_w | 0.424 | 0.509 | 0.860 | 0.455 | 0.215 | 0.874 | 0.256 |
| dense_bm25_25_reranker_01 | 0.501 | 0.536 | 0.804 | 0.484 | 0.267 | 0.885 | 0.461 |
| db_rrf | 0.446 | 0.531 | 0.892 | 0.442 | 0.280 | 0.883 | 0.259 |
| df_rrf | 0.441 | 0.524 | 0.885 | 0.434 | 0.273 | 0.880 | 0.256 |
| dbf_rrf | 0.419 | 0.532 | 0.890 | 0.463 | 0.286 | 0.883 | 0.259 |
| db_w_top25 | 0.482 | 0.556 | 0.862 | 0.509 | 0.281 | 0.882 | 0.461 |
| db_w_top50 | 0.534 | 0.548 | 0.780 | 0.496 | 0.307 | 0.885 | 0.461 |
| db_w_top75 | 0.510 | 0.545 | 0.731 | 0.539 | 0.309 | 0.888 | 0.461 |
| db_w_top100 | 0.505 | 0.551 | 0.728 | 0.569 | 0.290 | 0.882 | 0.461 |
| dbf_rrf_top25 | 0.534 | 0.574 | 0.871 | 0.496 | 0.331 | 0.889 | 0.455 |
| dbf_rrf_top25_thr-3 | 0.513 | 0.569 | 0.872 | 0.508 | 0.323 | 0.888 | 0.442 |
| dbf_rrf_top25_thr-2 | 0.506 | 0.571 | 0.849 | 0.528 | 0.317 | 0.889 | 0.438 |
| dbf_rrf_top25_thr-1.5 | 0.468 | 0.556 | 0.851 | 0.509 | 0.322 | 0.886 | 0.429 |
| dbf_rrf_top25_thr-1 | 0.513 | 0.567 | 0.858 | 0.509 | 0.305 | 0.882 | 0.415 |
| dbf_rrf_top50 | 0.549 | 0.569 | 0.746 | 0.544 | 0.365 | 0.888 | 0.456 |
| dbf_rrf_top100 | 0.522 | 0.555 | 0.676 | 0.578 | 0.353 | 0.885 | 0.454 |
| **dbf_rrf_top25_comb** | **0.580** | **0.592** | **0.857** | **0.535** | **0.303** | **0.887** | **0.462** |

`global_rag_score` = 0.25 × context_recall_llm + 0.35 × factual_correctness_recall + 0.15 × citation_recall_files + 0.25 × faithfulness

---

## Per-run details

---

### `baseline`

**Goal**: reference configuration, dense + BM25.

| Parameter | Value |
|-----------|--------|
| Collection | `baseline_fragus` |
| Ranker | `WeightedRanker` |
| Fields | `embedding` (w=0.5, COSINE) · `sparse_bm25` (w=0.5, IP) |
| Candidates | 100 |
| Top-N | 10 |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` · CPU · max_length=512 |
| ranking_mode | `reranker` |
| LLM | `mistral-large-latest` · max_tokens=1024 |
| Questions | 65 originals + 3 variants/question (195 variants) |

**Command**:
```bash
python prepare_eval_dataset.py data/rag_evaluation_dataset.xlsx \
  --experiment-id baseline \
  --dense-weight 0.5 --bm25-weight 0.5
```

---

### `baseline_raw`

**Goal**: same config as `baseline` but on the raw pypdf corpus (no OCR/Docling), to isolate the effect of parsing.

| Parameter | Value |
|-----------|--------|
| Collection | `brut_fragus` (pypdf corpus, 15,520 chunks) |
| Ranker | `WeightedRanker` |
| Fields | `embedding` (w=0.5, COSINE) · `sparse_bm25` (w=0.5, IP) |
| Candidates | 100 |
| Top-N | 10 |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| LLM | `mistral-large-latest` |

**Observations**: context_recall similar to the baseline (+3.2 pts) but lower global_rag_score (-2.9 pts), penalized by weaker faithfulness (0.775 vs 0.881) and a collapsed citation_recall (0.183 vs 0.267). The pypdf corpus recovers the text but loses table structure → the model generates less well-grounded answers.

---

### `dbf_w`

**Goal**: test the Fermi-1024 encoder as an additional signal alongside Dense and BM25.

| Parameter | Value |
|-----------|--------|
| Collection | `baseline_fragus` |
| Ranker | `WeightedRanker` |
| Fields | `embedding` (w=0.33, COSINE) · `sparse_bm25` (w=0.33, IP) · `sparse_fermi` (w=0.33, IP) |
| Candidates | 100 |
| Top-N | 10 |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| LLM | `mistral-large-latest` |

> ⚠️ **Command not tracked**: run launched before the `--ranker` / `--fermi-weight` arguments were added to `prepare_eval_dataset.py`. Config confirmed after the fact by the user.

**Observations**: marginal improvement in context_recall (+1.9 pts vs baseline) but a lower global_rag_score (-0.8 pts). The post-mortem analysis shows that WeightedRanker applies an arctan normalization that compresses Fermi's IP scores towards 1.0, destroying their discriminative power — hence the disappointing performance.

---

### `df_w`

**Goal**: Dense 50% + Fermi 50% without BM25, WeightedRanker — check whether Fermi can replace BM25.

| Parameter | Value |
|-----------|--------|
| Collection | `baseline_fragus` |
| Ranker | `WeightedRanker` |
| Fields | `embedding` (w=0.5, COSINE) · `sparse_fermi` (w=0.5, IP) |
| Candidates | 100 |
| Top-N | 10 |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| LLM | `mistral-large-latest` |

**Command**:
```bash
python prepare_eval_dataset.py data/rag_evaluation_dataset.xlsx \
  --experiment-id df_w \
  --dense-weight 0.5 --bm25-weight 0.0 --fermi-weight 0.5
```

**Observations**: +3.9 pts vs baseline on context_recall but WeightedRanker penalizes Fermi (same arctan issue as `dbf_w`). The diversity of retrieved chunks (0.256) is the lowest — a sign that Dense+Fermi without BM25 brings back more redundant chunks.

---

### `dense_bm25_25_reranker_01`

**Goal**: Dense+BM25 baseline with top_n=25 (instead of 10) to isolate the effect of the number of chunks passed to the LLM.

| Parameter | Value |
|-----------|--------|
| Collection | `baseline_fragus` |
| Ranker | `WeightedRanker` |
| Fields | `embedding` (w=0.5, COSINE) · `sparse_bm25` (w=0.5, IP) |
| Candidates | 100 |
| Top-N | 25 |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| LLM | `mistral-large-latest` |

**Command**:
```bash
python prepare_eval_dataset.py data/rag_evaluation_dataset.xlsx \
  --experiment-id dense_bm25_25_reranker_01 \
  --dense-weight 0.5 --bm25-weight 0.5 --top-n 25
```

**Observations**: +11.6 pts vs baseline on context_recall. Chunk diversity (0.461) is the highest of all weighted runs. Going from 10 to 25 chunks massively improves recall — the reranker selects better with more candidates exposed to the LLM.

---

### `db_rrf`

**Goal**: Dense+BM25 baseline with RRFRanker instead of WeightedRanker — isolate the effect of the ranker without Fermi.

| Parameter | Value |
|-----------|--------|
| Collection | `baseline_fragus` |
| Ranker | `RRFRanker` (k=60) |
| Fields | `embedding` (COSINE) · `sparse_bm25` (IP) |
| Candidates | 100 |
| Top-N | 10 |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` · CPU |
| ranking_mode | `reranker` |
| LLM | `mistral-large-latest` · max_tokens=1024 |

> ⚠️ **Config partially uncertain**: run launched manually, command not tracked. Top-N confirmed at 10 by counting in the result files.

**Observations**: RRF on Dense+BM25 (without Fermi) brings +6.1 pts of context_recall vs the WeightedRanker baseline (0.446 vs 0.385), but global_rag_score capped at 0.531 — the recall gain is partly absorbed by a slight degradation of factual_correctness. Adding Fermi (→ `dbf_rrf_top25`) is needed to reach the next level.

---

### `df_rrf`

**Goal**: Dense+Fermi with RRFRanker, without BM25.

| Parameter | Value |
|-----------|--------|
| Collection | `baseline_fragus` |
| Ranker | `RRFRanker` (k=60) |
| Fields | `embedding` (COSINE) · `sparse_fermi` (IP) |
| Candidates | 100 |
| Top-N | 10 |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` · CPU |
| ranking_mode | `reranker` |
| LLM | `mistral-large-latest` · max_tokens=1024 |

> ⚠️ **Config partially uncertain**: run launched manually, command not tracked. Top-N confirmed at 10 by counting in the result files.

**Observations**: slightly below `db_rrf` (0.524 vs 0.531 global_rag_score) — Fermi alone without BM25 does not outperform BM25 alone with RRF. The combination of the three fields (`dbf_rrf_top25`) is needed.

---

### `dbf_rrf`

**Goal**: 3 fields (Dense + BM25 + Fermi) with RRFRanker, top-n=10 — top-10 version of `dbf_rrf_top25`.

| Parameter | Value |
|-----------|--------|
| Collection | `baseline_fragus` |
| Ranker | `RRFRanker` (k=60) |
| Fields | `embedding` (COSINE, nprobe=16) · `sparse_bm25` (IP) · `sparse_fermi` (IP) |
| Candidates | 100 |
| Top-N | 10 |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` · CPU · max_length=512 |
| ranking_mode | `reranker` |
| LLM | `mistral-large-latest` · max_tokens=1024 |
| Questions | 65 originals + 3 variants/question (195 variants) |

**Command**:
```bash
uv run scripts/rag_evaluation/prepare_eval_dataset.py \
  scripts/rag_evaluation/data/rag_evaluation_dataset.xlsx \
  --experiment-id dbf_rrf \
  --ranker rrf \
  --fermi-weight 0.33 \
  --top-n 10
```

**Observations**:
- global_rag_score: **0.532** · context_recall_llm: 0.419 · faithfulness: 0.890
- factual_correctness_recall: 0.463 · citation_recall_files: 0.286
- paraphrase_robustness: 0.883 · diversity: 0.259

---

### `dbf_rrf_top*` series

**Goal**: isolate the effect of top-n on the 3-field + RRF config — same config as `dbf_rrf_top25` with top_n varying from 10 to 100.

| Parameter | Value |
|-----------|--------|
| Collection | `baseline_fragus` |
| Ranker | `RRFRanker` (k=60) |
| Fields | `embedding` (COSINE, nprobe=16) · `sparse_bm25` (IP) · `sparse_fermi` (IP) |
| Candidates | 100 |
| Top-N | **10 / 25 / 50 / 100** depending on the run |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` · CPU · max_length=512 |
| ranking_mode | `reranker` |
| LLM | `mistral-large-latest` · max_tokens=1024 |
| Questions | 65 originals + 3 variants/question (195 variants) |

**Commands**:
```bash
uv run scripts/rag_evaluation/prepare_eval_dataset.py \
  scripts/rag_evaluation/data/rag_evaluation_dataset.xlsx \
  --experiment-id dbf_rrf_top50 --ranker rrf --fermi-weight 0.33 --top-n 50
# same with --experiment-id dbf_rrf_top100 --top-n 100
```

**Observations**: `dbf_rrf_top25` is the best compromise (global=0.574). Beyond that, faithfulness collapses (0.871 → 0.746 → 0.676) and cancels out the recall gains. `diversity` plateaus from top-25 onwards (0.455), a sign that the extra chunks bring no complementary signal, only noise for the LLM.

| run | ctx_recall | global | faithfulness | fcr | cit_recall |
|---|:---:|:---:|:---:|:---:|:---:|
| `dbf_rrf` (top-10) | 0.419 | 0.532 | **0.890** | 0.463 | 0.286 |
| `dbf_rrf_top25` | 0.534 | **0.574** | 0.871 | 0.496 | 0.331 |
| `dbf_rrf_top50` | **0.549** | 0.569 | 0.746 | 0.544 | **0.365** |
| `dbf_rrf_top100` | 0.522 | 0.555 | 0.676 | **0.578** | 0.353 |

---

### `db_w_top*` series

**Goal**: isolate the effect of the number of chunks passed to the LLM — baseline config (Dense+BM25, WeightedRanker) with top_n varying from 25 to 100.

| Parameter | Value |
|-----------|--------|
| Collection | `baseline_fragus` |
| Ranker | `WeightedRanker` |
| Fields | `embedding` (w=0.5, COSINE) · `sparse_bm25` (w=0.5, IP) |
| Candidates | 100 |
| Top-N | **25 / 50 / 75 / 100** depending on the run |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` · CPU |
| ranking_mode | `reranker` |
| LLM | `mistral-large-latest` · max_tokens=1024 |
| Questions | 65 originals + 3 variants/question (195 variants) |

**Commands**:
```bash
uv run scripts/rag_evaluation/prepare_eval_dataset.py \
  scripts/rag_evaluation/data/rag_evaluation_dataset.xlsx \
  --experiment-id db_w_top25 --top-n 25
# same with --experiment-id db_w_top50 --top-n 50 / db_w_top75 --top-n 75 / db_w_top100 --top-n 100
```

**Observations**: `global_rag_score` plateaus quickly — the context_recall gain with more chunks is offset by a drop in faithfulness (0.862 → 0.728). `diversity` is identical across all runs (0.461), a sign that the bottleneck is upstream (vector fields, not top_n). `db_w_top25` is the best recall/faithfulness compromise of the series.

---

### `dbf_rrf_top25`

**Goal**: 3 vector fields (Dense + BM25 + Fermi) with RRFRanker — validated exhaustive config, top_n=25.

| Parameter | Value |
|-----------|--------|
| Collection | `baseline_fragus` |
| Ranker | `RRFRanker` (k=60) |
| Fields | `embedding` (COSINE, nprobe=16) · `sparse_bm25` (IP) · `sparse_fermi` (IP) |
| Candidates | 100 |
| Top-N | 25 |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` · CPU · max_length=512 |
| ranking_mode | `reranker` |
| LLM | `mistral-large-latest` · max_tokens=1024 |
| Questions | 65 originals + 3 variants/question (195 variants) |
| Question embeddings | 100% cache hits (`query_embeddings_cache.pkl`) — no Fermi inference at query time |

> Note: with RRFRanker, the `--fermi-weight` and `--dense-weight` weights do not influence the fusion (fusion is rank-based). Only the **presence** of the field in `search_config` matters; `--fermi-weight 0.33` is only used to activate the field.

**Command**:
```bash
python prepare_eval_dataset.py data/rag_evaluation_dataset.xlsx \
  --experiment-id dbf_rrf_top25 \
  --ranker rrf --fermi-weight 0.33 --top-n 25
# dense-weight=0.5, bm25-weight=0.5 by default → all 3 fields active
```

**Observations**:
- **+14.9 pts** on `context_recall_llm` vs baseline (0.534 vs 0.385)
- **+6.0 pts** on `global_rag_score` vs baseline (0.574 vs 0.514)
- `diversity` = 0.455: the 3 signal sources bring back complementary chunks
- `faithfulness` = 0.871: the LLM stays well grounded despite more context

**Why RRF > WeightedRanker with Fermi**: WeightedRanker normalizes Milvus scores with arctan before weighting them. Fermi's IP scores have a very different distribution from BM25 or dense cosine scores — the arctan normalization compresses them towards 1.0 and destroys their discriminative power. RRF is insensitive to score distributions (rank-based fusion): the 3 sources contribute on equal footing regardless of their score scales.

---

### `dbf_rrf_top25_comb` ⭐ best run

**Goal**: same config as `dbf_rrf_top25`, but final `combined` ranking — the top-25 is sorted by the 50/50 average of the min-max-normalized scores (Milvus RRF score + reranker logit) instead of the reranker logit alone. No threshold.

| Parameter | Value |
|-----------|--------|
| Collection | `baseline_fragus` |
| Ranker | `RRFRanker` (k=60) |
| Fields | `embedding` (COSINE, nprobe=16) · `sparse_bm25` (IP) · `sparse_fermi` (IP) |
| Candidates | 100 |
| Top-N | 25 |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` · CPU · max_length=512 |
| ranking_mode | `combined` (per-query min-max over the 100-candidate pool, 50/50 average) |
| reranker_threshold | none |
| LLM | `mistral-large-latest` · max_tokens=1024 |
| Questions | 65 originals + 3 variants/question (195 variants) |
| Question embeddings | 100% cache hits (`query_embeddings_cache.pkl`) |

**Command**:
```bash
python prepare_eval_dataset.py data/rag_evaluation_dataset.xlsx \
  --experiment-id dbf_rrf_top25_comb \
  --ranker rrf --fermi-weight 0.33 --top-n 25 \
  --ranking-mode combined
```

**Observations**:
- **No significant difference** vs `dbf_rrf_top25` (Wilcoxon + BH correction: 0/17 metrics, all p_adj > 0.77; see `comparison_dbf_rrf_top25_vs_dbf_rrf_top25_comb.xlsx`)
- Trends (not significant): combined shifts the balance towards **recall** — `context_recall_llm` +4.7 pts (0.580), `factual_correctness_recall` +3.9 pts, `grounded_rate` +5.0 pts, `hallucination_rate` −2.6 pts — at the cost of **precision**: `context_precision_llm` −5.1 pts (0.358), `hub_france_ia_fscore` −3.9 pts, `citation_recall_files` −2.8 pts, `faithfulness` −1.4 pts
- `global_rag_score` 0.592 (+1.8 pts, not significant) — best absolute score in the catalog, but to be interpreted with caution
- Reading: reinjecting the Milvus signal into the ordering favours high-coverage chunks that the reranker alone would have demoted, but lets more noise rise back up — exactly what a threshold on the reranker logit is supposed to correct. The combined + threshold combo (`dbf_rrf_top25_comb_thr*` series) is the next test.

---

### `dbf_rrf_top25_thr*` series

**Goal**: evaluate the effect of a minimum threshold on the reranker score — `dbf_rrf_top25` config with a post-reranking filter at different values.

| Parameter | Value |
|-----------|--------|
| Collection | `baseline_fragus` |
| Ranker | `RRFRanker` (k=60) |
| Fields | `embedding` (COSINE, nprobe=16) · `sparse_bm25` (IP) · `sparse_fermi` (IP) |
| Candidates | 100 |
| Top-N | 25 (before filtering) |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` · CPU · max_length=512 |
| ranking_mode | `reranker` |
| reranker_threshold | **−3.0 / −2.0 / −1.5 / −1.0** depending on the run |
| LLM | `mistral-large-latest` · max_tokens=1024 |

> **Note on the order of operations**: the threshold is applied **after** the top-n cut. The pipeline sorts the 100 candidates by `rerank_score`, cuts at 25, then filters out those below the threshold. The threshold therefore always operates on pure cross-encoder scores (even in `combined` mode).

**Commands**:
```bash
uv run scripts/rag_evaluation/prepare_eval_dataset.py \
  scripts/rag_evaluation/data/rag_evaluation_dataset.xlsx \
  --experiment-id dbf_rrf_top25_thr-3 --ranker rrf --fermi-weight 0.33 --top-n 25 --reranker-threshold -3.0
# same with --reranker-threshold -2.0 / -1.5 / -1.0
```

**Observations**: no threshold beats the unfiltered baseline on `global_rag_score`. The filter improves `factual_correctness_recall` (up to +3.2 pts at thr-2) but degrades `context_recall_llm` and `hub_france_ia_fscore` — the net balance is negative. The reranker score distribution for this config (median −0.83, 70% negative) makes any threshold ≥ 0 unusable (two questions have all their chunks in the negatives).

| run | ctx_recall | global | faithfulness | fcr | cit_recall | chunks/q |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `dbf_rrf_top25` (none) | **0.534** | **0.574** | **0.871** | 0.496 | **0.331** | 20.4 |
| `thr-3` | 0.513 | 0.569 | 0.872 | 0.508 | 0.323 | 19.9 |
| `thr-2` | 0.506 | 0.571 | 0.849 | **0.528** | 0.317 | 19.7 |
| `thr-1.5` | 0.468 | 0.556 | 0.851 | 0.509 | 0.322 | 19.4 |
| `thr-1` | 0.513 | 0.567 | 0.858 | 0.509 | 0.305 | 18.4 |

---

## Reproduction command

```bash
# Generate the pipeline answers
uv run scripts/rag_evaluation/prepare_eval_dataset.py \
  scripts/rag_evaluation/data/rag_evaluation_dataset.xlsx \
  --experiment-id <run_id> \
  [--ranker rrf] \
  [--fermi-weight 0.33] \
  [--dense-weight 0.5] \
  [--bm25-weight 0.5] \
  [--top-n 25]

# Evaluate
uv run scripts/rag_evaluation/rag_evaluation.py \
  --experiment-id <run_id>
```
