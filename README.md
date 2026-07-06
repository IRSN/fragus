# FRAGUS

Study of a **RAG retrieval** chain over the ASNR corpus (nuclear-safety
documents, mostly in French, on-prem / SecNumCloud). The project ingests the
corpus (PDF → chunks → embeddings → Milvus) and compares several
retrieval / fusion / reranking strategies.

Target pipeline: 3 retrievers (dense **bge-m3** / lexical **BM25** / sparse
**fermi**) → **RRF** fusion → **mmarco-mMiniLMv2-L12-H384-v1** rerank → LLM context.
See [`docs/target-pipeline.md`](docs/target-pipeline.md).

> **Note on this public repository** — this is a public snapshot of work carried
> out on an internal platform (Cleyrop / SecNumCloud). The corpus, the evaluation
> dataset and the raw evaluation results are not published: they derive from
> internal documents. Aggregate evaluation scores are available in
> [`docs/rag_runs_catalog.md`](docs/rag_runs_catalog.md).

## Getting started

See [`GETTING_STARTED.md`](GETTING_STARTED.md) (Cleyrop CLI, login, `uv sync`).
Check connectivity:

```bash
uv run scripts/ops/test_milvus.py     # Milvus connection
uv run scripts/ops/test_llm.py        # ai-gen LLM proxy
uv run scripts/ops/browse.py          # Cleyrop project tree
```

## Configuration

- `config.toml` — non-secret parameters (versioned): Milvus, LLM proxy,
  bge-m3 / fermi embeddings, raw chunking.
- `configs/chunker.yaml` — docling-serve + HybridChunker options.
- `.env` — secrets (`CLEYROP_*`, `DOCLING_SERVICE_URL`, …), not versioned.

## Ingestion pipeline

Docling chain (collection `documents_vectorises`):

```bash
uv run scripts/pipeline/build_manifest_fragus_clean.py   # 1. corpus manifest
uv run scripts/pipeline/convert_corpus.py                # 2. PDF → DoclingDocument JSON
uv run scripts/pipeline/chunk_corpus.py                  # 3. JSON → JSONL chunks (HybridChunker)
uv run scripts/pipeline/embed_corpus.py                  # 4. chunks → bge-m3 embeddings → Milvus
```

Variants:
- `scripts/pipeline/brute_corpus.py` — "raw" pypdf chain without OCR or docling
  (collection `documents_vectorises_brut`). See [`docs/milvus-collection.md`](docs/milvus-collection.md).
- `scripts/pipeline/embed_fermi.py` — **sparse** fermi-1024 embeddings (local
  inference) on the docling chunks (collection `documents_vectorises_fermi`). See
  [`docs/collection-fermi.md`](docs/collection-fermi.md).

The dedicated fermi inference server lives in [`fermi_server/`](fermi_server/).

## RAG pipeline (answer generation)

`scripts/rag/hybrid_rag_pipeline.py` exposes `HybridRAGPipeline`: Milvus hybrid
search (dense bge-m3 + BM25, optional sparse fermi) → fusion
(`WeightedRanker` or `RRFRanker`) → cross-encoder rerank → LLM generation via
the Cleyrop proxy.

```python
from scripts.rag.hybrid_rag_pipeline import HybridRAGPipeline

rag = HybridRAGPipeline()
result = rag.ask("Quels sont les critères d'agrément des colis de type B ?")
print(result["answer"])
```

Notable `HybridRAGPipeline` parameters:

| Parameter | Default | Role |
|---|---|---|
| `ranker` | `"weighted"` | Milvus fusion: `"weighted"` (WeightedRanker) or `"rrf"` (Reciprocal Rank Fusion) |
| `rrf_k` | `60` | RRF smoothing constant (only when `ranker="rrf"`) |
| `candidates` | `100` | Chunks retrieved before reranking |
| `top_n` | `10` | Chunks kept after reranking and passed to the LLM |
| `ranking_mode` | `"reranker"` | Final ranking: `"reranker"` (cross-encoder score only) or `"combined"` (normalized average of Milvus + reranker scores) |
| `reranker_threshold` | `None` | Minimum reranker score to include a chunk (absent = keep the full top-n) |
| `reranker_url` | `None` | URL of a remote reranking server (env `RERANKER_URL`). If absent, the model runs locally on CPU |

Configuration: [`scripts/rag/config.toml`](scripts/rag/config.toml) (Milvus,
LLM proxy, bge-m3 embedding).

The sparse fermi retriever is disabled by default (dense/BM25 weights 50/50).
To enable it, start [`fermi_server/`](fermi_server/) and set
`encoder_url` in the `[embedding_fermi]` section of the root `config.toml`.

## End-to-end RAG evaluation

`scripts/rag_evaluation/` runs an evaluation campaign in 2 steps,
linked by a shared `EXPERIMENT_ID`:

```bash
uv sync --extra eval   # dedicated dependencies: ragas, matplotlib, ...
cd scripts/rag_evaluation

# 1. Generate the RAG answers on the standard question set
uv run python prepare_eval_dataset.py data/rag_evaluation_dataset.xlsx \
  --experiment-id demo_01
# → data/results/eval_test_demo_01.xlsx

# 2. Compute the scores (RAGAS + custom metrics) on that file
uv run python rag_evaluation.py --experiment-id demo_01
# → data/results/score_test_demo_01_per_question.xlsx
# → data/results/score_test_demo_01_averages.xlsx
```

Both scripts resolve the judge LLM and the embeddings through the Cleyrop
proxy — no API key to configure. The expected evaluation dataset format is
documented in [`scripts/rag_evaluation/data/README.md`](scripts/rag_evaluation/data/README.md).

### Computed scores

`rag_evaluation.py` produces the following indicators, organized in three families:

**Answer quality**

| Score | Method | Description |
|---|---|---|
| `factual_correctness_recall` | RAGAS (internal NLI) | Fraction of the expected facts actually present in the answer |
| `faithfulness` | RAGAS | Fraction of the answer's claims supported by the retrieved context (anti-hallucination) |
| `paraphrase_robustness` | Embeddings (cosine) | Average semantic similarity of the answers to rephrasings of the same question — measures pipeline stability |
| `hallucination_rate` | LLM (factual analysis) | Fraction of the answer's facts absent from both the context AND the reference |

**Retrieved-context quality**

| Score | Method | Description |
|---|---|---|
| `context_recall_llm` | RAGAS | Fraction of the reference answer's facts covered by the retrieved chunks |
| `context_precision_llm` | RAGAS (rank-aware AP) | Rank-weighted Average Precision of the retrieved chunks against the reference |
| `hub_france_ia_fscore` | F-β (β=0.5) over the two above | Hub France IA score: sufficient context without excess (β<1 penalizes over-retrieval) |
| `context_relevance_files` | Files | File precision: `\|retrieved ∩ expected\| / \|retrieved\|` |
| `context_coverage_files` | Files | File recall: `\|retrieved ∩ expected\| / \|expected\|` |
| `diversity` | Embeddings (cosine) | `1 − average cosine similarity` of the retrieved chunks — measures context complementarity |

**Citations**

| Score | Method | Description |
|---|---|---|
| `citation_recall_files` | Files | Fraction of the expected sources actually cited in the answer |
| `citation_precision_files` | Files | Fraction of the cited sources that belong to the expected sources |
| `citation_fscore_files` | F-1 over the two above | — |

**Global score**

Weighted aggregation of 4 components (renormalized if a component is missing):

```
global_rag_score = 0.25 × context_recall_llm
                 + 0.35 × factual_correctness_recall
                 + 0.15 × citation_recall_files
                 + 0.25 × faithfulness
```

### `prepare_eval_dataset.py` parameters

| Parameter | Default | Role |
|---|---|---|
| `--experiment-id` | input file name | identifier shared with `rag_evaluation.py` |
| `--collection` | `baseline_fragus` | target Milvus collection |
| `--candidates` | `100` | chunks retrieved before reranking |
| `--top-n` | `10` | chunks kept after reranking (passed to the LLM) |
| `--dense-weight` | `0.5` | weight of the dense BGE-M3 field in the `WeightedRanker` |
| `--bm25-weight` | `0.5` | weight of the sparse BM25 field in the `WeightedRanker` |
| `--fermi-weight` | `0.0` | weight of the sparse Fermi field (0 = disabled by default) |
| `--reranker` | `mmarco-mMiniLMv2-L12-H384-v1` | cross-encoder reranking model |
| `--reranker-threshold` | none | minimum reranker score for a chunk to enter the context — accepts any float including negatives (absent = keep the full top-n) |
| `--ranker` | `weighted` | Milvus fusion: `weighted` (WeightedRanker) or `rrf` (Reciprocal Rank Fusion) |
| `--rrf-k` | `60` | RRF smoothing constant (ignored with `--ranker weighted`) |
| `--ranking-mode` | `reranker` | final ranking: `reranker` (cross-encoder score only) or `combined` (normalized average of Milvus + reranker scores) |
| `--query-cache` | none | path to a `.pkl` query-embedding cache file (avoids re-encoding the same questions across runs) |
| `--llm-model-key` | value from `scripts/rag/config.toml` | LLM generation model (Cleyrop proxy key) |
| `--paraphrase-variants` | `3` | variants per question for rephrasing robustness (0 = disabled) |
| `--nrows` | all | limit to N questions (useful for testing) |

Resuming after an interruption is automatic: rerunning with the same
`--experiment-id` picks up where the script stopped without overwriting the
existing answers.

Example comparison of retrieval configurations:

```bash
# Dense-heavy configuration
uv run python prepare_eval_dataset.py data/rag_evaluation_dataset.xlsx \
  --experiment-id dense_heavy --dense-weight 0.7 --bm25-weight 0.3

# With Fermi (requires a running fermi_server)
uv run python prepare_eval_dataset.py data/rag_evaluation_dataset.xlsx \
  --experiment-id hybrid_fermi \
  --dense-weight 0.34 --bm25-weight 0.33 --fermi-weight 0.33

# Generation-model comparison
uv run python prepare_eval_dataset.py data/rag_evaluation_dataset.xlsx \
  --experiment-id mistral_small --llm-model-key mistral-small-3.2-24b-instruct-2506

# With RRF + combined mode (Milvus + reranker scores) + quality threshold
uv run python prepare_eval_dataset.py data/rag_evaluation_dataset.xlsx \
  --experiment-id rrf_combined \
  --ranker rrf --ranking-mode combined --reranker-threshold 0.0
```

## Script tree

| Folder | Role | Scripts |
|---|---|---|
| `scripts/pipeline/` | ingestion chain | `build_manifest_fragus_clean`, `convert_corpus`, `chunk_corpus`, `embed_corpus`, `brute_corpus`, `embed_fermi`, `build_fragus_collection`, `_docling_client` |
| `scripts/rag/` | answer generation | `hybrid_rag_pipeline` (class `HybridRAGPipeline`), `graph_rag_pipeline` |
| `scripts/rag_evaluation/` | evaluation campaign | `prepare_eval_dataset`, `prepare_eval_dataset_graph`, `rag_evaluation`, `compare_rag_runs` |
| `scripts/inspect/` | stats / visualization / exploration | `milvus_recap`, `chunk_stats`, `explore_chunks`, `explore_manifest`, `view_chunks`, `view_parsed`, `visualize_chunks`, `view_duplicates`, `av_audit` |
| `scripts/diagnostics/` | one-off debugging / repair | `diagnose_zero_pages`, `probe_convert`, `repair_doc`, `recover_orphan_tasks` |
| `scripts/ops/` | connectivity & tokens | `browse`, `test_llm`, `test_milvus`, `keepalive_cleyrop`, `refresh_token_daemon` |

Each script carries its usage docstring at the top of the file.

## Documentation

- [`docs/target-pipeline.md`](docs/target-pipeline.md) — targeted retrieval / fusion / rerank architecture
- [`docs/reranking.md`](docs/reranking.md) — reranking work
- [`docs/convert-pipeline.md`](docs/convert-pipeline.md), [`docs/chunking.md`](docs/chunking.md) — ingestion steps
- [`docs/milvus-collection.md`](docs/milvus-collection.md), [`docs/collection-fermi.md`](docs/collection-fermi.md) — Milvus collections
- [`docs/conversion-benchmark.md`](docs/conversion-benchmark.md) — conversion benchmark
- [`docs/rag_eval_howto.md`](docs/rag_eval_howto.md) — evaluation how-to
- [`docs/rag_runs_catalog.md`](docs/rag_runs_catalog.md) — catalog of evaluation runs and aggregate scores
- [`docs/run_analysis.md`](docs/run_analysis.md) — parametric study: experimental analysis of the evaluation series
