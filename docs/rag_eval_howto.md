# RAG Evaluation — How to launch a run

## 2-step structure

```
Step 1: prepare_eval_dataset.py   →  eval_{id}.xlsx  (RAG answers)
Step 2: rag_evaluation.py         →  score_{id}_*.xlsx  (RAGAS + LLM judge scores)
```

---

> **Prerequisite**: install the evaluation dependencies with
> `uv sync --extra eval` (add `--extra graph` for the GraphRAG runs via
> `prepare_eval_dataset_graph.py`).

## Step 1 — Generating the RAG answers

```bash
nohup python scripts/rag_evaluation/prepare_eval_dataset.py \
  scripts/rag_evaluation/data/rag_evaluation_dataset.xlsx \
  --experiment-id baseline_fermi_02 \
  --collection baseline_fragus \
  --dense-weight 0.34 --bm25-weight 0.33 --fermi-weight 0.33 \
  --candidates 100 --top-n 10 \
  --reranker cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 \
  --query-cache scripts/rag_evaluation/data/query_embeddings_cache.pkl \
  > /tmp/prepare_{id}.log 2>&1 &
```

**Key parameters:**
| Parameter | Default value | Notes |
|-----------|------------------|-------|
| `--collection` | `baseline_fragus` | or `brut_fragus` for the A/B run |
| `--dense-weight / --bm25-weight / --fermi-weight` | 0.5 / 0.5 / 0.0 | Fermi: 0.34/0.33/0.33 |
| `--candidates` | 100 | Number of chunks before reranking |
| `--top-n` | 10 | Chunks kept after reranking |
| `--ranker` | `weighted` | Milvus fusion: `weighted` or `rrf` |
| `--rrf-k` | 60 | RRF smoothing constant (ignored if `--ranker weighted`) |
| `--ranking-mode` | `reranker` | Final ranking: `reranker` (cross-encoder score only) or `combined` (normalized average of Milvus + reranker scores) |
| `--reranker-threshold` | none | Minimum reranker score to include a chunk (e.g. `0.0` to drop negative scores) |
| `--query-cache` | none | `.pkl` cache file for query embeddings (speeds up subsequent runs on the same questions) |

Output: `scripts/rag_evaluation/data/results/eval_{experiment-id}.xlsx`  
**Checkpointing**: automatic resume if the file already exists.

---

## Step 2 — Scoring (RAGAS + LLM judge)

```bash
nohup python scripts/rag_evaluation/rag_evaluation.py \
  scripts/rag_evaluation/data/results/eval_{experiment-id}.xlsx \
  --experiment-id {experiment-id} \
  --llm-model-key mistral-small-3.2-24b-instruct-2506 \
  --ragas-workers 3 \
  --embedding-cache scripts/rag_evaluation/data/results/emb_cache_bge_multilingual_gemma2_{experiment-id}.pkl \
  > /tmp/eval_{id}.log 2>&1 &
```

**Key parameters:**
| Parameter | Default value | Notes |
|-----------|------------------|-------|
| `--llm-model-key` | mistral-large-latest | Use **mistral-small-3.2-24b-instruct-2506** for the judge |
| `--ragas-workers` | 3 | RAGAS parallelism (do not exceed 5, OVH quota 400 req/min) |
| `--no-ragas / --no-factual / --no-anchoring / --no-diversity` | — | Skip individual steps |

Outputs:
- `results/score_{id}_per_question.xlsx`
- `results/score_{id}_averages.xlsx`

**Checkpointing**: automatic resume from the last checkpoint.

---

## Global score formula

```
global_rag_score = 0.25 × context_recall_llm
                 + 0.35 × factual_correctness_recall
                 + 0.15 × citation_recall_files
                 + 0.25 × faithfulness
```

---

## Available collections

| Collection | Corpus | Chunks |
|------------|--------|--------|
| `baseline_fragus` | docling + OCR | 29,358 |
| `brut_fragus` | raw pypdf | 15,520 |

---

## Monitoring a running run

```bash
# Step 1
grep -E "group|ERROR" /tmp/prepare_{id}.log | tail -5

# Step 2
grep -E "Q[0-9]+|score|ERROR|exhausted" /tmp/eval_{id}.log | tail -10
```
