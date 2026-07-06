# Reranking — selected model

Scoping note for the *reranking* part of the RAG (ASNR corpus, French,
on-prem / SecNumCloud deployment, open weights).

## Role in the pipeline

The reranker is a **second stage** plugged into the retrieval output:

```
retrieval (top-100)  →  reranker (cross-encoder re-score)  →  final top-k (10)
```

Goal: turn broad recall into precise context. An **ON/OFF axis orthogonal**
to the retrieval method (dense / BM25 / sparse / hybrid).

## Selected model

**`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`** — loaded locally on CPU.

| Property | Value |
|---|---|
| Params | ~134M |
| Language | multilingual (trained on mMARCO, FR data included) |
| License | Apache 2.0 |
| Execution | CPU, `sentence-transformers` `CrossEncoder` |
| Latency | ~1 s / batch of 100 passages on the VM CPU |

Integrated into `HybridRAGPipeline` via the `RERANKER_MODEL` constant; the model
is downloaded automatically on first launch.

## Rejected candidates

| Model | Reason |
|---|---|
| Alibaba-NLP/gte-multilingual-reranker-base | not evaluated on the corpus |
| mixedbread-ai/mxbai-rerank-base-v2 | not evaluated on the corpus |
| jina-reranker-v2-base-multilingual | CC-BY-NC license (risk for ASNR) |

## Remote execution (optional)

`hybrid_rag_pipeline.py` supports a `RERANKER_URL` environment variable to
offload reranking to an external server (e.g. Mac MPS).
By default `RERANKER_URL` is empty → the model runs locally on CPU.

## Final ranking modes (`ranking_mode`)

After cross-encoder scoring, two ranking strategies are available via the
`ranking_mode` parameter of `HybridRAGPipeline` (or `--ranking-mode` in
`prepare_eval_dataset.py`):

| Mode | Sorted by | Usage |
|---|---|---|
| `"reranker"` (default) | cross-encoder score only | favors pure semantic relevance |
| `"combined"` | normalized average (Milvus + reranker scores) | keeps the vector-proximity signal; useful when the reranker is noisy |

In `combined` mode, both scores are min-max normalized then averaged 50/50
before sorting. The `reranker_threshold` parameter applies in both modes on the
raw cross-encoder score.

## Threshold filtering (`reranker_threshold`)

`reranker_threshold` is an optional float (default: `None`) that drops the
chunks whose cross-encoder score is below the threshold, after sorting and
truncating to `top_n`.

```python
# Drop chunks with a negative score (a sign of irrelevance)
rag = HybridRAGPipeline(reranker_threshold=0.0)

# Stricter threshold
rag = HybridRAGPipeline(reranker_threshold=1.5)
```

> **Caution**: a threshold set too high can return fewer than `top_n` chunks,
> or even an empty context for rare questions. Prefer values close to 0 to
> filter out only the clearly negative scores.
