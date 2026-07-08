# RAG Parametric Study — Experimental Analysis

> **Document scope:** systematic evaluation of the HybridRAG pipeline configurations across
> five experimental series. All runs are evaluated on the same 65 question-answer pairs
> (260 queries including paraphrase variants) with a fixed generation LLM, a fixed judge LLM,
> and a fixed prompt template. The companion document `run_analysis_graph.md` covers the
> GraphRAG experiments on the same question bank.

---

## 1. Experimental Protocol

### 1.1 Overview and Objective

This document presents a systematic parametric evaluation of a Retrieval-Augmented Generation
(RAG) pipeline applied to a corpus of regulatory and technical documents from the French
nuclear safety authority (ASNR). In a RAG system, the language model does not answer from
parametric memory alone: it is first provided a set of retrieved document excerpts (chunks)
that serve as grounding context. The quality of the final answer therefore depends jointly on
the model's generation ability and on the retrieval system's capacity to surface relevant,
accurate, and coherent content. This study focuses exclusively on the retrieval side: the
generation LLM, the prompt template, and the evaluation corpus are held fixed throughout, so
that all observed performance differences can be attributed solely to retrieval configuration
choices.

The pipeline operates in four sequential stages: (1) hybrid multi-field search over a Milvus
vector database, (2) score fusion across retrieval fields, (3) cross-encoder reranking of
candidate chunks, and (4) generation by a large language model conditioned on the selected
chunks. Five configuration axes are studied in sequence, each series promoting its best run
as the baseline of the next:

1. **Document ingestion strategy** — structured pre-processing vs. raw ingestion.
2. **Embedding field combination and rank fusion** — which vector representations are used,
   and how their scores are fused (weighted sum vs. Reciprocal Rank Fusion).
3. **Retrieved chunk count (top-n)** — how many reranked chunks are passed to the LLM.
4. **Combined ranking mode** — blending retrieval and reranker scores for the final ranking.
5. **Reranker score thresholding** — filtering low-confidence chunks out of the context.

### 1.2 Pipeline Architecture and Design Rationale

#### 1.2.1 Dense Retrieval

Dense retrieval encodes both documents and queries as continuous vectors using a bi-encoder —
here **BGE-M3** (BAAI/bge-m3), served via the Cleyrop proxy. At query time, the query is
encoded into the same vector space and the most semantically similar chunks are retrieved by
approximate nearest-neighbour search (cosine similarity in Milvus). Dense retrieval captures
semantic proximity: a question phrased differently from the source text can still match the
relevant passage. It is intrinsically robust to lexical variation and paraphrasing, but the
bi-encoder scores query and document independently, which limits matching quality — the model
cannot attend to the specific interaction between a given query and a given document.

#### 1.2.2 Sparse Retrieval — BM25

BM25 is a classic term-frequency ranking function: it scores a document against a query from
the frequency of shared terms, normalised by document length and weighted by inverse document
frequency. It cannot generalise beyond the vocabulary — brittle to paraphrasing, but extremely
reliable for rare, highly specific terms (technical identifiers, regulation article
references, proper nouns) that dense models may embed inconsistently. BM25 scores are computed
client-side from a persisted IDF/avgdl state fitted on the whole corpus and stored as sparse
vectors in Milvus (field `sparse_bm25`).

#### 1.2.3 Sparse Retrieval — Fermi

Fermi (`atomic-canyon/fermi-1024`) is a **learned sparse encoder** (SPLADE architecture).
Like BM25 it produces sparse token-weight vectors, but the weights are learned end-to-end by
a transformer trained to maximise retrieval quality — implicitly learning term expansion (a
chunk about "radioactive waste" may receive weight for "nuclear effluents" without the term
appearing). It bridges exact lexical matching (BM25) and full semantic generalisation
(dense). Fermi vectors live in the `sparse_fermi` field. BM25 and Fermi are complementary:
exact term coverage on one side, domain-specific implicit expansion on the other.

#### 1.2.4 Score Fusion: Weighted Sum vs. Reciprocal Rank Fusion

**Weighted fusion (`w`)** computes a linear combination of the raw field scores. It assumes
the scores are commensurable — but cosine similarities (dense) and inner products (sparse)
follow different distributions, and calibrating weights correctly without tuning data is
difficult.

**Reciprocal Rank Fusion (`rrf`)** avoids calibration entirely by operating on ranks: each
field produces an independent ranked list, and a chunk's fused score is `Σ 1/(k + rank_i)`
with k = 60. A chunk consistently near the top across fields accumulates a high score
regardless of absolute values. RRF is the established robust default when score
distributions are heterogeneous.

#### 1.2.5 Cross-Encoder Reranking

After fusion, Milvus returns a pool of 100 candidate chunks. A **cross-encoder**
(`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`) then scores each (query, chunk) pair jointly —
every token of the query can attend to every token of the chunk — producing far more accurate
relevance judgements than the bi-encoder, at a compute cost that restricts it to the candidate
pool. The top-n chunks by cross-encoder score form the LLM context.

#### 1.2.6 Combined Ranking Mode

The standard pipeline ranks the final chunks by cross-encoder score alone. **Combined mode**
(`_comb`) instead ranks by a 50/50 blend of the min-max-normalised Milvus fusion score and
cross-encoder score, hypothesising that chunks supported by both signals are safer picks.

#### 1.2.7 Reranker Score Thresholding

Even after reranking, the selected chunks may include weakly relevant material. A **score
threshold** (`_thr{v}`) drops any chunk whose raw cross-encoder score falls below a fixed
value, trading context breadth for context purity. The hypothesised benefit is lower
hallucination; the risk is losing recall — and, for aggressive thresholds, emptying the
context of hard questions entirely.

### 1.3 Experiment Naming Convention

```
{fields}_{ranker}[_top{n}][_comb][_thr{v}]
```

| Token | Values | Meaning |
|-------|--------|---------|
| `{fields}` | `db` / `df` / `dbf` | dense+BM25 / dense+Fermi / dense+BM25+Fermi |
| `{ranker}` | `w` / `rrf` | weighted fusion / Reciprocal Rank Fusion |
| `_top{n}` | e.g. `_top50` | chunks passed to the LLM (omitted when default = 10) |
| `_comb` | present/absent | combined ranking mode |
| `_thr{v}` | e.g. `_thr-1` | minimum cross-encoder score |

Two runs predate this convention: `baseline` (dense + BM25, weighted, top-10 — the original
pipeline) and `baseline_raw` (same retrieval, documents ingested without structured
pre-processing).

### 1.4 Evaluated Metrics

**Response quality** — Factual Correctness Recall (fraction of expected facts present in the
response), Faithfulness (fraction of response claims grounded in the retrieved context),
Reformulation Robustness (consistency across paraphrases), Hallucination Rate (fraction of
ungrounded claims — lower is better), Citation Recall (fraction of expected source documents
cited).

**Retrieved context quality** — Context Recall (fraction of the expected answer covered by
the retrieved chunks), Context Relevance (file-level precision of the retrieved set), Context
Coverage (file-level breadth over expected sources), Diversity (semantic dissimilarity of the
retrieved chunks), Hub France IA F-Score (F-measure of context precision and recall).

**Global RAG Score** — the primary decision criterion:
`0.25 × Context Recall + 0.35 × FC Recall + 0.15 × Citation Recall + 0.25 × Faithfulness`.

### 1.5 Statistical Methodology

Every pairwise comparison uses a two-tailed Wilcoxon signed-rank test on the 65 matched
question-level scores, with Benjamini-Hochberg correction at α = 0.05 across the metrics
tested. In the body of this document, differences are simply reported as **significant or
not** under that procedure; effect sizes and corrected p-values are listed in Appendix C.
A difference reported as significant can be read as "established"; everything else is at
best directional and must not be over-interpreted.

### 1.6 Series Overview

| Series | Baseline | Candidates | Question |
|--------|----------|------------|----------|
| 1 | `baseline` | `baseline_raw` | Does structured ingestion help? |
| 2 | `baseline` | `db_rrf`, `df_w`, `df_rrf`, `dbf_w`, `dbf_rrf` | Best fields × fusion combination? |
| 3 | `dbf_rrf` | `dbf_rrf_top{25,50,100}` | Optimal chunk count? |
| 4 | `dbf_rrf_top50` | `dbf_rrf_top50_comb` | Does combined ranking help? |
| 5 | `dbf_rrf_top50` | `dbf_rrf_top50_thr{−2,−1,0,+1}` | Does score thresholding help? |

---

## 2. Series 1 — Document Ingestion Strategy

**Question:** does structured pre-processing of documents at ingestion time improve retrieval
quality over raw ingestion?

### 2.1 Results

| Metric | baseline | baseline_raw | Δ (raw − base) | Significant |
|--------|:---:|:---:|:---:|:---:|
| FC Recall | 44.87% | 45.66% | +0.79 pp | No |
| Faithfulness | 88.09% | 77.49% | −10.59 pp | **Yes — baseline wins** |
| Robustness | 87.66% | 87.45% | −0.22 pp | No |
| Hallucination ↓ | 5.06% | 4.59% | −0.47 pp | No |
| Citation Recall | 26.72% | 18.26% | −8.46 pp | **Yes — baseline wins** |
| Context Recall | 38.48% | 41.72% | +3.24 pp | No |
| Diversity | 26.69% | 33.28% | +6.59 pp | **Yes — raw wins** |
| **Global RAG Score** | **51.35%** | **48.52%** | **−2.83 pp** | **No** |

### 2.2 Analysis

The global gap is not significant, but the three metrics that are tell a single, coherent
story about **where ingestion quality surfaces in a RAG pipeline** — and it is not where one
might first look.

Raw ingestion does not hurt *retrieval reach*: context recall is even directionally higher
(+3.24 pp), because raw chunks are more numerous and heterogeneous, so a wide net catches
slightly more of the expected content. What collapses is what happens *after* retrieval.
Faithfulness loses 10.59 pp: raw chunks carry OCR residue, broken formatting and poorly
delimited boundaries, so even when the right passage is retrieved, the LLM works from a
degraded substrate and produces claims that the judge cannot anchor in the context. Citation
recall loses 8.46 pp for the same underlying reason — sloppy chunk boundaries blur the
mapping between a passage and its source document, so the model cites less and cites worse.

The diversity result is the instructive trap of this series: it is the only metric where
raw ingestion wins significantly, and it is a *false positive for quality*. Raw chunks are
more semantically dissimilar from each other because they are noisier — the metric measures
heterogeneity, and noise is heterogeneous. The gain comes with zero improvement on any
answer-quality metric, which is exactly the signature separating "diverse because rich" from
"diverse because dirty". The same metric will reappear in Series 3 with the opposite
meaning, attached to genuine coverage gains.

The asymmetry matters for the global score: faithfulness weighs 25% of it, and the two
significant losses (faithfulness, citation) cannot be bought back by a marginal recall gain.
**Ingestion is not a retrieval parameter — it is a floor under everything downstream.** No
subsequent series revisits it: every later run inherits structured ingestion.

### 2.3 Verdict — Structured ingestion retained

`baseline` is the canonical starting point for all subsequent series.

---

## 3. Series 2 — Embedding Fields × Fusion Strategy

**Question:** which combination of retrieval fields (dense+BM25, dense+Fermi, all three) and
fusion strategy (weighted vs. RRF) retrieves best? All runs use top-10 chunks, isolating
these two axes from the chunk-count effect studied next.

### 3.1 Results

| Run | FC Recall | Faithfulness | Hall ↓ | Cit Recall | Ctx Recall | **Global** | Global significant vs baseline |
|-----|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| baseline (db_w) | 44.87% | 88.09% | 5.06% | 26.72% | 38.48% | **51.35%** | — |
| db_rrf | 44.17% | 89.19% | 5.47% | 27.99% | 44.56% | **53.06%** | No |
| df_w | 45.46% | 86.05% | 5.58% | 21.50% | 42.37% | **50.91%** | No |
| df_rrf | 43.38% | 88.48% | 2.84% | 27.33% | 44.08% | **52.42%** | No |
| dbf_w | 45.67% | 83.48% | 4.13% | 24.16% | 40.34% | **50.56%** | No |
| dbf_rrf | 46.34% | 89.02% | 6.21% | 28.55% | 41.92% | **53.24%** | No |

No metric of any candidate run differs significantly from `baseline`. The value of this
series lies in its directional patterns, which are unusually consistent.

### 3.2 Analysis

**The fusion axis produces a clean sweep: RRF beats weighted fusion in all three field
families, without a single exception.**

| Fields | Weighted | RRF | Δ |
|--------|:---:|:---:|:---:|
| db | 51.35% | 53.06% | +1.71 pp |
| df | 50.91% | 52.42% | +1.51 pp |
| dbf | 50.56% | 53.24% | +2.68 pp |

This pattern is exactly what the theory of §1.2.4 predicts. Weighted fusion adds raw scores
that live on incomparable scales — a cosine similarity bounded in [−1, 1] against unbounded
sparse inner products — so the weights encode an implicit calibration that nobody tuned.
RRF replaces scores by ranks and removes the calibration problem entirely. The telling
detail is that **the RRF advantage grows with the number of fields to calibrate**: +1.5 to
+1.7 pp with two fields, +2.68 pp with three. Each additional score distribution makes the
weighted sum harder to balance, while costing RRF nothing. Three consistent instances of a
mechanistically predicted direction is not statistical proof, but it is the right kind of
evidence for an architecture decision.

**The field axis shows that Fermi is an amplifier, not a substitute.** Two observations
lock this interpretation in place. First, under weighted fusion, adding Fermi actually
*hurts* (dbf_w 50.56% < baseline 51.35%): a third uncalibrated score destabilises the sum
faster than its signal helps. The value of a retrieval field is conditional on the fusion
strategy being able to absorb it. Second, *replacing* BM25 by Fermi (`df` runs) always
underperforms keeping BM25: on a corpus dense in exact technical identifiers (package
codes, regulation articles), nothing substitutes for literal term matching. Fermi's learned
term expansion only adds value *on top of* BM25, and only under RRF (dbf_rrf is the best
run of the series).

**Why nothing is significant here — and why that is expected.** At top-10, all
configurations feed the same cross-encoder from largely overlapping 100-candidate pools,
and the reranker cuts to a final selection so small that most configurations agree on it.
The retrieval differences exist but are attenuated by the two stages downstream of them.
This is not a reason to dismiss the series: it is the reason the next series exists. A
better retrieval mix can only express its advantage if the context is large enough to admit
the additional relevant material it finds — which is precisely what Series 3 tests.

### 3.3 Verdict — `dbf_rrf` promoted (Global = 53.24%)

All three fields, fused by RRF. The choice rests on directional consistency and mechanism
rather than significance — and Series 3 validates it a posteriori: the gains that were
latent at top-10 become the largest confirmed effects of the study once the context grows.

---

## 4. Series 3 — Retrieved Chunk Count (Top-n)

**Question:** how many reranked chunks should be passed to the LLM? Values tested within
`dbf_rrf`: 10 (series baseline), 25, 50, 100. The candidate pool is fixed at 100.

### 4.1 Results

| Run | FC Recall | Faithfulness | Hall ↓ | Cit Recall | Ctx Recall | Ctx Rel | Diversity | **Global** |
|-----|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| dbf_rrf (top-10) | 46.34% | 89.02% | 6.21% | 28.55% | 41.92% | 9.19% | 25.93% | **53.24%** |
| dbf_rrf_top25 | 49.55% | 89.62% | 6.37% | 33.10% | 52.86% | 6.29% | 27.41% | **57.93%** |
| **dbf_rrf_top50** | 54.40% | 88.83% | 3.86% | 36.47% | 75.32% | 4.98% | 29.26% | **65.55%** |
| dbf_rrf_top100 | 57.82% | 91.01% | 3.29% | 35.33% | 87.00% | 3.69% | 32.17% | **70.04%** |

Pairwise verdicts on the Global RAG Score:

| Comparison | Δ Global | Significant |
|-----------|:---:|:---:|
| top-10 → top-25 | +4.69 pp | No (borderline) |
| top-10 → top-50 | +12.31 pp | **Yes** |
| top-25 → top-50 | +7.62 pp | **Yes** |
| top-50 → top-100 | +4.49 pp | No |

### 4.2 Analysis

**This is the decisive axis of the entire study** — the two significant global gains
measured anywhere in it are both on this table, and they are large. The chain of causation
is worth walking through step by step, because each link is independently visible in the
metrics.

**Link 1 — more chunks, more of the answer.** Context recall climbs from 41.92% (top-10) to
75.32% (top-50) to 87.00% (top-100); every step is significant. This is near-mechanical:
the pool of 100 candidates contains far more relevant material than 10 slots can hold, and
each widening of the context admits more of it. That the curve is still rising steeply at
50 tells us the reranker's ranking quality is good deep into the pool — chunks ranked 26-50
are not filler, they carry answer content.

**Link 2 — the answer actually uses it.** Factual correctness recall follows context recall
upward (46.34% → 54.40% → 57.82%), and citation recall rises with it (28.55% → 36.47%).
The generation model does exploit the additional material rather than ignoring it — a
larger context translates into more complete, better-sourced answers, not just a bigger
prompt.

**Link 3 — and grounding does not pay for it.** This is the pivotal, least intuitive
finding: faithfulness is statistically flat across the whole range (88.8–91.0% from 10 to
100 chunks), and the hallucination rate *falls* as the context grows — 6.21% at top-10,
3.86% at top-50, 3.29% at top-100. The standard objection to large contexts is that the
model "gets lost" and drifts; on this corpus, the opposite happens. The mechanism: a
hallucination is typically the model bridging a gap in its evidence with parametric memory.
More retrieved evidence means fewer gaps to bridge. The classic trade-off "coverage vs.
grounding" simply does not materialise in the 10–100 range here — which is what makes the
recall gains free, and the top-n lever so powerful.

**The one real cost is precision dilution, and it is confined to context-side metrics.**
Context relevance falls monotonically (9.19% → 3.69%) and significantly at each step: with
more chunks, a smaller fraction is individually critical. But the dilution demonstrably
stops at the context — no answer-quality metric degrades with it. The reranker acts as the
safety net: the added chunks are lower-ranked but still relevant, noise-like chunks stay
below the cut.

**Why stop at 50 when 100 scores higher?** Top-100 posts the study's best global score
(70.04%), but its +4.49 pp over top-50 is not significant, while its *costs* are: the
significant metrics at top-100 are retrieval mechanics (recall, coverage, diversity) plus
significant additional precision losses. Three practical arguments close the case. First,
prompts of roughly 45k tokens per question double the latency and cost of top-50 for an
unproven gain. Second, they push both the generation model and any evaluation machinery
deep into the long-context regime where reliability empirically degrades. Third, the
marginal gains are flattening (+7.62 pp then +4.49 pp): the curve is saturating, and the
remaining headroom shrinks against fixed costs. Should the [50, 100] zone ever need to be
settled, the correct instrument is a larger question bank — the effect to detect is below
what 65 questions resolve — not intermediate top-n points, which would chase even smaller
effects with the same instrument.

### 4.3 Verdict — `dbf_rrf_top50` promoted, champion of the study (Global = 65.55%)

Top-50 is the largest context whose gain is statistically demonstrated end-to-end, with
flat faithfulness, the study's second-lowest hallucination rate, and manageable cost.
Top-100 remains a documented directional data point.

---

## 5. Series 4 — Combined Ranking Mode

**Question:** does re-ranking the final 50 chunks by a 50/50 blend of Milvus fusion score
and cross-encoder score improve over the cross-encoder alone?

### 5.1 Results

| Metric | dbf_rrf_top50 | dbf_rrf_top50_comb | Δ | Significant |
|--------|:---:|:---:|:---:|:---:|
| FC Recall | 54.40% | 55.76% | +1.36 pp | No |
| Faithfulness | 88.83% | 92.48% | +3.65 pp | No |
| Robustness | 88.79% | 88.54% | −0.25 pp | No |
| Hallucination ↓ | 3.86% | 4.35% | +0.49 pp | No |
| Citation Recall | 36.47% | 36.39% | −0.07 pp | No |
| Context Recall | 75.32% | 75.00% | −0.32 pp | No |
| Diversity | 29.26% | 29.26% | 0.00 pp | No |
| Hub IA F-Score | 27.89% | 29.07% | +1.18 pp | No |
| **Global RAG Score** | **65.55%** | **66.84%** | **+1.30 pp** | **No** |

Not a single metric out of the 17 tested is significant — the weakest differentiation
observed anywhere in the study.

### 5.2 Analysis

The hypothesis behind combined mode was reasonable: the cross-encoder judges each chunk in
isolation and might over-reward passages that *sound* like the question without informing
the answer; blending in the retrieval score — which reflects a different, embedding-space
notion of proximity — could re-anchor the selection. The result is a textbook null: nothing
moves, in either direction, on any metric.

The explanation is structural rather than statistical. Combined mode can only change the
outcome by changing *which* chunks enter the context or *in what order* — and at a 100 → 50
cut, it barely changes either. The Milvus fusion ranking and the cross-encoder ranking
disagree mostly about fine ordering near the top, not about which fifty of a hundred
candidates are worth keeping; the two signals were already chained (fusion selects the
pool, the reranker orders it), so their consensus is baked in before the blend applies.
Where the same blend might matter is at aggressive cuts (top-10 from a large pool) — at
top-50 the selection is too permissive for a re-weighting to bite. The seemingly attractive
faithfulness gain (+3.65 pp) sits well inside noise and is offset by an equally
insignificant hallucination regression; neither survives any statistical reading.

There is also a lesson in what did *not* happen: blending a noisier signal (raw retrieval
scores) into the ranking did not *hurt* either. The pipeline is robust to this class of
final-ranking perturbation — consistent with Series 3's finding that answer quality is
driven by what makes it into the context at all, not by ordering subtleties within it.

### 5.3 Verdict — Combined mode brings nothing; standard cross-encoder ranking retained

---

## 6. Series 5 — Reranker Score Thresholding

**Question:** does dropping low-scoring chunks from the top-50 context improve answer
quality — in particular, does it reduce hallucination?

### 6.1 Threshold Selection

Candidate thresholds are anchored on the cross-encoder score distribution of the 3,250
chunks actually served by the champion (65 questions × 50 chunks): median = 0.49,
mean = 0.66, P5 = −2.95, P25 = −1.15, P75 = 2.34, P95 = 4.94. The tested ladder samples the
removal curve while controlling the empty-context risk:

| Threshold | Chunks removed | Questions with empty context | Median chunks kept |
|:---:|:---:|:---:|:---:|
| −2.0 | 13.3% | 0/65 | 50/50 |
| −1.0 | 27.4% | 0/65 | 44/50 |
| 0.0 | 42.3% | 2/65 | 28/50 |
| +1.0 | 58.1% | 4/65 | 17/50 |

**Why the ladder stops at +1.** Thresholding carries a structural risk that grows with the
cut-off: for hard questions where even the best available chunks score low, the filter
silently removes the *entire* context, and the LLM answers with no grounding at all. At
+1.0 this already affects 4 of 65 questions; pushing further up the score scale
(median = 0.49) would turn a per-chunk filter into a per-question denial-of-context for a
growing share of the bank, making the measured averages a blend of two regimes (filtered
answers vs. context-free answers) that no longer isolates the effect under study. This
failure mode could be countered by a **fallback rule** — when the filter empties (or nearly
empties) a context, keep the top-n best-scored chunks regardless of threshold — which would
make aggressive thresholds testable safely. This fallback was not experimented in this
study; without it, thresholds beyond +1 are not meaningfully evaluable.

Methodology note: the thr−2 run performs the full retrieval; the higher thresholds are
derived from its stored per-chunk reranker scores and re-generated — all four variants
therefore filter exactly the same chunk sets.

### 6.2 Results

| Run | FC Recall | Faithfulness | Hall ↓ | Cit Recall | Ctx Recall | Ctx Rel | Diversity | **Global** | Global significant vs champion |
|-----|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **dbf_rrf_top50** | 54.40% | 88.83% | 3.86% | 36.47% | 75.32% | 4.98% | 29.26% | **65.55%** | — |
| thr−2 | 55.84% | 89.04% | 1.62% | 31.79% | 71.66% | 5.15% | 28.63% | **64.49%** | No |
| thr−1 | 53.28% | 89.68% | 1.73% | 35.16% | 66.00% | 5.60% | 27.67% | **62.84%** | No |
| thr0 | 53.78% | 90.17% | 5.96% | 29.67% | 57.40% | 6.39% | 26.80% | **60.17%** | No (borderline) |
| thr+1 | 47.18% | 87.97% | 5.77% | 27.36% | 51.77% | 7.56% | 26.10% | **55.55%** | **Yes — champion wins** |

Beyond the global score: diversity degrades significantly from thr−2 onward, paraphrase
robustness from thr−1 onward, context recall and coverage from thr0 onward, and at thr+1
citation recall joins them. The only significant win for any threshold anywhere is context
relevance at thr+1 — precision rising as the retained set shrinks.

### 6.3 Analysis

**Thresholding attacks exactly the mechanism that made the champion win.** Series 3
established that `dbf_rrf_top50`'s advantage is built on context recall; thresholding gives
that recall back, in direct proportion to the cut: 75.32% unfiltered → 71.66% → 66.00% →
57.40% → 51.77%. By thr+1 the median context is down to 17 chunks — *below* the top-25
regime — so the run pays reranker-filtering overhead to land on a configuration that
Series 3 had already shown to be inferior. Predictably, thr+1 posts the only significant
global regression of the whole study (−10.00 pp). The through-line from Series 3 to here is
one sentence: **on this corpus, whoever removes context loses.**

**The hallucination story is the interesting one — a U-shaped curve with a change of regime
in the middle.** Moderate filtering achieves the lowest hallucination rates observed
anywhere in this study: 1.62% at thr−2 and 1.73% at thr−1, versus 3.86% unfiltered — the
intended effect of thresholding, working as designed. But past thr−1 the curve *reverses*:
thr0 (5.96%) and thr+1 (5.77%) are worse than no filtering at all. The mechanism flips at
the point where the filter stops removing noise and starts removing evidence. Chunks
scoring between roughly −1 and 0 are lexically weak matches but still carry grounding value
— supporting details, adjacent passages the model can anchor secondary claims to. Remove
them and the model faces evidence gaps that it fills from parametric memory, which is
precisely the behaviour the threshold was meant to suppress. Filtering "against
hallucination" is thus only defensible in a narrow band — and even inside that band it buys
no global improvement, because the recall cost accrues from the very first removed chunk.

**Thresholds do not transfer across configurations.** A fixed score cut removes a similar
*proportion* of chunks whatever the base, but the damage depends on what that proportion
was buying. On a recall-built top-50 base, every threshold is net-negative; the same cuts
on a top-10 base would have little recall to destroy. Any future change of top-n therefore
invalidates previously validated thresholds — threshold and context size are coupled
parameters and must be tuned jointly, not sequentially.

### 6.4 Verdict — No threshold recommended

Every tested threshold loses more on recall than it gains anywhere else; the degradation
reaches global significance at thr+1 and near-significance at thr0. If a deployment context
specifically prioritises minimal hallucination over completeness (e.g. a compliance
use case), **thr−1 is the defensible option**: the study's lowest hallucination rate
(1.73%) at a statistically unconfirmed global cost of −2.71 pp. thr0 and beyond should be
avoided outright — they enter the zone where questions start losing their entire context.
Should thresholding ever be revisited, it should be paired with the empty-context fallback
described in §6.1 (keep the best-scored chunks when the filter would empty a context),
which removes the denial-of-context failure mode and would make the whole threshold range
safely explorable.

---

## 7. Final Conclusion

### 7.1 Established Findings (statistically significant)

1. **Structured ingestion is a prerequisite.** Raw ingestion significantly degrades
   faithfulness (−10.59 pp) and citation recall (−8.46 pp): chunk coherence at ingestion
   time propagates directly to answer grounding quality.

2. **Context size is the dominant retrieval lever on this corpus.** `dbf_rrf_top50` gains
   +12.31 pp of Global RAG Score over the top-10 configuration and +7.62 pp over top-25 —
   the two strongest effects of the study — driven by context recall (+33.41 pp from
   top-10).

3. **Faithfulness is invariant to context size in the 10–100 chunk range** (88.8–91.0%, no
   significant pairwise difference), and hallucination *decreases* as the context grows.
   The classic coverage-vs-grounding trade-off does not materialise on this corpus.

4. **Beyond top-50, gains stop being provable.** Top-100's +4.49 pp is not significant
   while its precision losses are; combined with doubled prompt cost, top-50 is the
   rational operating point.

5. **Neither reranker refinement helps.** Combined ranking moves nothing on any of 17
   metrics; every score threshold is net-negative on the global score, significantly so at
   thr+1 — the only significant global regression measured in the study.

### 7.2 Notable Directional Patterns (consistent but not significant)

- **RRF > weighted fusion in all three field families** (+1.5 to +2.7 pp each), with the
  advantage growing with the number of fields to calibrate.
- **Fermi adds value only under RRF**, and complements BM25 rather than replacing it.
- **Moderate thresholding (−2/−1) achieves the study's lowest hallucination rates**
  (1.6–1.7%) at a small, unconfirmed global cost — an option for hallucination-critical
  deployments only.

### 7.3 Recommended Configuration

| Parameter | Value |
|-----------|-------|
| Ingestion | Structured pre-processing |
| Embedding fields | Dense (BGE-M3) + BM25 sparse + Fermi sparse |
| Rank fusion | Reciprocal Rank Fusion (k = 60) |
| Candidate pool (pre-reranking) | 100 |
| Chunks passed to the LLM | **50** |
| Final ranking | Cross-encoder score only |
| Score threshold | None |

**Global RAG Score: 65.55%** — +14.20 pp over the original baseline, with the top-n step
alone accounting for +12.31 pp (statistically significant, large effect).

### 7.4 Limitations and Next Steps

- **Statistical power.** With 65 question groups, effects below ~5 pp of global score are
  not reliably detectable. The top-50 vs. top-100 question, and any future fine-tuning,
  require extending the question bank (150–200 groups) before further parametric splitting.
- **Single generation LLM and single judge.** All conclusions hold for the fixed
  model/prompt pair; a change of generation model (especially context window and
  long-context behaviour) reopens the top-n question.
- **Corpus specificity.** The dominance of context recall and the stability of faithfulness
  reflect a dense, redundant regulatory corpus; transfer to other document bases should be
  re-validated.
- **Cost dimension.** Top-50 doubles the prompt size of top-25 for +7.62 pp. The study
  optimises quality only; a deployment arbitrage between latency/cost and score remains to
  be made explicitly.

---

## 8. How to Reproduce

Prerequisites: Milvus populated with the pre-processed corpus, `uv` environment, Cleyrop
proxy reachable.

**Champion configuration:**
```bash
uv run scripts/rag_evaluation/prepare_eval_dataset.py \
    scripts/rag_evaluation/data/rag_evaluation_dataset.xlsx \
    --experiment-id dbf_rrf_top50 \
    --ranker rrf --candidates 100 --top-n 50 --fermi-weight 0.5
uv run scripts/rag_evaluation/rag_evaluation.py --experiment-id dbf_rrf_top50
```

Generation also writes `eval_test_{exp}_contexts.parquet` (full contexts, reranker scores,
chunk names) next to the Excel; the evaluator reads it in priority. Retrieval-only replays
and top-n / threshold variant derivations are available via
`scripts/rag_evaluation/replay_retrieval.py`; threshold variants can be re-generated without
retrieval via `prepare_eval_dataset.py --contexts-parquet <base parquet> --reranker-threshold <v>`.

**Other configurations** (flags on top of the champion command):

| Run | Modification |
|-----|--------------|
| `dbf_rrf_top50_comb` | add `--ranking-mode combined` |
| `dbf_rrf_top50_thr{v}` | add `--reranker-threshold {v}` |
| `dbf_rrf_top{25,100}` | `--top-n 25` / `--top-n 100` |
| `db_rrf`, `df_rrf`, `dbf_w`… | adjust `--fermi-weight`, `--bm25-weight`, `--ranker` |

**Pairwise statistics:**
```bash
cd scripts/rag_evaluation && uv run compare_rag_runs.py <run_A> <run_B>
```

---

## Appendix A — Run Parameters

| Run | Fields | Fusion | Pool | Top-n | Ranking | Threshold |
|-----|--------|--------|:---:|:---:|---------|:---:|
| baseline | dense, bm25 | weighted | — | 10 | standard | none |
| baseline_raw | dense, bm25 | weighted | — | 10 | standard | none (raw ingestion) |
| db_rrf | dense, bm25 | rrf | — | 10 | standard | none |
| df_w | dense, fermi | weighted | — | 10 | standard | none |
| df_rrf | dense, fermi | rrf | — | 10 | standard | none |
| dbf_w | dense, bm25, fermi | weighted | — | 10 | standard | none |
| dbf_rrf | dense, bm25, fermi | rrf | — | 10 | standard | none |
| dbf_rrf_top25 | dense, bm25, fermi | rrf | 100 | 25 | standard | none |
| **dbf_rrf_top50** | dense, bm25, fermi | rrf | 100 | 50 | standard | none |
| dbf_rrf_top100 | dense, bm25, fermi | rrf | 100 | 100 | standard | none |
| dbf_rrf_top50_comb | dense, bm25, fermi | rrf | 100 | 50 | combined | none |
| dbf_rrf_top50_thr-2 | dense, bm25, fermi | rrf | 100 | 50 | standard | −2.0 |
| dbf_rrf_top50_thr-1 | dense, bm25, fermi | rrf | 100 | 50 | standard | −1.0 |
| dbf_rrf_top50_thr0 | dense, bm25, fermi | rrf | 100 | 50 | standard | 0.0 |
| dbf_rrf_top50_thr1 | dense, bm25, fermi | rrf | 100 | 50 | standard | +1.0 |

## Appendix B — Full Score Table (all runs, all reported metrics)

| Run | FC Recall | Faithful | Rob | Hall ↓ | Cit Recall | Ctx Recall | Ctx Rel | Ctx Cov | Diversity | Hub IA F | **Global** |
|-----|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| baseline | 44.87% | 88.09% | 87.66% | 5.06% | 26.72% | 38.48% | 9.59% | 28.99% | 26.69% | 31.43% | **51.35%** |
| baseline_raw | 45.66% | 77.49% | 87.45% | 4.59% | 18.26% | 41.72% | 7.63% | 24.25% | 33.28% | 28.89% | **48.52%** |
| db_rrf | 44.17% | 89.19% | 88.26% | 5.47% | 27.99% | 44.56% | 10.21% | 29.87% | 25.86% | 31.77% | **53.06%** |
| df_w | 45.46% | 86.05% | 87.37% | 5.58% | 21.50% | 42.37% | 8.26% | 23.04% | 25.59% | 32.90% | **50.91%** |
| df_rrf | 43.38% | 88.48% | 88.01% | 2.84% | 27.33% | 44.08% | 9.71% | 28.72% | 25.60% | 30.28% | **52.42%** |
| dbf_w | 45.67% | 83.48% | 88.18% | 4.13% | 24.16% | 40.34% | 9.42% | 27.36% | 26.07% | 30.08% | **50.56%** |
| dbf_rrf | 46.34% | 89.02% | 88.34% | 6.21% | 28.55% | 41.92% | 9.19% | 29.78% | 25.93% | 29.83% | **53.24%** |
| dbf_rrf_top25 | 49.55% | 89.62% | 88.92% | 6.37% | 33.10% | 52.86% | 6.29% | 35.20% | 27.41% | 28.71% | **57.93%** |
| **dbf_rrf_top50** | 54.40% | 88.83% | 88.79% | 3.86% | 36.47% | 75.32% | 4.98% | 40.49% | 29.26% | 27.89% | **65.55%** |
| dbf_rrf_top100 | 57.82% | 91.01% | 88.54% | 3.29% | 35.33% | 87.00% | 3.69% | 45.97% | 32.17% | 27.42% | **70.04%** |
| dbf_rrf_top50_comb | 55.76% | 92.48% | 88.54% | 4.35% | 36.39% | 75.00% | 4.84% | 42.07% | 29.26% | 29.07% | **66.84%** |
| dbf_rrf_top50_thr-2 | 55.84% | 89.04% | 88.19% | 1.62% | 31.79% | 71.66% | 5.15% | 38.88% | 28.63% | 30.45% | **64.49%** |
| dbf_rrf_top50_thr-1 | 53.28% | 89.68% | 87.87% | 1.73% | 35.16% | 66.00% | 5.60% | 38.42% | 27.67% | 27.82% | **62.84%** |
| dbf_rrf_top50_thr0 | 53.78% | 90.17% | 87.37% | 5.96% | 29.67% | 57.40% | 6.39% | 33.06% | 26.80% | 27.82% | **60.17%** |
| dbf_rrf_top50_thr1 | 47.18% | 87.97% | 86.77% | 5.77% | 27.36% | 51.77% | 7.56% | 29.73% | 26.10% | 26.07% | **55.55%** |

## Appendix C — All Statistically Significant Pairwise Findings

Two-tailed Wilcoxon signed-rank tests on 65 matched questions, Benjamini-Hochberg corrected
at α = 0.05. Effect size r is the rank-biserial correlation (|r| < 0.2 negligible, 0.2–0.5
moderate, > 0.5 large).

| Comparison | Metric | Direction | Δ | r | p_BH |
|-----------|--------|-----------|:---:|:---:|:---:|
| baseline vs baseline_raw | Diversity | raw | +6.59 pp | 0.839 | <0.001 |
| baseline vs baseline_raw | Faithfulness | baseline | +10.59 pp | 0.348 | 0.029 |
| baseline vs baseline_raw | Citation Recall | baseline | +8.46 pp | 0.350 | 0.029 |
| dbf_rrf vs top50 | Global RAG Score | top50 | +12.31 pp | 0.575 | <0.0001 |
| dbf_rrf vs top50 | Context Recall | top50 | +33.41 pp | 0.591 | <0.0001 |
| dbf_rrf vs top50 | Context Coverage | top50 | +10.71 pp | 0.381 | 0.005 |
| dbf_rrf vs top50 | Citation Recall | top50 | +7.91 pp | 0.350 | 0.010 |
| dbf_rrf vs top50 | Context Relevance | dbf_rrf | +4.21 pp | 0.491 | <0.001 |
| top25 vs top50 | Global RAG Score | top50 | +7.62 pp | 0.435 | 0.0025 |
| top25 vs top50 | Context Recall | top50 | +22.46 pp | 0.538 | 0.0001 |
| top25 vs top50 | Context Relevance | top25 | +1.32 pp | 0.333 | 0.031 |
| top50 vs top100 | Context Recall | top100 | +11.68 pp | 0.405 | 0.006 |
| top50 vs top100 | Context Coverage | top100 | +5.48 pp | 0.332 | 0.025 |
| top50 vs top100 | Context Relevance | top50 | +1.29 pp | 0.472 | 0.001 |
| top50 vs thr−2 | Diversity | top50 | +0.63 pp | 0.389 | 0.030 |
| top50 vs thr−1 | Diversity | top50 | +1.59 pp | 0.539 | <0.001 |
| top50 vs thr−1 | Robustness | top50 | +0.92 pp | 0.388 | 0.015 |
| top50 vs thr0 | Context Recall | top50 | +17.92 pp | 0.469 | 0.0013 |
| top50 vs thr0 | Context Coverage | top50 | +7.44 pp | 0.315 | 0.048 |
| top50 vs thr+1 | Global RAG Score | top50 | +10.00 pp | 0.399 | 0.0056 |
| top50 vs thr+1 | Context Recall | top50 | +23.55 pp | 0.443 | 0.0020 |
| top50 vs thr+1 | Citation Recall | top50 | +9.10 pp | 0.349 | 0.014 |
| top50 vs thr+1 | Context Coverage | top50 | +10.77 pp | 0.350 | 0.014 |
| top50 vs thr+1 | Robustness | top50 | +2.03 pp | 0.451 | 0.0020 |
| top50 vs thr+1 | Context Relevance | thr+1 | +2.58 pp | 0.311 | 0.036 |
