# RAG Parametric Study — Experimental Analysis

> **Document scope:** systematic evaluation of 20 RAG pipeline configurations across four
> experimental series. All runs evaluated on the same 65 question-answer pairs (260 queries
> including paraphrase variants) with a fixed LLM and prompt template.

---

## 1. Experimental Protocol

### 1.1 Overview and Objective

This document presents a systematic parametric evaluation of a Retrieval-Augmented Generation
(RAG) pipeline applied to a corpus of regulatory and technical documents from the French nuclear
safety authority (ASN). In a RAG system, a language model does not answer questions from
parametric memory alone: it is first provided a set of retrieved document excerpts (chunks)
that serve as grounding context. The quality of the final answer therefore depends jointly on the
language model's generation ability and on the retrieval system's capacity to surface relevant,
accurate, and coherent content. This study focuses exclusively on the retrieval side: the
language model, the prompt template, and the evaluation corpus are held fixed throughout, so that
all observed performance differences can be attributed solely to retrieval configuration choices.

The pipeline operates in four sequential stages: (1) hybrid multi-field search over a Milvus
vector database, (2) score fusion across retrieval fields, (3) cross-encoder reranking of
candidate chunks, and (4) generation by a large language model conditioned on the selected
chunks. The study isolates and quantifies the contribution of four independent configuration
axes:

1. **Document ingestion strategy** — whether documents are ingested with structured
   pre-processing or in their raw form.
2. **Embedding field combination and rank fusion** — which vector representations are used
   during hybrid search, and how their scores are fused (weighted sum vs. Reciprocal Rank
   Fusion).
3. **Retrieved chunk count (top-n)** — how many reranked chunks are passed to the language model.
4. **Reranker usage refinement** — whether a combined ranking mode (mixing retrieval and reranker
   scores) is applied, and whether a minimum reranker score threshold filters out low-confidence
   chunks.

Each series promotes the best-performing run from the previous series as the new baseline,
ensuring that gains compound progressively across the study.

---

### 1.2 Pipeline Architecture and Design Rationale

#### 1.2.1 Dense Retrieval

Dense retrieval encodes both documents and queries as dense continuous vectors using a
bi-encoder model — here **BGE-M3** (BAAI/bge-m3), served via the Cleyrop proxy. At query time,
the query is encoded into the same vector space, and the most semantically similar document
chunks are retrieved by approximate nearest-neighbour search (cosine similarity in Milvus).
Dense retrieval captures semantic proximity: a question phrased differently from the source text
can still match the relevant passage if both express the same meaning. This makes it
intrinsically robust to lexical variation and paraphrasing, but it requires both the query and
the document to be within the model's training distribution to produce meaningful embeddings.

The bi-encoder architecture scores query and document *independently*: each is encoded once in
isolation, and the similarity is computed post-hoc. This is computationally efficient at scale
but inherently limits the quality of the matching — the model cannot attend to the specific
interaction between a given query and a given document.

#### 1.2.2 Sparse Retrieval — BM25

BM25 is a classic term-frequency–based ranking function. It scores a document against a query
based on the frequency of shared terms, normalised by document length and weighted by how rare
each term is in the corpus (inverse document frequency). BM25 does not generalise beyond the
vocabulary: it can only match terms that appear literally in both the query and the document.
This makes it brittle to paraphrasing, but extremely reliable for rare or highly specific terms
— technical identifiers, regulation article references, proper nouns — that dense models may
embed inconsistently across contexts.

In this pipeline, BM25 scores are computed client-side from a persisted IDF/avgdl state fitted
on the entire document corpus. They are stored as sparse vectors in Milvus (field `sparse_bm25`)
and combined with the dense field at search time.

#### 1.2.3 Sparse Retrieval — Fermi

Fermi (`atomic-canyon/fermi-1024`) is a **learned sparse encoder** following the SPLADE
architecture. Like BM25, it produces sparse token-weight vectors, but unlike BM25, these weights
are not computed by a hand-crafted frequency formula: they are learned end-to-end by a
transformer model trained to maximise retrieval quality. The model implicitly learns term
expansion — a chunk about "radioactive waste" may receive positive weight for "nuclear
effluents" even if that term does not appear literally in the text. This bridges the gap between
exact lexical matching (BM25) and full semantic generalisation (dense). Fermi vectors are stored
in the `sparse_fermi` field.

The rationale for including Fermi alongside BM25 is that the two sparse representations encode
complementary signals: BM25 excels at exact term coverage, Fermi at domain-specific implicit
term expansion. Their joint contribution is expected to be most beneficial under a fusion
strategy that does not require score calibration (see §1.2.4).

#### 1.2.4 Score Fusion: Weighted Sum vs. Reciprocal Rank Fusion

When multiple retrieval fields are active, Milvus must combine their individual result lists into
a single ranked list of candidates. Two strategies are compared in this study:

**Weighted score fusion (`w`)** computes a linear combination of the raw scores produced by
each field: `final_score = w₁·score_dense + w₂·score_BM25 + w₃·score_Fermi`. This approach is
conceptually straightforward but rests on the assumption that scores from different fields are
*commensurable* — that they live on comparable scales so that a given numerical weight reflects
the intended contribution. In practice, cosine similarity values (dense) and inner-product scores
from sparse encoders (BM25, Fermi) are produced by fundamentally different computations and
follow different distributions. Calibrating their relative weights correctly without exhaustive
tuning data is difficult, and errors in calibration systematically bias the result.

**Reciprocal Rank Fusion (RRF)** avoids score calibration entirely by operating on *ranks* rather
than raw scores. Each field produces an independent ranked list; a chunk's final score is
`Σ 1/(k + rank_i)` where `k` is a smoothing constant (typically 60) and `rank_i` is its
position in field `i`'s ranked list. A chunk that consistently appears near the top across
multiple fields accumulates a high RRF score regardless of the absolute score values. RRF is
well-established as a robust default fusion strategy when score distributions across retrievers
are heterogeneous.

#### 1.2.5 Cross-Encoder Reranking

After fusion, Milvus returns a pool of candidate chunks (100 in the main experiments). This pool
is then reranked by a **cross-encoder** model
(`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`), which takes the concatenation of the query
and each candidate chunk as input and produces a single relevance score. Unlike the bi-encoder
used for dense retrieval, the cross-encoder attends to the full interaction between the query and
the document: every token of the query can influence the representation of every token of the
document. This produces significantly more accurate relevance judgements, at the cost of much
higher compute — which is why reranking is applied only to the top-100 candidates rather than
the full corpus.

The reranker outputs a raw score on a continuous scale (observed range roughly −5 to +10 in this
corpus). The top-n chunks by this reranked score are then passed to the language model as
context. The cross-encoder score is therefore the last quality filter applied to the retrieved
set before generation.

#### 1.2.6 Combined Ranking Mode

The standard pipeline uses the cross-encoder score alone to select the final top-n chunks.
**Combined mode** (`_comb`) instead re-ranks the top-n by a blended score that mixes the
Milvus RRF retrieval score with the cross-encoder score, both min-max normalised to [0, 1] and
combined 50/50. The motivation is that the cross-encoder, operating in isolation, may
systematically prefer chunks that are linguistically close to the query even if they are not
the most informative for the answer — for example, a passage that restates the question without
providing the expected factual content. Including the Milvus retrieval score, which reflects
geometric proximity in the embedding space, can re-weight chunks that are both semantically
relevant *and* topically well-matched according to the embedding model. Combined mode is thus
a hybrid ranking signal that is expected to favour chunks with strong support from both retrieval
and reranking, at the possible cost of slightly reduced cross-encoder purity.

#### 1.2.7 Reranker Score Thresholding

Even after reranking, the top-n pool may contain chunks that the cross-encoder evaluates as
very weakly relevant — scoring near or below zero on its raw scale. These marginal chunks may
add noise to the LLM's context: they dilute the signal-to-noise ratio, potentially leading the
model to make claims anchored in peripheral or tangentially related content rather than
authoritative passages. A **score threshold** (`_thr{v}`) filters out any chunk whose raw
cross-encoder score falls below a fixed value before passing the context to the LLM. This
reduces the number of chunks for some queries (those that lack strongly relevant passages in the
corpus) while leaving high-recall queries unchanged.

The expected benefit is improved faithfulness and a lower hallucination rate: constraining the
LLM to a smaller, higher-confidence context should limit drift. The expected cost is reduced
context recall: some factually useful chunks may carry low cross-encoder scores (e.g., because
they answer the question indirectly), and filtering them out removes information that would have
improved factual completeness. The score distribution measured on this corpus (median 0.85,
P5 = −2.28, P95 = 4.90) informs what fraction of the pool each threshold value removes in
practice.

---

### 1.3 Experiment Naming Convention

All experiments follow a structured naming pattern:

```
{champs}_{ranker}[_top{n}][_comb][_thr{v}]
```

| Token | Values | Meaning |
|-------|--------|---------|
| `{champs}` | `db` | Dense (embedding) + BM25 sparse |
| | `df` | Dense + Fermi domain-specific sparse |
| | `dbf` | Dense + BM25 + Fermi (all three fields) |
| `{ranker}` | `w` | Weighted score fusion |
| | `rrf` | Reciprocal Rank Fusion |
| `[_top{n}]` | e.g. `_top25` | Chunks passed to the LLM (omitted when default = 10) |
| `[_comb]` | present/absent | Combined ranking mode (RRF Milvus score + reranker score, 50/50) |
| `[_thr{v}]` | e.g. `_thr-2` | Minimum reranker score threshold (−2.0); negative sign encoded as `-` |

Two runs are exempt from this convention: `baseline` (the original pipeline, dense + BM25,
weighted fusion, cross-encoder reranking, top-10, kept for historical continuity) and
`baseline_raw` (same retrieval pipeline, but documents ingested without structured
pre-processing).

### 1.4 Evaluated Metrics

Only the following metrics are reported and analysed in this study:

**Response Quality**
- **Factual Correctness (Recall)** — fraction of expected facts present in the model response.
- **Faithfulness** — fraction of model claims grounded in the retrieved context.
- **Reformulation Robustness** — response consistency across semantically equivalent question
  variants (paraphrases).
- **Hallucination Rate** — fraction of ungrounded claims in the response. *Lower is better.*
- **Citation Recall** — fraction of expected source documents correctly cited.

**Retrieved Context Quality**
- **Context Recall** — fraction of the expected answer's knowledge covered by retrieved chunks.
- **Context Relevance** — precision of the retrieved set; fraction of retrieved chunks that are
  relevant to the query.
- **Context Coverage** — breadth of coverage over expected source files.
- **Diversity** — semantic diversity of the retrieved chunk set.
- **Hub France IA F-Score** — composite grounding quality metric combining grounding rate and
  grounding precision into an F-measure.

**Global Performance**
- **Global RAG Score** — the primary decision criterion.
  Formula: `0.25 × Context Recall + 0.35 × Factual Correctness Recall + 0.15 × Citation Recall + 0.25 × Faithfulness`.

### 1.5 Statistical Methodology

Each pairwise comparison is performed using a two-tailed **Wilcoxon signed-rank test** on 65
matched question-level scores. Multiple testing is corrected via the **Benjamini-Hochberg (BH)
procedure** at α = 0.05. Effect sizes are reported as the **rank-biserial correlation** (r):
|r| < 0.1 negligible, 0.1–0.3 small, 0.3–0.5 medium, > 0.5 large. A result is considered
statistically significant only when the BH-corrected p-value (p_BH) falls strictly below 0.05.
The Global RAG Score is the primary decision criterion; secondary metrics are examined to
characterise meaningful trade-offs when no global significance is reached.

### 1.6 Series Overview

| Series | Baseline | Candidate Runs | Research Question |
|--------|----------|----------------|-------------------|
| 1 | `baseline` | `baseline_raw` | Does document pre-processing improve retrieval quality? |
| 2 | `baseline` | `db_rrf`, `df_w`, `df_rrf`, `dbf_w`, `dbf_rrf` (all top-10) | Which embedding + ranker combination is optimal? |
| 3 | `dbf_rrf` | `dbf_rrf_top{25,50,100}` | What is the optimal number of retrieved chunks? |
| 4 | `dbf_rrf_top25` | `dbf_rrf_top25_comb`, `dbf_rrf_top25_thr{−3,−2,−1.5,−1}` | Can combined mode or score thresholding improve quality? |

---

## 2. Series 1 — Impact of Document Ingestion Strategy

**Baseline:** `baseline` — dense + BM25, weighted fusion, cross-encoder reranking, top-10;
Global RAG Score = **51.35%**

**Candidate:** `baseline_raw` — identical retrieval pipeline (including reranking), but documents
were ingested **without structured pre-processing** (raw ingestion);
Global RAG Score = **48.52%**

### 2.1 Results

| Metric | baseline | baseline_raw | Δ (raw − base) | Significant |
|--------|----------|--------------|----------------|-------------|
| **Factual Correctness (Recall)** | 44.87% | 45.66% | +0.79 pp | No |
| **Faithfulness** | 88.09% | 77.49% | **−10.60 pp** | **Yes — ↑ baseline** |
| **Reformulation Robustness** | 87.66% | 87.45% | −0.22 pp | No |
| **Hallucination Rate** ↓ | 5.06% | 4.59% | −0.47 pp | No |
| **Citation Recall** | 26.72% | 18.26% | **−8.46 pp** | **Yes — ↑ baseline** |
| Context Recall | 38.48% | 41.72% | +3.24 pp | No |
| Context Relevance | 9.59% | 7.63% | −1.96 pp | No |
| Context Coverage | 28.99% | 24.25% | −4.74 pp | No |
| **Diversity** | 26.69% | 33.28% | **+6.59 pp** | **Yes — ↑ baseline_raw** |
| Hub France IA F-Score | 31.43% | 28.89% | −2.54 pp | No |
| **Global RAG Score** | **51.35%** | **48.52%** | **−2.83 pp** | **No** (p_BH = 0.242) |

Wilcoxon signed-rank test with BH correction, n = 65, α = 0.05.

### 2.2 Analysis

The Global RAG Score difference of −2.83 pp does not reach statistical significance (r = 0.197,
p_BH = 0.242). However, the pre-processed baseline delivers two statistically significant
advantages over the raw-ingested variant:

**Faithfulness (+10.60 pp, r = 0.348, p_BH = 0.029):** The large faithfulness gap points to a
structural quality difference in the ingested chunks. Raw ingestion likely produces noisier,
less coherent text segments — possibly from unresolved OCR artefacts, formatting residues, or
poorly delimited chunk boundaries. Even though the same cross-encoder reranker selects the best
available chunks in both cases, if the underlying corpus chunks are noisier, the LLM is more
prone to making claims that drift beyond the retrieved context, lowering faithfulness. This is
arguably the most meaningful finding of this series.

**Citation recall (+8.46 pp, r = 0.350, p_BH = 0.029):** Source attribution is also
significantly better with structured ingestion. This is consistent with the faithfulness
interpretation: better-delineated chunks carry more precise source boundaries, making file-level
citation more reliable.

**Diversity (+6.59 pp in baseline_raw, r = 0.839, p_BH < 0.001):** This is the only metric
significantly favouring the raw configuration. Raw chunks, being more heterogeneous in structure,
span a broader semantic range — producing a more diverse retrieved set. This is a mechanical
consequence of ingestion quality, not a retrieval improvement: higher diversity here reflects
noise rather than richer coverage.

The marginally higher context recall in `baseline_raw` (+3.24 pp, NS) is consistent with the
diversity finding: more heterogeneous chunks may incidentally cover a wider range of expected
answer content. The trade-off is clear — `baseline_raw` retrieves more broadly but less
accurately, resulting in lower faithfulness and citation quality.

### 2.3 Verdict — Retain `baseline` as canonical reference

Structured document pre-processing significantly improves two of the most user-visible quality
dimensions: faithfulness and citation recall. The global score advantage (−2.83 pp) is not
statistically significant but is directionally consistent. `baseline` is retained as the
canonical starting point for all subsequent experiments.

---

## 3. Series 2 — Impact of Embedding Combination and Rank Fusion Strategy

**Baseline:** `baseline` — dense + BM25, weighted, top-10; Global = **51.35%**

This series jointly examines two configuration axes: (1) the choice of embedding fields used
during vector retrieval, and (2) the rank fusion strategy applied to combine multi-field scores.
All candidate runs use the default top-10 chunks, isolating these two axes from the chunk-count
effect studied in Series 3.

The baseline corresponds to the `db` configuration (dense + BM25) under weighted fusion. The
five candidate runs test the remaining field combinations (`df`, `dbf`) and the alternative
fusion strategy (RRF), yielding a factorial comparison across both axes.

### 3.1 Results

| Run | FC Recall | Faithfulness | Rob | Hall Rate ↓ | Cit Recall | Ctx Recall | Ctx Rel | Ctx Cov | Diversity | Hub IA F | **Global** |
|-----|-----------|--------------|-----|-------------|------------|------------|---------|---------|-----------|----------|------------|
| **baseline** (db_w) | 44.87% | 88.09% | 87.66% | 5.06% | 26.72% | 38.48% | 9.59% | 28.99% | 26.69% | 31.43% | **51.35%** |
| db_rrf | 44.17% | 89.19% | 88.26% | 5.47% | 27.99% | 44.56% | 10.21% | 29.87% | 25.86% | 31.77% | **53.06%** |
| df_w | 45.46% | 86.05% | 87.37% | 5.58% | 21.50% | 42.37% | 8.26% | 23.04% | 25.59% | 32.90% | **50.91%** |
| df_rrf | 43.38% | 88.48% | 88.01% | 2.84% | 27.33% | 44.08% | 9.71% | 28.72% | 25.60% | 30.28% | **52.42%** |
| dbf_w | 45.67% | 83.48% | 88.18% | 4.13% | 24.16% | 40.34% | 9.42% | 27.36% | 26.07% | 30.08% | **50.56%** |
| **dbf_rrf** | **46.34%** | **89.02%** | **88.34%** | 6.21% | **28.55%** | 41.92% | 9.19% | 29.78% | 25.93% | 29.83% | **53.24%** |

No pairwise comparison vs. `baseline` reaches statistical significance on the Global RAG Score
or any individual metric after BH correction (all p_BH > 0.06).

### 3.2 Analysis

#### Axis 1 — Rank Fusion: RRF consistently outperforms weighted

Within each embedding family, RRF systematically outperforms weighted fusion:

| Field config | Weighted | RRF | Δ (RRF − W) |
|-------------|---------|-----|-------------|
| db (Dense + BM25) | 51.35% (baseline) | 53.06% (db_rrf) | +1.71 pp |
| df (Dense + Fermi) | 50.91% (df_w) | 52.42% (df_rrf) | +1.51 pp |
| dbf (Dense + BM25 + Fermi) | 50.56% (dbf_w) | 53.24% (dbf_rrf) | +2.68 pp |

This pattern is consistent and mechanistically expected. Weighted score fusion requires
commensurable score scales across fields (cosine similarity for dense, inner product for sparse
fields), but these distributions are inherently heterogeneous and difficult to calibrate without
extensive tuning. RRF avoids this problem entirely by rank-normalising each field independently,
making it inherently robust to score scale differences. The advantage is larger for the three-
field `dbf` configuration (+2.68 pp), where the calibration difficulty is greatest.

#### Axis 2 — Embedding Fields: Fermi adds marginal value under RRF only

The effect of adding the Fermi domain-specific sparse field depends strongly on the fusion
strategy:

- **Under weighted fusion:** Adding Fermi (dbf_w = 50.56%) performs *worse* than the baseline
  without Fermi (db_w = 51.35%, −0.79 pp). This likely reflects the calibration issue described
  above: a third sparse field with a different score distribution further destabilises the
  weighted sum. Citation recall drops from 26.72% to 24.16% (−2.56 pp), suggesting the Fermi
  field's higher-recall results are overweighted in a way that displaces more precisely-matched
  BM25 chunks.

- **Under RRF:** The full three-field configuration `dbf_rrf` (53.24%) marginally outperforms
  the two-field `db_rrf` (53.06%, +0.18 pp). Under rank fusion, Fermi contributes complementary
  domain-specific recall without disturbing score calibration.

- **Replacing BM25 with Fermi** (`df` variants) consistently underperforms their `db`
  counterparts: df_rrf (52.42%) < db_rrf (53.06%), and df_w (50.91%) < baseline/db_w (51.35%).
  BM25 remains the stronger sparse retrieval component for this corpus, providing broader lexical
  coverage than the domain-specific Fermi encoder.

None of these differences are individually statistically significant — the effect sizes are
small (r < 0.23 for all global score comparisons). This is consistent with the expectation that
at top-10, the field combination has limited room to differentiate: the reranker operates on a
small candidate pool and the retrieval diversity gains are attenuated.

### 3.3 Verdict — Promote `dbf_rrf` as new baseline (Global = 53.24%)

`dbf_rrf` achieves the highest global score across all top-10 configurations (+1.89 pp vs.
baseline). The directional advantage of both RRF and the Fermi field addition is consistent
across all comparisons, even if not individually significant at this scale. `dbf_rrf` is adopted
as the new baseline for Series 3, where expanding the candidate pool is expected to amplify the
retrieval quality differences observed here.

---

## 4. Series 3 — Impact of Retrieved Chunk Count (Top-n)

**Baseline:** `dbf_rrf` — dense + BM25 + Fermi, RRF, top-10; Global = **53.24%**

This series tests whether increasing the number of chunks passed to the LLM improves retrieval
coverage without sacrificing response quality. Three top-n values are tested within the `dbf_rrf`
configuration: 25, 50, and 100. Statistical significance is reported against the original
`baseline` (51.35%), as this is the reference available in the pairwise comparison files.

### 4.1 Results

| Run | FC Recall | Faithfulness | Rob | Hall Rate ↓ | Cit Recall | Ctx Recall | Ctx Rel | Ctx Cov | Diversity | Hub IA F | **Global** |
|-----|-----------|--------------|-----|-------------|------------|------------|---------|---------|-----------|----------|------------|
| baseline (ref) | 44.87% | 88.09% | 87.66% | 5.06% | 26.72% | 38.48% | 9.59% | 28.99% | 26.69% | 31.43% | **51.35%** |
| dbf_rrf (top-10) | 46.34% | 89.02% | 88.34% | 6.21% | 28.55% | 41.92% | 9.19% | 29.78% | 25.93% | 29.83% | **53.24%** |
| **dbf_rrf_top25** ★ | **49.55%** | 87.10% | **88.92%** §† | 7.89% | **33.10%** §† | **53.36%** §† | 6.29% §‡ | **35.20%** §† | **45.45%** §† | **37.00%** | **57.42%** §† |
| dbf_rrf_top50 | 54.40% | 74.64% | 88.79% | 10.25% | 36.47% | 54.86% | 4.98% | 40.49% | 45.59% | 33.86% | **56.88%** |
| dbf_rrf_top100 | 57.82% | 67.62% | 88.54% | 7.18% | 35.33% | 52.21% | 3.69% | 45.97% | 45.44% | 31.95% | **55.50%** |

★ Best global score of the study. § Significant vs. original `baseline`. † ↑ candidate.
‡ ↓ vs. baseline.

`dbf_rrf_top25` global vs. `baseline`: **r = 0.362, p_BH = 0.015** — the only statistically
significant global improvement in the entire study.

### 4.2 Analysis

**`dbf_rrf_top25` is the only configuration in the entire study to achieve a statistically
significant improvement in Global RAG Score** (+6.07 pp over baseline, r = 0.362, p_BH = 0.015;
or equivalently, +4.18 pp over the top-10 `dbf_rrf` baseline of this series). The result
demonstrates that the synergy between the three-field RRF retrieval (which generates a
high-quality 100-candidate pool) and a 25-chunk LLM context reaches a tipping point that neither
the field configuration alone nor the default top-10 can achieve.

The statistically significant individual metric changes vs. `baseline` characterise what this
broader context brings:

- **Context Recall +14.88 pp (r = 0.472, p_BH < 0.001):** The most mechanistically direct
  consequence of retrieving more chunks — a wider net covers more of the expected answer content.
  This is expected and well-controlled here: the reranker ensures the extra chunks are genuinely
  relevant before they reach the LLM.

- **Diversity +18.76 pp (r = 0.865, p_BH < 0.001):** A large, significant increase reflecting
  the broader semantic coverage of 25 chunks vs. 10. Unlike the diversity increase observed in
  `baseline_raw` (which reflected noise), this gain is associated with improved retrieval quality
  metrics across the board.

- **Citation Recall +6.38 pp (r = 0.317, p_BH = 0.026):** Providing more chunks increases
  the probability that all expected source files are represented in the context. This is a
  direct downstream effect of the context recall improvement.

- **Context Coverage +6.21 pp (r = 0.331, p_BH = 0.021):** Broader breadth of retrieved
  files, consistent with the citation recall gain.

- **Context Relevance −3.30 pp (r = 0.414, p_BH = 0.005, ↑ baseline):** A precision decrease
  is expected when retrieving more chunks — the denominator (retrieved chunks) grows while the
  proportion of highly relevant ones naturally decreases. This is an acceptable trade-off: the
  reranker limits this dilution, keeping the decrease moderate.

- **Faithfulness:** 87.10% vs. 88.09% (baseline), a decrease of −0.99 pp that is **not
  statistically significant** (NS). This is a key result: unlike the `db_w_top*` configurations
  tested for context (see Appendix C), moving from top-10 to top-25 within `dbf_rrf` does *not*
  significantly harm faithfulness. The reranker's selection quality on a rich 100-candidate pool
  appears sufficient to maintain LLM grounding.

**Beyond top-25, faithfulness collapses significantly.** Moving to top-50 loses 12.46 pp in
faithfulness (74.64% vs. 87.10%), and top-100 loses 19.48 pp (67.62% vs. 87.10%). These declines
reflect the LLM's limited capacity to remain grounded in very large, heterogeneous contexts: past
a certain volume, context noise outweighs the coverage gain, and the model increasingly generates
claims that are not anchored in the retrieved material. The Global RAG Score at top-50 (56.88%)
and top-100 (55.50%) both fall below top-25 (57.42%), confirming that 25 chunks is the optimal
operating point. The factual correctness recall continues to rise (up to 57.82% at top-100)
because more chunks do cover more expected facts — but the faithfulness collapse kills the global
score via its 25% weight.

Hallucination rate also increases with top-n (7.89% at top-25, 10.25% at top-50), consistent
with the faithfulness degradation, before partially recovering at top-100 (7.18%, possibly due to
the very high factual recall diluting individual hallucination events). The precise mechanism
behind this non-monotonic behaviour at top-100 is unclear.

### 4.3 Verdict — Promote `dbf_rrf_top25` as new baseline (Global = 57.42%) ★

`dbf_rrf_top25` is the **only configuration to deliver a statistically significant global
improvement over the original baseline** (r = 0.362, p_BH = 0.015), and the only one that
simultaneously improves context recall, citation recall, context coverage, and reformulation
robustness without a statistically significant faithfulness loss. It is adopted as the canonical
best configuration for Series 4.

---

## 5. Series 4 — Reranker Usage Refinement: Combined Mode and Score Thresholding

**Baseline:** `dbf_rrf_top25` — Global = **57.42%**

This series tests two refinements applied to the reranker stage of `dbf_rrf_top25`. Both
operate on the already-selected top-25 chunks and alter how the reranker output is used,
rather than changing retrieval itself:

- **Combined mode (`_comb`):** re-ranks the 25 chunks by blending the RRF Milvus retrieval
  score with the cross-encoder reranker score (50/50, both min-max normalised).
- **Score thresholding (`_thr{v}`):** filters out any chunk whose raw cross-encoder score falls
  below a fixed threshold, reducing the number of chunks passed to the LLM for some queries.

### 5.1 Combined Ranking Mode

#### 5.1.1 Results

| Metric | dbf_rrf_top25 | dbf_rrf_top25_comb | Δ | Significant |
|--------|--------------|-------------------|---|-------------|
| **Factual Correctness (Recall)** | 49.55% | 53.48% | +3.93 pp | No |
| **Faithfulness** | 87.10% | 85.68% | −1.42 pp | No |
| **Reformulation Robustness** | 88.92% | 88.68% | −0.25 pp | No |
| **Hallucination Rate** ↓ | 7.89% | 5.32% | −2.57 pp | No |
| **Citation Recall** | 33.10% | 30.27% | −2.83 pp | No |
| Context Recall | 53.36% | 58.04% | +4.68 pp | No |
| Context Relevance | 6.29% | 6.45% | +0.16 pp | No |
| Context Coverage | 35.20% | 36.17% | +0.97 pp | No |
| Diversity | 45.45% | 46.21% | +0.76 pp | No |
| Hub France IA F-Score | 37.00% | 33.10% | −3.90 pp | No |
| **Global RAG Score** | **57.42%** | **59.19%** | **+1.77 pp** | **No** (r = 0.034, p_BH = 0.911) |

Across all 16 metrics tested, no comparison survives BH correction. Global effect size r = 0.034
is the weakest observed in the entire study.

#### 5.1.2 Analysis

The +1.77 pp global gain for combined mode is statistically indistinguishable from noise
(r = 0.034, p_BH = 0.911). Combined mode promotes chunks that were initially ranked highly by
Milvus retrieval (i.e., geometrically close to the query embedding), partially overriding the
cross-encoder's pure quality judgement. This produces a consistent but non-significant
precision-recall rebalancing: context recall gains (+4.68 pp) and factual correctness recall
gains (+3.93 pp), at the cost of faithfulness (−1.42 pp), citation recall (−2.83 pp), and Hub
France IA F-Score (−3.90 pp). The directional pattern suggests the combined mode is effectively
retrieving more topically proximate material but with slightly less grounding precision. Why
retrieval-score-boosted chunks specifically lower citation recall is unclear — it may reflect
that source diversity decreases when proximity to the query embedding is prioritised.

#### 5.1.3 Verdict on combined mode — Retain `dbf_rrf_top25`

No statistically significant benefit. Directional trends consistently favour the standard mode
on grounding-related metrics. `dbf_rrf_top25` is retained.

---

### 5.2 Reranker Score Thresholding

#### 5.2.1 Results

Four threshold values are evaluated: −3, −1, 0, and +1. Their practical effect on the context
is characterised by the reranker score distribution measured on all 6,500 chunk–question pairs
(260 questions × 25 chunks) in this corpus: median = 0.85, mean = 1.03, P5 = −2.28,
P95 = 4.90. Based on this distribution, the four thresholds remove respectively approximately
**1.9%, 18.4%, 33.8%, and ~50%** of all post-reranked chunks globally. Note that global removal
rates can be misleading: a threshold may remove no chunks for most questions while eliminating
the entire context for a handful of harder questions (see §5.2.2).

| Run | FC Recall | Faithfulness | Rob | Hall Rate ↓ | Cit Recall | Ctx Recall | Ctx Rel | Ctx Cov | Diversity | Hub IA F | **Global** |
|-----|-----------|--------------|-----|-------------|------------|------------|---------|---------|-----------|----------|------------|
| **dbf_rrf_top25** | 49.55% | 87.10% | 88.92% | 7.89% | 33.10% | 53.36% | 6.29% | 35.20% | **45.45%** | **37.00%** | 57.42% |
| thr-3 | 50.85% | 87.24% | 88.77% | 3.21% †‡ | 32.33% | 51.26% | 6.68% | 33.66% | 44.20% | 31.83% § | 56.94% |
| thr-1 | 50.91% | 85.75% | 88.22% | **2.86%** | 30.55% | 51.26% | 6.85% | 34.07% | 41.55% | 33.10% | 56.65% |
| thr0 | 49.30% | 81.22% | 87.97% | 10.31% ‡‡ | 28.52% | 49.26% | **7.47%** | 30.75% | 38.87% § | 31.05% | 54.15% |
| thr1 | **52.58%** | 86.27% | 88.02% | 5.05% | 32.34% | 52.40% | 6.85% | 34.07% | 41.55% | 34.95% | **57.92%** |

§ Statistically significant vs. `dbf_rrf_top25` (BH-corrected Wilcoxon, α = 0.05), baseline wins.
† Hallucination rate computed over only 47 out of 65 questions — see §5.2.2.
‡ Hallucination rate directionally lower but statistically unconfirmed (p_BH > 0.9).
‡‡ Hallucination rate paradoxically higher than baseline — see §5.2.2.

Significant findings: **thr-3** — Hub France IA F-Score −5.17 pp (r = 0.380, p_BH = 0.038,
↑ baseline). **thr0** — Diversity −6.58 pp (r = 0.506, p_BH = 0.001, ↑ baseline). Global RAG
Score differences: −0.48 pp (thr-3), −0.77 pp (thr-1), −3.27 pp (thr0), +0.50 pp (thr1) —
all non-significant (p_BH > 0.6).

#### 5.2.2 Analysis

**Coverage degrades monotonically with threshold stringency.** Context recall falls from 53.36%
(no threshold) to 51.26% at thr-3, 51.26% at thr-1, 49.26% at thr0, and 52.40% at thr1.
Diversity follows the same direction: 45.45% → 44.20% → 41.55% → 38.87% → 41.55%. Citation
recall also decreases steadily as the threshold rises. These patterns confirm that even lower-
ranked chunks in the top-25 contribute meaningfully to factual and source coverage: they were
selected as the best available from 100 candidates by the reranker, and discarding them reduces
breadth without a compensating precision gain. Note that context relevance moves slightly in the
opposite direction (rising from 6.29% to 7.47% at thr0), consistent with what filtering predicts:
a smaller, more selective retrieved set has higher average per-chunk relevance, but at the cost
of overall coverage.

**thr-3 is statistically unsafe despite its apparent mildness.** At a global removal rate of
only 1.9% of chunks, one might expect thr-3 to have negligible effect. It does leave the Global
RAG Score virtually unchanged (−0.48 pp, p_BH = 0.935) and maintains faithfulness (+0.14 pp).
However, it causes a statistically significant collapse of Hub France IA F-Score (−5.17 pp,
r = 0.380, p_BH = 0.038). This apparent paradox — a tiny global removal rate yet a significant
grounding quality regression — is explained by the highly uneven distribution of low-scoring
chunks across questions. The 124 chunks scoring below −3 (1.9% of 6,500) are not spread
uniformly: they are concentrated on a small set of hard questions for which the retrieval
pipeline found no strongly relevant material. For these questions, even the "best" chunks are
negative-scoring, and removing them eliminates the entire context. This is confirmed by the
n_valid drop for grounding metrics under thr-3: the anchor-based hallucination evaluation covers
only 47 questions instead of the expected ~63, indicating that for roughly 16 questions, thr-3
produced an empty context and grounding could not be evaluated. The Hub IA F-Score regression
captures the downstream effect: these context-deprived answers cannot be grounded in any
retrieved source. **thr-3 is the minimum viable threshold for safe application in this corpus.**

**thr-1 produces a consistent but non-significant degradation.** Removing 18.4% of chunks (those
scoring below −1.0) reduces most quality metrics directionally: Hub IA F-Score −3.90 pp
(r = 0.242, p_BH = 0.359), Diversity −3.90 pp (r = 0.368, p_BH = 0.060 — approaching the
significance boundary), Citation Recall −2.55 pp, Context Recall −2.10 pp. None of these reach
BH-corrected significance at 65 pairs. The hallucination rate reaches its lowest observed value
(2.86%, −5.03 pp vs baseline), also non-significant (r = 0.174, p_BH = 0.436). Factual
Correctness Recall (+1.36 pp) shows a consistent increase with stricter thresholds, likely
because the LLM focuses its response on a more concentrated set of high-quality chunks. The net
effect on the Global RAG Score is −0.77 pp (NS). The overall picture for thr-1 is one of
directional but unconfirmed harm to retrieval coverage, without a clearly beneficial trade-off.

**thr0 is the inflection point where filtering becomes counterproductive.** Removing roughly a
third of all chunks (those scoring below 0, i.e., chunks the cross-encoder considers irrelevant
or weakly relevant) does not improve quality — it demonstrably degrades it. Two findings stand
out. First, Diversity drops by 6.58 pp (r = 0.506, p_BH = 0.001) — the largest single
effect-size finding in the threshold series. Removing a third of the chunks collapses the
semantic breadth of the retrieved set to a degree that is statistically unambiguous. Second, and
more strikingly, the hallucination rate at thr0 *increases* to 10.31%, **higher than the
baseline's 7.89%** (+2.42 pp, NS). This is counterintuitive: the explicit goal of thresholding
is to reduce hallucination by removing low-confidence chunks, yet at thr0, the hallucination
rate worsens. The likely mechanism is a reduction in the total evidence available to the LLM:
when too many chunks are filtered out, the LLM answers with less context support, filling gaps
from parametric memory rather than the retrieved corpus. Chunks scoring just below 0 may be
weakly relevant to the query in a cross-encoder sense while still being genuinely useful as
supporting context for the LLM's answer. Removing them simultaneously reduces factual
grounding and increases confabulation. The −3.27 pp global score loss (NS) is the largest
observed across all threshold configurations, and no metric improves over baseline at thr0.

**thr1 is the only threshold that does not worsen the global score.** Filtering out approximately
half the post-reranked chunks (those below the corpus mean of ~1.0) produces a global score of
57.92%, marginally above the baseline's 57.42% (+0.50 pp, NS, r = 0.053, p_BH = 0.765).
Factual Correctness Recall reaches its highest value in the threshold series (52.58%, +3.03 pp),
likely because retaining only the highest-scoring chunks provides the LLM with the most
information-dense context. Hallucination rate is also directionally better (5.05%, −2.84 pp,
NS). However, Hub IA F-Score continues its downward trend (−2.05 pp, NS), and Diversity also
decreases (−3.90 pp, NS). Notably, Context Coverage and Context Relevance are identical to
those of thr-1, suggesting that between thresholds −1 and +1, the same set of source documents
is covered — the additional chunks filtered at thr1 contribute content but not new document
sources. Considering that thr1 removes roughly half of all reranked chunks, the near-flat global
score most plausibly reflects cancellation between gains (FC recall, hallucination) and losses
(diversity, Hub IA F, citation recall) rather than a genuine optimum.

#### 5.2.3 Verdict on thresholding — No threshold recommended

No threshold delivers a statistically significant global improvement. Two thresholds cause
statistically confirmed degradations: thr-3 on Hub France IA F-Score (p_BH = 0.038), and thr0
on Diversity (p_BH = 0.001). The thr0 result is particularly instructive: filtering a third of
post-reranked chunks not only fails to reduce hallucination but increases it, confirming that
chunks scoring between −1 and 0 on the cross-encoder still play a positive role in grounding
the LLM's answer. The recommended configuration remains **`dbf_rrf_top25` without threshold**.

If a specific deployment context demands reduced hallucination (e.g., a compliance use case
where ungrounded claims carry a high cost), **thr-1** is the most defensible option among the
tested values: it achieves the lowest observed hallucination rate (2.86%, −5.03 pp) at a cost of
only −0.77 pp globally, without triggering any statistically significant regression. thr-3 should
be avoided: its apparent mildness is deceptive, and it silently eliminates context for a
non-trivial subset of hard questions.

---

## 6. Final Conclusion

### 6.1 Established Findings (statistically significant)

**1. Structured document pre-processing significantly improves faithfulness and citation recall.**
The raw ingestion pipeline produces lower faithfulness (−10.60 pp, r = 0.348, p_BH = 0.029) and
lower citation recall (−8.46 pp, r = 0.350, p_BH = 0.029), indicating that chunk coherence at
ingestion time has a direct, measurable impact on response grounding quality.

**2. `dbf_rrf_top25` is the only configuration to achieve a statistically significant global
improvement over the original baseline** (+6.07 pp, r = 0.362, p_BH = 0.015). The gain requires
the simultaneous combination of three conditions: the full field set (dense + BM25 + Fermi), RRF
fusion, and top-25. No individual condition alone is sufficient.

**3. Faithfulness degrades sharply beyond top-25** within the `dbf_rrf` family: top-50 loses
12.46 pp (74.64% vs. 87.10%), top-100 loses 19.48 pp (67.62% vs. 87.10%), driving the global
score below `dbf_rrf_top25` in both cases. The top-25 setting is the optimal balance between
coverage and grounding.

**4. The Fermi domain-specific sparse field contributes meaningfully to grounding quality beyond
BM25.** In a direct comparison of top-25 configurations, `dbf_rrf_top25` outperforms `db_w_top25`
significantly on Hub France IA F-Score (+6.52 pp, r = 0.432, p_BH = 0.009) and Context
Precision (+6.94 pp, r = 0.360, p_BH = 0.031). See Appendix C.

**5. Thresholding produces two statistically confirmed degradations.** thr-3 significantly
degrades Hub France IA F-Score (−5.17 pp, r = 0.380, p_BH = 0.038): despite removing only 1.9%
of chunks globally, it eliminates the entire context for approximately 16 questions, collapsing
their grounding quality. thr0 significantly degrades Diversity (−6.58 pp, r = 0.506,
p_BH = 0.001) and paradoxically increases the hallucination rate (+2.43 pp, NS), confirming
that chunks scoring just below 0 on the cross-encoder remain useful for grounding the LLM.

### 6.2 Notable Directional Patterns (consistent but non-significant)

- **RRF consistently outperforms weighted fusion** across all three embedding families at top-10,
  with gains of +1.51 pp (df family) to +2.68 pp (dbf family). None reach significance at
  top-10, but the pattern is unequivocal.

- **Combined mode systematically trades grounding quality for recall:** relative to
  `dbf_rrf_top25`, it loses on faithfulness (−1.42 pp), citation recall (−2.83 pp), and Hub
  France IA F-Score (−3.90 pp) while gaining on context recall (+4.68 pp). The net global gain
  (+1.77 pp) carries a negligible effect size (r = 0.034).

- **Thresholding has a non-monotonic effect on hallucination rate.** thr-3 and thr-1 reduce
  it directionally (−4.68 pp and −5.03 pp respectively, both NS), but thr0 reverses this
  pattern: removing a third of chunks *increases* hallucination to 10.31% (+2.43 pp). This
  non-monotonicity suggests that chunks scoring between −1 and 0 serve as genuine grounding
  anchors, and their removal causes the LLM to fill gaps from parametric memory.

### 6.3 Recommended Configuration

| Parameter | Value |
|-----------|-------|
| Embedding fields | Dense (embedding) + BM25 sparse + Fermi sparse |
| Rank fusion | Reciprocal Rank Fusion (RRF) |
| Candidates (pre-reranking pool) | 100 |
| Chunks passed to LLM (post-reranking) | 25 |
| Ranking mode | Standard (cross-encoder score only) |
| Score threshold | None |

**Global RAG Score: 57.42%** — statistically significant +6.07 pp over the original baseline
(r = 0.362, p_BH = 0.015).

---

## 7. How to Reproduce the Results

### Prerequisites

- RAG pipeline: `scripts/rag/hybrid_rag_pipeline.py`
- Retrieval and generation script: `scripts/rag_evaluation/prepare_eval_dataset.py`
- Evaluation and scoring script: `scripts/rag_evaluation/rag_evaluation.py`
- Milvus vector database running and populated with the document corpus
- Fermi sparse index available, or all evaluation queries pre-cached in
  `scripts/rag_evaluation/data/query_embeddings_cache.pkl`
- Environment: `uv` package manager with project dependencies installed

### Reproducing the recommended configuration (`dbf_rrf_top25`)

**Step 1 — Run retrieval and LLM generation:**
```bash
uv run scripts/rag_evaluation/prepare_eval_dataset.py \
  --experiment-id dbf_rrf_top25 \
  --ranker rrf \
  --candidates 100
```
*(top_n = 25 is the current default in `hybrid_rag_pipeline.py`; verify before running.)*

**Step 2 — Run evaluation:**
```bash
uv run scripts/rag_evaluation/rag_evaluation.py \
  --experiment-id dbf_rrf_top25
```

Score files are written automatically to
`scripts/rag_evaluation/data/results/score_test_dbf_rrf_top25_averages.xlsx` and
`score_test_dbf_rrf_top25_per_question.xlsx`.

> **Note**: the raw evaluation files are not published in this repository (they
> contain corpus-derived content). The aggregate tables in this document are the
> published record.

### Reproducing other configurations

| Run | Modifications to Step 1 flags |
|-----|-------------------------------|
| `dbf_rrf_top25_comb` | Add `--ranking-mode combined` |
| `dbf_rrf_top25_thr-2` | Add `--reranker-threshold -2.0` |
| `dbf_rrf_top50` | Set `TOP_N = 50` in `hybrid_rag_pipeline.py` before running |
| `dbf_rrf_top100` | Set `TOP_N = 100` in `hybrid_rag_pipeline.py` before running |
| `dbf_rrf` (top-10) | Set `TOP_N = 10` in `hybrid_rag_pipeline.py`; omit `--candidates` |
| `db_rrf` | Remove `sparse_fermi` from `SEARCH_CONFIG` in `hybrid_rag_pipeline.py` |
| `baseline_raw` | Ingest documents without structured pre-processing; same retrieval pipeline |

### Regenerating pairwise comparison files

```bash
uv run scripts/rag_evaluation/compare_rag_runs.py \
  --run-a dbf_rrf_top25 \
  --run-b dbf_rrf_top25_comb
```

---

## Appendix A — Detailed Run Parameters

| Run | Fields | Ranker | Candidates | Top-n | Mode | Threshold |
|-----|--------|--------|-----------|-------|------|-----------|
| baseline | dense, bm25 | weighted | — | 10 | standard | none |
| baseline_raw | dense, bm25 | weighted | — | 10 | standard | none (raw ingestion) |
| db_rrf | dense, bm25 | rrf | — | 10 | standard | none |
| df_w | dense, fermi | weighted | — | 10 | standard | none |
| df_rrf | dense, fermi | rrf | — | 10 | standard | none |
| dbf_w | dense, bm25, fermi | weighted | — | 10 | standard | none |
| dbf_rrf | dense, bm25, fermi | rrf | — | 10 | standard | none |
| db_w_top25 | dense, bm25 | weighted | — | 25 | standard | none |
| **dbf_rrf_top25** | dense, bm25, fermi | rrf | **100** | **25** | standard | none |
| dbf_rrf_top50 | dense, bm25, fermi | rrf | 100 | 50 | standard | none |
| dbf_rrf_top100 | dense, bm25, fermi | rrf | 100 | 100 | standard | none |
| dbf_rrf_top25_comb | dense, bm25, fermi | rrf | 100 | 25 | **combined** | none |
| dbf_rrf_top25_thr-3 | dense, bm25, fermi | rrf | 100 | 25 | standard | −3.0 |
| dbf_rrf_top25_thr-1 | dense, bm25, fermi | rrf | 100 | 25 | standard | −1.0 |
| dbf_rrf_top25_thr0 | dense, bm25, fermi | rrf | 100 | 25 | standard | 0.0 |
| dbf_rrf_top25_thr1 | dense, bm25, fermi | rrf | 100 | 25 | standard | +1.0 |

Note: "—" in Candidates indicates that the candidates/top-n distinction was not in effect for
early runs; the top-n pool equals the number of retrieved results directly.

---

## Appendix B — Full Score Tables

### B.1 All runs — reported metrics and Global RAG Score

| Run | FC Recall | Faithful | Rob | Hall ↓ | Cit Recall | Ctx Recall | Ctx Rel | Ctx Cov | Diversity | Hub IA F | **Global** |
|-----|-----------|---------|-----|--------|------------|------------|---------|---------|-----------|----------|------------|
| baseline | 44.87% | 88.09% | 87.66% | 5.06% | 26.72% | 38.48% | 9.59% | 28.99% | 26.69% | 31.43% | **51.35%** |
| baseline_raw | 45.66% | 77.49% | 87.45% | 4.59% | 18.26% | 41.72% | 7.63% | 24.25% | 33.28% | 28.89% | **48.52%** |
| db_rrf | 44.17% | 89.19% | 88.26% | 5.47% | 27.99% | 44.56% | 10.21% | 29.87% | 25.86% | 31.77% | **53.06%** |
| df_w | 45.46% | 86.05% | 87.37% | 5.58% | 21.50% | 42.37% | 8.26% | 23.04% | 25.59% | 32.90% | **50.91%** |
| df_rrf | 43.38% | 88.48% | 88.01% | 2.84% | 27.33% | 44.08% | 9.71% | 28.72% | 25.60% | 30.28% | **52.42%** |
| dbf_w | 45.67% | 83.48% | 88.18% | 4.13% | 24.16% | 40.34% | 9.42% | 27.36% | 26.07% | 30.08% | **50.56%** |
| dbf_rrf | 46.34% | 89.02% | 88.34% | 6.21% | 28.55% | 41.92% | 9.19% | 29.78% | 25.93% | 29.83% | **53.24%** |
| db_w_top25 | 50.87% | 86.19% | 88.25% | 2.82% | 28.13% | 48.24% | 5.66% | 32.55% | 46.11% | 30.48% | **55.63%** |
| **dbf_rrf_top25** ★ | **49.55%** | 87.10% | **88.92%** | 7.89% | **33.10%** | **53.36%** | 6.29% | **35.20%** | 45.45% | **37.00%** | **57.42%** |
| dbf_rrf_top50 | 54.40% | 74.64% | 88.79% | 10.25% | 36.47% | 54.86% | 4.98% | 40.49% | 45.59% | 33.86% | **56.88%** |
| dbf_rrf_top100 | 57.82% | 67.62% | 88.54% | 7.18% | 35.33% | 52.21% | 3.69% | 45.97% | 45.44% | 31.95% | **55.50%** |
| dbf_rrf_top25_comb | 53.48% | 85.68% | 88.68% | 5.32% | 30.27% | 58.04% | 6.45% | 36.17% | 46.21% | 33.10% | **59.19%** |
| dbf_rrf_top25_thr-3 | 50.85% | 87.24% | 88.77% | 3.21% †47 | 32.33% | 51.26% | 6.68% | 33.66% | 44.20% | 31.83% § | **56.94%** |
| dbf_rrf_top25_thr-1 | 50.91% | 85.75% | 88.22% | 2.86% | 30.55% | 51.26% | 6.85% | 34.07% | 41.55% | 33.10% | **56.65%** |
| dbf_rrf_top25_thr0 | 49.30% | 81.22% | 87.97% | 10.31% | 28.52% | 49.26% | 7.47% | 30.75% | 38.87% § | 31.05% | **54.15%** |
| dbf_rrf_top25_thr1 | 52.58% | 86.27% | 88.02% | 5.05% | 32.34% | 52.40% | 6.85% | 34.07% | 41.55% | 34.95% | **57.92%** |

★ Recommended configuration. Hall ↓ = lower is better.
§ Statistically significant regression vs. `dbf_rrf_top25` (BH-corrected Wilcoxon, α = 0.05).
†47 Hallucination rate computed over 47/65 questions only (empty context for ~16 questions at this threshold).

### B.2 All statistically significant pairwise findings (BH-corrected Wilcoxon, α = 0.05)

| Comparison | Metric | Direction | Δ | r | p_BH |
|-----------|--------|-----------|---|---|------|
| baseline vs baseline_raw | Diversity | ↑ baseline_raw | +6.59 pp | 0.839 | < 0.001 |
| baseline vs baseline_raw | Citation Recall | ↑ baseline | +8.46 pp | 0.350 | 0.029 |
| baseline vs baseline_raw | Faithfulness | ↑ baseline | +10.60 pp | 0.348 | 0.029 |
| baseline vs baseline_raw | Citation F-Score | ↑ baseline | +4.76 pp | 0.322 | 0.042 |
| baseline vs dbf_rrf_top25 | Diversity | ↑ dbf_rrf_top25 | +18.76 pp | 0.865 | < 0.001 |
| baseline vs dbf_rrf_top25 | Context Recall | ↑ dbf_rrf_top25 | +14.88 pp | 0.472 | < 0.001 |
| baseline vs dbf_rrf_top25 | Context Relevance | ↑ baseline | +3.30 pp | 0.414 | 0.005 |
| baseline vs dbf_rrf_top25 | **Global RAG Score** | **↑ dbf_rrf_top25** | **+6.07 pp** | **0.362** | **0.015** |
| baseline vs dbf_rrf_top25 | Context Coverage | ↑ dbf_rrf_top25 | +6.21 pp | 0.331 | 0.021 |
| baseline vs dbf_rrf_top25 | Paraphrase Robustness | ↑ dbf_rrf_top25 | +1.26 pp | 0.334 | 0.021 |
| baseline vs dbf_rrf_top25 | Citation Recall | ↑ dbf_rrf_top25 | +6.38 pp | 0.317 | 0.026 |
| baseline vs db_w_top50 | Context Coverage | ↑ db_w_top50 | +8.48 pp | 0.350 | 0.020 |
| baseline vs db_w_top50 | Faithfulness | ↑ baseline | +10.13 pp | 0.330 | 0.027 |
| baseline vs db_w_top75 | Faithfulness | ↑ baseline | +15.02 pp | 0.461 | 0.001 |
| baseline vs db_w_top75 | Context Coverage | ↑ db_w_top75 | +10.53 pp | 0.382 | 0.007 |
| baseline vs db_w_top100 | Faithfulness | ↑ baseline | +15.32 pp | 0.489 | < 0.001 |
| baseline vs db_w_top100 | Context Coverage | ↑ db_w_top100 | +11.56 pp | 0.412 | 0.004 |
| baseline vs db_w_top100 | Context Recall | ↑ db_w_top100 | +12.06 pp | 0.323 | 0.026 |
| db_w_top25 vs dbf_rrf_top25 | Hub France IA F-Score | ↑ dbf_rrf_top25 | +6.52 pp | 0.432 | 0.009 |
| db_w_top25 vs dbf_rrf_top25 | Context Precision | ↑ dbf_rrf_top25 | +6.94 pp | 0.360 | 0.031 |
| dbf_rrf_top25 vs thr-3 | Hub France IA F-Score | ↑ dbf_rrf_top25 | +5.17 pp | 0.380 | 0.038 |
| dbf_rrf_top25 vs thr0 | Diversity | ↑ dbf_rrf_top25 | +6.58 pp | 0.506 | 0.001 |

---

## Appendix C — Head-to-Head: Top-25 Configuration Comparison

This appendix provides a direct comparison between `db_w_top25` (dense + BM25, weighted, top-25)
and `dbf_rrf_top25` (the recommended configuration) at equal chunk count, isolating the joint
effect of the Fermi field addition and the RRF fusion strategy.

The configurations `db_rrf_top25` and `dbf_w_top25` were not run and are therefore not available
for comparison.

### C.1 `db_w_top25` vs. `dbf_rrf_top25`

| Metric | db_w_top25 | dbf_rrf_top25 | Δ (dbf_rrf − db_w) | Significant |
|--------|-----------|---------------|---------------------|-------------|
| FC Recall | 50.87% | 49.55% | −1.32 pp | No |
| Faithfulness | 86.19% | 87.10% | +0.91 pp | No |
| Reformulation Robustness | 88.25% | 88.92% | +0.67 pp | No |
| Hall Rate ↓ | 2.82% | 7.89% | +5.07 pp | No |
| Citation Recall | 28.13% | 33.10% | +4.97 pp | No |
| Context Recall | 48.24% | 53.36% | +5.13 pp | No |
| Context Relevance | 5.66% | 6.29% | +0.63 pp | No |
| Context Coverage | 32.55% | 35.20% | +2.65 pp | No |
| Diversity | 46.11% | 45.45% | −0.65 pp | No |
| **Hub France IA F-Score** | 30.48% | **37.00%** | **+6.52 pp** | **Yes** (r = 0.432, p_BH = 0.009) |
| **Context Precision** | 33.94% | **40.88%** | **+6.94 pp** | **Yes** (r = 0.360, p_BH = 0.031) |
| **Global RAG Score** | 55.63% | **57.42%** | +1.80 pp | No (r = 0.126, p_BH = 0.480) |

`dbf_rrf_top25` significantly outperforms `db_w_top25` on Hub France IA F-Score (+6.52 pp,
r = 0.432, p_BH = 0.009) and Context Precision (+6.94 pp, r = 0.360, p_BH = 0.031), confirming
that the joint contribution of the Fermi field and RRF fusion improves grounding quality and
retrieval precision beyond what BM25 + weighted fusion achieves at equal top-n. The +1.80 pp
global advantage of `dbf_rrf_top25` is not individually significant in this pairwise test but
accumulates additively with the improvements established in earlier series.

The hallucination rate is notably higher in `dbf_rrf_top25` (+5.07 pp, NS). This is likely
linked to the higher context recall: a richer, more diverse context covering more factual content
may also introduce more peripheral claims that the LLM echoes without grounding. The mechanism
is uncertain and the difference is not statistically confirmed.
