# GraphRAG Experimental Analysis

> **Document scope:** evaluation of five GraphRAG configurations on two Neo4j knowledge
> graphs built from the same regulatory corpus as the HybridRAG study
> (`run_analysis.md`). All runs are evaluated on the same 65 question-answer pairs
> (260 queries including paraphrase variants), with the same generation LLM, judge LLM,
> prompt template, and metrics — results are directly comparable across the two studies,
> within the limits discussed in §3.2.

---

## 1. Experimental Protocol

### 1.1 Objective

The HybridRAG study established that a flat vector+lexical retrieval pipeline
(`dbf_rrf_top50`, Global RAG Score 65.55%) is a strong configuration on this corpus. The
question addressed here is whether a **knowledge graph** adds retrieval value: can
graph-structural signals — passages connected through shared entities, LLM-generated graph
queries — surface evidence that flat retrieval misses, and at what cost in robustness?

Five experiments answer this in a controlled way: one **internal baseline** that uses the
graph as a plain document store (no structural signal), and two graph-exploiting modes
(**hybrid_cypher**, **text2cypher**) each run on two graphs of very different ontology
quality.

### 1.2 The Two Knowledge Graphs

| Graph | Prefix | Ontology | Schema size |
|-------|:---:|----------|:---:|
| Static | `st` | Manually curated, clean and consistent entity taxonomy | ~260 schema lines |
| Dynamic | `dy` | Automatically extracted, open-ended entity taxonomy | 11,245 schema lines (5,718 node types, 5,527 relation types) |

Both graphs contain the same document corpus as Chunk nodes (text, source, BGE-M3
embedding), linked to entity nodes via typed relations. They differ in how their entity
layer was built — and therefore in how exploitable it is by schema-dependent methods.

**Construction caveats, to keep in mind throughout this document:**

- The graphs are built from **about a hundred source documents** ingested with **basic
  chunking** — fixed-size text splitting, not the structure-aware pre-processing
  (Docling-style layout analysis) used for the Milvus corpus of the HybridRAG study. The
  HybridRAG Series 1 result showed that ingestion quality significantly drives faithfulness
  and citation quality; the graph corpus starts with that handicap built in.
- The **static ontology was not reviewed by a domain expert**, due to the time constraints
  of the project. It is "curated" relative to the auto-extracted dynamic one (consistent,
  compact), but its adequacy to the nuclear-regulatory domain — the right entity types, the
  right granularity, the right relations — has not been validated. An expert review is a
  prerequisite for the entity layer to reliably reflect what matters in the corpus, and
  therefore for graph-structural retrieval to reach its actual potential; the results below
  should be read as a lower bound obtained on an unvalidated ontology.

### 1.3 The Three Retrieval Modes

**`flat_rrf` — the internal baseline.** BGE-M3 ANN + BM25 Lucene over the Chunk nodes →
Reciprocal Rank Fusion (k = 60) over a 100-candidate pool → cross-encoder reranking →
top-50 chunks. This is architecturally the same recipe as the HybridRAG champion (dense +
lexical fusion, reranker, 50 chunks), minus the Fermi field, running on the Neo4j corpus.
It deliberately ignores the graph structure.

**Why it is the baseline for everything:** it isolates the variable under study. Any
difference between `flat_rrf` and a graph-exploiting mode is attributable to the
*structural exploitation itself* — same corpus, same chunks, same embeddings, same
reranker, same LLM. A graph mode that cannot beat `flat_rrf` provides no evidence that the
graph adds retrieval value; the graph is then just an expensive document store.

**`hybrid_cypher` — structural retrieval (HippoRAG-inspired).** BGE-M3 ANN retrieves 10
seed chunks; a 2-hop traversal walks from those seeds to their entities, to neighbouring
entities, and back to the chunks those entities appear in ("bridged" passages); seeds and
bridged chunks (up to 100) are reranked by the cross-encoder; top-50 form the context.
The hypothesis: passages thematically related through entity paths can be relevant without
being vectorially close to the query. The seed count follows the state of the art
(HippoRAG links ~5 nodes per query entity; Microsoft GraphRAG seeds local search with
8–12 items).

**`text2cypher` — LLM-generated queries.** The LLM receives the graph schema and the
question, generates a single Cypher query constrained to the pattern
`(entity)-[:FROM_CHUNK]->(chunk)`, and the raw query results (up to 50 records) form the
context. On the static graph the full schema is passed; on the dynamic graph its 11,245
lines exceed any context window, so a per-query embedding-based pruning (BGE-M3 cosine,
top-100 lines) selects the schema excerpt.

### 1.4 Experiments

| Experiment | Graph | Mode |
|------------|:---:|------|
| `graph_baseline` | static | flat_rrf |
| `graph_st_hybrid_cypher` | static | hybrid_cypher |
| `graph_dy_hybrid_cypher` | dynamic | hybrid_cypher |
| `graph_st_text2cypher` | static | text2cypher |
| `graph_dy_text2cypher` | dynamic | text2cypher |

Note: `graph_baseline` runs on the static graph, but since `flat_rrf` ignores the entity
layer entirely and both graphs hold the same Chunk corpus, it is an equally valid reference
for the dynamic-graph experiments.

### 1.5 Metrics and Statistics

Identical to the HybridRAG study: eleven reported metrics, Global RAG Score as primary
criterion (`0.25 × Context Recall + 0.35 × FC Recall + 0.15 × Citation Recall + 0.25 ×
Faithfulness`), pairwise two-tailed Wilcoxon signed-rank tests on the 65 matched questions
with Benjamini-Hochberg correction at α = 0.05. In the body of this document, differences
are reported simply as **significant or not** under that procedure; effect sizes and
corrected p-values are listed in Appendix C.

---

## 2. Results Overview

| Run | FC Recall | Faithfulness | Hall ↓ | Cit Recall | Ctx Recall | Ctx Cov | Hub IA F | **Global** |
|-----|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **graph_baseline** | 42.33% | 80.67% | 10.34% | 13.46% | 62.04% | 18.74% | 15.40% | **52.51%** |
| graph_st_hybrid_cypher | 37.48% | 78.14% | 11.26% | 10.31% | 55.11% | 19.69% | 19.65% | **47.97%** |
| graph_dy_hybrid_cypher | 34.43% | 79.73% | 10.25% | 12.42% | 43.57% | 13.46% | 17.56% | **44.83%** |
| graph_st_text2cypher | 17.79% | 55.73% | 36.25% | 1.92% | 19.49% | 5.00% | 1.17% | **25.32%** |
| graph_dy_text2cypher | 14.76% | 47.30% | 50.61% | 2.05% | 16.41% | 2.05% | 0.32% | **21.35%** |

Global RAG Score of each mode vs. `graph_baseline`:

| Comparison | Δ Global | Significant |
|-----------|:---:|:---:|
| st_hybrid_cypher | −4.54 pp | No |
| dy_hybrid_cypher | −7.68 pp | **Yes — baseline wins** |
| st_text2cypher | −27.19 pp | **Yes — baseline wins** |
| dy_text2cypher | −31.17 pp | **Yes — baseline wins** |

**The headline result is unambiguous: on this corpus, no graph-exploiting mode beats — or
even matches — flat retrieval over the same graph.** The ordering is strict:
flat retrieval > 2-hop traversal > LLM-generated Cypher, on both graphs.

---

## 3. The Baseline

### 3.1 What `graph_baseline` establishes

At 52.51% Global, the graph corpus served by flat retrieval is a workable RAG substrate:
context recall reaches 62.04% and faithfulness 80.67%. This is the bar the structural modes
must clear, and it is not an artificially low one — it embeds every good practice validated
in the HybridRAG study (rank fusion, cross-encoder reranking, 50-chunk context).

### 3.2 Comparison with the HybridRAG champion — and its limits

`graph_baseline` (52.51%) sits 13.04 pp below `dbf_rrf_top50` (65.55%). It is tempting to
read this as "Neo4j loses to Milvus", but the gap decomposes into differences that have
nothing to do with the database or the graph:

- **Citation recall: 13.46% vs. 36.47%.** The graph corpus's source metadata is poorer —
  file names are recovered from a path property on Chunk nodes rather than curated
  metadata. Citation recall weighs 15% of the global score, so this alone costs roughly
  3.5 pp mechanically, before any retrieval quality difference.
- **Ingestion quality.** The graph corpus was ingested with basic fixed-size chunking
  (§1.2), where the Milvus corpus benefited from structure-aware pre-processing — an axis
  the HybridRAG study measured as significantly impacting faithfulness and citation
  quality. The graph baseline's lower faithfulness (80.67% vs. 88.83%) is consistent with
  exactly that effect.
- **A missing field.** The Neo4j side fuses two signals (dense + BM25 Lucene) where the
  Milvus champion fuses three (dense + BM25 + Fermi).

The cross-backend comparison is therefore indicative, not a controlled experiment: it
measures the combined effect of ingestion, metadata and indexing choices, not "graph vs.
vector store". The controlled comparisons are the intra-graph ones — same corpus, same
ingestion, only the retrieval mode varies — and they are the object of this study.

---

## 4. Axis 1 — Does Graph Structure Add Retrieval Value?

### 4.1 hybrid_cypher vs. baseline

| Metric | baseline | st_hybrid_cypher | dy_hybrid_cypher | Significant |
|--------|:---:|:---:|:---:|:---:|
| Context Recall | 62.04% | 55.11% | 43.57% | dy only — baseline wins |
| FC Recall | 42.33% | 37.48% | 34.43% | No |
| Faithfulness | 80.67% | 78.14% | 79.73% | No |
| Context Coverage | 18.74% | 19.69% | 13.46% | No |
| Hub IA F-Score | 15.40% | 19.65% | 17.56% | No |
| Diversity | 30.10% | 31.40% | 32.50% | dy only — traversal wins |
| **Global** | **52.51%** | **47.97%** | **44.83%** | **dy only — baseline wins** |

On the static graph, the 2-hop traversal loses 4.54 pp of Global without any individual
metric reaching significance. On the dynamic graph the loss doubles (−7.68 pp) and becomes
significant, driven by a significant context recall collapse (−18.47 pp).

**The mechanism is a funnel problem, and the metrics let us watch it operate.** `flat_rrf`
draws its 100 candidates from the *entire* index through two complementary signals — every
chunk in the corpus is reachable at every question. `hybrid_cypher` draws its candidates
exclusively from the 2-hop entity neighbourhood of 10 ANN seeds: any relevant passage that
happens not to share an entity path with those seeds is structurally unreachable, however
strong its keyword or semantic match. Flat BM25 would have caught it; the traversal cannot.
The cost lands exactly where the funnel predicts — on recall (62.04% → 55.11% → 43.57%) —
while precision-side metrics move little.

**What the traversal finds, it finds well — it just does not find enough.** Two details
show the structural signal is real rather than pure loss. Context *coverage* on the static
graph matches the baseline (19.69% vs. 18.74%): the traversal reaches roughly as many of
the expected *documents*, through entity paths rather than similarity. And Hub IA F-Score —
the balance of context precision and recall — is directionally *better* than the baseline
on both graphs: the bridged passages that do arrive are pertinent. The traversal is a
precise but narrow instrument; the baseline is a wide net. On a corpus where the HybridRAG
study proved that recall is the dominant lever, narrow loses to wide.

**Why the dynamic graph makes it significantly worse.** The traversal is schema-agnostic
but not *structure*-agnostic: its candidate set is whatever the entity layer connects. On
the dynamic graph, 5,718 auto-extracted node types create connectivity that is dense but
weakly informative — near-duplicate entities fragment the neighbourhood, spurious entities
bridge unrelated passages. The 2-hop ball around the seeds fills its 100-chunk budget with
loosely related material (visible in the significantly *higher* diversity: 32.50%, the
highest of all five runs — heterogeneity as noise, the same signature as `baseline_raw` in
the HybridRAG study), crowding out the genuinely related passages. Structure quality is not
a detail: it is the resource this mode spends.

### 4.2 text2cypher vs. baseline

Both text2cypher runs collapse: −27.19 pp (static) and −31.17 pp (dynamic), the two largest
and most significant regressions of the entire study. Faithfulness falls to 55.73% / 47.30%
and hallucination rises to 36.25% / 50.61%, versus 10.34% for the baseline — all
significant. These are not degraded versions of the baseline's behaviour but a different
failure regime altogether; §6 dissects it.

### 4.3 Verdict

As implemented, graph-structural retrieval does not pay for itself on this corpus. The
traversal mode is a constrained version of flat retrieval — same reranker, narrower
candidate source, so it can only lose on recall-dominated ground. The query-generation mode
replaces a robust two-signal retrieval stack with a single brittle LLM decision. Neither
exploits what a graph is actually good at — *global* connectivity — which is what the
agentic and PPR perspectives in §7 address.

---

## 5. Axis 2 — Static vs. Dynamic Ontology

Same mode, different graph:

| Mode | Static | Dynamic | Δ (dy − st) | Significant |
|------|:---:|:---:|:---:|:---:|
| hybrid_cypher (Global) | 47.97% | 44.83% | −3.15 pp | No |
| text2cypher (Global) | 25.32% | 21.35% | −3.97 pp | No |
| text2cypher (Grounded rate) | 36.95% | 16.85% | −20.10 pp | **Yes — static wins** |

The static ontology wins directionally in both modes, and the sole significant difference —
the grounded rate in text2cypher mode — is the most diagnostic metric of the pair: on the
dynamic graph, barely one response claim in six is anchored in retrieved evidence.

**The two modes lose to the dynamic ontology through two different mechanisms**, which is
what makes the pattern credible rather than coincidental:

- **hybrid_cypher suffers through connectivity noise.** It never reads the schema; what
  hurts it is the *shape* of the entity layer. An uncurated taxonomy fragments and blurs
  the neighbourhood structure the traversal walks through (§4.1): context recall drops from
  55.11% to 43.57% because the 2-hop ball wastes its budget on weakly related passages.
- **text2cypher suffers through schema exposure.** It depends entirely on reading the
  schema, and the dynamic one is both too large to show (11,245 lines → lossy per-query
  pruning) and too inconsistent to guess (near-duplicate types, singletons). The
  consequence is measured in §6: the empty-context rate doubles from 40% to 80% of
  questions between the static and dynamic graphs.

**Takeaway:** ontology quality is not a nice-to-have — it is the substrate both
graph-exploiting modes consume, each in its own way. A curated taxonomy is a hard
precondition for schema-dependent methods to function at all, and even schema-agnostic
traversal degrades measurably on an uncurated entity layer. If the dynamic graph is the
production target, its entity layer needs consolidation (type normalisation, entity
resolution, pruning of singleton types) before graph-structural retrieval is worth
re-testing on it — and the static ontology itself still awaits the domain-expert review
noted in §1.2.

---

## 6. Axis 3 — hybrid_cypher vs. text2cypher, and Why text2cypher Fails

### 6.1 Head-to-head

At equal graph, the traversal mode dominates the query-generation mode on every
answer-quality metric:

| Graph | text2cypher | hybrid_cypher | Δ Global | Significant |
|-------|:---:|:---:|:---:|:---:|
| Static | 25.32% | 47.97% | +22.65 pp | **Yes — hybrid wins** |
| Dynamic | 21.35% | 44.83% | +23.48 pp | **Yes — hybrid wins** |

Faithfulness, context recall, FC recall and hallucination are all significantly in favour
of hybrid_cypher, on both graphs (details in Appendix C).

### 6.2 Anatomy of the text2cypher failure: the empty-context problem

The evaluation data localises the failure precisely — it is a **retrieval availability**
problem before being a quality problem. The central numbers:

| | Static graph | Dynamic graph |
|---|:---:|:---:|
| Questions with ZERO retrieved chunks | 26 / 65 (40%) | 52 / 65 (80%) |
| Questions with fewer than two chunks | 31 / 65 (48%) | 56 / 65 (86%) |
| Hallucination rate | 36.25% | 50.61% |
| Grounded rate (claims supported by retrieved evidence) | 36.95% | 16.85% |
| Citation recall | 1.92% | 2.05% |

The causal chain is direct: when the generated Cypher query returns nothing, the LLM
receives an empty context — and answers anyway, from parametric memory. Every empty context
converts into an ungrounded answer, which is exactly what the hallucination and grounded
rates measure. On the dynamic graph, one response claim out of two has no support in any
retrieved document.

**Why empty contexts happen in the general case.** The mode is single-shot with zero
fallback: one LLM call produces one Cypher query, executed once. Any of the usual
generation slips — an over-restrictive WHERE clause, a slightly wrong label, an entity name
that does not match the graph's surface form — yields zero rows, and nothing recovers. The
entity-matching problem is structural on this corpus: questions are in French while entity
names and labels are in English or mixed, so the generated
`WHERE entity.name CONTAINS "..."` filters miss translations, paraphrases and surface
variants (no entity linking or fuzzy matching exists between question terms and node
names). A single point of failure thus sits exactly where the flat pipeline has two
redundant retrieval signals plus a reranker.

**Why the dynamic ontology makes it far worse — logically, not accidentally.** On the
static graph the LLM sees the *full* schema (~260 lines): the labels it uses exist, and
failures come almost entirely from the entity-name filters. On the dynamic graph, the
11,245-line schema cannot be shown; the model sees a per-query pruned excerpt (top-100
lines by embedding similarity), which frequently omits the very labels needed — so the
model approximates or invents them, and the query matches nothing. On top of that, the
auto-extracted taxonomy is inconsistent (5,718 node types, near-duplicates, singletons):
even a label that exists may not be the one the relevant entities actually carry. Schema
exposure and schema quality degrade together, which is why the empty-context rate doubles
(40% → 80%) between the curated and the uncurated graph.

**A fourth, quality-side cause** compounds the availability problem: even when a query
succeeds, it returns up to 50 raw records in arbitrary order — no reranker, no relevance
scoring — so a "successful" text2cypher context is still weaker than a reranked pool.

None of these causes are inherent to the text2cypher *idea*; all are properties of the
current single-pass implementation. But fixing them amounts to rebuilding the mode (§7).

---

## 7. Perspectives — Where GraphRAG Value Could Come From

**An agentic loop is the natural fix for text2cypher.** The failure anatomy in §6.2 is a
list of things an agent recovers from and a single-pass pipeline cannot: an agent generates
a query, *inspects the result*, and reacts — relaxes the WHERE clause when zero rows come
back, re-reads the schema when a label errors out, reformulates entity names, and falls
back to vector search after N failed attempts. The 26/65 and 52/65 empty-context rates are
precisely the cases such a loop would catch. An agentic GraphRAG (ReAct-style: reason →
query → observe → refine, with text2cypher, vector search, and graph traversal as tools)
converts the single point of failure into a recoverable step, at the cost of latency and
LLM calls. Given that the current mode loses 27–31 pp to the baseline, the margin for
improvement is enormous.

**The traversal mode has a cheaper upgrade path.** Its deficit is the seed funnel, and the
state of the art addresses exactly that: HippoRAG replaces the fixed 2-hop walk with
Personalized PageRank from query-entity seeds (global graph signal instead of a fixed
radius), and entity-based seeding (link *question entities* to graph nodes, rather than
whole-question ANN over chunks) widens the entry point. Both are non-agentic and keep
latency predictable.

**The graphs themselves need work before re-testing.** Corpus-side: the ingestion should be
upgraded from basic fixed-size chunking to the structure-aware pre-processing validated in
the HybridRAG study, and source metadata aligned with the Milvus ingestion (the citation
gap alone costs the graph stack ~3.5 pp of global score mechanically). Static graph: the
ontology must go through the **domain-expert review** it has not yet received — entity
types, granularity and relations validated against actual regulatory practice — before the
entity layer can be trusted to encode what matters in the corpus. Dynamic graph:
entity-type normalisation and entity resolution are prerequisites — 5,718 node types is not
an ontology, it is a vocabulary dump.

**And the reference bar will move.** Flat retrieval on the graph corpus reached 52.51%
without the Fermi field; porting the full HybridRAG champion recipe to the Neo4j corpus
would raise the internal baseline further. Any future graph mode must be measured against
that moving bar, not against a convenient weaker reference.

---

## 8. Conclusion

1. **Flat retrieval wins on the graph corpus.** `graph_baseline` (52.51%) outperforms every
   graph-exploiting mode; the graph structure, as currently exploited, subtracts value —
   significantly so for hybrid_cypher on the dynamic graph (−7.68 pp) and catastrophically
   for text2cypher on both graphs (−27 to −31 pp, both significant).
2. **Static ontology > dynamic ontology**, directionally in both modes, significantly on
   grounding for text2cypher. Ontology curation is a precondition for schema-dependent
   retrieval, and helps even schema-agnostic traversal.
3. **hybrid_cypher > text2cypher by 22–23 pp at equal graph** (significant on both).
   If graph-structural retrieval must ship today, the traversal mode is the only defensible
   option.
4. **text2cypher as implemented is not viable**: 40–80% of questions end with an empty
   context, hallucination up to 50.61%, near-zero citation. Its failure modes are
   implementation properties (single-shot, no fallback, lexical cross-language matching,
   unranked results), not properties of the idea.
5. **These results are a lower bound.** The graphs were built from ~100 documents with
   basic chunking (no structure-aware pre-processing) and an ontology that has not been
   reviewed by a domain expert. Both handicaps are fixable and both plausibly depress every
   graph-side number in this study.
6. **Recommendation.** For production on this corpus today: the HybridRAG champion
   (`dbf_rrf_top50`, 65.55%) remains the system of reference; the graph stack does not
   currently justify its complexity. For the graph track: invest in (a) an agentic
   retrieval loop with text2cypher as one tool among several, (b) PPR-style traversal with
   entity-based seeding, and (c) graph curation — expert review of the static ontology,
   structure-aware ingestion, source metadata, entity resolution — then re-run this
   five-experiment protocol against a Fermi-complete internal baseline.

---

## 9. How to Reproduce

Prerequisites: Neo4j instances on ports 7687 (static) and 7688 (dynamic), populated with
the corpus; `uv` environment; Cleyrop proxy reachable.

```bash
# Internal baseline (static graph)
uv run scripts/rag_evaluation/prepare_eval_dataset_graph.py \
    scripts/rag_evaluation/data/rag_evaluation_dataset.xlsx \
    --experiment-id graph_baseline --mode flat_rrf \
    --neo4j-url bolt://localhost:7687

# Traversal mode — static / dynamic
uv run scripts/rag_evaluation/prepare_eval_dataset_graph.py \
    scripts/rag_evaluation/data/rag_evaluation_dataset.xlsx \
    --experiment-id graph_st_hybrid_cypher --mode hybrid_cypher \
    --neo4j-url bolt://localhost:7687
uv run scripts/rag_evaluation/prepare_eval_dataset_graph.py \
    scripts/rag_evaluation/data/rag_evaluation_dataset.xlsx \
    --experiment-id graph_dy_hybrid_cypher --mode hybrid_cypher

# Query-generation mode — static (full schema) / dynamic (embedding-pruned schema)
uv run scripts/rag_evaluation/prepare_eval_dataset_graph.py \
    scripts/rag_evaluation/data/rag_evaluation_dataset.xlsx \
    --experiment-id graph_st_text2cypher --mode text2cypher \
    --neo4j-url bolt://localhost:7687 --no-prune-schema
uv run scripts/rag_evaluation/prepare_eval_dataset_graph.py \
    scripts/rag_evaluation/data/rag_evaluation_dataset.xlsx \
    --experiment-id graph_dy_text2cypher --mode text2cypher

# Scoring (any experiment)
uv run scripts/rag_evaluation/rag_evaluation.py --experiment-id <id>

# Pairwise statistics
cd scripts/rag_evaluation && uv run compare_rag_runs.py <run_A> <run_B>
```

Pipeline defaults (set in `scripts/rag/graph_rag_pipeline.py`): candidate pool 100,
top-n 50, RRF k = 60, hybrid_cypher seeds 10, traversal limit 100, text2cypher schema
pruning top-100 lines.

---

## Appendix A — Run Parameters

| Run | Graph | Mode | Pool | Top-n | Seeds | Schema |
|-----|:---:|------|:---:|:---:|:---:|--------|
| graph_baseline | static | flat_rrf | 100 | 50 | — | — |
| graph_st_hybrid_cypher | static | hybrid_cypher | ≤100 | 50 | 10 | — |
| graph_dy_hybrid_cypher | dynamic | hybrid_cypher | ≤100 | 50 | 10 | — |
| graph_st_text2cypher | static | text2cypher | ≤50 records | — | — | full (~260 lines) |
| graph_dy_text2cypher | dynamic | text2cypher | ≤50 records | — | — | embedding-pruned (top-100 of 11,245 lines) |

## Appendix B — Full Score Table

| Run | FC Recall | Faithful | Rob | Hall ↓ | Cit Recall | Ctx Recall | Ctx Rel | Ctx Cov | Diversity | Hub IA F | **Global** |
|-----|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **graph_baseline** | 42.33% | 80.67% | 87.80% | 10.34% | 13.46% | 62.04% | 3.08% | 18.74% | 30.10% | 15.40% | **52.51%** |
| graph_st_hybrid_cypher | 37.48% | 78.14% | 86.69% | 11.26% | 10.31% | 55.11% | 2.52% | 19.69% | 31.40% | 19.65% | **47.97%** |
| graph_dy_hybrid_cypher | 34.43% | 79.73% | 86.65% | 10.25% | 12.42% | 43.57% | 3.76% | 13.46% | 32.50% | 17.56% | **44.83%** |
| graph_st_text2cypher | 17.79% | 55.73% | 86.68% | 36.25% | 1.92% | 19.49% | 1.92% | 5.00% | 31.88% | 1.17% | **25.32%** |
| graph_dy_text2cypher | 14.76% | 47.30% | 86.99% | 50.61% | 2.05% | 16.41% | 11.54% | 2.05% | 32.82% | 0.32% | **21.35%** |

Sample-size caveats: for the text2cypher runs, the file-level relevance and diversity
metrics could only be computed on the questions with non-empty retrieval (static: 39 and 34
of 65 respectively; dynamic: 13 and 9 of 65). In particular, the dynamic run's apparently
high Context Relevance (11.54%) rests on 13 questions and should not be interpreted.

## Appendix C — All Statistically Significant Pairwise Findings

Two-tailed Wilcoxon signed-rank tests on 65 matched questions, Benjamini-Hochberg corrected
at α = 0.05. Effect size r is the rank-biserial correlation (|r| < 0.2 negligible, 0.2–0.5
moderate, > 0.5 large).

| Comparison | Metric | Direction | Δ | r | p_BH |
|-----------|--------|-----------|:---:|:---:|:---:|
| baseline vs dy_hybrid_cypher | Global RAG Score | baseline | +7.68 pp | 0.394 | 0.010 |
| baseline vs dy_hybrid_cypher | Context Recall | baseline | +18.47 pp | 0.410 | 0.010 |
| baseline vs dy_hybrid_cypher | Robustness | baseline | +1.15 pp | 0.321 | 0.041 |
| baseline vs dy_hybrid_cypher | Diversity | dy_hybrid | +2.41 pp | 0.387 | 0.010 |
| baseline vs st_text2cypher | Global RAG Score | baseline | +27.19 pp | 0.686 | <0.0001 |
| baseline vs st_text2cypher | Context Recall | baseline | +42.55 pp | 0.511 | 0.0001 |
| baseline vs st_text2cypher | Faithfulness | baseline | +24.94 pp | 0.454 | 0.0005 |
| baseline vs st_text2cypher | Hallucination | baseline | −25.91 pp | 0.475 | 0.0003 |
| baseline vs st_text2cypher | FC Recall | baseline | +24.54 pp | 0.504 | 0.0001 |
| baseline vs st_text2cypher | Citation Recall | baseline | +11.54 pp | 0.382 | 0.0027 |
| baseline vs dy_text2cypher | Global RAG Score | baseline | +31.17 pp | 0.746 | <0.0001 |
| baseline vs dy_text2cypher | Context Recall | baseline | +45.63 pp | 0.590 | <0.0001 |
| baseline vs dy_text2cypher | Faithfulness | baseline | +33.37 pp | 0.553 | <0.0001 |
| baseline vs dy_text2cypher | Hallucination | baseline | −40.27 pp | 0.625 | <0.0001 |
| baseline vs dy_text2cypher | FC Recall | baseline | +27.57 pp | 0.553 | <0.0001 |
| baseline vs dy_text2cypher | Citation Recall | baseline | +11.41 pp | 0.397 | 0.0021 |
| st_text2cypher vs dy_text2cypher | Grounded Rate | static | +20.10 pp | 0.370 | 0.049 |
| st_text2cypher vs st_hybrid_cypher | Global RAG Score | hybrid | +22.65 pp | 0.642 | <0.0001 |
| st_text2cypher vs st_hybrid_cypher | Context Recall | hybrid | +35.62 pp | 0.470 | 0.0004 |
| st_text2cypher vs st_hybrid_cypher | Faithfulness | hybrid | +22.40 pp | 0.459 | 0.0005 |
| st_text2cypher vs st_hybrid_cypher | Hallucination | hybrid | −24.98 pp | 0.472 | 0.0004 |
| st_text2cypher vs st_hybrid_cypher | FC Recall | hybrid | +19.69 pp | 0.500 | 0.0002 |
| dy_text2cypher vs dy_hybrid_cypher | Global RAG Score | hybrid | +23.48 pp | 0.629 | <0.0001 |
| dy_text2cypher vs dy_hybrid_cypher | Context Recall | hybrid | +27.16 pp | 0.387 | 0.0037 |
| dy_text2cypher vs dy_hybrid_cypher | Faithfulness | hybrid | +32.43 pp | 0.493 | 0.0002 |
| dy_text2cypher vs dy_hybrid_cypher | Hallucination | hybrid | −40.36 pp | 0.635 | <0.0001 |
| dy_text2cypher vs dy_hybrid_cypher | FC Recall | hybrid | +19.67 pp | 0.480 | 0.0004 |
