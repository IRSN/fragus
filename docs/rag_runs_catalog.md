# Catalogue des runs d'évaluation RAG — IAGO / ASNR

Ce catalogue liste, pour chaque run retenu, sa configuration exacte et sa commande de
reproduction. Les 20 runs ci-dessous sont ceux effectivement commentés dans les deux
documents d'analyse — c'est là que se trouve le raisonnement complet (mécanismes,
significativité statistique, verdicts) :

- **`run_analysis.md`** — 15 runs HybridRAG (Milvus), protocole en 5 séries.
- **`run_analysis_graph.md`** — 5 runs GraphRAG (Neo4j), 2 graphes × 3 modes de retrieval.

Ce fichier est une référence technique (paramètres, commandes, tableaux de scores) ; pour
l'interprétation — pourquoi tel run gagne, quels effets sont statistiquement établis — se
reporter aux deux documents source, section par section (indiqué à chaque run).

Les fichiers de résultats sont dans `scripts/rag_evaluation/data/results/` :
- `eval_test_<run_id>.xlsx` / `eval_test_<run_id>_contexts.parquet` — réponses brutes du
  pipeline (question, réponse, contextes complets, scores reranker)
- `score_test_<run_id>_per_question.xlsx` — scores LLM par question
- `score_test_<run_id>_averages.xlsx` — moyennes par métrique

---

## Convention de nommage des experiment-id

**HybridRAG** — pattern : `{champs}_{ranker}[_top{n}][_comb][_thr{v}]`

| Segment | Valeur | Signification |
|---------|--------|---------------|
| champs | `db` | Dense + BM25 |
| | `df` | Dense + Fermi |
| | `dbf` | Dense + BM25 + Fermi |
| ranker | `w` | WeightedRanker |
| | `rrf` | RRFRanker |
| top-n | `_top50` | top-n=50 (absent = défaut 10) |
| ranking_mode | `_comb` | combined (absent = reranker seul) |
| seuil reranker | `_thr-1` | threshold=-1.0 (absent = pas de filtre) |

`baseline` (Dense+BM25, Weighted, top-10) est le run de référence de toute l'étude
HybridRAG et conserve son nom tel quel.

**GraphRAG** — pattern : `graph_{graphe}_{mode}` (`graph_baseline` pour le mode `flat_rrf`
sur le graphe statique)

| Segment | Valeur | Signification |
|---------|--------|---------------|
| graphe | `st` | ontologie statique, curatée manuellement |
| | `dy` | ontologie dynamique, extraite automatiquement par LLM |
| mode | `flat_rrf` | retrieval plat (dense+BM25 Lucene), ignore la structure du graphe |
| | `hybrid_cypher` | traversée 2-hop entités (style HippoRAG) |
| | `text2cypher` | requête Cypher générée par LLM |

---

## Rappel architecture

### Pipeline HybridRAG (Milvus)

```
Milvus hybrid_search (multi-vecteur)
    ↓  fusion : WeightedRanker ou RRFRanker (k=60)
Candidats (top-100)
    ↓  cross-encoder reranker (mmarco-mMiniLMv2-L12-H384-v1, CPU)
[seuil optionnel sur le score reranker]
Top-N chunks
    ↓  LLM (mistral-large-latest via proxy Cleyrop)
Réponse
```

**Collection** : `baseline_fragus` (corpus Docling+OCR, HybridChunker cible 512
tokens/chunk) sauf `baseline_raw` → `brut_fragus` (corpus pypdf, fenêtres de 500 tokens
BGE-M3, recouvrement 100 tokens, sans OCR, min 20 caractères).
3 champs vectoriels disponibles : `embedding` (BGE-M3 dense, COSINE), `sparse_bm25` (BM25
client-side, IP), `sparse_fermi` (Fermi-1024 sparse, IP).

### Pipeline GraphRAG (Neo4j)

```
flat_rrf         : BGE-M3 ANN + BM25 Lucene → RRF (k=60) → reranker → top-50
hybrid_cypher     : BGE-M3 ANN (10 seeds) → traversée 2-hop entités → reranker → top-50
text2cypher       : LLM génère une requête Cypher (schéma en contexte) → résultats bruts (≤50)
```

Deux graphes Neo4j sur le même corpus (~100 documents, chunking basique sans
pre-processing structuré) : statique (ontologie curatée, ~260 lignes de schéma, non
revue par un expert métier) et dynamique (ontologie auto-extraite, 11 245 lignes de
schéma, 5 718 types de nœuds). Détail : `run_analysis_graph.md` §1.2.

Reranker et LLM identiques au pipeline HybridRAG.

---

## Scores synthétiques — HybridRAG (15 runs, `run_analysis.md`)

| Run | FC Recall | Faithfulness | Cit. Recall | Ctx Recall | Diversity | **Global** |
|-----|:---:|:---:|:---:|:---:|:---:|:---:|
| baseline | 44.87% | 88.09% | 26.72% | 38.48% | 26.69% | **51.35%** |
| baseline_raw | 45.66% | 77.49% | 18.26% | 41.72% | 33.28% | **48.52%** |
| db_rrf | 44.17% | 89.19% | 27.99% | 44.56% | 25.86% | **53.06%** |
| df_w | 45.46% | 86.05% | 21.50% | 42.37% | 25.59% | **50.91%** |
| df_rrf | 43.38% | 88.48% | 27.33% | 44.08% | 25.60% | **52.42%** |
| dbf_w | 45.67% | 83.48% | 24.16% | 40.34% | 26.07% | **50.56%** |
| dbf_rrf | 46.34% | 89.02% | 28.55% | 41.92% | 25.93% | **53.24%** |
| dbf_rrf_top25 | 49.55% | 89.62% | 33.10% | 52.86% | 27.41% | **57.93%** |
| **dbf_rrf_top50** ⭐ | 54.40% | 88.83% | 36.47% | 75.32% | 29.26% | **65.55%** |
| dbf_rrf_top100 | 57.82% | 91.01% | 35.33% | 87.00% | 32.17% | **70.04%** |
| dbf_rrf_top50_comb | 55.76% | 92.48% | 36.39% | 75.00% | 29.26% | **66.84%** |
| dbf_rrf_top50_thr-2 | 55.84% | 89.04% | 31.79% | 71.66% | 28.63% | **64.49%** |
| dbf_rrf_top50_thr-1 | 53.28% | 89.68% | 35.16% | 66.00% | 27.67% | **62.84%** |
| dbf_rrf_top50_thr0 | 53.78% | 90.17% | 29.67% | 57.40% | 26.80% | **60.17%** |
| dbf_rrf_top50_thr1 | 47.18% | 87.97% | 27.36% | 51.77% | 26.10% | **55.55%** |

Table complète (11 métriques) : `run_analysis.md` Appendix B.
`global_rag_score` = 0.25 × Context Recall + 0.35 × FC Recall + 0.15 × Citation Recall + 0.25 × Faithfulness.

⭐ **`dbf_rrf_top50` est la configuration championne retenue** (§7.3 de `run_analysis.md`) :
dense + BM25 + Fermi, RRF, top-50, sans threshold ni combined ranking.

---

## Scores synthétiques — GraphRAG (5 runs, `run_analysis_graph.md`)

| Run | FC Recall | Faithfulness | Cit. Recall | Ctx Recall | Hall ↓ | **Global** |
|-----|:---:|:---:|:---:|:---:|:---:|:---:|
| **graph_baseline** | 42.33% | 80.67% | 13.46% | 62.04% | 10.34% | **52.51%** |
| graph_st_hybrid_cypher | 37.48% | 78.14% | 10.31% | 55.11% | 11.26% | **47.97%** |
| graph_dy_hybrid_cypher | 34.43% | 79.73% | 12.42% | 43.57% | 10.25% | **44.83%** |
| graph_st_text2cypher | 17.79% | 55.73% | 1.92% | 19.49% | 36.25% | **25.32%** |
| graph_dy_text2cypher | 14.76% | 47.30% | 2.05% | 16.41% | 50.61% | **21.35%** |

Table complète : `run_analysis_graph.md` Appendix B.

**Aucun mode exploitant la structure du graphe ne bat le retrieval plat** (`graph_baseline`)
sur ce corpus — verdict détaillé en `run_analysis_graph.md` §2, §4, §8.

---

## Détail par run — HybridRAG

### `baseline`

**Objectif** : configuration de référence, dense + BM25, WeightedRanker. Point de départ
de toute l'étude (§1.6, série 1 et 2 de `run_analysis.md`).

| Paramètre | Valeur |
|-----------|--------|
| Collection | `baseline_fragus` |
| Ranker | `WeightedRanker` |
| Champs | `embedding` (w=0.5, COSINE) · `sparse_bm25` (w=0.5, IP) |
| Candidates | 100 |
| Top-N | 10 |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` · CPU · max_length=512 |
| ranking_mode | `reranker` |
| LLM | `mistral-large-latest` · max_tokens=1024 |

**Commande** :
```bash
uv run scripts/rag_evaluation/prepare_eval_dataset.py \
  scripts/rag_evaluation/data/rag_evaluation_dataset.xlsx \
  --experiment-id baseline \
  --dense-weight 0.5 --bm25-weight 0.5
```

---

### `baseline_raw`

**Objectif** : Série 1 — isoler l'effet de l'ingestion (Docling structuré vs pypdf brut),
même retrieval que `baseline`. Détail : `run_analysis.md` §2.

| Paramètre | Valeur |
|-----------|--------|
| Collection | `brut_fragus` |
| Extraction | `pypdf`, texte natif page par page, **sans OCR** |
| Chunking | fenêtres de **500 tokens** (tokenizer BGE-M3), recouvrement **100 tokens**, filtre queue < 20 caractères (`config.toml` → `[chunking_brut]`) |
| Ranker | `WeightedRanker` |
| Champs | `embedding` (w=0.5, COSINE) · `sparse_bm25` (w=0.5, IP) |
| Candidates | 100 |
| Top-N | 10 |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| LLM | `mistral-large-latest` |

**Commande d'ingestion** :
```bash
uv run scripts/pipeline/brute_corpus.py --manifest output/manifest_fragus_clean.dedup.json
```
**Commande d'éval** : identique à `baseline`, `--experiment-id baseline_raw` sur la
collection `brut_fragus`.

**Verdict (§2.3)** : ingestion structurée retenue — écart global non significatif mais
faithfulness (−10.59 pp) et citation recall (−8.46 pp) significativement dégradés par le
brut ; la diversity, seule métrique où le brut gagne significativement, est un faux
positif (hétérogénéité = bruit, pas richesse).

---

### `db_rrf`, `df_w`, `df_rrf`, `dbf_w`, `dbf_rrf`

**Objectif** : Série 2 — combinaison de champs (dense+BM25 / dense+Fermi / les 3) ×
stratégie de fusion (weighted / RRF), top-10 fixe pour isoler ces deux axes. Détail :
`run_analysis.md` §3.

| Run | Champs actifs | Ranker |
|-----|---------------|--------|
| `db_rrf` | embedding, sparse_bm25 | RRFRanker (k=60) |
| `df_w` | embedding, sparse_fermi | WeightedRanker (0.5 / 0.5) |
| `df_rrf` | embedding, sparse_fermi | RRFRanker (k=60) |
| `dbf_w` | embedding, sparse_bm25, sparse_fermi | WeightedRanker (0.33 / 0.33 / 0.33) |
| `dbf_rrf` | embedding, sparse_bm25, sparse_fermi | RRFRanker (k=60) |

Communs aux cinq : Collection `baseline_fragus` · Candidates 100 · Top-N 10 · Reranker
`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` · ranking_mode `reranker` · LLM
`mistral-large-latest`.

> Avec `RRFRanker`, les poids `--*-weight` n'influencent pas la fusion (fusion rang-based) ;
> ils servent uniquement à activer le champ dans `search_config`.

**Commandes** :
```bash
uv run scripts/rag_evaluation/prepare_eval_dataset.py \
  scripts/rag_evaluation/data/rag_evaluation_dataset.xlsx \
  --experiment-id db_rrf --ranker rrf --bm25-weight 0.5

uv run scripts/rag_evaluation/prepare_eval_dataset.py \
  scripts/rag_evaluation/data/rag_evaluation_dataset.xlsx \
  --experiment-id df_w --dense-weight 0.5 --bm25-weight 0.0 --fermi-weight 0.5

uv run scripts/rag_evaluation/prepare_eval_dataset.py \
  scripts/rag_evaluation/data/rag_evaluation_dataset.xlsx \
  --experiment-id df_rrf --ranker rrf --bm25-weight 0.0 --fermi-weight 0.5

uv run scripts/rag_evaluation/prepare_eval_dataset.py \
  scripts/rag_evaluation/data/rag_evaluation_dataset.xlsx \
  --experiment-id dbf_w --dense-weight 0.33 --bm25-weight 0.33 --fermi-weight 0.33

uv run scripts/rag_evaluation/prepare_eval_dataset.py \
  scripts/rag_evaluation/data/rag_evaluation_dataset.xlsx \
  --experiment-id dbf_rrf --ranker rrf --fermi-weight 0.33 --top-n 10
```

**Verdict (§3.3)** : `dbf_rrf` promu (Global 53.24%, meilleur de la série mais aucun écart
individuellement significatif à top-10). Pattern directionnel net : RRF bat systématiquement
Weighted sur les 3 familles de champs (+1.5 à +2.7 pp), l'avantage croissant avec le nombre
de champs à calibrer ; Fermi est un amplificateur (utile seulement sous RRF, et seulement en
complément de BM25, jamais en remplacement).

---

### Série `dbf_rrf_top{25,50,100}`

**Objectif** : Série 3 — l'axe décisif de l'étude. Effet du nombre de chunks passés au LLM
sur la config 3-champs + RRF. Détail : `run_analysis.md` §4.

| Paramètre | Valeur |
|-----------|--------|
| Collection | `baseline_fragus` |
| Ranker | `RRFRanker` (k=60) |
| Champs | `embedding` (COSINE, nprobe=16) · `sparse_bm25` (IP) · `sparse_fermi` (IP) |
| Candidates | 100 |
| Top-N | **25 / 50 / 100** selon le run (10 = `dbf_rrf` ci-dessus) |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` · CPU · max_length=512 |
| ranking_mode | `reranker` |
| LLM | `mistral-large-latest` · max_tokens=1024 |

**Commandes** :
```bash
uv run scripts/rag_evaluation/prepare_eval_dataset.py \
  scripts/rag_evaluation/data/rag_evaluation_dataset.xlsx \
  --experiment-id dbf_rrf_top25 --ranker rrf --fermi-weight 0.5 --top-n 25

uv run scripts/rag_evaluation/prepare_eval_dataset.py \
  scripts/rag_evaluation/data/rag_evaluation_dataset.xlsx \
  --experiment-id dbf_rrf_top50 --ranker rrf --fermi-weight 0.5 --top-n 50

uv run scripts/rag_evaluation/prepare_eval_dataset.py \
  scripts/rag_evaluation/data/rag_evaluation_dataset.xlsx \
  --experiment-id dbf_rrf_top100 --ranker rrf --fermi-weight 0.5 --top-n 100
```

**Verdict (§4.3)** : `dbf_rrf_top50` promu **champion de l'étude** (Global 65.55%). Gain
top-10→top-50 significatif et large (+12.31 pp) tiré par le context recall (+33.41 pp,
significatif) ; la faithfulness reste statistiquement stable sur toute la plage 10→100
(88.8–91.0%) et l'hallucination *diminue* avec le contexte. Top-100 (70.04%) n'apporte pas
de gain significatif sur top-50 (+4.49 pp) pour un coût de prompt doublé — top-50 est le
point d'opération le plus haut dont le gain est prouvé de bout en bout.

---

### `dbf_rrf_top50_comb`

**Objectif** : Série 4 — classement final par moyenne 50/50 des scores min-max-normalisés
(fusion RRF Milvus + score reranker) au lieu du score reranker seul, sur la base top-50.
Détail : `run_analysis.md` §5.

| Paramètre | Valeur |
|-----------|--------|
| Collection | `baseline_fragus` |
| Ranker | `RRFRanker` (k=60) |
| Champs | `embedding` (COSINE, nprobe=16) · `sparse_bm25` (IP) · `sparse_fermi` (IP) |
| Candidates | 100 |
| Top-N | 50 |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` · CPU · max_length=512 |
| ranking_mode | `combined` (min-max par requête sur le pool de 100, moyenne 50/50) |
| reranker_threshold | aucun |
| LLM | `mistral-large-latest` · max_tokens=1024 |

**Commande** :
```bash
uv run scripts/rag_evaluation/prepare_eval_dataset.py \
  scripts/rag_evaluation/data/rag_evaluation_dataset.xlsx \
  --experiment-id dbf_rrf_top50_comb --ranker rrf --fermi-weight 0.5 --top-n 50 \
  --ranking-mode combined
```

**Verdict (§5.3)** : combined mode ne change rien — 0/17 métriques significatives, la plus
faible différenciation de toute l'étude. À une coupe 100→50, les classements RRF et
cross-encoder sont déjà largement d'accord ; le blend ne peut ni aider ni nuire.
Classement standard (cross-encoder seul) retenu.

---

### Série `dbf_rrf_top50_thr{-2,-1,0,+1}`

**Objectif** : Série 5 — filtrage post-reranking par seuil minimum sur le score
cross-encoder, base top-50. Détail : `run_analysis.md` §6.

| Paramètre | Valeur |
|-----------|--------|
| Collection | `baseline_fragus` |
| Ranker | `RRFRanker` (k=60) |
| Champs | `embedding` (COSINE, nprobe=16) · `sparse_bm25` (IP) · `sparse_fermi` (IP) |
| Candidates | 100 |
| Top-N | 50 (avant filtre) |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` · CPU · max_length=512 |
| ranking_mode | `reranker` |
| reranker_threshold | **−2.0 / −1.0 / 0.0 / +1.0** selon le run |
| LLM | `mistral-large-latest` · max_tokens=1024 |

Seuils ancrés sur la distribution réelle des scores reranker servis par le champion
(médiane 0.49, P5 −2.95, P95 4.94 — voir §6.1). Échelle arrêtée à +1.0 : au-delà, le filtre
commence à vider entièrement le contexte de certaines questions (4/65 déjà à +1.0), ce qui
mélangerait deux régimes de mesure différents.

**Commande** — le run `thr-2` effectue le retrieval complet ; les seuils supérieurs sont
dérivés sans nouveau retrieval à partir de son parquet de contextes (mêmes ensembles de
chunks filtrés à des seuils différents) :
```bash
uv run scripts/rag_evaluation/prepare_eval_dataset.py \
  scripts/rag_evaluation/data/rag_evaluation_dataset.xlsx \
  --experiment-id dbf_rrf_top50_thr-2 --ranker rrf --fermi-weight 0.5 --top-n 50 \
  --reranker-threshold -2.0

uv run scripts/rag_evaluation/prepare_eval_dataset.py \
  --contexts-parquet scripts/rag_evaluation/data/results/eval_test_dbf_rrf_top50_thr-2_contexts.parquet \
  --experiment-id dbf_rrf_top50_thr-1 --reranker-threshold -1.0
# idem pour thr0 (0.0) et thr1 (+1.0)
```

**Verdict (§6.4)** : aucun seuil recommandé — tous dégradent le context recall en
proportion directe de la coupe, et thr+1 est la seule régression globale significative de
toute l'étude (−10.00 pp). Point notable : thr-2 et thr-1 atteignent les hallucination rates
les plus bas de l'étude (1.6–1.7%) — option seulement pour un déploiement où minimiser
l'hallucination prime sur la complétude.

---

## Détail par run — GraphRAG

### `graph_baseline`

**Objectif** : baseline interne — le graphe Neo4j utilisé comme simple magasin de
documents, retrieval plat sans exploiter la structure entités. Référence pour les 4 autres
runs. Détail : `run_analysis_graph.md` §3.

| Paramètre | Valeur |
|-----------|--------|
| Graphe | statique (`bolt://localhost:7687`) |
| Mode | `flat_rrf` — BGE-M3 ANN + BM25 Lucene → RRF (k=60) |
| Candidates | 100 |
| Top-N | 50 |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| LLM | `mistral-large-latest` |

**Commande** :
```bash
uv run scripts/rag_evaluation/prepare_eval_dataset_graph.py \
    scripts/rag_evaluation/data/rag_evaluation_dataset.xlsx \
    --experiment-id graph_baseline --mode flat_rrf \
    --neo4j-url bolt://localhost:7687
```

**Verdict (§3.1)** : substrat RAG exploitable (Global 52.51%) mais 13.04 pp sous le
champion HybridRAG `dbf_rrf_top50` — écart attribuable à des facteurs non liés au graphe
(métadonnées de citation plus pauvres, ingestion basique sans Docling, absence du champ
Fermi), pas à Neo4j vs Milvus (§3.2).

---

### `graph_st_hybrid_cypher`, `graph_dy_hybrid_cypher`

**Objectif** : retrieval structurel par traversée 2-hop dans le graphe entités (inspiré
HippoRAG), sur chaque ontologie. Détail : `run_analysis_graph.md` §4.1.

| Paramètre | Valeur |
|-----------|--------|
| Graphe | statique (`st`, port 7687) / dynamique (`dy`, port 7688) |
| Mode | `hybrid_cypher` — 10 seeds ANN → traversée 2-hop entités → pool ≤100 → reranker → top-50 |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| LLM | `mistral-large-latest` |

**Commandes** :
```bash
uv run scripts/rag_evaluation/prepare_eval_dataset_graph.py \
    scripts/rag_evaluation/data/rag_evaluation_dataset.xlsx \
    --experiment-id graph_st_hybrid_cypher --mode hybrid_cypher \
    --neo4j-url bolt://localhost:7687

uv run scripts/rag_evaluation/prepare_eval_dataset_graph.py \
    scripts/rag_evaluation/data/rag_evaluation_dataset.xlsx \
    --experiment-id graph_dy_hybrid_cypher --mode hybrid_cypher
```

**Verdict (§4.1, §4.3)** : perd contre `graph_baseline` sur les deux graphes (−4.54 pp
statique, non significatif ; −7.68 pp dynamique, **significatif**) — problème d'entonnoir :
le candidate pool vient exclusivement du voisinage 2-hop des 10 seeds, contre l'index
entier pour le retrieval plat. Ce que la traversée trouve est pertinent (coverage et Hub IA
F-Score à parité ou meilleurs), mais elle n'en trouve pas assez.

---

### `graph_st_text2cypher`, `graph_dy_text2cypher`

**Objectif** : génération de requête Cypher par LLM à partir du schéma du graphe, sur
chaque ontologie. Détail : `run_analysis_graph.md` §4.2, §6.

| Paramètre | Valeur |
|-----------|--------|
| Graphe | statique (`st`) / dynamique (`dy`) |
| Mode | `text2cypher` — 1 requête Cypher générée (pattern `(entity)-[:FROM_CHUNK]->(chunk)`), ≤50 records bruts, pas de reranking |
| Schéma exposé | statique : schéma complet (~260 lignes) · dynamique : élagué par similarité embedding (BGE-M3, top-100 lignes sur 11 245) |
| LLM | `mistral-large-latest` |

**Commandes** :
```bash
uv run scripts/rag_evaluation/prepare_eval_dataset_graph.py \
    scripts/rag_evaluation/data/rag_evaluation_dataset.xlsx \
    --experiment-id graph_st_text2cypher --mode text2cypher \
    --neo4j-url bolt://localhost:7687 --no-prune-schema

uv run scripts/rag_evaluation/prepare_eval_dataset_graph.py \
    scripts/rag_evaluation/data/rag_evaluation_dataset.xlsx \
    --experiment-id graph_dy_text2cypher --mode text2cypher
```

**Verdict (§4.2, §6.2, §8)** : effondrement, les deux plus grandes régressions de l'étude
face au baseline (−27.19 pp statique, −31.17 pp dynamique, toutes deux significatives), et
significativement pires que `hybrid_cypher` à graphe égal (+22 à +23 pp pour hybrid). Cause
racine : problème de disponibilité du retrieval — 40% des questions (statique) à 80%
(dynamique) reçoivent **zéro chunk** (requête Cypher sans résultat, aucun fallback), et
chaque contexte vide se traduit mécaniquement en réponse non ancrée. Non viable en l'état ;
correctif naturel = boucle agentique (§7).

---

## Runs exploratoires antérieurs (hors périmètre — valeurs obsolètes)

Les runs suivants ont été lancés avant la mise en place du protocole en 5 séries et du
jeu de contextes complets (parquet) documentés ci-dessus. Ils utilisaient une échelle de
seuils différente (`thr-3/-2/-1.5/-1` sur base top-25, vs `thr-2/-1/0/+1` sur base top-50
aujourd'hui) et des scores calculés avant la régénération des contextes. **Conservés à
titre d'archive uniquement — ne pas citer ces chiffres, se référer aux runs ci-dessus.**

| run (obsolète) | Remplacé par |
|-----------------|--------------|
| `dense_bm25_25_reranker_01` | `dbf_rrf_top25` (précurseur non-Fermi de la série top-n) |
| `db_w_top25` / `_top50` / `_top75` / `_top100` | Série `dbf_rrf_top{25,50,100}` |
| `dbf_rrf_top25_comb` | `dbf_rrf_top50_comb` (le combined mode est désormais testé sur la base top-50, promue championne) |
| `dbf_rrf_top25_thr-3` / `-2` / `-1.5` / `-1` | Série `dbf_rrf_top50_thr{-2,-1,0,+1}` |

---

## Commandes de reproduction

**Éval (tout run HybridRAG ou GraphRAG déjà généré)** :
```bash
uv run scripts/rag_evaluation/rag_evaluation.py --experiment-id <run_id>
```

**Statistiques pairwise (Wilcoxon + correction Benjamini-Hochberg)** :
```bash
cd scripts/rag_evaluation && uv run compare_rag_runs.py <run_A> <run_B>
```

**Replay retrieval-only** (dérivation de variantes top-n / threshold sans regénération LLM,
HybridRAG et GraphRAG) :
```bash
uv run scripts/rag_evaluation/replay_retrieval.py --help
```
