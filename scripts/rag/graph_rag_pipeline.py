"""
graph_rag_pipeline.py
=====================

Non-agentic GraphRAG pipeline on a Neo4j graph. Three retrieval modes:

  flat_rrf      — BGE-M3 ANN + BM25 Lucene → RRF (k=60) → cross-encoder
                  reranker → top-25. Schema-agnostic baseline equivalent to
                  db_rrf_top25 on the Hybrid RAG side, but running on the
                  Neo4j corpus. Use this as the GraphRAG internal baseline.

  hybrid_cypher — BGE-M3 ANN (top-5 seeds) → 2-hop graph traversal to collect
                  bridged Chunk nodes (passages linked via shared entities) →
                  cross-encoder reranker → top-25. Adds graph-structural signal
                  on top of vector similarity: finds chunks that are
                  thematically related through entity paths even if not
                  vectorially close to the query (HippoRAG pattern).

  text2cypher   — LLM generates a Cypher query from the user question with
                  per-query keyword-based schema pruning (SOTA exact-match
                  approach per Neo4j 2025 research); executed directly on
                  Neo4j; results form the LLM context. Best suited for graphs
                  with a clean, well-named ontology.

Designed for two graph configurations:
  - static ontology  : clean pre-built schema — text2cypher works well.
  - dynamic ontology : large/inconsistent schema — hybrid_cypher is more robust
                       (schema-agnostic traversal).

Graph assumptions (node labels):
  - Chunk        : text passages with `text`, `source`, `embedding` properties.
  - SourceDocument: source document nodes.
  - All other labels are treated as entity nodes for graph traversal.

Quick start::

    from scripts.rag.graph_rag_pipeline import GraphRAGPipeline

    # Flat RRF baseline (equivalent to db_rrf_top25 on the Neo4j corpus)
    rag = GraphRAGPipeline(mode="flat_rrf")

    # Hybrid-Cypher (2-hop traversal + reranker)
    rag = GraphRAGPipeline(mode="hybrid_cypher", database="my_graph")

    # Text2Cypher on a graph with a clean ontology
    rag = GraphRAGPipeline(mode="text2cypher", database="static_graph")

    # Example query against the French corpus
    # ("What are the approval criteria for Type B packages?"):
    result = rag.ask("Quels sont les critères d'agrément des colis de type B ?")
    print(result["answer"])
    rag.close()
"""

from __future__ import annotations

import logging
import os
import re
import tomllib
from pathlib import Path
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from neo4j import GraphDatabase

_log = logging.getLogger(__name__)


# ─── Config (shared paths with hybrid_rag_pipeline.py) ───────────────────────

_CONFIG_PATH = Path(__file__).parent / "config.toml"

with open(_CONFIG_PATH, "rb") as _f:
    _CFG = tomllib.load(_f)

_LLM_CFG = _CFG.get("llm", {})
_EMB_CFG  = _CFG.get("embedding", {})

_PROXY_PREFIX   = _LLM_CFG.get("proxy_prefix", "http://localhost:8081")
LLM_MODEL_KEY   = _LLM_CFG.get("model_key", "mistral-large-latest")
LLM_MAX_TOKENS  = _LLM_CFG.get("max_tokens", 2048)
LLM_TIMEOUT     = _LLM_CFG.get("timeout", 120.0)
EMBEDDING_MODEL = _EMB_CFG.get("model_key", "BAAI/bge-m3")

# ─── Neo4j defaults (override per experiment) ─────────────────────────────────
# Defaults target a throwaway local Neo4j; set NEO4J_PASSWORD for anything else.

NEO4J_URL         = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USER        = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD    = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE    = os.getenv("NEO4J_DATABASE", "neo4j")
VECTOR_INDEX_NAME = "chunk-vector-index"
TOP_K             = 5
TRAVERSAL_LIMIT   = 100

# ─── flat_rrf defaults ────────────────────────────────────────────────────────

FULLTEXT_INDEX_NAME = "chunk-text-index"
CANDIDATES          = 100
TOP_N               = 25
RRF_K               = 60
RERANKER_MODEL      = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
RERANKER_DEVICE     = "cpu"
RERANKER_URL        = os.environ.get("RERANKER_URL") or None




# ─── Text2Cypher prompt ───────────────────────────────────────────────────────

_TEXT2CYPHER_PROMPT = """\
You are an expert in translating natural language questions into Cypher queries \
for a Neo4j graph database containing French nuclear regulatory documents.

Graph schema (relevant excerpt):
{schema}

{examples_block}\
Retrieval pattern (ALWAYS use this structure to get text):
  MATCH (entity:Package)-[:FROM_CHUNK]->(chunk:Chunk)
  WHERE toLower(entity.name) CONTAINS "keyword"
  RETURN chunk.text AS text, chunk.source AS source
  LIMIT 50

  Replace `Package` with the most relevant node label from the schema above.
  Use at most ONE simple WHERE condition on entity.name or entity.normalized_name.
  Do NOT chain multiple AND conditions — keep the filter broad so results are returned.

Rules:
- Use ONLY the node labels, relationship types and properties listed above.
- Do NOT invent labels, relationship types or properties.
- ALWAYS traverse via (entity:Label)-[:FROM_CHUNK]->(chunk:Chunk) to retrieve text.
- ALWAYS return chunk.text AS text and chunk.source AS source.
- Always include a LIMIT clause (max 50).
- Return the Cypher query ONLY — no explanation, no markdown fences.

Question: {question}
Cypher:\
"""

_EXAMPLES_BLOCK = "Examples:\n{examples}\n\n"


# ─── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are IAGO, an intelligent assistant working for ASNR (the French Nuclear Safety and Radiation Protection Authority).

### Method for answering a question
- Start by quoting the definitions needed to understand the question.
- In your final answer, include only elements that directly answer the question asked, with no digression: do not mention information unrelated to it.
- Support each of your statements with reliable sources (accessible documents), indicating at the end of each sentence, statement or paragraph the reference to the sources used to produce that statement.
- Be very careful not to confuse the different packages.
- Cite the document and the page where the information is found.
- Mark citations with the excerpt index in the form "[1]", "[3]", etc.
- At the end of your answer, always list the complete set of excerpts that were useful in answering the question (excerpt number in brackets, document name, page... - Example: [1] SSR-6, page 8).
- Use only the information present in the documentation provided to you, and nothing else. In particular, you are FORBIDDEN from guessing or from using your internal knowledge. Do not state anything that is not supported by the documentation at your disposal.
- If you cannot find the answer in the documentation, say that you do not know; do not make anything up.
- Always answer in French.

### Nuclear expertise
- Package types correspond to specific models of transport packagings used for the safe transport of radioactive materials.
"""


# ─── Internal helpers ─────────────────────────────────────────────────────────

_STOPWORDS = {
    "de", "du", "la", "le", "les", "un", "une", "des", "et", "ou", "en",
    "à", "au", "aux", "par", "pour", "sur", "dans", "avec", "que", "qui",
    "quels", "quelles", "quel", "quelle", "est", "sont", "the", "a", "of",
    "in", "for", "is", "are", "to", "and", "or",
}


def _query_tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower())) - _STOPWORDS


_ALWAYS_KEEP = {"chunk", "from_chunk", "from_document"}


def _prune_schema(schema: str, question: str) -> str:
    tokens = _query_tokens(question)
    pruned = [
        ln for ln in schema.splitlines()
        if _query_tokens(ln) & tokens or _query_tokens(ln) & _ALWAYS_KEEP
    ]
    return "\n".join(pruned) if pruned else schema


def _extract_cypher(text: str) -> str:
    return re.sub(r"^```(?:cypher)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE).strip()


def _lucene_query(text: str) -> str:
    """Strip Lucene special chars and return a plain keyword query."""
    cleaned = re.sub(r'[+\-&|!(){}\[\]^"~*?:\\/]', " ", text)
    return " ".join(cleaned.split()) or "*"


# ─── Pipeline class ───────────────────────────────────────────────────────────


class GraphRAGPipeline:
    """Non-agentic GraphRAG pipeline on a Neo4j graph.

    Parameters
    ----------
    neo4j_url:
        Bolt URI of the Neo4j instance (e.g. ``"bolt://localhost:7687"``).
    neo4j_user / neo4j_password:
        Authentication credentials.
    database:
        Name of the target Neo4j database.
    mode:
        ``"flat_rrf"``      — ANN + BM25 Lucene → RRF → reranker → top-25.
                              GraphRAG internal baseline (no graph traversal).
        ``"hybrid_cypher"`` — ANN seeds → 2-hop traversal → reranker → top-25.
        ``"text2cypher"``   — LLM-generated Cypher with per-query schema pruning.
    vector_index_name:
        BGE-M3 vector index name. Used by ``flat_rrf`` and ``hybrid_cypher``.
    top_k:
        Number of ANN seed nodes (``hybrid_cypher`` only). Default: 5.
    traversal_limit:
        Maximum number of bridged chunks collected by the 2-hop traversal
        (``hybrid_cypher`` only). Default: 100.
    cypher_examples:
        Few-shot (question, cypher) pairs for ``text2cypher`` mode.
        Each entry must be a dict with keys ``"question"`` and ``"cypher"``.
    prune_schema:
        Enable per-query exact-match keyword schema pruning (``text2cypher``).
        SOTA approach per Neo4j 2025 research. Default: True.
    llm_model_key:
        Cleyrop proxy model key for generation (and Cypher generation in
        ``text2cypher`` mode).
    fulltext_index_name / candidates / top_n / rrf_k:
        ``flat_rrf`` retrieval parameters (BM25 index, candidate pool, final
        top-N, RRF smoothing constant).
    reranker_model / reranker_device / reranker_url:
        Cross-encoder reranker config. Used by ``flat_rrf`` and
        ``hybrid_cypher``. Pass ``reranker_url`` to use a remote reranker
        service instead of a local model.
    """

    def __init__(
        self,
        neo4j_url: str                  = NEO4J_URL,
        neo4j_user: str                 = NEO4J_USER,
        neo4j_password: str             = NEO4J_PASSWORD,
        database: str                   = NEO4J_DATABASE,
        mode: str                       = "hybrid_cypher",
        vector_index_name: str          = VECTOR_INDEX_NAME,
        top_k: int                      = TOP_K,
        traversal_limit: int            = TRAVERSAL_LIMIT,
        cypher_examples: Optional[list] = None,
        prune_schema: bool              = True,
        llm_model_key: str              = LLM_MODEL_KEY,
        llm_max_tokens: int             = LLM_MAX_TOKENS,
        llm_timeout: float              = LLM_TIMEOUT,
        system_prompt: str              = SYSTEM_PROMPT,
        # flat_rrf params
        fulltext_index_name: str        = FULLTEXT_INDEX_NAME,
        candidates: int                 = CANDIDATES,
        top_n: int                      = TOP_N,
        rrf_k: int                      = RRF_K,
        reranker_model: str             = RERANKER_MODEL,
        reranker_device: str            = RERANKER_DEVICE,
        reranker_url: Optional[str]     = RERANKER_URL,
    ) -> None:
        if mode not in ("text2cypher", "hybrid_cypher", "flat_rrf"):
            raise ValueError(
                f"mode must be 'text2cypher', 'hybrid_cypher' or 'flat_rrf', got {mode!r}"
            )

        self.mode            = mode
        self.database        = database
        self.top_k           = top_k
        self.traversal_limit = traversal_limit
        self.vector_index    = vector_index_name
        self.prune_schema    = prune_schema
        self.system_prompt   = system_prompt
        self._examples       = cypher_examples or []

        # flat_rrf
        self._fulltext_index = fulltext_index_name
        self._candidates     = candidates
        self._top_n          = top_n
        self._rrf_k          = rrf_k
        self._reranker_url   = reranker_url

        # Neo4j driver
        self._driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))

        # Resolve LLM endpoint via Cleyrop proxy (reuse existing helper)
        from scripts.rag.hybrid_rag_pipeline import _resolve_all_providers_for_key
        providers = _resolve_all_providers_for_key(_PROXY_PREFIX, llm_model_key)
        if not providers:
            raise RuntimeError(f"Model '{llm_model_key}' not found on proxy {_PROXY_PREFIX}")
        entry = providers[0]

        self._llm = ChatOpenAI(
            base_url=entry["endpoint"],
            model=entry["name"],
            api_key="dummy",
            max_tokens=llm_max_tokens,
            timeout=llm_timeout,
        )
        self._cypher_llm = ChatOpenAI(
            base_url=entry["endpoint"],
            model=entry["name"],
            api_key="dummy",
            max_tokens=512,
            timeout=llm_timeout,
        )

        if mode == "text2cypher":
            _log.info("Fetching Neo4j schema for database '%s'…", database)
            self._schema = self._fetch_schema()
            _log.info("Schema fetched (%d chars).", len(self._schema))

        if mode in ("hybrid_cypher", "flat_rrf"):
            self._embed = self._build_embedder()

        if mode in ("flat_rrf", "hybrid_cypher"):
            if reranker_url:
                _log.info("Using remote reranker at %s", reranker_url)
                self._reranker = None
            else:
                from sentence_transformers import CrossEncoder
                _log.info("Loading cross-encoder (%s)…", reranker_model)
                self._reranker = CrossEncoder(
                    model_name_or_path=reranker_model,
                    device=reranker_device,
                )

    # ── Schema (text2cypher) ──────────────────────────────────────────────────

    def _fetch_schema(self) -> str:
        try:
            r = self._driver.execute_query(
                "CALL db.schema.nodeTypeProperties() "
                "YIELD nodeType, propertyName, propertyTypes "
                "RETURN nodeType, collect(propertyName + ': ' + propertyTypes[0]) AS props",
                database_=self.database,
            )
            node_lines = [
                f"Node {rec['nodeType']}: {', '.join(rec['props'])}"
                for rec in r.records
            ]
            r2 = self._driver.execute_query(
                "CALL db.schema.relTypeProperties() "
                "YIELD relType, propertyName "
                "RETURN relType, collect(propertyName) AS props",
                database_=self.database,
            )
            rel_lines = [
                f"Rel {rec['relType']}: {', '.join(rec['props']) or '(no properties)'}"
                for rec in r2.records
            ]
            return "\n".join(node_lines + rel_lines)
        except Exception as exc:
            _log.warning("Schema procedure failed (%s); falling back to label scan.", exc)
            r = self._driver.execute_query(
                "MATCH (n) UNWIND labels(n) AS lbl RETURN DISTINCT lbl LIMIT 50",
                database_=self.database,
            )
            labels = [rec["lbl"] for rec in r.records]
            r2 = self._driver.execute_query(
                "MATCH ()-[r]-() RETURN DISTINCT type(r) AS rel LIMIT 50",
                database_=self.database,
            )
            rels = [rec["rel"] for rec in r2.records]
            return (
                "Node labels: " + ", ".join(labels)
                + "\nRelationship types: " + ", ".join(rels)
            )

    # ── Embedder (hybrid_cypher) ──────────────────────────────────────────────

    def _build_embedder(self):
        from scripts.rag.hybrid_rag_pipeline import _build_embedding_client
        oai_client, model_name = _build_embedding_client(_PROXY_PREFIX, EMBEDDING_MODEL)

        def embed(text: str) -> list[float]:
            return oai_client.embeddings.create(model=model_name, input=[text]).data[0].embedding

        return embed

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, question: str) -> list[dict]:
        """Return context items: ``[{"text": ..., "metadata": {...}}, ...]``."""
        if self.mode == "text2cypher":
            return self._retrieve_text2cypher(question)
        if self.mode == "flat_rrf":
            return self._retrieve_flat_rrf(question)
        return self._retrieve_hybrid_cypher(question)

    def _retrieve_text2cypher(self, question: str) -> list[dict]:
        schema = _prune_schema(self._schema, question) if self.prune_schema else self._schema

        examples_block = ""
        if self._examples:
            ex_lines = "\n".join(
                f"Q: {ex['question']}\nCypher: {ex['cypher']}" for ex in self._examples
            )
            examples_block = _EXAMPLES_BLOCK.format(examples=ex_lines)

        prompt = _TEXT2CYPHER_PROMPT.format(
            schema=schema,
            examples_block=examples_block,
            question=question,
        )
        try:
            response = self._cypher_llm.invoke([HumanMessage(content=prompt)])
            cypher = _extract_cypher(response.content)
            _log.info("Generated Cypher:\n%s", cypher)
        except Exception as exc:
            _log.warning("Cypher generation failed: %s", exc)
            return []

        return self._execute_cypher(cypher)

    def _execute_cypher(self, cypher: str) -> list[dict]:
        try:
            result = self._driver.execute_query(cypher, database_=self.database)
            items = []
            for record in result.records:
                data = dict(record)
                text = data.get("text") or " | ".join(str(v) for v in data.values() if v is not None)
                source = data.get("source", "")
                name = source.split("/")[-1] if source else ""
                items.append({"text": text, "source": source, "name": name, "metadata": data, "cypher": cypher})
            return items
        except Exception as exc:
            _log.warning("Cypher execution failed: %s\nQuery: %s", exc, cypher)
            return []

    def _retrieve_flat_rrf(self, question: str) -> list[dict]:
        """ANN + BM25 Lucene → RRF fusion → cross-encoder reranking → top-n."""
        embedding = self._embed(question)

        # ANN search — top candidates by cosine similarity
        ann_res = self._driver.execute_query(
            "CALL db.index.vector.queryNodes($index, $k, $emb) "
            "YIELD node, score "
            "RETURN elementId(node) AS eid, node.text AS text, "
            "       coalesce(node.source, '') AS source, score AS ann_score",
            parameters_={"index": self.vector_index, "k": self._candidates, "emb": embedding},
            database_=self.database,
        )
        ann_rows = [dict(r) for r in ann_res.records if r["text"]]

        # BM25 full-text search — top candidates via Lucene
        lq = _lucene_query(question)
        bm25_res = self._driver.execute_query(
            "CALL db.index.fulltext.queryNodes($index, $query, {limit: $k}) "
            "YIELD node, score "
            "RETURN elementId(node) AS eid, node.text AS text, "
            "       coalesce(node.source, '') AS source, score AS bm25_score",
            parameters_={"index": self._fulltext_index, "query": lq, "k": self._candidates},
            database_=self.database,
        )
        bm25_rows = [dict(r) for r in bm25_res.records if r["text"]]

        # RRF fusion
        candidates = self._rrf_fuse(ann_rows, bm25_rows)

        # Cross-encoder reranking → top-n
        reranked = self._rerank_flat(question, candidates)
        return reranked[: self._top_n]

    def _rrf_fuse(self, *ranked_lists: list[dict]) -> list[dict]:
        """Fuse N ranked lists of chunk dicts using Reciprocal Rank Fusion."""
        rrf_scores: dict[str, float] = {}
        chunk_by_eid: dict[str, dict] = {}
        for ranked in ranked_lists:
            for rank, item in enumerate(ranked, 1):
                eid = item["eid"]
                rrf_scores[eid] = rrf_scores.get(eid, 0.0) + 1.0 / (self._rrf_k + rank)
                chunk_by_eid.setdefault(eid, item)
        sorted_eids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)
        return [chunk_by_eid[eid] for eid in sorted_eids]

    def _rerank_flat(self, query: str, chunks: list[dict]) -> list[dict]:
        """Score chunks with the cross-encoder and return sorted by score descending.

        Adds ``rerank_score`` and ``name`` (filename extracted from ``source``) to each chunk.
        The returned dicts expose: text, source, name, eid, rerank_score.
        """
        if not chunks:
            return []
        pairs = [[query, c.get("text") or ""] for c in chunks]
        if self._reranker_url:
            import requests
            resp = requests.post(
                f"{self._reranker_url}/score",
                json={"query": query, "texts": [p[1] for p in pairs]},
                timeout=60,
            )
            resp.raise_for_status()
            scores = resp.json()["scores"]
        else:
            scores = self._reranker.predict(pairs, show_progress_bar=False)
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)
            source = chunk.get("source") or ""
            chunk["name"] = source.split("/")[-1] if source else chunk.get("eid", "")
        return sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

    def _retrieve_hybrid_cypher(self, question: str) -> list[dict]:
        """ANN seeds → 2-hop graph traversal → disaggregated bridged chunks → cross-encoder → top-n."""
        embedding = self._embed(question)

        # ANN → seed chunks
        try:
            ann_res = self._driver.execute_query(
                "CALL db.index.vector.queryNodes($index, $k, $emb) "
                "YIELD node, score "
                "RETURN elementId(node) AS eid, node.text AS text, "
                "       coalesce(node.source, '') AS source",
                parameters_={"index": self.vector_index, "k": self.top_k, "emb": embedding},
                database_=self.database,
            )
        except Exception as exc:
            _log.warning("ANN search failed: %s", exc)
            return []

        seeds = [dict(r) for r in ann_res.records if r["text"]]
        if not seeds:
            return []

        # 2-hop traversal → bridged chunks (one row per chunk, deduplicated)
        seed_eids = [s["eid"] for s in seeds]
        try:
            traversal_res = self._driver.execute_query(
                "UNWIND $seed_eids AS seed_eid "
                "MATCH (node) WHERE elementId(node) = seed_eid "
                "OPTIONAL MATCH (node)--(entity) "
                "WHERE NOT (entity:Chunk OR entity:SourceDocument) "
                "OPTIONAL MATCH (entity)--(neighbor_entity) "
                "WHERE entity IS NOT NULL "
                "  AND NOT (neighbor_entity:Chunk OR neighbor_entity:SourceDocument) "
                "OPTIONAL MATCH (bridged:Chunk)--(neighbor_entity) "
                "WHERE neighbor_entity IS NOT NULL "
                "  AND bridged <> node AND bridged.text IS NOT NULL "
                "RETURN DISTINCT "
                "  elementId(bridged) AS eid, "
                "  bridged.text AS text, "
                "  coalesce(bridged.source, '') AS source "
                "LIMIT $limit",
                parameters_={"seed_eids": seed_eids, "limit": self.traversal_limit},
                database_=self.database,
            )
            bridged = [dict(r) for r in traversal_res.records if r["text"]]
        except Exception as exc:
            _log.warning("2-hop traversal failed: %s", exc)
            bridged = []

        # Merge seeds + bridged chunks, deduplicate by eid
        seen: dict[str, dict] = {}
        for s in seeds:
            seen[s["eid"]] = s
        for b in bridged:
            if b["eid"] not in seen:
                seen[b["eid"]] = b

        candidates = list(seen.values())
        if not candidates:
            return []

        # Cross-encoder rerank → top-n
        reranked = self._rerank_flat(question, candidates)
        return reranked[: self._top_n]

    # ── Generation ────────────────────────────────────────────────────────────

    def _format_context(self, items: list[dict]) -> str:
        if not items:
            return "(no context retrieved)"
        parts = []
        for i, item in enumerate(items, 1):
            text   = (item.get("text") or "").strip()
            source = item.get("name") or item.get("source", "").split("/")[-1] or "unknown source"
            parts.append(f"[{i}] {text}\n    Source: {source}")
        return "\n\n".join(parts)

    def ask(self, question: str) -> dict[str, Any]:
        """Retrieve context then generate an answer."""
        context_items = self.retrieve(question)
        context_str   = self._format_context(context_items)

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Context:\n{context_str}\n\nQuestion: {question}"),
        ]
        response = self._llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)

        return {
            "query":    question,
            "answer":   answer,
            "chunks":   context_items,
            "mode":     self.mode,
            "database": self.database,
        }

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
