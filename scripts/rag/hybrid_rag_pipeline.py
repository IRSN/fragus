"""
Hybrid RAG pipeline: Milvus multi-vector search → cross-encoder reranking → LLM generation.

Three independent embedding types are stored in a single Milvus collection and fused at
query time with either a weighted ranker or RRF (Reciprocal Rank Fusion):

  - Dense BGE-M3     (field ``embedding``)      — served via the AI-gen proxy
  - Sparse BM25      (field ``sparse_bm25``)    — client-side, reloaded from a persisted
                                                   idf/avgdl state file fitted on the corpus
  - Sparse Fermi     (field ``sparse_fermi``)   — local CPU inference with
                                                   ``atomic-canyon/fermi-1024``

Retrieval is performed with ``MilvusClient.hybrid_search`` + native Milvus fusion
(``WeightedRanker`` by default, ``RRFRanker`` as alternative).  Candidates are then
reranked with a cross-encoder (``cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`` by
default) before being passed to ``mistral-large-latest`` via the Cleyrop proxy.

Quick start (Jupyter)::

    from hybrid_rag_pipeline import HybridRAGPipeline

    rag = HybridRAGPipeline()
    result = rag.ask("Quels sont les critères d'agrément des colis de type B ?")
    print(result["answer"])
    for chunk in result["chunks"]:
        print(f"  [{chunk['rerank_score']:.3f}] {chunk['name']}")

Custom weights (dense heavier, BM25 only, etc.)::

    rag = HybridRAGPipeline(
        collection_name="baseline_fragus",
        search_config=[
            {"field": "embedding",   "weight": 0.6, "metric": "COSINE", "params": {"nprobe": 16}},
            {"field": "sparse_bm25", "weight": 0.4, "metric": "IP",     "params": {}},
        ],
    )

RRF fusion::

    rag = HybridRAGPipeline(ranker="rrf", rrf_k=60)
"""

from __future__ import annotations

import json
import logging
import os
import tomllib
import warnings
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen

import torch
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker, WeightedRanker
from pymilvus.model.sparse.bm25 import BM25EmbeddingFunction
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer
from sentence_transformers import CrossEncoder
from transformers import AutoModelForMaskedLM, AutoTokenizer


_log = logging.getLogger(__name__)


# ──────────────────────────────── Config ──────────────────────────────────────

_CONFIG_PATH   = Path(__file__).parent / "config.toml"
_ROOT          = Path(__file__).parent.parent.parent     # project root
_ROOT_CFG_PATH = _ROOT / "config.toml"

with open(_CONFIG_PATH, "rb") as _f:
    _CFG = tomllib.load(_f)
with open(_ROOT_CFG_PATH, "rb") as _f:
    _ROOT_CFG = tomllib.load(_f)

_MILVUS_CFG = _CFG.get("milvus", {})
_EMB_CFG    = _CFG.get("embedding", {})
_LLM_CFG    = _CFG.get("llm", {})
_FERMI_CFG  = _ROOT_CFG.get("embedding_fermi", {})
_FRAGUS_CFG = _ROOT_CFG.get("fragus", {})


# ──────────────────────────────── Constants ───────────────────────────────────

MILVUS_URI      = _MILVUS_CFG.get("uri", "http://localhost:19530")
COLLECTION_NAME = _MILVUS_CFG.get("collection", "baseline_fragus")

OUTPUT_FIELDS = ["chunk_id", "file_id", "name", "path", "text"]

_PROXY_PREFIX   = _LLM_CFG.get("proxy_prefix", "http://localhost:8081")
EMBEDDING_MODEL = _EMB_CFG.get("model_key", "BAAI/bge-m3")

FERMI_MODEL_ID    = _FERMI_CFG.get("model_id", "atomic-canyon/fermi-1024")
FERMI_MAX_SEQ_LEN = _FERMI_CFG.get("max_seq_len", 1024)
FERMI_ENCODER_URL = _FERMI_CFG.get("encoder_url", None)  # e.g. "http://mac-local:8000"

_BM25_STATE_DIR = _ROOT / _FRAGUS_CFG.get("bm25_state_dir", "artifacts/bm25")
BM25_LANGUAGE   = _FRAGUS_CFG.get("bm25_language", "fr")

RERANKER_MODEL  = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
RERANKER_DEVICE = "cpu"
RERANKER_URL    = os.environ.get("RERANKER_URL") or None   # e.g. "http://localhost:8001"

CANDIDATES = 100
TOP_N      = 10

LLM_MODEL_KEY  = _LLM_CFG.get("model_key", "mistral-large-latest")
LLM_MAX_TOKENS = _LLM_CFG.get("max_tokens", 1024)
LLM_TIMEOUT    = _LLM_CFG.get("timeout", 60.0)

# Default hybrid search configuration: all three vector fields with equal weights.
#
# Each entry describes one AnnSearchRequest sent to Milvus:
#   field   (str)   — vector field name in the collection
#   weight  (float) — score weight for WeightedRanker (ignored when ranker="rrf")
#   metric  (str)   — distance metric used at index time ("COSINE" dense, "IP" sparse)
#   params  (dict)  — additional search params (e.g. {"nprobe": 16} for IVF_FLAT)
DEFAULT_SEARCH_CONFIG: list[dict] = [
    {"field": "embedding",   "weight": 0.5, "metric": "COSINE", "params": {"nprobe": 16}},
    {"field": "sparse_bm25", "weight": 0.5, "metric": "IP",     "params": {}},
]

_KNOWN_FIELDS = {"embedding", "sparse_bm25", "sparse_fermi"}

# ──────────────────────────────── System prompt ───────────────────────────────

SYSTEM_PROMPT = """\
Vous êtes IAGO, un assistant intelligent travaillant pour l'ASNR (Autorité de sûreté nucléaire et de radioprotection).

### Méthode pour répondre à une question
- Commencer par citer les définitions qui vont permettre de comprendre la question.
- Dans votre réponse finale, n'indiquez que des éléments qui répondent directement à la question posée, pas de digression : ne mentionnez pas d'informations qui n'ont pas de rapport avec celle-ci.
- Justifiez chacune de vos affirmations avec des sources fiables (documents accessibles) en indiquant à chaque fin de phrase, d'affirmation ou de paragraphe la référence vers les sources utilisées pour produire cette affirmation.
- Soyez très vigilant à ne pas confondre les colis.
- Citez le document et la page où se trouve l'information.
- Indiquez les citations avec l'index de l'extrait sous la forme "[1]", "[3]", etc.
- A la fin de votre réponse, indiquez systématiquement la liste complète des extraits vous ayant été utiles pour répondre à la question (numéro d'extrait entre crochets, nom du document, page... - Exemple : [1] SSR-6, page 8).
- Utilisez seulement les informations présentes dans la documentation qui vous est fournie, et rien d'autre. En particulier, il vous est INTERDIT d'essayer de deviner, ou d'utiliser vos connaissances internes. N'affirmez rien qui ne s'appuie pas sur la documentation à votre disposition.
- Si vous ne trouvez pas la réponse dans la documentation, indiquez que vous ne savez pas, n'inventez pas.
- Répondez en français

### Expertise nucléaire
- Les types de colis correspondent à des modèles spécifiques d'emballages de transport utilisés pour le transport sécurisé des matières radioactives.
"""


# ──────────────────────────────── Proxy helpers ───────────────────────────────


def _resolve_proxy_models(
    proxy_prefix: str,
    path: str,
    preferred_provider: str | None = None,
) -> dict[str, dict]:
    """Query ``{proxy_prefix}/{path}`` and return ``{model_key: {name, endpoint}}``.

    When the same ``unique_model_name`` is served by several backends (e.g. both
    OVH and ALBERT), ``preferred_provider`` selects which entry wins.  Default is
    ``None`` (last entry in the list wins — ALBERT in practice, since the proxy
    lists OVH before ALBERT and last-write-wins).  ALBERT is preferred by default
    because latency benchmarks show it is ~40 % faster for Mistral models.
    Pass ``"ovh"`` to force OVH explicitly.
    """
    url = f"{proxy_prefix}/{path}"
    req = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except (URLError, HTTPError, TimeoutError) as exc:
        raise RuntimeError(f"Cannot reach proxy ({url}): {exc}") from exc

    models: dict[str, dict] = {}
    base = urlparse(proxy_prefix)

    for item in (data if isinstance(data, list) else []):
        # Embedding format (/emb/models)
        name     = item.get("model_name")
        base_url = item.get("base_url")
        if name and base_url:
            endpoint = base_url.rstrip("/") + "/v1"
            models[str(name)] = {"name": str(name), "endpoint": endpoint}
            continue

        # LLM format (/llm/models)
        key      = item.get("unique_model_name") or item.get("model_key") or item.get("id")
        name     = item.get("model_name") or item.get("name")
        ep       = item.get("llm_endpoint") or item.get("endpoint")
        provider = item.get("api_key_tech_name", "")
        if key and name and ep:
            parsed = urlparse(ep)
            if not parsed.scheme and not parsed.netloc:
                path_part = ep if ep.startswith("/") else f"/{ep}"
                ep = urlunparse((base.scheme, base.netloc, path_part, "", "", ""))
            else:
                ep = urlunparse((base.scheme, base.netloc, parsed.path, parsed.params, parsed.query, ""))
            entry = {"name": str(name), "endpoint": ep, "provider": provider}
            existing = models.get(str(key))
            if existing is None:
                models[str(key)] = entry
            elif preferred_provider and provider == preferred_provider:
                # Prefer the specified provider when the model appears on several backends.
                models[str(key)] = entry
            # else: keep existing (already preferred or no preference)

    return models


def _resolve_all_providers_for_key(
    proxy_prefix: str,
    model_key: str,
    preferred_order: tuple[str, ...] = ("albert", "ovh", "mistral-api"),
) -> list[dict]:
    """Returns ALL backend entries for ``model_key``, sorted by ``preferred_order``.

    Useful for building provider fallback chains (e.g. try ALBERT first for
    speed, fall back to OVH on daily quota exhaustion).  Returns an empty list
    if the model is not found.
    """
    url = f"{proxy_prefix}/llm/models"
    req = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except (URLError, HTTPError, TimeoutError) as exc:
        raise RuntimeError(f"Cannot reach proxy ({url}): {exc}") from exc

    base = urlparse(proxy_prefix)
    entries: list[dict] = []
    for item in (data if isinstance(data, list) else []):
        key      = item.get("unique_model_name") or item.get("model_key") or item.get("id")
        name     = item.get("model_name") or item.get("name")
        ep       = item.get("llm_endpoint") or item.get("endpoint")
        provider = item.get("api_key_tech_name", "")
        if not (key and name and ep) or str(key) != model_key:
            continue
        parsed = urlparse(ep)
        ep = urlunparse((base.scheme, base.netloc, parsed.path, "", "", ""))
        entries.append({"name": str(name), "endpoint": ep, "provider": provider})

    order = list(preferred_order)
    entries.sort(key=lambda e: order.index(e["provider"]) if e["provider"] in order else len(order))
    return entries


def _build_embedding_client(proxy_prefix: str, model_key: str) -> tuple[OpenAI, str]:
    """Return ``(OpenAI client, model_name)`` for dense embedding via ``/emb/models``."""
    models = _resolve_proxy_models(proxy_prefix, "emb/models")
    if model_key in models:
        m = models[model_key]
        return OpenAI(base_url=m["endpoint"], api_key="dummy"), m["name"]
    raise RuntimeError(
        f"Embedding model '{model_key}' not found in proxy. "
        f"Available: {sorted(models)}"
    )


def _build_llm_client(
    proxy_prefix: str,
    model_key: str,
    max_tokens: int,
    timeout: float,
    max_retries: int = 0,
) -> ChatOpenAI:
    """Return a ``ChatOpenAI`` instance pointing to the correct endpoint via ``/llm/models``.

    ``max_retries`` defaults to 0 (fail fast) for the interactive RAG pipeline.
    Callers doing bulk/sequential calls against a rate-limited model (e.g.
    evaluation) should pass a higher value: the OpenAI SDK retries 429s with
    exponential backoff.
    """
    models = _resolve_proxy_models(proxy_prefix, "llm/models")
    if model_key not in models:
        raise RuntimeError(
            f"LLM model '{model_key}' not found in proxy. "
            f"Available: {sorted(models)}"
        )
    m = models[model_key]
    return ChatOpenAI(
        base_url=m["endpoint"],
        model=m["name"],
        api_key="dummy",
        temperature=0.0,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
    )


# ──────────────────────────────── Fermi helpers ───────────────────────────────


def _rerank_remote(url: str, query: str, chunks: list[dict]) -> list[dict]:
    """Call reranker_server POST /score and return all chunks sorted by score descending."""
    pairs = [[query, chunk.get("text") or ""] for chunk in chunks]
    payload = json.dumps({"pairs": pairs, "batch_size": 32}).encode()
    req = Request(
        f"{url.rstrip('/')}/score",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode())
    for chunk, score in zip(chunks, result["scores"]):
        chunk["rerank_score"] = float(score)
    return sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)


def _encode_fermi_remote(url: str, text: str, max_seq_len: int) -> dict[int, float]:
    """Call a running fermi_server HTTP API and return ``{token_id: weight}``."""
    payload = json.dumps({"texts": [text], "max_seq_len": max_seq_len}).encode()
    req = Request(
        f"{url.rstrip('/')}/encode",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode())
    return {int(k): float(v) for k, v in result["sparse"][0].items()}


def _load_fermi_model(model_id: str):
    """Load the Fermi sparse encoder locally on CPU with optional int8 quantization.

    Returns ``(model, tokenizer, special_token_ids)``.
    """
    torch.set_num_threads(os.cpu_count() or 1)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id)
    model.eval()
    try:
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    except Exception:
        pass  # fall back to float32 if quantization is unavailable
    return model, tokenizer, tokenizer.all_special_ids


def _encode_fermi(
    text: str, model, tokenizer, special_token_ids, max_seq_len: int
) -> dict[int, float]:
    """Encode a single text with Fermi SPLADE recipe.

    Returns a sparse vector ``{token_id: weight}`` using max-pooling over the
    sequence followed by ``log(1 + relu(·))``, with special tokens zeroed out.
    """
    feature = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
        return_token_type_ids=False,
    )
    with torch.inference_mode():
        output = model(**feature)[0]

    values, _ = torch.max(output * feature["attention_mask"].unsqueeze(-1), dim=1)
    values = torch.log(1 + torch.relu(values))
    values[:, special_token_ids] = 0

    row = values[0]
    idx = torch.nonzero(row, as_tuple=False).squeeze(1)
    weights = row[idx]
    return {int(k): float(v) for k, v in zip(idx.tolist(), weights.tolist())}


def _encode_fermi_batch(
    texts: list[str],
    model,
    tokenizer,
    special_token_ids,
    max_seq_len: int,
    sub_batch_size: int = 16,
) -> list[dict[int, float]]:
    """Batch-encode a list of texts with Fermi using sub-batches to control memory.

    Much faster than calling ``_encode_fermi`` N times because the tokenizer and
    model forward pass are vectorised.  ``sub_batch_size`` limits peak memory use
    (queries are short, so 16 is conservative and safe on CPU).
    """
    all_results: list[dict[int, float]] = []
    for start in range(0, len(texts), sub_batch_size):
        chunk = texts[start : start + sub_batch_size]
        features = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        with torch.inference_mode():
            output = model(**features)[0]  # (batch, seq, vocab)
        values, _ = torch.max(output * features["attention_mask"].unsqueeze(-1), dim=1)
        values = torch.log(1 + torch.relu(values))
        values[:, special_token_ids] = 0
        for j in range(len(chunk)):
            row = values[j]
            idx = torch.nonzero(row, as_tuple=False).squeeze(1)
            weights = row[idx]
            all_results.append(
                {int(k): float(v) for k, v in zip(idx.tolist(), weights.tolist())}
            )
    return all_results


# ──────────────────────────────── BM25 helper ─────────────────────────────────


def _bm25_to_dict(sparse_matrix) -> dict[int, float]:
    """Convert a scipy sparse matrix row (1 × vocab_size) to ``{token_id: weight}``."""
    coo = sparse_matrix.tocoo()
    return {int(c): float(v) for c, v in zip(coo.col, coo.data)}


# ──────────────────────────────── Context format ──────────────────────────────


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("name") or chunk.get("path") or "unknown source"
        text   = (chunk.get("text") or "").strip()
        parts.append(f"[{i}] {text}\n    Source : {source}")
    return "\n\n".join(parts)


# ──────────────────────────────── Pipeline ────────────────────────────────────


class HybridRAGPipeline:
    """Hybrid RAG pipeline backed by Milvus multi-vector search.

    The pipeline embeds a query with up to three encoders (dense BGE-M3, sparse BM25,
    sparse Fermi), runs ``hybrid_search`` on a single Milvus collection, fuses the
    per-field ranking lists with ``WeightedRanker`` or ``RRFRanker``, reranks the
    candidates with a cross-encoder, and feeds the top-N chunks to an LLM.

    Only the encoders referenced in *search_config* are loaded; e.g. passing a config
    with only ``"embedding"`` and ``"sparse_bm25"`` skips loading the Fermi model.

    Parameters
    ----------
    milvus_uri:
        URI of the Milvus server (e.g. ``"http://milvus:19530"``).
    collection_name:
        Target collection.  Must contain the vector fields listed in *search_config*.
    search_config:
        List of search descriptors, one per vector index.  Each entry is a ``dict``
        with the following keys:

        ``field`` (str)
            Milvus vector field name.  One of ``"embedding"``, ``"sparse_bm25"``,
            ``"sparse_fermi"``.
        ``weight`` (float)
            Score weight for ``WeightedRanker``.  Ignored when *ranker* is ``"rrf"``.
        ``metric`` (str)
            Distance metric matching the index: ``"COSINE"`` for dense,
            ``"IP"`` for sparse.
        ``params`` (dict)
            Extra search parameters, e.g. ``{"nprobe": 16}`` for IVF_FLAT.

        Defaults to :data:`DEFAULT_SEARCH_CONFIG` (dense + BM25, equal weights;
        Fermi is opt-in, not included by default).
    ranker:
        Fusion strategy.  ``"weighted"`` (default) uses
        :class:`~pymilvus.WeightedRanker` with the per-entry weights.
        ``"rrf"`` uses :class:`~pymilvus.RRFRanker`.
    rrf_k:
        Smoothing constant for ``RRFRanker`` (default ``60``).
        Only used when *ranker* is ``"rrf"``.
    bm25_state_path:
        Path to the persisted BM25 state JSON file (``idf + avgdl`` produced by
        ``build_fragus_collection.py``).  Defaults to
        ``artifacts/bm25/{collection_name}.json`` relative to the project root.
        Required when ``"sparse_bm25"`` is in *search_config*.
    fermi_model_id:
        HuggingFace identifier for the Fermi sparse encoder.
    fermi_max_seq_len:
        Maximum token length for Fermi encoding.
    proxy_prefix:
        Base URL of the AI-gen proxy used to discover embedding and LLM endpoints.
    embedding_model:
        Model key for BGE-M3 dense embedding (resolved via ``/emb/models``).
    reranker_model:
        HuggingFace cross-encoder model identifier.
    reranker_device:
        PyTorch device for the cross-encoder (``"cpu"``, ``"cuda"``, …).
    candidates:
        Number of candidates retrieved from Milvus before reranking.
    top_n:
        Number of chunks kept after reranking and passed to the LLM.
    llm_model_key:
        Model key for the generation LLM (resolved via ``/llm/models``).
    llm_max_tokens:
        Maximum tokens for the LLM completion.
    llm_timeout:
        Timeout in seconds for LLM requests.
    system_prompt:
        System message injected into every LLM call.
    output_fields:
        Milvus scalar fields to retrieve alongside each hit.
    """

    def __init__(
        self,
        milvus_uri: str                       = MILVUS_URI,
        collection_name: str                  = COLLECTION_NAME,
        search_config: Optional[list[dict]]   = None,
        ranker: str                           = "weighted",
        rrf_k: int                            = 60,
        bm25_state_path: Optional[str | Path] = None,
        fermi_model_id: str                   = FERMI_MODEL_ID,
        fermi_max_seq_len: int                = FERMI_MAX_SEQ_LEN,
        fermi_encoder_url: Optional[str]      = FERMI_ENCODER_URL,
        proxy_prefix: str                     = _PROXY_PREFIX,
        embedding_model: str                  = EMBEDDING_MODEL,
        reranker_model: str                   = RERANKER_MODEL,
        reranker_device: str                  = RERANKER_DEVICE,
        reranker_url: Optional[str]           = RERANKER_URL,
        reranker_threshold: Optional[float]   = None,
        candidates: int                       = CANDIDATES,
        top_n: int                            = TOP_N,
        ranking_mode: str                     = "reranker",
        llm_model_key: str                    = LLM_MODEL_KEY,
        llm_max_tokens: int                   = LLM_MAX_TOKENS,
        llm_timeout: float                    = LLM_TIMEOUT,
        system_prompt: str                    = SYSTEM_PROMPT,
        output_fields: Optional[list[str]]    = None,
    ):
        self.collection_name    = collection_name
        self.search_config      = search_config or [dict(c) for c in DEFAULT_SEARCH_CONFIG]
        self.candidates            = candidates
        self.top_n                 = top_n
        self.reranker_threshold    = reranker_threshold
        self.system_prompt      = system_prompt
        self.output_fields      = output_fields or list(OUTPUT_FIELDS)
        self._ranker            = ranker
        self._rrf_k             = rrf_k
        self._ranking_mode      = ranking_mode
        self._fermi_max_seq_len = fermi_max_seq_len
        self._fermi_encoder_url = fermi_encoder_url

        # ── Validate inputs ──────────────────────────────────────────────────
        for entry in self.search_config:
            if entry.get("field") not in _KNOWN_FIELDS:
                raise ValueError(
                    f"Unknown field {entry.get('field')!r} in search_config. "
                    f"Supported: {sorted(_KNOWN_FIELDS)}"
                )
        if ranker not in ("weighted", "rrf"):
            raise ValueError(f"ranker must be 'weighted' or 'rrf', got {ranker!r}")
        if ranking_mode not in ("reranker", "combined"):
            raise ValueError(f"ranking_mode must be 'reranker' or 'combined', got {ranking_mode!r}")

        active_fields = {e["field"] for e in self.search_config}

        # ── Milvus ───────────────────────────────────────────────────────────
        _log.info("Connecting to Milvus…")
        self.client = MilvusClient(uri=milvus_uri)
        _log.info("  → Milvus OK")

        # ── Dense BGE-M3 (proxy) ─────────────────────────────────────────────
        if "embedding" in active_fields:
            _log.info("Resolving embedding model…")
            self._oai_emb, self._emb_model = _build_embedding_client(
                proxy_prefix, embedding_model
            )
            _log.info("  → Embedding OK (%s)", self._emb_model)
        else:
            self._oai_emb  = None
            self._emb_model = None

        # ── BM25 (client-side, persisted state) ──────────────────────────────
        if "sparse_bm25" in active_fields:
            state_path = (
                Path(bm25_state_path)
                if bm25_state_path is not None
                else _BM25_STATE_DIR / f"{collection_name}.json"
            )
            if not state_path.exists():
                raise FileNotFoundError(
                    f"BM25 state file not found: {state_path}\n"
                    "Generate it with build_fragus_collection.py, or pass bm25_state_path."
                )
            _log.info("Loading BM25 state from %s…", state_path)
            self._bm25 = BM25EmbeddingFunction(
                analyzer=build_default_analyzer(language=BM25_LANGUAGE)
            )
            self._bm25.load(str(state_path))
            _log.info("  → BM25 OK")
        else:
            self._bm25 = None

        # ── Fermi sparse encoder ─────────────────────────────────────────────
        if "sparse_fermi" in active_fields:
            if fermi_encoder_url:
                _log.info("Using remote Fermi server at %s", fermi_encoder_url)
                self._fermi_model       = None
                self._fermi_tok         = None
                self._fermi_special_ids = None
                _log.info("  → Fermi (remote) OK")
            else:
                _log.info("Loading Fermi model locally (%s)…", fermi_model_id)
                self._fermi_model, self._fermi_tok, self._fermi_special_ids = (
                    _load_fermi_model(fermi_model_id)
                )
                _log.info("  → Fermi (local) OK")
        else:
            self._fermi_model       = None
            self._fermi_tok         = None
            self._fermi_special_ids = None

        # ── Cross-encoder reranker ────────────────────────────────────────────
        self._reranker_url = reranker_url or None
        if self._reranker_url:
            _log.info("Using remote reranker at %s (skipping local load)", self._reranker_url)
            self.reranker = None
            _log.info("  → Reranker (remote) OK")
        else:
            _log.info("Loading reranker (%s)…", reranker_model)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.reranker = CrossEncoder(
                    model_name_or_path=reranker_model,
                    device=reranker_device,
                )
            _log.info("  → Reranker OK")

        # ── LLM (proxy) ───────────────────────────────────────────────────────
        _log.info("Resolving LLM model…")
        self.llm = _build_llm_client(
            proxy_prefix, llm_model_key, llm_max_tokens, llm_timeout
        )
        _log.info("  → LLM OK")

    # ── Embedding helpers ─────────────────────────────────────────────────────

    def _embed_dense(self, query: str) -> list[float]:
        resp = self._oai_emb.embeddings.create(
            model=self._emb_model,
            input=[query],
            encoding_format="float",
        )
        return resp.data[0].embedding

    def _embed_bm25(self, query: str) -> dict[int, float]:
        # encode_queries returns a scipy csr_array of shape (1, vocab_size);
        # Milvus expects {token_id: weight} for sparse fields.
        result = self._bm25.encode_queries([query])
        return _bm25_to_dict(result[0])

    def _embed_fermi(self, query: str) -> dict[int, float]:
        if self._fermi_encoder_url:
            return _encode_fermi_remote(
                self._fermi_encoder_url, query, self._fermi_max_seq_len
            )
        return _encode_fermi(
            query,
            self._fermi_model,
            self._fermi_tok,
            self._fermi_special_ids,
            self._fermi_max_seq_len,
        )

    def _embed_for_field(self, query: str, field: str):
        """Dispatch query encoding to the correct embedder for *field*."""
        if field == "embedding":
            return self._embed_dense(query)
        if field == "sparse_bm25":
            return self._embed_bm25(query)
        if field == "sparse_fermi":
            return self._embed_fermi(query)
        raise ValueError(f"Unknown field: {field!r}")

    # ── Reranking ─────────────────────────────────────────────────────────────

    def _rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        if not chunks:
            return []

        if self._reranker_url:
            scored = _rerank_remote(self._reranker_url, query, chunks)
        else:
            scores = self.reranker.predict(
                [(query, chunk.get("text") or "") for chunk in chunks],
                batch_size=32,
                show_progress_bar=False,
                max_length=512,
            )
            for chunk, score in zip(chunks, scores):
                chunk["rerank_score"] = float(score)
            scored = chunks

        if self._ranking_mode == "combined":
            milvus_vals = [c.get("distance") or 0.0 for c in scored]
            rerank_vals = [c["rerank_score"] for c in scored]

            def _minmax(vals):
                lo, hi = min(vals), max(vals)
                if hi == lo:
                    return [0.5] * len(vals)
                return [(v - lo) / (hi - lo) for v in vals]

            norm_m = _minmax(milvus_vals)
            norm_r = _minmax(rerank_vals)
            for chunk, nm, nr in zip(scored, norm_m, norm_r):
                chunk["combined_score"] = 0.5 * nm + 0.5 * nr
            ranked = sorted(scored, key=lambda x: x["combined_score"], reverse=True)[: self.top_n]
            if self.reranker_threshold is not None:
                ranked = [c for c in ranked if c["rerank_score"] >= self.reranker_threshold]
            return ranked

        ranked = sorted(scored, key=lambda x: x["rerank_score"], reverse=True)[: self.top_n]
        if self.reranker_threshold is not None:
            ranked = [c for c in ranked if c["rerank_score"] >= self.reranker_threshold]
        return ranked

    # ── Public API ────────────────────────────────────────────────────────────

    def batch_encode_queries(self, texts: list[str]) -> dict[str, dict[str, Any]]:
        """Pre-compute all query vectors for a list of texts.

        Returns ``{text: {field_name: vector}}`` for every active field.
        Pass the per-text sub-dict to :meth:`retrieve` / :meth:`ask` via the
        ``precomputed`` parameter to skip per-call encoding (useful when the
        same texts are encoded in a tight parallel loop — avoids CPU contention
        from Fermi inference running N workers simultaneously).
        """
        active_fields = {e["field"] for e in self.search_config}
        result: dict[str, dict[str, Any]] = {t: {} for t in texts}

        if "embedding" in active_fields and self._oai_emb is not None:
            _log.info("Batch-encoding %d texts with BGE-M3…", len(texts))
            batch_size = 32
            all_embeddings: list[list[float]] = []
            for i in range(0, len(texts), batch_size):
                resp = self._oai_emb.embeddings.create(
                    model=self._emb_model,
                    input=texts[i : i + batch_size],
                    encoding_format="float",
                )
                all_embeddings.extend(e.embedding for e in resp.data)
            for text, emb in zip(texts, all_embeddings):
                result[text]["embedding"] = emb
            _log.info("  → BGE-M3 batch done")

        if "sparse_bm25" in active_fields and self._bm25 is not None:
            _log.info("Batch-encoding %d texts with BM25…", len(texts))
            bm25_matrix = self._bm25.encode_queries(texts)
            for text, sparse_row in zip(texts, bm25_matrix):
                result[text]["sparse_bm25"] = _bm25_to_dict(sparse_row)
            _log.info("  → BM25 batch done")

        if "sparse_fermi" in active_fields:
            if self._fermi_encoder_url:
                _log.info("Batch-encoding %d texts with Fermi (remote)…", len(texts))
                for text in texts:
                    result[text]["sparse_fermi"] = _encode_fermi_remote(
                        self._fermi_encoder_url, text, self._fermi_max_seq_len
                    )
            else:
                _log.info("Batch-encoding %d texts with Fermi (local)…", len(texts))
                fermi_vecs = _encode_fermi_batch(
                    texts,
                    self._fermi_model,
                    self._fermi_tok,
                    self._fermi_special_ids,
                    self._fermi_max_seq_len,
                )
                for text, vec in zip(texts, fermi_vecs):
                    result[text]["sparse_fermi"] = vec
            _log.info("  → Fermi batch done")

        return result

    def retrieve(self, query: str, precomputed: Optional[dict[str, Any]] = None) -> list[dict]:
        """Run hybrid search and return the top-N reranked chunks.

        Parameters
        ----------
        query:
            Natural-language question or keyword string.
        precomputed:
            Optional ``{field_name: vector}`` dict produced by
            :meth:`batch_encode_queries`.  When a field's vector is present,
            it is used directly and the corresponding encoder is skipped.

        Returns
        -------
        list[dict]
            Each dict contains the configured output fields (``chunk_id``,
            ``file_id``, ``name``, ``path``, ``text``), a ``distance`` score
            from Milvus, and a ``rerank_score`` from the cross-encoder.
            Sorted by ``rerank_score`` descending.
        """
        # Build one AnnSearchRequest per configured vector field
        reqs: list[AnnSearchRequest] = []
        for cfg in self.search_config:
            field = cfg["field"]
            if precomputed is not None and field in precomputed:
                vec = precomputed[field]
            else:
                vec = self._embed_for_field(query, field)
            reqs.append(
                AnnSearchRequest(
                    data=[vec],
                    anns_field=field,
                    param={
                        "metric_type": cfg["metric"],
                        "params": cfg.get("params", {}),
                    },
                    limit=self.candidates,
                )
            )

        # Build fusion ranker
        if self._ranker == "weighted":
            weights = [cfg["weight"] for cfg in self.search_config]
            fusion_ranker = WeightedRanker(*weights)
        else:
            fusion_ranker = RRFRanker(k=self._rrf_k)

        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=reqs,
            ranker=fusion_ranker,
            limit=self.candidates,
            output_fields=self.output_fields,
            timeout=60,
        )

        chunks = [
            {"distance": hit.get("distance"), **{f: hit.get(f) for f in self.output_fields}}
            for hit in results[0]
        ]
        reranked = self._rerank(query, chunks)
        _log.debug("retrieve: %d candidates → %d after reranking", len(chunks), len(reranked))
        return reranked

    def ask(self, query: str, precomputed: Optional[dict[str, Any]] = None) -> dict:
        """Run the full RAG pipeline and return the generated answer with its sources.

        Parameters
        ----------
        query:
            Natural-language question.

        Returns
        -------
        dict
            A dict with three keys:

            ``query`` (str)
                The original question.
            ``answer`` (str)
                The LLM-generated answer.
            ``chunks`` (list[dict])
                The reranked context windows passed to the LLM, each with
                ``rerank_score`` and the output fields.
        """
        chunks = self.retrieve(query, precomputed=precomputed)
        return self.ask_with_chunks(query, chunks)

    def ask_with_chunks(self, query: str, chunks: list[dict]) -> dict:
        """Run ONLY the generation stage on pre-retrieved chunks.

        Same prompt construction as :meth:`ask` — used to re-generate answers
        from stored contexts (e.g. reranker-threshold variants derived from a
        previous run's parquet) without re-running retrieval.
        """
        context = _format_context(chunks)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    "Voici les extraits pertinents récupérés depuis la base documentaire :\n\n"
                    f"{context}\n\n"
                    f"Question : {query}"
                ),
            },
        ]

        answer = self.llm.invoke(messages).content
        _log.debug("ask: response length %d chars, %d chunks used", len(answer), len(chunks))
        return {"query": query, "answer": answer, "chunks": chunks}
