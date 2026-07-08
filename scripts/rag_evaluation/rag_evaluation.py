"""
rag_evaluation.py
=================

Evaluation module for RAG / GraphRAG systems.

Designed to be imported from a Jupyter notebook. It computes, per question, a
set of scores organized into three families (cf. evaluation framework), and
aggregates a global score following a structure inspired by the Hub France IA
guide ("Evaluation des Chaines de RAG", 09/2025), extended with a 4th
faithfulness component (hallucination control).

--------------------------------------------------------------------------
COMPUTED INDICATORS
--------------------------------------------------------------------------

1. Response Quality
    - factual_correctness_{precision,recall,f1}
          Recomputed internally (NOT via the native RAGAS mode) because the
          current version of RAGAS computes recall incorrectly (issue #2693,
          PR #2694 unmerged as of 2026-05-25). See compute_factual_correctness.
          The score retained in the global is the RECALL (cf. framework:
          priority on recalling the expected facts; correct additional
          information is not penalized).
    - faithfulness               (RAGAS) -> anti-hallucination safeguard
    - paraphrase_robustness      (custom, embeddings) -> robustness to question
          reformulation (state of the art: group similarity / Con-RAG 2025).
          This is NOT self-consistency.
    - anchoring / hallucination  (custom LLM, 4 fact categories)

2. Retrieved Context Quality
    - context_recall_llm         (RAGAS, LLMContextRecall)
    - context_precision_llm      (RAGAS, LLMContextPrecisionWithReference,
                                  rank-aware / average precision)
    - hub_france_ia_fscore       (custom, F-beta(precision_llm, recall_llm),
                                  beta=0.5 -> penalizes excessive context;
                                  diagnostic "sufficient without being excessive")
    - context_relevance_files    (custom, retrieved & expected / retrieved — precision)
    - context_coverage_files     (custom, retrieved & expected / expected — recall)
    - diversity                  (custom, 1 - mean cosine similarity of chunks)

3. Citations (cited files vs reference)
    - citation_precision_files   (cited & expected / cited)
    - citation_recall_files      (cited & expected / expected)
    - citation_fscore_files      (F-beta, beta=1 by default)

--------------------------------------------------------------------------
GLOBAL SCORE (4 components, "recall + faithfulness safeguard" logic)
--------------------------------------------------------------------------
    Global = W_R * context_recall_llm            (foundation: info retrieved)
           + W_G * factual_correctness_recall    (expected facts present)
           + W_C * citation_recall_files         (expected sources cited)
           + W_F * faithfulness                  (anti-hallucination safeguard)
    Default weights: W_R=0.25, W_G=0.35, W_C=0.15, W_F=0.25
    Automatic renormalization if a component is missing.

The first 3 components measure completeness (recall); the 4th measures
reliability (no invention). Recall is favored everywhere because "extra
information" is tolerable (context filtered by the LLM, correct elaboration,
harmless surnumerary citations); only "missing" information is penalized.
Faithfulness provides the anti-hallucination counterweight.

--------------------------------------------------------------------------
BACKENDS
--------------------------------------------------------------------------
Both the judge LLM (RAGAS) and the embeddings (paraphrase_robustness,
diversity) are resolved through the Cleyrop AI-gen proxy (``/llm/models``
and ``/emb/models``), exactly like ``HybridRAGPipeline``. There is no direct
OVH/Mistral API access and no separate API key to configure: any model
exposed by the proxy can be selected via its ``model_key`` (e.g.
``"gpt-oss-120b"``, ``"mistral-large-latest"``), Mistral-branded or not.

--------------------------------------------------------------------------
INPUT FILE
--------------------------------------------------------------------------
An .xlsx whose columns follow the RAGAS field names (see COLMAP). The module
produces two files:
    <prefix>_per_question.xlsx : one score per question
    <prefix>_averages.xlsx     : mean of each score

References:
    - RAGAS  : https://docs.ragas.io/
    - OVH    : https://www.ovhcloud.com/en/public-cloud/ai-endpoints/catalog/
    - Mistral: https://docs.mistral.ai/
    - Hub France IA, "Evaluation des Chaines de RAG", 09/2025.
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.rag.hybrid_rag_pipeline import (  # noqa: E402
    _PROXY_PREFIX,
    _build_llm_client,
    _resolve_proxy_models,
    _resolve_all_providers_for_key,
)

# ---------------------------------------------------------------------------
# Guarded RAGAS / LangChain imports: the module stays importable even if the
# stack is not installed (one can then use only the non-LLM custom metrics:
# citations, diversity, paraphrase_robustness).
# ---------------------------------------------------------------------------
try:
    with warnings.catch_warnings():
        # ragas 0.4 deprecates these import paths in favour of
        # ragas.metrics.collections + llm_factory (removal planned for v1.0).
        # Migrating now would mean rewriting compute_factual_correctness
        # (the #2693 bug workaround relies on decompose_claims/verify_claims,
        # which the new collections.FactualCorrectness no longer exposes) and
        # re-validating every metric against live LLM calls. Not worth the
        # risk while the old API still works; revisit when ragas 1.0 ships.
        warnings.simplefilter("ignore", DeprecationWarning)
        import ragas as _ragas_pkg
        from ragas import SingleTurnSample
        from ragas.metrics import (
            Faithfulness,
            LLMContextRecall,
            LLMContextPrecisionWithReference,
        )
        from ragas.metrics._factual_correctness import FactualCorrectness
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
    _RAGAS_AVAILABLE = True
    _RAGAS_VERSION = getattr(_ragas_pkg, "__version__", "unknown")
except Exception as _e:  # pragma: no cover
    _RAGAS_AVAILABLE = False
    _RAGAS_IMPORT_ERROR = _e
    _RAGAS_VERSION = None


# ===========================================================================
# 1. CONFIGURATION
# ===========================================================================

# --- Expected column names in the xlsx ------------------------------------
# Aligned with the native RAGAS fields (SingleTurnSample):
#   user_input          : the question
#   response            : the answer generated by the RAG
#   reference           : the expected answer (ground truth)
#   retrieved_contexts  : context snippets used (list)
# Additional fields (outside RAGAS):
#   reference_files     : file names expected to answer (list)
#   cited_files         : file names cited by the RAG (list)
#   paraphrase_group_id : shared identifier for reformulations of the same
#                         question (for paraphrase_robustness). Optional.
COLMAP: dict[str, str] = {
    "user_input": "user_input",
    "response": "response",
    "reference": "reference",
    "retrieved_contexts": "retrieved_contexts",
    "reference_files": "reference_files",
    "cited_files": "cited_files",
    "retrieved_files": "retrieved_files",
    "paraphrase_group_id": "paraphrase_group_id",
}

LIST_COLUMNS = ("retrieved_contexts", "reference_files", "cited_files", "retrieved_files")


@dataclass
class LLMConfig:
    """Judge LLM used by RAGAS and the custom LLM metrics, resolved via the
    Cleyrop AI-gen proxy (``/llm/models``)."""
    model_key: str = "mistral-small-3.2-24b-instruct-2506"
    proxy_prefix: str = _PROXY_PREFIX
    # RAGAS metrics (e.g. Faithfulness) generate verbose structured
    # intermediate output (claim lists, verdicts); 2048 truncates them
    # mid-JSON and the metric fails to parse the result.
    max_tokens: int = 8192
    # 300 s: anchoring prompts on large contexts (top-100 → ~45k tokens) can
    # exceed 120 s on the fallback backend without being stuck.
    timeout: float = 300.0
    # SDK-level retries are disabled (max_retries=0): the proxy returns two
    # distinct 429 flavours that need different handling —
    #   "N requests per minute exceeded" : recoverable → sleep 65 s, retry × 3
    #   "N requests per day exceeded"    : non-recoverable → NaN immediately
    # Timeouts are also non-recoverable (retrying a timed-out 120 B model call
    # several times wastes minutes per question for no gain). See _safe_ascore.
    max_retries: int = 0


@dataclass
class EmbeddingConfig:
    """Embeddings (paraphrase_robustness, diversity), resolved via the
    Cleyrop AI-gen proxy (``/emb/models``)."""
    model_key: str = "bge-multilingual-gemma2"  # OVH-recommended, most performant
    proxy_prefix: str = _PROXY_PREFIX


class _EmbeddingCache:
    """Disk-backed cache for embedding vectors (diversity, paraphrase_robustness).

    Stores {text → vector} in a pickle file so repeated evaluation runs skip
    already-computed embeddings. The cache path should include the model key
    to avoid cross-model pollution.
    """

    def __init__(self, embedder, cache_path: "str | Path") -> None:
        self._embedder = embedder
        self._path = Path(cache_path)
        self._store: dict[str, list] = {}
        if self._path.exists():
            try:
                import pickle as _pickle
                with open(self._path, "rb") as fh:
                    self._store = _pickle.load(fh)
                print(f"[embedding_cache] {len(self._store)} cached vectors loaded from {self._path}")
            except Exception as e:
                warnings.warn(f"[embedding_cache] Failed to load cache ({e}) — starting fresh.")
                self._store = {}

    def _save(self) -> None:
        import pickle as _pickle
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "wb") as fh:
            _pickle.dump(self._store, fh)

    def embed_documents(self, texts: list) -> list:
        missing = [t for t in texts if t not in self._store]
        if missing:
            new_vecs = self._embedder.embed_documents(missing)
            for t, v in zip(missing, new_vecs):
                self._store[t] = v
            self._save()
        return [self._store[t] for t in texts]


@dataclass
class EvalConfig:
    """Global configuration of an evaluation campaign."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)

    # --- Hub France IA F-Score (diagnostic "sufficient without being excessive") -
    # beta < 1 favors precision (penalizes excessive/costly context).
    hub_fscore_beta: float = 0.5

    # --- Beta of the citation F-score (diagnostic) ------------------------
    citation_fscore_beta: float = 1.0

    # --- Global Score weights (4 components) ------------------------------
    # Automatically renormalized if a component is missing.
    global_weights: dict[str, float] = field(default_factory=lambda: {
        "context_recall_llm": 0.25,            # Search (foundation)
        "factual_correctness_recall": 0.35,    # Generation (objective)
        "citation_recall_files": 0.15,         # Citation (traceability)
        "faithfulness": 0.25,                  # Faithfulness (anti-hallucination safeguard)
    })

    # Normalize file names (case, path) before comparison.
    normalize_filenames: bool = True

    # Atomicity of the claim decomposition (factual_correctness).
    factual_atomicity: str = "low"             # "low" | "high"

    # Disk cache for embedding vectors (diversity + paraphrase_robustness).
    # None = no cache; set to a .pkl path to persist vectors across runs.
    embedding_cache_path: "str | None" = None


# ===========================================================================
# 2. BACKENDS (LLM + EMBEDDINGS) — both via the Cleyrop AI-gen proxy
# ===========================================================================


def _check_ragas_version() -> None:
    """
    Notifies that factual_correctness is recomputed internally (workaround for
    bug #2693: the native RAGAS 'recall' mode returns a precision score).
    """
    if _RAGAS_VERSION and _RAGAS_VERSION != "unknown":
        warnings.warn(
            f"RAGAS {_RAGAS_VERSION} detected. NB: factual_correctness is "
            "recomputed internally (the native RAGAS 'recall' mode is buggy, "
            "issue #2693). The other RAGAS metrics are used as-is."
        )


class _FallbackChatModel:
    """Wraps multiple ChatOpenAI clients in priority order (ALBERT first, OVH
    fallback).  On daily quota exhaustion, switches permanently to the next
    provider.  Exposes ``invoke`` / ``ainvoke`` so that ``LangchainLLMWrapper``
    and direct ``_llm_invoke_text`` calls both work transparently.
    """

    def __init__(self, clients: list, labels: list[str] | None = None) -> None:
        self._clients = clients
        self._labels = labels or [f"backend-{i}" for i in range(len(clients))]
        self._idx = 0
        # Expose the LangChain interface attribute used by LangchainLLMWrapper.
        self.langchain_llm = self   # _llm_invoke_text does getattr(wrapper, "langchain_llm")

    # ---- LangChain BaseChatModel minimal interface -------------------------

    @property
    def _llm_type(self) -> str:
        return "provider-fallback"

    def _active(self):
        return self._clients[self._idx]

    def _should_retry(self, e: Exception, used_idx: int) -> bool:
        """Decide whether a failed call may be retried on the current provider.

        Handles the concurrent-switch race: if another in-flight call already
        advanced ``_idx`` past ``used_idx``, retry on the new active provider
        instead of giving up.
        """
        if self._idx != used_idx:
            return True  # someone else already switched — retry on new provider
        if _is_daily_quota_error(e) and self._idx < len(self._clients) - 1:
            old = self._labels[self._idx]
            self._idx += 1
            warnings.warn(
                f"Daily quota exhausted on [{old}] → switching to [{self._labels[self._idx]}]."
            )
            return True
        return False

    def _call(self, method: str, *args, **kwargs):
        for _ in range(len(self._clients) + 1):
            used = self._idx
            try:
                return getattr(self._clients[used], method)(*args, **kwargs)
            except Exception as e:
                if not self._should_retry(e, used):
                    raise
        raise RuntimeError("All LLM providers exhausted.")

    async def _acall(self, method: str, *args, **kwargs):
        for _ in range(len(self._clients) + 1):
            used = self._idx
            try:
                return await getattr(self._clients[used], method)(*args, **kwargs)
            except Exception as e:
                if not self._should_retry(e, used):
                    raise
        raise RuntimeError("All LLM providers exhausted.")

    def invoke(self, prompt):
        return self._call("invoke", prompt)

    async def ainvoke(self, prompt):
        return await self._acall("ainvoke", prompt)

    # ---- LangChain generation interface (used by LangchainLLMWrapper) ------

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._call("_generate", messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return await self._acall("_agenerate", messages, stop=stop, run_manager=run_manager, **kwargs)

    def generate_prompt(self, *args, **kwargs):
        return self._call("generate_prompt", *args, **kwargs)

    async def agenerate_prompt(self, *args, **kwargs):
        return await self._acall("agenerate_prompt", *args, **kwargs)


def build_llm(cfg: LLMConfig, verbose: bool = True):
    """Builds the judge LLM wrapped for RAGAS.

    When the configured model is available on several backends (e.g. both ALBERT
    and OVH), builds a :class:`_FallbackChatModel` that tries ALBERT first
    (faster) and switches permanently to OVH the first time a daily quota error
    is returned.  Falls back to a single-provider client when only one backend
    is available.
    """
    if not _RAGAS_AVAILABLE:
        raise ImportError(
            f"RAGAS / LangChain unavailable: {_RAGAS_IMPORT_ERROR}. "
            "Install: pip install ragas langchain-openai"
        )
    from langchain_openai import ChatOpenAI

    entries = _resolve_all_providers_for_key(cfg.proxy_prefix, cfg.model_key)

    if len(entries) > 1:
        clients, labels = [], []
        for e in entries:
            clients.append(ChatOpenAI(
                base_url=e["endpoint"], model=e["name"], api_key="dummy",
                max_tokens=cfg.max_tokens, timeout=cfg.timeout, max_retries=cfg.max_retries,
            ))
            labels.append(e["provider"])
        if verbose:
            print(f"[LLM] {cfg.model_key}: fallback chain {' → '.join(labels)}")
        llm = _FallbackChatModel(clients=clients, labels=labels)
    else:
        llm = _build_llm_client(
            cfg.proxy_prefix, cfg.model_key, cfg.max_tokens, cfg.timeout, cfg.max_retries
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return LangchainLLMWrapper(llm)


def build_embeddings(cfg: EmbeddingConfig):
    """
    Builds the embeddings object via the Cleyrop AI-gen proxy (``/emb/models``).
    Returns (raw_embeddings, ragas_wrapper).
    `raw_embeddings` exposes .embed_documents(list[str]) (LangChain interface).
    """
    from langchain_openai import OpenAIEmbeddings

    models = _resolve_proxy_models(cfg.proxy_prefix, "emb/models")
    if cfg.model_key not in models:
        raise RuntimeError(
            f"Embedding model '{cfg.model_key}' not found in proxy. "
            f"Available: {sorted(models)}"
        )
    m = models[cfg.model_key]
    raw = OpenAIEmbeddings(
        model=m["name"],
        base_url=m["endpoint"],
        api_key="dummy",
        check_embedding_ctx_length=False,   # non-OpenAI endpoints
        chunk_size=25,                      # OVH proxy hard limit per batch
    )

    wrapper = None
    if _RAGAS_AVAILABLE:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                wrapper = LangchainEmbeddingsWrapper(raw)
        except Exception:
            wrapper = None
    return raw, wrapper


# ===========================================================================
# 3. DATA LOADING AND PREPARATION
# ===========================================================================

def _parse_cell_to_list(value: Any) -> list[str]:
    """
    Converts a cell into a list of strings. Handles: actual list / JSON /
    simple separators (; | newline) / empty cell.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    s = str(value).strip()
    if not s:
        return []
    for parser in (ast.literal_eval, json.loads):
        try:
            parsed = parser(s)
            if isinstance(parsed, (list, tuple)):
                return [str(v).strip() for v in parsed if str(v).strip()]
        except Exception:
            pass
    return [p.strip() for p in re.split(r"[;\n|]+", s) if p.strip()]


def load_dataset(
    path: str,
    colmap: dict[str, str] | None = None,
    sheet_name: str | int = 0,
) -> pd.DataFrame:
    """
    Loads the xlsx and normalizes the columns to the canonical RAGAS names.
    List columns are parsed into list[str].
    """
    colmap = colmap or COLMAP
    df = pd.read_excel(path, sheet_name=sheet_name)

    rename = {src: canon for canon, src in colmap.items() if src in df.columns}
    df = df.rename(columns=rename)

    # Full contexts sidecar: the Excel cell caps at 32 767 chars (long contexts get
    # truncated there); prepare_eval_dataset*.py writes the complete version to a
    # parquet next to the Excel. Prefer it whenever present.
    _p = Path(str(path))
    pq_path = _p.with_name(_p.stem + "_contexts.parquet")
    if pq_path.exists() and "q_id" in df.columns:
        full = pd.read_parquet(pq_path)
        mapping = dict(zip(full["q_id"].astype(str), full["retrieved_contexts"]))
        mapped = df["q_id"].astype(str).map(mapping)
        mask = mapped.notna()
        df.loc[mask, "retrieved_contexts"] = mapped[mask]
        print(f"[load] Full contexts from {pq_path.name} ({int(mask.sum())}/{len(df)} rows)")

    required = ("user_input", "response", "reference", "retrieved_contexts")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns: {missing}. "
            f"Available: {list(df.columns)}. Check COLMAP."
        )

    for col in LIST_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(_parse_cell_to_list)

    for col in ("user_input", "response", "reference"):
        df[col] = df[col].fillna("").astype(str)

    return df.reset_index(drop=True)


# ===========================================================================
# 4. UTILITIES
# ===========================================================================

def _normalize_filename(name: str, enabled: bool = True) -> str:
    if not enabled:
        return str(name).strip()
    n = os.path.basename(str(name).strip()).lower()
    return re.sub(r"\s+", " ", n)


def _normalize_set(names: Iterable[str], enabled: bool = True) -> set[str]:
    return {_normalize_filename(n, enabled) for n in names if str(n).strip()}


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _fbeta(precision: float, recall: float, beta: float) -> float:
    """F-beta = (1+b^2)*P*R / (b^2*P + R). beta<1 favors precision."""
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta * beta
    denom = (b2 * precision) + recall
    return 0.0 if denom == 0 else (1 + b2) * precision * recall / denom


def _is_nan(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _is_daily_quota_error(e: Exception) -> bool:
    return "per day" in str(e).lower()


def _is_per_minute_quota_error(e: Exception) -> bool:
    return "per minute" in str(e).lower()


# ===========================================================================
# 5. ASYNC RUNTIME (notebook-compatible)
# ===========================================================================

_persistent_loop = None  # reused across calls so async HTTP clients stay bound to a live loop


def _run_async(coro):
    """Runs a coroutine, even if a loop is already running (notebook).

    Reuses a single persistent event loop across calls (instead of one
    ``asyncio.run()`` per call): the LangChain/OpenAI async HTTP client is
    created lazily and bound to whichever loop is running on first use: if
    that loop is closed (as ``asyncio.run()`` does after each call), every
    subsequent call fails with ``APIConnectionError``.
    """
    global _persistent_loop
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import nest_asyncio
        nest_asyncio.apply()
        return asyncio.get_event_loop().run_until_complete(coro)
    if _persistent_loop is None or _persistent_loop.is_closed():
        _persistent_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_persistent_loop)
    return _persistent_loop.run_until_complete(coro)


async def _safe_ascore(metric, sample, _rpm_retries: int = 3) -> float:
    """Score one sample, with per-minute rate-limit retry and fast-fail on others."""
    import asyncio as _asyncio
    name = type(metric).__name__
    for attempt in range(_rpm_retries + 1):
        try:
            return float(await metric.single_turn_ascore(sample))
        except Exception as e:
            if _is_daily_quota_error(e):
                warnings.warn(f"RAGAS {name}: daily quota exhausted → NaN (no retry).")
                return float("nan")
            if _is_per_minute_quota_error(e) and attempt < _rpm_retries:
                warnings.warn(
                    f"RAGAS {name}: per-minute rate limit → waiting 65 s "
                    f"(attempt {attempt + 1}/{_rpm_retries}) ..."
                )
                await _asyncio.sleep(65)
                continue
            warnings.warn(f"RAGAS metric {name} failed: {e}")
            return float("nan")
    return float("nan")


# ===========================================================================
# 6. FACTUAL CORRECTNESS (recomputed internally - bug #2693 workaround)
# ===========================================================================

def compute_factual_correctness(
    df: pd.DataFrame,
    fc_metric: "FactualCorrectness",
    pending_index: pd.Index | None = None,
    on_row_done=None,
) -> pd.DataFrame:
    """
    Computes precision / recall / f1 of factual correctness by reusing the
    RAGAS claim decomposition + NLI, BUT applying the correct formulas
    ourselves (the native RAGAS 'recall' mode is buggy, #2693).

    Two NLI directions:
      - ANSWER claims verified against the REFERENCE
            -> TP_precision (answer claims supported by reference)
      - REFERENCE claims verified against the ANSWER
            -> TP_recall (reference claims supported by answer)

    Correct formulas:
      precision = TP_precision / (number of answer claims)
      recall    = TP_recall    / (number of reference claims)
      f1        = 2PR / (P+R)

    Only ``pending_index`` rows are (re)computed (default: all of ``df``).
    ``on_row_done(idx, row_dict)``, if given, is called right after each row
    is scored — used by :func:`evaluate` to checkpoint progress incrementally.

    All rows run concurrently (asyncio.gather). Within each row, the two
    decompose_claims calls run in parallel, then the two verify_claims calls
    run in parallel — 2 round-trips instead of 4 sequential ones.
    """
    import asyncio as _asyncio
    eps = 1e-8
    idx_to_process = list(df.index if pending_index is None else pending_index)
    rows: dict = {}
    pbar = tqdm(total=len(idx_to_process), desc="factual_correctness")

    async def _compute_one(idx):
        r = df.loc[idx]
        response = str(r["response"])
        reference = str(r["reference"])
        precision = recall = f1 = float("nan")
        _fc_retries = 3
        for _fc_attempt in range(_fc_retries + 1):
            try:
                # Two decompose calls in parallel, then two verify calls in parallel.
                resp_claims, ref_claims = await _asyncio.gather(
                    fc_metric.decompose_claims(response, callbacks=None),
                    fc_metric.decompose_claims(reference, callbacks=None),
                )
                verdict_p, verdict_r = await _asyncio.gather(
                    fc_metric.verify_claims(premise=reference, hypothesis_list=resp_claims, callbacks=None),
                    fc_metric.verify_claims(premise=response, hypothesis_list=ref_claims, callbacks=None),
                )
                vp = np.asarray(verdict_p, dtype=bool)
                vr = np.asarray(verdict_r, dtype=bool)
                n_resp, n_ref = len(vp), len(vr)
                tp_precision = int(vp.sum())
                tp_recall = int(vr.sum())
                precision = tp_precision / (n_resp + eps) if n_resp else float("nan")
                recall = tp_recall / (n_ref + eps) if n_ref else float("nan")
                if _is_nan(precision) or _is_nan(recall):
                    f1 = float("nan")
                elif (precision + recall) == 0:
                    f1 = 0.0
                else:
                    f1 = 2 * precision * recall / (precision + recall)
                break
            except Exception as e:
                if _is_daily_quota_error(e):
                    warnings.warn("factual_correctness: daily quota exhausted → NaN (no retry).")
                    break
                if _is_per_minute_quota_error(e) and _fc_attempt < _fc_retries:
                    warnings.warn(
                        f"factual_correctness: per-minute rate limit → waiting 65 s "
                        f"(attempt {_fc_attempt + 1}/{_fc_retries}) ..."
                    )
                    await _asyncio.sleep(65)
                    continue
                warnings.warn(f"factual_correctness failed: {e}")
                break
        row = {
            "factual_correctness_precision": precision,
            "factual_correctness_recall": recall,
            "factual_correctness_f1": f1,
        }
        rows[idx] = row
        if on_row_done is not None:
            on_row_done(idx, row)
        pbar.update(1)

    async def _run_all():
        await _asyncio.gather(*[_compute_one(idx) for idx in idx_to_process])
        pbar.close()

    _run_async(_run_all())
    return pd.DataFrame.from_dict(rows, orient="index")


# ===========================================================================
# 7. RAGAS METRICS (context + faithfulness)
# ===========================================================================

def compute_ragas_metrics(
    df: pd.DataFrame,
    llm_wrapper,
    pending_index: pd.Index | None = None,
    on_row_done=None,
    ragas_workers: int = 3,
) -> pd.DataFrame:
    """
    Computes, per question:
      - faithfulness
      - context_recall_llm     (LLMContextRecall)
      - context_precision_llm  (LLMContextPrecisionWithReference, rank-aware)

    Only ``pending_index`` rows are (re)computed (default: all of ``df``).
    ``on_row_done(idx, row_dict)``, if given, is called right after each row
    is scored — used by :func:`evaluate` to checkpoint progress incrementally.
    ``ragas_workers`` controls how many questions are scored concurrently; within
    each question the 3 metrics are always computed in parallel via asyncio.gather.
    """
    import asyncio as _asyncio

    if not _RAGAS_AVAILABLE:
        raise ImportError(f"RAGAS unavailable: {_RAGAS_IMPORT_ERROR}")

    faithfulness_m = Faithfulness(llm=llm_wrapper)
    context_recall_m = LLMContextRecall(llm=llm_wrapper)
    context_precision_m = LLMContextPrecisionWithReference(llm=llm_wrapper)

    idx_to_process = list(df.index if pending_index is None else pending_index)
    pbar = tqdm(total=len(idx_to_process), desc="ragas")

    async def _process_one(idx, sem):
        r = df.loc[idx]
        sample = SingleTurnSample(
            user_input=r["user_input"],
            response=r["response"],
            reference=r["reference"],
            retrieved_contexts=r["retrieved_contexts"],
        )
        async with sem:
            fa, cr, cp = await _asyncio.gather(
                _safe_ascore(faithfulness_m, sample),
                _safe_ascore(context_recall_m, sample),
                _safe_ascore(context_precision_m, sample),
            )
        row = {
            "faithfulness": fa,
            "context_recall_llm": cr,
            "context_precision_llm": cp,
        }
        if on_row_done is not None:
            on_row_done(idx, row)
        pbar.update(1)
        return idx, row

    async def _all_questions():
        sem = _asyncio.Semaphore(ragas_workers)
        return await _asyncio.gather(*[_process_one(idx, sem) for idx in idx_to_process])

    all_results = _run_async(_all_questions())
    pbar.close()
    return pd.DataFrame.from_dict(dict(all_results), orient="index")


# ===========================================================================
# 8. HUB FRANCE IA F-SCORE (on the retrieved context)
# ===========================================================================

def hub_france_ia_fscore(
    context_precision_llm: float,
    context_recall_llm: float,
    beta: float = 0.5,
) -> float:
    """
    F-beta combining precision and recall OF THE RETRIEVED CONTEXT (not of the
    citations). Answers: "is the context sufficient without being excessive?".
    beta<1 (default 0.5) penalizes excessive context (cost/noise/dilution).
    """
    if _is_nan(context_precision_llm) or _is_nan(context_recall_llm):
        return float("nan")
    return _fbeta(context_precision_llm, context_recall_llm, beta)


# ===========================================================================
# 9. CITATIONS + CONTEXT RELEVANCE METRICS (files)
# ===========================================================================

def citation_scores_files(
    cited: Iterable[str],
    reference: Iterable[str],
    beta: float = 1.0,
    normalize: bool = True,
) -> dict[str, float]:
    """
    File-level precision / recall / F-beta of the citations.
      precision = |cited & expected| / |cited|
      recall    = |cited & expected| / |expected|
    Recall is the indicator retained in the global score (cite all the
    expected sources; correct surnumerary citations are tolerated).
    """
    c = _normalize_set(cited, normalize)
    ref = _normalize_set(reference, normalize)
    tp = len(c & ref)
    precision = tp / len(c) if c else float("nan")
    recall = tp / len(ref) if ref else float("nan")
    if _is_nan(precision) or _is_nan(recall):
        fbeta = float("nan")
    else:
        fbeta = _fbeta(precision, recall, beta)
    return {
        "citation_precision_files": precision,
        "citation_recall_files": recall,
        "citation_fscore_files": fbeta,
    }


def context_scores_files(
    retrieved_files: Iterable[str],
    reference: Iterable[str],
    normalize: bool = True,
) -> dict[str, float]:
    """
    File-level precision / recall of the RETRIEVED CONTEXT vs the expert reference set.
      context_relevance_files  = |retrieved & expected| / |retrieved|  (precision)
      context_coverage_files   = |retrieved & expected| / |expected|   (recall)

    Uses the files actually present in the retrieved chunks, NOT the cited files.
    """
    r = _normalize_set(retrieved_files, normalize)
    ref = _normalize_set(reference, normalize)
    tp = len(r & ref)
    return {
        "context_relevance_files": tp / len(r) if r else float("nan"),
        "context_coverage_files": tp / len(ref) if ref else float("nan"),
    }


# ===========================================================================
# 10. DIVERSITY (embeddings)
# ===========================================================================

def diversity_score(contexts: Sequence[str], raw_embeddings) -> float:
    """
    Diversity = 1 - (mean cosine similarity over all pairs of retrieved
    chunks). High = complementary chunks; low = near-duplicates.
    """
    _MAX_CHARS = 10000  # ~5600 tokens at ~1.78 chars/tok (French regulatory text), safely under the 8192-token embedding limit
    ctx = [str(c)[:_MAX_CHARS] for c in contexts if str(c).strip()]
    if len(ctx) < 2:
        return float("nan")
    vecs = np.asarray(raw_embeddings.embed_documents(list(ctx)))
    sims = [_cosine(vecs[i], vecs[j])
            for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
    return (1.0 - float(np.mean(sims))) if sims else float("nan")


# ===========================================================================
# 11. PARAPHRASE ROBUSTNESS (embeddings, per reformulation group)
# ===========================================================================

def paraphrase_robustness_by_group(df: pd.DataFrame, raw_embeddings) -> pd.Series:
    """
    Measures the semantic consistency of the answers to reformulations of the
    same question (state of the art: group similarity, Con-RAG 2025). This is
    NOT self-consistency (sampling variance) but robustness to input
    reformulation.

    For each group defined by `paraphrase_group_id`, the answers are embedded
    and the mean cosine similarity over all pairs is computed. All rows of a
    group receive the same value. Groups of size 1 -> NaN.
    """
    scores = pd.Series(np.nan, index=df.index, dtype=float)
    if "paraphrase_group_id" not in df.columns:
        return scores

    groups = list(df.groupby("paraphrase_group_id"))
    for _, grp in tqdm(groups, desc="paraphrase_robustness"):
        responses = [str(x) for x in grp["response"].tolist() if str(x).strip()]
        if len(responses) < 2:
            continue
        vecs = np.asarray(raw_embeddings.embed_documents(responses))
        sims = [_cosine(vecs[i], vecs[j])
                for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
        scores.loc[grp.index] = float(np.mean(sims)) if sims else float("nan")
    return scores


# ===========================================================================
# 12. ANCHORING / HALLUCINATION (LLM, 4 categories)
# ===========================================================================

_ANCHOR_PROMPT = """You are an expert evaluator specialized in the factual analysis of RAG answers.

You are given:
1. A GENERATED ANSWER
2. A REFERENCE ANSWER (expected facts)
3. A RETRIEVED CONTEXT (sources)

Steps:
A. Decompose the GENERATED ANSWER into atomic facts.
B. Classify EACH fact into exactly one of the 4 categories:
   - "expected_grounded"     : present in the reference AND supported by the context
   - "expected_ungrounded"   : present in the reference BUT not supported by the context
   - "additional_grounded"   : absent from the reference BUT supported by the context (valid enrichment)
   - "additional_ungrounded" : absent from the reference AND not supported by the context (hallucination)

Reply ONLY with a JSON object, with no text or Markdown, in the format:
{{"facts": [{{"fact": "...", "category": "..."}}]}}

GENERATED ANSWER:
{response}

REFERENCE ANSWER:
{reference}

RETRIEVED CONTEXT:
{context}
"""

_ANCHOR_CATEGORIES = (
    "expected_grounded",
    "expected_ungrounded",
    "additional_grounded",
    "additional_ungrounded",
)


def _extract_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON found.")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                chunk = text[start:i + 1]
                try:
                    return json.loads(chunk)
                except json.JSONDecodeError:
                    # Fix lone backslashes not part of a valid JSON escape sequence
                    chunk = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', chunk)
                    return json.loads(chunk)
    raise ValueError("Unbalanced JSON.")


def _llm_invoke_text(llm_wrapper, prompt: str, _rpm_retries: int = 3) -> str:
    """Invoke the LLM synchronously with per-minute retry and fast-fail on other errors."""
    import time as _time
    lc = getattr(llm_wrapper, "langchain_llm", None) or llm_wrapper
    for attempt in range(_rpm_retries + 1):
        try:
            out = lc.invoke(prompt)
            return getattr(out, "content", str(out))
        except Exception as e:
            if _is_daily_quota_error(e):
                raise RuntimeError(f"Daily quota exhausted: {e}") from e
            if _is_per_minute_quota_error(e) and attempt < _rpm_retries:
                warnings.warn(
                    f"LLM invoke: per-minute rate limit → waiting 65 s "
                    f"(attempt {attempt + 1}/{_rpm_retries}) ..."
                )
                _time.sleep(65)
                continue
            raise
    raise RuntimeError("_llm_invoke_text: max retries exceeded")


async def _llm_invoke_text_async(llm_wrapper, prompt: str, _rpm_retries: int = 3) -> str:
    """Async version of _llm_invoke_text using ainvoke."""
    import asyncio as _asyncio
    lc = getattr(llm_wrapper, "langchain_llm", None) or llm_wrapper
    for attempt in range(_rpm_retries + 1):
        try:
            out = await lc.ainvoke(prompt)
            return getattr(out, "content", str(out))
        except Exception as e:
            if _is_daily_quota_error(e):
                raise RuntimeError(f"Daily quota exhausted: {e}") from e
            if _is_per_minute_quota_error(e) and attempt < _rpm_retries:
                warnings.warn(
                    f"LLM invoke: per-minute rate limit → waiting 65 s "
                    f"(attempt {attempt + 1}/{_rpm_retries}) ..."
                )
                await _asyncio.sleep(65)
                continue
            raise
    raise RuntimeError("_llm_invoke_text_async: max retries exceeded")


def anchoring_analysis(
    response: str,
    reference: str,
    contexts: Sequence[str],
    llm_wrapper,
) -> dict[str, float]:
    """
    Classifies the atomic facts of the answer into 4 categories and derives:
      - counts per category
      - hallucination_rate = additional_ungrounded / total
      - grounded_rate      = (expected_grounded + additional_grounded) / total
    """
    context_str = "\n\n".join(str(c) for c in contexts if str(c).strip())
    prompt = _ANCHOR_PROMPT.format(
        response=response, reference=reference, context=context_str or "(empty)"
    )
    counts = {c: 0 for c in _ANCHOR_CATEGORIES}
    try:
        data = _extract_json(_llm_invoke_text(llm_wrapper, prompt))
        for f in data.get("facts", []):
            cat = f.get("category")
            if cat in counts:
                counts[cat] += 1
    except Exception as e:
        warnings.warn(f"anchoring_analysis failed: {e}")
        return {
            **{f"anchor_{c}": float("nan") for c in _ANCHOR_CATEGORIES},
            "hallucination_rate": float("nan"),
            "grounded_rate": float("nan"),
        }

    total = sum(counts.values())
    grounded = counts["expected_grounded"] + counts["additional_grounded"]
    halluc = counts["additional_ungrounded"]
    return {
        **{f"anchor_{c}": float(counts[c]) for c in _ANCHOR_CATEGORIES},
        "hallucination_rate": (halluc / total) if total else float("nan"),
        "grounded_rate": (grounded / total) if total else float("nan"),
    }


# ===========================================================================
# 13. GLOBAL SCORE (4 components)
# ===========================================================================

def global_rag_score(row: dict[str, float], weights: dict[str, float]) -> float:
    """
    Weighted aggregation of the 4 components. NaN/missing components are
    ignored and the weights renormalized over the ones present.
    """
    num, denom = 0.0, 0.0
    for metric, w in weights.items():
        v = row.get(metric)
        if not _is_nan(v):
            num += w * float(v)
            denom += w
    return (num / denom) if denom else float("nan")


# ===========================================================================
# 14. ORCHESTRATION
# ===========================================================================

def evaluate(
    df: pd.DataFrame,
    config: EvalConfig | None = None,
    *,
    use_ragas: bool = True,
    use_factual: bool = True,
    use_anchoring: bool = True,
    use_diversity: bool = True,
    use_paraphrase: bool = True,
    verbose: bool = True,
    checkpoint_path: str | Path | None = None,
    ragas_workers: int = 3,
) -> pd.DataFrame:
    """
    Computes all enabled metrics, per question, and the global score.

    Parameters
    ----------
    use_ragas      : faithfulness, context_recall_llm, context_precision_llm
                     (+ derived hub_france_ia_fscore).
    use_factual    : factual_correctness precision/recall/f1 (recomputed in-house).
    use_anchoring  : anchoring/hallucination analysis (LLM).
    use_diversity  : context diversity (embeddings).
    use_paraphrase : paraphrase robustness (embeddings, requires the
                     paraphrase_group_id column).
    checkpoint_path : if given, results are saved to this .xlsx after every
                     row, and reloaded from it on start (rows whose metrics
                     are already present are not recomputed). A long run
                     (e.g. ragas + anchoring on 65 questions, rate-limited
                     to ~10 req/min) can then be interrupted and resumed
                     without losing already-scored questions.

    Returns a DataFrame: one row per question, one column per metric, plus
    'global_rag_score'.
    """
    config = config or EvalConfig()

    # Variant rows (is_paraphrase=True) only contribute to paraphrase_robustness.
    # All other metrics run on original questions only; the score file only
    # contains original rows.
    if "is_paraphrase" in df.columns:
        df_eval = df[~df["is_paraphrase"].fillna(False)].copy()
    else:
        df_eval = df

    results = pd.DataFrame(index=df_eval.index)
    for col in ("q_id", "test_intent", "user_input"):
        if col in df_eval.columns:
            results[col] = df_eval[col]

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        prev = pd.read_excel(checkpoint_path)
        if len(prev) == len(df_eval):
            prev.index = df_eval.index
            _id_cols = {"q_id", "test_intent", "user_input"}
            for col in prev.columns:
                if col not in _id_cols:
                    results[col] = prev[col]
            if verbose:
                print(f"[resume] Loaded checkpoint {checkpoint_path} ({len(prev.columns) - 1} metric columns).")
        elif verbose:
            print(
                f"[resume] Checkpoint {checkpoint_path} has {len(prev)} rows, "
                f"expected {len(df_eval)} -> ignoring (recomputing from scratch)."
            )

    def _checkpoint() -> None:
        if checkpoint_path is not None:
            results.to_excel(checkpoint_path, index=False)

    def _pending(cols: list[str]) -> pd.Index:
        for c in cols:
            if c not in results.columns:
                results[c] = float("nan")
        return results.index[results[cols].isna().any(axis=1)]

    need_llm = use_ragas or use_factual or use_anchoring
    need_emb = use_diversity or use_paraphrase

    llm_wrapper = raw_emb = None
    if need_llm:
        _check_ragas_version()
        if verbose:
            print(f"[init] Judge LLM: {config.llm.model_key} (via {config.llm.proxy_prefix})")
        llm_wrapper = build_llm(config.llm, verbose=verbose)
    if need_emb:
        if verbose:
            print(f"[init] Embeddings: {config.embeddings.model_key} (via {config.embeddings.proxy_prefix})")
        raw_emb, _ = build_embeddings(config.embeddings)
        if config.embedding_cache_path:
            raw_emb = _EmbeddingCache(raw_emb, config.embedding_cache_path)

    # --- RAGAS: faithfulness, context recall/precision ---------------------
    if use_ragas:
        pending = _pending(["faithfulness", "context_recall_llm", "context_precision_llm"])
        if len(pending):
            if verbose:
                print(f"[ragas] faithfulness, context_recall_llm, context_precision_llm ({len(pending)}/{len(df_eval)} pending) ...")

            def _on_ragas_row(idx, row):
                for k, v in row.items():
                    results.loc[idx, k] = v
                _checkpoint()

            compute_ragas_metrics(df_eval, llm_wrapper, pending_index=pending, on_row_done=_on_ragas_row, ragas_workers=ragas_workers)
        elif verbose:
            print("[ragas] already complete (resumed from checkpoint).")
        # Hub France IA F-Score (derived, diagnostic "sufficient without excessive") — cheap, always recomputed.
        results["hub_france_ia_fscore"] = [
            hub_france_ia_fscore(
                results.loc[i, "context_precision_llm"],
                results.loc[i, "context_recall_llm"],
                config.hub_fscore_beta,
            )
            for i in results.index
        ]
        _checkpoint()

    # --- Factual Correctness (recomputed in-house) ------------------------
    if use_factual:
        pending = _pending(["factual_correctness_precision", "factual_correctness_recall", "factual_correctness_f1"])
        if len(pending):
            if verbose:
                print(f"[factual] precision/recall/f1 ({len(pending)}/{len(df_eval)} pending, internal recompute, bug #2693) ...")
            fc_metric = FactualCorrectness(llm=llm_wrapper, atomicity=config.factual_atomicity)

            def _on_factual_row(idx, row):
                for k, v in row.items():
                    results.loc[idx, k] = v
                _checkpoint()

            compute_factual_correctness(df_eval, fc_metric, pending_index=pending, on_row_done=_on_factual_row)
        elif verbose:
            print("[factual] already complete (resumed from checkpoint).")

    # --- Context metrics (retrieved_files vs reference) --------------------
    if "retrieved_files" in df_eval.columns and "reference_files" in df_eval.columns:
        pending = _pending(["context_relevance_files", "context_coverage_files"])
        if len(pending):
            if verbose:
                print(f"[context] relevance + coverage (retrieved files vs reference, {len(pending)}/{len(df_eval)} pending) ...")
            norm = config.normalize_filenames
            for idx in tqdm(pending, desc="context files"):
                r = df_eval.loc[idx]
                for k, v in context_scores_files(r["retrieved_files"], r["reference_files"], norm).items():
                    results.loc[idx, k] = v
                _checkpoint()
        elif verbose:
            print("[context] already complete (resumed from checkpoint).")
    elif verbose:
        print("[context] retrieved_files column missing -> context_relevance/coverage skipped.")

    # --- Citation metrics (cited_files vs reference) -----------------------
    if "cited_files" in df_eval.columns and "reference_files" in df_eval.columns:
        pending = _pending(["citation_precision_files", "citation_recall_files", "citation_fscore_files"])
        if len(pending):
            if verbose:
                print(f"[citations] precision/recall/fscore (cited files vs reference, {len(pending)}/{len(df_eval)} pending) ...")
            norm = config.normalize_filenames
            for idx in tqdm(pending, desc="citations"):
                r = df_eval.loc[idx]
                for k, v in citation_scores_files(
                    r["cited_files"], r["reference_files"], config.citation_fscore_beta, norm
                ).items():
                    results.loc[idx, k] = v
                _checkpoint()
        elif verbose:
            print("[citations] already complete (resumed from checkpoint).")
    elif verbose:
        print("[citations] cited_files/reference_files columns missing -> skipped.")

    # --- Diversity ---------------------------------------------------------
    if use_diversity:
        pending = _pending(["diversity"])
        if len(pending):
            if verbose:
                print(f"[diversity] chunk dissimilarity ({len(pending)}/{len(df_eval)} pending) ...")
            for idx in tqdm(pending, desc="diversity"):
                results.loc[idx, "diversity"] = diversity_score(df_eval.loc[idx, "retrieved_contexts"], raw_emb)
                _checkpoint()
        elif verbose:
            print("[diversity] already complete (resumed from checkpoint).")

    # --- Paraphrase robustness (whole-df groupby, embeddings only — cheap) -
    if use_paraphrase:
        if "paraphrase_group_id" in df.columns:
            pending = _pending(["paraphrase_robustness"])
            if len(pending):
                if verbose:
                    n_groups = df["paraphrase_group_id"].nunique()
                    print(f"[paraphrase] robustness to reformulation ({n_groups} groups, {len(df) - len(df_eval)} variants) ...")
                # Compute on full df (originals + variants), keep only original rows in results
                all_scores = paraphrase_robustness_by_group(df, raw_emb)
                results["paraphrase_robustness"] = all_scores.reindex(df_eval.index)
                _checkpoint()
            elif verbose:
                print("[paraphrase] already complete (resumed from checkpoint).")
        elif verbose:
            print("[paraphrase] paraphrase_group_id column missing -> skipped.")

    # --- Anchoring / hallucination -----------------------------------------
    if use_anchoring:
        import asyncio as _asyncio
        anchor_cols = [f"anchor_{c}" for c in _ANCHOR_CATEGORIES] + ["hallucination_rate", "grounded_rate"]
        pending = _pending(anchor_cols)
        if len(pending):
            if verbose:
                print(f"[anchoring] factual decomposition ({len(pending)}/{len(df_eval)} pending, LLM) ...")
            pbar_anchor = tqdm(total=len(pending), desc="anchoring")
            anchor_sem = _asyncio.Semaphore(ragas_workers)

            async def _anchor_one(idx):
                r = df_eval.loc[idx]
                context_str = "\n\n".join(str(c) for c in r["retrieved_contexts"] if str(c).strip())
                prompt = _ANCHOR_PROMPT.format(
                    response=r["response"], reference=r["reference"],
                    context=context_str or "(empty)"
                )
                counts = {c: 0 for c in _ANCHOR_CATEGORIES}
                try:
                    async with anchor_sem:
                        # Long-context judge calls occasionally return malformed
                        # JSON — resample up to 3 times before giving up.
                        for _attempt in range(3):
                            try:
                                data = _extract_json(await _llm_invoke_text_async(llm_wrapper, prompt))
                                break
                            except (ValueError, json.JSONDecodeError):
                                if _attempt == 2:
                                    raise
                    for f in data.get("facts", []):
                        cat = f.get("category")
                        if cat in counts:
                            counts[cat] += 1
                except Exception as e:
                    warnings.warn(f"anchoring_analysis failed: {e}")
                    for k in anchor_cols:
                        results.loc[idx, k] = float("nan")
                    pbar_anchor.update(1)
                    return
                total = sum(counts.values())
                grounded = counts["expected_grounded"] + counts["additional_grounded"]
                halluc = counts["additional_ungrounded"]
                anchor = {
                    **{f"anchor_{c}": float(counts[c]) for c in _ANCHOR_CATEGORIES},
                    "hallucination_rate": (halluc / total) if total else float("nan"),
                    "grounded_rate": (grounded / total) if total else float("nan"),
                }
                for k, v in anchor.items():
                    results.loc[idx, k] = v
                _checkpoint()
                pbar_anchor.update(1)

            async def _run_all_anchors():
                await _asyncio.gather(*[_anchor_one(idx) for idx in pending])
            _run_async(_run_all_anchors())
            pbar_anchor.close()
        elif verbose:
            print("[anchoring] already complete (resumed from checkpoint).")

    # --- Global score (4 components) ---------------------------------------
    if verbose:
        print("[global] 4-component aggregation (recall + faithfulness) ...")
    results["global_rag_score"] = [
        global_rag_score(results.loc[i].to_dict(), config.global_weights)
        for i in results.index
    ]
    _checkpoint()

    return results


# ===========================================================================
# 15. AGGREGATION AND EXPORT
# ===========================================================================

def compute_averages(results: pd.DataFrame) -> pd.DataFrame:
    """Mean (NaN ignored) of each numeric metric + number of valid values."""
    numeric = results.select_dtypes(include=[np.number])
    out = numeric.mean(skipna=True).to_frame(name="mean")
    out["n_valid"] = numeric.notna().sum()
    out.index.name = "metric"
    return out.reset_index()


def export_results(results: pd.DataFrame, averages: pd.DataFrame,
                   output_prefix: str) -> tuple[str, str]:
    per_q = f"{output_prefix}_per_question.xlsx"
    avg = f"{output_prefix}_averages.xlsx"
    results.to_excel(per_q, index=False)
    averages.to_excel(avg, index=False)
    return per_q, avg


def run_evaluation(
    input_path: str,
    output_prefix: str,
    config: EvalConfig | None = None,
    colmap: dict[str, str] | None = None,
    *,
    use_ragas: bool = True,
    use_factual: bool = True,
    use_anchoring: bool = True,
    use_diversity: bool = True,
    use_paraphrase: bool = True,
    verbose: bool = True,
    ragas_workers: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, str]]:
    """
    Full pipeline (main entry point for the notebook).

    The per-question output file doubles as a checkpoint: re-running with the
    same ``output_prefix`` resumes from it instead of recomputing already-scored
    questions (see :func:`evaluate`).

    Returns (results, averages, (per_question_path, averages_path)).
    """
    df = load_dataset(input_path, colmap=colmap)
    if verbose:
        print(f"[load] {len(df)} questions from {input_path}")
    results = evaluate(
        df, config,
        use_ragas=use_ragas, use_factual=use_factual,
        use_anchoring=use_anchoring, use_diversity=use_diversity,
        use_paraphrase=use_paraphrase, verbose=verbose,
        checkpoint_path=f"{output_prefix}_per_question.xlsx",
        ragas_workers=ragas_workers,
    )
    averages = compute_averages(results)
    paths = export_results(results, averages, output_prefix)
    if verbose:
        print(f"[export] {paths[0]}\n[export] {paths[1]}")
    return results, averages, paths


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="RAG / GraphRAG evaluation (RAGAS + custom).")
    p.add_argument(
        "input", nargs="?", default=None,
        help="Input .xlsx file (default: data/results/eval_test_<experiment-id>.xlsx, "
             "i.e. the file produced by prepare_eval_dataset.py for the same experiment)",
    )
    p.add_argument(
        "--experiment-id", dest="experiment_id", default=None,
        help="Experiment identifier shared with prepare_eval_dataset.py. Used to derive "
             "both the default input path and the output prefix.",
    )
    p.add_argument("-o", "--output-prefix", default=None)
    p.add_argument("--llm-model-key", default=LLMConfig().model_key, help="Cleyrop proxy LLM model key")
    p.add_argument("--emb-model-key", default=EmbeddingConfig().model_key, help="Cleyrop proxy embedding model key")
    p.add_argument("--proxy-prefix", default=_PROXY_PREFIX, help="Cleyrop AI-gen proxy base URL")
    p.add_argument("--no-ragas", action="store_true")
    p.add_argument("--no-factual", action="store_true")
    p.add_argument("--no-anchoring", action="store_true")
    p.add_argument("--no-diversity", action="store_true")
    p.add_argument("--no-paraphrase", action="store_true")
    p.add_argument("--hub-beta", type=float, default=EvalConfig().hub_fscore_beta)
    p.add_argument("--citation-beta", type=float, default=EvalConfig().citation_fscore_beta)
    p.add_argument("--ragas-workers", type=int, default=3, dest="ragas_workers",
                   help="Questions scored concurrently by RAGAS (default: 3). "
                        "Each question runs 3 metrics in parallel; watch rate limits.")
    p.add_argument(
        "--embedding-cache", default=None, dest="embedding_cache",
        help="Path to a .pkl file used to cache embedding vectors across runs "
             "(diversity, paraphrase_robustness). Auto-derived from experiment-id when omitted.",
    )
    a = p.parse_args()

    if a.input is None and a.experiment_id is None:
        p.error("either INPUT or --experiment-id is required")

    _results_dir = Path(__file__).parent / "data" / "results"
    _results_dir.mkdir(parents=True, exist_ok=True)
    input_path = a.input or str(_results_dir / f"eval_test_{a.experiment_id}.xlsx")
    output_prefix = a.output_prefix or (
        str(_results_dir / f"score_test_{a.experiment_id}") if a.experiment_id
        else "rag_eval"
    )

    # Default cache path: data/results/emb_cache_<model>_<experiment>.pkl
    _model_slug = a.emb_model_key.replace("/", "_").replace("-", "_")
    _cache_default = (
        str(_results_dir / f"emb_cache_{_model_slug}_{a.experiment_id}.pkl")
        if a.experiment_id else None
    )
    embedding_cache = a.embedding_cache if a.embedding_cache is not None else _cache_default

    cfg = EvalConfig(
        llm=LLMConfig(model_key=a.llm_model_key, proxy_prefix=a.proxy_prefix),
        embeddings=EmbeddingConfig(model_key=a.emb_model_key, proxy_prefix=a.proxy_prefix),
        hub_fscore_beta=a.hub_beta,
        citation_fscore_beta=a.citation_beta,
        embedding_cache_path=embedding_cache,
    )
    run_evaluation(
        input_path, output_prefix, cfg,
        use_ragas=not a.no_ragas, use_factual=not a.no_factual,
        use_anchoring=not a.no_anchoring, use_diversity=not a.no_diversity,
        use_paraphrase=not a.no_paraphrase,
        ragas_workers=a.ragas_workers,
    )
