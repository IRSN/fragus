"""cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 reranking server — run LOCALLY on the Mac (MPS/GPU).

Purpose: offload reranking (cross-encoder) to the Apple Silicon GPU (Metal/MPS),
much faster than the VM's CPU. The server receives a query + a list of passages
and returns a relevance score per passage; on the VM side, the reranking step
consumes this API to reorder the candidates coming out of retrieval
(dense/sparse Milvus) before generation.

Public model (no HF token), multilingual, aligned with the HybridRAGPipeline
default (RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1").
Override with RERANKER_MODEL_ID to use another cross-encoder.

Mac dependencies (in a venv):
    pip install torch transformers huggingface_hub

Launch:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python reranker_server.py
    # → listens on http://127.0.0.1:8001  (device: mps)

Quick test:
    curl -s localhost:8001/health
    curl -s -XPOST localhost:8001/score -H 'content-type: application/json' \\
         -d '{"pairs":[["criticality safety of the spent fuel cask",
                        "spent fuel is stored in a pool"],
                       ["criticality safety of the spent fuel cask",
                        "the weather is sunny"]]}' | head -c 200
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = os.environ.get("RERANKER_MODEL_ID", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
DEFAULT_MAX_LEN = int(os.environ.get("RERANKER_MAX_LEN", "512"))
DEFAULT_BATCH = int(os.environ.get("RERANKER_BATCH", "16"))


def pick_device() -> str:
    forced = os.environ.get("RERANKER_DEVICE", "").strip().lower()
    if forced:
        return forced
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


print(f"Loading {MODEL_ID}...", flush=True)
DEVICE = pick_device()
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(DEVICE).eval()
print(f"Model loaded on device: {DEVICE}", flush=True)

# ThreadingHTTPServer handles one request per thread, but MODEL is shared: a
# concurrent forward on the same module/device (MPS) is not thread-safe and can
# crash or corrupt scores. So we serialize GPU access. Tokenization (CPU, outside
# the lock) remains parallel across requests.
_GPU_LOCK = threading.Lock()


def score_pairs(
    pairs: list[list[str]],
    max_seq_len: int,
    batch_size: int,
) -> list[float]:
    """Score each (query, document) — raw logit, higher = more relevant."""
    scores: list[float] = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        feature = TOKENIZER(
            batch,
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )
        with _GPU_LOCK:
            feature = feature.to(DEVICE)
            with torch.inference_mode():
                logits = MODEL(**feature, return_dict=True).logits.view(-1).float()
            scores.extend(logits.to("cpu").tolist())
    return scores


def rerank(
    query: str,
    documents: list[str],
    max_seq_len: int,
    batch_size: int,
    top_n: int | None,
    return_documents: bool,
) -> list[dict]:
    """Return the passages reordered by decreasing score."""
    pairs = [[query, doc] for doc in documents]
    raw = score_pairs(pairs, max_seq_len, batch_size)
    order = sorted(range(len(documents)), key=lambda i: raw[i], reverse=True)
    if top_n is not None:
        order = order[:top_n]
    results: list[dict] = []
    for i in order:
        item: dict = {"index": i, "score": raw[i]}
        if return_documents:
            item["document"] = documents[i]
        results.append(item)
    return results


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            self._send(200, {"status": "ok", "device": DEVICE, "model": MODEL_ID})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length).decode())
            max_seq_len = int(req.get("max_seq_len", DEFAULT_MAX_LEN))
            batch_size = int(req.get("batch_size", DEFAULT_BATCH))

            if self.path == "/rerank":
                # {"query":str, "documents":[str], "top_n":int?, "return_documents":bool?}
                results = rerank(
                    req["query"],
                    req["documents"],
                    max_seq_len,
                    batch_size,
                    req.get("top_n"),
                    bool(req.get("return_documents", False)),
                )
                self._send(200, {"results": results})

            elif self.path == "/score":
                # {"pairs":[[query,doc],...]} → aligned scores (used by HybridRAGPipeline)
                raw = score_pairs(req["pairs"], max_seq_len, batch_size)
                self._send(200, {"scores": raw})

            else:
                self._send(404, {"error": "not found"})

        except Exception as e:  # pragma: no cover
            self._send(500, {"error": repr(e)})

    def log_message(self, *args):  # silence per-request logging
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-encoder reranking server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Listening on http://{args.host}:{args.port}  (device: {DEVICE})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
