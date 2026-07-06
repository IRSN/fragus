"""fermi-1024 (sparse) inference server — run LOCALLY on the Mac (MPS/GPU).

Purpose: offload fermi inference to the Apple Silicon GPU (Metal/MPS), much
faster than the VM's CPU. The server receives texts and returns their sparse
vectors {token_id: weight}; `scripts/pipeline/embed_fermi.py --encoder-url ...` (on the
VM) consumes this API and writes into Milvus.

Mac dependencies (in a venv):
    pip install torch transformers huggingface_hub

HuggingFace auth (gated model) — once:
    export HF_TOKEN=hf_xxx        # your token, from an account that accepted access to the model

Launch:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python fermi_server.py
    # → listens on http://127.0.0.1:8000  (device: mps)

Quick test:
    curl -s localhost:8000/health
    curl -s -XPOST localhost:8000/encode -H 'content-type: application/json' \\
         -d '{"texts":["criticality safety of the spent fuel cask"]}' | head -c 300
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

MODEL_ID = os.environ.get("FERMI_MODEL_ID", "atomic-canyon/fermi-1024")


def pick_device() -> str:
    # FERMI_DEVICE allows forcing the device. MPS (Metal GPU) can freeze after
    # the Mac goes to sleep and may even make the whole system sluggish: you can
    # therefore force "cpu" for robust operation (FERMI_DEVICE=cpu).
    forced = os.environ.get("FERMI_DEVICE", "").strip().lower()
    if forced:
        return forced
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# CPU throttling: caps the number of threads to keep the Mac responsive during
# the run (FERMI_NUM_THREADS, default = no change).
_NTHREADS = os.environ.get("FERMI_NUM_THREADS", "").strip()
if _NTHREADS:
    torch.set_num_threads(int(_NTHREADS))

print(f"Loading {MODEL_ID}...", flush=True)
DEVICE = pick_device()
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
MODEL = AutoModelForMaskedLM.from_pretrained(MODEL_ID).to(DEVICE).eval()
SPECIAL_TOKEN_IDS = TOKENIZER.all_special_ids
print(f"Model loaded on device: {DEVICE}", flush=True)

# MPS/Metal is not thread-safe: ThreadingHTTPServer spawns one thread per request,
# and two concurrent inferences on the Apple GPU segfault. So we serialize access
# to the model. Cost is ~zero when requests already arrive serially.
_MODEL_LOCK = threading.Lock()


def encode(texts: list[str], max_seq_len: int) -> list[dict[str, float]]:
    with _MODEL_LOCK:
        return _encode_locked(texts, max_seq_len)


def _encode_locked(texts: list[str], max_seq_len: int) -> list[dict[str, float]]:
    feature = TOKENIZER(
        texts,
        padding=True,
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
        return_token_type_ids=False,
    ).to(DEVICE)
    with torch.inference_mode():
        output = MODEL(**feature)[0]  # logits [B, L, V]
    values, _ = torch.max(output * feature["attention_mask"].unsqueeze(-1), dim=1)
    values = torch.log(1 + torch.relu(values))
    values[:, SPECIAL_TOKEN_IDS] = 0
    values = values.to("cpu")

    out: list[dict[str, float]] = []
    for i in range(values.shape[0]):
        row = values[i]
        idx = torch.nonzero(row, as_tuple=False).squeeze(1)
        weights = row[idx]
        out.append({str(int(k)): float(v) for k, v in zip(idx.tolist(), weights.tolist())})
    return out


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
        if self.path != "/encode":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length).decode())
            texts = req["texts"]
            max_seq_len = int(req.get("max_seq_len", 1024))
            self._send(200, {"sparse": encode(texts, max_seq_len)})
        except Exception as e:  # pragma: no cover
            self._send(500, {"error": repr(e)})

    def log_message(self, *args):  # silence per-request logging
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="fermi (sparse) inference server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Listening on http://{args.host}:{args.port}  (device: {DEVICE})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
