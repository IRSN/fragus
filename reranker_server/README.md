# reranker_server — mmarco-mMiniLMv2 reranking on a Mac (MPS/GPU)

Standalone HTTP server that offloads **reranking** (cross-encoder
`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`) to the Apple Silicon GPU (Metal/MPS)
of a Mac. Model aligned with the `HybridRAGPipeline` default. **Public** (no HF
token) and **multilingual/native French**.

Used by `HybridRAGPipeline` (via `RERANKER_URL`) and by `rag_evaluation.py` to
score candidates coming out of Milvus retrieval.

Same spirit as [`fermi_server`](../fermi_server).

```
Mac (MPS)                            VM
┌────────────────────┐   tunnel   ┌──────────────────────────────┐
│ reranker_server.py │ <───────── │ HybridRAGPipeline / eval      │
│  POST /score       │  pairs →   │  Milvus dense/sparse candidates│
│  → aligned scores  │ ← scores   │  → top-n to the LLM           │
└────────────────────┘            └──────────────────────────────┘
```

## 1. Get this folder onto the Mac

```bash
# if the Mac has a copy of the repo:
git pull && cd reranker_server
# otherwise:
scp -r <vm>:/home/cleyrop/projects/fragus/reranker_server .
```

## 2. Python environment (Mac)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers huggingface_hub
```

## 3. Start the server

```bash
./run_server.sh
# → Listening on http://127.0.0.1:<PORT>  (device: mps)
```

`run_server.sh` tries port `RERANKER_PORT` (default **8001**). If that port is
taken by another process, it automatically picks the first free port starting
from 8001 and records the result in `logs/server.port`. A warning is emitted on
stderr if the preferred port is not available.

The script throttles the Mac so it doesn't get saturated: `caffeinate -i` (no
sleep while it runs), `nice -n 5` and `OMP/MKL_NUM_THREADS=4` (≈4 CPU cores at
low priority — the bulk of the compute is on the GPU). `RERANKER_BATCH=32` by default.

> The model (`mmarco-mMiniLMv2-L12-H384-v1`, ~134M params, ~470 MB) is lightweight:
> it barely uses the GPU or RAM. `caffeinate -i` does not cover a
> **closed lid on battery** → keep the screen open, or stay on AC power with an
> external display.

Env variables: `RERANKER_DEVICE` (force cpu/mps/cuda), `RERANKER_MODEL_ID`,
`RERANKER_MAX_LEN` (512), `RERANKER_BATCH` (32), `RERANKER_PORT` (preferred port).
"Raw" launch without the wrapper: `PYTORCH_ENABLE_MPS_FALLBACK=1 python reranker_server.py`.

Local test (check the port in `logs/server.port`):

```bash
PORT=$(cat logs/server.port)
curl -s localhost:$PORT/health
# {"status":"ok","device":"mps","model":"cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"}
```

## 4. Reverse SSH tunnel (Mac → VM)

```bash
./run_tunnel.sh
```

`run_tunnel.sh` opens a reverse tunnel **with automatic reconnection** (`autossh`,
requires `brew install autossh`): `localhost:<PORT>` on the VM then points to the
Mac server. It reads the port from `logs/server.port` (written by `run_server.sh`)
to automatically use the right port, even if it differs from 8001. It detects a
dead connection in ~45 s and restarts, and retries cleanly if the port stayed
busy on the VM side after a disconnect.

The default host is the `~/.ssh/config` alias **`cleyrop-asnr-secnum-devspace`**;
override with `RERANKER_TUNNEL_HOST`:

```bash
RERANKER_TUNNEL_HOST=my-other-vm ./run_tunnel.sh
```

## 4 bis. Robust: auto start + restart on crash (launchd)

For the server **and** the tunnel to start at login and restart automatically
on crash:

```bash
./install_launchd.sh             # installs / reloads both LaunchAgents
./install_launchd.sh uninstall   # removes everything
```

Logs in `logs/` (`server.err.log`, `tunnel.err.log`). Status:
`launchctl list | grep reranker`. The target VM is set via `RERANKER_TUNNEL_HOST`
in `launchd/com.cleyrop.reranker.tunnel.plist`.

## 5. Use from the VM

Check the port used on the Mac side:

```bash
# On the Mac
cat reranker_server/logs/server.port   # e.g. 8001
```

**RAG pipeline** (`HybridRAGPipeline`):

```bash
export RERANKER_URL=http://localhost:<PORT>   # port printed by run_server.sh
```

or in Python:

```python
from scripts.rag.hybrid_rag_pipeline import HybridRAGPipeline
rag = HybridRAGPipeline(reranker_url="http://localhost:<PORT>")
```

**Evaluation** (`rag_evaluation.py`):

```bash
export RERANKER_URL=http://localhost:<PORT>
# then run the eval notebook / script as usual
```

## API

| Method  | Route      | Body / response |
|---------|------------|-----------------|
| `GET`   | `/health`  | `{"status","device","model"}` |
| `POST`  | `/rerank`  | req `{"query":str, "documents":[str], "top_n":int?, "return_documents":bool?}` → `{"results":[{"index","score","document"?}]}` (sorted desc) |
| `POST`  | `/score`   | req `{"pairs":[[query,doc], ...]}` → `{"scores":[float]}` (aligned with `pairs`, unsorted) — **native interface of `HybridRAGPipeline`** |

The `score` is the raw cross-encoder logit (unbounded). Only the relative order
matters. For a probability in `[0,1]`, apply a sigmoid on the client side.
