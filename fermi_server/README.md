# fermi_server — fermi-1024 sparse inference on a Mac (MPS/GPU)

Standalone HTTP server that offloads inference of the sparse model
`atomic-canyon/fermi-1024` to the Apple Silicon GPU (Metal/MPS) of a Mac, much
faster than the VM's CPU. It receives texts and returns their sparse vectors
`{token_id: weight}`. On the VM side, `scripts/pipeline/embed_fermi.py --encoder-url ...`
consumes this API and writes the vectors into Milvus.

```
Mac (MPS)                          VM
┌─────────────────┐   tunnel   ┌──────────────────────────┐
│ fermi_server.py │ <───────── │ embed_fermi.py            │
│  POST /encode   │  texts →   │  reads the chunk cache    │
│  → sparse vecs  │  ← sparse  │  → upsert Milvus (sparse) │
└─────────────────┘            └──────────────────────────┘
```

## 1. Get this folder onto the Mac

```bash
scp -r <vm>:/home/cleyrop/projects/fragus/fermi_server .
# or, if the Mac has a copy of the repo:
git pull && cd fermi_server
```

## 2. Python environment (Mac)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers huggingface_hub
export HF_TOKEN=hf_xxx      # HF token of an account that accepted access to the gated model
```

## 3. Start the server (MPS GPU enabled)

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python fermi_server.py
```

On first launch it downloads the model (~440 MB), then prints:

```
Listening on http://127.0.0.1:8000  (device: mps)
```

⚠️ Check `device: mps` (not `cpu`). Options: `--host`, `--port` (default
`127.0.0.1:8000`).

Local test:

```bash
curl -s localhost:8000/health
# {"status":"ok","device":"mps","model":"atomic-canyon/fermi-1024"}

curl -s -XPOST localhost:8000/encode -H 'content-type: application/json' \
     -d '{"texts":["criticality safety of the spent fuel cask"]}' | head -c 300
```

## 4. VM → Mac tunnel (reverse SSH)

The VM must be able to reach the server. **Reverse SSH tunnel started from the Mac**:
`localhost:8000` on the VM then points to the Mac server.

```bash
ssh -N -R 8000:localhost:8000 <user>@<vm-host>
```

Keep this connection open for the duration of the run. Add
`-o ServerAliveInterval=30 -o ServerAliveCountInterval=3` to prevent the
session from dropping on inactivity (each disconnect interrupts the run; resume
afterwards with `--skip-existing`).

## 5. Run the embedding (VM side)

```bash
.venv/bin/python scripts/pipeline/embed_fermi.py --reset \
    --encoder-url http://localhost:8000/encode --batch-size 8
```

(Smoke test first: add `--limit 50`.) Then cross-check the count with
`scripts/inspect/milvus_recap.py --collection documents_vectorises_fermi`.

## API

| Method  | Route      | Body / response |
|---------|------------|-----------------|
| `GET`   | `/health`  | `{"status","device","model"}` |
| `POST`  | `/encode`  | req `{"texts":[...], "max_seq_len":1024}` → `{"sparse":[{token_id: weight}, ...]}` |
