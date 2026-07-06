# Collection `documents_vectorises_fermi` — sparse fermi-1024

Third Milvus collection on the **same ASNR corpus**, alongside the two dense bge-m3
collections (see [milvus-collection.md](milvus-collection.md)). It indexes the **same
docling chunks** as `documents_vectorises`, but with a **learned sparse** embedding instead
of the dense one — for a dense vs sparse A/B at constant chunking.

| | |
|---|---|
| **Model** | `atomic-canyon/fermi-1024` (SPLADE/LSR, English BERT fine-tuned on the nuclear domain) |
| **Type** | **sparse** vector `{token_id: weight}` over a 30,522 vocab |
| **Similarity** | dot product → Milvus metric **`IP`** |
| **Index** | `SPARSE_INVERTED_INDEX` |
| **Entities** | **29,358** (1:1 with the cached docling chunks, 0 excluded) |
| **Source chunks** | `output/all_chunks_cache.jsonl` (docling chunks reused as-is) |
| **Built on** | 2026-05-29 |

> ⚠️ **Known language mismatch**: fermi is **English**, while the corpus and queries are almost
> entirely French. This collection serves an exploratory A/B, not production French retrieval
> (see the multilingual/native-French choice made for the main collections).

## Schema

Same business fields as the bge-m3 collections, with the dense vector replaced by the sparse one:

| Field | Type | Notes |
|---|---|---|
| `chunk_id` | VARCHAR (PK) | `{file_id}_{index}` — identical to `documents_vectorises` |
| `file_id` | VARCHAR | document identifier |
| `name` | VARCHAR | source PDF name |
| `path` | VARCHAR | path within the corpus |
| `corpus_minimal` | BOOL | membership in the minimal corpus |
| `text` | VARCHAR | chunk text (with docling context header) |
| `sparse` | **SPARSE_FLOAT_VECTOR** | fermi, IP metric, SPARSE_INVERTED_INDEX index |

The `text`/token/chunks-per-doc distributions are **those of `documents_vectorises`**
(same chunks) — see milvus-collection.md.

## fermi inference recipe (sparse)

```python
output = model(**feature)[0]                                  # logits [B, L, vocab]
values, _ = torch.max(output * attention_mask.unsqueeze(-1), dim=1)  # masked max-pool
values = torch.log(1 + torch.relu(values))                    # activation
values[:, special_token_ids] = 0                              # zero out special tokens
# → per row: {int(token_id): float(weight)} over the non-zero indices
```

## Building

Inference offloaded to a **Mac M3 (MPS/GPU)** — the VM's CPU was too slow (~27 h estimated).
A small HTTP server encodes on the Mac, exposed to the VM through a **reverse SSH tunnel**; the VM
reads the chunks from the cache and writes the vectors into Milvus.

```bash
# Mac: inference server (see fermi_server/README.md)
PYTORCH_ENABLE_MPS_FALLBACK=1 python fermi_server.py          # device: mps
ssh -N -R 8000:localhost:8000 <user>@<vm-host>                # reverse SSH tunnel

# VM: remote encoding → Milvus (the Mac server is on localhost:8000)
.venv/bin/python scripts/pipeline/embed_fermi.py --reset \
    --encoder-url http://localhost:8000/encode --batch-size 8
# resume after an interruption (idempotent on the PK):
.venv/bin/python scripts/pipeline/embed_fermi.py --skip-existing --encoder-url … --batch-size 8
```

Entity count recap: `.venv/bin/python scripts/inspect/milvus_recap.py --collection documents_vectorises_fermi`.

## Querying (sparse)

```python
q_sparse = encode_via_fermi("radioactive waste management and decommissioning")
client.search(
    "documents_vectorises_fermi", data=[q_sparse], anns_field="sparse",
    search_params={"metric_type": "IP"}, limit=5, output_fields=["name", "text"],
)
```

Sanity check at build time (English query) → coherent top hits: *SSG-33* (decommissioning),
*SSG-65* (emergency preparedness), *SSG-26* (advisory material), clear IP scores.

## Pitfalls encountered (remote Mac/tunnel run)

- **SSH tunnel drop**: any interruption of the SSH session kills the running job → keep
  the connection alive (`-o ServerAliveInterval=30`); otherwise resume with `--skip-existing`.
- **Mac OOM on long chunks**: the `[batch, seq_len, 30522]` logits take ~8 GB transiently
  at batch 32 × 1024 tokens → **use `--batch-size 8`** (peak ÷4, throughput ~11–12 chunks/s).
- **Milvus memory quota** (multi-tenant cluster): writes rejected when too many collections
  are loaded in RAM → `release_collection` on a large unused collection (reversible).
- Robustness: `encode_sparse_remote` retries with backoff on tunnel drops/timeouts.
