# Chunking

Parses the PDFs through docling-serve and uploads the JSONL chunks to Cleyrop (`corpus_chunks/`).
Local manifests (`output/manifests/`) act as a state cache for resuming.

## Prerequisites

docling-serve must be reachable. If the service runs on the Mac, open the SSH tunnel from the VM:

```bash
ssh -R 5001:localhost:5001 cleyrop-asnr-secnum-devspace
```

`DOCLING_SERVICE_URL` must be set in `.env`:

```
DOCLING_SERVICE_URL=http://localhost:5001
```

## Chunking

```bash
# Show status (done / pending / error / timeout) + timing stats
uv run scripts/pipeline/chunk_corpus.py --status

# Process the next N documents not yet processed
uv run scripts/pipeline/chunk_corpus.py --sample 10

# Full corpus with 4 parallel workers
uv run scripts/pipeline/chunk_corpus.py --workers 4

# Full corpus (sequential)
uv run scripts/pipeline/chunk_corpus.py

# Minimal corpus only (files flagged corpus_minimal in manifest.json)
uv run scripts/pipeline/chunk_corpus.py --corpus-minimal --workers 4

# Minimal corpus status
uv run scripts/pipeline/chunk_corpus.py --corpus-minimal --status
```

`Ctrl+C` interrupts cleanly — restarting resumes where it left off.  
`--force` re-chunks documents already processed.  
`--status` also shows an estimate of the remaining time depending on the number of workers.

## Exploring the chunks

```bash
# Global statistics
uv run scripts/inspect/explore_chunks.py

# Random document, middle chunks
uv run scripts/inspect/explore_chunks.py --random --middle

# Search by file name
uv run scripts/inspect/explore_chunks.py --name "report" --limit 5

# Chunks of a specific document
uv run scripts/inspect/explore_chunks.py --source-id <file_id>
```

## HTML visualization of the chunks

```bash
# N random documents → one self-contained HTML file (collapsible chunks + context)
uv run scripts/inspect/visualize_chunks.py --n 5 --out output/chunks_preview.html

# Chunks of a specific document, as HTML cards
uv run scripts/inspect/view_chunks.py --name "report"
uv run scripts/inspect/view_chunks.py --id <file_id>
```

## Global stats over all chunks

```bash
uv run scripts/inspect/chunk_stats.py
```
