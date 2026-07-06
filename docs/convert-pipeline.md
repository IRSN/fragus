# PDF → DoclingDocument conversion pipeline

Step 1 of the RAG pipeline: converting the PDFs into DoclingDocument JSON via docling-serve,
stored in the `docling_docs/` folder on Cleyrop.

## Prerequisites

```bash
# .env
DOCLING_SERVICE_URL=http://localhost:5001
DOCLING_SERVICE_API_KEY=...   # optional
CLEYROP_TOKEN=...             # or CLEYROP_CLIENT_ID / CLEYROP_CLIENT_SECRET
```

## Conversion

```bash
# Test on a sample
uv run scripts/pipeline/convert_corpus.py --sample 10

# Test in parallel (2 workers recommended for 1 GPU)
uv run scripts/pipeline/convert_corpus.py --sample 20 --workers 2

# Minimal corpus only
uv run scripts/pipeline/convert_corpus.py --corpus-minimal --workers 2

# Whole corpus
uv run scripts/pipeline/convert_corpus.py --workers 2

# Re-convert files already processed
uv run scripts/pipeline/convert_corpus.py --sample 5 --force

# Show progress and remaining-time estimates
uv run scripts/pipeline/convert_corpus.py --status
```

Each document is processed in two phases:
1. Automatic OCR detection (`needs_ocr`) on the first 3 pages — disables OCR if the PDF is native
2. Conversion via `/v1/convert/file/async` → JSON uploaded to `docling_docs/{file_id}.json`

Local manifests live in `output/convert_manifests/`. Errors are listed in `output/conversion_errors.txt`.

## Visualization

```bash
# Original PDF (left) ↔ Docling parsing (right), self-contained HTML
uv run scripts/inspect/view_parsed.py --name "fragment"
uv run scripts/inspect/view_parsed.py --id <file_id>
```

The generated HTML is self-contained (base64-embedded images): source PDF panel
on the left, structured Docling output on the right.
