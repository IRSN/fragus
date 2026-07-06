# PDF → DoclingDocument conversion benchmark

**Date:** 2026-04-23  
**Corpus:** 29,882 PDFs (52.5 GB)  
**Parameters:** `table_mode=fast`, `pdf_backend=pypdfium2`, auto-detected OCR (RapidOCR), 300s timeout

## Methodology

Four runs over a total sample of 95 documents:

| Run | Workers | Docs | Wall time | Total GPU |
|-----|---------|------|-----------|-----------|
| 1   | 1       | 5    | 79s       | 79s       |
| 2   | 1       | 20   | 169s      | 169s      |
| 3   | 2       | 20   | 283s      | 509s      |
| 4   | 1       | 50   | 1244s     | 1239s     |

## Measured metrics

| Metric | Value |
|---|---|
| Share of OCR docs (estimated over the 75 docs of runs 1+2+4) | **~30%** |
| Average time — OCR docs | ~45 s/doc |
| Average time — non-OCR docs | ~10 s/doc |
| **Weighted global average (95 docs)** | **21.0 s/doc** |

Run 4 (50 docs) refined the estimate: it included heavy documents (130p/253s, 66p/124.5s), confirming that the initial average of 16.8s/doc was underestimated.

## Impact of parallelism (single GPU)

With 2 workers, the measured speedup is **1.80x** (509s GPU / 283s wall).

This gain exceeds the mere masking of network I/O (~1s/doc): docling-serve pipelines the CPU preprocessing (layout, block detection) of the next request while the GPU processes the current one.

Beyond 2 workers, gains become marginal — the GPU is the only bottleneck.

## Projections over the full corpus (29,787 remaining docs)

### Entire corpus

```
 1 worker  :  ~174h  (7.2 days)
 2 workers :   ~97h  (4.0 days)  ← recommended
```

### Excluding the MEDIATHEQUE folder (~22,763 remaining docs)

MEDIATHEQUE accounts for 7,046 PDFs / 23.6% of the corpus.

```
 1 worker  :  ~133h  (5.5 days)
 2 workers :   ~74h  (3.1 days)  ← recommended
```

**Recommended command:**
```bash
uv run scripts/pipeline/convert_corpus.py --workers 2
```

## Limits of the estimate

- High variance: the distribution is strongly bimodal (OCR ~45s/doc vs non-OCR ~10s/doc). The actual confidence interval is wide.
- A few large files (up to 294 MB) may hit the 300s timeout — monitor `output/conversion_errors.txt`.
- The 2-worker speedup is empirical over 20 docs; to be confirmed on a longer run.
