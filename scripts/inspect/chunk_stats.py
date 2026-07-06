"""Global statistics over all produced chunks.

Downloads every JSONL referenced in chunk_manifests (status=done) from
Cleyrop and computes the token, page and chunks-per-doc distributions.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tomllib
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from cleyrop import CleyropClient, ClientConfig

load_dotenv(override=True)

_config_path = Path(__file__).parent.parent.parent / "config.toml"
with open(_config_path, "rb") as f:
    _config = tomllib.load(f)


def make_cleyrop_config() -> ClientConfig:
    domain = os.environ.get("CLEYROP_DOMAIN") or _config["cleyrop"].get("domain")
    if domain:
        os.environ.setdefault("CLEYROP_DOMAIN", domain)
        return ClientConfig.from_env()
    return ClientConfig.internal(
        client_id=os.environ.get("CLEYROP_CLIENT_ID", "cleyrop-cli"),
        client_secret=os.environ.get("CLEYROP_CLIENT_SECRET"),
        token=os.environ.get("CLEYROP_TOKEN"),
    )


def quantile(sorted_vals: list[int], q: float) -> int:
    if not sorted_vals:
        return 0
    idx = int(round(q * (len(sorted_vals) - 1)))
    return sorted_vals[idx]


def histogram(values: list[int], bins: list[tuple[int, int | None]]) -> list[tuple[str, int, float]]:
    n = len(values)
    out = []
    for lo, hi in bins:
        if hi is None:
            count = sum(1 for v in values if v >= lo)
            label = f">= {lo}"
        else:
            count = sum(1 for v in values if lo <= v < hi)
            label = f"{lo}–{hi-1}"
        out.append((label, count, count / n * 100 if n else 0))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="output/all_chunks_cache.jsonl",
                        help="Local cache to avoid re-downloading")
    parser.add_argument("--refresh", action="store_true", help="Force re-download")
    args = parser.parse_args()

    root = Path(__file__).parent.parent.parent
    manifests_dir = root / "output" / "chunk_manifests"
    cache_path = root / args.cache

    # 1. List the done manifests
    done = []
    for p in manifests_dir.glob("*.json"):
        d = json.loads(p.read_text("utf-8"))
        if d.get("status") == "done" and d.get("cleyrop_jsonl_file_id"):
            done.append(d)
    print(f"{len(done)} chunked documents to analyse")

    # 2. Load from cache or download
    all_chunks: list[dict] = []
    if cache_path.exists() and not args.refresh:
        print(f"Cache found: {cache_path}")
        for line in cache_path.read_text("utf-8").splitlines():
            if line.strip():
                all_chunks.append(json.loads(line))
        print(f"  ↳ {len(all_chunks)} chunks loaded from cache")
    else:
        print("Downloading from Cleyrop...")
        cleyrop_cfg = make_cleyrop_config()
        from cleyrop.exceptions import CleyropError
        skipped = []
        with CleyropClient(cleyrop_cfg) as cleyrop:
            if cleyrop_cfg.client_secret:
                cleyrop.login_client_credentials()
            for i, d in enumerate(done, 1):
                try:
                    jsonl_bytes = cleyrop.download_file_bytes(d["cleyrop_jsonl_file_id"])
                except CleyropError as e:
                    skipped.append((d["name"], str(e).split(":")[-1][:80].strip()))
                    continue
                for line in jsonl_bytes.decode("utf-8").splitlines():
                    if line.strip():
                        all_chunks.append(json.loads(line))
                if i % 25 == 0:
                    print(f"  {i}/{len(done)} docs · {len(all_chunks)} chunks total · {len(skipped)} skipped")
        if skipped:
            print(f"\n⚠ {len(skipped)} JSONL files not downloadable (likely AV pending):")
            for name, err in skipped[:5]:
                print(f"    - {name}: {err}")
            if len(skipped) > 5:
                print(f"    … +{len(skipped)-5} more")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as f:
            for c in all_chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"Cache written: {cache_path}")

    # 3. Stats
    tokens = sorted(c["docling_meta"]["num_tokens"] for c in all_chunks if c.get("docling_meta"))
    chars = sorted(len(c["text"]) for c in all_chunks)
    chunks_per_doc = Counter(c["file_id"] for c in all_chunks)
    pages_lists = [c.get("page_numbers") or [] for c in all_chunks]
    pages_per_chunk = sorted(len(p) for p in pages_lists)
    has_heading = sum(1 for c in all_chunks if c.get("docling_meta", {}).get("headings"))

    print(f"\n══════════════════════════════════════════════════════════════")
    print(f"GLOBAL STATS — {len(all_chunks)} chunks · {len(done)} docs")
    print(f"══════════════════════════════════════════════════════════════")

    print(f"\n┌─ Tokens / chunk (bge-m3 tokenizer) ───────────────────────────")
    print(f"│  min          : {min(tokens)}")
    print(f"│  p10          : {quantile(tokens, 0.10)}")
    print(f"│  p25          : {quantile(tokens, 0.25)}")
    print(f"│  median (p50) : {quantile(tokens, 0.50)}")
    print(f"│  p75          : {quantile(tokens, 0.75)}")
    print(f"│  p90          : {quantile(tokens, 0.90)}")
    print(f"│  p95          : {quantile(tokens, 0.95)}")
    print(f"│  p99          : {quantile(tokens, 0.99)}")
    print(f"│  max          : {max(tokens)}")
    print(f"│  mean         : {statistics.mean(tokens):.1f}")
    print(f"│  std dev      : {statistics.stdev(tokens):.1f}")
    print(f"└──────────────────────────────────────────────────────────────")

    print(f"\n┌─ Token distribution (bins) ───────────────────────────────────")
    bins = [(0, 50), (50, 100), (100, 256), (256, 512), (512, 768), (768, 1024), (1024, None)]
    hist = histogram(tokens, bins)
    for label, count, pct in hist:
        bar = "█" * int(pct / 1.5)
        print(f"│  {label:>12s}  {count:>6d}  {pct:5.1f}%  {bar}")
    print(f"└──────────────────────────────────────────────────────────────")

    over_max = sum(1 for t in tokens if t > 512)
    over_cap = sum(1 for t in tokens if t > 1024)
    print(f"\n┌─ Threshold compliance ────────────────────────────────────────")
    print(f"│  > 512 tok (target max_tokens) : {over_max} ({over_max/len(tokens)*100:.1f}%)")
    print(f"│  > 1024 tok (hard_cap)         : {over_cap} ({over_cap/len(tokens)*100:.1f}%)")
    print(f"└──────────────────────────────────────────────────────────────")

    print(f"\n┌─ Chunks / document ───────────────────────────────────────────")
    cpd = sorted(chunks_per_doc.values())
    print(f"│  min / median / mean / max : {min(cpd)} / {quantile(cpd, 0.5)} / {statistics.mean(cpd):.1f} / {max(cpd)}")
    print(f"│  docs with a single chunk  : {sum(1 for v in cpd if v == 1)}")
    print(f"│  docs with >= 200 chunks   : {sum(1 for v in cpd if v >= 200)}")
    print(f"└──────────────────────────────────────────────────────────────")

    print(f"\n┌─ Characters / chunk ──────────────────────────────────────────")
    print(f"│  min / median / mean / max : {min(chars)} / {quantile(chars, 0.5)} / {statistics.mean(chars):.0f} / {max(chars)}")
    print(f"└──────────────────────────────────────────────────────────────")

    print(f"\n┌─ Pages covered by a chunk ────────────────────────────────────")
    print(f"│  min / median / mean / max : {min(pages_per_chunk)} / {quantile(pages_per_chunk, 0.5)} / {statistics.mean(pages_per_chunk):.2f} / {max(pages_per_chunk)}")
    print(f"│  multi-page chunks         : {sum(1 for n in pages_per_chunk if n > 1)} ({sum(1 for n in pages_per_chunk if n > 1)/len(pages_per_chunk)*100:.1f}%)")
    print(f"└──────────────────────────────────────────────────────────────")

    print(f"\n┌─ Contextualization ───────────────────────────────────────────")
    print(f"│  chunks with heading(s)    : {has_heading} ({has_heading/len(all_chunks)*100:.1f}%)")
    print(f"│  root chunks (no heading)  : {len(all_chunks) - has_heading}")
    print(f"└──────────────────────────────────────────────────────────────")

    print(f"\nTotal retrieval corpus: {sum(tokens):,} tokens · {sum(chars):,} characters")


if __name__ == "__main__":
    main()
