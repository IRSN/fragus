"""AV audit: virus_scan_status for every parsed JSON and chunk JSONL
referenced in the local manifests, with the age of any blocked file.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import threading
import tomllib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from cleyrop import CleyropClient, ClientConfig
from cleyrop.exceptions import CleyropError

load_dotenv(override=True)

_config_path = Path(__file__).parent.parent.parent / "config.toml"
with open(_config_path, "rb") as f:
    _config = tomllib.load(f)
os.environ.setdefault("CLEYROP_DOMAIN", _config["cleyrop"]["domain"])


_local = threading.local()


def _get_client(cfg: ClientConfig) -> CleyropClient:
    if not hasattr(_local, "client"):
        c = CleyropClient(cfg)
        if cfg.client_secret:
            c.login_client_credentials()
        c.__enter__()
        _local.client = c
    return _local.client


def fmt_age(dt: datetime | None, now: datetime) -> str:
    if dt is None:
        return "—"
    delta = now - dt
    days = delta.days
    hours = delta.seconds // 3600
    if days >= 1:
        return f"{days}d {hours}h"
    minutes = (delta.seconds % 3600) // 60
    if hours >= 1:
        return f"{hours}h {minutes}min"
    return f"{minutes}min"


def query_one(cfg: ClientConfig, file_id: str, label: str, name: str) -> dict:
    try:
        client = _get_client(cfg)
        r = client.get_file(file_id)
        return {
            "kind": label,
            "name": name,
            "file_id": str(file_id),
            "status": str(r.virus_scan_status),
            "created_at": r.created_at,
            "virus_scan_at": r.virus_scan_at,
            "size": r.size,
        }
    except CleyropError as e:
        return {"kind": label, "name": name, "file_id": str(file_id),
                "status": "404/err", "created_at": None, "virus_scan_at": None,
                "size": 0, "error": str(e)[:60]}


def main() -> None:
    root = Path(__file__).parent.parent.parent
    convert_dir = root / "output" / "convert_manifests"
    chunk_dir = root / "output" / "chunk_manifests"

    targets = []  # (label, file_id, name)
    for p in convert_dir.glob("*.json"):
        d = json.loads(p.read_text("utf-8"))
        if d.get("cleyrop_json_file_id"):
            targets.append(("docling_docs/JSON", d["cleyrop_json_file_id"], d["name"]))
    for p in chunk_dir.glob("*.json"):
        d = json.loads(p.read_text("utf-8"))
        if d.get("cleyrop_jsonl_file_id"):
            targets.append(("corpus_chunks/JSONL", d["cleyrop_jsonl_file_id"], d["name"]))

    print(f"AV audit over {len(targets)} files (parsed JSON + chunk JSONL)...")

    cfg = ClientConfig.from_env()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(query_one, cfg, fid, label, name) for label, fid, name in targets]
        for i, f in enumerate(concurrent.futures.as_completed(futures), 1):
            results.append(f.result())
            if i % 100 == 0:
                print(f"  {i}/{len(targets)}")

    now = datetime.now(timezone.utc)

    # Buckets
    buckets = Counter(r["status"] for r in results)
    print(f"\n══ Overall AV status ══")
    for k, v in buckets.most_common():
        pct = v / len(results) * 100
        print(f"  {k:<15s}  {v:>5d}  ({pct:5.1f}%)")

    # Per-kind breakdown
    print(f"\n══ By folder ══")
    by_kind: dict[str, Counter] = {}
    for r in results:
        by_kind.setdefault(r["kind"], Counter())[r["status"]] += 1
    for kind, c in by_kind.items():
        total = sum(c.values())
        print(f"  {kind} ({total}):")
        for k, v in c.most_common():
            print(f"      {k:<15s}  {v:>4d}  ({v/total*100:5.1f}%)")

    # Non-CLEAN files: list with age
    blocked = [r for r in results if r["status"] not in ("CLEAN", "VirusScanStatus.CLEAN")]
    if blocked:
        blocked.sort(key=lambda r: r.get("created_at") or now, reverse=False)
        print(f"\n══ {len(blocked)} non-CLEAN files — detail with age ══")
        print(f"{'Status':<10s} {'Kind':<22s} {'Doc':<42s} {'Uploaded':<18s} {'Age':<10s} {'Size'}")
        print("─" * 130)
        for r in blocked:
            age = fmt_age(r.get("created_at"), now)
            up = r["created_at"].strftime("%Y-%m-%d %H:%M") if r.get("created_at") else "—"
            size_kb = r.get("size", 0) / 1024
            print(f"{r['status']:<10s} {r['kind']:<22s} {r['name'][:40]:<42s} {up:<18s} {age:<10s} {size_kb:>7.0f} KB")

    # Age distribution of blocked files
    if blocked:
        ages_days = [(now - r["created_at"]).total_seconds() / 86400 for r in blocked if r.get("created_at")]
        bins = [(0, 0.01), (0.01, 1), (1, 3), (3, 7), (7, None)]
        print(f"\n══ Age distribution of non-CLEAN files ══")
        labels = ["< 15 min", "15 min – 1 d", "1–3 days", "3–7 days", "> 7 days"]
        for (lo, hi), label in zip(bins, labels):
            if hi is None:
                count = sum(1 for a in ages_days if a >= lo)
            else:
                count = sum(1 for a in ages_days if lo <= a < hi)
            print(f"  {label:<18s}  {count}")


if __name__ == "__main__":
    main()
