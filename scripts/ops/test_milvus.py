"""Minimal Milvus connection test.

Usage:
    uv sync
    uv run scripts/ops/test_milvus.py

Configuration: config.toml, section [milvus]
"""

import sys
import tomllib
from pathlib import Path

from pymilvus import MilvusClient

_config_path = Path(__file__).parent.parent.parent / "config.toml"
with open(_config_path, "rb") as f:
    _config = tomllib.load(f)["milvus"]


def main() -> None:
    uri = _config["uri"]
    collection = _config["collection"]

    print(f"URI     : {uri}")
    print(f"Target collection: {collection}\n")

    client = MilvusClient(uri=uri)

    # List available collections
    collections = client.list_collections()
    print(f"Available collections ({len(collections)}):")
    for col in collections:
        marker = " <-- target" if col == collection else ""
        print(f"  {col}{marker}")
    print()

    # Info on the target collection
    if collection not in collections:
        print(f"[WARNING] Collection '{collection}' is missing.")
        client.close()
        return

    stats = client.get_collection_stats(collection_name=collection)
    row_count = stats.get("row_count", "N/A")
    print(f"Entity count in '{collection}': {row_count}")

    # Test query: fetch 1 entity
    results = client.query(
        collection_name=collection,
        filter="",
        output_fields=["chunk_id"],
        limit=1,
    )
    if results:
        print(f"Sample entity: {results[0]}")

    print("\n[OK] Milvus connection working.")
    client.close()


if __name__ == "__main__":
    main()
