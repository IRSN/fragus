"""Builds a manifest targeting fragus_corpus_clean/{baseline,documents_supplementaires}.

Recursively enumerates the PDFs of both folders and writes a manifest
in the format expected by convert_corpus.py.

Usage:
    uv run scripts/pipeline/build_manifest_fragus_clean.py [--output output/manifest_fragus_clean.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from cleyrop import CleyropClient, ClientConfig
from cleyrop.models import FileResponse, FolderResponse
from dotenv import load_dotenv

load_dotenv(override=True)

_config_path = Path(__file__).parent.parent.parent / "config.toml"
with open(_config_path, "rb") as f:
    _config = tomllib.load(f)


TARGET_TOP = "fragus_corpus_clean"
TARGET_SUBFOLDERS = ["baseline", "documents_supplementaires"]


def get_client() -> CleyropClient:
    domain = os.environ.get("CLEYROP_DOMAIN") or _config["cleyrop"].get("domain")
    if domain:
        os.environ.setdefault("CLEYROP_DOMAIN", domain)
        cfg = ClientConfig.from_env()
    else:
        cfg = ClientConfig.internal(
            client_id=os.environ.get("CLEYROP_CLIENT_ID", "cleyrop-cli"),
            client_secret=os.environ.get("CLEYROP_CLIENT_SECRET"),
            token=os.environ.get("CLEYROP_TOKEN"),
        )
    client = CleyropClient(cfg)
    if cfg.client_secret:
        client.login_client_credentials()
    return client


def resolve_project_id() -> str:
    if project_id := os.environ.get("PROJECT_ID"):
        return project_id
    return _config["project"]["id"]


def walk_folder(
    client: CleyropClient,
    project_id: str,
    folder_id: str,
    current_path: str,
) -> list[dict]:
    """Walk a folder recursively and return file records."""
    files: list[dict] = []
    for item in client.iter_project_contents(project_id, folder_id=folder_id):
        if isinstance(item, FileResponse):
            files.append({
                "id": str(item.id),
                "name": item.name,
                "path": f"{current_path}/{item.name}",
                "size": getattr(item, "size", 0) or 0,
                "mime_type": getattr(item, "mime_type", None),
            })
        elif isinstance(item, FolderResponse):
            sub_path = f"{current_path}/{item.name}"
            files.extend(walk_folder(client, project_id, str(item.id), sub_path))
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="output/manifest_fragus_clean.json",
        help="Output manifest path",
    )
    args = parser.parse_args()

    project_id = resolve_project_id()
    print(f"Project: {project_id}")
    print(f"Target : /{TARGET_TOP}/{{{','.join(TARGET_SUBFOLDERS)}}}")

    all_files: list[dict] = []
    with get_client() as client:
        # Find the fragus_corpus_clean root folder
        root_folder_id = None
        for item in client.iter_project_contents(project_id, folder_id=None):
            if isinstance(item, FolderResponse) and item.name == TARGET_TOP:
                root_folder_id = str(item.id)
                break
        if not root_folder_id:
            print(f"ERROR: folder {TARGET_TOP} not found", file=sys.stderr)
            sys.exit(1)

        # Find the target subfolders
        sub_ids: dict[str, str] = {}
        for item in client.iter_project_contents(project_id, folder_id=root_folder_id):
            if isinstance(item, FolderResponse) and item.name in TARGET_SUBFOLDERS:
                sub_ids[item.name] = str(item.id)
        missing = [n for n in TARGET_SUBFOLDERS if n not in sub_ids]
        if missing:
            print(f"ERROR: missing subfolders: {missing}", file=sys.stderr)
            sys.exit(1)

        for name in TARGET_SUBFOLDERS:
            base_path = f"/{TARGET_TOP}/{name}"
            print(f"  scanning {base_path}...", flush=True)
            recs = walk_folder(client, project_id, sub_ids[name], base_path)
            print(f"    → {len(recs)} files")
            all_files.extend(recs)

    # Filter PDFs
    pdfs = [f for f in all_files if (f.get("mime_type") or "").lower() == "application/pdf"]
    non_pdfs = len(all_files) - len(pdfs)
    print(f"\nTotal: {len(all_files)} files ({len(pdfs)} PDF, {non_pdfs} non-PDF)")

    # Flag corpus_minimal = True for documents_supplementaires
    for f in pdfs:
        f["corpus_minimal"] = "/documents_supplementaires/" in f["path"]

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stats": {
            "total_files_scanned": len(all_files),
            "total_pdfs": len(pdfs),
            "baseline": sum(1 for f in pdfs if "/baseline/" in f["path"]),
            "documents_supplementaires": sum(
                1 for f in pdfs if "/documents_supplementaires/" in f["path"]
            ),
        },
        "files": pdfs,
    }

    root = Path(__file__).parent.parent.parent
    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), "utf-8")
    print(f"\nManifest written: {output_path.resolve()}")
    print(f"  baseline                  : {output['stats']['baseline']}")
    print(f"  documents_supplementaires : {output['stats']['documents_supplementaires']}")


if __name__ == "__main__":
    main()
