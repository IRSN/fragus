"""Recover the results of orphaned docling-serve tasks (client-side timeout).

For each (task_id, filename) provided:
1. Checks that the task is in `success` state on the docling-serve side
2. GET /v1/result/{task_id} to fetch the DoclingDocument JSON
3. Finds the matching file_id in the fragus_clean manifest (via filename)
4. Uploads the JSON to Cleyrop (docling_docs folder) as {file_id}.json
5. Saves the local manifest (status=done)

Usage:
    uv run scripts/diagnostics/recover_orphan_tasks.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import tomllib
from datetime import datetime
from io import BytesIO
from pathlib import Path

import httpx
from cleyrop import CleyropClient, ClientConfig
from dotenv import load_dotenv

load_dotenv(override=True)

_root = Path(__file__).parent.parent.parent
_config_path = _root / "config.toml"
with open(_config_path, "rb") as f:
    _config = tomllib.load(f)

DOCLING_URL = os.environ["DOCLING_SERVICE_URL"]
DOCLING_API_KEY = os.environ.get("DOCLING_SERVICE_API_KEY")

# (task_id, filename) — fill in with your own orphaned tasks
# (fictional example values below)
ORPHAN_TASKS = [
    ("00000000-0000-0000-0000-000000000001", "report_A.pdf"),
    ("00000000-0000-0000-0000-000000000002", "technical_guide_B.pdf"),
    ("00000000-0000-0000-0000-000000000003", "standard_C.pdf"),
    ("00000000-0000-0000-0000-000000000004", "historical_report_D.pdf"),
]

MANIFEST_PATH = _root / "output" / "manifest_fragus_clean.json"
MANIFESTS_DIR = _root / "output" / "convert_manifests"
DOCLING_DOCS_FOLDER_NAME = "docling_docs"


def get_cleyrop_client() -> CleyropClient:
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


def get_project_id() -> str:
    return os.environ.get("PROJECT_ID") or _config["project"]["id"]


def get_docling_docs_folder_id(client: CleyropClient, project_id: str) -> str:
    contents = client.get_project_contents(project_id)
    for f in contents.folders:
        if f.name == DOCLING_DOCS_FOLDER_NAME:
            return str(f.id)
    raise RuntimeError(f"Folder {DOCLING_DOCS_FOLDER_NAME} not found")


def find_file_id_by_name(filename: str) -> tuple[str, str] | None:
    """Returns (file_id, path) by matching filename in manifest."""
    data = json.loads(MANIFEST_PATH.read_text("utf-8"))
    for entry in data["files"]:
        if entry["name"] == filename:
            return entry["id"], entry["path"]
    return None


def poll_task_status(task_id: str) -> dict | None:
    """Returns None if task not found (404)."""
    headers = {"Authorization": f"Bearer {DOCLING_API_KEY}"} if DOCLING_API_KEY else {}
    r = httpx.get(
        f"{DOCLING_URL}/v1/status/poll/{task_id}",
        params={"wait": 0},
        headers=headers,
        timeout=30,
    )
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def fetch_result(task_id: str) -> dict:
    headers = {"Authorization": f"Bearer {DOCLING_API_KEY}"} if DOCLING_API_KEY else {}
    r = httpx.get(
        f"{DOCLING_URL}/v1/result/{task_id}",
        headers=headers,
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def recover_one(
    task_id: str,
    filename: str,
    client: CleyropClient,
    project_id: str,
    folder_id: str,
) -> str:
    """Returns one of: 'done', 'pending', 'error', 'not_in_manifest'."""
    match = find_file_id_by_name(filename)
    if not match:
        print(f"  ⚠ {filename} not found in manifest")
        return "not_in_manifest"
    file_id, path = match

    local_manifest = MANIFESTS_DIR / f"{file_id}.json"
    if local_manifest.exists():
        existing = json.loads(local_manifest.read_text("utf-8"))
        if existing.get("status") == "done":
            print(f"  ✓ already done: {filename}")
            return "done"

    status = poll_task_status(task_id)
    if status is None:
        print(f"  ✗ {filename}: task gone on the docling-serve side (to reprocess in pass 3)")
        return "lost"
    s = status.get("task_status")
    if s != "success":
        print(f"  ⏳ {filename}: task_status={s} (pos={status.get('task_position')})")
        return "pending"

    print(f"  📥 Recovering {filename} (task={task_id[:8]})...", end=" ", flush=True)
    result = fetch_result(task_id)
    doc_json = result["document"]["json_content"]
    json_bytes = json.dumps(doc_json, ensure_ascii=False).encode("utf-8")
    num_pages = len(doc_json.get("pages", {}))
    json_kb = len(json_bytes) / 1024
    processing_time = result.get("processing_time", 0)
    print(f"{num_pages}p, {json_kb:.0f}KB, processing_time={processing_time:.1f}s")

    upload_resp = client.upload_file(
        BytesIO(json_bytes),
        project_id=project_id,
        folder_id=folder_id,
        filename=f"{file_id}.json",
    )

    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    local_manifest.write_text(
        json.dumps({
            "file_id": file_id,
            "name": filename,
            "path": path,
            "status": "done",
            "converted_at": datetime.now().isoformat(),
            "num_pages": num_pages,
            "json_size_bytes": len(json_bytes),
            "elapsed_s": round(processing_time, 1),
            "cleyrop_json_file_id": str(upload_resp.id),
            "recovered_from_orphan_task": task_id,
        }, indent=2, ensure_ascii=False),
        "utf-8",
    )
    return "done"


def main() -> None:
    project_id = get_project_id()
    with get_cleyrop_client() as client:
        folder_id = get_docling_docs_folder_id(client, project_id)

        # Loop until all tasks are recovered or abandoned
        remaining = list(ORPHAN_TASKS)
        attempt = 0
        while remaining:
            attempt += 1
            print(f"\n=== Attempt {attempt} ({len(remaining)} tasks remaining) ===")
            still_pending = []
            for task_id, filename in remaining:
                result = recover_one(task_id, filename, client, project_id, folder_id)
                if result == "pending":
                    still_pending.append((task_id, filename))
                # 'lost', 'done', 'not_in_manifest' = abandoned
            remaining = still_pending
            if remaining:
                wait_s = 60
                print(f"  ⏰ {len(remaining)} still pending, sleeping {wait_s}s")
                time.sleep(wait_s)

        print("\n✓ Recovery complete.")


if __name__ == "__main__":
    main()
