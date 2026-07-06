"""Print the level-1 tree of the Cleyrop project.

Usage (in-cluster / devspace):
    uv sync
    uv run browse.py

Non-secret configuration: config.toml (versioned)
Secrets: .env (CLEYROP_CLIENT_SECRET, etc.)
"""

import tomllib
import os
from pathlib import Path

from cleyrop import ClientConfig, CleyropClient
from cleyrop.models import FileResponse, FolderResponse
from dotenv import load_dotenv

load_dotenv(override=True)

_config_path = Path(__file__).parent.parent.parent / "config.toml"
with open(_config_path, "rb") as f:
    _config = tomllib.load(f)


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


def resolve_project_id(client: CleyropClient) -> str:
    if project_id := os.environ.get("PROJECT_ID"):
        return project_id
    if project_id := _config["project"].get("id"):
        return project_id
    project_slug = os.environ["PROJECT_SLUG"]
    return str(client.get_project_by_slug(project_slug).id)


def print_tree_level1(client: CleyropClient, project_id: str) -> None:
    """List only the root level of the project (depth=0)."""
    for item in client.iter_project_contents(project_id, folder_id=None):
        if isinstance(item, FileResponse):
            size = getattr(item, "size", 0) or 0
            print(f"📄 {item.name}  ({size:,} B)")
        elif isinstance(item, FolderResponse):
            print(f"📁 {item.name}/")


with get_client() as client:
    project_id = resolve_project_id(client)
    print(f"Project id={project_id}\n")
    print_tree_level1(client, project_id)
