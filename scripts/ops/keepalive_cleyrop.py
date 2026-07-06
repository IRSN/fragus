"""Keep the Cleyrop token fresh by making a lightweight call every ~4 minutes.

While this script is running, the SDK automatically refreshes the access_token
before it expires (5 min), which also extends the refresh_token (rolling refresh).
Avoids interactive re-logins during long pipelines.

Usage:
    setsid nohup uv run scripts/ops/keepalive_cleyrop.py > output/logs/keepalive.log 2>&1 < /dev/null & disown

Stop:
    pkill -f keepalive_cleyrop
"""

from __future__ import annotations

import logging
import os
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path

from cleyrop import CleyropClient, ClientConfig
from dotenv import load_dotenv

load_dotenv(override=True)

_root = Path(__file__).parent.parent.parent
with open(_root / "config.toml", "rb") as f:
    _config = tomllib.load(f)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PING_INTERVAL_S = 240  # 4 min < 5 min (access_token lifetime)


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


def main() -> None:
    project_id = os.environ.get("PROJECT_ID") or _config["project"]["id"]
    logger.info(f"Keepalive started, project_id={project_id}, interval={PING_INTERVAL_S}s")
    failures = 0
    while True:
        try:
            with get_client() as client:
                # Lightweight call that forces a refresh if needed
                client.get_project_contents(project_id, limit=1)
            logger.info("ping OK")
            failures = 0
        except Exception as e:
            failures += 1
            logger.error(f"ping FAILED (#{failures}): {e}")
            if failures >= 10:
                logger.error("10 consecutive failures, giving up.")
                sys.exit(1)
        time.sleep(PING_INTERVAL_S)


if __name__ == "__main__":
    main()
