"""Proactive Cleyrop token refresh daemon.

Refreshes every 3 min via a direct POST to Keycloak. Writes the new
tokens.json under a file lock (compatible with the Cleyrop SDK). As long as
the refresh window (1800s rolling) is not exceeded between two ticks, the
token stays alive indefinitely.

Logs to stderr. Exits non-zero if refresh fails (window closed or network down).
"""
from __future__ import annotations

import fcntl
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

TOKENS = Path.home() / ".cleyrop" / "tokens.json"
LOCK = Path.home() / ".cleyrop" / "tokens.json.lock"
TOKEN_URL = os.environ.get(
    "REFRESH_TOKEN_URL",
    "https://auth.asnr-secnum.cleyrop.net/realms/cleyrop/protocol/openid-connect/token",
)
CLIENT_ID = os.environ.get("REFRESH_CLIENT_ID", "cleyrop-cli")
INTERVAL_S = int(os.environ.get("REFRESH_INTERVAL_S", "180"))


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def refresh_once() -> tuple[float, float] | None:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.touch(exist_ok=True)
    with open(LOCK, "r+") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            with open(TOKENS) as f:
                data = json.load(f)
            key = next(iter(data))
            entry = data[key]
            body = urllib.parse.urlencode({
                "grant_type": "refresh_token",
                "client_id": CLIENT_ID,
                "refresh_token": entry["refresh_token"],
            }).encode()
            req = urllib.request.Request(
                TOKEN_URL, data=body,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            with urllib.request.urlopen(req, timeout=20) as r:
                tok = json.loads(r.read())
            now = time.time()
            entry["access_token"] = tok["access_token"]
            entry["refresh_token"] = tok.get("refresh_token", entry["refresh_token"])
            entry["expires_at"] = now + float(tok["expires_in"])
            entry["refresh_expires_at"] = now + float(tok.get("refresh_expires_in", 1800))
            data[key] = entry
            tmp = TOKENS.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(TOKENS)
            return entry["expires_at"], entry["refresh_expires_at"]
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def main() -> None:
    log(f"daemon start — refresh every {INTERVAL_S}s")
    fail_count = 0
    while True:
        try:
            result = refresh_once()
            if result is None:
                log("refresh returned None")
                fail_count += 1
            else:
                ae, re_ = result
                log(
                    "refresh ok — access_exp="
                    f"{datetime.fromtimestamp(ae).strftime('%H:%M:%S')}"
                    f" refresh_exp={datetime.fromtimestamp(re_).strftime('%H:%M:%S')}"
                )
                fail_count = 0
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:200] if hasattr(e, "read") else ""
            log(f"REFRESH FAIL HTTP {e.code}: {body}")
            fail_count += 1
        except Exception as e:  # noqa: BLE001
            log(f"REFRESH FAIL: {type(e).__name__}: {e}")
            fail_count += 1

        if fail_count >= 3:
            log("3 consecutive failures — exit (re-login required)")
            sys.exit(1)
        time.sleep(INTERVAL_S)


if __name__ == "__main__":
    main()
