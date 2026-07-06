"""Minimal LLM call test through the Cleyrop proxy.

Usage:
    uv sync
    uv run scripts/ops/test_llm.py

Configuration: config.toml, section [llm]
"""

import json
import sys
import tomllib
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen

from langchain_openai import ChatOpenAI

_config_path = Path(__file__).parent.parent.parent / "config.toml"
with open(_config_path, "rb") as f:
    _config = tomllib.load(f)["llm"]


def _fetch_models(proxy_prefix: str) -> dict[str, dict]:
    url = f"{proxy_prefix}/llm/models"
    req = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except (URLError, HTTPError, TimeoutError) as exc:
        print(f"[ERROR] Cannot reach the proxy: {exc}")
        print(f"        URL attempted: {url}")
        sys.exit(1)

    models = {}
    for item in data if isinstance(data, list) else []:
        key = item.get("uniq_model_name") or item.get("model_key") or item.get("id")
        name = item.get("model_name") or item.get("name")
        endpoint = item.get("llm_endpoint") or item.get("endpoint")

        if not key or not name or not endpoint:
            continue

        parsed = urlparse(endpoint)
        base = urlparse(proxy_prefix)
        if not parsed.scheme and not parsed.netloc:
            path = endpoint if endpoint.startswith("/") else f"/{endpoint}"
            endpoint = urlunparse((base.scheme, base.netloc, path, "", "", ""))
        else:
            endpoint = urlunparse(
                (base.scheme, base.netloc, parsed.path, parsed.params, parsed.query, "")
            )

        models[str(key)] = {"name": str(name), "endpoint": endpoint}

    return models


def main() -> None:
    proxy_prefix = _config["proxy_prefix"]
    model_key = _config["model_key"]

    print(f"Proxy: {proxy_prefix}")
    print(f"Requested model: {model_key}\n")

    models = _fetch_models(proxy_prefix)
    if not models:
        print("[ERROR] No model returned by the proxy.")
        sys.exit(1)

    print(f"Available models ({len(models)}):")
    for k, v in models.items():
        marker = " <-- selected" if k == model_key else ""
        print(f"  {k:<35} {v['name']}{marker}")
    print()

    if model_key not in models:
        print(f"[ERROR] Model '{model_key}' not found.")
        sys.exit(1)

    model_cfg = models[model_key]

    llm = ChatOpenAI(
        base_url=model_cfg["endpoint"],
        model=model_cfg["name"],
        api_key="dummy",
        temperature=0.0,
        max_tokens=_config["max_tokens"],
        timeout=_config["timeout"],
        max_retries=0,
    )

    prompt = "Reply only with 'pong'."
    print(f"Sending prompt: {prompt!r}")
    response = llm.invoke(prompt)
    print(f"Response: {getattr(response, 'content', str(response))!r}")
    print("\n[OK] LLM call working.")


if __name__ == "__main__":
    main()
