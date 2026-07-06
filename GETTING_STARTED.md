# Getting Started

> This project runs on the Cleyrop platform (SecNumCloud). The steps below
> assume access to a Cleyrop devspace; outside that environment the scripts can
> be read as reference but will not run as-is.

## Prerequisites

- `uv` installed
- Access to the Cleyrop devspace

---

## 1. Install the Cleyrop CLI

```bash
uv tool install /home/cleyrop/projects/internal/packages/cleyrop-sdk/
export PATH="/home/cleyrop/.local/bin:$PATH"

# Make it permanent
echo 'export PATH="/home/cleyrop/.local/bin:$PATH"' >> ~/.bashrc
```

## 2. Log in

```bash
cleyrop context asnr-secnum.cleyrop.net
cleyrop login
```

The terminal shows a URL + a code. Open the URL in your browser and
authenticate. The terminal confirms:

```
Authenticated as: First Last (user@example.com)
```

The token is cached in `~/.cleyrop/tokens.json` (validity ~30 min, automatic
refresh).

## 3. Run

```bash
cd ~/projects/fragus
uv sync
uv run scripts/ops/browse.py
```

The Cleyrop project tree is displayed in the terminal.

---

## If the token expires

```bash
cleyrop login
```

That's all — nothing needs to be reconfigured.
