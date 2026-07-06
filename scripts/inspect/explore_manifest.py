"""Generate a Markdown report of the corpus directory tree from manifest.json.

Local read only — no API calls.

Usage:
    uv run explore_manifest.py \\
        --manifest output/manifest.json \\
        --output output/tree.md \\
        [--depth 3]   # max displayed depth (default: 3)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path, PurePosixPath


# ─────────────────────────── helpers ─────────────────────────────────────────


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# ─────────────────────────── tree ────────────────────────────────────────────


class Node:
    def __init__(self, name: str) -> None:
        self.name = name
        self.children: dict[str, Node] = {}
        self.file_count = 0      # direct files
        self.file_size = 0       # direct file size
        self.total_count = 0     # recursive file count (computed afterwards)
        self.total_size = 0      # recursive size

    def add_file(self, size: int) -> None:
        self.file_count += 1
        self.file_size += size

    def get_or_create(self, name: str) -> "Node":
        if name not in self.children:
            self.children[name] = Node(name)
        return self.children[name]


def build_tree(files: list[dict]) -> Node:
    root = Node("/")
    for rec in files:
        parts = PurePosixPath(rec["path"]).parts  # ('/', 'Corpus root', ..., 'file.pdf')
        # Walk down to the parent folder
        node = root
        for part in parts[1:-1]:  # skip '/' and the file name
            node = node.get_or_create(part)
        node.add_file(rec["size"])
    return root


def propagate(node: Node) -> tuple[int, int]:
    """Recursively compute total_count and total_size."""
    node.total_count = node.file_count
    node.total_size = node.file_size
    for child in node.children.values():
        c, s = propagate(child)
        node.total_count += c
        node.total_size += s
    return node.total_count, node.total_size


# ─────────────────────────── Markdown rendering ──────────────────────────────


def render(node: Node, max_depth: int, lines: list[str], depth: int = 0) -> None:
    indent = "  " * depth
    # Sort: folders first (by file count desc), then by name
    children = sorted(
        node.children.values(),
        key=lambda n: (-n.total_count, n.name.lower()),
    )

    for child in children:
        marker = "📁"
        count_str = f"{child.total_count:,} files"
        size_str = human_size(child.total_size)

        # Flag files that live directly in this folder
        direct = f"  *(incl. {child.file_count:,} direct)*" if child.file_count and child.children else ""

        lines.append(f"{indent}- {marker} **{child.name}** — {count_str}, {size_str}{direct}")

        if depth + 1 < max_depth and child.children:
            render(child, max_depth, lines, depth + 1)
        elif child.children:
            # Summary at the depth limit
            nb_sub = len(child.children)
            lines.append(f"{indent}  *({nb_sub} subfolder{'s' if nb_sub > 1 else ''} not expanded)*")


# ─────────────────────────── main ────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Markdown report of the directory tree from manifest.json."
    )
    parser.add_argument(
        "--manifest",
        default="output/manifest.json",
        help="Path to manifest.json (default: output/manifest.json)",
    )
    parser.add_argument(
        "--output",
        default="output/tree.md",
        help="Output Markdown file (default: output/tree.md)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Max displayed depth (default: 3)",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    with manifest_path.open(encoding="utf-8") as f:
        m = json.load(f)

    files = m["files"]
    total_size = sum(r["size"] for r in files)
    generated_at = m.get("generated_at", "?")

    print(f"Loading: {len(files):,} files, {human_size(total_size)}")
    print("Building the tree…")

    root = build_tree(files)
    propagate(root)

    lines: list[str] = [
        f"# Corpus directory tree ({len(files):,} files — {human_size(total_size)})",
        f"\n*Generated from `{manifest_path}` (manifest dated {generated_at[:10]}). Displayed depth: {args.depth}.*\n",
    ]
    render(root, args.depth, lines)

    out_path = Path(args.output)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
