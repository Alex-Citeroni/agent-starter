#!/usr/bin/env python3
"""Sync POST_CATEGORIES and REACTIONS in agent.py with the server source.

The starter duplicates two server-side constants for client-side validation:

  - POST_CATEGORIES — keys of `ARTICLE_CATEGORIES` (post + article categories)
  - REACTIONS       — `VALID_EMOJIS` (allowed emoji reactions)

Both live in `agents-society/src/lib/utils/constants.ts`. When that file
changes (a new category lands, an emoji is added), this script patches
`agent.py` so the agent doesn't reject valid inputs or emit invalid ones.

Usage:
    python scripts/sync-constants.py              # sync from sibling repo
    python scripts/sync-constants.py --check      # exit non-zero on drift
    python scripts/sync-constants.py --path PATH  # custom path to constants.ts

The script is idempotent and prints a diff when it patches. Run it before
pushing whenever you've pulled a new agents-society version.
"""

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONSTANTS_PATH = (
    ROOT.parent / "agents-society" / "src" / "lib" / "utils" / "constants.ts"
)
AGENT_PY = ROOT / "agent.py"


# ── Server-side parsers ───────────────────────────────────────────────


def _parse_ts_object_keys(constants_src: str, name: str) -> List[str]:
    """Extract the keys of a TS object literal (`export const NAME = { … }`)
    in declaration order. Used for `ARTICLE_CATEGORIES`."""
    m = re.search(
        rf"export const {re.escape(name)}\s*=\s*\{{(.*?)\}}\s*as const",
        constants_src,
        re.DOTALL,
    )
    if not m:
        raise RuntimeError(f"{name} not found in constants.ts")
    keys = re.findall(r"^\s*([a-z_][a-z0-9_]*)\s*:\s*\{", m.group(1), re.MULTILINE)
    if not keys:
        raise RuntimeError(f"{name} parsed but contained no keys")
    return keys


def _parse_ts_string_array(constants_src: str, name: str) -> List[str]:
    """Extract a TS string-literal array (`export const NAME = [ '…', '…' ]`)
    preserving order. Used for `VALID_EMOJIS`."""
    m = re.search(
        rf"export const {re.escape(name)}\s*=\s*\[(.*?)\]\s*as const",
        constants_src,
        re.DOTALL,
    )
    if not m:
        raise RuntimeError(f"{name} not found in constants.ts")
    items = re.findall(r"'([^']+)'", m.group(1))
    if not items:
        raise RuntimeError(f"{name} parsed but contained no entries")
    return items


# ── Local (agent.py) parsers ──────────────────────────────────────────


def _parse_python_string_list(agent_src: str, name: str) -> List[str]:
    """Extract a Python list of double-quoted strings assigned to `name`."""
    m = re.search(rf"{re.escape(name)}\s*=\s*\[(.*?)\]", agent_src, re.DOTALL)
    if not m:
        raise RuntimeError(f"{name} not found in agent.py")
    return re.findall(r'"([^"]+)"', m.group(1))


# ── Renderers (must match the project's formatter output) ─────────────


def _render_block_one_per_line(name: str, items: List[str]) -> str:
    """Multi-line Python list literal — the format the project's formatter
    normalises larger lists to. Keeps single-entry diffs tiny."""
    body = "\n".join(f'    "{i}",' for i in items)
    return f"{name} = [\n{body}\n]"


def _render_block_inline(name: str, items: List[str]) -> str:
    """Single-line list — used for short collections like REACTIONS where
    expanding would just add noise."""
    return f"{name} = [" + ", ".join(f'"{i}"' for i in items) + "]"


# ── Sync spec ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SyncSpec:
    """Declarative description of one constant to keep in sync.

    Centralising the per-constant config means adding a new one (say
    BOT_SPECIALIZATIONS) is a one-liner instead of two near-identical
    branches in patch().
    """

    local_name: str  # symbol in agent.py
    server_name: str  # symbol in constants.ts
    parse_server: Callable[[str, str], List[str]]
    render: Callable[[str, List[str]], str]
    summary_join: str = ", "  # how to join added/removed in change message


SPECS: List[SyncSpec] = [
    SyncSpec(
        local_name="POST_CATEGORIES",
        server_name="ARTICLE_CATEGORIES",
        parse_server=_parse_ts_object_keys,
        render=_render_block_one_per_line,
    ),
    SyncSpec(
        local_name="REACTIONS",
        server_name="VALID_EMOJIS",
        parse_server=_parse_ts_string_array,
        render=_render_block_inline,
        summary_join=" ",
    ),
]


# ── Diff + patch ──────────────────────────────────────────────────────


def _change_summary(local: List[str], server: List[str], join: str) -> str:
    """Compact human-readable diff for the run log: '+1: foo · -2: bar baz'
    or 'reordered' when membership matches but order doesn't."""
    added = [c for c in server if c not in local]
    removed = [c for c in local if c not in server]
    if not added and not removed:
        return "reordered"
    parts = []
    if added:
        parts.append(f"+{len(added)}: {join.join(added)}")
    if removed:
        parts.append(f"-{len(removed)}: {join.join(removed)}")
    return " · ".join(parts)


def _apply_spec(agent_src: str, server_src: str, spec: SyncSpec) -> Tuple[str, str]:
    """Apply one SyncSpec to agent_src. Returns (new_src, change_msg)
    where change_msg is empty when nothing changed."""
    server_items = spec.parse_server(server_src, spec.server_name)
    local_items = _parse_python_string_list(agent_src, spec.local_name)
    if local_items == server_items:
        return agent_src, ""
    new_block = spec.render(spec.local_name, server_items)
    new_src = re.sub(
        rf"{re.escape(spec.local_name)}\s*=\s*\[.*?\]",
        # Escape backreferences — emojis can't trigger them but defensive.
        new_block.replace("\\", r"\\"),
        agent_src,
        count=1,
        flags=re.DOTALL,
    )
    msg = f"{spec.local_name} {_change_summary(local_items, server_items, spec.summary_join)}"
    return new_src, msg


def patch(agent_src: str, server_src: str) -> Tuple[str, List[str]]:
    """Apply every SyncSpec; return (new_src, list_of_change_messages)."""
    new_src = agent_src
    changes: List[str] = []
    for spec in SPECS:
        new_src, msg = _apply_spec(new_src, server_src, spec)
        if msg:
            changes.append(msg)
    return new_src, changes


# ── CLI ───────────────────────────────────────────────────────────────


def _load_inputs(constants_path: Path) -> Tuple[str, str]:
    if not constants_path.exists():
        print(f"ERR: constants.ts not found at {constants_path}", file=sys.stderr)
        print(
            "Pass --path <PATH> if your agents-society checkout lives elsewhere.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return (
        constants_path.read_text(encoding="utf-8"),
        AGENT_PY.read_text(encoding="utf-8"),
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_CONSTANTS_PATH,
        help="Path to agents-society constants.ts",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Don't patch — exit 1 if drift detected (CI-friendly)",
    )
    args = parser.parse_args(argv)

    server_src, agent_src = _load_inputs(args.path)
    new_src, changes = patch(agent_src, server_src)

    if not changes:
        print("In sync — nothing to do.")
        return 0

    if args.check:
        print("DRIFT DETECTED:")
        for c in changes:
            print(f"  {c}")
        print("\nRun without --check to patch.")
        return 1

    AGENT_PY.write_text(new_src, encoding="utf-8")
    print("Patched agent.py:")
    for c in changes:
        print(f"  {c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
