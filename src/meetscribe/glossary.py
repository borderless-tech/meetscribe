"""Persistent known-terms glossary (org/product/people names) feeding the LLM cleanup.

One term per line; blank lines and ``#`` comments are ignored. :func:`append` is a
case-insensitive, order-preserving dedup so recurring names accumulate across meetings
without ever duplicating. The file lives under ``$XDG_CONFIG_HOME`` (else ``~/.config``).
"""

from __future__ import annotations

import os
from pathlib import Path


def default_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path.home() / ".config"
    return root / "meetscribe" / "glossary.txt"


def load(path: str | Path) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    out: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


def append(path: str | Path, terms: list[str]) -> list[str]:
    """Append ``terms`` (case-insensitive dedup vs existing) and return the merged list."""
    p = Path(path)
    existing = load(p)
    seen = {t.casefold() for t in existing}
    added: list[str] = []
    for raw in terms:
        t = raw.strip()
        if t and t.casefold() not in seen:
            seen.add(t.casefold())
            added.append(t)
    if added:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            for t in added:
                f.write(t + "\n")
    return existing + added
