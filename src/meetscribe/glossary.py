"""Persistent known-terms glossary (org/product/people names) feeding the LLM cleanup.

One term per line; blank lines and ``#`` comments are ignored. :func:`append` is a
case-insensitive, order-preserving dedup so recurring names accumulate across meetings
without ever duplicating. The file lives next to the config file, under
``config.config_home()`` (``$XDG_CONFIG_HOME``, else ``~/.config``).
"""

from __future__ import annotations

from pathlib import Path


def default_path() -> Path:
    """``config_home()/meetscribe/glossary.txt`` — always the config file's directory."""
    from . import config

    return config.config_home() / "meetscribe" / "glossary.txt"


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


def base() -> list[str]:
    """The shipped base glossary (common tech/office vocabulary)."""
    from .glossary_base import BASE_TERMS

    return list(BASE_TERMS)


def effective(path: str | Path | None = None) -> list[str]:
    """Base glossary ∪ the user's persistent glossary (case-insensitive dedup, base first).

    This is what the flagger should use — the base list pre-empts common tech/office false
    positives, the user file adds org/meeting-specific names.
    """
    user = load(path if path is not None else default_path())
    seen: set[str] = set()
    out: list[str] = []
    for t in base() + user:
        k = t.casefold()
        if k not in seen:
            seen.add(k)
            out.append(t)
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
