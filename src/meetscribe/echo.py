"""Collapse ASR decoder-loop echoes.

A stuck decoder emits the *same* token back-to-back with a near-zero inter-token
gap (observed on real audio: four "und," in a row at gap ~0.000 s), whereas a
genuine repetition ("ja, ja") leaves a plausible pause between tokens. Duration is
not a reliable discriminator — the GAP is — so we key on ``next.start - prev.end``
and only fold runs of truly adjacent, case-insensitively-equal words down to their
first occurrence.
"""

from __future__ import annotations

from .types import Word


def collapse_echoes(
    words: list[Word], max_gap: float = 0.02
) -> tuple[list[Word], int]:
    """Fold each maximal run of consecutive equal words with gap <= ``max_gap``.

    Returns ``(cleaned_words, num_removed)``. A run is extended only while the next
    word matches (case-insensitively) *and* its gap from the previous word is within
    ``max_gap``; wider gaps break the run so genuine repetitions are preserved. The
    first occurrence keeps its original casing and timestamps.
    """
    if not words:
        return [], 0

    cleaned: list[Word] = [words[0]]
    removed = 0
    for prev, w in zip(words, words[1:]):
        # The loop signal is between *back-to-back emissions*, so the gap is always
        # measured against the immediately preceding original token — not the kept
        # anchor, whose ``end`` would drift as intervening echoes are dropped.
        same_word = w.w.casefold() == cleaned[-1].w.casefold()
        if same_word and (w.start - prev.end) <= max_gap:
            removed += 1
        else:
            cleaned.append(w)
    return cleaned, removed
