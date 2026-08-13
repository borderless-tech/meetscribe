"""Apply LLM corrections to the flagged broken words of an utterance, double-guarded.

A language model asked to fix a garbled ASR word will confidently invent a fluent
word that was never spoken. Two independent guards stop that: the candidate must
(1) *sound* like what the ASR heard — ``phonetic.acoustic_distance`` within
``max_distance`` — **and** (2) be a real word — ``is_known`` is True. Acoustic
closeness alone is not enough: a near-homophone non-word ("Ertier" → "Erter",
distance ~0.05) passes the sound test yet the dictionary vetoes it.

Only flagged indices are touched, and a fix replaces the word *text* alone —
the original ``start``/``end`` are carried over so the per-word timeline never
shifts. The corrector and dictionary are injected, so unit tests need no LLM or
hunspell.
"""

from __future__ import annotations

from typing import Callable

from . import phonetic
from .types import Word


def is_valid_correction(
    asr: str,
    candidate: str,
    max_distance: float,
    is_known: Callable[[str], bool],
) -> bool:
    """True iff ``candidate`` may replace ``asr``: close-sounding AND a real word.

    The tightened guard over bare acoustic distance: the dictionary check is what
    keeps an acoustically-close *non-word* from being applied.
    """
    if phonetic.acoustic_distance(asr, candidate) > max_distance:
        return False
    return is_known(candidate)


def repair_words(
    words: list[Word],
    flagged: list[int],
    correct_fn: Callable[[int, list[Word]], str],
    accept_fn: Callable[[str, str], bool],
) -> tuple[list[Word], int]:
    """Repair only ``flagged`` word indices; return ``(new_words, num_fixed)``.

    For each flagged ``i``: ask ``correct_fn(i, words)`` (the LLM span corrector,
    handed the position plus the full word list for context) for a candidate string.
    If ``accept_fn(words[i].w, candidate)`` is True, swap in a new ``Word`` with the
    candidate text but the *original* timestamps; otherwise keep the word unchanged.
    Non-flagged words and overall order are untouched.
    """
    new_words = list(words)
    num_fixed = 0
    for i in sorted(set(flagged)):  # dedup + ascending → deterministic corrector calls
        candidate = correct_fn(i, words)
        if accept_fn(words[i].w, candidate):
            new_words[i] = Word(candidate, words[i].start, words[i].end)
            num_fixed += 1
    return new_words, num_fixed
