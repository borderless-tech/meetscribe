"""Tests for collapsing ASR decoder-loop echoes (echo.py).

A decoder loop emits the same word back-to-back with a near-zero inter-token gap;
a genuine repetition has a plausible gap. The gap — not duration — is the discriminator.
"""

from __future__ import annotations

from meetscribe.echo import collapse_echoes
from meetscribe.types import Word


def test_clean_loop_collapses_to_first() -> None:
    # Four "und," in a row at gap ~0 → keep only the first, remove 3.
    words = [
        Word("und,", 1.00, 1.10),
        Word("und,", 1.10, 1.20),
        Word("und,", 1.20, 1.30),
        Word("und,", 1.30, 1.40),
    ]
    cleaned, removed = collapse_echoes(words)
    assert removed == 3
    assert cleaned == [Word("und,", 1.00, 1.10)]


def test_genuine_repetition_kept() -> None:
    # "ja, ja" with a plausible gap (0.24 s) is real speech, not a loop.
    words = [
        Word("ja", 0.50, 0.70),
        Word("ja", 0.94, 1.14),
    ]
    cleaned, removed = collapse_echoes(words)
    assert removed == 0
    assert cleaned == words


def test_mixed_sequence() -> None:
    # Loop of "the" (gap 0) collapses; distinct words untouched; a real "so, so"
    # (gap 0.30) is kept.
    words = [
        Word("hello", 0.00, 0.20),
        Word("the", 0.20, 0.30),
        Word("the", 0.30, 0.40),
        Word("the", 0.40, 0.50),
        Word("world", 0.50, 0.80),
        Word("so", 1.00, 1.20),
        Word("so", 1.50, 1.70),
    ]
    cleaned, removed = collapse_echoes(words)
    assert removed == 2
    assert cleaned == [
        Word("hello", 0.00, 0.20),
        Word("the", 0.20, 0.30),
        Word("world", 0.50, 0.80),
        Word("so", 1.00, 1.20),
        Word("so", 1.50, 1.70),
    ]


def test_empty_list() -> None:
    assert collapse_echoes([]) == ([], 0)


def test_single_word() -> None:
    words = [Word("solo", 0.00, 0.30)]
    cleaned, removed = collapse_echoes(words)
    assert removed == 0
    assert cleaned == words


def test_run_broken_by_different_word() -> None:
    # "a a b a" all at gap 0: the "b" breaks the run, so the trailing "a" is a new
    # run of length 1 and is kept. Only the second "a" (before "b") is removed.
    words = [
        Word("a", 0.00, 0.10),
        Word("a", 0.10, 0.20),
        Word("b", 0.20, 0.30),
        Word("a", 0.30, 0.40),
    ]
    cleaned, removed = collapse_echoes(words)
    assert removed == 1
    assert cleaned == [
        Word("a", 0.00, 0.10),
        Word("b", 0.20, 0.30),
        Word("a", 0.30, 0.40),
    ]


def test_case_insensitive_match() -> None:
    # "Und" / "und" are the same word for loop-detection; the first occurrence's
    # casing is preserved.
    words = [
        Word("Und", 0.00, 0.10),
        Word("und", 0.10, 0.20),
        Word("UND", 0.20, 0.30),
    ]
    cleaned, removed = collapse_echoes(words)
    assert removed == 2
    assert cleaned == [Word("Und", 0.00, 0.10)]
