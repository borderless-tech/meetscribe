"""Flag likely-broken ASR words: tokens unknown to *both* a German and an English
spell-checker and absent from the glossary.

Bilingual because meetings mix German and English: a de-only checker over-flags valid
English (``LinkedIn``, ``meeting``). The spell-checkers are injected **batch** callables
(``list[str] -> set[str]`` returning the known subset) so one hunspell call covers a whole
utterance instead of a subprocess per word, and so unit tests need no hunspell binary.
Pure/deterministic; stdlib only.
"""

from __future__ import annotations

import os
import subprocess
from typing import Callable, Iterable

# Trailing/leading punctuation to peel off before lookup — a word wearing a comma
# ("Hallo,") is the same lexical item as the bare word.
_STRIP = ".,;:!?…\"'`´()[]{}«»„“”‚‘’-–—"

# Below this length a token is too ambiguous to judge (function words, initials).
_MIN_LEN = 3

BatchCheck = Callable[[list[str]], "set[str]"]


def _norm(word: str) -> str:
    return word.strip().strip(_STRIP)


def _wordlike(norm: str) -> bool:
    """A token worth judging: long enough, has letters, and is a plain word (letters +
    optional interior hyphen/apostrophe). Skips numbers, symbols, and ASR junk like
    ``M<unk>A`` / ``KI-Use`` mixes that no dictionary could sensibly correct."""
    if len(norm) < _MIN_LEN or not any(c.isalpha() for c in norm):
        return False
    return all(c.isalpha() or c in "-'" for c in norm)


class Lexicon:
    """Bilingual dictionary check backed by two injected batch spell-checkers + a glossary.

    ``spell_de`` / ``spell_en`` are callables ``list[str] -> set[str]`` (the known subset).
    ``glossary`` is domain names (people, products) no general dictionary knows — matched
    case-insensitively, and the main lever for flag precision (fewer false flags on names).
    """

    def __init__(self, spell_de: BatchCheck, spell_en: BatchCheck, glossary: Iterable[str]) -> None:
        self._spell_de = spell_de
        self._spell_en = spell_en
        self._glossary = {g.lower() for g in glossary}

    def is_known(self, word: str) -> bool:
        """True if either dictionary accepts the (stripped) token or it is in the glossary."""
        norm = _norm(word)
        if not norm:
            return True
        if norm.lower() in self._glossary:
            return True
        return bool(self._spell_de([norm]) or self._spell_en([norm]))

    def flag_broken(self, words: list[str]) -> list[int]:
        """Indices of wordlike tokens that no dictionary/glossary recognises. Batches the
        spell lookups (one call per language for the whole list)."""
        candidates = [(i, _norm(w)) for i, w in enumerate(words)]
        candidates = [(i, n) for i, n in candidates if _wordlike(n)]
        ask = [n for _, n in candidates]
        known = self._spell_de(ask) | self._spell_en(ask)
        return [i for i, n in candidates if n not in known and n.lower() not in self._glossary]


def build_hunspell_lexicon(dicpath: str, glossary: Iterable[str]) -> Lexicon:
    """Wire a real ``Lexicon`` backed by the ``hunspell`` CLI (de_DE + en_US), batched.

    Each checker shells out ONCE per call to ``hunspell -d <lang> -l`` (``DICPATH=<dicpath>``);
    ``-l`` lists the misspelled words, so the known subset is ``input − misspelled``. Not
    unit-tested (needs the binary + dictionaries), like the real LLM path elsewhere.
    """
    env = {**os.environ, "DICPATH": dicpath}

    def _checker(lang: str) -> BatchCheck:
        def check(words: list[str]) -> set[str]:
            if not words:
                return set()
            res = subprocess.run(
                ["hunspell", "-d", lang, "-l"],
                input="\n".join(words), capture_output=True, text=True, env=env,
            )
            misspelled = set(res.stdout.split())
            return set(words) - misspelled

        return check

    return Lexicon(_checker("de_DE"), _checker("en_US"), glossary)


def build_hunspell_suggester(dicpath: str) -> Callable[[list[str]], "dict[str, list[str]]"]:
    """A ``list[str] -> {word: candidates}`` function via one ``hunspell -a`` call (de+en).

    hunspell's ``-a`` (ispell pipe) mode prints ``& <word> <n> <off>: c1, c2, …`` per
    misspelled input word with its ranked corrections — free, instant, affix-expanded,
    bilingual. Batched (all words in one call). Not unit-tested (needs the binary + dicts).
    """
    env = {**os.environ, "DICPATH": dicpath}

    def suggest(words: list[str]) -> dict[str, list[str]]:
        if not words:
            return {}
        res = subprocess.run(
            ["hunspell", "-d", "de_DE,en_US", "-a"],
            input="\n".join(words), capture_output=True, text=True, env=env,
        )
        out: dict[str, list[str]] = {}
        for line in res.stdout.splitlines():
            if line.startswith("& ") and ":" in line:
                word = line.split()[1]
                out[word] = [c.strip() for c in line.split(":", 1)[1].split(",") if c.strip()]
        return out

    return suggest
