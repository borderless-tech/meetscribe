"""Flag likely-broken ASR words: tokens unknown to *both* a German and an English
spell-checker and absent from the glossary.

Bilingual because meetings mix German and English: a de-only checker over-flags valid
English (``LinkedIn``, ``meeting``) as errors. The spell-checkers are injected callables
(``word -> bool``) so unit tests need no hunspell binary — mirroring how the LLM boundary
is injected elsewhere. Pure/deterministic; stdlib only.
"""

from __future__ import annotations

import os
import subprocess
from typing import Callable, Iterable

# Trailing/leading punctuation to peel off before lookup — a word wearing a comma
# ("Hallo,") is the same lexical item as the bare word.
_STRIP = ".,;:!?…\"'`´()[]{}«»„“”‚‘’-–—"

# Below this length a token is too ambiguous to judge (function words, initials); skipping
# them keeps the flag list to genuinely suspect content words.
_MIN_LEN = 3


class Lexicon:
    """Bilingual dictionary check backed by two injected spell-checkers + a glossary.

    ``spell_de`` / ``spell_en`` are callables ``word -> bool``. ``glossary`` is a list of
    domain names (people, products) that no general dictionary knows; matched
    case-insensitively.
    """

    def __init__(
        self,
        spell_de: Callable[[str], bool],
        spell_en: Callable[[str], bool],
        glossary: Iterable[str],
    ) -> None:
        self._spell_de = spell_de
        self._spell_en = spell_en
        self._glossary = {g.lower() for g in glossary}

    def is_known(self, word: str) -> bool:
        """True if either dictionary accepts the (punctuation-stripped) token or it is in
        the glossary (case-insensitive)."""
        norm = word.strip().strip(_STRIP)
        if not norm:
            return True  # nothing lexical left to judge
        if norm.lower() in self._glossary:
            return True
        return self._spell_de(norm) or self._spell_en(norm)

    def flag_broken(self, words: list[str]) -> list[int]:
        """Indices of wordlike tokens that no dictionary/glossary recognises.

        Skips pure punctuation, pure numbers, and tokens shorter than ``_MIN_LEN``: those
        are not the transcription errors we hunt for and only add noise.
        """
        broken: list[int] = []
        for i, word in enumerate(words):
            norm = word.strip().strip(_STRIP)
            if len(norm) < _MIN_LEN:
                continue
            if not any(c.isalpha() for c in norm):  # pure numbers / symbols
                continue
            if not self.is_known(word):
                broken.append(i)
        return broken


def build_hunspell_lexicon(dicpath: str, glossary: Iterable[str]) -> Lexicon:
    """Wire a real ``Lexicon`` backed by the ``hunspell`` CLI (de_DE + en_US).

    Each checker shells out to ``hunspell -d <lang> -l`` with ``DICPATH=<dicpath>`` in the
    environment; ``-l`` lists the *misspelled* words on stdin, so a word is *known* exactly
    when that output is empty. Not unit-tested (like the real LLM path elsewhere) — it needs
    the hunspell binary and the dictionary files on disk.
    """
    env = {**os.environ, "DICPATH": dicpath}

    def _checker(lang: str) -> Callable[[str], bool]:
        def check(word: str) -> bool:
            res = subprocess.run(
                ["hunspell", "-d", lang, "-l"],
                input=word,
                capture_output=True,
                text=True,
                env=env,
            )
            return res.stdout.strip() == ""

        return check

    return Lexicon(
        spell_de=_checker("de_DE"),
        spell_en=_checker("en_US"),
        glossary=glossary,
    )
