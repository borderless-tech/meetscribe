"""Acoustic-closeness guardrail for ASR word corrections.

When a garbled ASR word has several dictionary candidates, we must correct it
only to a word that plausibly *sounds* like what was heard — otherwise a
language model will happily "correct" garbage into a fluent but invented word
that was never spoken. We measure sound-alikeness two ways and combine them:

- **Kölner Phonetik** (the German counterpart to Soundex): maps a word to a
  digit code where letters that sound alike share a digit, so "Ertier" and
  "Erster" collapse toward the same code while unrelated words diverge. It is
  chosen over Soundex because it is designed for German phonology.
- **Levenshtein** on the raw letters, as a tiebreaker/back-off: the phonetic
  code alone is coarse (many words share a code), so orthographic edit distance
  keeps ranking sensible when codes tie or nearly tie.

Pure stdlib, deterministic. ``acoustic_distance`` is 0.0 for an exact match and
grows as words sound less alike; lower = closer.
"""

from __future__ import annotations

# Kölner Phonetik digit for each consonant that has a fixed, context-free code.
# Context-sensitive letters (C, D, T, P, X, H) are handled separately below.
_SIMPLE: dict[str, str] = {
    "a": "0", "e": "0", "i": "0", "j": "0", "o": "0", "u": "0", "y": "0",
    "ä": "0", "ö": "0", "ü": "0",
    "b": "1",
    "f": "3", "v": "3", "w": "3",
    "g": "4", "k": "4", "q": "4",
    "l": "5",
    "m": "6", "n": "6",
    "r": "7",
    "s": "8", "z": "8", "ß": "8",
}


def _normalize(word: str) -> str:
    """Lowercase and keep only alphabetic letters (incl. German umlauts/ß)."""
    return "".join(ch for ch in word.lower() if ch.isalpha())


def koelner_phonetik(word: str) -> str:
    """Return the standard German Kölner Phonetik code for ``word``.

    Three passes, per the published algorithm: (1) map each letter to a digit
    using its neighbours for the context-sensitive letters, (2) collapse runs
    of the same digit to one, (3) drop every ``0`` except a leading one (a
    leading vowel is significant; interior vowels are not).
    """
    letters = _normalize(word)
    if not letters:
        return ""

    codes: list[str] = []
    n = len(letters)
    for i, ch in enumerate(letters):
        prev = letters[i - 1] if i > 0 else ""
        nxt = letters[i + 1] if i + 1 < n else ""

        if ch == "h":
            # H is never coded (it only lengthens the preceding vowel).
            continue
        if ch in _SIMPLE:
            codes.append(_SIMPLE[ch])
        elif ch == "c":
            if i == 0:
                # Word-initial C: "4" before a/h/k/l/o/q/r/u/x, else "8".
                code = "4" if nxt in ("a", "h", "k", "l", "o", "q", "r", "u", "x") else "8"
            elif prev in ("s", "z"):
                code = "8"
            elif nxt in ("a", "h", "k", "o", "q", "u", "x"):
                code = "4"
            else:
                code = "8"
            codes.append(code)
        elif ch in ("d", "t"):
            codes.append("8" if nxt in ("c", "s", "z") else "2")
        elif ch == "p":
            codes.append("3" if nxt == "h" else "1")
        elif ch == "x":
            # X = ks; folds into the preceding sound after c/k/q, else "48".
            codes.append("8" if prev in ("c", "k", "q") else "48")
        # any other char produced nothing

    digits = "".join(codes)

    # Pass 2: collapse consecutive identical digits.
    collapsed: list[str] = []
    for d in digits:
        if not collapsed or collapsed[-1] != d:
            collapsed.append(d)

    # Pass 3: remove all "0" except a possible leading one.
    result = collapsed[0] if collapsed else ""
    result += "".join(d for d in collapsed[1:] if d != "0")
    return result


def levenshtein(a: str, b: str) -> int:
    """Classic edit distance (insert/delete/substitute cost 1). Symmetric."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def _norm_lev(a: str, b: str) -> float:
    """Levenshtein normalized to [0, 1] by the longer string (0 = identical)."""
    longest = max(len(a), len(b))
    if longest == 0:
        return 0.0
    return levenshtein(a, b) / longest


def acoustic_distance(asr_form: str, candidate: str) -> float:
    """How acoustically far ``candidate`` is from what the ASR heard (lower = closer).

    Combines phonetic-code distance (the dominant term — it decides whether the
    words *sound* alike) with normalized letter-level Levenshtein (a finer
    tiebreaker). Both terms are normalized to [0, 1] so the weights are
    comparable; identical inputs return exactly 0.0.
    """
    code_dist = _norm_lev(koelner_phonetik(asr_form), koelner_phonetik(candidate))
    letter_dist = _norm_lev(asr_form.lower(), candidate.lower())
    # Phonetic closeness is what the guardrail is really about; letters break ties.
    return 0.7 * code_dist + 0.3 * letter_dist


def rank_candidates(asr_form: str, candidates: list[str]) -> list[str]:
    """Return ``candidates`` sorted closest-sounding first (stable on ties)."""
    return sorted(candidates, key=lambda c: acoustic_distance(asr_form, c))


def accept_correction(asr_form: str, candidate: str, max_distance: float) -> bool:
    """True iff ``candidate`` is close enough to be an allowed correction.

    The guardrail: reject a candidate whose acoustic distance exceeds
    ``max_distance``, so a fluent-but-unrelated word can never replace what was
    actually heard.
    """
    return acoustic_distance(asr_form, candidate) <= max_distance
