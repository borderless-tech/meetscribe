"""Assemble broken-word correction candidates for human review in borderless-knowledge.

We deliberately do NOT auto-apply: acoustic distance cannot separate a correct fix from a
plausibly-wrong one (a wrong ``unterstatten→unterstellen`` scores *closer* than a correct
``vorhandliche→vorhandene``), so a human picks. The pipeline's job is to offer a good
shortlist. Sources are complementary and merged here:

- **hunspell** suggestions — free, instant, multi-candidate sound-alikes (catches e.g.
  ``geernt→gelernt``);
- an optional **LLM** guess — context-aware, catches what hunspell misses
  (``vorhandliche→vorhandene``).

Candidates are deduped, the original dropped, and ranked sound-alike-first via
``phonetic.acoustic_distance`` (since false positives are cheap under human review, we keep a
generous top-``max_n``). Pure/deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import phonetic


@dataclass
class Suggestion:
    original: str
    candidates: list[str]  # ranked, closest-sounding first


def gather_candidates(original: str, hunspell_suggestions: list[str],
                      llm_suggestion: str | None = None, max_n: int = 5) -> Suggestion:
    """Merge hunspell + (optional) LLM candidates → a deduped, sound-ranked shortlist."""
    pool = list(hunspell_suggestions)
    if llm_suggestion:
        pool.append(llm_suggestion)

    seen: set[str] = set()
    uniq: list[str] = []
    for cand in pool:
        c = cand.strip()
        key = c.casefold()
        if c and key != original.strip().casefold() and key not in seen:
            seen.add(key)
            uniq.append(c)

    ranked = phonetic.rank_candidates(original, uniq)  # closest-sounding first
    return Suggestion(original, ranked[:max_n])
