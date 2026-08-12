"""Glossary-informed LLM cleanup of transcript text (see docs/plans/…-cleanup-design.md).

The cleaner fixes garbled proper nouns / spelling from *context* + a known-terms glossary,
restores punctuation, and lightly smooths disfluencies. It is injected as a pipeline component
(:class:`NullCleaner` is the default so the unit suite stays LLM-free); the real
:class:`LlamaCleaner` drives a ``LlamaClient`` (a managed ``llama-server``, see ``llama.py``).

Only ``segment.text`` is ever rewritten — per-word timings are untouched — so the worst case
degrades, per segment, to the raw ASR text. Two anti-hallucination guards keep an LLM meltdown
from corrupting the transcript: an empty candidate and a length ratio outside sane bounds are
rejected in favour of the raw text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

# A candidate longer than this multiple of the input is a runaway generation.
_MAX_RATIO = 3.0
# A candidate shorter than this multiple is a collapse — but only trusted as a signal on
# inputs long enough that a big shrink is genuinely suspicious (short backchannels like
# "uh the the" → "the" are legitimate large-ratio shrinks).
_MIN_RATIO = 0.3
_MIN_LEN_FOR_COLLAPSE = 40


class LlamaClient(Protocol):
    def complete(self, prompt: str) -> str: ...


@dataclass
class CleanResult:
    texts: list[str]
    cleaned: int
    kept_raw: int
    active: bool  # did a real (non-Null) cleaner run? → drives meta's `cleaned` flag


class Cleaner(Protocol):
    def clean(self, texts: list[str], glossary: list[str], reporter) -> CleanResult: ...


def accept_candidate(original: str, candidate: str) -> bool:
    """True if ``candidate`` is a safe replacement for ``original`` (else keep raw)."""
    cand = candidate.strip()
    if not cand:
        return False
    n0 = len(original.strip())
    if n0 == 0:
        return False  # nothing to correct → never inject LLM text
    ratio = len(cand) / n0
    if ratio > _MAX_RATIO:
        return False  # runaway generation
    if n0 > _MIN_LEN_FOR_COLLAPSE and ratio < _MIN_RATIO:
        return False  # collapse on a substantial segment
    return True


_SYSTEM = (
    "You are a transcription cleanup tool. Correct obvious ASR errors — spelling, "
    "split/merged words, and misrecognized names using the glossary — and restore "
    "punctuation and capitalization. Do NOT translate, summarize, paraphrase, or add "
    "anything. Keep the original language. If the text is already fine, return it unchanged. "
    "Output only the corrected text, nothing else."
)


def build_prompt(glossary: list[str], prev: str, current: str) -> str:
    """Assemble the cleanup instruction. The real client wraps this in the model's chat
    template (ChatML for Qwen); tests only assert the pieces are present."""
    gloss = ", ".join(glossary) if glossary else "(none)"
    ctx = prev.strip() or "(start of meeting)"
    return (
        f"{_SYSTEM}\n\n"
        f"Known correct spellings (fix misspellings of these): {gloss}\n\n"
        f"Previous line (context, do not edit or repeat):\n{ctx}\n\n"
        f"Line to correct:\n{current}"
    )


class NullCleaner:
    """No-op cleaner (default). Returns text unchanged; marks the run inactive."""

    def clean(self, texts: list[str], glossary: list[str], reporter) -> CleanResult:
        return CleanResult(texts=list(texts), cleaned=0, kept_raw=0, active=False)


class LlamaCleaner:
    """Per-segment cleanup via an injected :class:`LlamaClient`."""

    def __init__(self, client: LlamaClient) -> None:
        self._client = client

    def clean(self, texts: list[str], glossary: list[str], reporter) -> CleanResult:
        out: list[str] = []
        cleaned = kept_raw = 0
        warned = False
        prev = ""
        for text in texts:
            candidate: str | None = None
            try:
                candidate = self._client.complete(build_prompt(glossary, prev, text))
            except Exception as exc:  # server died / timeout → keep raw, don't abort the run
                if reporter is not None and not warned:
                    reporter.warn(f"cleanup: LLM call failed ({exc}); keeping raw text")
                    warned = True
            if candidate is not None and accept_candidate(text, candidate):
                out.append(candidate.strip())
                cleaned += 1
            else:
                if candidate is not None and reporter is not None and not warned:
                    reporter.warn("cleanup: rejected an implausible LLM output; keeping raw text")
                    warned = True
                out.append(text)
                kept_raw += 1
            prev = out[-1]
        return CleanResult(texts=out, cleaned=cleaned, kept_raw=kept_raw, active=True)
