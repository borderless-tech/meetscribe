"""Flag-then-repair transcript cleanup (see docs/plans/2026-08-13-asr-artifact-repair.md).

The defects in these transcripts are ASR *decoder artifacts* — echo loops and garbled
words — not language errors, so we do NOT rewrite whole segments with an LLM (slow,
decode-bound, and it invents/over-corrects). Instead, per system-track utterance:

1. **echo collapse** (deterministic, ``echo.py``): drop decoder-loop repeats (gap≈0).
2. **flag** broken words (``lexicon.py``): tokens unknown to both de+en + not in glossary.
3. **repair** only those spans (``repair.py``): the LLM proposes a single word from context,
   applied only if it is *both* acoustically close to the ASR form *and* a real word
   (``phonetic`` + ``is_known``) — otherwise the raw word is kept.

Timestamps are never touched; only word text changes. The LLM client and the dictionary
are injected, so :class:`SpanRepairCleaner` is unit-tested with fakes; :class:`NullCleaner`
(the default) is a no-op so the suite stays model-free.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

from .echo import collapse_echoes
from .lexicon import Lexicon
from .repair import is_valid_correction, repair_words
from .suggest import gather_candidates
from .types import Utterance

# A correction must sound within this acoustic distance of the ASR word (else keep raw).
MAX_DISTANCE = 0.5
# Generating a single replacement word never needs many tokens.
_SPAN_MAX_TOKENS = 16


class LlamaClient(Protocol):
    def complete(self, prompt: str, max_tokens: int | None = None) -> str: ...


@dataclass
class RepairResult:
    utterances: list[Utterance]
    fixed: int  # broken words repaired
    echoes: int  # echo tokens collapsed
    active: bool  # did a real (non-Null) cleaner run? → drives meta's `cleaned` flag


class Cleaner(Protocol):
    def clean(self, utterances: list[Utterance], glossary: list[str], reporter) -> RepairResult: ...


class NullCleaner:
    """No-op cleaner (default): returns utterances unchanged, marks the run inactive."""

    def clean(self, utterances: list[Utterance], glossary: list[str], reporter) -> RepairResult:
        return RepairResult(list(utterances), fixed=0, echoes=0, active=False)


_SPAN_SYSTEM = (
    "A German meeting transcript sentence has ONE possibly-garbled word marked with «». "
    "If it is already a real word (including a name or English term), return it unchanged. "
    "Otherwise replace it with the single correct word. Output ONLY that one word, nothing else."
)


def build_span_prompt(words: list, i: int, glossary: list[str], window: int = 8) -> str:
    """Prompt to repair the single word at index ``i``, with ±``window`` words of context."""
    left = " ".join(w.w for w in words[max(0, i - window):i])
    right = " ".join(w.w for w in words[i + 1:i + 1 + window])
    gloss = ", ".join(glossary) if glossary else "(none)"
    return (
        f"{_SPAN_SYSTEM}\nKnown names: {gloss}\n"
        f"Sentence: {left} «{words[i].w}» {right}"
    )


def _first_word(text: str) -> str:
    parts = text.split()
    return parts[0].strip(".,?!:;«»\"'") if parts else text.strip()


class SpanRepairCleaner:
    """Echo-collapse + flag + guarded span repair over system-track utterances.

    ``client`` (LLM) and ``lexicon`` (dictionary) are injected. Rebuilds each utterance's
    ``text`` from its (collapsed + repaired) words and stashes the original in ``raw_text``.
    """

    def __init__(self, client: LlamaClient, lexicon: Lexicon,
                 max_distance: float = MAX_DISTANCE, max_gap: float = 0.02) -> None:
        self._client = client
        self._lex = lexicon
        self._max_distance = max_distance
        self._max_gap = max_gap

    def clean(self, utterances: list[Utterance], glossary: list[str], reporter) -> RepairResult:
        out: list[Utterance] = []
        total_fixed = total_echo = 0
        for u in utterances:
            words, removed = collapse_echoes(list(u.words), self._max_gap)
            total_echo += removed
            flagged = self._lex.flag_broken([w.w for w in words])

            def correct_fn(i: int, ws: list) -> str:
                return _first_word(
                    self._client.complete(build_span_prompt(ws, i, glossary),
                                          max_tokens=_SPAN_MAX_TOKENS)
                )

            def accept_fn(asr: str, cand: str) -> bool:
                return is_valid_correction(asr, cand, self._max_distance, self._lex.is_known)

            words, fixed = repair_words(words, flagged, correct_fn, accept_fn)
            total_fixed += fixed
            text = " ".join(w.w for w in words)  # rebuild from collapsed+repaired words
            out.append(replace(u, text=text, words=tuple(words), raw_text=u.text))
        if reporter is not None:
            reporter.info(
                f"repaired {total_fixed} word(s), collapsed {total_echo} echo(es) "
                f"across {len(utterances)} segment(s)"
            )
        return RepairResult(out, fixed=total_fixed, echoes=total_echo, active=True)


class ManagedSpanRepairCleaner:
    """Real cleaner: builds the hunspell lexicon (with the runtime glossary), spins up a
    ``llama-server``, delegates to :class:`SpanRepairCleaner`, tears the server down. If the
    server or dictionaries are unavailable it degrades gracefully — a `warn` + unchanged
    utterances marked inactive (so meta stays ``cleaned:false``)."""

    def __init__(self, model_path: str, dicpath: str, model_info: dict | None = None,
                 threads: int = 4, _server=None, _client=None, _lexicon=None) -> None:
        self.model_path = model_path
        self.dicpath = dicpath
        self.model_info = model_info
        self.threads = threads
        self._server = _server
        self._client = _client
        self._lexicon = _lexicon

    def clean(self, utterances: list[Utterance], glossary: list[str], reporter) -> RepairResult:
        try:
            lexicon = self._lexicon or self._default_lexicon(glossary)
            server_cm = self._server() if self._server else self._default_server()
            with server_cm as server:
                make_client = self._client or (lambda url: self._default_client(url))
                client = make_client(server.base_url)
                return SpanRepairCleaner(client, lexicon).clean(utterances, glossary, reporter)
        except Exception as exc:
            if reporter is not None:
                reporter.warn(f"cleanup skipped: repair backend unavailable ({exc})")
            return RepairResult(list(utterances), fixed=0, echoes=0, active=False)

    def _default_lexicon(self, glossary: list[str]) -> Lexicon:
        from .lexicon import build_hunspell_lexicon

        return build_hunspell_lexicon(self.dicpath, glossary)

    def _default_server(self):
        from .llama import LlamaServer

        return LlamaServer(self.model_path, threads=self.threads)

    def _default_client(self, base_url: str):
        from .llama import LlamaClient

        return LlamaClient(base_url)


# ---- suggest-mode (human-in-the-loop; the decided direction) ------------------------

@dataclass
class SuggestResult:
    utterances: list[Utterance]  # echo-collapsed; broken words left raw (not applied)
    suggestions: list[dict]  # per flagged word: segment/word_index/start/end/original/candidates
    echoes: int
    active: bool


class SuggestCleaner:
    """Echo-collapse (auto, safe) + flag broken words + emit ranked correction *candidates*
    for a human to accept in borderless-knowledge. Broken words are NOT auto-applied (acoustic
    distance can't separate right from plausibly-wrong). Candidates = hunspell sound-alikes
    (``suggest_fn``) unioned with an optional LLM guess (``client``); both injected."""

    def __init__(self, suggest_fn, lexicon: Lexicon, client=None,
                 max_candidates: int = 5, max_gap: float = 0.02) -> None:
        self._suggest = suggest_fn
        self._lex = lexicon
        self._client = client
        self._max = max_candidates
        self._max_gap = max_gap

    def clean(self, utterances: list[Utterance], glossary: list[str], reporter) -> SuggestResult:
        out: list[Utterance] = []
        suggestions: list[dict] = []
        total_echo = 0
        for seg, u in enumerate(utterances):
            words, removed = collapse_echoes(list(u.words), self._max_gap)
            total_echo += removed
            flagged = self._lex.flag_broken([w.w for w in words])
            hun_map = self._suggest([words[j].w for j in flagged]) if flagged else {}
            for j in flagged:
                w = words[j]
                llm = None
                if self._client is not None:
                    llm = _first_word(self._client.complete(
                        build_span_prompt(words, j, glossary), max_tokens=_SPAN_MAX_TOKENS))
                cand = gather_candidates(w.w, hun_map.get(w.w, []), llm, self._max)
                if cand.candidates:
                    suggestions.append({
                        "segment": seg, "word_index": j, "start": w.start, "end": w.end,
                        "original": w.w, "candidates": cand.candidates,
                    })
            out.append(replace(u, words=tuple(words),
                               text=" ".join(x.w for x in words), raw_text=u.text))
        if reporter is not None:
            reporter.info(f"collapsed {total_echo} echo(es); "
                          f"{len(suggestions)} broken-word suggestion(s) for review")
        return SuggestResult(out, suggestions, total_echo, active=True)


class ManagedSuggestCleaner:
    """Real suggest cleaner: builds the hunspell lexicon + suggester (runtime glossary), and —
    when ``use_llm`` — a ``llama-server`` for an extra contextual candidate. Graceful fallback
    (warn + unchanged, inactive) if the backend is unavailable."""

    def __init__(self, model_path: str, dicpath: str, model_info: dict | None = None,
                 threads: int = 4, use_llm: bool = True,
                 _server=None, _client=None, _suggester=None, _lexicon=None) -> None:
        self.model_path = model_path
        self.dicpath = dicpath
        self.model_info = model_info
        self.threads = threads
        self.use_llm = use_llm
        self._server = _server
        self._client = _client
        self._suggester = _suggester
        self._lexicon = _lexicon

    def clean(self, utterances: list[Utterance], glossary: list[str], reporter) -> SuggestResult:
        try:
            suggester = self._suggester or self._default_suggester()
            lexicon = self._lexicon or self._default_lexicon(glossary)
            if not self.use_llm:
                return SuggestCleaner(suggester, lexicon, client=None).clean(
                    utterances, glossary, reporter)
            server_cm = self._server() if self._server else self._default_server()
            with server_cm as server:
                make_client = self._client or (lambda url: self._default_client(url))
                client = make_client(server.base_url)
                return SuggestCleaner(suggester, lexicon, client=client).clean(
                    utterances, glossary, reporter)
        except Exception as exc:
            if reporter is not None:
                reporter.warn(f"cleanup skipped: suggest backend unavailable ({exc})")
            return SuggestResult(list(utterances), [], 0, active=False)

    def _default_suggester(self):
        from .lexicon import build_hunspell_suggester

        return build_hunspell_suggester(self.dicpath)

    def _default_lexicon(self, glossary: list[str]) -> Lexicon:
        from .lexicon import build_hunspell_lexicon

        return build_hunspell_lexicon(self.dicpath, glossary)

    def _default_server(self):
        from .llama import LlamaServer

        return LlamaServer(self.model_path, threads=self.threads)

    def _default_client(self, base_url: str):
        from .llama import LlamaClient

        return LlamaClient(base_url)
