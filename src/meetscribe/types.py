"""Shared, JSON-friendly data types. Times are float seconds throughout."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Word:
    w: str
    start: float
    end: float


@dataclass(frozen=True)
class Segment:
    """One ASR utterance on one track (speaker not yet assigned for the system track)."""

    start: float
    end: float
    text: str
    words: tuple[Word, ...]


@dataclass(frozen=True)
class DiarSegment:
    """A diarization span with a cluster label."""

    start: float
    end: float
    speaker: str  # "spk_0", "spk_1", ...


@dataclass(frozen=True)
class Utterance:
    """A final, merged unit written to transcript.json.

    ``text`` is the human-readable form (possibly LLM-cleaned); ``raw_text`` is the verbatim
    ASR text. They are equal until the cleanup stage runs — which is the *only* place
    ``raw_text`` is ever set (never at align/mic construction or in coalesce). When unset it
    defaults to "" and serialization/read falls back to ``text``.
    """

    start: float
    end: float
    speaker: str  # "me" | "spk_N"
    track: str  # "mic" | "system"
    text: str
    words: tuple[Word, ...]
    raw_text: str = ""
