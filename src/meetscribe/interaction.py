"""Front-end seam for the bk review loop. Task-level, not prompt-level.

The engine (``review.py``) exchanges complete, presentation-neutral *view-models* for
*answers* — one method per bk gate type — so a CLI can walk speakers sequentially while a
later GUI renders one form per gate, both against the same :class:`Interaction` protocol.
Playback is a front-end concern: views carry :class:`Snippet` refs into local WAVs, never
audio bytes.

``NullInteraction`` (the default everywhere) raises on any question so no non-interactive
path can block on input; ``ScriptedInteraction`` feeds canned answers to engine tests —
both live here next to the protocol, mirroring ``NullReporter`` in ``progress.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Union


# ---- View-models (frozen; times are float seconds, matching types.py) -----------------

@dataclass(frozen=True)
class Snippet:
    """A playable span in a local WAV file."""

    path: Path
    start: float
    duration: float


@dataclass(frozen=True)
class Suggestion:
    person_id: str
    name: str
    confidence: str | None
    source: str
    preselected: bool = False


@dataclass(frozen=True)
class SpeakerView:
    label: str
    talk_time_s: float
    samples: tuple[str, ...]  # sample utterance texts, longest first
    suggestions: tuple[Suggestion, ...]
    snippets: tuple[Snippet, ...]  # longest first; empty when raw/ is gone


@dataclass(frozen=True)
class RosterEntry:
    person_id: str
    email: str | None
    name: str | None


@dataclass(frozen=True)
class SpeakerReviewView:
    speakers: tuple[SpeakerView, ...]
    roster: tuple[RosterEntry, ...]
    web_url: str


@dataclass(frozen=True)
class MappingCandidate:
    meeting_id: str
    title: str
    start: str | None
    end: str | None
    match: str


@dataclass(frozen=True)
class MappingReviewView:
    candidates: tuple[MappingCandidate, ...]
    allow_create: bool
    web_url: str


# ---- Answers ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NewPerson:
    name: str
    email: str | None = None


@dataclass(frozen=True)
class Assignment:
    """Exactly one of ``person_id`` / ``new_person`` set, or neither (= unassigned)."""

    label: str
    person_id: str | None = None
    new_person: NewPerson | None = None


@dataclass(frozen=True)
class SpeakerAnswers:
    assignments: tuple[Assignment, ...]


@dataclass(frozen=True)
class MappingAnswer:
    """Exactly one of ``meeting_id`` / ``create_title`` set."""

    meeting_id: str | None = None
    create_title: str | None = None


class OpenWeb:
    """Sentinel answer: hand off to bk's web UI (compared by isinstance)."""


class Quit:
    """Sentinel answer: leave the review loop cleanly (compared by isinstance)."""


SpeakerResult = Union[SpeakerAnswers, OpenWeb, Quit]
MappingResult = Union[MappingAnswer, OpenWeb, Quit]


# ---- Protocol --------------------------------------------------------------------------

class Interaction(Protocol):
    def review_speakers(self, view: SpeakerReviewView) -> SpeakerResult: ...
    def review_mapping(self, view: MappingReviewView) -> MappingResult: ...
    def notify(self, message: str) -> None: ...


# ---- Null (default) --------------------------------------------------------------------

class NullInteraction:
    """Raises on any question — no non-interactive path may ever block on input."""

    _MSG = "interactive review requires a front-end — this is a bug, not a user error"

    def review_speakers(self, view: SpeakerReviewView) -> SpeakerResult:
        raise RuntimeError(self._MSG)

    def review_mapping(self, view: MappingReviewView) -> MappingResult:
        raise RuntimeError(self._MSG)

    def notify(self, message: str) -> None:
        pass


# ---- Scripted (tests) ------------------------------------------------------------------

class ScriptedInteraction:
    """Pops the next canned answer per review call; records every view and notify."""

    def __init__(self, answers: list) -> None:
        self._answers = list(answers)
        self.seen: list[SpeakerReviewView | MappingReviewView] = []
        self.messages: list[str] = []

    def review_speakers(self, view: SpeakerReviewView) -> SpeakerResult:
        self.seen.append(view)
        return self._answers.pop(0)

    def review_mapping(self, view: MappingReviewView) -> MappingResult:
        self.seen.append(view)
        return self._answers.pop(0)

    def notify(self, message: str) -> None:
        self.messages.append(message)
