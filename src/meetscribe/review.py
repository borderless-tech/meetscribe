"""The bk review loop: view-model building (this file) + the gate-walking engine (Task 4).

View-model building is pure and local: given a bk task dict (one entry of a workflow's
``tasks``, carrying ``payload`` + ``web_url``) and the meeting directory, derive everything
a front-end needs to render the gate — talk time, sample utterances, and playable
:class:`~meetscribe.interaction.Snippet` refs from ``transcript.json``, while suggestions,
roster, and mapping candidates mirror the task payload field-for-field. A missing ``raw/``
(cleaned up, moved, old meeting) degrades to empty snippets, never an error — the front-end
renders "(audio unavailable)".
"""

from __future__ import annotations

from pathlib import Path

from .interaction import (
    MappingCandidate,
    MappingReviewView,
    RosterEntry,
    Snippet,
    SpeakerReviewView,
    SpeakerView,
    Suggestion,
)
from .output import read_transcript
from .types import Utterance

# Sample utterances are display context, not the transcript — keep them to one line.
SAMPLE_MAX_CHARS = 120
# Playback cycles the longest segments; more than a few adds nothing.
MAX_SNIPPETS = 3


def _truncate_sample(text: str) -> str:
    """Cut at the last word boundary at or before SAMPLE_MAX_CHARS; mark the cut."""
    if len(text) <= SAMPLE_MAX_CHARS:
        return text
    head = text[:SAMPLE_MAX_CHARS]
    if " " in head:
        head = head.rsplit(" ", 1)[0]
    return head.rstrip() + "…"


def _speaker_view(label: str, suggestions: tuple[Suggestion, ...],
                  segments: list[Utterance], meeting_dir: Path) -> SpeakerView:
    # Longest segments carry the most identity — order samples and snippets by duration.
    segments = sorted(segments, key=lambda u: u.end - u.start, reverse=True)
    playable = [
        (u, wav)
        for u in segments
        if (wav := meeting_dir / "raw" / f"{u.track}.wav").exists()
    ]
    return SpeakerView(
        label=label,
        talk_time_s=sum(u.end - u.start for u in segments),
        samples=tuple(_truncate_sample(u.text) for u in segments),
        suggestions=suggestions,
        snippets=tuple(
            Snippet(wav, u.start, u.end - u.start) for u, wav in playable[:MAX_SNIPPETS]
        ),
    )


def build_speaker_view(task: dict, meeting_dir: str | Path) -> SpeakerReviewView:
    """Build the speaker-annotation gate's view from a bk task + local artifacts.

    The payload's speaker list is authoritative (bk may know speakers we lost locally —
    those appear with zero talk time and no samples, so the user can still assign them);
    the local transcript only enriches each label with derived context.
    """
    meeting_dir = Path(meeting_dir)
    _, _, utterances = read_transcript(meeting_dir / "transcript.json")
    by_speaker: dict[str, list[Utterance]] = {}
    for u in utterances:
        by_speaker.setdefault(u.speaker, []).append(u)
    speakers = tuple(
        _speaker_view(
            spk["label"],
            tuple(
                Suggestion(
                    person_id=s["person_id"],
                    name=s["name"],
                    confidence=s["confidence"],
                    source=s["source"],
                    preselected=s.get("preselected", False),
                )
                for s in spk["suggestions"]
            ),
            by_speaker.get(spk["label"], []),
            meeting_dir,
        )
        for spk in task["payload"]["speakers"]
    )
    roster = tuple(
        RosterEntry(person_id=r["person_id"], email=r["email"], name=r["name"])
        for r in task["payload"]["roster"]
    )
    return SpeakerReviewView(speakers=speakers, roster=roster, web_url=task["web_url"])


def build_mapping_view(task: dict) -> MappingReviewView:
    """Build the meeting-mapping gate's view — a pure mirror of the task payload."""
    payload = task["payload"]
    return MappingReviewView(
        candidates=tuple(
            MappingCandidate(
                meeting_id=c["meeting_id"],
                title=c["title"],
                start=c["start"],
                end=c["end"],
                match=c["match"],
            )
            for c in payload["candidates"]
        ),
        allow_create=payload["allow_create"],
        web_url=task["web_url"],
    )
