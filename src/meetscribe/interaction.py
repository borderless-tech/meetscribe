"""Front-end seam for the bk review loop. Task-level, not prompt-level.

The engine (``review.py``) exchanges complete, presentation-neutral *view-models* for
*answers* — one method per bk gate type — so a CLI can walk speakers sequentially while a
later GUI renders one form per gate, both against the same :class:`Interaction` protocol.
Playback is a front-end concern: views carry :class:`Snippet` refs into local WAVs, never
audio bytes.

``NullInteraction`` (the default everywhere) raises on any question so no non-interactive
path can block on input; ``ScriptedInteraction`` feeds canned answers to engine tests —
both live here next to the protocol, mirroring ``NullReporter`` in ``progress.py``.
``RichInteraction`` is the CLI front-end (stderr, like ``RichReporter``): one key per
speaker in the common case, ffplay for snippet playback — all four of its seams
(console, readkey, read_line, player) are injectable, so its tests stay scripted and
offline.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime
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


@dataclass(frozen=True)
class CalendarCandidate:
    """A calendar event near record start (bk ``/meetings?around=``). Attendees reuse
    :class:`RosterEntry`; ``person_id`` is unused for calendar attendees and left ``""``."""

    meeting_id: str
    title: str
    start: str | None  # tz-aware ISO, or None
    end: str | None
    attendees: tuple[RosterEntry, ...]


@dataclass(frozen=True)
class MeetingPickView:
    candidates: tuple[CalendarCandidate, ...]


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
    def pick_meeting(self, view: MeetingPickView) -> str | None: ...  # id | None = skip
    def ask_retry(self, message: str) -> bool: ...
    def notify(self, message: str) -> None: ...


# ---- Null (default) --------------------------------------------------------------------

class NullInteraction:
    """Raises on any question — no non-interactive path may ever block on input."""

    _MSG = "interactive review requires a front-end — this is a bug, not a user error"

    def review_speakers(self, view: SpeakerReviewView) -> SpeakerResult:
        raise RuntimeError(self._MSG)

    def review_mapping(self, view: MappingReviewView) -> MappingResult:
        raise RuntimeError(self._MSG)

    def pick_meeting(self, view: MeetingPickView) -> str | None:
        raise RuntimeError(self._MSG)

    def ask_retry(self, message: str) -> bool:
        raise RuntimeError(self._MSG)

    def notify(self, message: str) -> None:
        pass


# ---- Scripted (tests) ------------------------------------------------------------------

class ScriptedInteraction:
    """Pops the next canned answer per review call; records every view and notify."""

    def __init__(self, answers: list) -> None:
        self._answers = list(answers)
        self.seen: list[SpeakerReviewView | MappingReviewView | MeetingPickView] = []
        self.messages: list[str] = []

    def review_speakers(self, view: SpeakerReviewView) -> SpeakerResult:
        self.seen.append(view)
        return self._answers.pop(0)

    def review_mapping(self, view: MappingReviewView) -> MappingResult:
        self.seen.append(view)
        return self._answers.pop(0)

    def pick_meeting(self, view: MeetingPickView) -> str | None:
        self.seen.append(view)
        return self._answers.pop(0)

    def ask_retry(self, message: str) -> bool:
        self.messages.append(message)
        return self._answers.pop(0)  # scripted True (retry) / False (quit)

    def notify(self, message: str) -> None:
        self.messages.append(message)


# ---- ffplay playback ---------------------------------------------------------------------

def ffplay_argv(snippet: Snippet) -> list[str]:
    """The exact ffplay command for a snippet — pure, so it is testable without audio."""
    return [
        "ffplay", "-nodisp", "-autoexit", "-loglevel", "error",
        "-ss", f"{snippet.start:.3f}", "-t", f"{snippet.duration:.3f}", str(snippet.path),
    ]


def play_snippet(snippet: Snippet, *, run=subprocess.run) -> str | None:
    """Play a snippet blocking (snippets are seconds long). Returns ``None`` on success,
    or a human message for ``notify`` when ffplay is missing — never raises for that."""
    try:
        run(ffplay_argv(snippet), check=False)
    except FileNotFoundError:
        return "ffplay not found — install ffmpeg"
    return None


# ---- Rich (CLI) --------------------------------------------------------------------------

_ENTER_KEYS = ("\r", "\n")


def _read_key() -> str:
    """Read one raw keypress from the tty (no Enter needed). Ctrl-C still interrupts —
    raw mode suppresses the signal, so we re-raise it ourselves."""
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    if ch == "\x03":
        raise KeyboardInterrupt
    return ch


def _fmt_talk_time(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


def _fmt_local_range(start: str | None, end: str | None) -> str | None:
    """``HH:MM–HH:MM`` in local time, or ``None`` when either bound is missing or
    unparsable — the values come from an untrusted bk ref, and a bad timestamp must
    cost the row its time range, never the pick."""
    if not start or not end:
        return None
    def hhmm(iso: str) -> str:
        return datetime.fromisoformat(iso).astimezone().strftime("%H:%M")
    try:
        return f"{hhmm(start)}–{hhmm(end)}"
    except (TypeError, ValueError):
        return None


def _preselection(suggestions: tuple[Suggestion, ...]) -> Suggestion | None:
    """bk's explicit ``preselected`` wins; else the top high-confidence suggestion."""
    for s in suggestions:
        if s.preselected:
            return s
    return next((s for s in suggestions if s.confidence == "high"), None)


class RichInteraction:
    """Terminal front-end: renders gates to stderr, reads single keys, plays snippets.

    One keystroke per speaker in the common case (Enter = confirm the preselection).
    All four seams are injectable for tests; the defaults are the real tty/ffplay."""

    def __init__(self, console=None, *, readkey=None, read_line=None, player=None) -> None:
        if console is None:
            from rich.console import Console

            console = Console(stderr=True)
        self.console = console
        self._readkey = readkey if readkey is not None else _read_key
        self._read_line = read_line if read_line is not None else input
        self._player = player if player is not None else play_snippet

    def notify(self, message: str) -> None:
        self.console.print(message)

    def ask_retry(self, message: str) -> bool:
        """A submit failed in transit but the answers are still in memory: offer
        re-sending the same body instead of re-walking the gate."""
        self.console.print(f"[yellow]{message}[/yellow]")
        self.console.print("  [dim]Enter/r retry · q quit[/dim]")
        while True:
            key = self._readkey()
            if key == "r" or key in _ENTER_KEYS:
                return True
            if key == "q":
                return False
            # unknown key — ignore, re-read

    # -- speaker annotation gate --

    def review_speakers(self, view: SpeakerReviewView) -> SpeakerResult:
        while True:  # "e" at the summary backs out and restarts the walk
            assignments: list[Assignment] = []
            for spk in view.speakers:
                answer = self._ask_speaker(spk, view.roster)
                if isinstance(answer, (OpenWeb, Quit)):
                    return answer
                assignments.append(answer)
            confirmed = self._confirm_summary(view.speakers, assignments)
            if isinstance(confirmed, (OpenWeb, Quit)):
                return confirmed
            if confirmed:
                return SpeakerAnswers(assignments=tuple(assignments))

    def _ask_speaker(
        self, spk: SpeakerView, roster: tuple[RosterEntry, ...]
    ) -> Assignment | OpenWeb | Quit:
        self._render_speaker(spk)
        pre = _preselection(spk.suggestions)
        plays = 0  # "p" cycles through the snippets, longest first
        while True:
            key = self._readkey()
            if key == "q":
                return Quit()
            if key == "w":
                return OpenWeb()
            if key in _ENTER_KEYS:
                if pre is None:
                    continue  # nothing to confirm — ignore, re-read
                return Assignment(label=spk.label, person_id=pre.person_id)
            if key.isdigit() and key != "0":
                i = int(key) - 1
                if i < len(spk.suggestions):
                    return Assignment(
                        label=spk.label, person_id=spk.suggestions[i].person_id
                    )
                continue
            if key == "u":
                return Assignment(label=spk.label)
            if key == "n":
                name = self._read_line("  name: ").strip()
                if not name:
                    continue  # a person needs a name — treat empty as cancel
                email = self._read_line("  email (optional): ").strip() or None
                return Assignment(
                    label=spk.label, new_person=NewPerson(name=name, email=email)
                )
            if key == "r":
                picked = self._pick_roster(roster)
                if isinstance(picked, (OpenWeb, Quit)):
                    return picked
                if picked is None:
                    self._render_speaker(spk)  # cancelled — show the block again
                    continue
                return Assignment(label=spk.label, person_id=picked.person_id)
            if key == "p":
                if not spk.snippets:
                    self.notify("(audio unavailable)")
                    continue
                message = self._player(spk.snippets[plays % len(spk.snippets)])
                plays += 1
                if message:
                    self.notify(message)
                continue
            # unknown key — ignore, re-read

    def _render_speaker(self, spk: SpeakerView) -> None:
        self.console.print(
            f"\n[bold]{spk.label}[/bold] — {_fmt_talk_time(spk.talk_time_s)} talk time"
        )
        for sample in spk.samples[:2]:
            self.console.print(f'  [dim]"{sample}"[/dim]')
        for i, s in enumerate(spk.suggestions, start=1):
            hint = f"{s.source}, {s.confidence}" if s.confidence else s.source
            marker = " [green]←[/green]" if _preselection(spk.suggestions) is s else ""
            self.console.print(f"  {i}) {s.name}  [dim]({hint})[/dim]{marker}")
        actions = "Enter confirm · 1-9 pick · r roster · n new · u unassigned"
        if spk.snippets:
            actions += " · p play"
        self.console.print(f"  [dim]{actions} · w web · q quit[/dim]")

    def _pick_roster(
        self, roster: tuple[RosterEntry, ...]
    ) -> RosterEntry | OpenWeb | Quit | None:
        for i, entry in enumerate(roster, start=1):
            self.console.print(f"  {i}) {entry.name or entry.email or entry.person_id}")
        while True:
            key = self._readkey()
            if key == "q":
                return Quit()
            if key == "w":
                return OpenWeb()
            if key.isdigit() and key != "0":
                i = int(key) - 1
                if i < len(roster):
                    return roster[i]
                continue
            return None  # any other key backs out to the speaker

    def _confirm_summary(
        self, speakers: tuple[SpeakerView, ...], assignments: list[Assignment]
    ) -> bool | OpenWeb | Quit:
        names = {
            s.person_id: s.name for spk in speakers for s in spk.suggestions
        }
        self.console.print("\n[bold]summary[/bold]")
        for a in assignments:
            if a.new_person is not None:
                chosen = f"{a.new_person.name} [dim](new)[/dim]"
            elif a.person_id is not None:
                chosen = names.get(a.person_id, a.person_id)
            else:
                chosen = "[dim]unassigned[/dim]"
            self.console.print(f"  {a.label} → {chosen}")
        self.console.print("  [dim]y/Enter submit · e edit again · w web · q quit[/dim]")
        while True:
            key = self._readkey()
            if key == "q":
                return Quit()
            if key == "w":
                return OpenWeb()
            if key == "y" or key in _ENTER_KEYS:
                return True
            if key == "e":
                return False

    # -- meeting mapping gate --

    def review_mapping(self, view: MappingReviewView) -> MappingResult:
        self._render_mapping(view)
        # Preselect only a *sole* high-match candidate — two highs mean bk isn't sure.
        high = [c for c in view.candidates if c.match == "high"]
        pre = high[0] if len(high) == 1 else None
        while True:
            key = self._readkey()
            if key == "q":
                return Quit()
            if key == "w":
                return OpenWeb()
            if key in _ENTER_KEYS:
                if pre is None:
                    continue
                return MappingAnswer(meeting_id=pre.meeting_id)
            if key.isdigit() and key != "0":
                i = int(key) - 1
                if i < len(view.candidates):
                    return MappingAnswer(meeting_id=view.candidates[i].meeting_id)
                continue
            if key == "n" and view.allow_create:
                title = self._read_line("  title: ").strip()
                return MappingAnswer(
                    create_title=title or f"Meeting {date.today().isoformat()}"
                )
            # unknown key (incl. a hidden "n") — ignore, re-read

    # -- calendar meeting pick (record flow, not a bk gate) --

    def pick_meeting(self, view: MeetingPickView) -> str | None:
        self._render_pick(view)
        while True:
            key = self._readkey()
            if key == "s":
                return None
            if key in _ENTER_KEYS:
                if len(view.candidates) == 1:
                    return view.candidates[0].meeting_id
                return None  # several candidates — Enter won't guess, skip
            if key.isdigit() and key != "0":
                i = int(key) - 1
                if i < len(view.candidates):
                    return view.candidates[i].meeting_id
                continue
            # unknown key — ignore, re-read

    def _render_pick(self, view: MeetingPickView) -> None:
        self.console.print("\n[bold]calendar match?[/bold]")
        for i, c in enumerate(view.candidates, start=1):
            when = _fmt_local_range(c.start, c.end)
            extra = f"  [dim]{when}[/dim]" if when else ""
            names = ", ".join(a.name or a.email or "" for a in c.attendees)
            if names:
                extra += f"  [dim]{names}[/dim]"
            self.console.print(f"  {i}) {c.title}{extra}")
        if len(view.candidates) == 1:
            actions = "Enter confirm · 1 pick · s skip"
        else:
            actions = "1-9 pick · Enter/s skip"
        self.console.print(f"  [dim]{actions}[/dim]")

    def _render_mapping(self, view: MappingReviewView) -> None:
        self.console.print("\n[bold]which meeting is this?[/bold]")
        for i, c in enumerate(view.candidates, start=1):
            when = f"  [dim]{c.start} – {c.end}[/dim]" if c.start else ""
            self.console.print(f"  {i}) {c.title}  [dim]({c.match})[/dim]{when}")
        actions = "Enter confirm · 1-9 pick"
        if view.allow_create:
            actions += " · n new meeting"
        self.console.print(f"  [dim]{actions} · w web · q quit[/dim]")
