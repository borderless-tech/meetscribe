"""The bk review loop: view-model building + the gate-walking engine.

View-model building is pure and local: given a bk task dict (one entry of a workflow's
``tasks``, carrying ``payload`` + ``web_url``) and the meeting directory, derive everything
a front-end needs to render the gate — talk time, sample utterances, and playable
:class:`~meetscribe.interaction.Snippet` refs from ``transcript.json``, while suggestions,
roster, and mapping candidates mirror the task payload field-for-field. A missing ``raw/``
(cleaned up, moved, old meeting) degrades to empty snippets, never an error — the front-end
renders "(audio unavailable)".

The engine (:func:`run_review`) is pure logic over injected seams — a bk client, an
:class:`~meetscribe.interaction.Interaction`, a ``Reporter``, ``sleep``, ``open_url`` —
so tests drive it entirely offline with the vendored contract fixtures. It polls
``GET /workflows/{id}`` while ``processing``, dispatches ``awaiting_review`` tasks to the
front-end, translates answers into contract submit bodies, and feeds each submit's 200
response (the full fresh workflow) straight back into the state machine. bk's web UI is
the universal fallback (contract rule 1): unknown task types, unknown workflow states,
and ``OpenWeb`` answers all hand off to ``web_url`` with exit 0.
"""

from __future__ import annotations

import time
import webbrowser
from datetime import datetime
from pathlib import Path

from .bk import (
    BkAuthError,
    BkError,
    BkInvalidSubmission,
    BkRevisionConflict,
    read_workflow_ref,
    write_workflow_ref,
)
from .interaction import (
    Interaction,
    MappingAnswer,
    MappingCandidate,
    MappingReviewView,
    OpenWeb,
    Quit,
    RosterEntry,
    Snippet,
    SpeakerAnswers,
    SpeakerReviewView,
    SpeakerView,
    Suggestion,
)
from .output import read_transcript
from .progress import NullReporter, Reporter
from .types import Utterance

# Sample utterances are display context, not the transcript — keep them to one line.
SAMPLE_MAX_CHARS = 120
# Playback cycles the longest segments; more than a few adds nothing.
MAX_SNIPPETS = 3

# Polling cadence (design §engine): 5 s while fresh, 15 s once a processing stint has
# lasted a minute — review sessions are human-paced, hammering bk buys nothing.
POLL_INTERVAL_S = 5.0
POLL_INTERVAL_SLOW_S = 15.0
POLL_BACKOFF_AFTER_S = 60.0
# A single blip must not lose the session; a dead bk must not poll forever.
POLL_FAILURE_LIMIT = 5


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


# ---- submission translation (pure: answers + task revision → contract body) -----------


def mapping_submission(answer: MappingAnswer, revision: int) -> dict:
    """The contract's meeting-mapping submit body — select XOR create."""
    if answer.meeting_id is not None:
        return {"revision": revision, "meeting_id": answer.meeting_id}
    return {"revision": revision, "create": {"title": answer.create_title}}


def speaker_submission(answers: SpeakerAnswers, revision: int) -> dict:
    """The contract's speaker-annotation submit body.

    Three assignment forms per the contract: ``person_id`` (assign), ``person``
    (create), and an explicit ``"person_id": null`` (deliberately unassigned — the
    absence of a key would read as "forgot to answer")."""
    assignments: list[dict] = []
    for a in answers.assignments:
        if a.new_person is not None:
            assignments.append(
                {"label": a.label, "person": {"email": a.new_person.email, "name": a.new_person.name}}
            )
        else:
            assignments.append({"label": a.label, "person_id": a.person_id})
    return {"revision": revision, "assignments": assignments}


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


# ---- the engine -------------------------------------------------------------------------


def _persist_state(meeting_dir: Path, ref: dict, wf: dict) -> dict:
    """Merge a fetched workflow into the ref file (same shape as ``cli._run_status``)
    so ``status`` stays accurate even when the loop is interrupted mid-session."""
    ref = {
        **ref,
        "state": str(wf.get("state") or ref.get("state") or "unknown"),
        "web_url": wf.get("web_url") or ref.get("web_url"),
        "checked_at": datetime.now().astimezone().isoformat(),
    }
    write_workflow_ref(meeting_dir, ref)
    return ref


def _web_fallback(url, interaction: Interaction, open_url) -> int:
    """Contract rule 1: whatever the terminal cannot handle, the web UI can."""
    interaction.notify(f"continue this review in the browser: {url}")
    open_url(url)
    return 0


def run_review(
    meeting_dir,
    client,
    interaction: Interaction,
    reporter: Reporter = NullReporter(),
    *,
    sleep=time.sleep,
    open_url=webbrowser.open,
    emit=print,
) -> int:
    """Walk the uploaded meeting's bk review gates to a terminal state.

    Returns a process exit code: 0 for ``done``, a user quit, Ctrl-C, or a web
    handoff (quitting is legitimate — gates wait 7 days); 1 for ``failed`` or an
    unreachable bk; 2 when the meeting has no usable workflow reference or bk
    rejects the token.

    ``emit`` receives the single machine-useful stdout line
    (``<workflow_id> <state>``, design §gate-rendering) — all human rendering goes
    to stderr via the interaction/reporter seams, so scripts read the outcome here."""
    meeting_dir = Path(meeting_dir)
    ref = read_workflow_ref(meeting_dir)
    if ref is None or not ref.get("workflow_id"):
        interaction.notify(
            f"no uploaded workflow found in {meeting_dir}"
            f" — run: meetscribe upload {meeting_dir}"
        )
        return 2
    try:
        return _loop(meeting_dir, ref, client, interaction, reporter, sleep, open_url, emit)
    except KeyboardInterrupt:
        # Gates wait 7 days — quitting mid-poll is a legitimate, clean outcome.
        interaction.notify(f"review paused — resume with: meetscribe review {meeting_dir}")
        # The loop may have persisted fresher state than our pre-loop snapshot.
        current = read_workflow_ref(meeting_dir) or ref
        emit(f"{ref['workflow_id']} {current.get('state') or 'unknown'}")
        return 0


def _loop(
    meeting_dir: Path,
    ref: dict,
    client,
    interaction: Interaction,
    reporter: Reporter,
    sleep,
    open_url,
    emit,
) -> int:
    workflow_id = ref["workflow_id"]
    wf: dict | None = None  # None = needs a fetch; a submit's 200 body fills it directly
    failures = 0
    stint_slept_s = 0.0  # how long the current processing stint has been polling
    stage = None  # an open Reporter.stage spanning the current processing stint

    def poll_wait() -> None:
        nonlocal stint_slept_s
        interval = (
            POLL_INTERVAL_SLOW_S
            if stint_slept_s >= POLL_BACKOFF_AFTER_S
            else POLL_INTERVAL_S
        )
        sleep(interval)
        stint_slept_s += interval

    def finish(code: int) -> int:
        # The one machine-useful stdout line (design §gate-rendering): everything
        # human-facing renders to stderr, so scripts read the outcome from here.
        emit(f"{workflow_id} {ref.get('state') or 'unknown'}")
        return code

    def end_stage() -> None:
        nonlocal stage
        if stage is not None:
            stage.__exit__(None, None, None)
            stage = None

    try:
        while True:
            if wf is None:
                try:
                    wf = client.workflow(workflow_id)
                except BkAuthError as e:
                    # Contract: "401 → actionable client error, never a retry loop" —
                    # re-sending a known-bad token only wastes the human's time.
                    interaction.notify(str(e))
                    return finish(2)
                except BkError as e:
                    failures += 1
                    if failures >= POLL_FAILURE_LIMIT:
                        reporter.warn(
                            f"giving up after {failures} consecutive failures ({e})"
                            f" — resume with: meetscribe review {meeting_dir}"
                        )
                        return finish(1)
                    reporter.warn(f"could not refresh workflow state ({e}) — retrying")
                    poll_wait()
                    continue
                failures = 0
                ref = _persist_state(meeting_dir, ref, wf)

            state = str(wf.get("state") or "")
            if state != "processing":
                end_stage()  # the stint (if any) is over — close the spinner
            if state == "done":
                reporter.info(f"review complete — {ref.get('web_url')}")
                return finish(0)
            if state == "failed":
                error = wf.get("error") or {}
                reporter.warn(
                    f"bk processing failed ({error.get('code', 'unknown')}):"
                    f" {error.get('message', '')}"
                )
                return finish(1)
            if state == "processing":
                # Design §engine: a spinner shows the loop is alive during the whole
                # (potentially minutes-long) stint — one stage, not one per poll.
                if stage is None:
                    stage = reporter.stage("waiting for bk to process the meeting")
                    stage.__enter__()
                poll_wait()
                wf = None
                continue
            if state != "awaiting_review":
                # An unknown state means bk grew a workflow step this client predates.
                return finish(
                    _web_fallback(wf.get("web_url") or ref.get("web_url"), interaction, open_url)
                )

            stint_slept_s = 0.0  # a later processing stint starts fresh at 5 s
            task = wf["tasks"][0]  # contract: awaiting_review carries exactly one task
            task_type = task.get("type")
            if task_type == "meeting_mapping":
                answer = interaction.review_mapping(build_mapping_view(task))
            elif task_type == "speaker_annotation":
                answer = interaction.review_speakers(build_speaker_view(task, meeting_dir))
            else:
                return finish(
                    _web_fallback(task.get("web_url") or wf.get("web_url"), interaction, open_url)
                )

            if isinstance(answer, Quit):
                interaction.notify(
                    f"review paused — resume with: meetscribe review {meeting_dir}"
                )
                return finish(0)
            if isinstance(answer, OpenWeb):
                return finish(
                    _web_fallback(task.get("web_url") or wf.get("web_url"), interaction, open_url)
                )

            if task_type == "meeting_mapping":
                body = mapping_submission(answer, task["revision"])
            else:
                body = speaker_submission(answer, task["revision"])
            submitted: dict | None = None
            while submitted is None:
                try:
                    submitted = client.submit_task(task["submit_url"], body)
                except BkAuthError as e:
                    # 401 is final everywhere (contract rule): surface the token
                    # problem instead of re-asking for answers bk will reject again.
                    interaction.notify(str(e))
                    return finish(2)
                except BkRevisionConflict as e:
                    # The task moved under us: re-fetch and re-render the same gate
                    # with the fresh payload/revision. No retry cap — each round
                    # needs a human answer.
                    interaction.notify(str(e))
                    wf = None
                    break
                except BkInvalidSubmission as e:
                    # bk rejected the answer body: show the server's message, re-ask
                    # the gate (wf is unchanged, so the same gate re-renders).
                    interaction.notify(str(e))
                    break
                except BkError as e:
                    # Transport trouble during submit — the answers are still in
                    # memory (design §errors), so offer re-sending the SAME body
                    # instead of re-walking the gate. A lost submit that actually
                    # landed answers the resend with a clean 409, so retrying is safe.
                    if interaction.ask_retry(f"submit failed ({e}) — retry?"):
                        continue
                    interaction.notify(
                        f"review paused — resume with: meetscribe review {meeting_dir}"
                    )
                    return finish(0)
            if submitted is None:
                continue  # 409 (re-fetch) or 422 (re-ask) — nothing new to persist
            wf = submitted
            ref = _persist_state(meeting_dir, ref, wf)
    finally:
        # Ctrl-C or an early return must never leave a spinner on screen.
        end_stage()
