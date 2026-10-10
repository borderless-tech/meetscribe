# Phase 4 — interactive review loop (CLI), design

Status: validated in design session 2026-10-10 → ships as **0.3.1**.
Contract basis: `docs/meetscribe-bk-contract-v1.md` (v1 AGREED, fixtures vendored in
`tests/fixtures/bk/`). Roadmap context: `docs/plans/2026-09-08-transition-roadmap.md`.

## Goal

After a bundle is uploaded, the user completes bk's review gates without leaving the
terminal: confirm speaker suggestions, map the meeting to a calendar event, listen to
audio snippets, and end at a terminal workflow state — with bk's web UI always available
as fallback (`web_url`, contract rule 1). The second deliverable is the **Interaction
seam**: a front-end protocol the later tray GUI (P5) reuses unchanged.

## Command surface

- **New subcommand `meetscribe review [dir]`** — dir defaults like `status` to the newest
  meeting with a `bk-workflow.json`. Reads the persisted ref, fetches state, enters the
  loop. Exit codes: config problems **2**; workflow `failed` or unrecoverable API error
  **1**; `done` or user quit (`q`/Ctrl-C — gates wait 7 days, quitting is legitimate)
  **0** with a resume hint.
- **Auto-continue:** after a successful `upload`, and after auto-upload in
  `record`/`process`, flow straight into the loop — only when **stderr is a TTY**.
  Non-interactive contexts keep today's behavior (print URL, exit 0). No opt-out flag in
  v1 (quitting is one keypress; add `--no-review` later only if it annoys).
- `bk.py` grows `submit_task` (POST, revision echo, 200/409/422 → typed results), same
  stdlib-urllib + injected-opener style as the existing calls.

**Out of scope for 0.3.1:** `--yes` bulk auto-confirm, `?meeting_id=` on upload (P3),
any GUI.

## The Interaction seam (`interaction.py`)

**Task-level, not prompt-level.** A primitive `select/confirm` protocol would force every
front-end into a sequential wizard; a GUI wants one form per gate (all speakers, dropdowns,
play buttons, one Submit). So the seam exchanges **view-models for answers, one method per
gate type**:

```python
class Interaction(Protocol):
    def review_speakers(self, view: SpeakerReviewView) -> SpeakerAnswers | OpenWeb | Quit: ...
    def review_mapping(self, view: MappingReviewView) -> MappingAnswer | OpenWeb | Quit: ...
    def notify(self, message: str) -> None: ...
```

View-models are frozen, presentation-neutral, **complete**: per speaker the label, talk
time, sample utterances, ranked suggestions (confidence/source), roster, and
`Snippet(path, start, duration)` refs — so each front-end owns playback natively (CLI
shells out to ffplay; GUI uses a play button). Playback is not an engine concern.

A new bk task type ⇒ new seam method + client update anyway; until then the engine's
`web_url` fallback covers it — mirroring the contract's own extension rule.

Calls are synchronous; the GUI later runs the engine on a worker thread and bridges to
its event loop (standard pattern).

Implementations: **`RichInteraction`** (CLI-only, stderr, like `RichReporter`),
**`ScriptedInteraction`** (tests: canned answers, records prompts),
**`NullInteraction`** (default everywhere; raises on any question so no non-interactive
path can block on input).

## The review loop engine (`review.py`)

Pure logic; injected seams only: a bk client (the opener-seam functions in `bk.py`), an
`Interaction`, a `Reporter`, and a `sleep` callable (tests pass a recording fake).

State machine driven by `GET /workflows/{id}`:

- **processing** → poll 5 s, backoff to 15 s after the first minute; spinner via
  Reporter. Ctrl-C = clean exit 0 + resume hint.
- **awaiting_review** → exactly one task (contract). Dispatch on `type`; unknown types
  and `OpenWeb` answers → print `web_url`, try `webbrowser.open`, exit 0.
- **done** → summary + `web_url`, exit 0. **failed** → `error.code/message`, exit 1.

**Submit:** translate answers + task `revision` to the contract body. `200` returns the
full fresh workflow state — feed it straight back into the state machine (no extra GET).
`409` → notify, re-fetch, re-render the same gate with fresh payload/revision (no retry
cap — each round needs a human answer). `422` → show server message, re-ask.

**View-model building** also lives in `review.py`: talk time + sample utterances from
`transcript.json` (longest segments first); snippets reference the system-track WAV in
`raw/`. Missing `raw/` → no snippets in the view; CLI renders "(audio unavailable)".

After every state change, rewrite `bk-workflow.json` (`state`, `checked_at`) — same
non-crashing read semantics as today, so `status` stays accurate for free.

## Gate rendering (RichInteraction)

**Speaker annotation.** One block per speaker, top to bottom: header
(`spk_0 — 14:32 talk time`), two sample utterances (longest, truncated), ranked
suggestions as a numbered list with hint column (`1) Anna Meier  (voice match, high)`),
letter actions: `p` play snippet (cycles top-3 longest segments), `n` new person
(name, email optional → contract's `{"person": {...}}` create form), `u` leave
unassigned, `w` open web UI, `q` quit. **Enter confirms the preselection**: bk's
`preselected` when present, else the top high-confidence suggestion — one keystroke per
speaker in the common case, including `me` (owner-sourced, high). A final summary table
(label → name → source) with one confirm gates the submit; backing out restarts the
gate's walk (nothing has left the machine yet).

**Meeting mapping.** Candidates as a numbered list with `match` quality + time range;
preselect only a sole `high` candidate. `n` create ad-hoc (title prompt, default
`"Meeting <date>"`), hidden when `allow_create` is false. `w`/`q` as above.

**Playback:** `ffplay -nodisp -autoexit -loglevel error -ss <start> -t <dur> <wav>`,
blocking (snippets are seconds long); missing ffplay degrades to one notify.

All rendering to **stderr**; stdout gets only the final machine-useful line
(workflow id + state).

## Errors

- Network/5xx while polling: warn, keep polling with backoff — a flaky connection must
  not lose the session.
- Network failure during submit: answers are still in memory → offer retry/quit.
- `401` → exit 2 immediately, never a retry loop (contract rule). `BkError` messages
  keep naming the knob to turn.
- Corrupt/absent `bk-workflow.json` on `review <dir>` → clear exit-2 message pointing at
  `meetscribe upload`, never a crash.

## Testing

Offline, fixture-driven (vendored canonical set). No tautological tests (no
string-containment / was-called asserts) — assert behavior: submitted bodies (full
shape), exit codes, state transitions, derivations.

- `bk.submit_task` against `task-submit-*.json` fixtures: body shape, revision echo,
  200/409/422 mapping.
- Engine tests drive `review.run` with a scripted bk-client fake returning fixture
  sequences (`processing → awaiting_review(mapping) → awaiting_review(speakers) → done`;
  the 409 path; unknown-task fallback; `failed`) + `ScriptedInteraction`; assert
  submitted bodies and exit codes. Injected `sleep` records poll cadence.
- View-model tests: talk-time/sample/snippet derivation from a synthetic
  `transcript.json`, incl. the no-`raw/` degradation.
- `RichInteraction`: thin smoke tests (stderr rendering, scripted keypresses via
  injected input) — heavy logic deliberately lives engine-side.

## Versioning

0.3.1; changelog under `[Unreleased]` as pieces land. No `format_version` change
(`bk-workflow.json` was never a bundle member). No new runtime deps — stdlib + existing
rich/ffmpeg only.
