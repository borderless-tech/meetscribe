# Phase 4 — interactive review loop: Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this
> plan task-by-task (or drive it via the ultracode workflow, one agent per task).

**Goal:** `meetscribe review <dir>` (plus auto-continue after upload) walks bk's review
gates in the terminal: speaker annotation with one-key confirm + ffplay snippets, meeting
mapping, revision-guarded submits, `web_url` fallback — per the validated design
`docs/plans/2026-10-10-phase4-review-loop-design.md` (read it first, it is binding).

**Architecture:** three new seams/modules — `bk.submit_task` (transport), `interaction.py`
(task-level front-end protocol + view-models), `review.py` (pure engine + view-model
building) — wired by a `review` subcommand in `cli.py`. Everything testable offline via
injected fakes; `RichInteraction` is the only rich/tty code.

**Tech stack:** stdlib only (urllib, subprocess, webbrowser) + existing `rich`. No new
runtime deps. All commands run as `nix develop -c pytest -q …` (plain pytest is NOT on
PATH).

**Test policy (binding, user requirement):** no tautological tests — never assert "string
X in output Y" as the main claim, never assert "function was called" on an internal.
Assert *behavior*: full submitted request bodies vs. the contract fixtures, returned
answers for scripted keypresses, exit codes, state transitions, computed view-model
values, written files. Asserting what an injected *seam* received (e.g. the recorded
`sleep` cadence, the URL an injected `open_url` got) is fine — that's the observable
contract of the seam.

**Contract fixtures** (vendored, canonical): `tests/fixtures/bk/
workflow-{processing,awaiting-review-meeting-mapping,awaiting-review-speaker-annotation,done,failed}.json`,
`task-submit-{speaker-annotation,meeting-mapping-select,meeting-mapping-create}.json`,
`error.json`. Engine tests load these — never hand-write workflow JSON that a fixture
already covers. NOTE: `submit_url`/`state_url` in fixtures are **host-relative paths**
(`/api/meetscribe/v1/...`), NOT ids — build the full URL as
`config.base_url.rstrip('/') + submit_url`.

Commit after every task; run the full suite (`nix develop -c pytest -q`) before each
commit. Branch: `feat/review-loop`.

---

### Task 1: `bk.submit_task` + typed 409/422

**Files:** Modify `src/meetscribe/bk.py`; Test `tests/test_bk.py` (append).

1. Failing tests first (scripted `opener` seam, exactly like the existing `test_bk.py`
   tests — read a few before writing):
   - `submit_task(submit_url, body)` POSTs to `base_url + submit_url` with
     `Content-Type: application/json`; the opener-recorded request body, parsed back from
     JSON, **equals** the loaded `task-submit-speaker-annotation.json` fixture dict when
     that fixture is passed as `body`. 200 response (use
     `workflow-processing.json` bytes) → returns the parsed workflow state dict.
   - HTTP 409 → raises `BkRevisionConflict` (new subclass of `BkError`); **no retry**
     (opener called exactly once — the count on the scripted seam is the behavior).
   - HTTP 422 with `error.json` body → raises `BkInvalidSubmission` carrying the
     contract's `error.message`.
   - HTTP 500 then 200 → retried like the other calls (existing retry discipline).
2. `nix develop -c pytest tests/test_bk.py -q` → new tests FAIL (AttributeError).
3. Implement: two exception subclasses; `submit_task(self, submit_url: str, body: dict)
   -> dict` using `_request` — but 409/422 must bypass the generic retry/raise: extend
   `_request` with an optional hook or catch `HTTPError` codes 409/422 before the
   generic branch (keep `_request`'s shape; a small `final_codes` param mapping code →
   exception factory is acceptable; do NOT duplicate the retry loop).
4. Full suite green → commit `feat(bk): submit_task with typed 409/422 results`.

### Task 2: `interaction.py` — view-models, answers, protocol, test doubles

**Files:** Create `src/meetscribe/interaction.py`; Test: none standalone (consumed by
Tasks 3–5; do not write structural "dataclass has field" tests — that's tautology).

Frozen dataclasses (times are float seconds, matching `types.py`):

```python
@dataclass(frozen=True)
class Snippet:            # a playable span in a local WAV
    path: Path; start: float; duration: float

@dataclass(frozen=True)
class Suggestion:
    person_id: str; name: str; confidence: str | None
    source: str; preselected: bool = False

@dataclass(frozen=True)
class SpeakerView:
    label: str; talk_time_s: float
    samples: tuple[str, ...]          # sample utterance texts, longest first
    suggestions: tuple[Suggestion, ...]
    snippets: tuple[Snippet, ...]     # longest first; empty when raw/ is gone

@dataclass(frozen=True)
class RosterEntry:
    person_id: str; email: str | None; name: str | None

@dataclass(frozen=True)
class SpeakerReviewView:
    speakers: tuple[SpeakerView, ...]; roster: tuple[RosterEntry, ...]; web_url: str

@dataclass(frozen=True)
class MappingCandidate:
    meeting_id: str; title: str; start: str | None; end: str | None; match: str

@dataclass(frozen=True)
class MappingReviewView:
    candidates: tuple[MappingCandidate, ...]; allow_create: bool; web_url: str

@dataclass(frozen=True)
class NewPerson:
    name: str; email: str | None = None

@dataclass(frozen=True)
class Assignment:         # exactly one of person_id / new_person set, or neither (= unassigned)
    label: str; person_id: str | None = None; new_person: NewPerson | None = None

@dataclass(frozen=True)
class SpeakerAnswers:
    assignments: tuple[Assignment, ...]

@dataclass(frozen=True)
class MappingAnswer:      # exactly one of meeting_id / create_title set
    meeting_id: str | None = None; create_title: str | None = None

class OpenWeb: ...        # singleton-style sentinels (instances compared by isinstance)
class Quit: ...

class Interaction(Protocol):
    def review_speakers(self, view) -> SpeakerAnswers | OpenWeb | Quit: ...
    def review_mapping(self, view) -> MappingAnswer | OpenWeb | Quit: ...
    def notify(self, message: str) -> None: ...
```

`NullInteraction`: every review method raises `RuntimeError("interactive review requires
a front-end — this is a bug, not a user error")`; `notify` is a no-op.
`ScriptedInteraction(answers: list)`: pops the next canned answer per review call and
records every view it was shown (`.seen`), `notify` appends to `.messages`. Both live in
`interaction.py` (NullInteraction is production code — the default; ScriptedInteraction
is three lines, keeping it next to the protocol mirrors `NullReporter`).

Commit: `feat(interaction): task-level front-end seam (views, answers, doubles)`.

### Task 3: `review.py` — view-model building from local artifacts

**Files:** Create `src/meetscribe/review.py`; Test: Create `tests/test_review_views.py`.

1. Failing tests: build a tmp meeting dir with a synthetic `transcript.json` via
   `output.write_transcript` (3 speakers; give `spk_0` segments of 10 s + 4 s + 1 s on
   track "system", `me` 5 s on "mic") and empty `raw/mic.wav`/`raw/system.wav` files.
   Then with the `workflow-awaiting-review-speaker-annotation.json` task payload:
   - `build_speaker_view(payload, meeting_dir)` → `spk_0.talk_time_s == 15.0`; samples
     are the segment texts longest-first; `snippets[0] == Snippet(raw/system.wav, start
     of the 10 s segment, 10.0)` and at most 3 snippets; `me`'s snippets point at
     `raw/mic.wav`; suggestions/roster mirror the payload (incl. `preselected`,
     `confidence: null` → `None`); labels in the payload but absent from the transcript
     still appear (talk time 0, no samples — bk knows speakers we might have lost).
   - delete `raw/` → same call yields `snippets == ()` for every speaker (no error).
   - `build_mapping_view(payload)` from the mapping fixture → candidates/allow_create
     mapped faithfully (null start/end → None).
2. Run → FAIL. 3. Implement (pure functions; read the transcript via
   `output.read_transcript`; truncate samples to ~120 chars at word boundary).
4. Suite green → commit `feat(review): view-model building from transcript + raw WAVs`.

### Task 4: `review.py` — the engine

**Files:** Modify `src/meetscribe/review.py`; Test: Create `tests/test_review_engine.py`.

Signature (all seams injected; NO direct `time.sleep`, `webbrowser`, `print`):

```python
def run_review(meeting_dir, client, interaction, reporter=NullReporter(),
               *, sleep=time.sleep, open_url=webbrowser.open) -> int
```

Behavior to implement AND assert (drive with a `FakeBkClient` returning fixture
sequences + `ScriptedInteraction`; every test asserts exit code + what reached the fake
client / ref file):

- **Happy path:** `processing ×3 → awaiting_review(mapping) → [submit] → processing →
  awaiting_review(speakers) → [submit] → done`. Scripted answers:
  `MappingAnswer(meeting_id=…)`, then `SpeakerAnswers(…)` matching the
  `task-submit-speaker-annotation.json` fixture. Assert: the two bodies the fake client
  received **equal** the corresponding `task-submit-*.json` fixtures (full dict
  equality — this is the contract test); recorded sleeps are 5,5,5 (then 15 s after the
  first 60 s — cover with a 14-poll sequence asserting the cadence switch); exit 0;
  `bk-workflow.json` ends with `state: done` and a fresh `checked_at`.
- **Submit response is the next state:** after a 200 the returned workflow dict is used
  directly (fake client counts `workflow()` GETs — exactly the polls, no extra fetch).
- **409 path:** first submit raises `BkRevisionConflict` → engine re-fetches, re-renders
  (ScriptedInteraction `.seen` grows by one more view carrying the **new** revision's
  payload), second submit body carries the new revision. Exit 0.
- **422 path:** `BkInvalidSubmission("…")` → `interaction.notify` received the server
  message, same gate re-asked, next answer submitted.
- **Unknown task type** (mutate a fixture copy's `type` to `"zz_new"`) → `open_url`
  seam received the task's `web_url`, exit 0, nothing submitted.
- **OpenWeb / Quit answers** → same web fallback / clean exit 0, nothing submitted.
- **failed** fixture → exit 1. **done** immediately → exit 0.
- **Polling network blip:** `workflow()` raising `BkError` once mid-poll → engine warns
  (reporter) and keeps polling; only N consecutive failures (use 5) abort with exit 1.
- **KeyboardInterrupt** during sleep (scripted sleep raises on call 2) → exit 0 and the
  ref file still holds the last known state.
- **Submission translation** is a pure function pair — `speaker_submission(answers,
  revision) -> dict`, `mapping_submission(answer, revision) -> dict` — unit-tested
  against ALL THREE `task-submit-*.json` fixtures (select, create, annotation incl.
  unassigned `person_id: null` and `person` create forms).

Update `bk-workflow.json` after every fetched state change (merge pattern copied from
`cli._run_status`). TDD in the listed order (one behavior at a time), then commit
`feat(review): gate-walking engine (poll, submit, 409/422, fallbacks)`.

### Task 5: `RichInteraction` + ffplay playback

**Files:** Modify `src/meetscribe/interaction.py`; Test: Create
`tests/test_interaction_rich.py`.

`RichInteraction(console=None, *, readkey=None, read_line=None, player=None)` — console
defaults to `rich.console.Console(stderr=True)` (match `RichReporter`), `readkey`
defaults to a tiny raw-tty single-key reader (termios; implement in `interaction.py`),
`read_line` to `input`, `player` to `play_snippet` (below). Tests inject all four.

Keymap per speaker (design §gate-rendering): Enter → preselection (bk `preselected`,
else top high-confidence suggestion; **no preselection → Enter is ignored, re-read key**);
`1..9` → that suggestion; `r` → pick from full roster (numbered page, digit choice);
`n` → new person (read_line name, then optional email, empty → None); `u` → unassigned;
`p` → play `snippets[i]`, cycling on repeat, re-ask same speaker (missing snippets →
notify "(audio unavailable)"); `w`/`q` anywhere → OpenWeb/Quit immediately. After the
walk: summary + confirm key — `y`/Enter submits, `e` restarts the walk. Mapping gate
analogous (digits, `n` create with title read_line defaulting to `Meeting <today>`,
hidden when `allow_create` is false; preselect only a *sole* `match == "high"`
candidate).

Tests are behavioral — scripted keypresses in, returned answer out (NOT output-string
asserts): Enter,Enter,… on the speaker fixture view → `SpeakerAnswers` equal to the
preselected set; `2` → second suggestion; `n`+lines → `NewPerson`; `u` → unassigned;
`q` mid-walk → `Quit`; `p` → recorded player got `snippets[0]`, then the answer still
collected; `e` at summary → walk restarts (ScriptedInteraction-style key list long
enough, final answers from the second walk win).

`play_snippet(snippet, *, run=subprocess.run)`: builds
`["ffplay","-nodisp","-autoexit","-loglevel","error","-ss",f"{s.start:.3f}","-t",f"{s.duration:.3f}",str(s.path)]`
— expose `ffplay_argv(snippet)` as a pure function and test THAT; `FileNotFoundError`
→ return a "ffplay not found — install ffmpeg" message for `notify`, don't raise.

Commit: `feat(interaction): rich terminal front-end + ffplay snippet playback`.

### Task 6: CLI wiring — `review` subcommand + auto-continue

**Files:** Modify `src/meetscribe/cli.py`, `src/meetscribe/pipeline.py`;
Test: `tests/test_cli.py` (append; read the existing style first).

1. Parser: `review` subcommand, `dir` positional `nargs="?"`. Default dir = newest
   meeting dir (by name, they're timestamped) under the configured meetings dir that has
   a `bk-workflow.json` — factor the meeting-scan from `_run_status` into a helper both
   use (`_find_meetings(root)`).
2. `_run_review(dir_arg)`: config/`_resolve_bk_client` errors → exit 2 (reuse
   `config_error_message`); absent/corrupt ref or no meeting found → exit 2 with a
   message pointing at `meetscribe upload <dir>`; else build `RichInteraction` +
   reporter and return `review.run_review(...)`.
3. Auto-continue: in `_run_upload` after the success prints and in
   `pipeline._upload_to_bk` after persisting the ref: when `sys.stderr.isatty()`, call
   `review.run_review` with a fresh RichInteraction — but its exit code must NOT replace
   upload semantics: `_run_upload` returns the review exit code (review is now the task);
   `_upload_to_bk` ignores it (process run stays exit 0, design §command-surface).
   Factor as `cli._maybe_review(meeting_dir, client, reporter, isatty=sys.stderr.isatty)`
   so tests inject `isatty=lambda: False/True`.
4. Tests: `review` with no uploaded meeting → exit 2 + hint; `_run_upload` with
   `isatty` False behaves byte-identically to today (existing tests must stay green
   unchanged — that IS the assertion); with True, a monkeypatched `run_review` recorder
   receives the meeting dir and its return value becomes the exit code.
5. Suite green → commit `feat(cli): review subcommand + post-upload auto-continue`.

### Task 7: docs + changelog (no release yet)

**Files:** Modify `CHANGELOG.md` (`[Unreleased]`), `CLAUDE.md` (short § for the review
loop: seam locations, exit-code rules, how to fake it in tests), `README.md` usage block
if it lists subcommands. No version bump — release 0.3.1 is cut after the review passes.
Commit: `docs: review loop usage + changelog`.

---

**Final gate (after all tasks):** `nix develop -c pytest -q` fully green AND
`nix flake check` green. Then code review (lenses: contract compliance against
`docs/meetscribe-bk-contract-v1.md`; test quality per the policy above; conformance to
design doc + CLAUDE.md conventions — stderr/stdout split, exit codes, None-default flag
pattern, no new deps) with adversarial verification of findings before fixing.
