# Phase 3 — calendar-aware recording: Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this
> plan task-by-task (or drive it via the ultracode workflow, one agent per task).

**Goal:** opportunistic calendar lookup at record start, post-recording roster confirm
feeding glossary + `--speakers`, and `?meeting_id=` on upload — per the binding design
`docs/plans/2026-10-10-phase3-calendar-design.md` (read it first).

**Architecture:** one new client call (`bk.meetings_around`, single-attempt + short
timeout), one new persisted ref (`bk-calendar.json`, sibling of `bk-workflow.json`), one
new Interaction method (`pick_meeting`), wired into `record.run`'s existing
post-recording prompt block and both upload paths. No new deps; all tests offline.

**Test policy (binding, as in P4):** no tautological tests — assert behavior: request
URLs/bodies at the opener seam, file contents, the values handed to injected seams,
returned answers for scripted keypresses, exit codes. Run everything as
`nix develop -c pytest -q`. Commit per task with the trailer
`Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. Branch: `feat/calendar-aware`.

Fixtures: `tests/fixtures/bk/meetings-around.json` (one candidate, one attendee with
`name: null`) and `meetings-around-empty.json`. Contract endpoint 2 says `id` is opaque
and attendees carry email AND possibly-null name.

---

### Task 1: `bk.meetings_around` + `bk-calendar.json` persistence

**Files:** Modify `src/meetscribe/bk.py`; Test `tests/test_bk.py` (append).

1. Failing tests (scripted opener, existing style):
   - `meetings_around("2026-09-08T09:00:25+02:00")` GETs
     `<base>/api/meetscribe/v1/meetings?around=<urlencoded iso>` with the bearer header;
     200 with the `meetings-around.json` fixture bytes → returns the parsed dict
     (assert equality with the loaded fixture). Empty fixture → `{"meetings": []}`.
   - **Single attempt:** opener scripted to raise a 500 → `BkError` after exactly ONE
     opener call (count asserted; contrast: the existing calls retry 3×).
   - **Timeout override:** the opener seam records the `timeout` argument it received —
     assert `3.0` (default) and a caller-passed value. Other calls still see
     `config.timeout_s`.
   - `write_calendar_ref(dir, ref)` / `read_calendar_ref(dir)` round-trip
     (`bk-calendar.json`); absent and corrupt file → `None` (mirror
     `read_workflow_ref`'s shape and tests).
2. Run → FAIL. 3. Implement: extend `_request` with `attempts: int = RETRY_ATTEMPTS`
   and `timeout_s: float | None = None` (None → config value) — do NOT duplicate the
   loop; `CALENDAR_REF_NAME = "bk-calendar.json"`; `meetings_around` with
   `timeout_s: float = 3.0`.
4. Full suite green → commit `feat(bk): meetings_around (opportunistic) + calendar ref`.

### Task 2: `[bk].calendar` config key

**Files:** Modify `src/meetscribe/config.py` (resolver + `TEMPLATE` `[bk]` block +
known-keys set); Test `tests/test_config.py` (append, existing style).

Tests: `bk_calendar(None, env, {})` → `Resolved(True, "default")`; config
`[bk] calendar = false` → `(False, "config")`; flag False over config true → `(False,
"flag")`; string `"false"` in TOML → `ConfigError`; the key does NOT warn as unknown.
No env layer (same as `auto_upload`). Implement, suite green, commit
`feat(config): [bk].calendar toggle (default on)`.

### Task 3: `Interaction.pick_meeting`

**Files:** Modify `src/meetscribe/interaction.py`; Test `tests/test_interaction_rich.py`
(append).

- `CalendarCandidate` / `MeetingPickView` frozen dataclasses per the design (reuse
  `RosterEntry`; `person_id=""` for calendar attendees).
- Protocol + `NullInteraction.pick_meeting` raises; `ScriptedInteraction` pops a canned
  answer and records the view in `.seen`.
- `RichInteraction.pick_meeting`: numbered list (title, local `HH:MM–HH:MM` when
  start/end present, attendee names comma-joined, null names rendered as the email);
  keys: digit → that candidate's `meeting_id`; Enter → the sole candidate's id when
  `len(candidates) == 1`, else skip (`None`); `s` → `None`. Behavioral tests: scripted
  keypresses → returned id/None for the single- and multi-candidate cases.
- Suite green → commit `feat(interaction): calendar meeting pick`.

### Task 4: record flow wiring

**Files:** Modify `src/meetscribe/record.py`, `src/meetscribe/cli.py` (parser:
`--calendar/--no-calendar` on record, default None; pass a `RichInteraction` to
`record.run` when stdin+stderr are ttys); Test `tests/test_record.py` (append; read the
existing `run()` tests first — they monkeypatch `record_tracks` and `pipeline.run`).

Behavior (TDD one case at a time):

1. **Query at start:** a pure-ish helper `query_calendar(cfg, reporter, *, client_factory,
   now_iso)` — resolves `[bk].calendar` (+ base_url/token incl. one `fetch_secret`),
   builds the client, calls `meetings_around(now_iso)`, returns the `meetings` list or
   `None`. EVERY exception path (`ConfigError`, `BkError`, missing config) returns
   `None` after at most a `reporter.info`. Tests inject a `client_factory` returning a
   fake (or raising) and assert: configured+ok → list; bk unconfigured / calendar=false
   / flag --no-calendar / BkError / BkAuthError → `None` and **no raise**.
2. In `run()`: call it right before `record_tracks` (after the out dir is created),
   with `now_iso = datetime.now().astimezone().isoformat()`; non-empty result →
   `bk.write_calendar_ref(root, {queried_at, around, meetings})`.
3. **Post-recording pick** (inside the existing `_stdin_is_tty()` block, BEFORE
   `ask_participants`): when a ref with candidates exists AND `interaction` is not None:
   `picked = interaction.pick_meeting(view)`; on a pick: rewrite the ref with
   `picked_meeting_id`, append the roster's non-null names to the glossary
   (`glossary.append`), and use `speakers_from_count(len(attendees))` as the default —
   the count prompt becomes "(Enter = N)" accepting the roster size, and the manual
   name prompts are skipped. On skip/None: today's flow untouched.
4. Tests (fake `record_tracks` writing silent-proof WAV stubs is already the pattern;
   fake `pipeline.run` recording its kwargs): picked → glossary file contains "Anna
   Meier" but NOT the null-named guest, `pipeline.run` got `num_speakers ==
   speakers_from_count(2)` (fixture: 2 attendees), ref file has `picked_meeting_id`;
   skipped → `num_speakers` from the normal prompt path; query returning `None` → no
   ref file, flow identical (assert the same kwargs as a no-bk run).
5. Suite green → commit `feat(record): calendar lookup + roster confirm`.

### Task 5: `?meeting_id=` on upload

**Files:** Modify `src/meetscribe/cli.py` (`_run_upload` + `--meeting-id` flag on the
upload subparser), `src/meetscribe/pipeline.py` (`_upload_to_bk`); Test
`tests/test_cli.py` (append).

- Precedence: `--meeting-id` flag > `read_calendar_ref(dir)['picked_meeting_id']` >
  nothing. Both upload paths pass it as `upload_bundle(..., calendar_meeting_id=...)`.
- Tests: fake `BkClient` records the kwarg — cache with picked id → forwarded; corrupt/
  absent ref → `None`; flag wins over cache; auto-upload path (via `pipeline`) forwards
  it too. Existing upload tests stay green unchanged.
- Suite green → commit `feat(upload): forward picked calendar meeting id`.

### Task 6: docs + changelog (no release in this task)

`CHANGELOG.md` `[Unreleased]`, CLAUDE.md (extend the bk/review sections with the
calendar flow + new key/flags + `bk-calendar.json`), README subcommand/flag mentions.
Suite green (test_version guards changelog shape) → commit `docs: calendar-aware
recording`.

---

**Final gate:** `nix develop -c pytest -q` AND `nix flake check` green. Then review
lenses (contract compliance incl. "recording NEVER blocks on bk"; test quality per
policy; repo conventions — None-default flags, stderr/stdout, exit codes, no new deps)
with adversarial verification, then fixes. Release 0.3.2 happens in the main session
after the workflow, not by an agent.
