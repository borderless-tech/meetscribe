# Phase 3 — calendar-aware recording, design

Status: decided 2026-10-10 (autonomous session; decisions below are final) → ships as
**0.3.2**. Contract basis: endpoint 2 of `docs/meetscribe-bk-contract-v1.md` (verified
live in bk: `apps/api/src/routes/meetscribe.ts::meetingsAround`); fixtures
`tests/fixtures/bk/meetings-around{,-empty}.json`. Roadmap: the vacation note's P3 —
"`GET /meetings?around=now` at record start (opportunistic, NEVER blocks recording),
roster confirm → glossary/keyterms + `--speakers`, `?meeting_id=` on upload."

## Shape of the feature

Recording stays sacred: the calendar is consulted in a way that can never delay, block,
or fail a recording. The flow splits the work across the two moments the user is
naturally present:

1. **At record start** (before ffmpeg): one opportunistic `GET /meetings?around=<now>` —
   single attempt, **3 s timeout, no retries** (the retry discipline exists for uploads;
   here a slow bk must cost at most 3 s once). Every failure (bk unconfigured, token
   missing, network, 4xx/5xx, bad JSON) silently degrades to today's behavior — one
   `reporter.info` line in verbose mode, nothing else. The raw candidate list is cached
   to the meeting dir as **`bk-calendar.json`** (`{queried_at, around, meetings: [...]}`;
   never a bundle member, tolerant reader like `bk-workflow.json`).
2. **After Ctrl-C** (where the participant prompts already live, same TTY gate): when
   candidates were cached, the user picks the matching event or skips — then the picked
   event's **named attendees are appended to the glossary** (spelling hints → LLM cleanup
   + nova-3 keyterms, exactly what participant names do today), the **attendee count
   becomes the default** answer of the existing "how many people?" prompt (Enter
   accepts), the manual name prompts are skipped (the roster already provided the
   names), and `picked_meeting_id` is added to `bk-calendar.json`. Skipping (or no
   candidates, or no TTY) leaves today's flow byte-identical.
3. **At upload** (`upload` subcommand AND record/process auto-upload): a picked id in
   `bk-calendar.json` is sent as `?meeting_id=` (the `upload_bundle` parameter has
   existed since P2). bk re-verifies attendance server-side and degrades a bad id to its
   normal matching — params never widen scope — so sending it is always safe. New
   `upload --meeting-id <id>` flag overrides the cache (manual escape hatch; would have
   bypassed the `[object Object]` auto-match found on 2026-10-10).

**No reconcile prompt at upload time** (deliberate YAGNI): for recordings made offline
or with bk unconfigured, bk already reconciles server-side from `meta.started_at`
(`meetingCandidates`) and parks on the `meeting_mapping` gate that P4 renders. A second
client-side pick UX would duplicate that for no gain.

## Interaction seam growth

`Interaction` gains a third gate-shaped method (the tray GUI will want the same form):

```python
@dataclass(frozen=True)
class CalendarCandidate:
    meeting_id: str; title: str
    start: str | None; end: str | None            # tz-aware ISO or null
    attendees: tuple[RosterEntry, ...]             # person_id unused here → ""

@dataclass(frozen=True)
class MeetingPickView:
    candidates: tuple[CalendarCandidate, ...]

class Interaction(Protocol):  # addition
    def pick_meeting(self, view: MeetingPickView) -> str | None: ...  # id | None=skip
```

`NullInteraction` raises (as everywhere), `ScriptedInteraction` pops a canned answer,
`RichInteraction` renders a numbered list (title, local time range, attendee names) —
digits pick, **Enter accepts the sole candidate when there is exactly one**, otherwise
Enter/`s` skip. `record.run` gains `interaction: Interaction | None = None`; the CLI
passes a `RichInteraction` only when stdin+stderr are ttys (same rule as the review
auto-continue); `None` skips the pick entirely.

## Configuration

New `[bk].calendar` bool (**default true** — the query only happens when bk is already
explicitly configured, so no new opt-in is needed; the key is the off switch), flag
`--calendar/--no-calendar` on `record`, None-default pattern, no env layer (like
`auto_upload`/`bundle`). Non-bool value = `ConfigError`. The record-start query resolves
base_url + token itself (a `token_cmd` runs once here — pinentry at record start is
fine, the established pattern); any `ConfigError` just skips the query.

## New bk client call

`BkClient.meetings_around(around_iso, timeout_s=3.0)` → the parsed `{"meetings": [...]}`
dict. GET `build_url(config, "/meetings") + "?around=" + urlencode(iso)`; **attempts=1**
(a new `_request` knob, default unchanged for all other calls) and the short per-call
timeout override. 401/403 still raise `BkAuthError` (callers of the opportunistic path
swallow it like any `BkError` — but record's eager validation means it rarely fires).

## Out of scope

Doctor/status additions, meta.json changes (`format_version` stays 2, bundle members
untouched), auto-accepting a candidate without a TTY (bk's server-side matching covers
it), re-querying at process/upload time. No new runtime deps.

## Testing

Offline, fixture-driven: `meetings-around{,-empty}.json` for the client call (URL/query
encoding, single-attempt behavior via opener call count, timeout override observed at
the opener seam); `bk-calendar.json` round-trip incl. corrupt-file tolerance; record
flow driven with fakes (fake `record_tracks`, fake bk client, `ScriptedInteraction`) —
assert the glossary file contents, the `num_speakers` handed to `pipeline.run`, the
cache file, and that a failing query leaves the flow identical to today; upload tests
assert the `calendar_meeting_id` the fake client received (flag > cache > absent). Same
binding test policy as P4: no tautological tests.

## Versioning

0.3.2; changelog under `[Unreleased]` as pieces land, release cut at the end.
