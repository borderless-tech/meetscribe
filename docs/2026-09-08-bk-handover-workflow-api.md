# Handover: meetscribe integration API (for borderless-knowledge)

Audience: the implementor on the bk side. Goal: make bk **meetscribe-aware** — a small,
versioned API through which meetscribe (the meeting recorder/transcriber CLI, soon
tray app) queries calendar context, uploads `.mscribe` bundles, and drives bk's existing
async annotation workflow interactively from the user's machine.

Read in this order:
1. `mscribe-format-v2.md` — the bundle you ingest (you have this already; v2 is additive).
2. `meetscribe-bk-contract-v1.md` — **the contract. It is the spec; this handover is the
   narrative around it.** A copy of the contract must live in the bk repo and change in
   lockstep with ours.

## Why this exists (the UX we're building toward)

Today the flow after a recording is: user uploads the bundle to bk, then has to log into
bk, find the meeting, annotate speakers one by one, and map the transcript to a calendar
meeting. We're inverting the interactive part: **bk keeps the storage, workflow, and
matching intelligence; meetscribe becomes the front-end for the review steps.** The user
records, gets a small confirm/correct dialog right in the tool they already have open, and
the finished transcript+note appears in bk — without opening bk at all.

meetscribe has two things your web UI can't have: the raw audio on local disk (it can play
a 3-second snippet of an unknown speaker during annotation — in local-transcription mode
that audio never left the machine) and presence at the moment the meeting ends, when the
user still remembers who "the deep voice" was.

## What to implement (summary — details in the contract)

Five endpoints under `/api/meetscribe/v1/`, Bearer-token auth:

| # | Endpoint | Purpose | Likely maps to |
|---|---|---|---|
| 1 | `GET /capabilities` | contract + supported `format_version`s preflight | new, trivial |
| 2 | `GET /meetings?around=<ts>` | calendar events near an instant, **attendees with email AND name** (contacts-resolved) | your calendar + contacts models |
| 3 | `POST /bundles` | bundle upload → `202` + `workflow_id`; idempotent via `Idempotency-Key` | your existing upload → processing pipeline |
| 4 | `GET /workflows/{id}` | state machine + pending tasks (JSON) | your existing async workflow, exposed |
| 5 | `POST …/tasks/{id}/submit` | user's answers; `revision`-guarded; returns fresh state | your existing annotation actions |

The workflow states are `processing → awaiting_review → done | failed`. Pending review
steps are **tasks**: `speaker_annotation` and `meeting_mapping` at launch (you may later
merge them into one review task — see the forward-compat rules).

## Design decisions you should know the reasons for

- **`web_url` everywhere, always.** Every state and every task carries the bk web page for
  the same step. That's the escape hatch that lets you ship new workflow steps without
  waiting for a meetscribe release — old clients just deep-link the user to your UI. Please
  treat this as a hard requirement, it's the whole forward-compat story.
- **Speakers are referenced by transcript label only** (`me`, `spk_0`, …). meetscribe
  produced the transcript and still has it; don't re-serve transcript content in task
  payloads. The labels are the shared keys — they're identical in `transcript.json` and
  `embeddings.npz` inside the bundle.
- **Suggestions, not questions.** The `speaker_annotation` payload should carry your best
  guesses: pgvector matches against previous meetings' cluster/turn vectors
  (`source: "voice_match"`), calendar attendees (`source: "roster"`), and the uploading
  user for `me` (`source: "owner"`). The common case in the client is then one-click
  confirmation. The embeddings you need ship in every bundle (192-dim CAM++; dimension and
  model identity in `meta.json` — reject vectors whose `embedding_model_sha256` doesn't
  match your stored history, they'd be incomparable).
- **Attendees carry `email` AND `name`** (nullable) — you manage contacts, so resolve names
  where you can; meetscribe feeds them into its ASR keyterm boost, which measurably improves
  proper-noun recognition. This is why the meetings query happens *before* recording.
- **Idempotency + revisions.** Upload dedupe via `Idempotency-Key` (meetscribe sends its
  `meeting_id` from `meta.json`); task submits carry `revision`, `409` on mismatch. Users
  will close laptops mid-annotation and retry hours later; every mutating call must be
  retry-safe, and workflows must persist until completed.
- **Empty results are normal.** No calendar match (`meetings: []`) and ad-hoc meetings
  (`meeting_mapping` with `create`) are first-class paths, not errors.

## What meetscribe will do (so you don't build it)

- All interactive rendering: talk times, sample utterances, local audio playback, pickers.
- Roster confirmation **before** upload (names → keyterms, count → diarizer hint), so
  bundles arrive with a `?meeting_id=` most of the time.
- Polling (~5 s with backoff), resume via persisted `workflow_id`, `web_url` fallback for
  anything it doesn't understand.
- meetscribe stores **no person identities** — `person_id` is opaque to it.

## Rollout & testing

- **Fixtures are the contract tests.** Please produce canonical JSON fixtures for every
  response/state/task in the contract (the contract doc's examples are the starting point)
  and keep them in the bk repo next to your contract copy; we vendor the same set and run
  our client against them offline. A contract change without a fixture change in the same
  PR is the drift that bit us with the bundle `format_version` — let's not repeat it.
- Suggested build order: 1 (capabilities) → 3 (upload+workflow_id) → 4 (state, tasks
  read-only) → 5 (submits) → 2 (meetings query). meetscribe adopts in the same order
  (see `docs/plans/2026-09-08-transition-roadmap.md`), so we can integrate incrementally —
  capabilities+upload alone already kills the manual-upload step.

## Open points for you to decide (tell us, we'll follow)

1. Token issuance/scoping: per-user API token from bk settings? That's our assumption.
2. The matching window for `?around` (we assumed ±30 min; it's your `capabilities` field).
3. Whether/when to merge `speaker_annotation` + `meeting_mapping` into one review task.
4. Where the contract copy + fixtures live in the bk repo.
