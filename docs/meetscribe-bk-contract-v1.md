# meetscribe ↔ borderless-knowledge API contract, version 1

Status: **draft — pending agreement from the bk side.** This document is the single source of
truth for the integration; a copy lives in both repos and MUST be changed in lockstep (same
review, both copies). Consumer-side context: `docs/mscribe-format-v2.md` (the bundle format),
`docs/plans/2026-09-08-transition-roadmap.md` (when meetscribe adopts which endpoint).

## Roles

- **bk** owns storage, the async processing workflow, person identities, and the matching
  intelligence (voice ↔ person via pgvector, roster from calendar + contacts).
- **meetscribe** owns capture, transcription, and **all interactive steps**: it renders bk's
  pending tasks locally (it has the transcript and the raw audio — including local playback
  of speaker snippets) and submits the user's answers back. meetscribe never stores person
  identities; the transcript speaker labels (`me`, `spk_N`) are the shared keys.

## Conventions

- Base path: `https://<bk-host>/api/meetscribe/v1/` — the `v1` is `contract_version` 1.
- Auth: `Authorization: Bearer <token>` on every endpoint. bk issues the token; meetscribe
  stores it in its config file. `401` → actionable client error, never a retry loop.
- All timestamps are tz-aware ISO 8601 **with offset** (same convention as `meta.json`).
- All responses are JSON; errors use `{"error": {"code": "...", "message": "..."}}`.
- **Additive changes do not bump the version** (clients must ignore unknown fields).
  Renames/removals/semantic changes bump `contract_version` (= new base path).

## Forward-compatibility rules (the load-bearing part)

1. Every workflow state and every task carries a **`web_url`** — the bk web page where the
   same step can be completed manually. A client that encounters an unknown `state` or task
   `type` MUST fall back to showing `web_url`. This is what lets bk add workflow steps
   without breaking older meetscribe versions.
2. meetscribe calls `capabilities` before uploading and refuses (with a clear message) if its
   bundle `format_version` is unsupported — the "unsupported format version" failure becomes
   a pre-flight message instead of a rejected upload.
3. Task submissions carry the task's `revision`; bk answers `409` on mismatch, the client
   re-fetches state. Combined with an `Idempotency-Key` on upload this makes every mutating
   call retry-safe.

## Endpoints

### 1. `GET /capabilities`

```jsonc
{
  "contract_version": 1,
  "mscribe_format_versions": [1, 2],     // bundle versions bk can ingest
  "meeting_query_window_minutes": 30     // informational: bk's matching window (±)
}
```

### 2. `GET /meetings?around=<iso8601>`

Calendar events overlapping / near the given instant (typically "now" at record start, or
`meta.json.started_at` when reconciling later). Multiple candidates are possible — the
client picks or asks the user. Empty list is a normal answer (ad-hoc meeting).

```jsonc
{
  "meetings": [
    {
      "id": "cal_evt_8f3a",              // bk's stable event id — echoed back on upload
      "title": "Weekly Sync",
      "start": "2026-09-08T09:00:00+02:00",
      "end":   "2026-09-08T09:30:00+02:00",
      "attendees": [
        { "email": "anna@example.com", "name": "Anna Meier" },  // name from bk contacts…
        { "email": "guest@other.org",  "name": null }           // …null when unknown
      ]
    }
  ]
}
```

Attendees carry **email AND name** — bk manages contacts, not just the calendar, so it
resolves names where it can. meetscribe uses names for the glossary → ASR keyterm boost and
the roster UI; the attendee count seeds the local diarizer's `--speakers` hint.

### 3. `POST /bundles`

Body: the `.mscribe` zip, `Content-Type: application/zip`.
Headers: `Idempotency-Key: <meeting_id from meta.json>` — re-uploading the same bundle MUST
return the existing workflow, not create a duplicate.
Optional query: `?meeting_id=cal_evt_8f3a` when the calendar event is already known.

Response `202 Accepted`:

```jsonc
{
  "workflow_id": "wf_01j9",
  "state_url": "/api/meetscribe/v1/workflows/wf_01j9",
  "web_url": "https://bk.example.com/workflows/wf_01j9"
}
```

### 4. `GET /workflows/{workflow_id}`

```jsonc
{
  "workflow_id": "wf_01j9",
  "contract_version": 1,
  "state": "processing",              // "processing" | "awaiting_review" | "done" | "failed"
  "web_url": "https://bk.example.com/workflows/wf_01j9",   // ALWAYS present (rule 1)
  "error": null,                      // {"code","message"} when state == "failed"
  "tasks": []                         // non-empty when state == "awaiting_review"
}
```

Polling guidance: every ~5 s while `processing`, with backoff after the first minute.
Workflows persist — a client may re-attach hours or days later (meetscribe stores the
`workflow_id` in the meeting directory).

#### Task envelope

```jsonc
{
  "task_id": "task_a1",
  "type": "speaker_annotation",       // unknown type → client falls back to web_url
  "revision": 3,                      // echoed on submit; 409 on mismatch
  "submit_url": "/api/meetscribe/v1/workflows/wf_01j9/tasks/task_a1/submit",
  "web_url": "https://bk.example.com/workflows/wf_01j9#task_a1",
  "payload": { }                      // type-specific, below
}
```

#### Task type `speaker_annotation`

bk references speakers **only by transcript label** — meetscribe renders talk time, sample
utterances, and local audio snippets from its own data. `suggestions` are bk's ranked
guesses; `source` is `voice_match` (pgvector against previous meetings), `roster` (calendar
attendee), or `owner` (the uploading user, for `me`).

```jsonc
{
  "speakers": [
    { "label": "me",
      "suggestions": [ { "person_id": "p_owner", "name": "Christian H.", "confidence": "high", "source": "owner" } ] },
    { "label": "spk_0",
      "suggestions": [
        { "person_id": "p_123", "name": "Anna Meier", "confidence": "high", "source": "voice_match" },
        { "person_id": "p_456", "name": "Ben Otto",   "confidence": null,   "source": "roster" } ] },
    { "label": "spk_1", "suggestions": [] }
  ],
  "roster": [                          // calendar attendees + contact resolution, for pickers
    { "person_id": "p_123", "email": "anna@example.com", "name": "Anna Meier" },
    { "person_id": "p_456", "email": "ben@example.com",  "name": "Ben Otto" }
  ]
}
```

Submission body:

```jsonc
{
  "revision": 3,
  "assignments": [
    { "label": "me",    "person_id": "p_owner" },
    { "label": "spk_0", "person_id": "p_123" },
    { "label": "spk_1", "person": { "email": null, "name": "Externer Gast" } },  // create new
    { "label": "spk_2", "person_id": null }                                       // leave unassigned
  ]
}
```

#### Task type `meeting_mapping`

```jsonc
{
  "candidates": [
    { "meeting_id": "cal_evt_8f3a", "title": "Weekly Sync",
      "start": "2026-09-08T09:00:00+02:00", "end": "2026-09-08T09:30:00+02:00",
      "match": "high" }               // "high" | "medium" | "low" — timestamp/roster overlap
  ],
  "allow_create": true
}
```

Submission: `{ "revision": 2, "meeting_id": "cal_evt_8f3a" }` — or, for ad-hoc meetings,
`{ "revision": 2, "create": { "title": "Spontanes Gespräch mit X" } }`.

bk MAY combine annotation + mapping into a single review task type later; until then they
are two tasks that clients typically render as one screen.

### 5. `POST /workflows/{workflow_id}/tasks/{task_id}/submit`

Body: the type-specific submission (above), always including `revision`.
Responses: `200` with the **fresh full workflow state** (so the client needs no follow-up
GET), `409` on revision mismatch (client re-fetches and re-renders), `422` with the standard
error shape on invalid input.

## Client obligations (meetscribe side, for the record)

- Recording NEVER blocks on bk: the `meetings` query at record start is opportunistic
  (short timeout, cached to the meeting dir); reconciliation can happen at process/upload
  time via `meta.started_at`.
- Unknown states/task types → `web_url` fallback (rule 1). No retry loops on `401`.
- Both repos test against the **shared fixture set**: canonical JSON examples of every
  response/state/task in this document live as fixture files next to the contract copy in
  each repo; contract changes must update fixtures in the same change.
