# meetscribe Upload Bundle (`.mscribe`) — Design

Status: **design approved, not yet implemented.** Goal: turn the three loose output artifacts
(`transcript.json`, `embeddings.npz`, `meta.json`) into a single self-contained file that a
stateless HTTP sink (borderless-knowledge) can accept in one authenticated upload and ingest
asynchronously into a transcript-note.

## Problem / framing

- The sink is **borderless-knowledge** (a Notion clone). Upload → background workflow ingests the
  data, creates a transcript-note, and reconciles it with the uploader's calendar (matching a
  meeting name by time window).
- Backend is **stateless**; the upload is **authenticated**, so *who* uploaded (and their calendar)
  comes from the auth layer, **not** the payload.
- Three separate file uploads risk split-brain ("transcript landed, embeddings didn't"). We want
  **one upload → one thing**, atomic.
- **No human inspection** of the payload is required, so re-encoding the dense float32 embeddings
  into JSON (3–4× larger, lossy round-trip) buys nothing. Keep the binary layout.

## Decision: a zip container (Option A)

Bundle the three files we **already** write into one zip archive. `.npz` is itself a zip
internally, so nesting a zip-of-artifacts is idiomatic. Zero re-encoding; the existing `output.py`
writers are untouched.

Rejected alternatives:
- **Single `.npz`** (stuff the JSON in as byte arrays): forces numpy just to read text; weird idiom;
  saves nothing over zip.
- **MessagePack envelope**: re-encodes embeddings, needs a shared schema both ends, no size win. YAGNI.

## Container spec (the contract)

- **Extension:** `.mscribe` (a standard zip; any zip tool opens it).
- **Naming:** `meeting-<meeting_id>.mscribe`.
- **Compression:** `ZIP_DEFLATED`. Text compresses well; the already-packed `embeddings.npz` barely
  changes — harmless.
- **Internal layout — flat, fixed names:**

  ```
  transcript.json     # unchanged
  embeddings.npz      # unchanged
  meta.json           # enriched (below)
  ```

### `meta.json` enrichment

Add the fields the server needs that cannot come from auth:

```jsonc
{
  // ... existing model / embedding_model_sha256 / embedding_dim / sample_rate / version fields ...
  "meeting_id": "abc123",                       // promote here so meta alone identifies the bundle
  "started_at": "2026-08-02T14:03:11+02:00",    // wall-clock, ISO 8601 WITH offset
  "ended_at":   "2026-08-02T14:47:52+02:00",
  "duration_s": 2681.0,                          // mirrored from transcript.json for convenience
  "format_version": 1                            // container contract version; server gates on this
}
```

- `started_at` / `ended_at` are the **calendar-reconciliation match window**. The timezone offset is
  load-bearing: calendar events are wall-clock, so a bare UTC epoch would misalign the match.
- `format_version` lets borderless-knowledge branch/reject on future layout changes without guessing.

## Producer side (meetscribe)

Bundling is **decoupled from the pipeline** so it's reusable (future TUI: "bundle this directory").

- **Always write the directory** (unchanged; handy locally, keeps loose-file debuggability).
- **`bundle_dir(src_dir, out_path)`** in `output.py`: reads the three known filenames from a
  directory and zips them. **Single place** for validation — asserts the three files exist and
  `meta.json.format_version` is present before zipping. Uses `io.BytesIO` only if it ever needs to
  build parts in memory; the directory-first path just reads files off disk.
- **`run(..., bundle_path=None)`**: writes the dir always, then if `bundle_path` set, calls
  `bundle_dir(out_dir, bundle_path)`.
- **CLI:** `--bundle PATH` is **additive** (B1) — dir is always produced, bundle on demand.
- **Standalone subcommand** `meetscribe bundle <dir> [-o out.mscribe]` reuses the same `bundle_dir` —
  the TUI "bundle a certain directory" affordance for free.
- **Timestamps:** the recorder knows wall-clock start/stop. Capture `started_at` at record start and
  `ended_at` at stop, thread through the pipeline `result`, and pass into `build_meta`.

## Consumer contract (borderless-knowledge, informational)

Not owned here, but the format is built to make it easy:

1. **One POST**, body = `.mscribe` bytes, `Content-Type: application/zip`. Identity/calendar from
   auth. No multipart.
2. **Read order:** open zip → parse `meta.json` first → gate on `format_version` (reject unknown
   major versions with a clear error) → then `transcript.json` and `embeddings.npz`.
3. **Cheap self-consistency asserts** (catch truncated/corrupt uploads):
   - `embeddings.npz` `turn_vectors.shape[1] == meta.embedding_dim`
   - every `turn_ids` entry resolves to a segment/speaker in `transcript.json`
   - `started_at` / `ended_at` parse as tz-aware ISO 8601
4. **Atomicity:** one file → a partial upload is a short/unopenable zip → reject the whole thing. No
   split-brain. This is the stateless-backend win.

## Testing strategy

- `bundle_dir` round-trip: write a fixture dir, bundle it, re-open the zip, assert the three members
  are present and byte-identical to the originals (npz especially — no re-encoding).
- Validation: `bundle_dir` raises a clear error when a member is missing or `format_version` absent.
- `meta.json` enrichment: `build_meta` includes `started_at`/`ended_at` as tz-aware ISO 8601 and
  `format_version`.
- CLI: `--bundle` produces the file *and* leaves the directory intact; `meetscribe bundle <dir>`
  produces an equivalent archive.
- Keep the existing pure-pipeline tests rich-free and unchanged — bundling is an additive step.

## Open items / future

- `format_version` bump policy (additive members = same version; renamed/removed members = major bump).
- Optional later: a checksum member or speaker display-name hints, only if borderless-knowledge asks.
