# Phase 2 design: bk client + auto-upload

Roadmap P2 (`docs/plans/2026-09-08-transition-roadmap.md`); contract:
`docs/meetscribe-bk-contract-v1.md` (**v1 AGREED 2026-09-09** — the agreed semantics in it
are binding: single task per review, `mscribe_format_versions: [2]`, `max_bundle_bytes`,
idempotency dedupes live workflows only, 7-day gate timeout). This phase consumes bk
endpoints 1 (capabilities), 3 (upload), and 4 (workflow state, read-only display). Task
submits (endpoint 5) and the meetings query (endpoint 2) are Phases 4/3 — out of scope here.

Fixtures: bk's canonical set is not published yet — we author
`tests/fixtures/bk/*.json` from the contract's examples now, structured so bk's files are a
drop-in replacement later; divergence then = red test = contract bug to resolve.

Goal (the 0.3.0 headline): after `record`/`process`, the `.mscribe` bundle is uploaded to bk
automatically (explicit opt-in), the returned workflow reference is persisted with the
meeting, and `meetscribe status` shows where every meeting stands. Zero new dependencies
(stdlib urllib, same pattern as `deepgram.py`).

## Config additions (`[bk]` section stops being reserved)

```toml
#[bk]
#base_url = ""            # e.g. "https://bk.example.com"  (env MEETSCRIBE_BK_URL wins)
#token = ""               # personal bk API key            (env MEETSCRIBE_BK_TOKEN wins)
#token_cmd = ""           # shell command printing the token; mutually exclusive with token
#auto_upload = false      # upload the bundle to bk after record/process (explicit opt-in;
#                         # flag --upload/--no-upload wins per run)
```

- `token`/`token_cmd` reuse the lazy-secret pattern from `[deepgram].api_key_cmd`: the
  generic marker/executor moves to a shared helper (`SecretCmd` + `fetch_secret`;
  `ApiKeyCmd`/`fetch_api_key` stay as thin aliases so nothing breaks). Same rules: both set
  = `ConfigError`; never executed by `config` display, `validate`, doctor; executed eagerly
  at record start when auto-upload is on.
- `auto_upload` is a strict TOML bool, default **false** (privacy roadmap rule: uploading
  is never a surprise). Resolution: flag > config > default (no env layer, like
  bundle/cleanup).
- Precedence/origins/`meetscribe config` display extend automatically (token masked like
  api_key; `(via token_cmd)` for the marker).

## New module `src/meetscribe/bk.py`

Mirror of `deepgram.py`'s shape — frozen config, transport seam, typed error:

```python
@dataclass(frozen=True)
class BkConfig:
    base_url: str          # normalized: trailing slash stripped
    token: str
    timeout_s: float = 120.0   # uploads carry MBs on slow links

class BkError(RuntimeError): ...   # message always actionable (which knob to turn)

class BkClient:                    # opener/sleep seams exactly like DeepgramClient
    def capabilities(self) -> dict                      # GET  /api/meetscribe/v1/capabilities
    def upload_bundle(self, bundle_path, *, meeting_id,
                      calendar_meeting_id=None) -> dict # POST /bundles (zip body,
                                                        #   Idempotency-Key: meeting_id,
                                                        #   ?meeting_id= only when given —
                                                        #   Phase 3 wires it; None here)
    def workflow(self, workflow_id) -> dict             # GET  /workflows/{id}
```

- Auth `Authorization: Bearer <token>`; retries ×3 with backoff on 429/5xx/transport
  (uploads are retry-safe by Idempotency-Key); 401/403 name `[bk].token` /
  `MEETSCRIBE_BK_TOKEN`; 413 names `max_bundle_bytes`; non-JSON 200 gets the
  captive-portal-style error (copy the deepgram.py handling incl. response-phase
  exceptions).
- `preflight(caps, bundle_bytes) -> str | None`: contract_version 1 present in bk's,
  `output.FORMAT_VERSION` ∈ `mscribe_format_versions` ("update bk or pin meetscribe"),
  `bundle_bytes` ≤ `max_bundle_bytes` when present. Returns the human message, no raising.
- Workflow reference persistence, next to the artifacts (NOT a bundle member —
  `BUNDLE_MEMBERS` is untouched): `bk-workflow.json` =
  `{workflow_id, state_url, web_url, uploaded_at, state, checked_at}` with
  `write_workflow_ref(dir, ...)` / `read_workflow_ref(dir) -> dict | None`.

## Wiring

**`pipeline.run`** — inside the existing config guard: resolve `upload` (flag>config>
default); when on, require `base_url` + token (fetch `token_cmd` eagerly — same guard,
exit 2) and require the bundle (`--no-bundle` + auto-upload = exit 2 with "drop
--no-bundle or disable [bk].auto_upload"). After the bundle is written: capabilities →
`preflight` → upload → write `bk-workflow.json` → print `workflow_id` + `web_url`.
**Upload failure does NOT fail the run**: the artifacts are complete and local; failure is
a loud `reporter.warn` + printed retry hint (`meetscribe upload <dir>`), exit stays 0.
(Rationale: exit 2 would make automation treat a *processed* meeting as failed; the retry
path is first-class instead.)

**`record.run`** — eager validation at record start (before ffmpeg, same rationale as the
deepgram key): when upload resolves on, base_url+token must resolve and `token_cmd` is
executed once. The upload itself happens in the `pipeline.run` hand-off.

**`cli.py`**:
- `--upload/--no-upload` (BooleanOptionalAction, default None) on `record` + `process`;
  `run()` treats None = resolve from config (same tri-state contract as bundle/cleanup).
- New `upload <dir>` subcommand — the retry/backfill path: bundles the dir first if no
  `.mscribe` is present (reuse `bundle_dir`), then capabilities→preflight→upload→ref.
  Here upload IS the task: failure is exit 1 (network/server) or 2 (config), with the
  actionable message on stdout.
- New `status [dir]` subcommand: scans `dir` (default: resolved `meetings_dir`) for
  meeting directories (contain `meta.json`); prints one line each — meeting id, started_at,
  duration, backend, bundled?, upload state. For non-terminal persisted workflows it
  refreshes via `GET /workflows/{id}` and updates `bk-workflow.json` (`--offline` skips;
  refresh failure degrades to the cached state + a warning). Stdout is the data output.

**`doctor.py`** — when bk is configured (base_url set) or auto-upload on: base_url
reachability (TCP/TLS, no request), token presence (marker counts, never executed), and —
only when a *static* token exists — a live `capabilities` call validating auth + format
support (free endpoint; skipped for `token_cmd` to avoid pinentry in headless runs).

## Error-handling matrix (the part reviews should check hardest)

| Situation | Behavior |
|---|---|
| auto-upload on, base_url/token missing | exit 2 at run/record start (fail fast, before ffmpeg) |
| auto-upload on + `--no-bundle` | exit 2, message names both knobs |
| capabilities preflight fails (version/size) | upload skipped: loud warn + retry hint, run exits 0; `upload` subcommand exits 1 |
| upload network/5xx after retries | same as above |
| 401 from bk | actionable message naming token knobs (no retry loop) |
| re-upload same meeting, workflow live | bk returns existing workflow (Idempotency-Key) — we overwrite the local ref with the response either way |
| re-upload after done/failed | bk starts a NEW workflow (contract carve-out) — ref overwritten, correct by construction |
| `status` refresh fails | cached state shown + warning, exit 0 |

## Out of scope (later phases)

Task rendering/submits (P4), `?meeting_id=`/meetings query/roster (P3), polling loops
beyond the one-shot `status` refresh (P4), any GUI (P5).
