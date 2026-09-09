# Phase 2 implementation plan: bk client + auto-upload

Prereq reading: `2026-09-09-phase2-bk-upload-design.md` (the spec — config keys, client
API, wiring, the error-handling matrix), `docs/meetscribe-bk-contract-v1.md` (agreed
payload shapes — the fixtures MUST match its examples), `CLAUDE.md`. Branch:
`feat/bk-upload`. Tests via `nix develop -c pytest -q`; baseline **446 passed, 3
skipped** — green after every task. TDD: failing test first, always. No new dependencies.
Config/XDG tests monkeypatch `MEETSCRIBE_CONFIG`/`XDG_*` — never the real home; no test
touches the network (transport seams only).

## Task 1 — config: `[bk]` section + shared secret-cmd helper

Files: `src/meetscribe/config.py`, `tests/test_config.py`.

- Generalize the lazy-secret machinery: `SecretCmd` NamedTuple + `fetch_secret(value,
  timeout_s)` carrying a *label* for error messages (e.g. `[bk].token_cmd`);
  `ApiKeyCmd`/`fetch_api_key` become thin aliases (existing tests must keep passing
  unchanged).
- `_SCHEMA` gains `bk: {base_url, token, token_cmd, auto_upload}`; `bk` leaves
  `_RESERVED_SECTIONS` (it is real now — unknown `bk.*` keys start warning).
- Resolvers (+ `validate()` coverage): `bk_base_url` (env `MEETSCRIBE_BK_URL` > config >
  `None`; strip trailing `/`), `bk_token` (env `MEETSCRIBE_BK_TOKEN` > config `token` >
  config `token_cmd` as unexecuted `SecretCmd`; both config keys set = `ConfigError`),
  `bk_auto_upload` (flag > config bool > default `False`; strict bool).
- TEMPLATE: replace the reserved-`[bk]` comment line with the commented-out section from
  the design doc (template still parses to `{}` — the existing test enforces it).
- `insecure_api_key_perms` → also warn when `[bk].token` is set (rename/extend to cover
  both secrets; keep the old name as alias if other modules import it).

## Task 2 — `bk.py` + contract fixtures (new files only; parallel-safe with Task 1)

Files: `src/meetscribe/bk.py`, `tests/test_bk.py`, `tests/fixtures/bk/*.json`.

`BkConfig`/`BkError`/`BkClient` exactly as specced (opener+sleep seams copied from
`deepgram.py`, including response-phase exception retries and the non-JSON-200 error).
`preflight(caps, bundle_bytes)`. `write_workflow_ref`/`read_workflow_ref`
(`bk-workflow.json`; `read` returns `None` when absent/invalid-JSON — a corrupt ref must
not crash `status`).

Fixtures authored FROM THE CONTRACT DOC's examples (drop-in replaceable by bk's canonical
set later): `capabilities.json`, `upload_accepted.json` (202 body),
`workflow_processing.json`, `workflow_awaiting_review.json` (ONE `speaker_annotation`
task — the agreed single-task shape), `workflow_done.json`, `workflow_failed.json`,
`error.json`. TDD order: URL building + auth header → capabilities → preflight verdicts
(ok / bad contract_version / format_version not listed / bundle too big) → upload (body,
Idempotency-Key header, optional `?meeting_id=`, 202 parse) → retry/backoff (429→200,
URLError→200, 401 names both token knobs and does NOT retry, 413 names
`max_bundle_bytes`) → workflow GET → ref read/write roundtrip + corrupt-file tolerance.

## Task 3 — wiring: pipeline + record (needs Tasks 1+2)

Files: `src/meetscribe/pipeline.py`, `src/meetscribe/record.py`, `tests/test_pipeline.py`,
`tests/test_record.py`.

- `pipeline.run(upload: bool | None = None, ...)`: resolve inside the existing config
  guard; when on → require base_url+token (fetch `SecretCmd` eagerly, same guard = exit
  2) and bundle-on (`--no-bundle` conflict = exit 2, message names both knobs). After the
  bundle is written: capabilities → preflight → upload → write ref → print
  `workflow_id` + `web_url`. Any upload-stage failure (preflight, network, 4xx/5xx):
  `reporter.warn` + printed `meetscribe upload <dir>` retry hint, **exit stays 0** —
  tests must pin the exit code AND the hint. Client construction goes through a
  module-level seam (`bk.BkClient`) so tests monkeypatch one symbol.
- `record.run`: when upload resolves on → base_url+token resolve + eager `fetch_secret`
  in the existing preflight guard (test: broken `token_cmd` fails before
  `record_tracks`, mirroring the deepgram test).

## Task 4 — CLI: `--upload/--no-upload`, `upload <dir>`, `status` (needs Tasks 1+2; files disjoint from Task 3)

Files: `src/meetscribe/cli.py`, `tests/test_cli.py`.

- Tri-state `--upload/--no-upload` on record+process, default None, forwarded like
  bundle/cleanup (getattr fallbacks None; guard tests for forwarding).
- `upload <dir>`: load config (malformed = exit 2); bundle the dir if no `.mscribe`
  exists; capabilities → preflight → upload → write ref → print `workflow_id` +
  `web_url`. Config problems exit 2; upload failures exit 1 (here upload IS the task).
- `status [dir] [--offline]`: scan for `meta.json` dirs under the argument (default:
  resolved `meetings_dir`); per meeting print id, started_at, duration_s, backend,
  bundled yes/no, upload state (`not uploaded` / state from ref). Non-terminal ref +
  not `--offline` → refresh via `BkClient.workflow`, update the ref file; refresh failure
  → cached state + `(stale)` marker + warning, exit 0. Plain prints, stdout is data.

## Task 5 — doctor + docs (needs Tasks 1+2; files disjoint from 3/4)

Files: `src/meetscribe/doctor.py`, `tests/test_doctor.py`, `CHANGELOG.md`, `CLAUDE.md`,
`README.md`.

- bk checks appended when base_url resolves or auto-upload is on: reachability
  (TCP/TLS to the base_url host, injected connect seam), token presence (SecretCmd
  counts, never executed), live capabilities+format-support check ONLY with a static
  token (injected opener; skipped with `token_cmd` — no pinentry in doctor).
- CHANGELOG `[Unreleased]`: auto-upload headline (explicit opt-in + privacy note),
  `upload`/`status` subcommands, `[bk]` config section. CLAUDE.md: extend the
  configuration + remote sections (bk client, ref file, the exit-0-on-upload-failure
  rationale). README: quickstart (`config init` → set `[bk]` → `meetscribe status`).

## Sequencing

Tasks 1 ∥ 2 first (disjoint files). Then 3 ∥ 4 ∥ 5 (disjoint files; the shared contract —
`upload=None` tri-state and `bk.BkClient` as the patch seam — is fixed HERE). Then review
(design-correctness incl. the error matrix, invariants, test quality) → adversarial verify
→ fix → full suite + `nix flake check`.

## Acceptance (whole phase)

- No `[bk]` config → byte-identical behavior to v0.2.1+P1 (auto-upload strictly opt-in).
- `[bk]` configured + `auto_upload = true`: `process` ends with `workflow_id`/`web_url`
  printed and `bk-workflow.json` next to the artifacts; bk down → artifacts intact,
  warning + retry hint, exit 0.
- `meetscribe upload <dir>` retrofits + retries any meeting dir; `meetscribe status`
  lists every local meeting with its upload state.
- Suite green + offline + sub-second; `nix flake check` green; `pyproject.toml`/`uv.lock`
  untouched.
