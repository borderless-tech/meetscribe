# Changelog

All notable changes to meetscribe are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Note: the artifact contract (what borderless-knowledge ingests) is versioned separately by the
integer `format_version` in `meta.json` (see `docs/mscribe-format-v2.md`). A `format_version`
bump always warrants at least a minor version bump here, and is called out in the entry.

## [Unreleased]

### Added
- TOML config file at `$XDG_CONFIG_HOME/meetscribe/config.toml` (`~/.config` fallback;
  `MEETSCRIBE_CONFIG=<path>` overrides). All keys optional; schema v1 covers
  `[stt] backend/language`, `[deepgram] api_key`, `[storage] meetings_dir`,
  `[record] system_source`, `[output] bundle/cleanup`. One precedence rule everywhere:
  **flag > env > config > default**. Malformed TOML exits 2 with file + line (never a
  silent fall-back to defaults); unknown keys warn. Read via stdlib `tomllib` — no new
  dependencies.
- `meetscribe config` subcommand: bare = effective values as a table (value, origin,
  api_key masked) plus the config path; `config init` writes a commented template
  (0600, refuses to overwrite); `config path` prints the path for scripting.
- XDG data dir: recordings now have a standard home under
  `$XDG_DATA_HOME/meetscribe/meetings/` (`~/.local/share` fallback), configurable via
  `[storage] meetings_dir`.
- `doctor` gains a config check (runs first, never aborts the audio checks): path in
  use + existence, parse status (malformed = red with the line), unknown-key and
  api_key file-permission warnings (0600 advice).

### Changed
- **Default recordings location:** `meetscribe record` without `-o` now writes to
  `$XDG_DATA_HOME/meetscribe/meetings/meetscribe-<timestamp>/` instead of littering the
  current working directory with `./meetscribe-<timestamp>/`. `-o` behaves exactly as
  before. This is the only behavior change for existing setups.

## [0.2.1] - 2026-09-08

Feature release shipped as a patch (0.x pragmatism — same-day follow-up to 0.2.0). No
breaking changes: the artifact contract stays at `format_version` 2, `local` remains the
default backend, and all CLI/meta additions are backwards-compatible.

### Added
- Optional remote transcription backend: `--backend deepgram` on `record`/`process` (or
  `STT_BACKEND=deepgram`), with `--language`/`STT_LANGUAGE` (default `de`). Both tracks are
  transcribed — and the system track diarized — by Deepgram nova-3 over plain stdlib HTTP
  (no new dependencies). Requires `DEEPGRAM_API_KEY`; a missing key fails fast with exit 2 —
  there is **no silent fallback** to local. Glossary terms are forwarded as nova-3 `keyterm`
  boosts. Remote mode loads neither Parakeet nor the diarizer/VAD/GGUF cleaner.
- `meta.json` gains a top-level `backend` (`local`/`deepgram`) plus, for remote runs, the
  service's model identity (`backend_model_versions`, `request_ids`) — the remote substitute
  for local SHA-256 pins. Additive only: `format_version` stays 2.
- `doctor` preflights the remote backend when `STT_BACKEND=deepgram`: key present +
  DNS/TCP/TLS reachability of `api.deepgram.com:443` (no billable API call).

### Privacy
- With `--backend deepgram` the meeting **audio leaves the machine** (both tracks are
  uploaded to Deepgram's API). Speaker embeddings are still computed locally (CAM++) and
  never uploaded; the default remains fully local/offline.

### Fixed
- `nix flake check` (the CI gate) had failed since the 0.2.0 SemVer setup: the hermetic
  test derivation was missing `uv.lock` and `CHANGELOG.md`, which `tests/test_version.py`
  guards against.

## [0.2.0] - 2026-09-08

### Added
- Transcript-cleanup stage (default-on, `--no-cleanup` to skip): echo-collapse + guarded
  repair of system-track text via a managed local `llama-server` (Qwen2.5-7B GGUF, opt-in
  `models-llm` output). The mic (`me`) track passes through untouched; per-word timings are
  never modified; anti-hallucination guards fall back to raw ASR per segment.
- Broken-word suggest mode: likely-broken ASR words are flagged with ranked correction
  candidates in a top-level `suggestions` array for human review (never auto-applied).
  Frami German hunspell dict + shipped base glossary keep false flags low.
- Persistent glossary (`~/.config/meetscribe/glossary.txt`); the record flow optionally
  prompts for participant names and appends them.
- `clean <dir>` subcommand — re-run cleanup on an existing transcript non-destructively
  into `<dir>-cleanup/`.
- **Artifact `format_version` 1 → 2** (additive): per-segment `raw_text`, top-level
  `cleaned` flag, optional `cleanup_model` identity in `meta.json`, `suggestions` in
  `transcript.json`. Reader-side spec: `docs/mscribe-format-v2.md`.
- `.mscribe` upload bundle (zip of the three artifacts) + `bundle <dir>` subcommand;
  `meta.json` enriched with `meeting_id`, wall-clock `started_at`/`ended_at`,
  `format_version`.
- Rich terminal UI: live record level meters, per-stage progress bars, summary table
  (all on stderr; stdout stays clean).

### Changed
- `record`/`process` now emit the `.mscribe` bundle **by default**; `--no-bundle` opts out.
- Longer ASR windows (0.7 s VAD silence gap) and same-speaker paragraph coalescing —
  fewer word errors and spurious-English backchannels.

### Fixed
- System audio is captured from the *default* sink's monitor, not the first sink.
- Embeddings ship only for speakers that appear in the transcript (upload sinks reject
  vectors without a transcript segment).
- ASR word splitting handles space-prefixed tokens, not just the `▁` marker.
- `doctor` mic check waits out the Bluetooth A2DP→HFP profile switch.

## [0.1.0] - 2026-08-02

Initial release: local, offline, CPU-only dual-track meeting recorder (mic + system audio)
producing a diarized transcript with per-word timestamps and 192-dim speaker embeddings
(per-turn + per-cluster centroid). Silero VAD, Parakeet TDT ASR, sherpa-onnx diarization,
`doctor` preflight checks, reproducible Nix/uv2nix build with SHA-256-pinned models.

[Unreleased]: https://github.com/borderless-tech/meetscribe/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/borderless-tech/meetscribe/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/borderless-tech/meetscribe/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/borderless-tech/meetscribe/releases/tag/v0.1.0
