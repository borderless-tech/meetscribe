# Changelog

All notable changes to meetscribe are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Note: the artifact contract (what borderless-knowledge ingests) is versioned separately by the
integer `format_version` in `meta.json` (see `docs/mscribe-format-v2.md`). A `format_version`
bump always warrants at least a minor version bump here, and is called out in the entry.

## [Unreleased]

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

[Unreleased]: https://github.com/borderless-tech/meetscribe/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/borderless-tech/meetscribe/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/borderless-tech/meetscribe/releases/tag/v0.1.0
