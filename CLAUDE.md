# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`meetscribe` is a local, offline, CPU-only CLI (Linux + macOS) that records a meeting as **two
separate tracks** (microphone + system audio) and produces a diarized transcript plus speaker
embeddings. The embeddings are a first-class output — they are meant to be ingested into pgvector
later so speakers can be matched *across* meetings — which is why the raw vectors ship alongside
the transcript. No cloud, no PyTorch, no CUDA, no HF token.

## Commands

**Everything runs inside the Nix devshell.** `pytest`, `python` deps, `ffmpeg`, and the sherpa-onnx
native stack are only present there — plain `pytest`/`uv` are **not** on `PATH` outside it.

```bash
nix develop                       # enter the shell (Python 3.12, src/meetscribe editable-mounted)
pytest                            # fast unit suite — the ML boundary is mocked, runs in <1s
pytest tests/test_bundle.py -v    # a single test file
pytest tests/test_output.py::test_write_transcript_matches_schema -v   # a single test

# One-shot without entering the shell:
nix develop -c pytest -q

# Full end-to-end with the REAL pinned models (otherwise test_e2e.py skips):
export MEETSCRIBE_MODELS=$(nix build .#models --no-link --print-out-paths)
pytest tests/test_e2e.py          # EN mic + DE system → real transcript + 192-dim embeddings

nix flake check                   # runs the unit suite hermetically (this is what CI gates on)
```

Run the app itself (models + ffmpeg are wired in by the flake wrapper):

```bash
nix run .#doctor                  # preflight audio-setup checks — run this first
nix run .                         # record (Ctrl-C stops) + process (asks participant count)
nix run . -- process ./meeting-dir [--no-bundle] [--speakers N] [--no-cleanup]   # N = people incl. the user
nix run . -- clean ./meeting-dir  # re-run LLM cleanup on an existing transcript → ./meeting-dir-cleanup/

# The LLM transcript-cleanup pass (§ below) is default-on but needs the opt-in models-llm output.
# The default `nix run .` lacks the ~4.7 GB GGUF, so cleanup gracefully no-ops (raw transcript).
# For actual cleanup use the meetscribe-llm variant (bundles the GGUF + llama-server):
nix run .#meetscribe-llm -- process ./meeting-dir
nix build .#models-llm            # build just the LLM model tree (MEETSCRIBE_MODELS target)
```

## Architecture

The **central design decision** is dual-track asymmetry (see `what-we-build.md` §2.1): the mic
track is definitionally the user — hard-labelled `speaker="me"`, never diarized or clustered —
while the system track holds everyone else and is the only track that gets diarized. This halves
compute and removes the most common failure mode (your own voice colliding with a participant's
cluster).

**Pipeline** (`pipeline.py::process`): each track is loaded to 16 kHz mono, VAD-chunked, and
transcribed (Parakeet TDT, native word timestamps). The system track is additionally diarized;
`align.py` assigns each word to the max-overlap diarization segment, then `merge.py` interleaves
both tracks by timestamp and coalesces consecutive same-speaker utterances into paragraphs
(`coalesce_utterances`, order-safe because the list is time-sorted, so a different speaker always
breaks a run — unlike widening the VAD window, which would let one speaker swallow others' turns).
VAD chunks speech with a 0.7 s silence gap (not 0.25 s) so the recognizer sees longer,
context-rich windows; short chunks decoded cold were the main source of word errors and spurious
English on backchannels. Embeddings are computed in a **second pass** over the diarization segments
(sherpa's diarization API does not expose its internal vectors) — per-turn plus one per-cluster
centroid (computed by concatenating a cluster's audio, not averaging vectors). **Invariant:
embedding speakers == transcript speakers.** Only speakers that won at least one word are
embedded (upload sinks reject vectors without a transcript segment), turn vectors skip
sub-0.8 s segments, and centroids use all of a speaker's segments unfiltered so every
transcript speaker has a vector. `output.py` writes the artifacts.

A final **LLM cleanup stage** (`cleanup.py`, default-on, disable with `--no-cleanup`) rewrites
only the **system-track** `text` — fixing garbled proper nouns from context + a persistent
glossary (`glossary.py`, `~/.config/meetscribe/glossary.txt`; the record flow optionally prompts
for participant names and appends them). The mic (`me`) track is the clean user audio and passes
through untouched, and **per-word timings are never touched** — only `text` changes, with the
verbatim ASR text kept in `raw_text`. Rationale (see `docs/plans/2026-08-12-transcript-cleanup-*`):
transcript-quality is capped by the mixed, VoIP-compressed system downmix, not the ASR model, so
the lever is a text layer. The real cleaner (`ManagedLlamaCleaner`) drives a managed `llama-server`
subprocess (Qwen2.5-7B GGUF, `llama.py`) over loopback HTTP; anti-hallucination guards keep raw
text on empty/runaway/collapse output, so the worst case degrades per-segment to raw ASR.
`meetscribe clean <dir>` re-runs cleanup on an existing transcript non-destructively into
`<dir>-cleanup/`.

**Two seams keep the core pure and testable:**
- `Components` (a dataclass of `vad`/`recognizer`/`diarizer`/`embedder`/`cleaner`) is injected into
  `process()`. Unit tests pass fakes (`cleaner` defaults to `NullCleaner`); `build_components()`
  wires the real sherpa models (and, when the opt-in GGUF is present, the LLM cleaner). This is
  why the suite needs no models and runs in milliseconds.
- A `Reporter` protocol (`progress.py`) is injected for all terminal UI. `NullReporter` is the
  default arg everywhere, so pipeline/record logic stays UI-free and tests stay `rich`-free;
  `RichReporter` (live level-meters, progress bars, summary) is built only by the CLI. All UI goes
  to **stderr**; stdout stays clean.

`record.py` is the **only** platform-specific module: macOS uses one `avfoundation` input against
an aggregate device split by channel; Linux uses two `pulse` inputs (mic + `<sink>.monitor`).
Device indices are matched by **name**, never hard-coded. ffmpeg is stopped by writing `q` to its
stdin (a hard kill corrupts WAV headers).

## Remote backend (Deepgram)

Opt-in alternative to the local sherpa path: `--backend deepgram` on `record`/`process`, or
`STT_BACKEND=deepgram`; language via `--language`/`STT_LANGUAGE` (default `de`). Precedence is
always **flag > env > config > default** (see § Configuration), resolved once in `pipeline.run`
(`resolve_backend`/`resolve_language`). `backends.py` defines the seam (`TranscriptionBackend` →
`BackendResult`); `deepgram.py` implements it over stdlib `urllib` (no SDK, no new deps),
uploading the mic track plain and the system track with `diarize=true`, forwarding glossary
terms as nova-3 `keyterm` boosts. Rules:

- `DEEPGRAM_API_KEY` is required — missing key is exit 2 with an actionable message,
  **never** a silent fallback to local. `pipeline.check_backend` is the single validator
  (name + key) and runs both in `pipeline.run` and at the TOP of `record.run`, so
  `record --backend deepgram` without a key fails *before* ffmpeg starts — not after an
  hour of recording. A `DeepgramError` at process time (e.g. invalid key → 401) is also
  a clean exit 2. Doctor preflights key + reachability when `STT_BACKEND=deepgram`.
- `--speakers` / the participant-count prompt steers only the local diarizer; remote mode
  warns and ignores it (participant *names* still matter — they feed the glossary →
  nova-3 keyterms).
- **Embeddings stay local in every backend** (CAM++ via `build_embed_components`); only
  audio is uploaded, vectors never leave the machine. Remote mode must not load Parakeet,
  the diarizer/VAD, or the GGUF cleaner (`NullCleaner`; transcript ships `cleaned: false`).
- `meta.json` records `backend` plus the service's `backend_model_versions`/`request_ids`
  (the remote substitute for SHA-256 pins); additions are additive, `format_version` stays 2.
- Privacy: with this backend the meeting **audio leaves the machine**. The default remains
  fully local/offline.
- Real-API e2e: `tests/test_deepgram_e2e.py` (paid, ~$0.01/run; double-gated on
  `DEEPGRAM_API_KEY` + `MEETSCRIBE_MODELS`).

## Configuration

`config.py` owns all of it — the TOML file, the XDG paths, and the **one precedence rule
everywhere: flag > env > config > default**. Read-only stdlib `tomllib`; we never write TOML
(the `config init` template is the plain string `config.TEMPLATE`).

**Paths** (Linux + macOS, both via XDG):
- Config: `$XDG_CONFIG_HOME/meetscribe/config.toml` (`~/.config` fallback);
  `MEETSCRIBE_CONFIG=<path>` overrides — tests monkeypatch these and must **never** read the
  real home.
- Recordings default: `$XDG_DATA_HOME/meetscribe/meetings/` (`~/.local/share` fallback) —
  `record` without `-o` writes `<meetings_dir>/meetscribe-<timestamp>/` there.
- Glossary (unchanged): `$XDG_CONFIG_HOME/meetscribe/glossary.txt` (`glossary.py` uses
  `config.config_home()`).

**Schema v1** (all keys optional; empty strings count as unset): `[stt]
backend`/`language`, `[deepgram] api_key`, `[storage] meetings_dir`, `[record]
system_source`, `[output] bundle`/`cleanup`. `[bk]` is reserved for Phase 2 and never
reported as unknown. Unknown keys warn; **malformed TOML is exit 2 with file + line**
(`ConfigError`) — a typo'd config silently degrading to defaults is the worst failure mode.
Booleans must be real TOML booleans (`"false"` is a `ConfigError`, never truthy).

Every resolver in `config.py` returns `Resolved(value, origin)` with origin ∈
`flag|env|config|default`, so `meetscribe config` can show *why* each value is what it is
(`config init` writes the 0600 template, `config path` prints the path). `doctor` runs a
config check first (path + parse status red-with-line, unknown-key and api_key-permission
warnings) that never aborts the audio checks.

**The None-default flag pattern:** tri-state CLI flags (`--bundle/--no-bundle`,
`--cleanup/--no-cleanup`, `--backend`, `--language`, `--system-source`) have parser default
`None` = "flag not given → resolve env/config/default downstream" in
`pipeline.run`/`record.run`. An explicit flag value always wins. When adding a config-backed
option, keep this shape: never give the parser a concrete default, or the config layer
underneath becomes unreachable.

## Outputs

Three artifacts (`output.py`), plus an optional single-file bundle:
- `transcript.json` — segments with per-word timestamps, `speaker` (`me`/`spk_N`), `track`, the
  (possibly cleaned) `text`, and the verbatim `raw_text`; plus a top-level `cleaned` bool.
  `read_transcript` is the inverse of `write_transcript` (used by `clean <dir>`).
- `embeddings.npz` — `turn_vectors` / `cluster_vectors` (`float32`), plus id/speaker str arrays.
- `meta.json` — model names + the embedding model's SHA-256, `embedding_dim`, timestamps,
  `cleaned` + (when cleanup ran) `cleanup_model` (name/SHA-256/params), `format_version` (now `2`:
  adds `raw_text`/`cleaned`/`cleanup_model`). **Not optional:** vectors/text from different models
  are incomparable, so the model identity must travel with the artifact (and later into the DB).
- `meeting-<id>.mscribe` (default-on for `record`/`process`, opt out with `--no-bundle`; also
  retrofittable via the `meetscribe bundle <dir>` subcommand) — a plain
  zip of the three files above, for one atomic authenticated upload to a stateless sink. See
  `docs/plans/2026-08-02-bundle-format-design.md`. `bundle_dir` / `default_bundle_name` in
  `output.py` are the *only* places that know the member names and naming scheme — keep it that way.

## Versioning & releases

SemVer + [Keep a Changelog](CHANGELOG.md). To cut a release:

1. Bump the version in **both** `pyproject.toml` and `src/meetscribe/__init__.py`, then run
   `nix develop -c uv lock` (uv.lock records the project's own version).
2. Move the `## [Unreleased]` notes into a new `## [x.y.z] - <date>` section (+ compare link
   at the bottom) — and keep `[Unreleased]` fed as features land, not at release time.
3. Commit, then `git tag vX.Y.Z` (push tags with `git push --tags`).

`tests/test_version.py` enforces the sync (pyproject == `__version__` == uv.lock, changelog
entry exists), so forgetting a step fails the suite. The artifact contract is versioned
separately by the integer `format_version` (`output.py`); bumping it warrants at least a
minor release and an explicit changelog callout.

## Invariants / gotchas

- **Embedding dimension is 192** (CAM++), read at runtime from `extractor.dim` — **never hard-code
  it.** `what-we-build.md` §6 and §2.x say `512`; that number is **stale/wrong** — trust the code,
  the README, and `meta.json`. The dim is stored in `meta.json` and sizes the npz arrays.
- Models are pinned by SHA-256 in `nix/models.nix`, so a silent model swap (which would make the
  whole pgvector history incomparable) is structurally impossible. The wrapper sets
  `MEETSCRIBE_MODELS` to the models store path; the pipeline reads it.
- `sherpa-onnx` is split across two PyPI dists (`sherpa-onnx` + `sherpa-onnx-core`); **both** are
  pinned in `pyproject.toml`, else the native lib is missing at runtime. Versions are pinned with
  `==`, never `>=`, to keep the `uv.lock` / uv2nix closure reproducible.
- Times are float seconds everywhere; the shared types in `types.py` are frozen, JSON-friendly
  dataclasses.
- Timestamps in `meta.json` are tz-aware ISO 8601 **with offset** (`datetime.now().astimezone()` /
  `datetime.fromtimestamp(ts).astimezone()`) — a bare UTC epoch would misalign downstream
  calendar matching.

`what-we-build.md` is the original German bootstrap spec — useful for *why* decisions were made,
but written from memory before implementation, so verify specifics against the code (esp. the
512-vs-192 dim). Design/plan docs live in `docs/plans/`.
