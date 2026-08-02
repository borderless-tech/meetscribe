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
nix run .                         # record (Ctrl-C stops) + process
nix run . -- process ./meeting-dir [--bundle]
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
both tracks by timestamp. Embeddings are computed in a **second pass** over the same segments
(sherpa's diarization API does not expose its internal vectors) — per-turn plus one per-cluster
centroid (computed by concatenating a cluster's audio, not averaging vectors). `output.py` writes
the artifacts.

**Two seams keep the core pure and testable:**
- `Components` (a dataclass of `vad`/`recognizer`/`diarizer`/`embedder`) is injected into
  `process()`. Unit tests pass fakes; `build_components()` wires the real sherpa models. This is
  why the suite needs no models and runs in milliseconds.
- A `Reporter` protocol (`progress.py`) is injected for all terminal UI. `NullReporter` is the
  default arg everywhere, so pipeline/record logic stays UI-free and tests stay `rich`-free;
  `RichReporter` (live level-meters, progress bars, summary) is built only by the CLI. All UI goes
  to **stderr**; stdout stays clean.

`record.py` is the **only** platform-specific module: macOS uses one `avfoundation` input against
an aggregate device split by channel; Linux uses two `pulse` inputs (mic + `<sink>.monitor`).
Device indices are matched by **name**, never hard-coded. ffmpeg is stopped by writing `q` to its
stdin (a hard kill corrupts WAV headers).

## Outputs

Three artifacts (`output.py`), plus an optional single-file bundle:
- `transcript.json` — segments with per-word timestamps, `speaker` (`me`/`spk_N`), `track`.
- `embeddings.npz` — `turn_vectors` / `cluster_vectors` (`float32`), plus id/speaker str arrays.
- `meta.json` — model names + the embedding model's SHA-256, `embedding_dim`, timestamps,
  `format_version`. **Not optional:** vectors from different models are incomparable, so the model
  identity must travel with the vectors (and later into the DB).
- `meeting-<id>.mscribe` (via `--bundle` or the `meetscribe bundle <dir>` subcommand) — a plain
  zip of the three files above, for one atomic authenticated upload to a stateless sink. See
  `docs/plans/2026-08-02-bundle-format-design.md`. `bundle_dir` / `default_bundle_name` in
  `output.py` are the *only* places that know the member names and naming scheme — keep it that way.

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
