# meetscribe

Local, offline, cross-platform (Linux + macOS) CLI that records a meeting — **microphone
and system audio as two separate tracks** — and produces a diarized transcript plus speaker
embeddings, entirely on CPU. No cloud, no PyTorch, no CUDA, no HF token.

The speaker embeddings are a first-class output: they are meant to be ingested into a pgvector
instance later so speakers can be recognised **across** meetings. That's why the raw vectors ship
alongside the transcript.

## What it produces

Running against a recording yields three files:

- `transcript.json` — text with per-word timestamps, speaker label (`me` / `spk_N`) and track.
- `embeddings.npz` — speaker embeddings per turn and per cluster (`float32`, **192-dim**).
- `meta.json` — model names, the embedding model's hash, the embedding dimension, sample rate.

`meta.json` is **not** optional: vectors from different models are not comparable, so the model
name + dimension must travel with the vectors (and later into the DB).

## Quick start

Requires only [Nix](https://nixos.org/download) with flakes enabled. Nothing else is
preinstalled — ffmpeg and all models come from the flake (exception: BlackHole on macOS, see
below).

```bash
nix run github:borderless-tech/meetscribe#doctor              # check the audio setup first
nix run github:borderless-tech/meetscribe                     # record (Ctrl-C stops) + process
nix run github:borderless-tech/meetscribe -- process ./meeting-dir   # process an existing recording
```

`process` accepts a directory containing `raw/mic.wav` + `raw/system.wav`, or a single `.wav`
(treated as the system track and diarized).

## ⚠️ Wear headphones

v1 requires headphones. Without them the far side leaks from your speakers into the mic track
(echo/bleed) and pollutes the diarization. This is a known limitation (a WebRTC AEC sidecar is
possible later).

## Audio setup

### Linux

Nothing to set up — PipeWire exposes a `<sink>.monitor` source out of the box, which is the far
side. `doctor` verifies PipeWire is up and a monitor source exists.

### macOS

Nix can't install a CoreAudio HAL plugin or create an aggregate device, so two one-time manual
steps are needed (walked through by `doctor`):

1. Install BlackHole: `brew install blackhole-2ch`
2. In **Audio MIDI Setup**, create a **Multi-Output Device** (BlackHole + your headphones) and
   set it as the system output, then create an **Aggregate Device** (BlackHole 2ch + your Mic)
   named exactly `meetscribe`. The aggregate device also does clock-drift correction between the
   two inputs.

The most important `doctor` check is the 1-second mic RMS test: **without a TCC (microphone)
permission, ffmpeg records silence with no error** — you'd only notice after the meeting.

## Development

Everything runs from within the Nix shell — `nix develop` (or plain `nix-shell` via
flake-compat) gives a Python 3.12 env with the sherpa-onnx native stack, ffmpeg and pytest, with
`src/meetscribe` mounted editable:

```bash
nix develop
pytest                       # fast unit suite (ML boundary mocked)

# Full end-to-end smoke with the real pinned models:
export MEETSCRIBE_MODELS=$(nix build .#models --no-link --print-out-paths)
pytest tests/test_e2e.py     # EN mic + DE system → real transcript + 192-dim embeddings
```

`nix flake check` runs the unit suite hermetically. See `docs/plans/` for the architecture
research report and the implementation plan.

## Models (pinned)

| role | model |
|---|---|
| ASR | Parakeet-TDT-0.6b **v3** (multilingual incl. German), int8 |
| diarization segmentation | pyannote-segmentation-3.0 (ONNX, int8) |
| speaker embedding | 3D-Speaker CAM++ (192-dim) |
| VAD | Silero VAD |

All four are pinned by SHA-256 in `nix/models.nix`, so a silent model swap — which would make the
whole pgvector history incomparable — is structurally impossible.

## Not in scope (separate tools)

pgvector ingest and cross-meeting profile matching consume `embeddings.npz` + `meta.json` and live
elsewhere. Realtime/streaming and the macOS Core-Audio-Taps helper are future work.
