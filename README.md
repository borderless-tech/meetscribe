# meetscribe

Local, offline, cross-platform (Linux + macOS) CLI that records a meeting — **microphone
and system audio as two separate tracks** — and produces a diarized transcript plus speaker
embeddings, entirely on CPU. No cloud, no PyTorch, no CUDA.

Outputs three artifacts: `transcript.json`, `embeddings.npz`, `meta.json`. See
[what-we-build.md](what-we-build.md) for the full architecture and rationale.

## Status

Bootstrapping. See `docs/plans/` for the research report and implementation plan, and the
task list for progress. Full usage docs (incl. the macOS BlackHole setup and the mandatory
headphone requirement) land with chunk C10.

## Quick start (once built)

```bash
nix run .#doctor          # check the audio setup
nix run .                 # record (Ctrl-C stops) + process
nix run . -- process path/to/audio.wav
```

## Development

Everything runs from within the Nix shell:

```bash
nix develop          # or: nix-shell  (flake-compat)
pytest               # run the test suite
```
