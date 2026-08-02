# meetscribe Implementation Plan

> **For Claude:** Execute task-by-task with TDD (superpowers:test-driven-development).
> Run every test inside the Nix shell: `nix develop --command pytest ...`.

**Goal:** A local, offline, CPU-only CLI that records mic + system audio as two tracks and
produces `transcript.json`, `embeddings.npz`, `meta.json` with speaker diarization + embeddings.

**Architecture:** Two-track pipeline (see what-we-build.md §4). The only platform-specific
module is `record.py`. All ML calls (sherpa-onnx) sit behind thin, injectable interfaces so the
deterministic logic is unit-tested with mocks; one real end-to-end smoke run on Linux proves the
wiring. Every third-party dep (incl. the sherpa native stack) comes from the uv2nix flake.

**Tech Stack:** Python 3.12, sherpa-onnx 1.13.4 (Parakeet-TDT-v3 ASR, Silero VAD, pyannote-seg-3.0
diarization, CAM++ 192-dim embeddings), numpy, ffmpeg; Nix flake (uv2nix), pytest.

**Authoritative facts:** see `docs/plans/2026-08-02-meetscribe-research.md`. Notably the embedding
dim is **192, not 512** (spec §6 is wrong); read `extractor.dim` at runtime.

---

## Shared data types (`src/meetscribe/types.py`) — build first in C1

Plain dataclasses, JSON-friendly. Times are float seconds throughout.

```python
@dataclass(frozen=True)
class Word:
    w: str
    start: float
    end: float

@dataclass(frozen=True)
class Segment:       # one ASR utterance on one track, speaker not yet assigned for system track
    start: float
    end: float
    text: str
    words: tuple[Word, ...]

@dataclass(frozen=True)
class DiarSegment:   # a diarization span
    start: float
    end: float
    speaker: str     # "spk_0", "spk_1", ...

@dataclass(frozen=True)
class Utterance:     # final merged unit written to transcript.json
    start: float
    end: float
    speaker: str     # "me" | "spk_N"
    track: str       # "mic" | "system"
    text: str
    words: tuple[Word, ...]
```

---

## C1 — Pure-logic core (align, merge, output)  [task #2]

**Files:** create `src/meetscribe/types.py`, `align.py`, `merge.py`, `output.py`; tests
`tests/test_align.py`, `tests/test_merge.py`, `tests/test_output.py`.

### align.py — `assign_words_to_speakers(words, diar) -> list[Utterance]`
WhisperX-style max-overlap (what-we-build.md §4.1):
1. For each `Word`, compute temporal overlap with each `DiarSegment`; the segment with max
   overlap wins (overlap = `max(0, min(w.end,d.end) - max(w.start,d.start))`). Ties → earliest
   segment. A word overlapping nothing → nearest segment by gap (or its own "spk_?" — pick
   nearest to avoid dropping words).
2. Group consecutive words with the same speaker into one `Utterance` (track="system").

**Tests:** word fully inside one segment; word straddling two (bigger overlap wins); tie →
earliest; word before all segments → nearest; consecutive same-speaker words merged into one
utterance; speaker change splits utterances; empty words → [].

### merge.py — `merge_tracks(mic, system) -> list[Utterance]`
Concatenate the mic utterances (speaker forced `"me"`, track `"mic"`) with the system utterances,
sort by `start` (stable; tie-break by track so "mic" before "system"). Mic utterances are never
reclustered.

**Tests:** interleaving by start; mic speaker always "me"; stable tie-break; empty mic; empty system.

### output.py
- `build_meta(models, embedding_dim, sample_rate=16000) -> dict` → the §6 `meta.json` (embedding_dim
  from runtime, NOT hard-coded).
- `write_transcript(path, meeting_id, duration_s, utterances)` → §6 `transcript.json`.
- `write_embeddings(path, turns, clusters)` where each turn = (id, vector(np.float32 (dim,)),
  speaker); writes npz arrays `turn_ids, turn_vectors (N,dim), turn_speakers, cluster_ids,
  cluster_vectors (K,dim)`.
- `write_meta(path, meta)`.

**Tests:** transcript round-trips to the exact schema; npz arrays have dtype/shape (N,dim) with
dim read from the vectors (parametrize dim=192); meta contains all §6 keys incl. embedding_dim=192;
empty turns → (0,dim) arrays.

**Done:** `nix develop --command pytest tests/test_align.py tests/test_merge.py tests/test_output.py`
green. Commit.

---

## C2 — record.py command-building (both OS)  [task #3]

**Files:** `src/meetscribe/record.py` (rewrite), `tests/test_record.py`, fixtures
`tests/fixtures/avfoundation_devices.txt`, `tests/fixtures/pactl_sources.txt`.

Pure, injectable helpers (no ffmpeg exec in unit tests):
- `parse_avfoundation_devices(stderr_text) -> list[(index, name)]` — parse `ffmpeg -f avfoundation
  -list_devices true -i ""` stderr.
- `match_device(devices, name) -> index` — match by name (never hard-code index); raise if absent.
- `build_macos_ffmpeg_cmd(aggregate_index, mic_out, system_out) -> list[str]` — one avfoundation
  input against the aggregate device, `pan` filter splitting channels to two mono files.
- `parse_pw_sources(text) -> list[str]` / `find_monitor_source(sources) -> str`.
- `build_linux_ffmpeg_cmd(mic_source, monitor_source, mic_out, system_out) -> list[str]` — two
  `-f pulse` inputs.
- `rms(samples) -> float` and `warn_if_silent(path) -> str|None`.
- SIGINT handler sends `q` to ffmpeg stdin (not SIGKILL) — helper `stop_ffmpeg(proc)`.

**Tests (both OS):** device-list parsing from fixtures; name match + missing-name error; macOS cmd
has the pan filter and two outputs; linux cmd has two `-f pulse` inputs matched by name; rms of a
known buffer; silent-track detection. Real Linux path: a marked integration test that actually runs
`pw-record`/ffmpeg for ~1 s and asserts two non-empty WAVs (skipped if no PipeWire).

**Done:** unit tests green on both OS paths; one real Linux capture verified. Commit.

---

## C3 — audio.py + vad.py  [task #4]

**Files:** `audio.py`, `vad.py`, `tests/test_audio.py`, `tests/test_vad.py`.

- `audio.build_resample_cmd(src, dst, rate=16000) -> list[str]` (`ffmpeg -i src -ar 16000 -ac 1
  dst`); `load_wav_f32(path) -> (np.float32, rate)`.
- `vad.VadRunner(protocol)` wraps `sherpa_onnx.VoiceActivityDetector`. Pure part:
  `speech_chunks(segments) -> list[Chunk]` where each chunk carries `offset` (start seconds =
  `seg.start / sample_rate`, since `SpeechSegment` has NO `.end`) and samples. Tested with a fake
  VAD yielding scripted `(start_sample, samples)` — assert absolute offsets recomputed correctly.

**Tests:** resample cmd string; offset math (chunk starting at sample 48000 @16k → 3.0 s); chunk
end = start + len(samples)/rate; empty → []. Mock the sherpa VAD.

**Done:** green. Commit.

---

## C4 — asr.py (Parakeet wrapper)  [task #5]

**Files:** `asr.py`, `tests/test_asr.py`.

- `Recognizer` protocol → `recognize(samples) -> RawResult(text, tokens, timestamps, durations)`.
- Pure `tokens_to_words(tokens, timestamps, durations) -> list[Word]`: group SentencePiece tokens
  into words on the `▁` word-boundary marker; word.start = first token ts, word.end = last token
  ts + its duration. (r.words is often empty for subword models — build from tokens.)
- `transcribe_chunks(recognizer, chunks) -> list[Segment]`: recognize each VAD chunk, add the
  chunk offset to every timestamp, one Segment per chunk.

**Tests:** `▁he llo ▁world` → 2 words with right spans; offset applied; empty tokens → []; a chunk
with offset 10.0 shifts word starts by 10.0. Mock the recognizer.

**Done:** green. Commit.

---

## C5 — diarize.py + embed.py  [task #6]

**Files:** `diarize.py`, `embed.py`, `tests/test_diarize.py`, `tests/test_embed.py`.

- `diarize.run(diarizer, samples) -> list[DiarSegment]` mapping sherpa int speaker id → `spk_N`;
  diarizer behind protocol.
- `embed.filter_short(segments, min_dur=0.8) -> list` (drop < 0.8 s — unusable embeddings).
- `embed.embed_turns(extractor, wav, segments) -> list[(turn_id, vec, speaker)]` — one embedding
  per turn; assert `extractor.dim == 192`.
- `embed.cluster_centroids(extractor, wav, segments_by_cluster) -> list[(cluster_id, vec)]` —
  **concatenate** each cluster's audio spans, embed once (NOT vector mean; §2.7).
- `embed.slice_audio(wav, rate, start, end) -> np.float32` (pure).

**Tests:** filter drops < 0.8 s; slice returns right sample range; centroid concatenates (fake
extractor records the total sample count it received == sum of spans); dim assertion; speaker-id
mapping. Mock extractor/diarizer.

**Done:** green. Commit.

---

## C6 — doctor.py  [task #7]

**Files:** `doctor.py` (rewrite), `tests/test_doctor.py`.

- Pure `Check(name, ok, hint)` + `format_report(checks) -> str` (the §8 ✓/✗ layout with hints).
- `run()` composes probes behind interfaces: ffmpeg present; models dir present (from
  `MEETSCRIBE_MODELS`); platform audio (Linux: PipeWire + a monitor source; macOS: BlackHole HAL +
  aggregate device "meetscribe"); the **RMS test-capture** (1 s, RMS>0) that catches macOS TCC
  silent capture. Exit code non-zero if any required check fails.

**Tests:** format_report renders ✓/✗ + hints; all-pass → exit 0; a failing required check → non-0;
RMS check fails on a silent buffer. Mock probes.

**Done:** green; `nix run .#doctor` wired in C7. Commit.

---

## C7 — Flake runtime wrapper + apps + overlays + HM module  [task #8]

Extend `flake.nix`; add `nix/package.nix`, `nix/module.nix`, `bin/meetscribe`.
- `bin/meetscribe`: `#!/bin/sh` → `exec python -m meetscribe "$@"`.
- Runtime venv `mkVirtualEnv "meetscribe-env" workspace.deps.default` (no dev group).
- `packages.default` = makeWrapper: install `bin/meetscribe`, `--prefix PATH` with the venv +
  `ffmpeg-headless`, `--set MEETSCRIBE_MODELS ${models}`.
- `apps.default` (record) + `apps.doctor`.
- `overlays.default`; `homeManagerModules.default` (installs meetscribe, optional
  systemd/launchd user service for auto-processing).
- 4 systems already via `eachDefaultSystem`. `nixConfig` cachix (finalized C10).

**Done:** `nix run .#doctor` runs on Linux; `nix flake check` evaluates. Commit.

---

## C8 — models.nix pinned fetchurl  [task #9]

`nix/models.nix`: `fetchurl` the 4 assets (URLs in research §3) with real sha256 (compute via
`nix store prefetch-file --unpack`? no — plain file, use `nix store prefetch-file <url>`; for the
tar.bz2 keep the archive hash and untar in the runCommand). `runCommand "meetscribe-models"`
assembles `$out/{asr,seg,spk,vad}`; use `model.int8.onnx` for segmentation, `silero_vad.onnx` for
vad. Wire `models` into the C7 wrapper.

**Done:** `nix build .#models` produces the tree; wrapper exposes `MEETSCRIBE_MODELS`. Commit.

---

## C9 — Pipeline wiring + real E2E smoke  [task #10]

`pipeline.py` orchestrates §4: resample both tracks → mic: VAD→ASR (speaker="me"); system:
diarize + VAD→ASR→align→embed(turns+centroids) → merge → write 3 artifacts. `cli process` calls it.
Real construction of sherpa objects from `MEETSCRIBE_MODELS` lives here behind the same interfaces.

**Tests:** an integration test with the real models on a short synthesized `mic.wav`+`system.wav`
asserting the 3 artifacts exist and validate (transcript schema; npz shapes (N,192); meta dim 192).
Marked slow; runs in the nix shell with `MEETSCRIBE_MODELS` set.

**Done:** one real E2E run produces valid artifacts. Commit.

---

## C10 — CI + Cachix + README  [task #11]

`.github/workflows/build.yml` matrix (ubuntu-latest + macos-14): install nix, cachix, `nix flake
check`, `nix build`. `nixConfig` substituters/keys. Flesh out README (headphone requirement, macOS
BlackHole + aggregate-device setup, doctor walkthrough). Confirm nix-shell parity note.

**Done:** CI green on both platforms (or documented gaps). Commit. Finish branch.
```
