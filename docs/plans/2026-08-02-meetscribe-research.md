# meetscribe — Pre-Implementation Research Report

Consolidated authoritative reference (from a 7-agent research + adversarial cross-check
workflow). Where a cross-check disagreed with an original finding, the check is preferred
and the correction is noted inline. Verified 2026-08-02 against primary sources.

> **This report resolves what-we-build.md §10 ("Vor dem Bauen verifizieren").** Two facts
> here *contradict* the spec and the report wins: (a) the speaker-embedding dimension is
> **192, not 512**; (b) sherpa-onnx is one pip dep (`sherpa-onnx`), which pins the split
> native package transitively.

---

## 1. sherpa-onnx: pinned version + pyproject.toml packages

**Pinned version: `1.13.4`** (latest stable, released 2026-07-07, tag `v1.13.4`; bundles
onnxruntime 1.27.0).

**Package split — CONFIRMED.** As of 1.13.4 the distribution is three PyPI packages,
published in lockstep at `==1.13.4`:

| PyPI package | Role | requires_dist | Needed for Python API? |
|---|---|---|---|
| `sherpa-onnx` | pybind11 extension (`import sherpa_onnx`); ABI-tagged cp3xx wheels | `sherpa-onnx-core==1.13.4` | **Yes** |
| `sherpa-onnx-core` | native runtime shared lib; `py3-none-manylinux2014` wheels | `null` | **Yes** (transitive) |
| `sherpa-onnx-bin` | CLI executables only | `sherpa-onnx-core==1.13.4` | No |

**FINAL — list in `pyproject.toml`:** `sherpa-onnx==1.13.4` alone. It transitively pins
`sherpa-onnx-core==1.13.4` (the native lib). Do not add `sherpa-onnx-bin`.

Sources: pypi.org/pypi/sherpa-onnx/json · /sherpa-onnx-core/1.13.4/json · github.com/k2-fsa/sherpa-onnx/releases

---

## 2. Python API cheat-sheet

All class/factory names and result properties confirmed against primary pybind11 source.

### 2a. OfflineRecognizer — Parakeet-TDT with timestamps

```python
import sherpa_onnx
recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder="encoder.int8.onnx", decoder="decoder.int8.onnx",
    joiner="joiner.int8.onnx", tokens="tokens.txt",
    num_threads=4, sample_rate=16000, feature_dim=80,
    decoding_method="greedy_search",
    model_type="nemo_transducer",   # REQUIRED for Parakeet-TDT / NeMo transducer
    provider="cpu", debug=False,
)
s = recognizer.create_stream()
s.accept_waveform(16000, samples)   # samples: float32 numpy, mono, 16 kHz
recognizer.decode_stream(s)         # or decode_streams([...]) for batch
r = s.result
r.text        # str
r.tokens      # list[str] per-token
r.timestamps  # list[float] seconds; timestamps[i] aligns 1:1 with tokens[i]
r.durations   # list[float] per-token durations
r.words       # list[str] — OFTEN EMPTY for BPE/subword models → build words from tokens
```

- `from_transducer` is a `@classmethod`. `model_type="nemo_transducer"` confirmed via C++
  `offline-recognizer-impl.cc` + the NeMo-transducer models doc. Set it explicitly.
- Result props confirmed in `python/csrc/offline-stream.cc`: `text, tokens, words,
  timestamps, durations, lang, emotion, event, segment_texts, ...`.
- `r.words` is often empty for subword models → group `tokens`/`timestamps` into words in
  `asr.py` (word boundary = token starting with the SentencePiece "▁" marker).

### 2b. Silero VAD (+ offset pattern)

```python
config = sherpa_onnx.VadModelConfig()
config.silero_vad.model = "silero_vad.onnx"
config.silero_vad.threshold = 0.5
config.silero_vad.min_silence_duration = 0.25
config.silero_vad.min_speech_duration = 0.25
config.silero_vad.max_speech_duration = 20.0
config.silero_vad.window_size = 512     # Silero v4 @16k — VERIFY vs the .onnx (v5 differs)
config.sample_rate = 16000
config.num_threads = 1
config.provider = "cpu"
vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=100)
vad.accept_waveform(samples_window)     # len(window) == window_size
while not vad.empty():
    seg = vad.front                     # SpeechSegment
    audio_chunk   = seg.samples         # float32 numpy of detected speech
    start_seconds = seg.start / config.sample_rate     # OFFSET → absolute time
    # NOTE: seg.end does NOT exist; end = start_seconds + len(seg.samples)/sample_rate
    vad.pop()
```

- `SpeechSegment` exposes read-only `start` (int sample idx) and `samples` only. **No
  `seg.end`.** Derive end from `len(samples)`.
- ⚠️ `window_size=512` is Silero v4; confirm against the pinned `silero_vad.onnx`.

### 2c. OfflineSpeakerDiarization (result fields)

```python
config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
    segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
        pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
            model="model.int8.onnx")),   # pyannote segmentation-3.0 ONNX (int8 for CPU)
    embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model="3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
        num_threads=1, provider="cpu"),
    clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=0.5),
    min_duration_on=0.3, min_duration_off=0.5,
)
sd = sherpa_onnx.OfflineSpeakerDiarization(config)
result = sd.process(audio, callback=None).sort_by_start_time()   # audio: f32 mono @ sd.sample_rate
for seg in result:
    seg.start   # float s
    seg.end     # float s
    seg.speaker # int speaker id
```

- `num_clusters=-1` + `threshold` = agglomerative, no fixed K (matches spec §12).
- Result seg fields confirmed: read-only `start, end, duration, speaker`.

### 2d. SpeakerEmbeddingExtractor (dim)

```python
config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
    model="3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
    num_threads=1, provider="cpu")
extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
stream = extractor.create_stream()
stream.accept_waveform(sample_rate=16000, waveform=samples)   # arg names confirmed
stream.input_finished()
embedding = extractor.compute(stream)   # float32 vector, length == extractor.dim
dim = extractor.dim                     # AUTHORITATIVE — read at runtime
```

- **CRITICAL — embedding dim is `192`, NOT 512.** The cross-check downloaded the actual
  `.onnx`; `metadata_props` literally contains `output_dim=192` (no "512" anywhere). The
  spec §6's `(N, 512)` / `embedding_dim: 512` is WRONG. Read `extractor.dim` at runtime,
  store it in `meta.json`, size all npz arrays as `(N, dim)`, and `assert extractor.dim == 192`.

---

## 3. Models table

All four URLs are real, current release assets (verified via GitHub REST API + download).

| key | url | filename | packaging | inner files (used) | size |
|---|---|---|---|---|---|
| **asr** | `https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2` | `sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2` | tar.bz2 | `encoder.int8.onnx`, `decoder.int8.onnx`, `joiner.int8.onnx`, `tokens.txt` | ~465 MB archive |
| **speaker** | `https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx` | same | single `.onnx` | n/a | ~27 MB |
| **segmentation** | `https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2` | `sherpa-onnx-pyannote-segmentation-3-0.tar.bz2` | tar.bz2 | `model.int8.onnx` (use int8) | ~7 MB |
| **vad** | `https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx` | `silero_vad.onnx` | single `.onnx` | n/a | ~640 KB |

- ASR **v3 is MULTILINGUAL** (25 European langs incl. German `de`) — NOT v2 (English-only).
- Speaker tag spelling is `speaker-recongition-models` (upstream typo — keep it).
- Use `model.int8.onnx` from the segmentation archive for CPU.

---

## 4. uv2nix / pyproject-nix

**Org is `github:pyproject-nix/*` now (NOT `adisbladis/*`).**

### Flake inputs

```nix
inputs = {
  nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  pyproject-nix = { url = "github:pyproject-nix/pyproject.nix"; inputs.nixpkgs.follows = "nixpkgs"; };
  uv2nix = {
    url = "github:pyproject-nix/uv2nix";
    inputs.pyproject-nix.follows = "pyproject-nix";
    inputs.nixpkgs.follows = "nixpkgs";
  };
  pyproject-build-systems = {
    url = "github:pyproject-nix/build-system-pkgs";
    inputs.pyproject-nix.follows = "pyproject-nix";
    inputs.uv2nix.follows = "uv2nix";
    inputs.nixpkgs.follows = "nixpkgs";
  };
};
```

### Build snippet (with native-wheel fixup)

```nix
let
  inherit (nixpkgs) lib;
  pkgs = nixpkgs.legacyPackages.${system};
  python = pkgs.python312;   # must satisfy requires-python

  workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };
  overlay = workspace.mkPyprojectOverlay { sourcePreference = "wheel"; };

  # Only supply the missing native lib. Do NOT add autoPatchelfHook — pyprojectWheelHook
  # already runs it on binary wheels. Apply to BOTH split packages.
  pyprojectOverrides = final: prev: {
    sherpa-onnx = prev.sherpa-onnx.overrideAttrs (old: {
      buildInputs = (old.buildInputs or [ ]) ++ [ pkgs.stdenv.cc.cc.lib ];
    });
    sherpa-onnx-core = prev.sherpa-onnx-core.overrideAttrs (old: {
      buildInputs = (old.buildInputs or [ ]) ++ [ pkgs.stdenv.cc.cc.lib ];
    });
  };

  pythonSet = (pkgs.callPackage pyproject-nix.build.packages { inherit python; }).overrideScope
    (lib.composeManyExtensions [
      pyproject-build-systems.overlays.wheel   # NOT .default (== sdist)
      overlay
      pyprojectOverrides                        # AFTER overlay
    ]);

  venv = pythonSet.mkVirtualEnv "meetscribe-env" workspace.deps.default;
in venv
```

**Corrections vs the spec skeleton:**
- Use `pyproject-build-systems.overlays.wheel`, not `.default`, with `sourcePreference="wheel"`.
- Do NOT add `autoPatchelfHook` — the wheel builder already runs it. Only add `stdenv.cc.cc.lib`.
- Apply the override to BOTH `sherpa-onnx` and `sherpa-onnx-core`.
- devShell hygiene: `UV_NO_SYNC=1`, `UV_PYTHON=pythonSet.python.interpreter`,
  `UV_PYTHON_DOWNLOADS=never`, `unset PYTHONPATH`; never `uv run` inside the shell.

---

## 5. "Verify at build time" checklist (ranked by risk)

1. **[HIGH]** uv.lock attribute key form (`sherpa-onnx` vs `sherpa_onnx`) for the override —
   wrong key = silent no-op → runtime libstdc++ load failure. Verify both attrs exist.
2. **[HIGH]** Confirm `uv.lock` pins `sherpa-onnx==1.13.4` (split); older = no `-core` attr.
3. **[HIGH]** If autoPatchelf reports a missing `.so` beyond libstdc++/libgcc_s, add its
   nixpkgs pkg to `buildInputs` (onnxruntime is statically bundled, so likely fine).
4. **[MED]** Silero VAD `window_size` — confirm v4 (512) vs v5 from the `.onnx` metadata.
5. **[MED]** `assert extractor.dim == 192` at load; no downstream hard-coded 512.
6. **[MED]** Compute SHA-256 for the 4 model assets for `fetchurl` pinning (do at build time).
7. **[LOW]** `model_type="nemo_transducer"` explicit is safe; verify ONNX metadata present.
8. **[LOW]** aarch64-linux / macOS wheels exist but were not exercised; Linux x86_64 is clean.
