# Analysis: Deepgram as a remote ASR/diarization backend (dual-mode pipeline)

Status: analysis — no implementation yet. Feature branch: `feat/deepgram-remote`.

## Why

The local CPU-only stack is too slow and transcript quality is only okay-ish. The known quality
ceiling is the mixed VoIP system downmix (see `diarization-and-asr-quality-limits`), but nova-3
is simply a much stronger model than what we can run on CPU, and Deepgram's diarizer sees the
same audio with a better acoustic front end. Cost is ~$0.0043/min (≈ €0.26/audio-hour) — note
we upload **two tracks**, so a 1 h meeting bills ≈ 2 h of audio ≈ **~52 ct/meeting-hour**.
Local mode stays fully supported (offline, free, private).

## What Deepgram offers (the subset we need)

Everything below is `POST https://api.deepgram.com/v1/listen` (pre-recorded REST, Nova family),
binary WAV body, `Authorization: Token $DEEPGRAM_API_KEY`, synchronous JSON response.

| Feature | Param | What we get |
|---|---|---|
| ASR | `model=nova-3` | words with `start`/`end` (float seconds — same convention as ours), `confidence`, `punctuated_word` |
| German / mixed | `language=de` or `language=multi` | `multi` = native code-switching incl. **German + English** in one stream (nova-2 only has es+en) — matches our EN-mic/DE-system reality |
| Diarization | `diarize=true` (v1) / `diarize_model=latest` | **per-word integer `speaker` + `speaker_confidence`**. No speaker-count hint. **No embedding vectors.** |
| Utterance segmentation | `utterances=true` (+`utt_split`) | ready-made utterance list: `start`/`end`/`speaker`/`transcript`/`words` — replaces our VAD-chunking + `align.py` word→segment assignment |
| Formatting | `smart_format=true` / `punctuate=true` | punctuation/casing at the source |
| Glossary boost | `keyterm=X` (repeatable) | nova-3 keyterm prompting, ≤100 terms / 500 tokens, plain terms, works with `multi` — **our `glossary.txt` + participant names, applied at recognition time instead of post-hoc LLM repair** |
| Multichannel | `multichannel=true` | per-channel transcripts (≤20 ch) — **not useful for us**: params are global per request, and we need diarize on the system track but not on mic → two separate requests map better |

Limits: 2 GB/file (16 kHz mono 16-bit ≈ 115 MB/h — no issue), sync processing capped at 10 min
(Nova transcribes far faster than realtime; multi-hour meetings fine), 429 = concurrent-request
limit (we do ≤2 sequential requests). Response `metadata` carries model name + version/UUID —
our substitute for the SHA-256 identity pin.

Tooling: `deepctl` (`dg`) is pinned in the dev group (`deepctl==0.3.0`, project-local) — docs
MCP (`.mcp.json` → `nix develop --command dg mcp`), `dg listen` for manual API experiments,
`dg models`/`dg usage` for ops. There is also a `deepgram-sdk` Python package; see "SDK or not".

## The architecture fit

### What maps cleanly

The **dual-track asymmetry survives unchanged** — it becomes two API requests:

- mic track → `nova-3`, no diarize → all words are `me` (exactly like today)
- system track → `nova-3` + `diarize` + `utterances` → speaker-labelled utterances directly

`record.py` (capture), `merge.py` (interleave + coalesce into paragraphs), `output.py`
(artifacts, bundle), `glossary.py`, and the whole terminal-UI seam stay as they are.
Deepgram's per-word speakers arrive already aligned, so **`align.py` and `vad.py` are simply
not used** on the remote path (they stay for local mode).

### The critical constraint: embeddings stay LOCAL

Deepgram returns **no speaker embedding vectors**, and even if it did, they would be from a
different model — incomparable with every vector already in pgvector (this is exactly why
`meta.json` pins the embedding model SHA). Therefore remote mode is a **hybrid**:

> Remote: ASR + diarization (the expensive, quality-critical stages).
> Local: CAM++ embedding second pass over Deepgram's diarization segments (cheap on CPU).

The embedding pass already runs from `(samples, DiarSegment list)` — we synthesize
`DiarSegment`s from Deepgram utterances (`speaker` int → `spk_N`). All existing invariants keep
holding: embedding speakers == transcript speakers, ≥0.8 s turn filter, unfiltered centroids,
dim read at runtime (192). **pgvector history stays comparable across both modes.**

### Where the seam goes

`Components` (vad/recognizer/diarizer/embedder/cleaner) is the wrong granularity for remote —
Deepgram collapses vad+recognizer+diarizer+align into one call. The seam moves one level up:

```
Backend protocol:  transcribe_tracks(mic_wav, system_wav, glossary, reporter)
                       -> (mic_utts, system_utts, system_diar_segments)

LocalBackend     = today's per-track logic (VAD → Parakeet → diarize → align), Components inside
DeepgramBackend  = two /v1/listen requests + response mapping
```

`pipeline.process()` keeps the shared tail for both: merge → coalesce → **local embed** →
cleanup → `Result`. Unit tests keep working against fakes at the new seam; the existing
Components fakes drive `LocalBackend`.

### What changes per module

| Module | Change |
|---|---|
| `pipeline.py` | extract `Backend` seam; shared tail unchanged; `build_components` → local-backend wiring |
| new `deepgram.py` | HTTP client + response→`Utterance`/`Word`/`DiarSegment` mapping; retries; clear errors (401/429/504) |
| `cli.py` | `--backend local\|deepgram` on `record`/`process` (+ env `MEETSCRIBE_BACKEND`); key via `DEEPGRAM_API_KEY` |
| `output.py` / meta | additive fields, stays `format_version: 2`: `backend`, remote `asr_model: "nova-3"` + Deepgram's model version/UUID from response metadata as the identity pin; embedding fields unchanged (still local CAM++ + SHA) |
| `glossary.py` | feed glossary + participant names as `keyterm`s (trim to limits, longest/most-specific first) |
| `doctor.py` | remote checks: key present, api.deepgram.com reachable |
| `cleanup.py` | unchanged, but see open question — likely default-off for remote |
| `record.py`, `merge.py`, `align.py`, `vad.py`, `embed.py` | untouched |

### What stops working / changes behavior (remote mode)

- **Offline/privacy invariant is broken by choice**: meeting audio (third parties' voices!)
  is uploaded to Deepgram's US cloud. Must be an explicit opt-in, never a silent default —
  and worth a GDPR moment before using it for real client meetings.
- **`--speakers N` has no remote equivalent** (no speaker-count hint on the diarizer) —
  warn + ignore in remote mode.
- Requires network + `DEEPGRAM_API_KEY`; failures need clean degradation (suggest: fail the
  run with a clear message, do NOT silently fall back to local — silent fallback would produce
  a mixed-quality artifact series without the user noticing).
- The `models-llm`/GGUF stack becomes optional for remote users (keyterms replace most of what
  cleanup did); local mode keeps it.
- Nix "no cloud" story: remote mode needs no model downloads at all — `nix run .` without the
  4.7 GB closure becomes genuinely useful.

## SDK or not

Recommendation: **no `deepgram-sdk` runtime dep** — it's one REST endpoint with a binary body;
stdlib `urllib.request` (or a tiny `httpx`-free client) keeps the uv2nix closure unchanged and
the `==`-pin policy trivial. The SDK earns its keep for streaming/agent use we don't have.
`deepctl` stays dev-group-only for docs/experiments.

## Testing strategy

- Unit: `DeepgramBackend` with an injected transport returning **captured real response JSON**
  (fixture from one paid call against a short two-speaker WAV) — response mapping, speaker
  renaming, keyterm assembly, error paths. Suite stays offline + <1 s.
- E2E: `tests/test_deepgram_e2e.py`, skip-gated on `DEEPGRAM_API_KEY` (mirrors the
  `MEETSCRIBE_MODELS` gating pattern); costs ~$0.005/run on the existing short fixtures.

## Decisions (settled with the user, 2026-09-08)

1. **Backend selection**: `local` is the default; override via **`STT_BACKEND=deepgram`**
   (env var, plus a `--backend` CLI flag mirroring it). `deepgram` requires
   `DEEPGRAM_API_KEY` to be present — fail fast with a clear message if it isn't.
2. **Language**: default **`language=multi`**. Rationale: meetings are mostly German with
   real English mixed in (and the mic track can be English); `multi` does word-level
   code-switching and the premium is negligible (batch $0.0052 vs $0.0043/min ≈ +5 ct per
   meeting-hour). **Scoping `multi` to en+de is not possible** — it's the full 10-language
   set or a single language; in practice stray third-language misdetections are rare when
   the audio only contains de/en, and each word carries a `language` tag + confidence we
   could surface later. A `--language de` escape hatch stays for pure-German meetings
   (monolingual squeezes out the last bit of accuracy and is 21% cheaper).
3. **Sample rate**: nothing to decide — `record.py` already writes both tracks as
   **16 kHz mono WAV at capture time** (the 48 kHz source is downsampled by ffmpeg during
   recording; no higher-rate original exists on disk). That matches Deepgram guidance:
   linear16 at the audio's native rate, never upsample (upsampling creates artifacts that
   hurt WER; downsampling 48→16 kHz at capture is harmless — speech ASR is wideband-16 kHz
   internally). We upload the on-disk WAVs as-is.
4. **No `deepgram-sdk`**: plain stdlib HTTP. The SDK (v7.x) would add httpx + pydantic +
   pydantic-core (native wheel) + websockets to the runtime closure for what is, for us,
   one `POST` with a bytes body and JSON response — against the repo's `==`-pin/uv2nix
   philosophy and a wider supply-chain surface. What we'd gain (typed models, live
   WebSocket streaming support) we don't need today; **revisit iff we ever stream audio
   live during recording**, which is the one scenario where the SDK pays for itself.
5. **Cleanup default on remote**: off (keyterms + `smart_format` address it at the source);
   the flag stays available.

## Suggested next step

Design doc + implementation plan (TDD, seam-first: extract `LocalBackend` with tests green,
then add `DeepgramBackend` against fixtures, then CLI/meta/doctor wiring, then the paid e2e).
