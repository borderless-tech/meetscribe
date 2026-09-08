# Implementation plan: Deepgram remote backend (dual-mode pipeline)

Prereq reading: `docs/plans/2026-09-08-deepgram-remote-analysis.md` (all decisions settled
there), `CLAUDE.md`. Branch: `feat/deepgram-remote`. All tests run via `nix develop -c pytest -q`
(suite is currently 245 passed, 2 skipped — it must stay green after every task).

**Non-negotiables** (from CLAUDE.md + analysis):
- No new runtime dependencies (stdlib `urllib` for HTTP; `deepctl` stays dev-group-only).
- Embeddings stay local CAM++ in BOTH modes; dim read at runtime; never hard-code 192.
- Times are float seconds; `types.py` dataclasses frozen; all UI via injected `Reporter` to
  stderr; stdout clean.
- Remote is opt-in (`STT_BACKEND=deepgram`); missing `DEEPGRAM_API_KEY` fails fast; NO silent
  fallback to local.
- `format_version` stays 2 (all meta additions are additive).
- TDD: red test first, then implementation, per task.

## Architecture (target state)

```
                 ┌────────────────────────────────────────────┐
                 │ pipeline.process(mic_wav, system_wav,       │
                 │   components, backend=None, …)              │  shared tail:
                 │   backend=None → LocalBackend(components)   │  merge → coalesce →
                 └───────────────┬────────────────────────────┘  local embed → cleanup
                                 │ backend.transcribe(...)
              ┌──────────────────┴──────────────────┐
   backends.py: LocalBackend                deepgram.py: DeepgramBackend
   (today's per-track logic:                (2× POST /v1/listen: mic plain,
    VAD → Parakeet → diarize → align)        system + diarize; response mapping)
```

### New module `src/meetscribe/backends.py`

```python
@dataclass
class BackendResult:
    mic_utts: list[Utterance]      # track="mic"; merge_tracks forces speaker="me"
    system_utts: list[Utterance]   # speaker="spk_N", track="system"
    system_diar: list[DiarSegment] # feeds the local embedding pass
    models_meta: dict              # {"asr_model": ..., "segmentation_model": ...} (+ backend extras)

class TranscriptionBackend(Protocol):
    name: str                      # "local" | "deepgram"
    def transcribe(self, mic: TrackInput | None, system: TrackInput | None,
                   glossary: list[str], reporter) -> BackendResult: ...

@dataclass(frozen=True)
class TrackInput:
    wav_path: str
    samples: np.ndarray            # 16 kHz mono float32, already loaded by process()
```

`LocalBackend(components)` moves the existing mic/system per-track blocks out of
`pipeline.process()` **verbatim** (VAD→ASR, diarize→ASR→align, the dropped-cluster warning).
`process()` keeps: loading samples, duration, the embedding pass (mic `me` + system from
`BackendResult.system_diar`), merge/coalesce, `apply_cleanup`, `Result`. `models_meta` for
local is the current hard-coded dict in `run()` — it moves into `LocalBackend`.

### New module `src/meetscribe/deepgram.py`

```python
@dataclass(frozen=True)
class DeepgramConfig:
    api_key: str
    language: str = "de"           # STT_LANGUAGE / --language
    model: str = "nova-3"
    timeout_s: float = 300.0

class DeepgramClient:              # transport seam — tests inject `opener`
    def __init__(self, config, opener=None): ...   # opener: callable(Request, timeout) → response
    def transcribe_file(self, wav_path: str, *, diarize: bool,
                        keyterms: list[str]) -> dict:  # parsed response JSON
```

- URL: `https://api.deepgram.com/v1/listen` + query
  (`model`, `language`, `smart_format=true`, `utterances=true`, and for the system track
  `diarize=true`); `keyterm` repeated per term (only sent for nova-3).
- Headers: `Authorization: Token <key>`, `Content-Type: audio/wav`; body = WAV bytes.
- Retries: 3 attempts with backoff on 429/5xx/URLError; 401/403 → actionable error naming
  `DEEPGRAM_API_KEY`; anything non-retryable raises `DeepgramError` with status + server msg.
- `keyterms_from_glossary(glossary) -> list[str]`: dedupe (case-insensitive), keep order,
  cap at 100 terms AND ≤500 whitespace-tokens total (drop overflow, longest-first is NOT
  needed — keep input order, glossary is already curated).

`DeepgramBackend(config, client=None)`:
- mic: `transcribe_file(diarize=False)` → utterances → `Utterance(start, end, "me", "mic",
  transcript, words)`.
- system: `transcribe_file(diarize=True)` → utterances → speaker int `n` → `f"spk_{n}"`;
  one `DiarSegment(start, end, spk)` per utterance.
- words: prefer `punctuated_word`, fall back to `word`; `Word(w, start, end)`.
- `models_meta`: `{"asr_model": "deepgram-<model>", "segmentation_model":
  "deepgram-diarizer", "backend_model_versions": <metadata.model_info from the system-track
  response (or mic if no system)>, "request_ids": [...]}` — the remote substitute for SHA
  pins.
- Missing `utterances` key / empty channels → clear `DeepgramError`, not a crash.
- Remote mode runs NO cleanup: `process()` is unchanged (cleaner comes from components; see
  Task 3 — remote wiring passes `NullCleaner`), transcript `cleaned:false`, `suggestions:[]`.

### Response fixture (for unit tests)

`tests/fixtures/deepgram_system.json` + `deepgram_mic.json`, hand-built to the documented
schema (`results.channels[0].alternatives[0]`, `results.utterances[]` with
`start/end/transcript/speaker/id/confidence/words[]`, words with
`word/punctuated_word/start/end/confidence/speaker`, top-level `metadata` with
`request_id/model_info/duration`). Two speakers on the system fixture, German text with one
English code-switch word, ≥1 utterance pair that must coalesce (same speaker, gap < 2 s).

## Tasks

### Task 1 — Backend seam (refactor only, no behavior change)
Files: `backends.py` (new), `pipeline.py`, `tests/test_backends.py` (new), existing tests.
1. Red: `test_backends.py` — `LocalBackend.transcribe` with the existing Components fakes
   returns mic utts labelled from ASR, system utts via align, diar segments, and the local
   `models_meta` dict.
2. Extract `LocalBackend`; rewrite `process()` to the shared-tail shape with
   `backend=None → LocalBackend(components)`; `run()` takes `models_meta` from the result
   (embedding fields still added centrally in `run()`).
3. The dropped-cluster warning moves with the system block; `Result` gains nothing.
Acceptance: full suite green with NO changes to existing test *assertions* (mechanical
signature updates are fine); `git diff` on `merge.py`/`align.py`/`embed.py` empty.

### Task 2 — `deepgram.py` (new files only; do not touch existing modules)
Files: `deepgram.py`, `tests/test_deepgram.py`, `tests/fixtures/deepgram_*.json`.
TDD the pieces in this order: query/URL building (incl. repeated `keyterm`, no keyterm when
glossary empty) → `keyterms_from_glossary` caps → response→Utterance/DiarSegment mapping
(fixtures; punctuated_word preference; spk naming; me labelling; coalescible utterances
preserved as separate utterances — coalescing is the shared tail's job) → retry/backoff with
a fake opener (429 then 200; URLError then 200; 401 message names the env var; 3rd failure
raises) → `models_meta` extraction. No network in any test.

### Task 3 — wiring: CLI/env, build_components split, meta, docs
Files: `cli.py`, `pipeline.py`, `record.py`, `output.py` (only if meta helper needs it),
`tests/test_cli.py`, `tests/test_output.py`, `CHANGELOG.md`, `CLAUDE.md`, `README.md`.
1. `--backend {local,deepgram}` + `--language` on `record` and `process`; precedence
   flag > env (`STT_BACKEND`, `STT_LANGUAGE`) > default (`local`, `de`). Bare-invocation
   getattr fallbacks must match (see the guard test pattern in `test_cli.py`).
2. `pipeline.run`: resolve backend; for `deepgram` — require `DEEPGRAM_API_KEY` (exit 2 with
   actionable message), build `DeepgramBackend`, and build components WITHOUT
   vad/recognizer/diarizer and WITHOUT the LLM cleaner (embedder only + `NullCleaner`) —
   remote must not load Parakeet (~600 MB) or the GGUF.
3. meta.json: add top-level `"backend"` (`"local"`/`"deepgram"`) + merge the backend's
   `models_meta`; `format_version` stays 2. Remote runs still record
   `embedding_model`+sha as today (local CAM++).
4. `record.run` passes backend/language through to `pipeline.run`.
5. CHANGELOG `[Unreleased]`: Added (deepgram backend, STT_BACKEND/STT_LANGUAGE, keyterm
   glossary boost) + privacy note. CLAUDE.md: short "Remote backend (Deepgram)" section
   (env vars, no-fallback rule, embeddings-stay-local invariant). README: usage snippet.
Acceptance: suite green; `meetscribe process x --backend deepgram` without key exits 2 with
a message naming DEEPGRAM_API_KEY (test via monkeypatched env).

### Task 4 — doctor remote checks
Files: `doctor.py`, `tests/test_doctor.py`.
When backend resolves to `deepgram` (env or flag): check key present (fail), DNS+TLS reach
`api.deepgram.com:443` (fail with offline hint). No billable API call. Injected socket/opener
for tests, matching doctor's existing check style.

### Task 5 — e2e (skip-gated, paid)
Files: `tests/test_deepgram_e2e.py`.
Skip unless `DEEPGRAM_API_KEY` set AND `MEETSCRIBE_MODELS` present (needs test wavs + spk
model for the local embedding pass). Mirrors `test_e2e.py` structure: run `process` with
`--backend deepgram` on the EN-mic/DE-system fixtures; assert transcript has `me` + ≥1
`spk_*`, embeddings dim==meta dim, `meta.backend == "deepgram"`, bundle builds. Mark cost in
a comment (~$0.01/run).

## Sequencing

Task 1 first (it rewrites `process()`). Tasks 2, 4, 5 are then parallel-safe (disjoint new
files / doctor-only). Task 3 last (touches `pipeline.py`+`cli.py`, needs 1's seam and 2's
module). Reviews after: correctness vs this plan, CLAUDE.md-invariant audit, test-quality
pass — findings adversarially verified before fixing.

## Explicitly out of scope (parked in the analysis doc)

Merged-single-track upload, asymmetric local-mic/remote-system mode, live streaming (SDK
revisit trigger), surfacing per-word `language` tags, `language=multi` accuracy tuning.
