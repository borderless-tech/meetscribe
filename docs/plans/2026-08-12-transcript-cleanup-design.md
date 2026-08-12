# Transcript cleanup: glossary-informed local-LLM pass

**Status:** design agreed 2026-08-12, not yet implemented.

## Why

Transcript quality on real meetings is limited by the **capture**, not the ASR model.
The system track is the default sink's `.monitor` — i.e. audio that the conferencing app
has already (1) decoded from a lossy VoIP codec, (2) **summed into one mixed stream**, and
(3) run through its own AGC/noise-suppression. Consequences, both verified on
`meetscribe-2026-08-12T12-00-55`:

- **Speaker separation is unrecoverable from our capture** — the voices are already added
  together before we see them. Diarization over-merges (a single ~1500 s CAM++ cluster that
  never splits regardless of `num_clusters` or threshold). Only the platform (per-participant
  recording / a meeting bot) could supply separable audio. Out of scope here.
- **ASR errors are dominated by the degraded audio, not the model.** Parakeet-TDT-int8 and
  Canary-180m make the *same* errors on the same spans (both mangle "Borderless"). Beam search
  is *worse* than greedy (drops words). Canary can't be a drop-in (no timestamps, >40 s crash,
  melts down on wrong forced language).

So the lever is a **post-ASR enhancement layer**, not the mic path. Two candidate levers were
considered; one was dropped:

- **Hotwords / contextual biasing (dropped).** sherpa transducer hotwords require
  `modified_beam_search` (which regressed quality in testing) *and* a `bpe_vocab` the pinned
  Parakeet tarball does not ship (only `tokens.txt`). Not practical on our model.
- **Glossary-informed local-LLM cleanup (this design).** An LLM corrects garbled words from
  *context* + a known-terms glossary, restores punctuation, lightly smooths disfluencies. It
  fixes "Borderlestern"→"Borderless" the way re-listening cannot, because it reasons from the
  surrounding words rather than the already-lossy audio.

Transcript goal (per product owner): human reading/archive + summarizer input + search — **not
verbatim**, so LLM paraphrase/smoothing is acceptable. Local/offline/CPU constraint stands; a
CPU GGUF via llama.cpp is acceptable.

## Architecture

New pure stage after `merge`/`coalesce`, operating on the final `Utterance` list (best context).
Follows the two existing seams:

- **`cleaner` injected into `Components`** (alongside vad/recognizer/diarizer/embedder),
  satisfying a `Cleaner` protocol: `clean(segments, glossary) -> list[str]`. Default is
  **`NullCleaner`** (no-op) so the unit suite stays LLM-free and fast, mirroring `NullReporter`.
  `build_components()` wires the real llama.cpp cleaner.
- **`Reporter`** gains a `"cleanup (llm)"` stage with a per-segment progress bar and `warn`s.

**Timestamps are never touched.** We clean only `segment.text`; per-word `words` (with timings)
stay verbatim → diarization alignment, embeddings, and provenance intact. `text` will no longer
equal `" ".join(words)` — intentional, documented.

### CLI surface

- **On by default.** `meetscribe` (record) and `meetscribe process <dir>` run cleanup as the
  final stage. Because it's default-on, the GGUF is a **pinned flake model** (SHA-256) so
  `nix run .` always has it. Costs on the record: default closure +~5 GB; normal processing
  gains a few minutes of CPU inference (bounded by segment count).
- **`--no-cleanup`** skips it entirely → raw ASR text (the fast path).
- **`meetscribe clean <dir>`** — new subcommand (sibling to `bundle`). Runs cleanup on an
  **existing** `transcript.json` (no audio, no re-ASR). **Non-destructive:** writes
  `<dir>-cleanup/` with cleaned `transcript.json` + copies of `embeddings.npz` / updated
  `meta.json`; original untouched. The A/B tool.

## Glossary & name prompt

- **Persistent glossary** at `~/.config/meetscribe/glossary.txt`, one term/phrase per line
  (org, products, recurring colleagues, jargon). Read every run. Seeded with the obvious
  (`Borderless`, the user's name). Missing → empty; cleanup still runs (general smoothing).
- **Optional name prompt.** Hooks onto the existing participant-count question. Only when a
  count ≥ 2 was given **and** stdin is a TTY, follow with a skippable loop ("Name of a
  participant? (Enter to skip)"). Each entry optional; empty line stops early; entered names
  **appended to the persistent glossary** (case-insensitive dedup) so recurring people are
  remembered. Entering 2 of 4 and skipping keeps the 2 — nothing discarded. Any decline
  (empty/EOF/Ctrl-C) never blocks processing.
- **No speaker mapping.** Names are flat spelling hints only — we do *not* attach them to
  `spk_N` (diarization is unreliable on the mixed downmix).

## LLM stage

- **Model:** Qwen2.5-7B-Instruct GGUF `Q4_K_M` (~4.7 GB), pinned by SHA-256 in `nix/models.nix`.
  Strong German, permissive license, deterministic at temp 0. Swappable.
- **Integration — managed `llama-server` subprocess** (from nixpkgs). Spawn once, poll
  `/health` until ready, POST all segments over local HTTP, terminate. Same lifecycle pattern
  as ffmpeg in `record.py`. Keeps the Python dep surface tiny (stdlib `urllib` only; no native
  `llama-cpp-python` in the uv2nix closure) and loads weights once. HTTP client behind a thin
  `LlamaClient.complete(prompt) -> str` protocol for testability.
- **Port robustness.** llama.cpp is HTTP-only (no stdio transport). Bind **loopback only**
  (`127.0.0.1`). Pick a random high port (20000–60000), test-bind, retry up to **5×** on
  collision; all 5 fail → `warn` + skip cleanup (non-destructive, = `--no-cleanup`).
- **Prompting — one call per segment, greedy/temp 0** (deterministic → reproducible + stable
  A/B). Each call: fixed system instruction + glossary + previous segment as read-only context
  + current segment to fix. Instruction scoped: correct spelling / split-merged words /
  misrecognized names (via glossary), restore punctuation & capitalization; **do NOT** translate,
  summarize, paraphrase, or add; keep original language; return unchanged if already fine;
  output only the corrected text.
- **Anti-hallucination guards:** length ratio >3× or <0.3× → discard, keep raw; empty /
  echoed-prompt → keep raw. Every trip `warn`s and is counted. Worst case degrades per-segment
  to the current raw transcript.

## Data flow & outputs

- **`transcript.json`:** each segment keeps verbatim `words` (timings) + gains cleaned `text`;
  add `raw_text` (original ASR text) for provenance / the verbatim edge case. Top-level
  `"cleaned": true|false`.
- **`meta.json`:** add `cleanup_model` name + **SHA-256** + params (temp, quant) — same
  identity-travels-with-artifact rule as embeddings. Bump `format_version` 1 → 2.
  `--no-cleanup` → `cleaned:false`, no `cleanup_model` block.
- **`clean <dir>`:** writes `<dir>-cleanup/` (cleaned transcript + copied embeddings + updated
  meta); a complete, bundle-able set to diff against the original.
- **Bundle:** `bundle_dir` / `default_bundle_name` unchanged — same three members.

## Error handling & determinism

Failure ladder, every rung non-destructive (→ raw text):
binary/model missing → skip; no free port after 5 → skip; server unhealthy within ~120 s → skip;
per-segment HTTP error/timeout (~60 s cap) → keep that segment raw; guard trip → keep raw.
Summary reports `cleaned N/M (K kept raw)`. Server owned by a `try-finally` that **always
terminates** it (success/exception/Ctrl-C) — no orphan, like `stop_ffmpeg`.

Determinism: greedy (temp 0) + fixed seed + SHA-256-pinned GGUF → same in, same out. Honest
caveat: llama.cpp multithreaded CPU math isn't bit-identical across thread counts, but greedy
argmax essentially never flips — stable in practice; not chasing single-thread bit-repro.

## Testing

HTTP behind `LlamaClient` protocol → fakes everywhere, suite stays <1 s.

- **Unit:** glossary read/append/dedup; name-prompt loop (gating, early-stop keeps prior, EOF
  safe); guard logic (good/runaway/empty/echo → cleaned vs keep-raw + counters); `process()`
  with FakeCleaner (default-on, `--no-cleanup` bypass, `words` untouched, `text`/`raw_text`,
  `cleaned` flag); output/meta schema (raw_text/cleaned, cleanup_model+sha, format_version 2);
  `clean <dir>` non-destructive; free-port finder (retry ≤5 then give-up); server lifecycle
  (`terminate` in `finally`).
- **E2E (opt-in, pinned GGUF, mirrors `test_e2e.py` skip):** real `llama-server` on a short
  transcript with a seeded garbled glossary term → corrected + timings intact.
