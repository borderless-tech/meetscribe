# meetscribe Terminal UI — Design

Status: **design approved, not yet implemented.** Goal: make it visible "what is going on and
when" across `record`, `process`, and `doctor`, without touching the pure, tested pipeline logic.

## Decisions

- **Library:** `rich` (pure-Python wheel, no native code → trivial in the uv2nix closure). Pinned
  to the current stable version at implementation time; lands in `deps.default` so the runtime
  wrapper ships it.
- **Fidelity:** full experience — live level-meters during `record`, a staged progress display
  for `process`, and a summary table.
- **All UI goes to stderr** (`Console(stderr=True)`), keeping stdout clean/composable. rich
  auto-detects non-TTY (pipes, CI) and degrades to plain line logs.

## Architecture — one injected seam

The pure pipeline stays pure. Introduce `src/meetscribe/progress.py`:

- **`Reporter` protocol** with a few hooks:
  - `stage(label)` → context manager: spinner while running → `✓ label   <elapsed>` on exit.
  - `track(label, total)` → handle with `.advance(n)` for progress bars (with ETA).
  - `info(msg)` → a log line (shown only in `--verbose`).
  - `summary(result)` → renders the final table.
- **`NullReporter`** — every method a no-op / dummy context manager. It is the **default**
  argument to `pipeline.process()` and `record.record_tracks()`, so the existing 70 tests run
  unchanged, fast, and rich-free.
- **`RichReporter`** — wraps `rich.Console` / `rich.Progress` / `rich.Table`. Built by the CLI and
  injected.

Injection sites are surgical: `process()` already loops over VAD chunks (→ wrap in `track`) and
calls diarization (→ sherpa's `process(callback=num_done,num_total)` feeds a `track`); each stage
gets a `stage(...)` context. `record_tracks()` gets the meter loop.

CLI flags: `--quiet` forces `NullReporter`; `--verbose` enables `info` lines. Default = rich on a
TTY, plain logs otherwise.

## Per-command UX

### record — live panel, refreshed in place
```
● recording 02:14   (Ctrl-C to stop)
  mic     ▇▇▇▆▅▂  -18 dB
  system  ▇▇▇▇▇▆   -9 dB
```
ffmpeg gains an added `astats` analysis output that writes per-stream RMS to a pipe; a pure
`parse_astats(line) -> dbfs` feeds the two meter bars. A track flatlined at silence for >~3 s
turns its meter red — a dead track is caught **during** the meeting, not after. On Ctrl-C: the
existing clean `q`-stop, then meters resolve to `mic ✓ / system ✓` (or `⚠ silent`).

### process — live stage list (mirrors §4)
```
✓ resample            0.4s
✓ VAD (system)        98 chunks
⠹ ASR (system)   ▕████████░░░░▏ 62/98   ~0:40 left
· diarize             (queued)
· embeddings          (queued)
```
The ASR bar is real — VAD yields the chunk total up front, so ETA works. Diarization is a real bar
too, driven by sherpa's progress callback. Mic and system shown as grouped sub-trees.

### summary — rich.Table on completion
```
meeting  57m 12s · 41 segments · 3 speakers
  me       23m   (mic)
  spk_0    19m
  spk_1    15m
→ transcript.json · embeddings.npz · meta.json   (in ./meeting-…)
```
Per-speaker talk time = summed utterance durations.

### doctor — keep the ✓/✗ report; add a spinner during the 1 s mic RMS capture so it never looks
frozen.

## What is pure / tested

- `parse_astats(line) -> dbfs` and `rms_to_meter(dbfs, width) -> str` — fragile parsing + bar
  rendering, TDD'd against captured ffmpeg-output fixtures (same pattern as the device-list
  fixtures).
- `summarize(result) -> Summary` — duration, segment count, per-speaker talk time. Pure, TDD'd.
- `NullReporter` — exercised by all existing tests (they stay rich-free).
- `RichReporter` — one smoke test: renders without error, and `force_terminal=False` produces
  plain output (no ANSI), proving the non-TTY fallback.

## Rollout (chunks, TDD)

- **U1** — `progress.py`: `Reporter` + `NullReporter` + `RichReporter`; `--quiet/--verbose`;
  inject null default into `process()`/`record_tracks()` (zero behavior change; all tests green).
- **U2** — `process` stage UI + `summarize()` table.
- **U3** — `record` astats side-channel + meter parsing + live meters.
- **U4** — `doctor` spinner; README screenshot/gif.

Add `rich` to `pyproject.toml`, `uv lock`, rebuild before U1.
