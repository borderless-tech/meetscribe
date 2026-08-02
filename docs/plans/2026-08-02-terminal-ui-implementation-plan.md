# Terminal UI Implementation Plan (U1–U4)

> Execute task-by-task with TDD. Run tests in the nix shell: `nix develop --command pytest`.
> Design + rationale: `docs/plans/2026-08-02-terminal-ui-design.md`.

**Goal:** Visible progress across `record`/`process`/`doctor` via `rich`, without touching the
pure pipeline logic — one injected `Reporter` seam, `NullReporter` by default (tests stay green).

**Dependency:** add `rich` (pinned to whatever `uv` resolves) to `pyproject.toml` `dependencies`,
`uv lock`, rebuild. Pure-Python wheel — no closure pain.

---

## U1 — `progress.py` Reporter seam  [task #12]

**Files:** create `src/meetscribe/progress.py`, `tests/test_progress.py`; modify
`pyproject.toml`, `uv.lock`, `src/meetscribe/pipeline.py`, `src/meetscribe/record.py`,
`src/meetscribe/cli.py`.

**Interface (`progress.py`):**
```python
class Reporter(Protocol):
    def stage(self, label: str) -> ContextManager[None]: ...
    def track(self, label: str, total: int) -> "TrackHandle": ...   # .advance(n=1)
    def info(self, msg: str) -> None: ...
    def summary(self, summary: "Summary") -> None: ...

class NullReporter:   # all no-ops; stage()/track() return dummy CMs/handles
class RichReporter:   # Console(stderr=True); verbose flag gates info()
```
- `pipeline.process(..., reporter: Reporter = NullReporter())` and
  `record.record_tracks(..., reporter: Reporter = NullReporter())` — default keeps behavior + all
  70 tests unchanged.
- `cli.build_parser`: add `--quiet` / `--verbose` (global). `main` builds `RichReporter` unless
  `--quiet` → `NullReporter`.

**Tests (`test_progress.py`):**
- `NullReporter().stage("x")` usable as a context manager, no output; `track().advance()` no-ops.
- `RichReporter` renders a stage without error; with `force_terminal=False` the captured output
  contains **no ANSI escape** (plain fallback).
- `--quiet` selects NullReporter; `--verbose` sets the verbose flag (parser-level test).

**Done:** all existing tests still green + new progress tests. Commit.

---

## U2 — `process` stage UI + summary table  [task #13]

**Files:** modify `pipeline.py`; create `summarize` in `progress.py` (or `summary.py`); tests in
`tests/test_summary.py`, extend `tests/test_pipeline.py`.

- Pure `summarize(result: Result) -> Summary` — `duration_s`, `n_segments`, and speakers with
  talk time = summed utterance durations, ordered by talk time (mic "me" flagged).
- `process()` wraps each stage in `reporter.stage(...)`; ASR loop uses
  `reporter.track("ASR", len(chunks))` + `.advance()`; diarization passes a `callback` to sherpa's
  `process(callback=…)` that advances a `track`. All via the injected reporter (Null in tests).
- `RichReporter.summary` renders a `rich.Table`.

**Tests:** `summarize` — talk-time per speaker, ordering, empty; a pipeline test asserting a
`RecordingReporter` (test double) receives the expected stage/track/summary calls.

**Done:** green. Commit.

---

## U3 — `record` live level meters  [task #14]

**Files:** modify `record.py`; create `tests/fixtures/astats_output.txt`; extend
`tests/test_record.py`; meter rendering in `progress.py`.

- Pure `parse_astats(line) -> float | None` (dBFS from ffmpeg `astats`/`ametadata` RMS lines) and
  `rms_to_meter(dbfs, width=8) -> str` (block-bar). TDD against a **captured real** ffmpeg astats
  fixture.
- Extend the ffmpeg command with an `astats` analysis output writing metadata to a pipe; a reader
  thread updates a `RichReporter` live two-track meter panel (red when a track's dBFS stays at
  silence >~3 s). Keep the clean `q`-stop.

**Tests:** `parse_astats` on real fixture lines (value + non-match → None); `rms_to_meter`
monotonic + clamps at 0 dB / −∞. (The live panel + ffmpeg wiring verified by a real short capture,
like C2.)

**Done:** green + real capture check. Commit.

---

## U4 — `doctor` spinner + README + review  [task #15]

- `doctor`: wrap the 1 s mic RMS capture in `reporter.stage("testing microphone")` (spinner).
- README: document `--quiet`/`--verbose`; add a UI example block.
- Final verification: full suite + `nix flake check` + real e2e; then an adversarial multi-agent
  review workflow over the whole UI addition; address findings.

**Done:** all green; branch finished.
