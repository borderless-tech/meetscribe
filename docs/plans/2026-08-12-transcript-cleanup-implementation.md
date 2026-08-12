# Transcript Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a default-on, glossary-informed local-LLM cleanup stage that corrects garbled
proper nouns / spelling and lightly smooths ASR transcript text, fully offline on CPU.

**Architecture:** New pure stage after `merge`/`coalesce`, injected as a `cleaner` component
(NullCleaner default → tests stay LLM-free). Real cleaner drives a managed `llama-server`
subprocess (Qwen2.5-7B GGUF) over loopback HTTP. Cleans only `segment.text`; per-word timings
untouched. Design: `docs/plans/2026-08-12-transcript-cleanup-design.md`.

**Tech Stack:** Python 3.12, sherpa-onnx (existing), llama.cpp `llama-server` (nixpkgs), a
pinned Qwen2.5-7B-Instruct Q4_K_M GGUF, stdlib `urllib`/`socket`/`subprocess`, pytest, Nix.

**Sequencing:** Tasks 1–7 are pure/injected and fully verifiable in the existing fast suite
without the model. Tasks 8–11 need the real binary+GGUF (heavy download/build) and are the
network/ML boundary — unit-tested with mocks, real path exercised only by the opt-in e2e test.

---

## Review amendments (2026-08-12, AUTHORITATIVE — override task bodies where they conflict)

Folded in from a 4-reviewer plan review (verified against the codebase). Verdict: sound to
execute after these; no architectural rethink.

- **A. New Task 4a — `read_transcript` (in `output.py`).** No JSON→Utterance reader exists
  (`output.py` is write-only). Add `read_transcript(path) -> (meeting_id, duration_s,
  list[Utterance])`, symmetric to `write_transcript`, rebuilding `Word(w,start,end)` tuples
  exactly (Task 11 asserts byte-identical timings). Apply `raw_text = seg.get("raw_text",
  seg["text"])` for v1 back-compat; for an already-cleaned v2 input, preserve the original
  `raw_text` (don't overwrite). Own TDD test. `clean_existing` (Task 7) depends on it.
- **B. `cleaned` flag needs a real seam (Tasks 4/5/6).** It is a *run-level* fact, NOT
  derivable from `text != raw_text` (a real pass may no-op; NullCleaner leaves them equal).
  So: add `cleaned: bool = False` param to `write_transcript`; add `cleaned: bool` +
  `cleanup_model: dict | None` fields to the `Result` dataclass (`process()` is the single
  source of truth, set from whether a non-Null cleaner actually ran via `CleanResult`);
  `run()` passes both to `write_transcript(cleaned=...)` and `build_meta`. Graceful fallback
  to NullCleaner → `cleaned:false`, no `cleanup_model`.
- **C. Sequencing (Task 7 vs 9).** Task 7's `run()` must NOT pass `cleanup` to
  `build_components`; it only selects process/NullCleaner behaviour. Defer the
  `build_components(cleanup=...)` signature change **and** the three `test_pipeline.py`
  monkeypatch lambda fixes (lines ~336/358/378 — add `cleanup=True`/`**kwargs`) to Task 9,
  in one commit, so the suite never goes red on the wrong task.
- **D. process() stays cleanup-agnostic (Task 5).** No `cleanup` bool on `process()`; drive
  it purely from `components.cleaner` (default `NullCleaner`), exactly like `num_speakers`
  flows `run→build_components`.
- **E. `raw_text` is serialize/read-time fallback, not a dataclass default (Task 4).**
  `Utterance` gains `raw_text: str = ""`. In `_utterance_to_dict`: `"raw_text": u.raw_text
  or u.text`. Populate `raw_text` ONLY in the Task-5 cleaner stage via `replace(u,
  text=cleaned, raw_text=u.text)` — never at align/mic construction, never in
  `coalesce_utterances` (add a comment there saying so). Task-4 test asserts `raw_text ==
  text` (fallback) since no cleaner runs yet.
- **F. Task 4 red step edits the EXISTING schema test.** `test_write_transcript_matches_schema`
  asserts `seg == {exact dict}` (no `raw_text`); rewrite that dict to include `raw_text` and
  assert the top-level `cleaned` key in the red step (not just add a new test). `format_version`
  tests read the constant — keep it that way through the 1→2 bump.
- **G. Cleaner protocol signature (Tasks 3/5).** Authoritative: `clean(texts, glossary,
  reporter) -> CleanResult`. The Task-5 `FakeCleaner` returns a `CleanResult`, not a list.
- **H. Guards (Task 3).** `accept_candidate(input, candidate)`: guard the zero-division for
  empty `input` (test `("", "x")`); ratio is `len(candidate)/len(input)`; keep the `>3×`
  runaway guard unconditional but apply the `<0.3×` collapse guard **only when
  `len(input) > 40` chars** (short backchannels like `"uh the the"→"the"` are legit). Add a
  boundary test. **Drop the echo guard (YAGNI)** — empty + ratio already cover the failure
  modes, and `accept_candidate` has no access to the prompt anyway.
- **I. Mic track is NOT cleaned (Task 5).** Clean only `track=='system'` utterances (the
  degraded downmix); pass mic (`me`) text through unchanged (still `raw_text==text`). The
  glossary/garble rationale doesn't apply to the clean user track. Add a pass-through test.
- **J. Task 8 (llama.cpp client) specifics.** Qwen2.5-Instruct needs **ChatML** — use
  `/v1/chat/completions` (or build `<|im_start|>` framing for `/completion`); document the
  choice. Make readiness (cold 4.7 GB load can exceed 120 s) and per-request timeouts
  generous + configurable; the skip-fallback `warn` must say cleanup was skipped and why.
  Pin `llama-server --threads` to a fixed value so the greedy+seed determinism claim holds.
  State a realistic total latency budget (sequential per-segment CPU inference over hundreds
  of segments is minutes-to-tens-of-minutes); optionally cap total time and fall back to raw
  for the remainder.
- **K. Task 10 (Nix) recipe fixes + OPEN DECISION.** (1) Pin
  `bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/<COMMIT-SHA>/Qwen2.5-7B-Instruct-Q4_K_M.gguf`
  (single 4.68 GB file) at a **commit revision**, not `main` (mirrors the release-tag
  immutability; the official Qwen repo splits Q4_K_M into 2 parts → single fetchurl 404s).
  Get the hash via `nix store prefetch-file --hash-type sha256 <url>` (not deprecated
  `nix-prefetch-url`). (2) Add `pkgs.llama-cpp` (it provides the `llama-server` binary; there
  is no `pkgs.llama-server`, `mainProgram` is `llama`) to the wrapper `makeBinPath` in
  `nix/package.nix`; invoke `llama-server` by name; don't strip (GGML_BACKEND_DL RPATH).
  (3) **OPEN DECISION — confirm with the user before executing Task 10:** default-on cleanup
  puts the 4.7 GB GGUF in the default `models` closure → CI (`build.yml`, both ubuntu+macos,
  **no cache**) fetches 4.7 GB + builds llama-cpp from source per runner. Options: **(a)** keep
  it in the default closure (true out-of-the-box on-by-default, expensive CI — likely needs
  Cachix first, and/or drop the models/default build from the macOS CI leg), or **(b)** gate
  the GGUF behind a separate flake output `packages.models-llm` (lean default/CI; cleanup then
  falls back to raw on a fresh machine until the user builds that output — i.e. on-by-default
  at runtime but opt-in at the *build* layer). This trades the user's "on by default" wish
  against CI cost/closure size.

---

### Task 1: Glossary module (read / append / dedup)

**Files:**
- Create: `src/meetscribe/glossary.py`
- Test: `tests/test_glossary.py`

Pure. Path resolves `$XDG_CONFIG_HOME/meetscribe/glossary.txt` else `~/.config/...`. One
term per line; blank lines and lines starting `#` ignored. Append is case-insensitive dedup,
order-preserving, creates the file/dir if missing.

**Step 1 — failing tests:**
```python
# tests/test_glossary.py
from meetscribe import glossary

def test_load_missing_file_is_empty(tmp_path):
    assert glossary.load(tmp_path / "glossary.txt") == []

def test_load_skips_blanks_and_comments(tmp_path):
    p = tmp_path / "g.txt"; p.write_text("Borderless\n\n# a comment\nGeorg\n")
    assert glossary.load(p) == ["Borderless", "Georg"]

def test_append_dedups_case_insensitively(tmp_path):
    p = tmp_path / "g.txt"; p.write_text("Borderless\n")
    glossary.append(p, ["georg", "BORDERLESS", "Christian"])
    assert glossary.load(p) == ["Borderless", "georg", "Christian"]

def test_append_creates_missing_file(tmp_path):
    p = tmp_path / "sub" / "g.txt"
    glossary.append(p, ["Ana"])
    assert glossary.load(p) == ["Ana"]

def test_default_path_honours_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert glossary.default_path() == tmp_path / "meetscribe" / "glossary.txt"
```

**Step 2 — run, expect FAIL** (`module has no attribute`): `pytest tests/test_glossary.py -v`

**Step 3 — implement `glossary.py`:**
```python
"""Persistent known-terms glossary (org/product/people names) feeding the LLM cleanup.

One term per line; blanks and ``#`` comments ignored. Append is case-insensitive,
order-preserving dedup so recurring names accumulate without duplicating."""
from __future__ import annotations
import os
from pathlib import Path

def default_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path.home() / ".config"
    return root / "meetscribe" / "glossary.txt"

def load(path: str | Path) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    out: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out

def append(path: str | Path, terms: list[str]) -> list[str]:
    p = Path(path)
    existing = load(p)
    seen = {t.casefold() for t in existing}
    added = [t.strip() for t in terms if t.strip() and t.strip().casefold() not in seen]
    for t in added:
        seen.add(t.casefold())
    if added:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            for t in added:
                f.write(t + "\n")
    return existing + added
```

**Step 4 — run, expect PASS.** **Step 5 — commit** `feat: persistent known-terms glossary`.

---

### Task 2: Participant name prompt

**Files:**
- Modify: `src/meetscribe/record.py` (near `ask_participants`, ~line 412)
- Test: `tests/test_record.py`

Pure loop with injected `input_fn` (mirrors `ask_participants`). Only meaningful when a count
≥ 2 was given. Empty line stops early keeping prior entries; EOF/KeyboardInterrupt → return
what we have. Caller appends results to the glossary.

**Step 1 — failing tests:**
```python
def test_ask_names_collects_until_blank():
    answers = iter(["Georg", "Christian", ""])
    from meetscribe.record import ask_participant_names
    assert ask_participant_names(4, input_fn=lambda _: next(answers)) == ["Georg", "Christian"]

def test_ask_names_skipped_when_count_below_two():
    from meetscribe.record import ask_participant_names
    assert ask_participant_names(1, input_fn=lambda _: "x") == []

def test_ask_names_eof_keeps_prior():
    def boom(_):
        if boom.n == 0:
            boom.n += 1; return "Georg"
        raise EOFError
    boom.n = 0
    from meetscribe.record import ask_participant_names
    assert ask_participant_names(3, input_fn=boom) == ["Georg"]
```

**Step 2 — FAIL. Step 3 — implement:**
```python
def ask_participant_names(count: int, input_fn=input) -> list[str]:
    """Optionally collect participant names (spelling hints for the glossary).

    Only prompts when ``count`` >= 2. Each entry optional; a blank line stops early.
    Any decline (blank/EOF/Ctrl-C) never blocks — returns whatever was entered."""
    if count < 2:
        return []
    names: list[str] = []
    for _ in range(count):
        try:
            ans = input_fn("Name of a participant? (Enter to skip) ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not ans:
            break
        names.append(ans)
    return names
```

**Step 4 — PASS. Step 5 — commit** `feat: optional participant-name prompt for the glossary`.

(Wiring into `record.run` — appending to `glossary.default_path()` — lands in Task 7 with the
rest of the CLI/flow changes so the seam is tested end to end there.)

---

### Task 3: Cleaner protocol, NullCleaner, guards

**Files:**
- Create: `src/meetscribe/cleanup.py`
- Test: `tests/test_cleanup.py`

`LlamaClient` protocol (`complete(prompt: str) -> str`). `Cleaner` protocol
(`clean(texts, glossary, reporter) -> CleanResult`). `NullCleaner` returns inputs unchanged.
Guard logic is pure: reject a candidate (keep raw) when it is empty, or its length ratio to the
input is > 3.0 or < 0.3, or it echoes the prompt. `CleanResult` carries `texts` + counts
(`cleaned`, `kept_raw`).

**Step 1 — failing tests** (fake client, no LLM):
```python
from meetscribe.cleanup import LlamaCleaner, NullCleaner, accept_candidate

def test_accept_candidate_rules():
    assert accept_candidate("Borderless GmbH", "Borderlestern GmbH")      # ok
    assert not accept_candidate("Borderless GmbH", "")                     # empty
    assert not accept_candidate("hi", "hi " * 50)                          # runaway >3x
    assert not accept_candidate("a very long original sentence here", "a") # collapse <0.3x

class FakeClient:
    def __init__(self, mapping): self.mapping = mapping; self.calls = []
    def complete(self, prompt):
        self.calls.append(prompt)
        for k, v in self.mapping.items():
            if k in prompt: return v
        return "UNMATCHED"

def test_null_cleaner_is_identity():
    assert NullCleaner().clean(["a", "b"], [], None).texts == ["a", "b"]

def test_llama_cleaner_applies_good_keeps_bad():
    client = FakeClient({"Borderlestern": "Borderless.", "runaway": "x " * 99})
    res = LlamaCleaner(client).clean(["Borderlestern", "runaway text here ok"], ["Borderless"], None)
    assert res.texts[0] == "Borderless."         # applied
    assert res.texts[1] == "runaway text here ok" # guard kept raw
    assert res.cleaned == 1 and res.kept_raw == 1

def test_llama_cleaner_passes_glossary_and_context():
    client = FakeClient({"seg-two": "cleaned two"})
    LlamaCleaner(client).clean(["seg one", "seg-two"], ["Borderless", "Georg"], None)
    p = client.calls[1]
    assert "Borderless" in p and "Georg" in p and "seg one" in p and "seg-two" in p
```

**Step 2 — FAIL. Step 3 — implement `cleanup.py`** with `accept_candidate`, `build_prompt`
(system instruction + glossary block + previous-segment context + current), `NullCleaner`,
`LlamaCleaner` (loops segments, calls client, applies `accept_candidate`, on reject/exception
keeps raw + increments `kept_raw`, emits `reporter.warn` on the first reject), and a
`CleanResult` dataclass. Full code inline in this task when executing.

**Step 4 — PASS. Step 5 — commit** `feat: LLM cleanup orchestration + anti-hallucination guards`.

---

### Task 4: `raw_text` on Utterance + transcript schema

**Files:**
- Modify: `src/meetscribe/types.py` (add `raw_text: str = ""` to `Utterance`)
- Modify: `src/meetscribe/output.py` (`write_transcript`: per-segment `raw_text`, top-level `cleaned`)
- Test: `tests/test_output.py`

TDD: extend the schema test to assert a segment carries `raw_text` and the doc has a top-level
`cleaned` boolean; assert `words` still present with timings. Keep back-compat: `raw_text`
defaults to `text` when unset. Commit `feat: transcript carries raw_text + cleaned flag`.

---

### Task 5: Wire cleaner into `process()` + Components

**Files:**
- Modify: `src/meetscribe/pipeline.py` (`Components` gains `cleaner`; `process` runs it last)
- Test: `tests/test_pipeline.py`

`Components` gains `cleaner=NullCleaner()` default. After
`utterances = coalesce_utterances(...)`, run the cleaner over `[u.text for u in utterances]`,
producing new utterances with cleaned `text` and original `raw_text`; wrap in a
`reporter.stage("cleanup (llm)")` + progress bar; report `cleaned N/M (K kept raw)`.

**Step 1 — failing tests:** a `FakeCleaner` that uppercases; assert default-on cleaning sets
`text` uppercased and `raw_text` original; assert `words` untouched; assert a `NullCleaner`
run leaves `text == raw_text`. **Steps 2–4** red→green. **Step 5 — commit**
`feat: run cleanup as the final pipeline stage`.

---

### Task 6: meta.json cleanup identity + format_version

**Files:**
- Modify: `src/meetscribe/output.py` (`build_meta`) and `src/meetscribe/pipeline.py` (`run`)
- Test: `tests/test_output.py`

`build_meta` gains an optional `cleanup_model` dict (name, sha256, temp, quant); when present,
`format_version` = 2; absent (or `--no-cleanup`) → no block, and `cleaned:false`. TDD both
branches. Commit `feat: record cleanup-model identity in meta.json`.

---

### Task 7: CLI — `--no-cleanup`, `clean <dir>`, glossary/name wiring

**Files:**
- Modify: `src/meetscribe/cli.py` (flags + `clean` subcommand)
- Modify: `src/meetscribe/pipeline.py` (`run` accepts `cleanup: bool`; add `clean_existing(dir)`)
- Modify: `src/meetscribe/record.py` (`run`: after count, prompt names, append to glossary)
- Test: `tests/test_cli.py`, `tests/test_pipeline.py`, `tests/test_record.py`

- `process`/`record` gain `--no-cleanup` (default cleanup on). Parser tests assert the flag maps
  to `cleanup=False`.
- New `clean <dir>` subcommand → `pipeline.clean_existing(dir)`: read `transcript.json`, run the
  cleaner over segment texts, write `<dir>-cleanup/` with cleaned `transcript.json` + copied
  `embeddings.npz` + updated `meta.json`; **original untouched** (assert in test via FakeCleaner
  through a `build_components` monkeypatch, like existing run tests).
- `record.run`: `names = ask_participant_names(count); glossary.append(default_path(), names)` —
  TTY-gated exactly as `ask_participants` already is.

Commit per sub-piece (`feat: --no-cleanup`, `feat: clean subcommand`, `feat: name prompt wiring`).

---

### Task 8: Real LlamaClient — free port, server lifecycle, HTTP

**Files:**
- Create: `src/meetscribe/llama.py`
- Test: `tests/test_llama.py`

- `find_free_port(rng=(20000,60000), tries=5, _bind=...)`: attempt a loopback bind; return the
  first free port; after `tries` raise `NoFreePort`. Test with an injected `_bind` that fails N
  times then succeeds, and one that always fails (→ raises after 5).
- `LlamaServer` context manager: spawn `llama-server --host 127.0.0.1 --port P -m <gguf>`; poll
  `/health` until ready or readiness timeout; **`__exit__` always terminates** (test with a
  mocked `Popen` — assert `terminate` called even when the body raises).
- `LlamaClient.complete(prompt)`: POST to `/completion` (temp 0, seed fixed, n_predict cap),
  return the text; per-request timeout. (Real HTTP exercised only by the e2e test.)

Commit `feat: managed llama-server client with robust port selection`.

---

### Task 9: `build_components` wires the real cleaner

**Files:**
- Modify: `src/meetscribe/pipeline.py` (`build_components`)
- Modify: `tests/test_pipeline.py`

`build_components(models_dir, num_speakers, cleanup=True)`: when `cleanup`, construct
`LlamaCleaner(LlamaClient(...))` pointed at `<models_dir>/llm/model.gguf` and the `llama-server`
on PATH; else `NullCleaner()`. On construction failure (binary/model missing) → `warn` +
`NullCleaner` (graceful). Test the fallback with a monkeypatched constructor. Commit
`feat: wire real LLM cleaner into build_components with graceful fallback`.

---

### Task 10: Nix — pin the GGUF + add llama.cpp

**Files:**
- Modify: `nix/models.nix` (fetch GGUF by sha256 → `$out/llm/model.gguf`)
- Modify: `flake.nix` / wrapper (add `pkgs.llama-cpp` to the runtime PATH)

Get the sha256 via `nix-prefetch-url` of the pinned Qwen2.5-7B-Instruct-Q4_K_M GGUF; add it to
`models.nix` alongside the existing assets; ensure `llama-server` is on the wrapper PATH. Run
`nix flake check`. **Heavy (~4.7 GB fetch + rebuild); validate build here but expect minutes.**
Commit `build: pin Qwen2.5-7B GGUF and add llama.cpp to the runtime`.

---

### Task 11: E2E cleanup test (opt-in)

**Files:**
- Modify/Create: `tests/test_e2e.py`

Skipped unless `MEETSCRIBE_MODELS` (with the GGUF) is present, like the existing e2e. Build a
2-segment transcript with a deliberately garbled glossary term, run the real `clean_existing`
path, assert the term is corrected and per-word timings are byte-identical to the input.
Commit `test: e2e cleanup corrects a garbled glossary term, preserves timings`.

---

## Notes for the executor
- DRY/YAGNI/TDD, commit per task. Run `nix develop -c pytest -q` after each task; the whole
  suite must stay green and <1 s for Tasks 1–9 (no model touched).
- Keep the pipeline core UI-free: all messages via the injected `reporter` (`warn`/`stage`).
- Never touch `words`/timings — clean only `text`. `raw_text` preserves the original.
- Consistency: pin the GGUF by sha256 exactly like `nix/models.nix` does today; the model
  identity MUST travel into `meta.json` (same rule as embeddings).
