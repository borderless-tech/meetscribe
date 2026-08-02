# `.mscribe` Upload Bundle Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Emit a single `.mscribe` zip (transcript + embeddings + enriched meta) so an authenticated, stateless HTTP sink can ingest one atomic upload.

**Architecture:** Keep the three existing `output.py` writers untouched. Enrich `meta.json` with `meeting_id`, wall-clock `started_at`/`ended_at`, mirrored `duration_s`, and `format_version`. Add a standalone `bundle_dir(src_dir, out_path)` that zips the three fixed-name members from a directory (single validation point). The pipeline always writes the directory; `--bundle` additively produces the zip; a `meetscribe bundle <dir>` subcommand reuses `bundle_dir`.

**Tech Stack:** Python 3, `zipfile` (stdlib), `numpy` (existing), `pytest`. Design: `docs/plans/2026-08-02-bundle-format-design.md`.

**Conventions in this repo:** tests run with `uv run pytest` (or `pytest` if the env is active). Times are tz-aware ISO 8601 **with offset** — use `datetime.now().astimezone()` / `datetime.fromtimestamp(ts).astimezone()`, never bare UTC epochs. Follow @superpowers:test-driven-development. Commit after each task.

---

## Task 1: Enrich `build_meta` (new fields + `FORMAT_VERSION`)

**Files:**
- Modify: `src/meetscribe/output.py:24-34` (`build_meta`), top of file for the constant
- Test: `tests/test_output.py:21-42`

**Step 1: Write the failing test**

Add to `tests/test_output.py` (and extend the imports on line 8 to include `FORMAT_VERSION`):

```python
from meetscribe.output import FORMAT_VERSION  # add to existing import line


def test_build_meta_includes_bundle_fields():
    meta = build_meta(
        MODELS,
        embedding_dim=DIM,
        meeting_id="abc123",
        started_at="2026-08-02T14:03:11+02:00",
        ended_at="2026-08-02T14:47:52+02:00",
        duration_s=2681.0,
    )
    assert meta["meeting_id"] == "abc123"
    assert meta["started_at"] == "2026-08-02T14:03:11+02:00"
    assert meta["ended_at"] == "2026-08-02T14:47:52+02:00"
    assert meta["duration_s"] == 2681.0
    assert meta["format_version"] == FORMAT_VERSION
```

Also update the existing `test_build_meta_has_all_required_keys` and `test_write_meta_round_trips` calls to pass the new required keyword args (use the same literals as above), so they still construct a valid meta.

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_output.py::test_build_meta_includes_bundle_fields -v`
Expected: FAIL — `TypeError: build_meta() missing ... keyword arguments` / `ImportError: cannot import name 'FORMAT_VERSION'`.

**Step 3: Write minimal implementation**

At the top of `src/meetscribe/output.py`, below the imports:

```python
FORMAT_VERSION = 1
```

Replace `build_meta` with:

```python
def build_meta(
    models: dict,
    embedding_dim: int,
    *,
    meeting_id: str,
    started_at: str,
    ended_at: str,
    duration_s: float,
    sample_rate: int = 16000,
) -> dict:
    """Assemble meta.json. ``started_at``/``ended_at`` are tz-aware ISO 8601 strings
    (with offset) — the calendar-reconciliation match window."""
    return {
        "embedding_model": models["embedding_model"],
        "embedding_model_sha256": models["embedding_model_sha256"],
        "embedding_dim": embedding_dim,
        "asr_model": models["asr_model"],
        "segmentation_model": models["segmentation_model"],
        "sample_rate": sample_rate,
        "meetscribe_version": __version__,
        "meeting_id": meeting_id,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_s": duration_s,
        "format_version": FORMAT_VERSION,
    }
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_output.py -v`
Expected: PASS (all, including the two updated tests).

**Step 5: Commit**

```bash
git add src/meetscribe/output.py tests/test_output.py
git commit -m "feat: enrich meta.json with meeting_id, timestamps, format_version"
```

---

## Task 2: `bundle_dir` — zip the three members from a directory

**Files:**
- Modify: `src/meetscribe/output.py` (append; add `import zipfile` at top)
- Test: `tests/test_bundle.py` (create)

**Step 1: Write the failing test**

Create `tests/test_bundle.py`:

```python
"""The .mscribe upload bundle: zip the three artifacts from a directory (design §Producer)."""

import json
import zipfile

import numpy as np
import pytest

from meetscribe.output import (
    BUNDLE_MEMBERS,
    bundle_dir,
    build_meta,
    write_embeddings,
    write_meta,
    write_transcript,
)
from meetscribe.types import Utterance, Word

DIM = 192
MODELS = {
    "embedding_model": "cam++",
    "embedding_model_sha256": "abc123",
    "asr_model": "parakeet-tdt-0.6b-v3",
    "segmentation_model": "pyannote-segmentation-3.0",
}


def _make_artifacts(d):
    write_transcript(
        d / "transcript.json",
        meeting_id="m1",
        duration_s=1.0,
        utterances=[Utterance(0.0, 1.0, "me", "mic", "hi", (Word("hi", 0.0, 1.0),))],
    )
    write_embeddings(
        d / "embeddings.npz",
        [("turn_0", np.ones(DIM, dtype=np.float32), "me")],
        [],
        dim=DIM,
    )
    write_meta(
        d / "meta.json",
        build_meta(
            MODELS, embedding_dim=DIM, meeting_id="m1",
            started_at="2026-08-02T14:03:11+02:00",
            ended_at="2026-08-02T14:03:12+02:00", duration_s=1.0,
        ),
    )


def test_bundle_dir_contains_the_three_members_byte_identical(tmp_path):
    src = tmp_path / "meeting"
    src.mkdir()
    _make_artifacts(src)
    out = tmp_path / "meeting-m1.mscribe"

    bundle_dir(src, out)

    assert out.exists()
    with zipfile.ZipFile(out) as z:
        assert sorted(z.namelist()) == sorted(BUNDLE_MEMBERS)
        for name in BUNDLE_MEMBERS:
            assert z.read(name) == (src / name).read_bytes()  # no re-encoding
    # embeddings survive the round-trip through the zip
    with zipfile.ZipFile(out) as z, z.open("embeddings.npz") as f:
        assert np.load(f)["turn_vectors"].shape == (1, DIM)


def test_bundle_dir_rejects_missing_member(tmp_path):
    src = tmp_path / "meeting"
    src.mkdir()
    _make_artifacts(src)
    (src / "embeddings.npz").unlink()
    with pytest.raises(FileNotFoundError, match="embeddings.npz"):
        bundle_dir(src, tmp_path / "x.mscribe")


def test_bundle_dir_rejects_meta_without_format_version(tmp_path):
    src = tmp_path / "meeting"
    src.mkdir()
    _make_artifacts(src)
    (src / "meta.json").write_text(json.dumps({"meeting_id": "m1"}))  # no format_version
    with pytest.raises(ValueError, match="format_version"):
        bundle_dir(src, tmp_path / "x.mscribe")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_bundle.py -v`
Expected: FAIL — `ImportError: cannot import name 'bundle_dir'`.

**Step 3: Write minimal implementation**

Add `import zipfile` to the top of `src/meetscribe/output.py`, then append:

```python
# The .mscribe bundle: a plain zip of these three fixed-name members (design doc).
BUNDLE_MEMBERS = ("transcript.json", "embeddings.npz", "meta.json")


def bundle_dir(src_dir: str | Path, out_path: str | Path) -> Path:
    """Zip the three artifacts from ``src_dir`` into a single ``.mscribe`` file.

    Single validation point: every member must exist and ``meta.json`` must carry a
    ``format_version`` before we write anything.
    """
    src = Path(src_dir)
    for name in BUNDLE_MEMBERS:
        if not (src / name).exists():
            raise FileNotFoundError(f"cannot bundle: missing {name} in {src}")
    meta = json.loads((src / "meta.json").read_text())
    if "format_version" not in meta:
        raise ValueError(f"cannot bundle: meta.json in {src} lacks format_version")
    out = Path(out_path)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name in BUNDLE_MEMBERS:
            z.write(src / name, arcname=name)
    return out
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_bundle.py -v`
Expected: PASS (3 tests).

**Step 5: Commit**

```bash
git add src/meetscribe/output.py tests/test_bundle.py
git commit -m "feat: bundle_dir zips the three artifacts into one .mscribe"
```

---

## Task 3: Thread timestamps + `format_version` through `pipeline.run`

**Files:**
- Modify: `src/meetscribe/pipeline.py:188-231` (`run`)
- Test: `tests/test_pipeline.py`

**Context:** `run` builds `meta` and writes the dir. It must now compute `started_at`/`ended_at` and pass a `bundle` flag through. When a caller (record flow, Task 5) supplies `started_at`, use it; otherwise derive it from the input WAV's mtime. `ended_at = started_at + duration_s`. Add `default_bundle_name` for reuse.

**Step 1: Write the failing test**

Add to `tests/test_pipeline.py` (adapt imports/fakes to the file's existing `process`/`run` test style; if `run` is not currently exercised there, drive `build_meta` args via a small unit test of the new helper instead):

```python
from datetime import datetime, timezone

from meetscribe.output import default_bundle_name


def test_default_bundle_name_uses_meeting_id():
    assert default_bundle_name({"meeting_id": "abc123"}) == "meeting-abc123.mscribe"


def test_derive_window_from_started_at():
    from meetscribe.pipeline import _window
    start = datetime(2026, 8, 2, 14, 0, 0, tzinfo=timezone.utc)
    started, ended = _window(started_at=start, duration_s=60.0, mic_wav=None, system_wav=None)
    assert started == "2026-08-02T14:00:00+00:00"
    assert ended == "2026-08-02T14:01:00+00:00"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline.py -k "bundle_name or window" -v`
Expected: FAIL — `ImportError` / `AttributeError: _window`.

**Step 3: Write minimal implementation**

In `src/meetscribe/output.py` add:

```python
def default_bundle_name(meta: dict) -> str:
    return f"meeting-{meta['meeting_id']}.mscribe"
```

In `src/meetscribe/pipeline.py`, add a helper above `run`:

```python
def _window(started_at, duration_s, mic_wav, system_wav):
    """Return (started_at, ended_at) as tz-aware ISO 8601 strings (with offset).

    ``started_at`` (a tz-aware datetime) wins; otherwise fall back to the input WAV's
    mtime as a best-effort capture time for the ``process`` path.
    """
    from datetime import datetime, timedelta

    if started_at is None:
        src = mic_wav or system_wav
        ts = Path(src).stat().st_mtime if src else 0.0
        started_at = datetime.fromtimestamp(ts).astimezone()
    ended_at = started_at + timedelta(seconds=duration_s)
    return started_at.isoformat(), ended_at.isoformat()
```

Change `run`'s signature to:

```python
def run(audio=None, out_dir=None, bundle=False, started_at=None, reporter=None) -> int:
```

Replace the `build_meta(...)` call (lines ~217-225) with one that passes the new args:

```python
    started_iso, ended_iso = _window(started_at, result.duration_s, mic_wav, system_wav)
    meta = build_meta(
        {
            "embedding_model": "3dspeaker_campplus_sv_zh_en_16k",
            "embedding_model_sha256": _sha256(spk_model),
            "asr_model": "parakeet-tdt-0.6b-v3",
            "segmentation_model": "pyannote-segmentation-3.0",
        },
        embedding_dim=result.dim,
        meeting_id=meeting_id,
        started_at=started_iso,
        ended_at=ended_iso,
        duration_s=result.duration_s,
    )
```

Leave the three `write_*` calls as-is (Task 4 adds bundling after them).

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pipeline.py tests/test_output.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/meetscribe/pipeline.py src/meetscribe/output.py tests/test_pipeline.py
git commit -m "feat: compute started_at/ended_at window and pass into build_meta"
```

---

## Task 4: `--bundle` produces the `.mscribe` after writing the dir

**Files:**
- Modify: `src/meetscribe/pipeline.py` (`run`, after the `write_*` calls)
- Modify: `src/meetscribe/cli.py:46-52` (process/record parsers), `:74-85` (dispatch)
- Test: `tests/test_pipeline.py`, `tests/test_cli.py`

**Step 1: Write the failing test**

In `tests/test_pipeline.py`, add an end-to-end-ish test using the existing fake `Components` pattern in that file (mirror how the file already calls `process`/`run`; if `run` needs `MEETSCRIBE_MODELS`, prefer testing the bundling branch by calling `bundle_dir` from a populated dir instead). Minimum viable assertion:

```python
def test_run_with_bundle_writes_mscribe(tmp_path, monkeypatch):
    # Arrange a dir with the three artifacts (reuse the Task 2 helper or fakes),
    # then assert that after run(..., bundle=True) a meeting-<id>.mscribe exists
    # in the out dir alongside the loose files.
    ...
    assert (out / "meeting-<id>.mscribe").exists()
    assert (out / "transcript.json").exists()  # dir is still produced
```

In `tests/test_cli.py`, assert the parser accepts `--bundle` on both subcommands:

```python
def test_process_parser_accepts_bundle():
    args = build_parser().parse_args(["process", "x.wav", "--bundle"])
    assert args.bundle is True


def test_record_parser_accepts_bundle():
    args = build_parser().parse_args(["record", "--bundle"])
    assert args.bundle is True
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -k bundle tests/test_pipeline.py -k bundle -v`
Expected: FAIL — unrecognized argument `--bundle` / no `.mscribe` written.

**Step 3: Write minimal implementation**

In `pipeline.py` `run`, after the three `write_*` calls and before `reporter.summary(...)`:

```python
    from .output import bundle_dir, default_bundle_name

    if bundle:
        bundle_path = out / default_bundle_name(meta)
        bundle_dir(out, bundle_path)
        print(f"wrote {bundle_path.name}")
```

In `cli.py`, add to **both** `p_process` and `p_record` parsers:

```python
    p_process.add_argument(
        "--bundle", action="store_true",
        help="also emit a single meeting-<id>.mscribe upload bundle",
    )
    p_record.add_argument(
        "--bundle", action="store_true",
        help="also emit a single meeting-<id>.mscribe upload bundle",
    )
```

Pass it through in the dispatch:

```python
    if command == "record":
        return record.run(
            out_dir=getattr(args, "out", None),
            bundle=getattr(args, "bundle", False),
            reporter=reporter,
        )
    if command == "process":
        return pipeline.run(
            audio=getattr(args, "audio", None),
            out_dir=getattr(args, "out", None),
            bundle=getattr(args, "bundle", False),
            reporter=reporter,
        )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py tests/test_pipeline.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/meetscribe/pipeline.py src/meetscribe/cli.py tests/test_cli.py tests/test_pipeline.py
git commit -m "feat: --bundle additively emits meeting-<id>.mscribe"
```

---

## Task 5: Thread `started_at` from the record flow

**Files:**
- Modify: `src/meetscribe/record.py:311-330` (`run`)
- Test: `tests/test_record.py`

**Context:** Recording knows the real wall-clock start. Capture it just before `record_tracks` and pass it (plus the `bundle` flag) into `pipeline.run` so the meta window reflects the actual meeting, not the WAV mtime.

**Step 1: Write the failing test**

Add to `tests/test_record.py` (mirror the file's existing monkeypatch style — stub `record_tracks`, `warn_if_silent`, and `pipeline.run` to capture kwargs):

```python
def test_run_passes_started_at_and_bundle_to_pipeline(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr("meetscribe.record.record_tracks", lambda *a, **k: None)
    monkeypatch.setattr("meetscribe.record.warn_if_silent", lambda p: None)
    import meetscribe.pipeline as pl
    monkeypatch.setattr(pl, "run", lambda **k: captured.update(k) or 0)

    from meetscribe import record
    record.run(out_dir=str(tmp_path / "m"), bundle=True)

    assert captured["bundle"] is True
    assert captured["started_at"] is not None  # a tz-aware datetime
    assert captured["started_at"].tzinfo is not None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_record.py -k started_at -v`
Expected: FAIL — `run()` has no `bundle` param / `started_at` not passed.

**Step 3: Write minimal implementation**

Change `record.run` to:

```python
def run(out_dir: str | None = None, bundle: bool = False, reporter=None) -> int:
    from datetime import datetime, timezone

    root = Path(out_dir or f"meetscribe-{datetime.now(timezone.utc):%Y-%m-%dT%H-%M-%S}")
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    mic_out = str(raw / "mic.wav")
    system_out = str(raw / "system.wav")

    print(f"Recording to {root} — press Ctrl-C to stop.")
    started_at = datetime.now().astimezone()  # tz-aware wall-clock, real meeting start
    record_tracks(mic_out, system_out, reporter=reporter)

    for track in (mic_out, system_out):
        warning = warn_if_silent(track)
        if warning:
            print(f"⚠ {warning}")

    from . import pipeline

    return pipeline.run(
        audio=str(root), out_dir=str(root),
        bundle=bundle, started_at=started_at, reporter=reporter,
    )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_record.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/meetscribe/record.py tests/test_record.py
git commit -m "feat: record passes real started_at + bundle flag into the pipeline"
```

---

## Task 6: `meetscribe bundle <dir>` standalone subcommand

**Files:**
- Modify: `src/meetscribe/cli.py` (new `bundle` subparser + dispatch)
- Test: `tests/test_cli.py`

**Context:** Reuse `bundle_dir`; `-o` defaults to `meeting-<id>.mscribe` (from the dir's `meta.json`) in the current directory. This is the TUI-friendly "bundle any directory" affordance.

**Step 1: Write the failing test**

Add to `tests/test_cli.py`:

```python
def test_bundle_parser_accepts_dir_and_out():
    args = build_parser().parse_args(["bundle", "some/dir", "-o", "x.mscribe"])
    assert args.dir == "some/dir"
    assert args.out == "x.mscribe"


def test_bundle_command_writes_default_named_archive(tmp_path, monkeypatch):
    # populate tmp_path with the three artifacts (reuse tests/test_bundle helper),
    # cd into tmp_path, run main(["bundle", str(src)]), assert meeting-<id>.mscribe exists.
    ...
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -k bundle -v`
Expected: FAIL — `invalid choice: 'bundle'`.

**Step 3: Write minimal implementation**

In `build_parser`, add:

```python
    p_bundle = sub.add_parser("bundle", help="zip an artifact directory into one .mscribe")
    p_bundle.add_argument("dir", help="directory holding transcript.json / embeddings.npz / meta.json")
    p_bundle.add_argument(
        "-o", "--out", default=None,
        help="output path (default: ./meeting-<id>.mscribe)",
    )
```

In `main`, add a dispatch branch:

```python
    if command == "bundle":
        import json
        from pathlib import Path

        from .output import bundle_dir, default_bundle_name

        src = Path(args.dir)
        meta = json.loads((src / "meta.json").read_text())
        out = args.out or default_bundle_name(meta)
        bundle_dir(src, out)
        print(f"wrote {out}")
        return 0
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/meetscribe/cli.py tests/test_cli.py
git commit -m "feat: meetscribe bundle <dir> subcommand reuses bundle_dir"
```

---

## Task 7: Full suite + docs touch-up

**Step 1: Run the whole suite**

Run: `uv run pytest -q`
Expected: PASS (all prior tests + the new ones; ~70 existing tests unchanged).

**Step 2: Update the design doc status**

In `docs/plans/2026-08-02-bundle-format-design.md`, change the status line from
`design approved, not yet implemented` to `implemented`.

Also update `src/meetscribe/output.py`'s module docstring (line 1) and `pipeline.py`'s
(line 1) if they enumerate "the three artifacts" — add a note that `--bundle` wraps them
into one `.mscribe`.

**Step 3: Commit**

```bash
git add -A
git commit -m "docs: mark .mscribe bundle format implemented"
```

---

## Notes / guardrails

- **DRY:** `bundle_dir` and `default_bundle_name` are the *only* places that know member names and the naming scheme. Do not re-implement zipping in the CLI or pipeline.
- **YAGNI:** no checksum member, no manifest file, no speaker-name hints — the design defers these until borderless-knowledge asks. `format_version = 1` is the only forward-compat lever.
- **Timezone:** every timestamp is `.astimezone()`-localized (carries an offset). A bare UTC epoch would misalign the calendar match window — see design §meta enrichment.
- **The loose directory is always produced** — `--bundle` is strictly additive. Never gate the `write_*` calls behind it.
