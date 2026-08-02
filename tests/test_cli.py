"""CLI argument-parsing tests (pure, no side effects)."""

import pytest

from meetscribe.cli import build_parser


def test_bare_invocation_has_no_command():
    # Bare invocation parses cleanly; main() defaults it to "record".
    args = build_parser().parse_args([])
    assert args.command is None


def test_process_takes_optional_audio_path():
    args = build_parser().parse_args(["process", "foo.wav"])
    assert args.command == "process"
    assert args.audio == "foo.wav"


def test_process_audio_is_optional():
    args = build_parser().parse_args(["process"])
    assert args.command == "process"
    assert args.audio is None


def test_doctor_command():
    args = build_parser().parse_args(["doctor"])
    assert args.command == "doctor"


def test_record_out_flag():
    args = build_parser().parse_args(["record", "-o", "/tmp/out"])
    assert args.command == "record"
    assert args.out == "/tmp/out"


def test_quiet_and_verbose_flags():
    args = build_parser().parse_args(["--quiet", "process", "x.wav"])
    assert args.quiet is True and args.verbose is False
    args = build_parser().parse_args(["-v", "doctor"])
    assert args.verbose is True and args.quiet is False


def test_version_exits_zero():
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--version"])
    assert exc.value.code == 0


def test_process_parser_accepts_bundle():
    args = build_parser().parse_args(["process", "x.wav", "--bundle"])
    assert args.bundle is True


def test_record_parser_accepts_bundle():
    args = build_parser().parse_args(["record", "--bundle"])
    assert args.bundle is True


def test_main_record_forwards_bundle_flag(monkeypatch):
    # Guards the CLI->record.run wiring: `record --bundle` must actually reach
    # record.run(bundle=True), not just parse into args and get dropped.
    from meetscribe.cli import main

    captured = {}
    import meetscribe.record as rec
    monkeypatch.setattr(rec, "run", lambda **k: captured.update(k) or 0)

    assert main(["record", "--bundle"]) == 0
    assert captured["bundle"] is True


def test_bundle_parser_accepts_dir_and_out():
    args = build_parser().parse_args(["bundle", "some/dir", "-o", "x.mscribe"])
    assert args.command == "bundle"
    assert args.dir == "some/dir"
    assert args.out == "x.mscribe"


def test_bundle_parser_out_defaults_to_none():
    args = build_parser().parse_args(["bundle", "some/dir"])
    assert args.out is None


def _make_artifacts(d):
    """Populate a directory with the three real bundle members (mirrors test_bundle)."""
    import numpy as np

    from meetscribe.output import build_meta, write_embeddings, write_meta, write_transcript
    from meetscribe.types import Utterance, Word

    dim = 192
    models = {
        "embedding_model": "cam++",
        "embedding_model_sha256": "abc123",
        "asr_model": "parakeet-tdt-0.6b-v3",
        "segmentation_model": "pyannote-segmentation-3.0",
    }
    write_transcript(
        d / "transcript.json",
        meeting_id="m1",
        duration_s=1.0,
        utterances=[Utterance(0.0, 1.0, "me", "mic", "hi", (Word("hi", 0.0, 1.0),))],
    )
    write_embeddings(
        d / "embeddings.npz",
        [("turn_0", np.ones(dim, dtype=np.float32), "me")],
        [],
        dim=dim,
    )
    write_meta(
        d / "meta.json",
        build_meta(
            models, embedding_dim=dim, meeting_id="m1",
            started_at="2026-08-02T14:03:11+02:00",
            ended_at="2026-08-02T14:03:12+02:00", duration_s=1.0,
        ),
    )


def test_bundle_command_writes_default_named_archive(tmp_path, monkeypatch):
    import zipfile

    from meetscribe.cli import main

    src = tmp_path / "meeting"
    src.mkdir()
    _make_artifacts(src)

    monkeypatch.chdir(tmp_path)  # default output lands in the current directory
    assert main(["bundle", str(src)]) == 0

    out = tmp_path / "meeting-m1.mscribe"
    assert out.exists()
    with zipfile.ZipFile(out) as z:
        assert sorted(z.namelist()) == ["embeddings.npz", "meta.json", "transcript.json"]


def test_bundle_command_honours_explicit_out(tmp_path):
    from meetscribe.cli import main

    src = tmp_path / "meeting"
    src.mkdir()
    _make_artifacts(src)
    out = tmp_path / "custom.mscribe"

    assert main(["bundle", str(src), "-o", str(out)]) == 0
    assert out.exists()
