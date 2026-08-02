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
