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


def test_record_help_documents_meetings_dir_default(capsys):
    # Without -o, recordings land under the configured meetings dir — the help
    # must not keep advertising the removed CWD-littering default.
    with pytest.raises(SystemExit):
        build_parser().parse_args(["record", "--help"])
    out = capsys.readouterr().out
    assert "meetings" in out
    assert "./meetscribe-" not in out


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


def test_record_parser_bundle_defaults_to_none():
    # None (not True) so an unset flag is distinguishable from an explicit one —
    # the resolved default (flag > env > config > default) lives in config.py.
    args = build_parser().parse_args(["record"])
    assert args.bundle is None


def test_process_parser_bundle_defaults_to_none():
    args = build_parser().parse_args(["process", "x.wav"])
    assert args.bundle is None


def test_record_parser_accepts_no_bundle():
    args = build_parser().parse_args(["record", "--no-bundle"])
    assert args.bundle is False


def test_process_parser_accepts_no_bundle():
    args = build_parser().parse_args(["process", "x.wav", "--no-bundle"])
    assert args.bundle is False


def test_main_record_forwards_bundle_flag(monkeypatch):
    # Guards the CLI->record.run wiring: explicit `--bundle`/`--no-bundle` must
    # reach record.run as True/False; without the flag, main forwards None and
    # record.run resolves the default from config (contract with pipeline/record).
    from meetscribe.cli import main

    captured = {}
    import meetscribe.record as rec
    monkeypatch.setattr(rec, "run", lambda **k: captured.update(k) or 0)

    assert main(["record", "--bundle"]) == 0
    assert captured["bundle"] is True

    assert main(["record", "--no-bundle"]) == 0
    assert captured["bundle"] is False

    assert main(["record"]) == 0
    assert captured["bundle"] is None


def test_main_bare_invocation_forwards_bundle_none(monkeypatch):
    # Bare `meetscribe` skips the subparser entirely, so record.run must get the
    # bundle default via the getattr fallback — None, i.e. "resolve from config".
    from meetscribe.cli import main

    captured = {}
    import meetscribe.record as rec
    monkeypatch.setattr(rec, "run", lambda **k: captured.update(k) or 0)

    assert main(["--quiet"]) == 0
    assert captured["bundle"] is None
    assert captured["cleanup"] is None


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


def test_bundle_command_writes_default_named_archive(isolated_config, tmp_path, monkeypatch):
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


def test_bundle_command_honours_explicit_out(isolated_config, tmp_path):
    from meetscribe.cli import main

    src = tmp_path / "meeting"
    src.mkdir()
    _make_artifacts(src)
    out = tmp_path / "custom.mscribe"

    assert main(["bundle", str(src), "-o", str(out)]) == 0
    assert out.exists()


def test_bundle_command_malformed_config_exits_2(isolated_config, tmp_path, capsys):
    # Acceptance: EVERY subcommand exits 2 on a malformed config (except
    # `config path`) — bundle included.
    from meetscribe.cli import main

    src = tmp_path / "meeting"
    src.mkdir()
    _make_artifacts(src)
    isolated_config.parent.mkdir(parents=True)
    isolated_config.write_text("[output\nbundle = false\n")

    assert main(["bundle", str(src)]) == 2
    printed = capsys.readouterr().out
    assert "config.toml" in printed and "line 1" in printed
    assert not list(tmp_path.glob("*.mscribe"))  # nothing was written


# ---- process --speakers --------------------------------------------------------------

def test_process_parser_accepts_speakers():
    from meetscribe.cli import build_parser

    args = build_parser().parse_args(["process", "x", "--speakers", "4"])
    assert args.speakers == 4


def test_process_speakers_defaults_to_none():
    from meetscribe.cli import build_parser

    args = build_parser().parse_args(["process", "x"])
    assert args.speakers is None


def test_cleanup_parser_defaults_and_spellings():
    # BooleanOptionalAction: default None ("resolve from config"), and the
    # historical `--no-cleanup` spelling must keep working alongside `--cleanup`.
    for base in (["record"], ["process", "x"]):
        assert build_parser().parse_args(base).cleanup is None
        assert build_parser().parse_args(base + ["--cleanup"]).cleanup is True
        assert build_parser().parse_args(base + ["--no-cleanup"]).cleanup is False


def test_main_process_forwards_cleanup_flag(monkeypatch):
    from meetscribe import pipeline
    from meetscribe.cli import main

    captured = {}
    monkeypatch.setattr(pipeline, "run", lambda **k: captured.update(k) or 0)

    assert main(["process", "x"]) == 0
    assert captured["cleanup"] is None  # unset flag → pipeline resolves config/default
    assert captured["bundle"] is None

    assert main(["process", "x", "--no-cleanup"]) == 0
    assert captured["cleanup"] is False

    assert main(["process", "x", "--cleanup"]) == 0
    assert captured["cleanup"] is True


def test_main_clean_dispatches_to_clean_existing(monkeypatch):
    from meetscribe import pipeline
    from meetscribe.cli import main

    captured = {}
    monkeypatch.setattr(pipeline, "clean_existing", lambda **k: captured.update(k) or 0)

    assert main(["clean", "some/dir"]) == 0
    assert captured["audio_dir"] == "some/dir"


def test_main_process_forwards_speakers(monkeypatch):
    # `--speakers N` means people in the meeting including the user — the same
    # semantic as the post-recording prompt — so the diarizer gets N-1; without
    # the flag (or with just the user) the pipeline gets -1 (automatic clustering).
    from meetscribe import pipeline
    from meetscribe.cli import main

    captured = {}
    monkeypatch.setattr(pipeline, "run", lambda **k: captured.update(k) or 0)

    assert main(["process", "x", "--speakers", "4"]) == 0
    assert captured["num_speakers"] == 3

    assert main(["process", "x"]) == 0
    assert captured["num_speakers"] == -1

    assert main(["process", "x", "--speakers", "1"]) == 0
    assert captured["num_speakers"] == -1


# ---- --backend / --language ----------------------------------------------------------

def test_backend_and_language_flags_parse_on_process_and_record():
    for base in (["process", "x"], ["record"]):
        args = build_parser().parse_args(base + ["--backend", "deepgram", "--language", "en"])
        assert args.backend == "deepgram"
        assert args.language == "en"


def test_backend_and_language_default_to_none():
    # None (not "local"/"de") so pipeline.run can apply flag > env > default precedence:
    # a concrete parser default would shadow STT_BACKEND / STT_LANGUAGE.
    for base in (["process", "x"], ["record"]):
        args = build_parser().parse_args(base)
        assert args.backend is None
        assert args.language is None


def test_backend_flag_rejects_unknown_value():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["process", "x", "--backend", "whisper"])


def test_main_process_forwards_backend_and_language(monkeypatch):
    from meetscribe import pipeline
    from meetscribe.cli import main

    captured = {}
    monkeypatch.setattr(pipeline, "run", lambda **k: captured.update(k) or 0)

    assert main(["process", "x", "--backend", "deepgram", "--language", "en"]) == 0
    assert captured["backend"] == "deepgram"
    assert captured["language"] == "en"

    assert main(["process", "x"]) == 0
    assert captured["backend"] is None  # unset flag → pipeline resolves env/default
    assert captured["language"] is None


def test_main_record_forwards_backend_and_language(monkeypatch):
    from meetscribe.cli import main

    captured = {}
    import meetscribe.record as rec
    monkeypatch.setattr(rec, "run", lambda **k: captured.update(k) or 0)

    assert main(["record", "--backend", "deepgram", "--language", "multi"]) == 0
    assert captured["backend"] == "deepgram"
    assert captured["language"] == "multi"


def test_main_bare_invocation_backend_fallbacks_match_parser_defaults(monkeypatch):
    # Bare `meetscribe` never runs the subparser, so record.run gets backend/language via
    # the getattr fallbacks — they must equal the parser defaults (None → env/default).
    from meetscribe.cli import main

    captured = {}
    import meetscribe.record as rec
    monkeypatch.setattr(rec, "run", lambda **k: captured.update(k) or 0)

    assert main(["--quiet"]) == 0
    assert captured["backend"] is None
    assert captured["language"] is None


# ---- config subcommand ---------------------------------------------------------------


@pytest.fixture
def isolated_config(monkeypatch, tmp_path):
    """Point every config/data path at tmp_path and scrub the STT env vars, so no
    test ever reads the real home or inherits the developer's environment."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))
    cfg_file = tmp_path / "conf" / "config.toml"
    monkeypatch.setenv("MEETSCRIBE_CONFIG", str(cfg_file))
    for var in ("STT_BACKEND", "STT_LANGUAGE", "DEEPGRAM_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    return cfg_file


def test_config_parser_actions():
    assert build_parser().parse_args(["config"]).action is None
    assert build_parser().parse_args(["config", "init"]).action == "init"
    assert build_parser().parse_args(["config", "path"]).action == "path"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["config", "bogus"])


def test_config_path_prints_path_only(isolated_config, capsys):
    from meetscribe.cli import main

    assert main(["config", "path"]) == 0
    assert capsys.readouterr().out == f"{isolated_config}\n"


def test_config_path_works_with_malformed_config(isolated_config, capsys):
    # Explicit acceptance carve-out: `config path` is the escape hatch a user
    # needs to LOCATE the broken file — it must not exit 2 on it.
    from meetscribe.cli import main

    isolated_config.parent.mkdir(parents=True)
    isolated_config.write_text("[stt\nbackend = 'x'\n")

    assert main(["config", "path"]) == 0
    assert capsys.readouterr().out == f"{isolated_config}\n"


def test_config_init_writes_template_0600(isolated_config, capsys):
    import os

    from meetscribe.cli import main
    from meetscribe.config import TEMPLATE

    # Pin the umask: under umask 077 write_text alone already yields 0600, which
    # would let a deleted chmod call escape this test on hardened runners.
    old = os.umask(0o022)
    try:
        assert main(["config", "init"]) == 0
    finally:
        os.umask(old)
    assert isolated_config.read_text() == TEMPLATE  # parent dirs were created
    assert (isolated_config.stat().st_mode & 0o777) == 0o600
    assert str(isolated_config) in capsys.readouterr().out


def test_config_init_refuses_overwrite(isolated_config):
    from meetscribe.cli import main

    isolated_config.parent.mkdir(parents=True)
    isolated_config.write_text("[stt]\nbackend = \"deepgram\"\n")

    assert main(["config", "init"]) != 0
    # The existing file is untouched.
    assert isolated_config.read_text() == "[stt]\nbackend = \"deepgram\"\n"


def test_config_show_defaults_when_missing(isolated_config, capsys):
    from meetscribe.cli import main

    assert main(["config"]) == 0
    out = capsys.readouterr().out
    assert str(isolated_config) in out and "missing" in out
    assert "backend" in out and "local" in out
    assert "cleanup" in out and "true" in out
    assert "(default)" in out


def test_config_show_reports_config_and_env_origins(isolated_config, monkeypatch, capsys):
    from meetscribe.cli import main

    isolated_config.parent.mkdir(parents=True)
    isolated_config.write_text("[stt]\nbackend = \"deepgram\"\n[output]\nbundle = false\n")

    assert main(["config"]) == 0
    out = capsys.readouterr().out
    assert "exists" in out
    backend_line = next(l for l in out.splitlines() if l.startswith("backend"))
    assert "deepgram" in backend_line and "(config)" in backend_line
    bundle_line = next(l for l in out.splitlines() if l.startswith("bundle"))
    assert "false" in bundle_line and "(config)" in bundle_line

    monkeypatch.setenv("STT_BACKEND", "local")
    assert main(["config"]) == 0
    backend_line = next(
        l for l in capsys.readouterr().out.splitlines() if l.startswith("backend")
    )
    assert "local" in backend_line and "(env)" in backend_line


def test_config_show_masks_api_key(isolated_config, capsys):
    from meetscribe.cli import main

    isolated_config.parent.mkdir(parents=True)
    isolated_config.write_text('[deepgram]\napi_key = "dg_supersecret123"\n')

    assert main(["config"]) == 0
    out = capsys.readouterr().out
    assert "supersecret" not in out  # the secret itself never hits stdout
    assert "secret123" not in out
    key_line = next(l for l in out.splitlines() if l.startswith("api_key"))
    # the exact mask: 3-char prefix only — anything longer leaks the secret
    assert "dg_…****" in key_line and "(config)" in key_line


def test_config_show_malformed_exits_2(isolated_config, capsys):
    from meetscribe.cli import main

    isolated_config.parent.mkdir(parents=True)
    isolated_config.write_text("[stt\nbackend = 'x'\n")  # unclosed section header

    assert main(["config"]) == 2
    err = capsys.readouterr().err
    assert str(isolated_config) in err


def test_config_show_wrongly_typed_value_exits_2(isolated_config, capsys):
    from meetscribe.cli import main

    isolated_config.parent.mkdir(parents=True)
    isolated_config.write_text('[output]\nbundle = "false"\n')  # string, not TOML bool

    assert main(["config"]) == 2
    assert "bundle" in capsys.readouterr().err


def test_config_show_api_key_cmd_shown_without_executing(isolated_config, capsys, tmp_path):
    # `meetscribe config` must indicate the lazy source but NEVER run the command.
    from meetscribe.cli import main

    canary = tmp_path / "canary"
    isolated_config.parent.mkdir(parents=True)
    isolated_config.write_text(f'[deepgram]\napi_key_cmd = "touch {canary}"\n')

    assert main(["config"]) == 0
    out = capsys.readouterr().out
    key_line = next(l for l in out.splitlines() if l.startswith("api_key"))
    assert "via api_key_cmd" in key_line and "(config)" in key_line
    assert str(canary) not in out  # don't echo the command either
    assert not canary.exists()
