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
    for var in (
        "STT_BACKEND", "STT_LANGUAGE", "DEEPGRAM_API_KEY",
        "MEETSCRIBE_BK_URL", "MEETSCRIBE_BK_TOKEN",
    ):
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


def test_config_show_includes_bk_defaults(isolated_config, capsys):
    # The [bk] section is part of the show-every-effective-value promise: each key
    # appears with its origin even when nothing is configured (auto_upload is the
    # privacy-critical one — it must be visible as false/default).
    from meetscribe.cli import main

    assert main(["config"]) == 0
    lines = capsys.readouterr().out.splitlines()
    base_line = next(l for l in lines if l.startswith("bk_base_url"))
    assert "(not set)" in base_line and "(default)" in base_line
    token_line = next(l for l in lines if l.startswith("bk_token"))
    assert "(not set)" in token_line and "(default)" in token_line
    auto_line = next(l for l in lines if l.startswith("bk_auto_upload"))
    assert "false" in auto_line and "(default)" in auto_line


def test_config_show_bk_values_and_masked_token(isolated_config, capsys):
    from meetscribe.cli import main

    isolated_config.parent.mkdir(parents=True)
    isolated_config.write_text(
        '[bk]\nbase_url = "https://bk.example.com"\n'
        'token = "bk_supersecret123"\nauto_upload = true\n'
    )

    assert main(["config"]) == 0
    out = capsys.readouterr().out
    assert "supersecret" not in out  # the secret itself never hits stdout
    assert "secret123" not in out
    base_line = next(l for l in out.splitlines() if l.startswith("bk_base_url"))
    assert "https://bk.example.com" in base_line and "(config)" in base_line
    token_line = next(l for l in out.splitlines() if l.startswith("bk_token"))
    # the exact mask: 3-char prefix only, same rule as api_key
    assert "bk_…****" in token_line and "(config)" in token_line
    auto_line = next(l for l in out.splitlines() if l.startswith("bk_auto_upload"))
    assert "true" in auto_line and "(config)" in auto_line


def test_config_show_bk_env_origin_masks_token(isolated_config, monkeypatch, capsys):
    from meetscribe.cli import main

    monkeypatch.setenv("MEETSCRIBE_BK_URL", "https://env.example.com/")
    monkeypatch.setenv("MEETSCRIBE_BK_TOKEN", "env_supersecret")

    assert main(["config"]) == 0
    out = capsys.readouterr().out
    assert "supersecret" not in out
    base_line = next(l for l in out.splitlines() if l.startswith("bk_base_url"))
    # normalized (trailing slash stripped), env origin
    assert "https://env.example.com " in base_line and "(env)" in base_line
    token_line = next(l for l in out.splitlines() if l.startswith("bk_token"))
    assert "env…****" in token_line and "(env)" in token_line


def test_config_show_bk_token_cmd_shown_without_executing(
    isolated_config, capsys, tmp_path
):
    # Same rule as api_key_cmd: indicate the lazy source, NEVER run the command —
    # and never echo the command itself (it may embed paths/pinentry details).
    from meetscribe.cli import main

    canary = tmp_path / "canary"
    isolated_config.parent.mkdir(parents=True)
    isolated_config.write_text(
        f'[bk]\nbase_url = "https://bk.example.com"\ntoken_cmd = "touch {canary}"\n'
    )

    assert main(["config"]) == 0
    out = capsys.readouterr().out
    token_line = next(l for l in out.splitlines() if l.startswith("bk_token"))
    assert "(via token_cmd)" in token_line and "(config)" in token_line
    assert str(canary) not in out
    assert not canary.exists()


# ---- --upload / --no-upload ----------------------------------------------------------


def test_upload_flag_parses_tristate_on_record_and_process():
    # BooleanOptionalAction with default None: unset must stay distinguishable from an
    # explicit --upload/--no-upload — None means "resolve [bk].auto_upload from config".
    for base in (["record"], ["process", "x"]):
        assert build_parser().parse_args(base).upload is None
        assert build_parser().parse_args(base + ["--upload"]).upload is True
        assert build_parser().parse_args(base + ["--no-upload"]).upload is False


def test_main_record_forwards_upload_flag(monkeypatch):
    from meetscribe.cli import main

    captured = {}
    import meetscribe.record as rec
    monkeypatch.setattr(rec, "run", lambda **k: captured.update(k) or 0)

    assert main(["record", "--upload"]) == 0
    assert captured["upload"] is True

    assert main(["record", "--no-upload"]) == 0
    assert captured["upload"] is False

    assert main(["record"]) == 0
    assert captured["upload"] is None


def test_main_process_forwards_upload_flag(monkeypatch):
    from meetscribe import pipeline
    from meetscribe.cli import main

    captured = {}
    monkeypatch.setattr(pipeline, "run", lambda **k: captured.update(k) or 0)

    assert main(["process", "x", "--upload"]) == 0
    assert captured["upload"] is True

    assert main(["process", "x", "--no-upload"]) == 0
    assert captured["upload"] is False

    assert main(["process", "x"]) == 0
    assert captured["upload"] is None


def test_main_bare_invocation_forwards_upload_none(monkeypatch):
    # Bare `meetscribe` never runs the subparser: record.run must get upload via the
    # getattr fallback — None, i.e. "resolve from config" (contract with the wiring).
    from meetscribe.cli import main

    captured = {}
    import meetscribe.record as rec
    monkeypatch.setattr(rec, "run", lambda **k: captured.update(k) or 0)

    assert main(["--quiet"]) == 0
    assert captured["upload"] is None


# ---- upload subcommand ---------------------------------------------------------------


def _bk_fixture(name):
    import json
    from pathlib import Path

    return json.loads((Path(__file__).parent / "fixtures" / "bk" / name).read_text())


def _write_bk_config(cfg_file, body=None):
    cfg_file.parent.mkdir(parents=True, exist_ok=True)
    if body is None:
        body = '[bk]\nbase_url = "https://bk.example.com"\ntoken = "tok-secret"\n'
    cfg_file.write_text(body)


def _fake_bk_client(
    monkeypatch, *, caps=None, accepted=None, workflow=None,
    caps_error=None, upload_error=None, workflow_error=None,
):
    """Patch the module-level seam (bk.BkClient) with a scripted fake; returns the
    list of constructed instances so tests can inspect config + calls."""
    from meetscribe import bk

    created = []

    class Fake:
        def __init__(self, config, opener=None, sleep=None):
            self.config = config
            self.upload_calls = []
            self.workflow_calls = []
            created.append(self)

        def capabilities(self):
            if caps_error is not None:
                raise caps_error
            return caps if caps is not None else _bk_fixture("capabilities.json")

        def upload_bundle(self, bundle_path, *, meeting_id, calendar_meeting_id=None):
            self.upload_calls.append((str(bundle_path), meeting_id, calendar_meeting_id))
            if upload_error is not None:
                raise upload_error
            return accepted if accepted is not None else _bk_fixture("bundle-accepted.json")

        def workflow(self, workflow_id):
            self.workflow_calls.append(workflow_id)
            if workflow_error is not None:
                raise workflow_error
            return workflow if workflow is not None else _bk_fixture("workflow-processing.json")

    monkeypatch.setattr(bk, "BkClient", Fake)
    return created


def _meeting_dir(tmp_path, name="meeting"):
    d = tmp_path / name
    d.mkdir(parents=True)
    _make_artifacts(d)
    return d


def test_upload_parser_accepts_dir():
    args = build_parser().parse_args(["upload", "some/dir"])
    assert args.command == "upload"
    assert args.dir == "some/dir"


def test_upload_command_malformed_config_exits_2(isolated_config, tmp_path, capsys):
    from meetscribe.cli import main

    src = _meeting_dir(tmp_path)
    _write_bk_config(isolated_config, "[bk\nbase_url = 'x'\n")

    assert main(["upload", str(src)]) == 2
    assert "config.toml" in capsys.readouterr().out


def test_upload_command_unconfigured_bk_exits_2(isolated_config, tmp_path, capsys, monkeypatch):
    # No [bk].base_url anywhere: a config problem — exit 2 with the knobs named,
    # before any client is even constructed.
    from meetscribe.cli import main

    src = _meeting_dir(tmp_path)
    created = _fake_bk_client(monkeypatch)

    assert main(["upload", str(src)]) == 2
    out = capsys.readouterr().out
    assert "[bk].base_url" in out
    assert created == []


def test_upload_command_missing_token_exits_2(isolated_config, tmp_path, capsys, monkeypatch):
    from meetscribe.cli import main

    src = _meeting_dir(tmp_path)
    _write_bk_config(isolated_config, '[bk]\nbase_url = "https://bk.example.com"\n')
    created = _fake_bk_client(monkeypatch)

    assert main(["upload", str(src)]) == 2
    out = capsys.readouterr().out
    assert "[bk].token" in out
    assert created == []


def test_upload_command_broken_token_cmd_exits_2(isolated_config, tmp_path, capsys, monkeypatch):
    # token_cmd is executed eagerly here (the secret is actually needed); its failure
    # is exit-2 config material naming the knob.
    from meetscribe.cli import main

    src = _meeting_dir(tmp_path)
    _write_bk_config(
        isolated_config,
        '[bk]\nbase_url = "https://bk.example.com"\ntoken_cmd = "false"\n',
    )
    created = _fake_bk_client(monkeypatch)

    assert main(["upload", str(src)]) == 2
    assert "[bk].token_cmd" in capsys.readouterr().out
    assert created == []


def test_upload_command_happy_path_with_existing_bundle(
    isolated_config, tmp_path, capsys, monkeypatch
):
    from datetime import datetime

    from meetscribe import bk
    from meetscribe.cli import main
    from meetscribe.output import bundle_dir

    src = _meeting_dir(tmp_path)
    bundle_dir(src, src / "meeting-m1.mscribe")
    _write_bk_config(isolated_config)
    created = _fake_bk_client(monkeypatch)

    assert main(["upload", str(src)]) == 0

    out = capsys.readouterr().out
    # the fixture's own (opaque) id + web_url must surface on stdout
    assert "wf-3f9a1b2c-4d5e-4f60-8172-93a4b5c6d7e8" in out
    assert "https://bk.example.com/transcripts/wf-3f9a1b2c-4d5e-4f60-8172-93a4b5c6d7e8" in out

    (client,) = created
    assert client.config.base_url == "https://bk.example.com"
    assert client.config.token == "tok-secret"
    path, meeting_id, calendar_id = client.upload_calls[0]
    assert path.endswith("meeting-m1.mscribe")
    assert meeting_id == "m1"
    assert calendar_id is None
    # the existing bundle was reused, not re-created next to it
    assert [p.name for p in src.glob("*.mscribe")] == ["meeting-m1.mscribe"]

    ref = bk.read_workflow_ref(src)
    accepted = _bk_fixture("bundle-accepted.json")  # ids are opaque — compare to fixture
    assert ref["workflow_id"] == accepted["workflow_id"]
    assert ref["state_url"] == accepted["state_url"]
    assert ref["web_url"] == accepted["web_url"]
    assert ref["state"] == "processing"
    # timestamps are tz-aware ISO 8601 with offset (the meta.json convention)
    assert datetime.fromisoformat(ref["uploaded_at"]).tzinfo is not None
    assert datetime.fromisoformat(ref["checked_at"]).tzinfo is not None


def test_upload_command_bundles_first_when_no_mscribe(
    isolated_config, tmp_path, capsys, monkeypatch
):
    import zipfile

    from meetscribe.cli import main

    src = _meeting_dir(tmp_path)
    _write_bk_config(isolated_config)
    created = _fake_bk_client(monkeypatch)

    assert main(["upload", str(src)]) == 0

    out = src / "meeting-m1.mscribe"
    assert out.exists()
    with zipfile.ZipFile(out) as z:
        assert sorted(z.namelist()) == ["embeddings.npz", "meta.json", "transcript.json"]
    assert created[0].upload_calls[0][0] == str(out)


def test_upload_command_token_cmd_supplies_the_token(
    isolated_config, tmp_path, monkeypatch
):
    from meetscribe.cli import main

    src = _meeting_dir(tmp_path)
    _write_bk_config(
        isolated_config,
        '[bk]\nbase_url = "https://bk.example.com"\ntoken_cmd = "printf tok-from-cmd"\n',
    )
    created = _fake_bk_client(monkeypatch)

    assert main(["upload", str(src)]) == 0
    assert created[0].config.token == "tok-from-cmd"


def test_upload_command_preflight_failure_exits_1(
    isolated_config, tmp_path, capsys, monkeypatch
):
    # Unlike auto-upload (warn, exit 0), here upload IS the task: a failed preflight
    # is exit 1, the upload is never attempted, and no ref file appears.
    from meetscribe import bk
    from meetscribe.cli import main

    src = _meeting_dir(tmp_path)
    _write_bk_config(isolated_config)
    caps = {**_bk_fixture("capabilities.json"), "mscribe_format_versions": [1]}
    created = _fake_bk_client(monkeypatch, caps=caps)

    assert main(["upload", str(src)]) == 1
    assert "format_version" in capsys.readouterr().out
    assert created[0].upload_calls == []
    assert bk.read_workflow_ref(src) is None


def test_upload_command_upload_failure_exits_1(isolated_config, tmp_path, capsys, monkeypatch):
    from meetscribe import bk
    from meetscribe.cli import main

    src = _meeting_dir(tmp_path)
    _write_bk_config(isolated_config)
    _fake_bk_client(
        monkeypatch, upload_error=bk.BkError("bk request failed after 3 attempts (down)")
    )

    assert main(["upload", str(src)]) == 1
    assert "3 attempts" in capsys.readouterr().out
    assert bk.read_workflow_ref(src) is None


def test_upload_command_capabilities_failure_exits_1(
    isolated_config, tmp_path, capsys, monkeypatch
):
    from meetscribe import bk
    from meetscribe.cli import main

    src = _meeting_dir(tmp_path)
    _write_bk_config(isolated_config)
    created = _fake_bk_client(monkeypatch, caps_error=bk.BkError("network error: down"))

    assert main(["upload", str(src)]) == 1
    assert "down" in capsys.readouterr().out
    assert created[0].upload_calls == []


def test_upload_command_missing_meta_exits_1(isolated_config, tmp_path, capsys, monkeypatch):
    from meetscribe.cli import main

    src = tmp_path / "empty"
    src.mkdir()
    _write_bk_config(isolated_config)
    created = _fake_bk_client(monkeypatch)

    assert main(["upload", str(src)]) == 1
    assert "meta.json" in capsys.readouterr().out
    assert created == [] or created[0].upload_calls == []


# ---- status subcommand ---------------------------------------------------------------


def test_status_parser_dir_optional_and_offline():
    args = build_parser().parse_args(["status"])
    assert args.command == "status"
    assert args.dir is None
    assert args.offline is False

    args = build_parser().parse_args(["status", "some/dir", "--offline"])
    assert args.dir == "some/dir"
    assert args.offline is True


def test_status_command_malformed_config_exits_2(isolated_config, tmp_path, capsys):
    from meetscribe.cli import main

    _write_bk_config(isolated_config, "[bk\n")

    assert main(["status", str(tmp_path)]) == 2
    assert "config.toml" in capsys.readouterr().out


def test_status_empty_dir_exits_zero(isolated_config, tmp_path, capsys):
    from meetscribe.cli import main

    assert main(["status", str(tmp_path / "nothing")]) == 0
    assert "no meetings" in capsys.readouterr().out


def test_status_lists_not_uploaded_meeting(isolated_config, tmp_path, capsys):
    from meetscribe.cli import main

    root = tmp_path / "meetings"
    _meeting_dir(root, "a")

    assert main(["status", str(root)]) == 0
    out = capsys.readouterr().out
    line = next(l for l in out.splitlines() if "m1" in l)
    assert "2026-08-02T14:03:11+02:00" in line  # started_at
    assert "1s" in line  # duration_s
    assert "local" in line  # backend
    assert "bundled=no" in line
    assert "not uploaded" in line


def test_status_marks_bundled_meetings(isolated_config, tmp_path, capsys):
    from meetscribe.cli import main
    from meetscribe.output import bundle_dir

    root = tmp_path / "meetings"
    d = _meeting_dir(root, "a")
    bundle_dir(d, d / "meeting-m1.mscribe")

    assert main(["status", str(root)]) == 0
    line = next(l for l in capsys.readouterr().out.splitlines() if "m1" in l)
    assert "bundled=yes" in line


def test_status_default_dir_is_resolved_meetings_dir(isolated_config, tmp_path, capsys):
    # Without an argument, status scans the resolved meetings dir — here the
    # XDG_DATA_HOME default from the isolated fixture.
    from meetscribe.cli import main

    root = tmp_path / "xdg-data" / "meetscribe" / "meetings"
    _meeting_dir(root, "a")

    assert main(["status"]) == 0
    assert "m1" in capsys.readouterr().out


def test_status_terminal_ref_shows_state_without_network(
    isolated_config, tmp_path, capsys, monkeypatch
):
    # done/failed are terminal: no refresh even without --offline — the fake client
    # must never be constructed.
    from meetscribe import bk
    from meetscribe.cli import main

    root = tmp_path / "meetings"
    d = _meeting_dir(root, "a")
    bk.write_workflow_ref(d, {"workflow_id": "wf_01j9", "state": "done"})
    _write_bk_config(isolated_config)
    created = _fake_bk_client(monkeypatch)

    assert main(["status", str(root)]) == 0
    line = next(l for l in capsys.readouterr().out.splitlines() if "m1" in l)
    assert "done" in line
    assert "(stale)" not in line
    assert created == []


def test_status_failed_ref_is_terminal_no_network(
    isolated_config, tmp_path, capsys, monkeypatch
):
    # "failed" is terminal exactly like "done": no refresh, no client construction —
    # otherwise every `status` run would re-fetch and rewrite a settled ref. And per
    # contract rule 1 the bk page is where the user can inspect/act on the failure,
    # so the persisted web_url must be shown.
    from meetscribe import bk
    from meetscribe.cli import main

    root = tmp_path / "meetings"
    d = _meeting_dir(root, "a")
    bk.write_workflow_ref(
        d,
        {
            "workflow_id": "wf_01j9",
            "state": "failed",
            "web_url": "https://bk.example.com/workflows/wf_01j9",
        },
    )
    _write_bk_config(isolated_config)
    created = _fake_bk_client(monkeypatch)

    assert main(["status", str(root)]) == 0
    line = next(l for l in capsys.readouterr().out.splitlines() if "m1" in l)
    assert "failed" in line
    assert "(stale)" not in line
    assert "https://bk.example.com/workflows/wf_01j9" in line
    assert created == []


def test_status_offline_skips_refresh(isolated_config, tmp_path, capsys, monkeypatch):
    from meetscribe import bk
    from meetscribe.cli import main

    root = tmp_path / "meetings"
    d = _meeting_dir(root, "a")
    bk.write_workflow_ref(d, {"workflow_id": "wf_01j9", "state": "processing"})
    _write_bk_config(isolated_config)
    created = _fake_bk_client(monkeypatch)

    assert main(["status", str(root), "--offline"]) == 0
    line = next(l for l in capsys.readouterr().out.splitlines() if "m1" in l)
    assert "processing" in line
    assert "(stale)" not in line  # skipping was requested, not a degradation
    assert created == []


def test_status_refreshes_non_terminal_ref_and_updates_file(
    isolated_config, tmp_path, capsys, monkeypatch
):
    from meetscribe import bk
    from meetscribe.cli import main

    root = tmp_path / "meetings"
    d = _meeting_dir(root, "a")
    bk.write_workflow_ref(
        d,
        {
            "workflow_id": "wf_01j9",
            "state": "processing",
            "web_url": "https://bk.example.com/workflows/wf_01j9",
            "checked_at": "2026-09-09T10:00:00+02:00",
        },
    )
    _write_bk_config(isolated_config)
    created = _fake_bk_client(monkeypatch, workflow=_bk_fixture("workflow-done.json"))

    assert main(["status", str(root)]) == 0
    line = next(l for l in capsys.readouterr().out.splitlines() if "m1" in l)
    assert "done" in line

    assert created[0].workflow_calls == ["wf_01j9"]
    ref = bk.read_workflow_ref(d)
    assert ref["state"] == "done"
    assert ref["checked_at"] != "2026-09-09T10:00:00+02:00"  # refresh bumped it
    assert ref["workflow_id"] == "wf_01j9"


def test_status_refreshes_awaiting_review_ref_and_shows_web_url(
    isolated_config, tmp_path, capsys, monkeypatch
):
    # awaiting_review is NOT terminal — the gate can resolve or time out server-side,
    # so a persisted awaiting_review ref must refresh like "processing" (a non-refresh
    # would go silently stale forever). Task rendering is Phase 4: until then the
    # web_url is the ONLY way the user can complete the gate, so it must be printed.
    from meetscribe import bk
    from meetscribe.cli import main

    root = tmp_path / "meetings"
    d = _meeting_dir(root, "a")
    bk.write_workflow_ref(
        d,
        {
            "workflow_id": "wf_01j9",
            "state": "awaiting_review",
            "checked_at": "2026-09-09T10:00:00+02:00",
        },
    )
    _write_bk_config(isolated_config)
    created = _fake_bk_client(
        monkeypatch, workflow=_bk_fixture("workflow-awaiting-review-speaker-annotation.json")
    )

    assert main(["status", str(root)]) == 0
    line = next(l for l in capsys.readouterr().out.splitlines() if "m1" in l)
    assert "awaiting_review" in line
    assert "https://bk.example.com/transcripts/wf-33333333-3333-4333-8333-333333333333" in line

    assert created[0].workflow_calls == ["wf_01j9"]
    ref = bk.read_workflow_ref(d)
    assert ref["state"] == "awaiting_review"
    # the refresh adopted the workflow response's web_url (fixture value)
    wf = _bk_fixture("workflow-awaiting-review-speaker-annotation.json")
    assert ref["web_url"] == wf["web_url"]
    assert ref["checked_at"] != "2026-09-09T10:00:00+02:00"  # refresh bumped it


def test_status_unknown_state_falls_back_to_web_url_offline(
    isolated_config, tmp_path, capsys
):
    # Contract forward-compat rule 1: a client that encounters an unknown state
    # MUST fall back to showing web_url — bk may add workflow steps without
    # breaking older meetscribe versions.
    from meetscribe import bk
    from meetscribe.cli import main

    root = tmp_path / "meetings"
    d = _meeting_dir(root, "a")
    bk.write_workflow_ref(
        d,
        {
            "workflow_id": "wf_01j9",
            "state": "archived",  # a state this client has never heard of
            "web_url": "https://bk.example.com/workflows/wf_01j9",
        },
    )

    assert main(["status", str(root), "--offline"]) == 0
    line = next(l for l in capsys.readouterr().out.splitlines() if "m1" in l)
    assert "archived" in line
    assert "https://bk.example.com/workflows/wf_01j9" in line


def test_status_unknown_state_from_refresh_falls_back_to_web_url(
    isolated_config, tmp_path, capsys, monkeypatch
):
    # Same rule 1, online path: bk's refresh answer carries a new state — the
    # web_url from THAT response must be shown (and persisted in the ref).
    from meetscribe import bk
    from meetscribe.cli import main

    root = tmp_path / "meetings"
    d = _meeting_dir(root, "a")
    bk.write_workflow_ref(d, {"workflow_id": "wf_01j9", "state": "processing"})
    _write_bk_config(isolated_config)
    _fake_bk_client(
        monkeypatch,
        workflow={
            "workflow_id": "wf_01j9",
            "contract_version": 1,
            "state": "archived",
            "web_url": "https://bk.example.com/workflows/wf_01j9",
            "error": None,
            "tasks": [],
        },
    )

    assert main(["status", str(root)]) == 0
    line = next(l for l in capsys.readouterr().out.splitlines() if "m1" in l)
    assert "archived" in line
    assert "https://bk.example.com/workflows/wf_01j9" in line
    ref = bk.read_workflow_ref(d)
    assert ref["state"] == "archived"
    assert ref["web_url"] == "https://bk.example.com/workflows/wf_01j9"


def test_status_refresh_failure_degrades_to_stale(
    isolated_config, tmp_path, capsys, monkeypatch
):
    # A dead bk must never fail `status`: cached state + "(stale)" + a stderr
    # warning, exit 0 — stdout stays clean data.
    from meetscribe import bk
    from meetscribe.cli import main

    root = tmp_path / "meetings"
    d = _meeting_dir(root, "a")
    bk.write_workflow_ref(d, {"workflow_id": "wf_01j9", "state": "processing"})
    _write_bk_config(isolated_config)
    _fake_bk_client(monkeypatch, workflow_error=bk.BkError("network error: down"))

    assert main(["status", str(root)]) == 0
    captured = capsys.readouterr()
    line = next(l for l in captured.out.splitlines() if "m1" in l)
    assert "processing (stale)" in line
    assert "down" in captured.err
    assert "down" not in captured.out

    # the cached ref is untouched — no half-refresh written
    assert bk.read_workflow_ref(d) == {"workflow_id": "wf_01j9", "state": "processing"}


def test_status_unconfigured_bk_degrades_to_stale(
    isolated_config, tmp_path, capsys, monkeypatch
):
    # A non-terminal ref but no [bk] config: refresh is impossible — cached state
    # + "(stale)" + warning, exit 0 (never exit 2: status is read-only reporting).
    from meetscribe import bk
    from meetscribe.cli import main

    root = tmp_path / "meetings"
    d = _meeting_dir(root, "a")
    bk.write_workflow_ref(d, {"workflow_id": "wf_01j9", "state": "processing"})
    created = _fake_bk_client(monkeypatch)

    assert main(["status", str(root)]) == 0
    captured = capsys.readouterr()
    line = next(l for l in captured.out.splitlines() if "m1" in l)
    assert "processing (stale)" in line
    assert captured.err  # the reason is reported
    assert created == []


def test_status_tolerates_corrupt_meta_and_ref(isolated_config, tmp_path, capsys):
    from meetscribe.bk import WORKFLOW_REF_NAME
    from meetscribe.cli import main

    root = tmp_path / "meetings"
    broken = root / "broken"
    broken.mkdir(parents=True)
    (broken / "meta.json").write_text("{not json")
    ok = _meeting_dir(root, "ok")
    (ok / WORKFLOW_REF_NAME).write_text("{not json")  # corrupt ref → "not uploaded"

    assert main(["status", str(root)]) == 0
    out = capsys.readouterr().out
    assert "broken" in out  # listed, not crashed
    line = next(l for l in out.splitlines() if "m1" in l)
    assert "not uploaded" in line


def test_status_scans_a_single_meeting_dir_directly(isolated_config, tmp_path, capsys):
    # Pointing status at a meeting dir itself (not a parent) must work too.
    from meetscribe.cli import main

    d = _meeting_dir(tmp_path, "a")

    assert main(["status", str(d)]) == 0
    assert "m1" in capsys.readouterr().out
