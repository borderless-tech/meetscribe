"""Preflight checks: report formatting + exit code. Platform probes are injected."""

import json
from pathlib import Path

import numpy as np
import pytest

from meetscribe.doctor import (
    Check,
    bk_capabilities_check,
    bk_checks,
    bk_reachability_check,
    bk_token_check,
    checks_pass,
    config_checks,
    deepgram_key_check,
    deepgram_reachability_check,
    format_report,
    linux_monitor_check,
    remote_checks,
    rms_after_warmup,
    run,
)


@pytest.fixture(autouse=True)
def _isolated_config(monkeypatch, tmp_path):
    """No test may read the real home: remote_checks now consults the config layer,
    so the config path and XDG dirs always point into tmp_path, and the STT env vars
    from the runner's shell are scrubbed. Tests that need a config layer write
    ``tmp_path / "config.toml"``."""
    monkeypatch.setenv("MEETSCRIBE_CONFIG", str(tmp_path / "config.toml"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))
    for var in (
        "STT_BACKEND", "STT_LANGUAGE", "DEEPGRAM_API_KEY",
        "MEETSCRIBE_BK_URL", "MEETSCRIBE_BK_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)


def test_format_report_marks_ok_and_failures():
    checks = [
        Check("ffmpeg 7.x", True),
        Check("BlackHole not found", False, "brew install blackhole-2ch"),
    ]
    report = format_report(checks)
    assert "✓ ffmpeg 7.x" in report
    assert "✗ BlackHole not found" in report
    assert "→ brew install blackhole-2ch" in report


def test_checks_pass_true_when_all_ok():
    assert checks_pass([Check("a", True), Check("b", True)])


def test_checks_pass_false_on_any_failure():
    assert not checks_pass([Check("a", True), Check("b", False, "fix it")])


def test_linux_monitor_check_shows_chosen_default_sink_monitor():
    # doctor must display the SAME source record.py would tap — the default sink's monitor —
    # so a mis-selection (e.g. a silent HDMI monitor) is obvious at a glance in the preflight.
    sources = [
        "alsa_output.hdmi3.monitor",
        "alsa_input.builtin_mic",
        "bluez_output.AC_80_0A_F3_FB_F1.1.monitor",
    ]
    check = linux_monitor_check(sources, default_sink="bluez_output.AC_80_0A_F3_FB_F1.1")
    assert check.ok
    assert "bluez_output.AC_80_0A_F3_FB_F1.1.monitor" in check.name


def test_linux_monitor_check_fails_when_no_monitor():
    check = linux_monitor_check(["alsa_input.builtin_mic"], default_sink=None)
    assert not check.ok
    assert check.hint is not None


def test_rms_after_warmup_ignores_transition_silence():
    # A Bluetooth headset emits digital silence for ~1s while WirePlumber switches A2DP→HFP.
    # Measuring across the whole clip would read that as a dead mic; the warm-up window must be
    # skipped so the real signal in the tail is what counts.
    sr = 16000
    silent_head = np.zeros(sr, dtype=np.float32)          # 1.0 s transition silence
    voiced_tail = np.ones(sr * 2, dtype=np.float32)       # 2.0 s of signal
    samples = np.concatenate([silent_head, voiced_tail])
    assert rms_after_warmup(samples, sr, warmup_s=1.5) > 0.0


def test_rms_after_warmup_zero_for_fully_silent_capture():
    # A genuinely muted/absent mic stays silence even after the warm-up → still a failure.
    sr = 16000
    assert rms_after_warmup(np.zeros(sr * 3, dtype=np.float32), sr, warmup_s=1.5) == 0.0


def test_rms_after_warmup_falls_back_when_clip_shorter_than_warmup():
    # Never discard the entire clip: a short capture must still be measured, not read as silent.
    sr = 16000
    samples = np.ones(sr // 2, dtype=np.float32)          # 0.5 s, shorter than warm-up
    assert rms_after_warmup(samples, sr, warmup_s=1.5) > 0.0


def test_run_returns_zero_when_all_pass(capsys):
    class FakeProbe:
        def checks(self):
            return [Check("ffmpeg", True), Check("mic RMS > 0", True)]

    assert run(probe=FakeProbe()) == 0
    assert "✓ ffmpeg" in capsys.readouterr().out


def test_run_returns_nonzero_on_failure(capsys):
    class FakeProbe:
        def checks(self):
            return [Check("mic RMS > 0", False, "grant mic permission to the terminal")]

    assert run(probe=FakeProbe()) == 1
    out = capsys.readouterr().out
    assert "✗ mic RMS > 0" in out
    assert "→ grant mic permission" in out


# --- Remote-backend checks (STT_BACKEND=deepgram) --------------------------------------------
# The remote backend has NO silent fallback to local, so doctor must surface a missing key and
# an unreachable API up front. The TLS connector is injected — no network (and no billable
# Deepgram call) in tests.


def _ok_connect(host, port):
    _ok_connect.calls.append((host, port))


_ok_connect.calls = []


def test_remote_checks_empty_when_backend_is_local():
    # Doctor stays purely local unless the env opts into the remote backend.
    assert remote_checks(env={}, connect=_ok_connect) == []
    assert remote_checks(env={"STT_BACKEND": "local"}, connect=_ok_connect) == []


def test_remote_checks_run_for_deepgram_backend_and_probe_the_right_endpoint():
    _ok_connect.calls.clear()
    checks = remote_checks(
        env={"STT_BACKEND": "deepgram", "DEEPGRAM_API_KEY": "tok"}, connect=_ok_connect
    )
    assert len(checks) == 2
    assert all(c.ok for c in checks)
    # DNS+TLS reachability must probe the real API endpoint — and nothing else.
    assert _ok_connect.calls == [("api.deepgram.com", 443)]


def test_remote_checks_backend_from_config_triggers_checks(tmp_path):
    # [stt].backend = "deepgram" with no env var: the pipeline WILL use deepgram,
    # so doctor must run the key + reachability checks — not silently skip them.
    (tmp_path / "config.toml").write_text(
        '[stt]\nbackend = "deepgram"\n[deepgram]\napi_key = "dg_cfg_key"\n'
    )
    _ok_connect.calls.clear()
    checks = remote_checks(env={}, connect=_ok_connect)
    assert len(checks) == 2
    assert all(c.ok for c in checks), [(c.name, c.hint) for c in checks]
    assert _ok_connect.calls == [("api.deepgram.com", 443)]


def test_remote_checks_env_backend_with_config_key_is_not_a_false_red(tmp_path):
    # STT_BACKEND=deepgram in the env, key only in the config: record/process work,
    # so doctor must not fail the key check on this working setup.
    (tmp_path / "config.toml").write_text('[deepgram]\napi_key = "dg_cfg_key"\n')
    checks = remote_checks(env={"STT_BACKEND": "deepgram"}, connect=_ok_connect)
    assert all(c.ok for c in checks), [(c.name, c.hint) for c in checks]


def test_remote_checks_env_backend_beats_config_backend(tmp_path):
    # env > config: STT_BACKEND=local overrides a config-selected deepgram.
    (tmp_path / "config.toml").write_text('[stt]\nbackend = "deepgram"\n')
    assert remote_checks(env={"STT_BACKEND": "local"}, connect=_ok_connect) == []


def test_deepgram_key_check_fails_with_actionable_hint_when_missing():
    for absent in (None, "", "   "):
        check = deepgram_key_check(absent)
        assert not check.ok
        assert "DEEPGRAM_API_KEY" in (check.hint or "")
        # both fixes are offered, matching check_backend's fail-fast message
        assert "[deepgram].api_key" in (check.hint or "")


def test_deepgram_key_check_passes_when_set():
    check = deepgram_key_check("dg_secret")
    assert check.ok
    assert "dg_secret" not in check.name  # never echo the secret into the report


def test_deepgram_reachability_check_fails_offline_with_hint():
    def down(host, port):
        raise OSError("Name or service not known")

    check = deepgram_reachability_check(connect=down)
    assert not check.ok
    assert "offline" in (check.hint or "").lower()


def test_deepgram_reachability_check_passes_when_tls_connects():
    check = deepgram_reachability_check(connect=lambda host, port: None)
    assert check.ok
    assert "api.deepgram.com" in check.name


def test_remote_checks_report_missing_key_and_unreachable_api_together():
    # Both failures must be visible in one doctor run, not discovered one at a time.
    def down(host, port):
        raise OSError("network is unreachable")

    checks = remote_checks(env={"STT_BACKEND": "deepgram"}, connect=down)
    assert [c.ok for c in checks] == [False, False]


# --- Config check ------------------------------------------------------------------------------
# Doctor surfaces the config file's health up front: path in use + existence, parse status
# (malformed = red with the line number), unknown keys and api_key file permissions as warnings.
# A broken config must never abort the other checks — config_checks returns Checks, never raises.


def test_format_report_renders_warnings_distinctly_with_hint():
    checks = [Check("all good", True), Check("config keys", True, "unknown key: stt.foo", warn=True)]
    report = format_report(checks)
    assert "✓ all good" in report
    assert "! config keys" in report
    assert "→ unknown key: stt.foo" in report
    assert "✗" not in report


def test_checks_pass_treats_warnings_as_passing():
    # Warnings are advisory: doctor still exits 0 on warnings alone.
    assert checks_pass([Check("a", True), Check("b", True, "advice", warn=True)])


def test_config_checks_missing_file_is_ok_and_names_the_path(tmp_path):
    p = tmp_path / "config.toml"
    checks = config_checks(p)
    assert len(checks) == 1
    assert checks[0].ok
    assert str(p) in checks[0].name
    assert "defaults" in checks[0].name


def test_config_checks_valid_file_passes(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text('[stt]\nbackend = "local"\n')
    checks = config_checks(p)
    assert all(c.ok for c in checks)
    assert not any(c.warn for c in checks)
    assert str(p) in checks[0].name


def test_config_checks_malformed_toml_is_red_with_line_and_does_not_raise(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text('[stt]\nbackend = not-a-string\n')
    checks = config_checks(p)  # must not raise — the other doctor checks still run
    assert len(checks) == 1
    assert not checks[0].ok
    assert "line 2" in (checks[0].hint or "")
    assert str(p) in checks[0].name


def test_config_checks_wrongly_typed_value_is_red(tmp_path):
    # Valid TOML, but every record/process run would exit 2 on it — a green doctor
    # would defeat the preflight's purpose, so the resolvers run here too.
    p = tmp_path / "config.toml"
    p.write_text('[output]\nbundle = "false"\n')
    checks = config_checks(p)
    assert not checks_pass(checks)
    bad = next(c for c in checks if not c.ok)
    assert "bundle" in (bad.hint or "")


def test_config_checks_unreadable_file_is_red_not_a_crash(tmp_path):
    # config_checks promises to never raise; a chmod-000 file must not abort the
    # audio checks that follow.
    p = tmp_path / "config.toml"
    p.write_text("[stt]\n")
    p.chmod(0o000)
    try:
        checks = config_checks(p)
    finally:
        p.chmod(0o600)
    assert not checks_pass(checks)


def test_config_checks_warn_on_unknown_keys_but_still_pass(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text('[stt]\nbackedn = "local"\n')  # typo'd key
    checks = config_checks(p)
    warns = [c for c in checks if c.warn]
    assert len(warns) == 1
    assert "stt.backedn" in (warns[0].hint or "")
    assert checks_pass(checks)  # warning, not failure


def test_config_checks_warn_when_api_key_file_is_group_readable(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text('[deepgram]\napi_key = "dg_secret"\n')
    p.chmod(0o644)
    checks = config_checks(p)
    warns = [c for c in checks if c.warn]
    assert len(warns) == 1
    assert "0600" in (warns[0].hint or "")
    assert "dg_secret" not in format_report(checks)  # never echo the secret
    assert checks_pass(checks)


def test_config_checks_perm_warning_names_bk_token_too(tmp_path):
    # [bk].token is secret material like the deepgram key — a group-readable file
    # must warn and name the key that is actually set, not the deepgram one.
    p = tmp_path / "config.toml"
    p.write_text('[bk]\ntoken = "bk-secret"\n')
    p.chmod(0o644)
    checks = config_checks(p)
    warns = [c for c in checks if c.warn]
    assert len(warns) == 1
    assert "[bk].token" in (warns[0].hint or "")
    assert "bk-secret" not in format_report(checks)


def test_config_checks_no_perm_warning_when_file_is_0600(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text('[deepgram]\napi_key = "dg_secret"\n')
    p.chmod(0o600)
    assert not any(c.warn for c in config_checks(p))


def test_config_checks_default_path_honors_meetscribe_config_env(tmp_path, monkeypatch):
    p = tmp_path / "elsewhere.toml"
    p.write_text('[output]\nbundle = true\n')
    monkeypatch.setenv("MEETSCRIBE_CONFIG", str(p))
    checks = config_checks()
    assert checks[0].ok
    assert str(p) in checks[0].name


def test_deepgram_key_check_accepts_lazy_cmd_marker():
    # A configured api_key_cmd counts as "key available" — and must not be
    # executed (doctor may run headless; pass/gpg could block on pinentry).
    from meetscribe.config import ApiKeyCmd
    from meetscribe.doctor import deepgram_key_check

    check = deepgram_key_check(ApiKeyCmd("pass show deepgram"))
    assert check.ok is True


# --- bk (borderless-knowledge) checks ----------------------------------------------------------
# Appended only when [bk].base_url resolves or auto-upload is on: host reachability (injected
# connect seam, no real network), token presence (a SecretCmd marker counts and is NEVER
# executed — no pinentry in doctor), and — only with a *static* token — a live capabilities +
# format-support call through an injected opener.

_BK_FIXTURES = Path(__file__).parent / "fixtures" / "bk"


def _caps():
    return json.loads((_BK_FIXTURES / "capabilities.json").read_text())


class _RecordingConnect:
    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def __call__(self, host, port):
        self.calls.append((host, port))
        if self.error is not None:
            raise self.error


class _CapsOpener:
    """Fake transport for the live capabilities check (never touches the network)."""

    def __init__(self, outcome):
        self.outcome = outcome
        self.requests = []

    def __call__(self, request, timeout):
        self.requests.append(request)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        payload = json.dumps(self.outcome).encode("utf-8")

        class _Response:
            def read(self):
                return payload

        return _Response()


def test_bk_checks_empty_when_bk_unconfigured():
    # Without a base_url and with auto-upload off, doctor stays bk-silent.
    connect = _RecordingConnect()
    assert bk_checks(env={}, cfg={}, connect=connect, opener=_CapsOpener(_caps())) == []
    assert connect.calls == []


def test_bk_checks_all_green_with_static_token(tmp_path):
    (tmp_path / "config.toml").write_text(
        '[bk]\nbase_url = "https://bk.example.com"\ntoken = "tok-secret"\n'
    )
    connect = _RecordingConnect()
    opener = _CapsOpener(_caps())
    checks = bk_checks(env={}, connect=connect, opener=opener)
    assert len(checks) == 3
    assert all(c.ok for c in checks), [(c.name, c.hint) for c in checks]
    # reachability probes exactly the configured host, default HTTPS port
    assert connect.calls == [("bk.example.com", 443)]
    # the live check hit /capabilities with the bearer token
    assert len(opener.requests) == 1
    req = opener.requests[0]
    assert req.full_url == "https://bk.example.com/api/meetscribe/v1/capabilities"
    assert req.get_header("Authorization") == "Bearer tok-secret"
    assert "tok-secret" not in format_report(checks)  # never echo the secret


def test_bk_checks_env_url_wins_and_custom_port_is_probed(tmp_path):
    (tmp_path / "config.toml").write_text('[bk]\nbase_url = "https://wrong.example.com"\n')
    connect = _RecordingConnect()
    checks = bk_checks(
        env={"MEETSCRIBE_BK_URL": "https://bk.example.com:8443", "MEETSCRIBE_BK_TOKEN": "t"},
        connect=connect,
        opener=_CapsOpener(_caps()),
    )
    assert connect.calls == [("bk.example.com", 8443)]
    assert all(c.ok for c in checks), [(c.name, c.hint) for c in checks]


def test_bk_checks_auto_upload_without_base_url_is_red(tmp_path):
    # auto_upload = true with no URL: every record/process run would exit 2 on it,
    # so doctor must go red — and still report the token state in the same run.
    (tmp_path / "config.toml").write_text('[bk]\nauto_upload = true\n')
    connect = _RecordingConnect()
    checks = bk_checks(env={}, connect=connect, opener=_CapsOpener(_caps()))
    assert checks, "auto-upload on must trigger the bk checks even without a base_url"
    assert not checks_pass(checks)
    url_check = checks[0]
    assert not url_check.ok
    assert "[bk].base_url" in (url_check.hint or "")
    assert "MEETSCRIBE_BK_URL" in (url_check.hint or "")
    assert connect.calls == []  # nothing to probe without a URL


def test_bk_checks_report_missing_token_and_unreachable_host_together(tmp_path):
    # Both failures must show in one doctor run, not be discovered one at a time.
    (tmp_path / "config.toml").write_text('[bk]\nbase_url = "https://bk.example.com"\n')
    connect = _RecordingConnect(error=OSError("network is unreachable"))
    checks = bk_checks(env={}, connect=connect, opener=_CapsOpener(_caps()))
    assert [c.ok for c in checks] == [False, False]
    # no static token → the live capabilities call must not happen at all
    assert len(checks) == 2


def test_bk_token_cmd_counts_as_present_and_is_never_executed(tmp_path):
    # doctor may run headless — a keyring command could block on pinentry, so the
    # marker counts as "token available" but the command must never run.
    canary = tmp_path / "executed"
    (tmp_path / "config.toml").write_text(
        f'[bk]\nbase_url = "https://bk.example.com"\ntoken_cmd = "touch {canary}"\n'
    )
    checks = bk_checks(env={}, connect=_RecordingConnect(), opener=_CapsOpener(_caps()))
    assert checks_pass(checks), [(c.name, c.hint) for c in checks]
    assert not canary.exists(), "doctor must never execute [bk].token_cmd"
    # the live capabilities check is skipped (needs the secret) — visibly, as a warning
    skipped = [c for c in checks if c.warn]
    assert len(skipped) == 1
    assert "token_cmd" in (skipped[0].hint or "")


def test_bk_token_check_missing_is_red_with_actionable_hint():
    for absent in (None, "", "   "):
        check = bk_token_check(absent)
        assert not check.ok
        for knob in ("[bk].token", "token_cmd", "MEETSCRIBE_BK_TOKEN"):
            assert knob in (check.hint or "")


def test_bk_token_check_passes_without_echoing_the_secret():
    check = bk_token_check("tok-secret")
    assert check.ok
    assert "tok-secret" not in check.name


def test_bk_reachability_check_fails_offline_with_hint():
    check = bk_reachability_check(
        "https://bk.example.com", connect=_RecordingConnect(error=OSError("no route"))
    )
    assert not check.ok
    assert "bk.example.com" in check.name
    assert check.hint is not None


def test_bk_reachability_check_rejects_a_relative_base_url():
    # urlsplit("bk.example.com") has no hostname — probing would fail confusingly,
    # so the check itself must say the URL is malformed.
    connect = _RecordingConnect()
    check = bk_reachability_check("bk.example.com", connect=connect)
    assert not check.ok
    assert "https://" in (check.hint or "")
    assert connect.calls == []


def test_bk_capabilities_check_red_when_format_unsupported():
    caps = {**_caps(), "mscribe_format_versions": [1]}
    check = bk_capabilities_check("https://bk.example.com", "tok", opener=_CapsOpener(caps))
    assert not check.ok
    assert "format_version" in (check.hint or "")


def test_bk_capabilities_check_red_on_401_names_the_token_knobs():
    import io
    import urllib.error

    err = urllib.error.HTTPError(
        "https://bk.example.com/api/meetscribe/v1/capabilities", 401, "unauthorized",
        {}, io.BytesIO(b"{}"),
    )
    check = bk_capabilities_check("https://bk.example.com", "tok", opener=_CapsOpener(err))
    assert not check.ok
    assert "[bk].token" in (check.hint or "")
