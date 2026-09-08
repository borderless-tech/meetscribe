"""Preflight checks: report formatting + exit code. Platform probes are injected."""

import numpy as np

from meetscribe.doctor import (
    Check,
    checks_pass,
    deepgram_key_check,
    deepgram_reachability_check,
    format_report,
    linux_monitor_check,
    remote_checks,
    rms_after_warmup,
    run,
)


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


def test_deepgram_key_check_fails_with_actionable_hint_when_missing():
    for absent in (None, "", "   "):
        check = deepgram_key_check(absent)
        assert not check.ok
        assert "DEEPGRAM_API_KEY" in (check.hint or "")


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
