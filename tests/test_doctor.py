"""Preflight checks: report formatting + exit code. Platform probes are injected."""

from meetscribe.doctor import Check, checks_pass, format_report, linux_monitor_check, run


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
