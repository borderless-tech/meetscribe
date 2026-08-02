"""record.py — device parsing, name matching, ffmpeg command building (both OS).

Pure helpers only; no ffmpeg is executed here (the real Linux capture is a separate, skippable
integration test). Device indices are matched by NAME — never hard-coded (§5).
"""

from pathlib import Path

import numpy as np
import pytest

from meetscribe.record import (
    _drive_meters,
    _popen_kwargs,
    build_linux_ffmpeg_cmd,
    build_macos_ffmpeg_cmd,
    find_monitor_source,
    match_device,
    parse_astats,
    parse_avfoundation_devices,
    parse_pw_sources,
    rms,
    stop_ffmpeg,
    warn_if_silent,
)

FIX = Path(__file__).parent / "fixtures"


# ---- macOS device parsing / matching -------------------------------------------------

def test_parse_avfoundation_audio_devices_only():
    text = (FIX / "avfoundation_devices.txt").read_text()
    devices = parse_avfoundation_devices(text)
    assert devices == [(0, "Built-in Microphone"), (1, "meetscribe"), (2, "BlackHole 2ch")]


def test_match_device_by_name_returns_index():
    devices = [(0, "Built-in Microphone"), (1, "meetscribe"), (2, "BlackHole 2ch")]
    assert match_device(devices, "meetscribe") == 1


def test_match_device_missing_raises():
    with pytest.raises(ValueError, match="nope"):
        match_device([(0, "Built-in Microphone")], "nope")


def test_macos_cmd_has_pan_split_and_two_outputs():
    cmd = build_macos_ffmpeg_cmd(1, "raw/mic.wav", "raw/system.wav")
    joined = " ".join(cmd)
    assert "avfoundation" in joined
    assert '-i' in cmd and ':1' in joined            # aggregate device index, matched by name
    assert "pan=" in joined                          # channel split, not stereo downmix
    assert cmd.count("-map") == 2
    assert "raw/mic.wav" in cmd and "raw/system.wav" in cmd


# ---- Linux source parsing / command --------------------------------------------------

def test_parse_pw_sources_returns_names():
    text = (FIX / "pactl_sources.txt").read_text()
    names = parse_pw_sources(text)
    assert "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor" in names
    assert "alsa_input.pci-0000_00_1f.3.analog-stereo" in names


def test_find_monitor_source_prefers_monitor():
    names = parse_pw_sources((FIX / "pactl_sources.txt").read_text())
    assert find_monitor_source(names).endswith(".monitor")


def test_linux_cmd_has_two_pulse_inputs_and_two_outputs():
    cmd = build_linux_ffmpeg_cmd("mic_src", "sink.monitor", "raw/mic.wav", "raw/system.wav")
    assert cmd.count("pulse") == 2
    assert "mic_src" in cmd and "sink.monitor" in cmd
    assert cmd.count("-map") == 2
    assert cmd[-1] == "raw/system.wav"


# ---- RMS / silence / graceful stop ---------------------------------------------------

def test_rms_of_known_buffer():
    assert rms(np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)) == pytest.approx(1.0)


def test_rms_zero_for_silence():
    assert rms(np.zeros(100, dtype=np.float32)) == 0.0


def test_warn_if_silent_flags_silent_wav(tmp_path):
    import wave

    p = tmp_path / "silent.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.zeros(16000, dtype="<i2").tobytes())
    msg = warn_if_silent(p)
    assert msg is not None and "silent" in msg.lower()


## ---- astats metering ---------------------------------------------------------------

def test_parse_astats_per_channel():
    assert parse_astats("lavfi.astats.1.RMS_level=-29.537922") == (1, -29.537922)
    assert parse_astats("lavfi.astats.2.RMS_level=-16.258925") == (2, -16.258925)


def test_parse_astats_inf_is_negative_infinity():
    assert parse_astats("lavfi.astats.1.RMS_level=-inf") == (1, float("-inf"))


def test_parse_astats_non_matching_returns_none():
    assert parse_astats("frame:0    pts:0       pts_time:0") is None
    assert parse_astats("lavfi.astats.Overall.Peak_level=-3.0") is None


def test_linux_metered_cmd_has_astats_amerge_and_two_wavs():
    cmd = build_linux_ffmpeg_cmd("mic_src", "sink.monitor", "mic.wav", "sys.wav", metered=True)
    j = " ".join(cmd)
    assert "astats" in j and "amerge" in j and "ametadata" in j
    assert j.count("pulse") == 2
    assert "mic.wav" in cmd and "sys.wav" in cmd


def test_macos_metered_cmd_has_astats_and_pan():
    cmd = build_macos_ffmpeg_cmd(1, "mic.wav", "sys.wav", metered=True)
    j = " ".join(cmd)
    assert "astats" in j and "amerge" in j and "pan=" in j


def test_record_cmds_suppress_ffmpeg_log_noise():
    # ffmpeg's banner + astats per-frame report go to stderr, where the rich UI also renders;
    # -loglevel error -nostats keeps them from interleaving (meter data is on stdout, unaffected).
    for cmd in (
        build_linux_ffmpeg_cmd("m", "s", "a", "b"),
        build_linux_ffmpeg_cmd("m", "s", "a", "b", metered=True),
        build_macos_ffmpeg_cmd(0, "a", "b"),
        build_macos_ffmpeg_cmd(0, "a", "b", metered=True),
    ):
        assert "-loglevel" in cmd and "error" in cmd
        assert "-nostats" in cmd


def test_non_metered_cmds_unchanged():
    # default (metered=False) keeps the simple two-output command
    assert build_linux_ffmpeg_cmd("m", "s", "a", "b").count("-map") == 2
    assert "astats" not in " ".join(build_macos_ffmpeg_cmd(0, "a", "b"))


def test_popen_kwargs_metered_keeps_stdin_binary():
    import subprocess

    # metered must NOT use text mode: stop_ffmpeg writes b"q" to stdin, and a text-mode
    # stdin would raise TypeError inside the SIGINT handler (regression guard).
    kw = _popen_kwargs(metered=True)
    assert kw.get("text") in (None, False)
    assert kw["stdin"] == subprocess.PIPE
    assert kw["stdout"] == subprocess.PIPE
    # non-metered: no stdout pipe needed
    assert "stdout" not in _popen_kwargs(metered=False)


def test_drive_meters_parses_bytes_and_updates_channels():
    class RecMeters:
        def __init__(self):
            self.updates = []

        def update(self, ch, db):
            self.updates.append((ch, db))

    stream = [
        b"frame:0    pts_time:0\n",
        b"lavfi.astats.1.RMS_level=-20.0\n",
        b"lavfi.astats.2.RMS_level=-10.0\n",
        b"lavfi.astats.Overall.Peak_level=-3.0\n",  # ignored (not per-channel RMS)
    ]
    m = RecMeters()
    _drive_meters(iter(stream), m)
    assert m.updates == [(1, -20.0), (2, -10.0)]


def test_stop_ffmpeg_tolerates_write_typeerror():
    # defense-in-depth: a mode mismatch must not propagate out of the SIGINT handler.
    class BadStdin:
        def write(self, b):
            raise TypeError("write() argument must be str, not bytes")

        def flush(self):
            pass

    class Proc:
        stdin = BadStdin()

        def wait(self, timeout=None):
            return 0

    stop_ffmpeg(Proc())  # must not raise


def test_stop_ffmpeg_sends_q_not_kill():
    class FakeStdin:
        def __init__(self):
            self.written = b""
            self.flushed = False

        def write(self, b):
            self.written += b

        def flush(self):
            self.flushed = True

    class FakeProc:
        def __init__(self):
            self.stdin = FakeStdin()
            self.killed = False
            self.waited = False

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            self.waited = True
            return 0

    proc = FakeProc()
    stop_ffmpeg(proc)
    assert proc.stdin.written == b"q"
    assert proc.stdin.flushed
    assert not proc.killed


# ---- run() threads the real capture start into the pipeline --------------------------

def test_run_passes_started_at_and_bundle_to_pipeline(tmp_path, monkeypatch):
    # record.run knows the real wall-clock meeting start; it must hand a tz-aware
    # started_at (and the --bundle flag) to pipeline.run rather than let the pipeline
    # fall back to the WAV mtime.
    captured = {}
    monkeypatch.setattr("meetscribe.record.record_tracks", lambda *a, **k: None)
    monkeypatch.setattr("meetscribe.record.warn_if_silent", lambda p: None)
    import meetscribe.pipeline as pl
    monkeypatch.setattr(pl, "run", lambda **k: captured.update(k) or 0)

    from meetscribe import record
    assert record.run(out_dir=str(tmp_path / "m"), bundle=True) == 0

    assert captured["bundle"] is True
    assert captured["started_at"] is not None  # a tz-aware datetime
    assert captured["started_at"].tzinfo is not None
