"""record.py — device parsing, name matching, ffmpeg command building (both OS).

Pure helpers only; no ffmpeg is executed here (the real Linux capture is a separate, skippable
integration test). Device indices are matched by NAME — never hard-coded (§5).
"""

from pathlib import Path

import numpy as np
import pytest

from meetscribe.record import (
    build_linux_ffmpeg_cmd,
    build_macos_ffmpeg_cmd,
    find_monitor_source,
    match_device,
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
