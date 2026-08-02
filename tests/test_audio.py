"""ffmpeg resample command construction + WAV loading."""

import wave

import numpy as np

from meetscribe.audio import build_resample_cmd, load_wav_f32


def test_build_resample_cmd():
    cmd = build_resample_cmd("in.wav", "out.wav")
    assert cmd == ["ffmpeg", "-y", "-i", "in.wav", "-ar", "16000", "-ac", "1", "out.wav"]


def test_build_resample_cmd_custom_rate():
    cmd = build_resample_cmd("a", "b", rate=8000)
    assert "8000" in cmd and cmd[-1] == "b"


def _write_wav(path, samples_int16, rate=16000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples_int16.astype("<i2").tobytes())


def test_load_wav_f32_round_trip(tmp_path):
    raw = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
    p = tmp_path / "s.wav"
    _write_wav(p, raw, rate=16000)
    samples, rate = load_wav_f32(p)
    assert rate == 16000
    assert samples.dtype == np.float32
    # int16 normalised to [-1, 1) by /32768
    np.testing.assert_allclose(samples, raw.astype(np.float32) / 32768.0, atol=1e-6)


def test_load_wav_f32_is_mono_1d(tmp_path):
    p = tmp_path / "s.wav"
    _write_wav(p, np.zeros(64, dtype=np.int16))
    samples, _ = load_wav_f32(p)
    assert samples.ndim == 1
    assert len(samples) == 64
