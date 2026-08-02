"""Pipeline orchestration: input resolution + two-track wiring (fast, with fake stages).

The heavy real-model end-to-end smoke lives in tests/test_e2e.py (opt-in, needs MEETSCRIBE_MODELS).
"""

import wave

import numpy as np

from meetscribe.asr import RawResult
from meetscribe.pipeline import Components, process, resolve_inputs
from meetscribe.vad import Chunk


def _write_wav(path, seconds=1.0, rate=16000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.zeros(int(seconds * rate), dtype="<i2").tobytes())


# ---- resolve_inputs -----------------------------------------------------------------

def test_resolve_inputs_directory_with_raw(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_wav(raw / "mic.wav")
    _write_wav(raw / "system.wav")
    mic, system = resolve_inputs(str(tmp_path))
    assert mic.endswith("raw/mic.wav") and system.endswith("raw/system.wav")


def test_resolve_inputs_single_file_is_system(tmp_path):
    f = tmp_path / "meeting.wav"
    _write_wav(f)
    mic, system = resolve_inputs(str(f))
    assert mic is None and system.endswith("meeting.wav")


def test_resolve_inputs_mic_only(tmp_path):
    _write_wav(tmp_path / "mic.wav")
    mic, system = resolve_inputs(str(tmp_path))
    assert mic.endswith("mic.wav") and system is None


# ---- process (fake stages) ----------------------------------------------------------

MARK = "▁"


class FakeVad:
    def chunks(self, samples):
        return [Chunk(0.0, np.asarray(samples, dtype=np.float32), 16000)]


class FakeRecognizer:
    def recognize(self, samples):
        # spans 0.0–1.0 s so the utterance survives the 0.8 s short-segment filter
        return RawResult("hello world", [f"{MARK}hello", f"{MARK}world"], [0.0, 0.6], [0.3, 0.4])


class FakeDiarizer:
    def segments(self, samples):
        from dataclasses import dataclass

        @dataclass
        class S:
            start: float
            end: float
            speaker: int

        return [S(0.0, 1.0, 0)]


class FakeEmbedder:
    dim = 192

    def embed(self, samples):
        return np.ones(192, dtype=np.float32)


def _components():
    return Components(FakeVad(), FakeRecognizer(), FakeDiarizer(), FakeEmbedder())


def test_process_both_tracks(tmp_path):
    _write_wav(tmp_path / "mic.wav")
    _write_wav(tmp_path / "system.wav")
    res = process(str(tmp_path / "mic.wav"), str(tmp_path / "system.wav"), _components())

    speakers = {u.speaker for u in res.utterances}
    tracks = {u.track for u in res.utterances}
    assert "me" in speakers and "spk_0" in speakers
    assert tracks == {"mic", "system"}
    assert res.dim == 192
    # turns from both tracks, ids prefixed & unique
    ids = [t[0] for t in res.turns]
    assert any(i.startswith("mic_") for i in ids)
    assert any(i.startswith("sys_") for i in ids)
    assert len(ids) == len(set(ids))
    # a "me" centroid and a spk_0 centroid
    cluster_ids = {c[0] for c in res.clusters}
    assert "me" in cluster_ids and "spk_0" in cluster_ids
    assert res.duration_s == 1.0


def test_process_system_only(tmp_path):
    _write_wav(tmp_path / "system.wav")
    res = process(None, str(tmp_path / "system.wav"), _components())
    assert all(u.track == "system" for u in res.utterances)
    assert all(t[0].startswith("sys_") for t in res.turns)
