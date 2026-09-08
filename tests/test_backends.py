"""Backend seam: LocalBackend must reproduce today's per-track transcription behavior.

The backend owns VAD→ASR (mic), diarize→ASR→align (system), and the dropped-cluster
warning; pipeline.process() keeps the shared tail (embed, merge/coalesce, cleanup).
"""

import numpy as np

from meetscribe.asr import RawResult
from meetscribe.backends import BackendResult, LocalBackend, TrackInput
from meetscribe.pipeline import Components
from meetscribe.progress import NullReporter
from meetscribe.vad import Chunk

MARK = "▁"


class FakeVad:
    def chunks(self, samples):
        return [Chunk(0.0, np.asarray(samples, dtype=np.float32), 16000)]


class FakeRecognizer:
    def recognize(self, samples):
        return RawResult(
            "hello world", [f"{MARK}hello", f"{MARK}world"], [0.0, 0.6], [0.3, 0.4]
        )


class ScriptedDiarizer:
    def __init__(self, segs):
        self._segs = segs

    def segments(self, samples, on_progress=None):
        from dataclasses import dataclass

        @dataclass
        class S:
            start: float
            end: float
            speaker: int

        if on_progress is not None:
            on_progress(1, 1)
        return [S(*t) for t in self._segs]


class FakeEmbedder:
    dim = 192

    def embed(self, samples):
        return np.ones(192, dtype=np.float32)


class RecordingReporter(NullReporter):
    def __init__(self):
        self.stages = []
        self.warns = []

    def stage(self, label):
        self.stages.append(label)
        return super().stage(label)

    def warn(self, msg):
        self.warns.append(msg)


def _components(diar_segs=((0.0, 1.0, 0),)):
    return Components(
        FakeVad(), FakeRecognizer(), ScriptedDiarizer(list(diar_segs)), FakeEmbedder()
    )


def _track(seconds=1.0):
    return TrackInput("unused.wav", np.zeros(int(seconds * 16000), dtype=np.float32))


def test_local_backend_name():
    assert LocalBackend(_components()).name == "local"


def test_local_backend_both_tracks():
    res = LocalBackend(_components()).transcribe(
        _track(), _track(), [], NullReporter()
    )

    assert isinstance(res, BackendResult)
    # mic: hard-labelled "me", never diarized
    assert [(u.speaker, u.track, u.text) for u in res.mic_utts] == [
        ("me", "mic", "hello world")
    ]
    assert [w.w for w in res.mic_utts[0].words] == ["hello", "world"]
    # system: words assigned to the diarization cluster via align
    assert [(u.speaker, u.track, u.text) for u in res.system_utts] == [
        ("spk_0", "system", "hello world")
    ]
    # diar segments for the embedding pass
    assert [(s.start, s.end, s.speaker) for s in res.system_diar] == [(0.0, 1.0, "spk_0")]
    # local model identity moves here (embedding fields stay in run())
    assert res.models_meta == {
        "asr_model": "parakeet-tdt-0.6b-v3",
        "segmentation_model": "pyannote-segmentation-3.0",
    }


def test_local_backend_mic_only():
    res = LocalBackend(_components()).transcribe(_track(), None, [], NullReporter())
    assert [u.speaker for u in res.mic_utts] == ["me"]
    assert res.system_utts == []
    assert res.system_diar == []


def test_local_backend_system_only():
    res = LocalBackend(_components()).transcribe(None, _track(), [], NullReporter())
    assert res.mic_utts == []
    assert [u.speaker for u in res.system_utts] == ["spk_0"]


def test_local_backend_reports_stages():
    rep = RecordingReporter()
    LocalBackend(_components()).transcribe(_track(), _track(), [], rep)
    assert "transcribe (mic)" in rep.stages
    assert "diarize (system)" in rep.stages
    assert "transcribe (system)" in rep.stages


def test_local_backend_warns_on_dropped_cluster_and_filters_diar():
    """A cluster that wins no words is warned about AND excluded from system_diar
    (embedding speakers must equal transcript speakers)."""
    rep = RecordingReporter()
    # words span 0.0–1.0 s → all land on spk_0; spk_1 (2.0–3.0) wins nothing
    res = LocalBackend(_components(diar_segs=[(0.0, 1.0, 0), (2.0, 3.0, 1)])).transcribe(
        None, _track(3.0), [], rep
    )

    assert {u.speaker for u in res.system_utts} == {"spk_0"}
    assert {s.speaker for s in res.system_diar} == {"spk_0"}
    assert any("spk_1" in w for w in rep.warns), rep.warns


def test_process_defaults_to_local_backend(tmp_path):
    """process(backend=None) must behave exactly as before the seam existed."""
    import wave

    from meetscribe.pipeline import process

    with wave.open(str(tmp_path / "system.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.zeros(16000, dtype="<i2").tobytes())

    res = process(None, str(tmp_path / "system.wav"), _components())
    assert {u.speaker for u in res.utterances} == {"spk_0"}
    assert res.models_meta == {
        "asr_model": "parakeet-tdt-0.6b-v3",
        "segmentation_model": "pyannote-segmentation-3.0",
    }
