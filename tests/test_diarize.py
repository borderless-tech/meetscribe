"""Diarization: map sherpa int speaker ids to spk_N labels. Diarizer is mocked."""

from dataclasses import dataclass

import numpy as np

from meetscribe.diarize import run, to_diar_segments
from meetscribe.types import DiarSegment


@dataclass
class RawSeg:
    start: float
    end: float
    speaker: int


def test_to_diar_segments_maps_int_ids_to_labels():
    raw = [RawSeg(0.0, 1.0, 0), RawSeg(1.0, 2.0, 1), RawSeg(2.0, 3.0, 0)]
    out = to_diar_segments(raw)
    assert out == [
        DiarSegment(0.0, 1.0, "spk_0"),
        DiarSegment(1.0, 2.0, "spk_1"),
        DiarSegment(2.0, 3.0, "spk_0"),
    ]


def test_run_uses_backend_and_returns_labelled_segments():
    class FakeDiarizer:
        def segments(self, samples):
            return [RawSeg(0.0, 1.0, 2), RawSeg(1.0, 2.0, 2)]

    out = run(FakeDiarizer(), np.zeros(16000, dtype=np.float32))
    assert [s.speaker for s in out] == ["spk_2", "spk_2"]


def test_empty():
    assert to_diar_segments([]) == []
