"""VAD chunk offset math (what-we-build.md §2.4). The sherpa VAD is mocked via fake segments.

sherpa's SpeechSegment exposes .start (int sample index) and .samples only — NO .end — so the
absolute end is derived from the sample count.
"""

from dataclasses import dataclass

import numpy as np

from meetscribe.vad import Chunk, segments_to_chunks


@dataclass
class FakeSeg:
    start: int
    samples: np.ndarray


def test_offset_is_start_sample_over_rate():
    seg = FakeSeg(start=48000, samples=np.zeros(1600, dtype=np.float32))
    chunk = segments_to_chunks([seg], sample_rate=16000)[0]
    assert chunk.offset == 3.0


def test_end_is_offset_plus_duration():
    # offset 3.0 s, 16000 samples @16k = 1.0 s -> end 4.0
    seg = FakeSeg(start=48000, samples=np.zeros(16000, dtype=np.float32))
    chunk = segments_to_chunks([seg], sample_rate=16000)[0]
    assert chunk.end == 4.0
    assert chunk.duration == 1.0


def test_samples_preserved():
    s = np.arange(10, dtype=np.float32)
    chunk = segments_to_chunks([FakeSeg(0, s)], sample_rate=16000)[0]
    assert np.array_equal(chunk.samples, s)


def test_multiple_segments_in_order():
    segs = [
        FakeSeg(0, np.zeros(8000, dtype=np.float32)),
        FakeSeg(32000, np.zeros(8000, dtype=np.float32)),
    ]
    chunks = segments_to_chunks(segs, sample_rate=16000)
    assert [c.offset for c in chunks] == [0.0, 2.0]


def test_empty_segments():
    assert segments_to_chunks([], sample_rate=16000) == []


def test_chunk_is_hashable_dataclass():
    c = Chunk(offset=1.0, samples=np.zeros(3, dtype=np.float32), sample_rate=16000)
    assert c.start == 1.0
