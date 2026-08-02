"""Speaker embeddings: short-segment filter, audio slicing, per-turn + per-cluster centroids.

Extractor is mocked. Cluster centroids concatenate a cluster's audio then embed ONCE
(what-we-build.md §2.7), rather than averaging per-turn vectors.
"""

import numpy as np

from meetscribe.embed import cluster_centroids, embed_turns, filter_short, slice_audio
from meetscribe.types import DiarSegment

DIM = 192


class FakeExtractor:
    """Returns a deterministic vector and records the sample counts it was asked to embed."""

    def __init__(self, dim=DIM):
        self.dim = dim
        self.received_lengths = []

    def embed(self, samples):
        self.received_lengths.append(len(samples))
        return np.full(self.dim, float(len(samples)), dtype=np.float32)


def test_filter_short_drops_sub_threshold():
    segs = [DiarSegment(0.0, 0.5, "spk_0"), DiarSegment(1.0, 2.0, "spk_0")]
    assert filter_short(segs, min_dur=0.8) == [DiarSegment(1.0, 2.0, "spk_0")]


def test_slice_audio_returns_sample_range():
    wav = np.arange(16000, dtype=np.float32)
    sl = slice_audio(wav, 16000, 0.5, 1.0)
    assert len(sl) == 8000
    assert sl[0] == 8000.0


def test_embed_turns_one_vector_per_turn_with_speaker():
    wav = np.ones(16000, dtype=np.float32)
    segs = [DiarSegment(0.0, 1.0, "spk_0"), DiarSegment(1.0, 2.0, "spk_1")]
    ext = FakeExtractor()
    turns = embed_turns(ext, wav, 16000, segs)
    assert [t[0] for t in turns] == ["turn_0", "turn_1"]
    assert [t[2] for t in turns] == ["spk_0", "spk_1"]
    assert turns[0][1].shape == (DIM,)


def test_embed_turns_asserts_dim_192():
    ext = FakeExtractor(dim=512)
    try:
        embed_turns(ext, np.ones(16000, dtype=np.float32), 16000, [DiarSegment(0.0, 1.0, "spk_0")])
        assert False, "expected AssertionError on wrong dim"
    except AssertionError as e:
        assert "192" in str(e)


def test_cluster_centroids_concatenate_audio_then_embed_once():
    wav = np.arange(48000, dtype=np.float32)  # 3 s @ 16k
    clusters = {
        "spk_0": [DiarSegment(0.0, 0.5, "spk_0"), DiarSegment(1.0, 1.5, "spk_0")],  # 8000+8000
    }
    ext = FakeExtractor()
    cents = cluster_centroids(ext, wav, 16000, clusters)
    assert [c[0] for c in cents] == ["spk_0"]
    # embed called ONCE with the concatenated audio (16000 samples), not twice-averaged
    assert ext.received_lengths == [16000]
    assert cents[0][1].shape == (DIM,)
