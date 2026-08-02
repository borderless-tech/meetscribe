"""Speaker embeddings — the first-class output (what-we-build.md §2.5–2.7).

A second pass over the diarization/VAD segments with the CAM++ extractor produces one embedding
per turn, and one *centroid* per cluster. The centroid is computed by CONCATENATING all of a
cluster's audio and embedding it once — markedly more stable than averaging many short-turn vectors.
Segments shorter than ~0.8 s yield unusable embeddings and are dropped.
"""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence

import numpy as np

from .types import DiarSegment

EMBEDDING_DIM = 192  # CAM++ (research report) — NOT the 512 in the spec

# (turn_id, vector (dim,), speaker)
Turn = tuple[str, np.ndarray, str]
# (cluster_id, vector (dim,))
Centroid = tuple[str, np.ndarray]


class Extractor(Protocol):
    dim: int

    def embed(self, samples: np.ndarray) -> np.ndarray: ...


def _check_dim(extractor: Extractor) -> None:
    assert (
        extractor.dim == EMBEDDING_DIM
    ), f"expected embedding dim {EMBEDDING_DIM}, got {extractor.dim}"


def filter_short(
    segments: Sequence[DiarSegment], min_dur: float = 0.8
) -> list[DiarSegment]:
    return [s for s in segments if (s.end - s.start) >= min_dur]


def slice_audio(
    wav: np.ndarray, rate: int, start: float, end: float
) -> np.ndarray:
    return wav[int(round(start * rate)) : int(round(end * rate))]


def embed_turns(
    extractor: Extractor,
    wav: np.ndarray,
    rate: int,
    segments: Sequence[DiarSegment],
) -> list[Turn]:
    _check_dim(extractor)
    turns: list[Turn] = []
    for i, seg in enumerate(segments):
        vec = extractor.embed(slice_audio(wav, rate, seg.start, seg.end))
        turns.append((f"turn_{i}", np.asarray(vec, dtype=np.float32), seg.speaker))
    return turns


def cluster_centroids(
    extractor: Extractor,
    wav: np.ndarray,
    rate: int,
    segments_by_cluster: Mapping[str, Sequence[DiarSegment]],
) -> list[Centroid]:
    _check_dim(extractor)
    centroids: list[Centroid] = []
    for cluster_id, segs in segments_by_cluster.items():
        pieces = [slice_audio(wav, rate, s.start, s.end) for s in segs]
        concatenated = (
            np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
        )
        vec = extractor.embed(concatenated)
        centroids.append((cluster_id, np.asarray(vec, dtype=np.float32)))
    return centroids


class SpeakerEmbedder:
    """sherpa-onnx SpeakerEmbeddingExtractor wrapper (CAM++, 192-dim)."""

    def __init__(self, model_path: str, sample_rate: int = 16000) -> None:
        import sherpa_onnx

        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=model_path,
            num_threads=1,
            provider="cpu",
            debug=False,
        )
        self._ext = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        self._rate = sample_rate

    @property
    def dim(self) -> int:
        return self._ext.dim

    def embed(self, samples: np.ndarray) -> np.ndarray:
        stream = self._ext.create_stream()
        stream.accept_waveform(
            sample_rate=self._rate, waveform=np.asarray(samples, dtype=np.float32)
        )
        stream.input_finished()
        return np.asarray(self._ext.compute(stream), dtype=np.float32)
