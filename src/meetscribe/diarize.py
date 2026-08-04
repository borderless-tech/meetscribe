"""Speaker diarization on the system track via sherpa-onnx OfflineSpeakerDiarization.

Only the int-id → ``spk_N`` mapping is pure/tested; the sherpa diarizer sits behind a protocol.
Clustering defaults to ``num_clusters=-1`` + a threshold (agglomerative, no fixed K — see §12);
a known speaker count (``num_clusters > 0``) forces exactly that many clusters and makes
sherpa ignore the threshold.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from .types import DiarSegment


class _RawSeg(Protocol):
    start: float
    end: float
    speaker: int


class Diarizer(Protocol):
    def segments(self, samples: np.ndarray, on_progress=None) -> Sequence[_RawSeg]: ...


def to_diar_segments(raw: Sequence[_RawSeg]) -> list[DiarSegment]:
    return [DiarSegment(s.start, s.end, f"spk_{s.speaker}") for s in raw]


def run(
    diarizer: Diarizer, samples: np.ndarray, on_progress=None
) -> list[DiarSegment]:
    """``on_progress(processed_chunks, total_chunks)`` is called as the backend advances."""
    return to_diar_segments(diarizer.segments(samples, on_progress=on_progress))


class OfflineDiarizer:
    """sherpa-onnx OfflineSpeakerDiarization wrapper (pyannote seg + CAM++ embedding)."""

    def __init__(
        self,
        segmentation_model: str,
        embedding_model: str,
        threshold: float = 0.5,
        min_duration_on: float = 0.3,
        min_duration_off: float = 0.5,
        num_clusters: int = -1,
    ) -> None:
        import sherpa_onnx

        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=segmentation_model,
                ),
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=embedding_model,
                num_threads=1,
                provider="cpu",
            ),
            clustering=sherpa_onnx.FastClusteringConfig(
                num_clusters=num_clusters,
                threshold=threshold,
            ),
            min_duration_on=min_duration_on,
            min_duration_off=min_duration_off,
        )
        self._sd = sherpa_onnx.OfflineSpeakerDiarization(config)

    def segments(self, samples: np.ndarray, on_progress=None):
        callback = None
        if on_progress is not None:

            def callback(processed: int, total: int) -> int:
                on_progress(processed, total)
                return 0  # non-zero would abort the diarization

        result = self._sd.process(
            np.asarray(samples, dtype=np.float32), callback=callback
        ).sort_by_start_time()
        return list(result)
