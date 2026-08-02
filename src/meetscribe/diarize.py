"""Speaker diarization on the system track via sherpa-onnx OfflineSpeakerDiarization.

Only the int-id → ``spk_N`` mapping is pure/tested; the sherpa diarizer sits behind a protocol.
Clustering uses ``num_clusters=-1`` + a threshold (agglomerative, no fixed K — see §12).
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
    def segments(self, samples: np.ndarray) -> Sequence[_RawSeg]: ...


def to_diar_segments(raw: Sequence[_RawSeg]) -> list[DiarSegment]:
    return [DiarSegment(s.start, s.end, f"spk_{s.speaker}") for s in raw]


def run(diarizer: Diarizer, samples: np.ndarray) -> list[DiarSegment]:
    return to_diar_segments(diarizer.segments(samples))


class OfflineDiarizer:
    """sherpa-onnx OfflineSpeakerDiarization wrapper (pyannote seg + CAM++ embedding)."""

    def __init__(
        self,
        segmentation_model: str,
        embedding_model: str,
        threshold: float = 0.5,
        min_duration_on: float = 0.3,
        min_duration_off: float = 0.5,
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
                num_clusters=-1,
                threshold=threshold,
            ),
            min_duration_on=min_duration_on,
            min_duration_off=min_duration_off,
        )
        self._sd = sherpa_onnx.OfflineSpeakerDiarization(config)

    def segments(self, samples: np.ndarray):
        result = self._sd.process(
            np.asarray(samples, dtype=np.float32), callback=None
        ).sort_by_start_time()
        return list(result)
