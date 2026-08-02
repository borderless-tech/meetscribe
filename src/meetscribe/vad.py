"""Voice-activity detection: split a track into speech chunks with absolute offsets.

The offset bookkeeping is pure and unit-tested (:func:`segments_to_chunks`); the sherpa-onnx
Silero backend (:class:`SileroVad`) is a thin wrapper exercised by the end-to-end smoke test.
A sherpa ``SpeechSegment`` exposes ``.start`` (int sample index) and ``.samples`` only — there is
no ``.end`` — so the chunk end is derived from the sample count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np


@dataclass
class Chunk:
    """A contiguous span of detected speech, positioned on the original timeline."""

    offset: float  # absolute start time in seconds
    samples: np.ndarray  # float32 mono audio
    sample_rate: int

    @property
    def start(self) -> float:
        return self.offset

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate

    @property
    def end(self) -> float:
        return self.offset + self.duration


class _VadSegment(Protocol):
    start: int  # sample index
    samples: np.ndarray


def segments_to_chunks(
    segments: Iterable[_VadSegment], sample_rate: int
) -> list[Chunk]:
    """Map raw VAD segments to :class:`Chunk`s, recomputing absolute offsets from sample index."""
    return [
        Chunk(
            offset=seg.start / sample_rate,
            samples=np.asarray(seg.samples, dtype=np.float32),
            sample_rate=sample_rate,
        )
        for seg in segments
    ]


class SileroVad:
    """sherpa-onnx Silero VAD wrapper. Constructed from the pinned silero_vad.onnx model."""

    def __init__(
        self,
        model_path: str,
        sample_rate: int = 16000,
        window_size: int = 512,
        threshold: float = 0.5,
        min_silence_duration: float = 0.25,
        min_speech_duration: float = 0.25,
        max_speech_duration: float = 20.0,
    ) -> None:
        import sherpa_onnx

        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = model_path
        config.silero_vad.threshold = threshold
        config.silero_vad.min_silence_duration = min_silence_duration
        config.silero_vad.min_speech_duration = min_speech_duration
        config.silero_vad.max_speech_duration = max_speech_duration
        config.silero_vad.window_size = window_size
        config.sample_rate = sample_rate
        config.num_threads = 1
        config.provider = "cpu"
        self._config = config
        self._sample_rate = sample_rate
        self._window = window_size
        self._sherpa = sherpa_onnx

    def chunks(self, samples: np.ndarray) -> list[Chunk]:
        samples = np.asarray(samples, dtype=np.float32)
        vad = self._sherpa.VoiceActivityDetector(
            self._config, buffer_size_in_seconds=100
        )
        collected: list[Chunk] = []
        for i in range(0, len(samples), self._window):
            vad.accept_waveform(samples[i : i + self._window])
            while not vad.empty():
                collected.extend(segments_to_chunks([vad.front], self._sample_rate))
                vad.pop()
        vad.flush()
        while not vad.empty():
            collected.extend(segments_to_chunks([vad.front], self._sample_rate))
            vad.pop()
        return collected
