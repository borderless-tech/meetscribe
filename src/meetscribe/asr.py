"""ASR via sherpa-onnx Parakeet-TDT-v3, with word reconstruction from subword tokens.

The recognizer returns per-token text/timestamps/durations. ``result.words`` is usually empty for
subword models, so words are rebuilt here from the SentencePiece word-boundary marker ``▁``. The
recognizer itself sits behind the :class:`Recognizer` protocol so the grouping logic is unit-tested
with a fake; the real :class:`ParakeetRecognizer` is exercised by the end-to-end smoke test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .types import Segment, Word
from .vad import Chunk

WORD_MARKER = "▁"  # ▁ SentencePiece space marker


@dataclass
class RawResult:
    text: str
    tokens: list[str]
    timestamps: list[float]
    durations: list[float]


class Recognizer(Protocol):
    def recognize(self, samples: np.ndarray) -> RawResult: ...


def tokens_to_words(
    tokens: Sequence[str],
    timestamps: Sequence[float],
    durations: Sequence[float],
) -> list[Word]:
    """Group subword tokens into words on the ``▁`` boundary marker.

    A word's start is its first token's timestamp; its end is the last token's timestamp plus that
    token's duration.
    """
    words: list[Word] = []
    cur_text = ""
    cur_start = 0.0
    cur_end = 0.0
    have = False

    def flush() -> None:
        nonlocal have
        if have:
            words.append(Word(cur_text, cur_start, cur_end))
            have = False

    for tok, ts, dur in zip(tokens, timestamps, durations):
        if tok.startswith(WORD_MARKER) or not have:
            flush()
            cur_text = tok.lstrip(WORD_MARKER)
            cur_start = ts
            cur_end = ts + dur
            have = True
        else:
            cur_text += tok
            cur_end = ts + dur
    flush()
    return words


def transcribe_chunks(recognizer: Recognizer, chunks: Sequence[Chunk]) -> list[Segment]:
    """Recognize each VAD chunk and shift its word timestamps by the chunk's absolute offset."""
    segments: list[Segment] = []
    for chunk in chunks:
        res = recognizer.recognize(chunk.samples)
        local_words = tokens_to_words(res.tokens, res.timestamps, res.durations)
        if not local_words:
            continue  # silence / no decoded words
        words = tuple(
            Word(w.w, w.start + chunk.offset, w.end + chunk.offset) for w in local_words
        )
        segments.append(
            Segment(
                start=words[0].start,
                end=words[-1].end,
                text=res.text,
                words=words,
            )
        )
    return segments


class ParakeetRecognizer:
    """sherpa-onnx OfflineRecognizer for a Parakeet-TDT (NeMo transducer) model."""

    def __init__(self, model_dir: str, num_threads: int = 4) -> None:
        import os

        import sherpa_onnx

        self._sherpa = sherpa_onnx
        self._rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=os.path.join(model_dir, "encoder.int8.onnx"),
            decoder=os.path.join(model_dir, "decoder.int8.onnx"),
            joiner=os.path.join(model_dir, "joiner.int8.onnx"),
            tokens=os.path.join(model_dir, "tokens.txt"),
            num_threads=num_threads,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
            model_type="nemo_transducer",
            provider="cpu",
            debug=False,
        )

    def recognize(self, samples: np.ndarray) -> RawResult:
        stream = self._rec.create_stream()
        stream.accept_waveform(16000, np.asarray(samples, dtype=np.float32))
        self._rec.decode_stream(stream)
        r = stream.result
        return RawResult(
            text=r.text,
            tokens=list(r.tokens),
            timestamps=list(r.timestamps),
            durations=list(r.durations),
        )
