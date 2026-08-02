"""ASR: SentencePiece token -> word grouping + chunk-offset application.

Parakeet-TDT returns per-token text/timestamps/durations; r.words is often empty for subword
models, so words are rebuilt from the ``▁`` (SentencePiece space) word-boundary marker.
"""

import numpy as np

from meetscribe.asr import RawResult, tokens_to_words, transcribe_chunks
from meetscribe.vad import Chunk

MARK = "▁"  # ▁


def test_tokens_to_words_groups_on_boundary_marker():
    tokens = [f"{MARK}he", "llo", f"{MARK}world"]
    ts = [0.0, 0.2, 0.5]
    dur = [0.2, 0.3, 0.4]
    words = tokens_to_words(tokens, ts, dur)
    assert [w.w for w in words] == ["hello", "world"]
    assert words[0].start == 0.0 and words[0].end == 0.5  # 0.2 + 0.3
    assert words[1].start == 0.5 and words[1].end == 0.9  # 0.5 + 0.4


def test_single_word_multiple_pieces():
    words = tokens_to_words([f"{MARK}un", "believ", "able"], [0.0, 0.1, 0.2], [0.1, 0.1, 0.3])
    assert len(words) == 1
    assert words[0].w == "unbelievable"
    assert words[0].start == 0.0 and words[0].end == 0.5


def test_leading_token_without_marker_starts_a_word():
    words = tokens_to_words(["hi", f"{MARK}there"], [0.0, 0.4], [0.2, 0.3])
    assert [w.w for w in words] == ["hi", "there"]


def test_empty_tokens():
    assert tokens_to_words([], [], []) == []


def test_transcribe_chunks_applies_offset():
    class FakeRecognizer:
        def recognize(self, samples):
            return RawResult(
                text="hello world",
                tokens=[f"{MARK}he", "llo", f"{MARK}world"],
                timestamps=[0.0, 0.2, 0.5],
                durations=[0.2, 0.3, 0.4],
            )

    chunk = Chunk(offset=10.0, samples=np.zeros(16000, dtype=np.float32), sample_rate=16000)
    segs = transcribe_chunks(FakeRecognizer(), [chunk])
    assert len(segs) == 1
    assert segs[0].text == "hello world"
    assert segs[0].words[0].start == 10.0  # 0.0 + offset
    assert segs[0].words[1].end == 10.9  # 0.5 + 0.4 + offset
    assert segs[0].start == 10.0 and segs[0].end == 10.9


def test_transcribe_chunks_calls_on_advance_once_per_chunk():
    class R:
        def recognize(self, samples):
            return RawResult("hi", [f"{MARK}hi"], [0.0], [0.2])

    chunks = [
        Chunk(0.0, np.zeros(10, dtype=np.float32), 16000),
        Chunk(1.0, np.zeros(10, dtype=np.float32), 16000),
    ]
    calls = []
    transcribe_chunks(R(), chunks, on_advance=lambda: calls.append(1))
    assert len(calls) == 2


def test_transcribe_skips_silent_chunk_with_no_words():
    class Silent:
        def recognize(self, samples):
            return RawResult(text="", tokens=[], timestamps=[], durations=[])

    chunk = Chunk(offset=0.0, samples=np.zeros(10, dtype=np.float32), sample_rate=16000)
    assert transcribe_chunks(Silent(), [chunk]) == []
