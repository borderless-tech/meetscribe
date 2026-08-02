"""Assign each word to a speaker by maximum temporal overlap (what-we-build.md §4.1).

Standard WhisperX-style procedure: for every word, pick the diarization segment it overlaps
most; ties go to the earliest segment; a word overlapping nothing is assigned to the nearest
segment (so no words are dropped). Consecutive words with the same speaker collapse into one
utterance.
"""

from __future__ import annotations

from .types import DiarSegment, Utterance, Word


def _overlap(w: Word, d: DiarSegment) -> float:
    return max(0.0, min(w.end, d.end) - max(w.start, d.start))


def _gap(w: Word, d: DiarSegment) -> float:
    """Distance between a word and a non-overlapping segment (0 if they touch/overlap)."""
    if w.end < d.start:
        return d.start - w.end
    if d.end < w.start:
        return w.start - d.end
    return 0.0


def _best_speaker(w: Word, diar: list[DiarSegment]) -> str:
    best_idx = 0
    best_overlap = _overlap(w, diar[0])
    for i in range(1, len(diar)):
        ov = _overlap(w, diar[i])
        if ov > best_overlap:  # strict > keeps the earliest segment on ties
            best_overlap = ov
            best_idx = i
    if best_overlap > 0.0:
        return diar[best_idx].speaker
    # No overlap with any segment: assign the nearest by gap (earliest on ties).
    nearest_idx = min(range(len(diar)), key=lambda i: _gap(w, diar[i]))
    return diar[nearest_idx].speaker


def assign_words_to_speakers(
    words: list[Word], diar: list[DiarSegment]
) -> list[Utterance]:
    if not words or not diar:
        return []

    labelled = [(w, _best_speaker(w, diar)) for w in words]

    utterances: list[Utterance] = []
    run: list[Word] = []
    run_speaker: str | None = None

    def flush() -> None:
        if not run:
            return
        assert run_speaker is not None
        utterances.append(
            Utterance(
                start=run[0].start,
                end=run[-1].end,
                speaker=run_speaker,
                track="system",
                text=" ".join(w.w for w in run),
                words=tuple(run),
            )
        )

    for w, spk in labelled:
        if spk != run_speaker and run:
            flush()
            run = []
        run_speaker = spk
        run.append(w)
    flush()
    return utterances
