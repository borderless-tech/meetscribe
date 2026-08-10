"""Merge the mic and system utterance lists into one timeline (what-we-build.md §4.2).

Mic utterances are the user by definition: they are hard-labelled ``speaker="me"`` /
``track="mic"`` and never touched by clustering. Both lists are combined and sorted by start
time; ties are broken deterministically with mic before system.
"""

from __future__ import annotations

from dataclasses import replace

from .types import Utterance


def merge_tracks(
    mic: list[Utterance], system: list[Utterance]
) -> list[Utterance]:
    forced_mic = [replace(u, speaker="me", track="mic") for u in mic]
    # rank 0 = mic, 1 = system → mic wins ties at equal start (stable within each track).
    tagged = [(u.start, 0, u) for u in forced_mic] + [
        (u.start, 1, u) for u in system
    ]
    tagged.sort(key=lambda t: (t[0], t[1]))
    return [u for _, _, u in tagged]


def coalesce_utterances(
    utterances: list[Utterance], max_gap: float = 2.0
) -> list[Utterance]:
    """Merge consecutive utterances of the same speaker+track into one paragraph.

    The input must be time-ordered (``merge_tracks`` output). Because it is ordered,
    two same-speaker utterances are only adjacent in the list when *no other speaker
    started between them* — so merging adjacent runs can never reorder the
    conversation (unlike widening the VAD silence window, which lets a speaker's
    segment swallow others' turns). A gap wider than ``max_gap`` seconds starts a new
    utterance, so genuine turn breaks are kept.
    """
    if not utterances:
        return []

    out: list[Utterance] = []
    group: list[Utterance] = [utterances[0]]

    def flush() -> None:
        if len(group) == 1:
            out.append(group[0])
            return
        first, last = group[0], group[-1]
        text = " ".join(u.text.strip() for u in group if u.text.strip())
        words = tuple(w for u in group for w in u.words)
        out.append(replace(first, end=last.end, text=text, words=words))

    for u in utterances[1:]:
        prev = group[-1]
        same_speaker = u.speaker == prev.speaker and u.track == prev.track
        if same_speaker and (u.start - prev.end) <= max_gap:
            group.append(u)
        else:
            flush()
            group = [u]
    flush()
    return out
