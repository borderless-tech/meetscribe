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
