"""Merge the two tracks into one timeline (what-we-build.md §4.2)."""

from meetscribe.merge import merge_tracks
from meetscribe.types import Utterance, Word


def _u(start, end, speaker="spk_0", track="system", text="x"):
    return Utterance(start, end, speaker, track, text, (Word(text, start, end),))


def test_interleaves_by_start():
    mic = [_u(0.0, 1.0, "me", "mic", "a"), _u(4.0, 5.0, "me", "mic", "c")]
    system = [_u(2.0, 3.0, "spk_0", "system", "b")]
    order = [u.text for u in merge_tracks(mic, system)]
    assert order == ["a", "b", "c"]


def test_mic_speaker_is_always_me_and_track_mic():
    # even if a mic utterance arrives mislabelled, merge forces me/mic.
    mic = [_u(0.0, 1.0, "spk_9", "system", "a")]
    out = merge_tracks(mic, [])
    assert out[0].speaker == "me"
    assert out[0].track == "mic"


def test_stable_tie_break_mic_before_system():
    mic = [_u(1.0, 2.0, "me", "mic", "m")]
    system = [_u(1.0, 2.0, "spk_0", "system", "s")]
    assert [u.text for u in merge_tracks(mic, system)] == ["m", "s"]


def test_empty_mic():
    system = [_u(1.0, 2.0, "spk_0", "system", "s")]
    assert [u.text for u in merge_tracks([], system)] == ["s"]


def test_empty_system():
    mic = [_u(1.0, 2.0, "me", "mic", "m")]
    assert [u.text for u in merge_tracks(mic, [])] == ["m"]


def test_system_utterances_pass_through_unchanged():
    system = [_u(1.0, 2.0, "spk_3", "system", "s")]
    out = merge_tracks([], system)
    assert out[0].speaker == "spk_3" and out[0].track == "system"
