"""Merge the two tracks into one timeline (what-we-build.md §4.2)."""

from meetscribe.merge import coalesce_utterances, merge_tracks
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


# ---- coalesce_utterances ------------------------------------------------------------

def test_coalesce_merges_consecutive_same_speaker():
    # a single speaker's VAD fragments (small gaps) become one readable utterance.
    utts = [
        _u(0.0, 1.0, "me", "mic", "Hallo"),
        _u(1.5, 2.5, "me", "mic", "wie"),
        _u(3.0, 4.0, "me", "mic", "geht's"),
    ]
    out = coalesce_utterances(utts, max_gap=2.0)
    assert len(out) == 1
    assert out[0].start == 0.0 and out[0].end == 4.0
    assert out[0].text == "Hallo wie geht's"
    assert [w.w for w in out[0].words] == ["Hallo", "wie", "geht's"]
    assert out[0].speaker == "me" and out[0].track == "mic"


def test_coalesce_does_not_cross_another_speaker():
    # a remote interjection between two "me" fragments must keep them separate,
    # so the conversation order is preserved.
    utts = [
        _u(0.0, 1.0, "me", "mic", "ich"),
        _u(1.2, 2.0, "spk_0", "system", "moment"),
        _u(2.2, 3.0, "me", "mic", "sage"),
    ]
    out = coalesce_utterances(utts, max_gap=2.0)
    assert [u.text for u in out] == ["ich", "moment", "sage"]


def test_coalesce_breaks_on_large_gap():
    utts = [
        _u(0.0, 1.0, "me", "mic", "erst"),
        _u(8.0, 9.0, "me", "mic", "später"),  # 7 s gap → a real turn break
    ]
    out = coalesce_utterances(utts, max_gap=2.0)
    assert [u.text for u in out] == ["erst", "später"]


def test_coalesce_same_speaker_different_track_not_merged():
    utts = [
        _u(0.0, 1.0, "spk_0", "system", "a"),
        _u(1.2, 2.0, "spk_0", "mic", "b"),  # contrived: same label, other track
    ]
    out = coalesce_utterances(utts, max_gap=2.0)
    assert len(out) == 2


def test_coalesce_single_and_empty_pass_through():
    assert coalesce_utterances([]) == []
    one = [_u(0.0, 1.0, "me", "mic", "hi")]
    assert coalesce_utterances(one) == one


def test_coalesce_merges_system_speaker_too():
    utts = [
        _u(0.0, 2.0, "spk_1", "system", "Also"),
        _u(2.5, 4.0, "spk_1", "system", "ich denke"),
    ]
    out = coalesce_utterances(utts, max_gap=2.0)
    assert len(out) == 1 and out[0].text == "Also ich denke"
