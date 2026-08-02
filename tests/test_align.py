"""Word -> speaker assignment via max temporal overlap (what-we-build.md §4.1)."""

from meetscribe.align import assign_words_to_speakers
from meetscribe.types import DiarSegment, Word


def _spk(utts):
    return [u.speaker for u in utts]


def test_word_fully_inside_one_segment():
    words = [Word("hi", 1.0, 1.4)]
    diar = [DiarSegment(0.0, 5.0, "spk_0")]
    utts = assign_words_to_speakers(words, diar)
    assert len(utts) == 1
    assert utts[0].speaker == "spk_0"
    assert utts[0].track == "system"
    assert utts[0].text == "hi"
    assert utts[0].start == 1.0 and utts[0].end == 1.4


def test_word_straddling_two_segments_bigger_overlap_wins():
    # word [1.0,2.0]: 0.3 in spk_0 (..1.3), 0.7 in spk_1 (1.3..) -> spk_1
    words = [Word("x", 1.0, 2.0)]
    diar = [DiarSegment(0.0, 1.3, "spk_0"), DiarSegment(1.3, 3.0, "spk_1")]
    assert _spk(assign_words_to_speakers(words, diar)) == ["spk_1"]


def test_tie_goes_to_earliest_segment():
    # word [1.0,2.0]: exactly 0.5 in each -> earliest (spk_0)
    words = [Word("x", 1.0, 2.0)]
    diar = [DiarSegment(0.0, 1.5, "spk_0"), DiarSegment(1.5, 3.0, "spk_1")]
    assert _spk(assign_words_to_speakers(words, diar)) == ["spk_0"]


def test_word_before_all_segments_assigned_nearest():
    words = [Word("x", 0.0, 0.2)]
    diar = [DiarSegment(1.0, 2.0, "spk_0"), DiarSegment(5.0, 6.0, "spk_1")]
    assert _spk(assign_words_to_speakers(words, diar)) == ["spk_0"]


def test_consecutive_same_speaker_words_merge_into_one_utterance():
    words = [Word("hello", 1.0, 1.4), Word("world", 1.5, 2.0)]
    diar = [DiarSegment(0.0, 5.0, "spk_0")]
    utts = assign_words_to_speakers(words, diar)
    assert len(utts) == 1
    assert utts[0].text == "hello world"
    assert utts[0].start == 1.0 and utts[0].end == 2.0
    assert len(utts[0].words) == 2


def test_speaker_change_splits_utterances():
    words = [Word("a", 0.1, 0.4), Word("b", 2.1, 2.4)]
    diar = [DiarSegment(0.0, 1.0, "spk_0"), DiarSegment(2.0, 3.0, "spk_1")]
    utts = assign_words_to_speakers(words, diar)
    assert _spk(utts) == ["spk_0", "spk_1"]
    assert [u.text for u in utts] == ["a", "b"]


def test_empty_words_returns_empty():
    assert assign_words_to_speakers([], [DiarSegment(0.0, 1.0, "spk_0")]) == []
