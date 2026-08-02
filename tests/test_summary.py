"""summarize(Result) -> Summary: per-speaker talk time, ordering."""

from meetscribe.pipeline import Result, summarize
from meetscribe.types import Utterance, Word


def _u(start, end, speaker, track):
    return Utterance(start, end, speaker, track, "x", (Word("x", start, end),))


def test_summarize_counts_segments_and_duration():
    utts = [_u(0, 10, "me", "mic"), _u(10, 15, "spk_0", "system")]
    s = summarize(Result(utts, [], [], 192, 15.0))
    assert s.n_segments == 2
    assert s.duration_s == 15.0


def test_summarize_talk_time_per_speaker_ordered_desc():
    utts = [
        _u(0, 10, "me", "mic"),          # me: 10
        _u(10, 15, "spk_0", "system"),   # spk_0: 5 + 5 = 10
        _u(20, 25, "spk_0", "system"),
        _u(30, 33, "spk_1", "system"),   # spk_1: 3
    ]
    s = summarize(Result(utts, [], [], 192, 33.0))
    # ordered by talk time desc; me and spk_0 tie at 10 → stable (me first seen)
    assert [sp for sp, _, _ in s.speakers] == ["me", "spk_0", "spk_1"]
    talk = {sp: t for sp, t, _ in s.speakers}
    assert talk["spk_0"] == 10.0 and talk["spk_1"] == 3.0
    track = {sp: tr for sp, _, tr in s.speakers}
    assert track["me"] == "mic" and track["spk_0"] == "system"


def test_summarize_empty():
    s = summarize(Result([], [], [], 192, 0.0))
    assert s.n_segments == 0 and s.speakers == []
