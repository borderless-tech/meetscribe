"""Repair only the flagged broken words, each fix double-guarded, timestamps kept."""

from meetscribe.repair import is_valid_correction, repair_words
from meetscribe.types import Word


def _w(text, start, end):
    return Word(text, start, end)


# ---- is_valid_correction (the tightened, double guard) ------------------------------

def test_accepts_close_and_known():
    # acoustically close AND a real dictionary word → accepted.
    assert is_valid_correction("Erter", "Erster", max_distance=0.5, is_known=lambda w: True)


def test_rejects_close_but_unknown():
    # the Ertier → Erter case: acoustically near-identical (dist ~0.05) but NOT a
    # real word, so the dictionary check vetoes it. This is why is_known exists.
    assert not is_valid_correction("Ertier", "Erter", max_distance=0.5, is_known=lambda w: False)


def test_rejects_far_even_if_known():
    # a fluent, real word that sounds nothing like what was heard must be rejected.
    assert not is_valid_correction(
        "Erter", "Fahrrad", max_distance=0.1, is_known=lambda w: True
    )


def test_both_guards_must_pass_close_unknown_and_far_known_both_rejected():
    # neither guard alone is sufficient.
    assert not is_valid_correction("Erter", "Erster", max_distance=0.5, is_known=lambda w: False)
    assert not is_valid_correction("Erter", "Zebra", max_distance=0.01, is_known=lambda w: True)


# ---- repair_words -------------------------------------------------------------------

def _accept_all(asr, cand):
    return True


def _reject_all(asr, cand):
    return False


def test_applies_accepted_fix_keeping_timestamps():
    words = [_w("Ertier", 1.0, 1.5)]
    correct = lambda i, ws: "Erster"
    new, n = repair_words(words, [0], correct_fn=correct, accept_fn=_accept_all)
    assert n == 1
    assert new[0].w == "Erster"
    # timestamps preserved exactly.
    assert new[0].start == 1.0 and new[0].end == 1.5


def test_keeps_original_when_rejected():
    words = [_w("Ertier", 1.0, 1.5)]
    correct = lambda i, ws: "Erster"
    new, n = repair_words(words, [0], correct_fn=correct, accept_fn=_reject_all)
    assert n == 0
    assert new[0] == words[0]  # untouched, same text + timestamps


def test_empty_flagged_returns_input_and_zero():
    words = [_w("hallo", 0.0, 0.5), _w("welt", 0.6, 1.0)]
    new, n = repair_words(words, [], correct_fn=lambda i, ws: "x", accept_fn=_accept_all)
    assert n == 0
    assert new == words


def test_non_flagged_words_untouched_and_order_preserved():
    words = [_w("a", 0.0, 0.5), _w("bad", 0.6, 1.0), _w("c", 1.1, 1.5)]
    correct = lambda i, ws: "fixed"
    new, n = repair_words(words, [1], correct_fn=correct, accept_fn=_accept_all)
    assert n == 1
    assert [w.w for w in new] == ["a", "fixed", "c"]
    # the untouched neighbours are the very same objects.
    assert new[0] == words[0] and new[2] == words[2]
    # the repaired word kept its own timestamps.
    assert new[1].start == 0.6 and new[1].end == 1.0


def test_counts_multiple_fixes():
    words = [_w("x", 0.0, 0.5), _w("y", 0.6, 1.0), _w("z", 1.1, 1.5)]
    new, n = repair_words(words, [0, 2], correct_fn=lambda i, ws: "!", accept_fn=_accept_all)
    assert n == 2
    assert [w.w for w in new] == ["!", "y", "!"]


def test_correct_fn_receives_index_and_full_word_list():
    # the LLM span corrector needs surrounding context, so it is handed the position
    # and the whole word list, not just the single broken token.
    words = [_w("guten", 0.0, 0.4), _w("Tug", 0.5, 0.9), _w("alle", 1.0, 1.4)]
    seen = {}

    def correct(i, ws):
        seen["i"] = i
        seen["ws"] = ws
        return "Tag"

    new, n = repair_words(words, [1], correct_fn=correct, accept_fn=_accept_all)
    assert seen["i"] == 1
    assert seen["ws"] is words  # full list passed through
    assert new[1].w == "Tag" and n == 1


def test_mixed_accept_and_reject_uses_asr_and_candidate():
    # accept_fn is called with (original word text, candidate) so both guards apply.
    words = [_w("aaa", 0.0, 0.5), _w("bbb", 0.6, 1.0)]
    calls = []

    def accept(asr, cand):
        calls.append((asr, cand))
        return asr == "aaa"  # accept only the first

    new, n = repair_words(words, [0, 1], correct_fn=lambda i, ws: "C", accept_fn=accept)
    assert n == 1
    assert [w.w for w in new] == ["C", "bbb"]
    assert calls == [("aaa", "C"), ("bbb", "C")]
