"""Merge + rank broken-word correction candidates for human review."""

from meetscribe.suggest import Suggestion, gather_candidates


def test_ranks_sound_alikes_first_and_dedups():
    # hunspell offers several; the closest-sounding to "geernt" should lead.
    s = gather_candidates("geernt", ["geerbt", "gelernt", "gern", "geerntet"], max_n=5)
    assert isinstance(s, Suggestion) and s.original == "geernt"
    assert s.candidates[0] in ("gelernt", "geerntet", "geerbt")  # a near sound-alike, not "gern"
    assert "gern" in s.candidates  # still offered, just ranked lower
    assert len(s.candidates) == len(set(c.lower() for c in s.candidates))  # deduped


def test_merges_llm_suggestion_with_hunspell():
    s = gather_candidates("vorhandliche", ["vorbildliche", "handliche"],
                          llm_suggestion="vorhandene", max_n=5)
    assert "vorhandene" in s.candidates  # LLM candidate included
    assert "vorbildliche" in s.candidates


def test_drops_the_original_and_empty():
    s = gather_candidates("relev", ["relev", "relevant", ""], max_n=5)
    assert "relev" not in s.candidates and "" not in s.candidates
    assert s.candidates == ["relevant"]


def test_caps_at_max_n():
    s = gather_candidates("x", ["aa", "bb", "cc", "dd", "ee", "ff"], max_n=3)
    assert len(s.candidates) == 3


def test_no_candidates_yields_empty():
    s = gather_candidates("Backcounter", [], llm_suggestion=None, max_n=5)
    assert s.candidates == []
