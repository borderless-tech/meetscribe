"""Acoustic-closeness guardrail for ASR word corrections (phonetic.py).

The point of these tests: a garbled ASR word may have several dictionary
candidates, and we must prefer the one that actually *sounds* like what was
heard — never a confidently-invented but acoustically-unrelated word.
"""

from meetscribe.phonetic import (
    accept_correction,
    acoustic_distance,
    koelner_phonetik,
    levenshtein,
    rank_candidates,
)


# --- Kölner Phonetik: known reference codes -------------------------------


def test_koelner_known_codes():
    # Textbook examples of the standard German Kölner Phonetik.
    assert koelner_phonetik("Müller") == "657"
    assert koelner_phonetik("Wikipedia") == "3412"
    assert koelner_phonetik("Meyer") == "67"
    assert koelner_phonetik("Schmidt") == "862"


def test_koelner_case_insensitive():
    assert koelner_phonetik("müller") == koelner_phonetik("MÜLLER")


def test_koelner_empty_is_empty():
    assert koelner_phonetik("") == ""


def test_koelner_collapses_repeated_codes():
    # Doubled letters map to one code digit (no repeated adjacent digits).
    assert koelner_phonetik("Wassermann") == koelner_phonetik("Wasermann")


# --- Levenshtein ----------------------------------------------------------


def test_levenshtein_identical_is_zero():
    assert levenshtein("erster", "erster") == 0


def test_levenshtein_empty():
    assert levenshtein("", "abc") == 3
    assert levenshtein("abc", "") == 3


def test_levenshtein_single_edits():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("Ertier", "Erster") == 2


def test_levenshtein_symmetric():
    assert levenshtein("Ertier", "Erster") == levenshtein("Erster", "Ertier")


# --- acoustic_distance / rank / accept ------------------------------------


def test_acoustic_distance_identical_is_zero():
    assert acoustic_distance("Erster", "Erster") == 0.0


def test_near_word_is_closer_than_distant_word():
    # "Ertier" is a garbled "Erster"; "Amtstraeger" sounds nothing like it.
    near = acoustic_distance("Ertier", "Erster")
    far = acoustic_distance("Ertier", "Amtstraeger")
    assert near < far


def test_rank_candidates_orders_by_closeness():
    ranked = rank_candidates("Ertier", ["Amtstraeger", "Erster", "Erdteil"])
    assert ranked[0] == "Erster"
    assert ranked[-1] == "Amtstraeger"


def test_rank_candidates_empty():
    assert rank_candidates("Ertier", []) == []


def test_accept_near_reject_far():
    near = acoustic_distance("Ertier", "Erster")
    far = acoustic_distance("Ertier", "Amtstraeger")
    # A threshold between the two accepts the near word, rejects the far one.
    threshold = (near + far) / 2
    assert accept_correction("Ertier", "Erster", threshold) is True
    assert accept_correction("Ertier", "Amtstraeger", threshold) is False
