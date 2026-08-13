"""Bilingual known-word check + broken-word flagging (spell-checkers are injected)."""

from meetscribe.lexicon import Lexicon


def _lex(de=(), en=(), glossary=()):
    de_set = {w.lower() for w in de}
    en_set = {w.lower() for w in en}
    return Lexicon(
        spell_de=lambda w: w.lower() in de_set,
        spell_en=lambda w: w.lower() in en_set,
        glossary=list(glossary),
    )


def test_german_word_known_via_spell_de():
    lex = _lex(de=["Vielen"])
    assert lex.is_known("Vielen") is True


def test_english_word_known_via_spell_en_not_de():
    lex = _lex(en=["LinkedIn"])
    assert lex.is_known("LinkedIn") is True


def test_glossary_match_is_case_insensitive():
    lex = _lex(glossary=["Borderless"])
    assert lex.is_known("borderless") is True
    assert lex.is_known("BORDERLESS") is True


def test_surrounding_punctuation_is_stripped_before_lookup():
    lex = _lex(de=["Hallo"])
    assert lex.is_known("Hallo,") is True
    assert lex.is_known("(Hallo)") is True


def test_garbled_word_unknown_to_both_is_not_known():
    lex = _lex(de=["Hallo"], en=["hello"])
    assert lex.is_known("Halllooo") is False


def test_flag_broken_flags_only_the_garbled_token():
    lex = _lex(de=["Vielen", "Dank"], en=["hello"])
    words = ["Vielen", "Dank", "Xqwrz"]
    assert lex.flag_broken(words) == [2]


def test_flag_broken_skips_punctuation_numbers_and_short_tokens():
    lex = _lex(de=["Hallo"])
    # index 1 "." punctuation, 2 "42" number, 3 "ok" too short -> only 4 "Zzzqx" flagged
    words = ["Hallo", ".", "42", "ok", "Zzzqx"]
    assert lex.flag_broken(words) == [4]


def test_flag_broken_mixed_bilingual_list_returns_correct_indices():
    lex = _lex(de=["Wir", "nutzen"], en=["LinkedIn", "meeting"])
    words = ["Wir", "nutzen", "LinkedIn", "im", "meeting", "Grbldxz"]
    # "im" is short (<3), the rest are known except the last garbled token
    assert lex.flag_broken(words) == [5]


def test_flag_broken_empty_list():
    lex = _lex()
    assert lex.flag_broken([]) == []
