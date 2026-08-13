"""Flag-then-repair cleanup: NullCleaner, span-prompt, SpanRepairCleaner, managed fallback.

The LLM client and lexicon are injected, so no model/hunspell is needed here.
"""

from meetscribe.cleanup import (
    ManagedSpanRepairCleaner,
    NullCleaner,
    SpanRepairCleaner,
    SuggestCleaner,
    build_span_prompt,
)
from meetscribe.lexicon import Lexicon
from meetscribe.types import Utterance, Word


def _utt(text, words, track="system"):
    return Utterance(words[0].start, words[-1].end, "spk_0", track, text, tuple(words))


def _lex(known, glossary=()):
    kn = {w.lower() for w in known}
    return Lexicon(spell_de=lambda ws: {w for w in ws if w.lower() in kn},
                   spell_en=lambda ws: set(), glossary=list(glossary))


class FakeClient:
    """Returns a scripted correction based on the marked word in the span prompt."""

    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def complete(self, prompt, max_tokens=None):
        self.calls.append(prompt)
        for key, val in self.mapping.items():
            if f"«{key}" in prompt or f"«{key}»" in prompt:
                return val
        return "UNMATCHED"


class RecordingReporter:
    def __init__(self):
        self.warns = []
        self.infos = []

    def info(self, msg):
        self.infos.append(msg)

    def warn(self, msg):
        self.warns.append(msg)


# ---- NullCleaner --------------------------------------------------------------------

def test_null_cleaner_unchanged_and_inactive():
    u = _utt("hello world", [Word("hello", 0.0, 0.5), Word("world", 0.5, 1.0)])
    res = NullCleaner().clean([u], [], None)
    assert res.utterances == [u]
    assert res.active is False and res.fixed == 0 and res.echoes == 0


# ---- build_span_prompt --------------------------------------------------------------

def test_span_prompt_marks_word_with_context_and_glossary():
    words = [Word("du", 0, 0.3), Word("als", 0.4, 0.7), Word("Ertier", 0.8, 1.5),
             Word("hier", 1.6, 1.9)]
    p = build_span_prompt(words, 2, ["Borderless"])
    assert "«Ertier»" in p and "du als" in p and "hier" in p and "Borderless" in p


# ---- SpanRepairCleaner --------------------------------------------------------------

def test_repairs_flagged_word_preserving_timestamps():
    lex = _lex(known=["sagt", "Hallo", "bitte"])
    client = FakeClient({"Halllo": "Hallo"})
    u = _utt("sagt Halllo bitte",
             [Word("sagt", 0.0, 0.3), Word("Halllo", 0.4, 0.9), Word("bitte", 1.0, 1.3)])
    res = SpanRepairCleaner(client, lex).clean([u], [], None)
    out = res.utterances[0]
    assert out.words[1] == Word("Hallo", 0.4, 0.9)  # text fixed, timestamps preserved
    assert out.text == "sagt Hallo bitte"
    assert out.raw_text == "sagt Halllo bitte"
    assert res.fixed == 1 and res.active is True


def test_guard_rejects_acoustically_distant_correction():
    lex = _lex(known=["sagt", "Hallo", "bitte", "Tschüss"])
    client = FakeClient({"Halllo": "Tschüss"})  # a real word but nothing like "Halllo"
    u = _utt("sagt Halllo bitte",
             [Word("sagt", 0.0, 0.3), Word("Halllo", 0.4, 0.9), Word("bitte", 1.0, 1.3)])
    res = SpanRepairCleaner(client, lex).clean([u], [], None)
    assert res.utterances[0].words[1].w == "Halllo"  # kept raw
    assert res.fixed == 0


def test_guard_rejects_non_word_even_if_close():
    lex = _lex(known=["sagt", "bitte"])  # "Hallllo" is NOT known
    client = FakeClient({"Halllo": "Hallllo"})  # close but not a real word
    u = _utt("sagt Halllo bitte",
             [Word("sagt", 0.0, 0.3), Word("Halllo", 0.4, 0.9), Word("bitte", 1.0, 1.3)])
    res = SpanRepairCleaner(client, lex).clean([u], [], None)
    assert res.utterances[0].words[1].w == "Halllo"  # dictionary vetoes it
    assert res.fixed == 0


def test_collapses_echoes():
    lex = _lex(known=["und", "ja"])
    u = _utt("und und und ja",
             [Word("und", 0.0, 0.1), Word("und", 0.1, 0.2), Word("und", 0.2, 0.3),
              Word("ja", 0.4, 0.6)])
    res = SpanRepairCleaner(FakeClient({}), lex).clean([u], [], None)
    assert res.echoes == 2
    assert [w.w for w in res.utterances[0].words] == ["und", "ja"]


# ---- SuggestCleaner (suggest, don't apply) ------------------------------------------

def test_suggest_mode_flags_and_offers_candidates_without_applying():
    lex = _lex(known=["sagt", "bitte"])  # "Halllo" is broken
    suggest_fn = lambda ws: {w: ["Hallo", "Hallöchen"] for w in ws if w == "Halllo"}
    client = FakeClient({"Halllo": "Hallo"})
    u = _utt("sagt Halllo bitte",
             [Word("sagt", 0.0, 0.3), Word("Halllo", 0.4, 0.9), Word("bitte", 1.0, 1.3)])
    res = SuggestCleaner(suggest_fn, lex, client=client).clean([u], [], None)
    # text is NOT changed (suggest, don't apply)
    assert res.utterances[0].text == "sagt Halllo bitte"
    assert len(res.suggestions) == 1
    s = res.suggestions[0]
    assert s["original"] == "Halllo" and s["word_index"] == 1
    assert s["start"] == 0.4 and s["end"] == 0.9
    assert "Hallo" in s["candidates"]  # from both hunspell + LLM, deduped
    assert res.active is True


def test_suggest_mode_collapses_echoes_and_skips_clean_words():
    lex = _lex(known=["und", "ja"])
    u = _utt("und und ja", [Word("und", 0.0, 0.1), Word("und", 0.1, 0.2), Word("ja", 0.3, 0.5)])
    res = SuggestCleaner(lambda ws: {}, lex).clean([u], [], None)
    assert res.echoes == 1 and res.suggestions == []
    assert [w.w for w in res.utterances[0].words] == ["und", "ja"]


# ---- ManagedSpanRepairCleaner -------------------------------------------------------

class _FakeServer:
    base_url = "http://127.0.0.1:0"

    def __enter__(self):
        return self

    def __exit__(self, *e):
        return False


def test_managed_delegates_on_success():
    lex = _lex(known=["sagt", "Hallo", "bitte"])
    u = _utt("sagt Halllo bitte",
             [Word("sagt", 0.0, 0.3), Word("Halllo", 0.4, 0.9), Word("bitte", 1.0, 1.3)])
    cleaner = ManagedSpanRepairCleaner(
        "m.gguf", "dic", model_info={"name": "q"},
        _server=lambda: _FakeServer(),
        _client=lambda url: FakeClient({"Halllo": "Hallo"}),
        _lexicon=lex,
    )
    res = cleaner.clean([u], [], None)
    assert res.utterances[0].text == "sagt Hallo bitte" and res.active is True


def test_managed_falls_back_when_backend_unavailable():
    def boom():
        raise RuntimeError("no llama-server")

    u = _utt("sagt Halllo bitte",
             [Word("sagt", 0.0, 0.3), Word("Halllo", 0.4, 0.9), Word("bitte", 1.0, 1.3)])
    rep = RecordingReporter()
    res = ManagedSpanRepairCleaner("m", "dic", _server=boom, _lexicon=_lex(known=[])).clean(
        [u], [], rep)
    assert res.utterances == [u] and res.active is False
    assert rep.warns and "skipped" in rep.warns[0]
