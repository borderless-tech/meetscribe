"""LLM cleanup orchestration + anti-hallucination guards (no real model)."""

from meetscribe.cleanup import (
    LlamaCleaner,
    ManagedLlamaCleaner,
    NullCleaner,
    accept_candidate,
    build_prompt,
    token_budget,
)


# ---- token_budget (per-segment generation cap) --------------------------------------

def test_token_budget_floor_for_short_text():
    assert token_budget("") == 64
    assert token_budget("uh the the") == 64


def test_token_budget_proportional_for_medium_text():
    assert token_budget("x" * 400) == 216  # len//2 + 16


def test_token_budget_capped_for_long_text():
    assert token_budget("x" * 4000) == 512  # never exceeds the old flat default


# ---- accept_candidate (pure guard) --------------------------------------------------

def test_accept_good_correction():
    assert accept_candidate("Borderlestern GmbH", "Borderless GmbH")


def test_reject_empty_candidate():
    assert not accept_candidate("Borderless GmbH", "")
    assert not accept_candidate("Borderless GmbH", "   ")


def test_reject_empty_input():
    # No input to correct → never inject LLM text.
    assert not accept_candidate("", "anything")


def test_reject_runaway_over_3x():
    assert not accept_candidate("hi there", "ha " * 40)


def test_reject_collapse_only_for_long_inputs():
    long = "this is a genuinely long original sentence with plenty of words here"  # >40 chars
    assert not accept_candidate(long, "a")                       # <0.3× → collapse, rejected
    # short backchannel legitimately collapses: "uh the the" -> "the"
    assert accept_candidate("uh the the", "the")                 # input <=40 chars → allowed


# ---- prompt construction ------------------------------------------------------------

def test_build_prompt_includes_glossary_context_and_current():
    p = build_prompt(["Borderless", "Georg"], prev="hello there", current="borderles stuff")
    assert "Borderless" in p and "Georg" in p
    assert "hello there" in p          # previous-segment context
    assert "borderles stuff" in p      # the text to fix


# ---- NullCleaner --------------------------------------------------------------------

def test_null_cleaner_is_identity_and_inactive():
    res = NullCleaner().clean(["a", "b"], [], None)
    assert res.texts == ["a", "b"]
    assert res.active is False
    assert res.cleaned == 0 and res.kept_raw == 0


# ---- LlamaCleaner (fake client) -----------------------------------------------------

class FakeClient:
    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []
        self.budgets = []

    def complete(self, prompt, max_tokens=None):
        self.calls.append(prompt)
        self.budgets.append(max_tokens)
        for key, val in self.mapping.items():
            if key in prompt:
                return val
        return "UNMATCHED"


class RecordingReporter:
    def __init__(self):
        self.warns = []

    def warn(self, msg):
        self.warns.append(msg)


def test_llama_cleaner_applies_good_keeps_guarded():
    client = FakeClient({"Borderlestern": "Borderless.", "runaway": "na " * 99})
    rep = RecordingReporter()
    res = LlamaCleaner(client).clean(
        ["Borderlestern", "runaway text that is quite long here"], ["Borderless"], rep
    )
    assert res.texts[0] == "Borderless."                              # applied
    assert res.texts[1] == "runaway text that is quite long here"     # guard kept raw
    assert res.cleaned == 1 and res.kept_raw == 1
    assert res.active is True
    assert rep.warns  # a guard trip was surfaced


def test_llama_cleaner_passes_glossary_and_context():
    # seg one maps to itself → the cleaned previous line ("seg one") is the context for seg two
    client = FakeClient({"seg one": "seg one", "seg-two": "cleaned two"})
    LlamaCleaner(client).clean(["seg one", "seg-two"], ["Borderless", "Georg"], None)
    p = client.calls[1]
    assert "Borderless" in p and "Georg" in p
    assert "seg one" in p and "seg-two" in p  # prior (cleaned) line as context + current


def test_llama_cleaner_caps_tokens_per_segment():
    # Each segment requests only ~its own length of generation, not a flat 512 → the
    # long tail of slow CPU calls collapses.
    client = FakeClient({})
    LlamaCleaner(client).clean(["short one", "x" * 400], [], None)
    assert client.budgets == [token_budget("short one"), token_budget("x" * 400)]
    assert client.budgets[0] == 64 and client.budgets[1] == 216


def test_llama_cleaner_survives_client_error():
    class Boom:
        def complete(self, prompt, max_tokens=None):
            raise RuntimeError("server died")

    rep = RecordingReporter()
    res = LlamaCleaner(Boom()).clean(["keep me"], [], rep)
    assert res.texts == ["keep me"]
    assert res.kept_raw == 1 and res.cleaned == 0
    assert res.active is True


# ---- ManagedLlamaCleaner (server lifecycle mocked) ----------------------------------

class _FakeServer:
    base_url = "http://127.0.0.1:0"

    def __enter__(self):
        return self

    def __exit__(self, *e):
        return False


def test_managed_cleaner_delegates_on_success():
    cleaner = ManagedLlamaCleaner(
        "m.gguf", model_info={"name": "q"},
        _server=lambda: _FakeServer(),
        _client=lambda url: FakeClient({"borderles": "Borderless"}),
    )
    res = cleaner.clean(["borderles"], ["Borderless"], None)
    assert res.texts == ["Borderless"] and res.active is True


def test_managed_cleaner_falls_back_when_server_unavailable():
    def boom():
        raise RuntimeError("llama-server not found")

    rep = RecordingReporter()
    res = ManagedLlamaCleaner("missing.gguf", _server=boom).clean(["keep"], [], rep)
    assert res.texts == ["keep"]          # raw text preserved
    assert res.active is False            # → meta stays cleaned:false
    assert rep.warns and "skipped" in rep.warns[0]
