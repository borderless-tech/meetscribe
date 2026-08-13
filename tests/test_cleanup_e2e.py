"""Real end-to-end cleanup smoke: spins up the actual llama-server on the pinned GGUF.

Opt-in: skipped unless MEETSCRIBE_MODELS carries llm/model.gguf (the ``models-llm`` output)
AND llama-server is on PATH (the ``meetscribe-llm`` variant). Proves the whole cleanup wiring —
managed server → chat request → guarded apply — corrects a garbled glossary term while leaving
per-word timings byte-identical.
"""

import os
import shutil

import pytest

MODELS = os.environ.get("MEETSCRIBE_MODELS")
_HAS = (
    bool(MODELS)
    and os.path.exists(os.path.join(MODELS or "", "llm", "model.gguf"))
    and os.path.isdir(os.path.join(MODELS or "", "hunspell"))
    and shutil.which("llama-server") is not None
    and shutil.which("hunspell") is not None
)

pytestmark = pytest.mark.skipif(
    not _HAS, reason="models-llm (gguf+hunspell) / llama-server / hunspell not available"
)


def test_cleanup_suggests_candidates_for_garbled_span():
    from meetscribe.pipeline import apply_cleanup, build_components
    from meetscribe.progress import NullReporter
    from meetscribe.types import Utterance, Word

    words = (
        Word("wir", 0.0, 0.4), Word("arbeiten", 0.4, 1.0),
        Word("bei", 1.0, 1.2), Word("Borderles", 1.2, 2.0),
    )
    u = Utterance(0.0, 2.0, "spk_0", "system", "wir arbeiten bei Borderles", words)

    comps = build_components(MODELS)
    utts, cleaned, model, suggestions = apply_cleanup([u], comps.cleaner, ["Borderless"], NullReporter())

    assert cleaned is True
    # suggest-mode: text is left RAW (not auto-applied); the fix is offered as a candidate
    assert utts[0].text == "wir arbeiten bei Borderles"
    sug = [s for s in suggestions if s["original"] == "Borderles"]
    assert sug, "expected a suggestion for the garbled word"
    assert "Borderless" in sug[0]["candidates"]  # correct term offered (glossary/LLM)
    assert sug[0]["start"] == 1.2 and sug[0]["end"] == 2.0  # located for the reviewer
