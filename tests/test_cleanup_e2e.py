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


def test_cleanup_repairs_garbled_span_and_preserves_timings():
    from meetscribe.pipeline import apply_cleanup, build_components
    from meetscribe.progress import NullReporter
    from meetscribe.types import Utterance, Word

    words = (
        Word("wir", 0.0, 0.4), Word("arbeiten", 0.4, 1.0),
        Word("bei", 1.0, 1.2), Word("Borderles", 1.2, 2.0),
    )
    u = Utterance(0.0, 2.0, "spk_0", "system", "wir arbeiten bei Borderles", words)

    comps = build_components(MODELS)
    utts, cleaned, model = apply_cleanup([u], comps.cleaner, ["Borderless"], NullReporter())

    assert cleaned is True
    assert model and model["name"].startswith("Qwen2.5")
    assert utts[0].raw_text == "wir arbeiten bei Borderles"  # original preserved
    assert "Borderless" in utts[0].text                      # garble repaired via glossary
    # only the flagged word's TEXT changed; its timestamps (and the others) are intact
    assert utts[0].words[3].start == 1.2 and utts[0].words[3].end == 2.0
    assert [w.w for w in utts[0].words[:3]] == ["wir", "arbeiten", "bei"]
