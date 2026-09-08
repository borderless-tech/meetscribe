"""Real end-to-end smoke test against the Deepgram API — THIS TEST COSTS MONEY.

Each run uploads the two short model test clips to Deepgram (~$0.01/run at nova-3
pay-as-you-go rates). It is therefore double-gated and never runs in CI:

- ``DEEPGRAM_API_KEY`` must be set (opts in to the paid call), AND
- ``MEETSCRIBE_MODELS`` must point at the assembled model tree — the embedding pass is
  local in every backend, so the CAM++ speaker model is still required, and the tree also
  ships the EN/DE test clips used as the mic/system fixtures.

Mirrors ``tests/test_e2e.py``: EN clip as the mic track, DE clip as the system track,
run the full ``process`` flow with the deepgram backend, then validate the artifacts
(me + spk_* speakers, embedding dim consistency, ``meta.backend``, and the bundle).
"""

import json
import os
import subprocess
import zipfile

import numpy as np
import pytest

MODELS = os.environ.get("MEETSCRIBE_MODELS")

pytestmark = pytest.mark.skipif(
    not os.environ.get("DEEPGRAM_API_KEY")
    or not MODELS
    or not os.path.isdir(os.path.join(MODELS or "", "asr", "test_wavs")),
    reason="paid remote e2e: needs DEEPGRAM_API_KEY and MEETSCRIBE_MODELS with test_wavs",
)


def _resample_16k(src, dst):
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-i", src, "-ar", "16000", "-ac", "1", dst],
        check=True,
    )


def test_deepgram_backend_produces_valid_artifacts(tmp_path, monkeypatch):
    from meetscribe import pipeline

    meeting = tmp_path / "meeting"
    raw = meeting / "raw"
    raw.mkdir(parents=True)
    _resample_16k(os.path.join(MODELS, "asr", "test_wavs", "en.wav"), str(raw / "mic.wav"))
    _resample_16k(os.path.join(MODELS, "asr", "test_wavs", "de.wav"), str(raw / "system.wav"))

    # Route the run through the remote backend via the documented env knob (flag > env >
    # default); keep the user's real glossary out of the paid request's keyterms.
    monkeypatch.setenv("STT_BACKEND", "deepgram")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

    assert pipeline.run(audio=str(meeting), bundle=True) == 0

    # transcript: both tracks made it through — mic hard-labelled "me", system diarized
    doc = json.loads((meeting / "transcript.json").read_text())
    assert doc["segments"], "empty transcript"
    speakers = {s["speaker"] for s in doc["segments"]}
    assert "me" in speakers
    assert any(sp.startswith("spk_") for sp in speakers)
    assert all(s["text"].strip() for s in doc["segments"])

    # meta: remote backend identity recorded; embeddings stay local CAM++
    meta = json.loads((meeting / "meta.json").read_text())
    assert meta["backend"] == "deepgram"
    dim = meta["embedding_dim"]

    # embeddings: vectors exist and match the meta-declared dim (never hard-code 192)
    z = np.load(meeting / "embeddings.npz")
    assert z["turn_vectors"].shape[0] > 0
    assert z["turn_vectors"].shape[1] == dim
    assert z["cluster_vectors"].shape[0] > 0
    assert z["cluster_vectors"].shape[1] == dim

    # bundle: default-on artifact zip built and holds exactly the three members
    bundles = list(meeting.glob("*.mscribe"))
    assert len(bundles) == 1
    with zipfile.ZipFile(bundles[0]) as zf:
        assert set(zf.namelist()) == {"transcript.json", "embeddings.npz", "meta.json"}
