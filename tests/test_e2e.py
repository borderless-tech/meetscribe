"""Real end-to-end smoke test with the actual pinned models.

Opt-in: skipped unless MEETSCRIBE_MODELS points at the assembled model tree (set automatically
inside `nix run`, or by the flake). Uses the multilingual test clips shipped with the Parakeet
model — English for the mic track, German for the system track — to prove the whole wiring:
VAD → ASR (multilingual) → diarization → align → 192-dim embeddings → merge → artifacts.
"""

import json
import os
import subprocess

import numpy as np
import pytest

MODELS = os.environ.get("MEETSCRIBE_MODELS")

pytestmark = pytest.mark.skipif(
    not MODELS or not os.path.isdir(os.path.join(MODELS or "", "asr", "test_wavs")),
    reason="MEETSCRIBE_MODELS with test_wavs not available",
)


def _resample_16k(src, dst):
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-i", src, "-ar", "16000", "-ac", "1", dst],
        check=True,
    )


def test_full_pipeline_produces_valid_artifacts(tmp_path):
    from meetscribe import pipeline
    from meetscribe.output import build_meta, write_embeddings, write_meta, write_transcript

    mic = tmp_path / "mic.wav"
    system = tmp_path / "system.wav"
    _resample_16k(os.path.join(MODELS, "asr", "test_wavs", "en.wav"), str(mic))
    _resample_16k(os.path.join(MODELS, "asr", "test_wavs", "de.wav"), str(system))

    comp = pipeline.build_components(MODELS)
    assert comp.embedder.dim == 192

    res = pipeline.process(str(mic), str(system), comp)

    # both tracks produced an utterance with real transcribed text
    tracks = {u.track for u in res.utterances}
    assert tracks == {"mic", "system"}
    assert all(u.text.strip() for u in res.utterances)
    assert any(u.speaker == "me" for u in res.utterances)
    assert any(u.speaker.startswith("spk_") for u in res.utterances)

    # embeddings: per-turn + per-cluster, all 192-dim
    assert res.dim == 192
    assert res.turns and all(t[1].shape == (192,) for t in res.turns)
    assert res.clusters and all(c[1].shape == (192,) for c in res.clusters)

    # artifacts write + validate
    write_transcript(tmp_path / "transcript.json", "e2e", res.duration_s, res.utterances)
    write_embeddings(tmp_path / "embeddings.npz", res.turns, res.clusters, dim=res.dim)
    write_meta(
        tmp_path / "meta.json",
        build_meta(
            {
                "embedding_model": "cam++",
                "embedding_model_sha256": "x",
                "asr_model": "parakeet-tdt-0.6b-v3",
                "segmentation_model": "pyannote-segmentation-3.0",
            },
            embedding_dim=res.dim,
        ),
    )

    doc = json.loads((tmp_path / "transcript.json").read_text())
    assert len(doc["segments"]) == len(res.utterances)
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["embedding_dim"] == 192
    z = np.load(tmp_path / "embeddings.npz")
    assert z["turn_vectors"].shape[1] == 192
    assert z["cluster_vectors"].shape[1] == 192
