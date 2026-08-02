"""Artifact writers: transcript.json / embeddings.npz / meta.json (what-we-build.md §6)."""

import json

import numpy as np

from meetscribe import __version__
from meetscribe.output import build_meta, write_embeddings, write_meta, write_transcript
from meetscribe.types import Utterance, Word

DIM = 192  # CAM++ embedding dim (research report — NOT the 512 in the spec)

MODELS = {
    "embedding_model": "cam++",
    "embedding_model_sha256": "abc123",
    "asr_model": "parakeet-tdt-0.6b-v3",
    "segmentation_model": "pyannote-segmentation-3.0",
}


def test_build_meta_has_all_required_keys():
    meta = build_meta(MODELS, embedding_dim=DIM)
    for key in [
        "embedding_model",
        "embedding_model_sha256",
        "embedding_dim",
        "asr_model",
        "segmentation_model",
        "sample_rate",
        "meetscribe_version",
    ]:
        assert key in meta
    assert meta["embedding_dim"] == DIM
    assert meta["sample_rate"] == 16000
    assert meta["meetscribe_version"] == __version__


def test_write_meta_round_trips(tmp_path):
    meta = build_meta(MODELS, embedding_dim=DIM)
    p = tmp_path / "meta.json"
    write_meta(p, meta)
    assert json.loads(p.read_text()) == meta


def test_write_transcript_matches_schema(tmp_path):
    utts = [
        Utterance(12.4, 18.9, "me", "mic", "hello there",
                  (Word("hello", 12.4, 12.7), Word("there", 12.8, 18.9))),
        Utterance(19.0, 20.0, "spk_0", "system", "hi", (Word("hi", 19.0, 20.0),)),
    ]
    p = tmp_path / "transcript.json"
    write_transcript(p, meeting_id="2026-08-02T14-30-00", duration_s=3412.5, utterances=utts)
    data = json.loads(p.read_text())
    assert data["meeting_id"] == "2026-08-02T14-30-00"
    assert data["duration_s"] == 3412.5
    assert len(data["segments"]) == 2
    seg = data["segments"][0]
    assert seg == {
        "start": 12.4, "end": 18.9, "speaker": "me", "track": "mic",
        "text": "hello there",
        "words": [
            {"w": "hello", "start": 12.4, "end": 12.7},
            {"w": "there", "start": 12.8, "end": 18.9},
        ],
    }


def test_write_embeddings_shapes_and_dtypes(tmp_path):
    turns = [
        ("turn_0", np.ones(DIM, dtype=np.float32), "me"),
        ("turn_1", np.zeros(DIM, dtype=np.float32), "spk_0"),
    ]
    clusters = [("spk_0", np.full(DIM, 0.5, dtype=np.float32))]
    p = tmp_path / "embeddings.npz"
    write_embeddings(p, turns, clusters, dim=DIM)
    z = np.load(p, allow_pickle=False)
    assert z["turn_ids"].tolist() == ["turn_0", "turn_1"]
    assert z["turn_vectors"].shape == (2, DIM)
    assert z["turn_vectors"].dtype == np.float32
    assert z["turn_speakers"].tolist() == ["me", "spk_0"]
    assert z["cluster_ids"].tolist() == ["spk_0"]
    assert z["cluster_vectors"].shape == (1, DIM)
    assert z["cluster_vectors"].dtype == np.float32


def test_write_embeddings_empty_gives_zero_row_arrays(tmp_path):
    p = tmp_path / "embeddings.npz"
    write_embeddings(p, [], [], dim=DIM)
    z = np.load(p, allow_pickle=False)
    assert z["turn_vectors"].shape == (0, DIM)
    assert z["cluster_vectors"].shape == (0, DIM)
    assert z["turn_ids"].shape == (0,)
