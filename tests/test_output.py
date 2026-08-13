"""Artifact writers: transcript.json / embeddings.npz / meta.json (what-we-build.md §6)."""

import json

import numpy as np

from meetscribe import __version__
from meetscribe.output import (
    FORMAT_VERSION,
    build_meta,
    write_embeddings,
    write_meta,
    write_transcript,
)
from meetscribe.types import Utterance, Word

DIM = 192  # CAM++ embedding dim (research report — NOT the 512 in the spec)

MODELS = {
    "embedding_model": "cam++",
    "embedding_model_sha256": "abc123",
    "asr_model": "parakeet-tdt-0.6b-v3",
    "segmentation_model": "pyannote-segmentation-3.0",
}

# The bundle fields build_meta now requires (design §meta enrichment).
BUNDLE_FIELDS = {
    "meeting_id": "abc123",
    "started_at": "2026-08-02T14:03:11+02:00",
    "ended_at": "2026-08-02T14:47:52+02:00",
    "duration_s": 2681.0,
}


def test_build_meta_has_all_required_keys():
    meta = build_meta(MODELS, embedding_dim=DIM, **BUNDLE_FIELDS)
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


def test_build_meta_defaults_to_uncleaned():
    meta = build_meta(MODELS, embedding_dim=DIM, **BUNDLE_FIELDS)
    assert meta["cleaned"] is False
    assert "cleanup_model" not in meta


def test_build_meta_records_cleanup_identity():
    cm = {"name": "Qwen2.5-7B-Instruct-Q4_K_M", "sha256": "abc123", "temp": 0.0}
    meta = build_meta(MODELS, embedding_dim=DIM, cleaned=True, cleanup_model=cm, **BUNDLE_FIELDS)
    assert meta["cleaned"] is True
    assert meta["cleanup_model"] == cm


def test_build_meta_includes_bundle_fields():
    meta = build_meta(
        MODELS,
        embedding_dim=DIM,
        meeting_id="abc123",
        started_at="2026-08-02T14:03:11+02:00",
        ended_at="2026-08-02T14:47:52+02:00",
        duration_s=2681.0,
    )
    assert meta["meeting_id"] == "abc123"
    assert meta["started_at"] == "2026-08-02T14:03:11+02:00"
    assert meta["ended_at"] == "2026-08-02T14:47:52+02:00"
    assert meta["duration_s"] == 2681.0
    assert meta["format_version"] == FORMAT_VERSION


def test_write_meta_round_trips(tmp_path):
    meta = build_meta(MODELS, embedding_dim=DIM, **BUNDLE_FIELDS)
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
    assert data["cleaned"] is False  # default: no cleanup pass ran
    seg = data["segments"][0]
    assert seg == {
        "start": 12.4, "end": 18.9, "speaker": "me", "track": "mic",
        "text": "hello there",
        "raw_text": "hello there",  # falls back to text when unset
        "words": [
            {"w": "hello", "start": 12.4, "end": 12.7},
            {"w": "there", "start": 12.8, "end": 18.9},
        ],
    }


def test_write_transcript_cleaned_flag_and_raw_text(tmp_path):
    from meetscribe.types import Utterance, Word

    u = Utterance(0.0, 1.0, "spk_0", "system", "Borderless GmbH",
                  (Word("borderlestern", 0.0, 0.5), Word("gmbh", 0.5, 1.0)),
                  raw_text="borderlestern gmbh")
    p = tmp_path / "t.json"
    write_transcript(p, meeting_id="m", duration_s=1.0, utterances=[u], cleaned=True)
    data = json.loads(p.read_text())
    assert data["cleaned"] is True
    assert data["segments"][0]["text"] == "Borderless GmbH"
    assert data["segments"][0]["raw_text"] == "borderlestern gmbh"


def test_read_transcript_round_trips(tmp_path):
    from meetscribe.output import read_transcript
    from meetscribe.types import Utterance, Word

    # raw_text is given explicitly on both (write normalizes an unset "" to text, so the
    # exact round-trip is over already-normalized utterances).
    utts = [
        Utterance(12.4, 18.9, "me", "mic", "hello there",
                  (Word("hello", 12.4, 12.7), Word("there", 12.8, 18.9)),
                  raw_text="hello there"),
        Utterance(19.0, 20.0, "spk_0", "system", "Borderless",
                  (Word("borderles", 19.0, 20.0),), raw_text="borderles"),
    ]
    p = tmp_path / "t.json"
    write_transcript(p, meeting_id="M1", duration_s=3412.5, utterances=utts, cleaned=True)
    meeting_id, duration_s, got = read_transcript(p)
    assert meeting_id == "M1" and duration_s == 3412.5
    assert got == utts  # words, timings, raw_text all reconstructed exactly


def test_read_transcript_v1_backcompat_raw_text_defaults_to_text(tmp_path):
    # A v1 transcript (no raw_text/cleaned) reads with raw_text == text.
    import json as _json
    from meetscribe.output import read_transcript

    p = tmp_path / "old.json"
    p.write_text(_json.dumps({
        "meeting_id": "old", "duration_s": 2.0,
        "segments": [{"start": 0.0, "end": 1.0, "speaker": "spk_0", "track": "system",
                      "text": "hi", "words": [{"w": "hi", "start": 0.0, "end": 1.0}]}],
    }))
    _, _, got = read_transcript(p)
    assert got[0].text == "hi" and got[0].raw_text == "hi"


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
