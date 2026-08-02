"""The .mscribe upload bundle: zip the three artifacts from a directory (design §Producer)."""

import json
import zipfile

import numpy as np
import pytest

from meetscribe.output import (
    BUNDLE_MEMBERS,
    bundle_dir,
    build_meta,
    write_embeddings,
    write_meta,
    write_transcript,
)
from meetscribe.types import Utterance, Word

DIM = 192
MODELS = {
    "embedding_model": "cam++",
    "embedding_model_sha256": "abc123",
    "asr_model": "parakeet-tdt-0.6b-v3",
    "segmentation_model": "pyannote-segmentation-3.0",
}


def _make_artifacts(d):
    write_transcript(
        d / "transcript.json",
        meeting_id="m1",
        duration_s=1.0,
        utterances=[Utterance(0.0, 1.0, "me", "mic", "hi", (Word("hi", 0.0, 1.0),))],
    )
    write_embeddings(
        d / "embeddings.npz",
        [("turn_0", np.ones(DIM, dtype=np.float32), "me")],
        [],
        dim=DIM,
    )
    write_meta(
        d / "meta.json",
        build_meta(
            MODELS, embedding_dim=DIM, meeting_id="m1",
            started_at="2026-08-02T14:03:11+02:00",
            ended_at="2026-08-02T14:03:12+02:00", duration_s=1.0,
        ),
    )


def test_bundle_dir_contains_the_three_members_byte_identical(tmp_path):
    src = tmp_path / "meeting"
    src.mkdir()
    _make_artifacts(src)
    out = tmp_path / "meeting-m1.mscribe"

    bundle_dir(src, out)

    assert out.exists()
    with zipfile.ZipFile(out) as z:
        assert sorted(z.namelist()) == sorted(BUNDLE_MEMBERS)
        for name in BUNDLE_MEMBERS:
            assert z.read(name) == (src / name).read_bytes()  # no re-encoding
    # embeddings survive the round-trip through the zip
    with zipfile.ZipFile(out) as z, z.open("embeddings.npz") as f:
        assert np.load(f)["turn_vectors"].shape == (1, DIM)


def test_bundle_dir_rejects_missing_member(tmp_path):
    src = tmp_path / "meeting"
    src.mkdir()
    _make_artifacts(src)
    (src / "embeddings.npz").unlink()
    with pytest.raises(FileNotFoundError, match="embeddings.npz"):
        bundle_dir(src, tmp_path / "x.mscribe")


def test_bundle_dir_rejects_meta_without_format_version(tmp_path):
    src = tmp_path / "meeting"
    src.mkdir()
    _make_artifacts(src)
    (src / "meta.json").write_text(json.dumps({"meeting_id": "m1"}))  # no format_version
    with pytest.raises(ValueError, match="format_version"):
        bundle_dir(src, tmp_path / "x.mscribe")
