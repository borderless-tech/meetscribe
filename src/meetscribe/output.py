"""Write the three artifacts: transcript.json, embeddings.npz, meta.json (what-we-build.md §6).

The embedding dimension is a runtime value (192 for CAM++, read from ``extractor.dim``) — it is
stored in meta.json and used to size the npz arrays. Never hard-code 512.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from . import __version__
from .types import Utterance

# Bumped whenever the on-disk artifact/meta shape changes (forward-compat lever).
FORMAT_VERSION = 1

# (id, vector of shape (dim,), speaker)
Turn = tuple[str, np.ndarray, str]
# (cluster_id, vector of shape (dim,))
Cluster = tuple[str, np.ndarray]


def build_meta(
    models: dict,
    embedding_dim: int,
    *,
    meeting_id: str,
    started_at: str,
    ended_at: str,
    duration_s: float,
    sample_rate: int = 16000,
) -> dict:
    """Assemble meta.json. ``models`` supplies the model names + embedding hash.

    ``started_at``/``ended_at`` are tz-aware ISO 8601 strings (with offset) — the
    calendar-reconciliation match window.
    """
    return {
        "embedding_model": models["embedding_model"],
        "embedding_model_sha256": models["embedding_model_sha256"],
        "embedding_dim": embedding_dim,
        "asr_model": models["asr_model"],
        "segmentation_model": models["segmentation_model"],
        "sample_rate": sample_rate,
        "meetscribe_version": __version__,
        "meeting_id": meeting_id,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_s": duration_s,
        "format_version": FORMAT_VERSION,
    }


def write_meta(path: str | Path, meta: dict) -> None:
    Path(path).write_text(json.dumps(meta, indent=2))


def _utterance_to_dict(u: Utterance) -> dict:
    return {
        "start": u.start,
        "end": u.end,
        "speaker": u.speaker,
        "track": u.track,
        "text": u.text,
        "words": [{"w": w.w, "start": w.start, "end": w.end} for w in u.words],
    }


def write_transcript(
    path: str | Path,
    meeting_id: str,
    duration_s: float,
    utterances: Sequence[Utterance],
) -> None:
    doc = {
        "meeting_id": meeting_id,
        "duration_s": duration_s,
        "segments": [_utterance_to_dict(u) for u in utterances],
    }
    Path(path).write_text(json.dumps(doc, indent=2))


def _stack(vectors: Sequence[np.ndarray], dim: int) -> np.ndarray:
    if not vectors:
        return np.empty((0, dim), dtype=np.float32)
    return np.stack([np.asarray(v, dtype=np.float32) for v in vectors])


def write_embeddings(
    path: str | Path,
    turns: Sequence[Turn],
    clusters: Sequence[Cluster],
    dim: int,
) -> None:
    turn_ids = np.array([t[0] for t in turns], dtype=np.str_)
    turn_vectors = _stack([t[1] for t in turns], dim)
    turn_speakers = np.array([t[2] for t in turns], dtype=np.str_)
    cluster_ids = np.array([c[0] for c in clusters], dtype=np.str_)
    cluster_vectors = _stack([c[1] for c in clusters], dim)
    np.savez(
        path,
        turn_ids=turn_ids,
        turn_vectors=turn_vectors,
        turn_speakers=turn_speakers,
        cluster_ids=cluster_ids,
        cluster_vectors=cluster_vectors,
    )
