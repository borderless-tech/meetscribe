"""Write the three artifacts: transcript.json, embeddings.npz, meta.json (what-we-build.md §6).

The embedding dimension is a runtime value (192 for CAM++, read from ``extractor.dim``) — it is
stored in meta.json and used to size the npz arrays. Never hard-code 512.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from . import __version__
from .types import Utterance, Word

# Bumped whenever the on-disk artifact/meta shape changes (forward-compat lever).
# v2 adds per-segment ``raw_text`` and the top-level ``cleaned`` flag.
FORMAT_VERSION = 2

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
    cleaned: bool = False,
    cleanup_model: dict | None = None,
    backend: str = "local",
) -> dict:
    """Assemble meta.json. ``models`` supplies the model names + embedding hash.

    ``started_at``/``ended_at`` are tz-aware ISO 8601 strings (with offset) — the
    calendar-reconciliation match window. ``cleanup_model`` (name + SHA-256 + params) travels
    with the artifact when the LLM cleanup ran — text cleaned by different models isn't
    equivalent, same rule as the embedding model identity. ``backend`` names the STT path
    (``local``/``deepgram``); any extra keys in ``models`` (a remote backend's identity, e.g.
    ``backend_model_versions``/``request_ids``) are merged additively — ``format_version``
    stays 2.
    """
    meta = {
        "embedding_model": models["embedding_model"],
        "embedding_model_sha256": models["embedding_model_sha256"],
        "embedding_dim": embedding_dim,
        "asr_model": models["asr_model"],
        "segmentation_model": models["segmentation_model"],
        "backend": backend,
        "sample_rate": sample_rate,
        "meetscribe_version": __version__,
        "meeting_id": meeting_id,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_s": duration_s,
        "cleaned": cleaned,
        "format_version": FORMAT_VERSION,
    }
    if cleanup_model is not None:
        meta["cleanup_model"] = cleanup_model
    # Additive merge of whatever else the backend recorded about itself (the remote
    # substitute for local SHA-256 pins) — model identity must travel with the artifact.
    for key, value in models.items():
        meta.setdefault(key, value)
    return meta


def default_bundle_name(meta: dict) -> str:
    return f"meeting-{meta['meeting_id']}.mscribe"


def write_meta(path: str | Path, meta: dict) -> None:
    Path(path).write_text(json.dumps(meta, indent=2))


def _utterance_to_dict(u: Utterance) -> dict:
    return {
        "start": u.start,
        "end": u.end,
        "speaker": u.speaker,
        "track": u.track,
        "text": u.text,
        "raw_text": u.raw_text or u.text,  # fallback: raw == text when no cleanup ran
        "words": [{"w": w.w, "start": w.start, "end": w.end} for w in u.words],
    }


def _dict_to_utterance(seg: dict) -> Utterance:
    return Utterance(
        start=seg["start"],
        end=seg["end"],
        speaker=seg["speaker"],
        track=seg["track"],
        text=seg["text"],
        words=tuple(Word(w["w"], w["start"], w["end"]) for w in seg["words"]),
        # v1 back-compat: no raw_text → equals text. v2: preserve the original raw_text.
        raw_text=seg.get("raw_text", seg["text"]),
    )


def write_transcript(
    path: str | Path,
    meeting_id: str,
    duration_s: float,
    utterances: Sequence[Utterance],
    cleaned: bool = False,
    suggestions: Sequence[dict] | None = None,
) -> None:
    doc = {
        "meeting_id": meeting_id,
        "duration_s": duration_s,
        "cleaned": cleaned,
        # broken-word correction candidates for human review (borderless-knowledge); the
        # transcript text is left raw — these are suggestions, not applied edits.
        "suggestions": list(suggestions) if suggestions else [],
        "segments": [_utterance_to_dict(u) for u in utterances],
    }
    Path(path).write_text(json.dumps(doc, indent=2, ensure_ascii=False))


def read_transcript(path: str | Path) -> tuple[str, float, list[Utterance]]:
    """Inverse of :func:`write_transcript`: rebuild ``(meeting_id, duration_s, utterances)``
    with words/timings/raw_text reconstructed exactly. Used by the ``clean`` retrofit path."""
    doc = json.loads(Path(path).read_text())
    utts = [_dict_to_utterance(seg) for seg in doc["segments"]]
    return doc["meeting_id"], doc["duration_s"], utts


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


# The .mscribe bundle: a plain zip of these three fixed-name members (design doc).
BUNDLE_MEMBERS = ("transcript.json", "embeddings.npz", "meta.json")


def bundle_dir(src_dir: str | Path, out_path: str | Path) -> Path:
    """Zip the three artifacts from ``src_dir`` into a single ``.mscribe`` file.

    Single validation point: every member must exist and ``meta.json`` must carry a
    ``format_version`` before we write anything.
    """
    src = Path(src_dir)
    for name in BUNDLE_MEMBERS:
        if not (src / name).exists():
            raise FileNotFoundError(f"cannot bundle: missing {name} in {src}")
    meta = json.loads((src / "meta.json").read_text())
    if "format_version" not in meta:
        raise ValueError(f"cannot bundle: meta.json in {src} lacks format_version")
    out = Path(out_path)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name in BUNDLE_MEMBERS:
            z.write(src / name, arcname=name)
    return out
