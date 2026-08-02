"""End-to-end processing orchestration: audio → transcript/embeddings/meta.

Implemented in chunk C9. Wires the per-stage modules (audio, vad, asr, diarize,
embed, align, merge, output) into the two-track pipeline of what-we-build.md §4.
"""

from __future__ import annotations


def run(audio: str | None = None, out_dir: str | None = None) -> int:
    raise NotImplementedError("process/pipeline is implemented in chunk C9")
