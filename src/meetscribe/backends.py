"""Transcription backends: the seam between pipeline.process() and how audio becomes text.

A backend turns the (already loaded) mic/system tracks into utterances plus the system-track
diarization segments; ``process()`` keeps everything downstream — the local embedding pass,
merge/coalesce, and cleanup — identical across backends. :class:`LocalBackend` is today's
offline path (VAD → Parakeet, diarize → align) extracted verbatim from ``process()``; a
remote backend implements the same :class:`TranscriptionBackend` protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .align import assign_words_to_speakers
from .asr import transcribe_chunks
from .diarize import run as diarize_run
from .types import DiarSegment, Utterance


@dataclass(frozen=True)
class TrackInput:
    """One audio track, loaded by ``process()``: 16 kHz mono float32 samples."""

    wav_path: str
    samples: np.ndarray


@dataclass
class BackendResult:
    mic_utts: list[Utterance]  # track="mic"; merge_tracks forces speaker="me"
    system_utts: list[Utterance]  # speaker="spk_N", track="system"
    system_diar: list[DiarSegment]  # feeds the local embedding pass
    models_meta: dict = field(default_factory=dict)  # model identity for meta.json


class TranscriptionBackend(Protocol):
    name: str  # "local" | "deepgram"

    def transcribe(
        self,
        mic: TrackInput | None,
        system: TrackInput | None,
        glossary: list[str],
        reporter,
    ) -> BackendResult: ...


class LocalBackend:
    """The offline sherpa-onnx path (§2.1 asymmetry): mic is VAD→ASR and hard-labelled
    ``me``; the system track is diarized and words are align-assigned to clusters."""

    name = "local"

    def __init__(self, components):
        self.components = components

    def transcribe(self, mic, system, glossary, reporter) -> BackendResult:
        components = self.components
        mic_utts: list[Utterance] = []
        system_utts: list[Utterance] = []
        system_diar: list[DiarSegment] = []

        # ---- mic track: user, no diarization ---------------------------------------
        if mic is not None:
            with reporter.stage("transcribe (mic)"):
                segs = _transcribe(components, mic.samples, reporter, "ASR (mic)")
            mic_utts = [
                Utterance(s.start, s.end, "me", "mic", s.text, s.words) for s in segs
            ]

        # ---- system track: everyone else, diarized ---------------------------------
        if system is not None:
            with reporter.stage("diarize (system)"):
                # sherpa reports (processed_chunks, total_chunks); map to a 0–100 bar.
                with reporter.track("diarization (system)", total=100) as bar:
                    done = {"pct": 0}

                    def _on_diar_progress(processed, total):
                        pct = min(100, int(processed * 100 / total)) if total else 100
                        if pct > done["pct"]:
                            bar.advance(pct - done["pct"])
                            done["pct"] = pct

                    diar = diarize_run(
                        components.diarizer, system.samples, on_progress=_on_diar_progress
                    )
            with reporter.stage("transcribe (system)"):
                segs = _transcribe(components, system.samples, reporter, "ASR (system)")
            words = [w for s in segs for w in s.words]
            system_utts = assign_words_to_speakers(words, diar)

            # Embed only speakers that made it into the transcript: a word-less cluster
            # must not ship vectors (sinks reject embedding speakers without a transcript
            # segment), and every transcript speaker must get a centroid — so centroids
            # use all of a speaker's segments, unfiltered.
            spoken = {u.speaker for u in system_utts}
            # A forced/auto cluster that wins no words vanishes from the transcript
            # silently — which reads as "--speakers N was ignored". Name the dropped
            # clusters so a low count is explainable (someone barely spoke) rather than
            # mysterious. (This can be legitimate; it is a heads-up, not an error.)
            clusters_found = {s.speaker for s in diar}
            dropped = sorted(clusters_found - spoken)
            if dropped:
                reporter.warn(
                    f"diarization formed {len(clusters_found)} speaker(s) but "
                    f"{len(dropped)} produced no transcribed words and were dropped: "
                    f"{', '.join(dropped)}"
                )
            system_diar = [s for s in diar if s.speaker in spoken]

        return BackendResult(
            mic_utts,
            system_utts,
            system_diar,
            models_meta={
                "asr_model": "parakeet-tdt-0.6b-v3",
                "segmentation_model": "pyannote-segmentation-3.0",
            },
        )


def _transcribe(components, samples, reporter, label):
    """VAD-chunk then transcribe, driving a progress bar over the chunk count."""
    chunks = components.vad.chunks(samples)
    with reporter.track(label, total=len(chunks)) as bar:
        return transcribe_chunks(components.recognizer, chunks, on_advance=bar.advance)
