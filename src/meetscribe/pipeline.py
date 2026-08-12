"""End-to-end processing: audio tracks → transcript.json / embeddings.npz / meta.json (§4).

The two tracks are handled asymmetrically (§2.1): the mic track is the user (hard-labelled
``me``, no diarization), while the system track is diarized. Both are transcribed and embedded.
Stage objects are injected via :class:`Components` so this orchestration can be driven with real
sherpa models (the end-to-end smoke test) or fakes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .align import assign_words_to_speakers
from .asr import transcribe_chunks
from .audio import load_wav_f32
from .diarize import run as diarize_run
from .embed import EMBEDDING_DIM, cluster_centroids, embed_turns, filter_short, slice_audio
from .merge import coalesce_utterances, merge_tracks
from .output import build_meta, write_embeddings, write_meta, write_transcript
from .types import DiarSegment, Utterance

SAMPLE_RATE = 16000


@dataclass
class Components:
    vad: object  # .chunks(samples) -> list[Chunk]
    recognizer: object  # .recognize(samples) -> RawResult
    diarizer: object  # .segments(samples) -> raw segs
    embedder: object  # .dim ; .embed(samples) -> vector
    cleaner: object = None  # .clean(texts, glossary, reporter) -> CleanResult; None → NullCleaner

    def __post_init__(self):
        if self.cleaner is None:
            from .cleanup import NullCleaner

            self.cleaner = NullCleaner()


@dataclass
class Result:
    utterances: list[Utterance]
    turns: list
    clusters: list
    dim: int
    duration_s: float
    cleaned: bool = False
    cleanup_model: dict | None = None


def _group_by_speaker(segments: list[DiarSegment]) -> dict[str, list[DiarSegment]]:
    out: dict[str, list[DiarSegment]] = {}
    for s in segments:
        out.setdefault(s.speaker, []).append(s)
    return out


def _prefix_turns(turns, prefix):
    return [(f"{prefix}_{i}", vec, spk) for i, (_, vec, spk) in enumerate(turns)]


def process(
    mic_wav: str | None,
    system_wav: str | None,
    components: Components,
    reporter=None,
    glossary: list[str] | None = None,
) -> Result:
    from .progress import NullReporter

    reporter = reporter or NullReporter()
    glossary = glossary or []
    mic_utts: list[Utterance] = []
    system_utts: list[Utterance] = []
    turns: list = []
    clusters: list = []
    dim = getattr(components.embedder, "dim", EMBEDDING_DIM)
    duration = 0.0

    # ---- mic track: user, no diarization -------------------------------------------
    if mic_wav:
        samples, _ = load_wav_f32(mic_wav)
        duration = max(duration, len(samples) / SAMPLE_RATE)
        with reporter.stage("transcribe (mic)"):
            segs = _transcribe(components, samples, reporter, "ASR (mic)")
        mic_utts = [
            Utterance(s.start, s.end, "me", "mic", s.text, s.words) for s in segs
        ]
        # §2.6: embed the mic segments too → a clean "me" profile without clustering risk.
        # Turn vectors only for segments long enough to embed alone; the centroid uses
        # every segment (concatenation supplies the duration), so "me" has a vector
        # whenever it is in the transcript.
        me_segs = [DiarSegment(s.start, s.end, "me") for s in segs]
        with reporter.stage("embed (mic)"):
            me_long = filter_short(me_segs)
            total = len(me_long) + (1 if me_segs else 0)
            with reporter.track("embedding (mic)", total=total) as bar:
                turns += _prefix_turns(
                    embed_turns(
                        components.embedder, samples, SAMPLE_RATE, me_long,
                        on_advance=bar.advance,
                    ),
                    "mic",
                )
                if me_segs:
                    clusters += cluster_centroids(
                        components.embedder, samples, SAMPLE_RATE, {"me": me_segs},
                        on_advance=bar.advance,
                    )

    # ---- system track: everyone else, diarized -------------------------------------
    if system_wav:
        samples, _ = load_wav_f32(system_wav)
        duration = max(duration, len(samples) / SAMPLE_RATE)
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
                    components.diarizer, samples, on_progress=_on_diar_progress
                )
        with reporter.stage("transcribe (system)"):
            segs = _transcribe(components, samples, reporter, "ASR (system)")
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
        diar_spoken = [s for s in diar if s.speaker in spoken]
        with reporter.stage("embed (system)"):
            sys_long = filter_short(diar_spoken)
            groups = _group_by_speaker(diar_spoken)
            with reporter.track(
                "embedding (system)", total=len(sys_long) + len(groups)
            ) as bar:
                turns += _prefix_turns(
                    embed_turns(
                        components.embedder, samples, SAMPLE_RATE, sys_long,
                        on_advance=bar.advance,
                    ),
                    "sys",
                )
                clusters += cluster_centroids(
                    components.embedder, samples, SAMPLE_RATE, groups,
                    on_advance=bar.advance,
                )

    utterances = coalesce_utterances(merge_tracks(mic_utts, system_utts))
    duration = max([duration] + [u.end for u in utterances])

    # ---- cleanup (llm): rewrite only system-track text -----------------------------
    # The mic track is the user's clean audio; only the degraded system downmix is worth
    # cleaning (and risking paraphrase over). raw_text preserves the original ASR text.
    from dataclasses import replace

    sys_idx = [i for i, u in enumerate(utterances) if u.track == "system"]
    cleaned = False
    cleanup_model = None
    if sys_idx:
        with reporter.stage("cleanup (llm)"):
            res = components.cleaner.clean(
                [utterances[i].text for i in sys_idx], glossary, reporter
            )
        for j, i in enumerate(sys_idx):
            u = utterances[i]
            utterances[i] = replace(u, text=res.texts[j], raw_text=u.text)
        cleaned = res.active
        if res.active:
            cleanup_model = getattr(components.cleaner, "model_info", None)
            reporter.info(
                f"cleaned {res.cleaned}/{len(sys_idx)} system segments "
                f"({res.kept_raw} kept raw)"
            )
    return Result(utterances, turns, clusters, dim, duration, cleaned, cleanup_model)


def _transcribe(components, samples, reporter, label):
    """VAD-chunk then transcribe, driving a progress bar over the chunk count."""
    chunks = components.vad.chunks(samples)
    with reporter.track(label, total=len(chunks)) as bar:
        return transcribe_chunks(components.recognizer, chunks, on_advance=bar.advance)


def summarize(result: Result) -> "Summary":
    """Aggregate a Result into a Summary (per-speaker talk time, ordered by talk time)."""
    from .progress import Summary

    talk: dict[str, float] = {}
    track: dict[str, str] = {}
    order: list[str] = []
    for u in result.utterances:
        if u.speaker not in talk:
            talk[u.speaker] = 0.0
            order.append(u.speaker)
        talk[u.speaker] += u.end - u.start
        track[u.speaker] = u.track
    speakers = sorted(order, key=lambda sp: (-talk[sp], order.index(sp)))
    return Summary(
        duration_s=result.duration_s,
        n_segments=len(result.utterances),
        speakers=[(sp, talk[sp], track[sp]) for sp in speakers],
    )


def resolve_inputs(path: str) -> tuple[str | None, str | None]:
    """Resolve an input path to (mic_wav, system_wav).

    A directory is expected to hold raw/{mic,system}.wav (or {mic,system}.wav at its root); a
    single .wav file is treated as the system track (unknown speakers → diarize it).
    """
    p = Path(path)
    if p.is_dir():
        for base in (p / "raw", p):
            mic = base / "mic.wav"
            system = base / "system.wav"
            if mic.exists() or system.exists():
                return (
                    str(mic) if mic.exists() else None,
                    str(system) if system.exists() else None,
                )
        return (None, None)
    return (None, str(p))


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def build_components(models_dir: str, num_speakers: int = -1) -> Components:
    """Wire the real sherpa models. ``num_speakers`` is the expected speaker count on
    the system track (everyone except the user); -1 means threshold clustering."""
    from .asr import ParakeetRecognizer
    from .diarize import OfflineDiarizer
    from .embed import SpeakerEmbedder
    from .vad import SileroVad

    m = Path(models_dir)
    spk = str(m / "spk" / "model.onnx")
    return Components(
        vad=SileroVad(str(m / "vad" / "silero_vad.onnx")),
        recognizer=ParakeetRecognizer(str(m / "asr")),
        diarizer=OfflineDiarizer(
            str(m / "seg" / "model.int8.onnx"), spk, num_clusters=num_speakers
        ),
        embedder=SpeakerEmbedder(spk),
    )


def _window(started_at, duration_s, mic_wav, system_wav):
    """Return (started_at, ended_at) as tz-aware ISO 8601 strings (with offset).

    ``started_at`` (a tz-aware datetime) wins; otherwise fall back to the input WAV's
    mtime as a best-effort capture time for the ``process`` path.
    """
    from datetime import datetime, timedelta

    if started_at is None:
        src = mic_wav or system_wav
        ts = Path(src).stat().st_mtime if src else 0.0
        started_at = datetime.fromtimestamp(ts).astimezone()
    ended_at = started_at + timedelta(seconds=duration_s)
    return started_at.isoformat(), ended_at.isoformat()


def run(
    audio: str | None = None,
    out_dir: str | None = None,
    bundle: bool = False,
    started_at=None,
    reporter=None,
    num_speakers: int = -1,
) -> int:
    import os
    from datetime import datetime, timezone

    if audio is None:
        print("process: no input given (expected a directory or a .wav file)")
        return 2

    models_dir = os.environ.get("MEETSCRIBE_MODELS")
    if not models_dir:
        print("MEETSCRIBE_MODELS is not set — run via `nix run` or the wrapper")
        return 2

    mic_wav, system_wav = resolve_inputs(audio)
    if not mic_wav and not system_wav:
        print(f"no mic.wav / system.wav found under {audio}")
        return 2

    out = Path(out_dir or audio)
    out.mkdir(parents=True, exist_ok=True)
    meeting_id = out.name if out.name else f"{datetime.now(timezone.utc):%Y-%m-%dT%H-%M-%S}"

    from .progress import NullReporter

    reporter = reporter or NullReporter()
    # Loading the ~600 MB Parakeet model takes several seconds — show a spinner so the
    # record→process transition isn't a silent gap.
    with reporter.stage("loading models"):
        components = build_components(models_dir, num_speakers)
    result = process(mic_wav, system_wav, components, reporter=reporter)

    spk_model = str(Path(models_dir) / "spk" / "model.onnx")
    started_iso, ended_iso = _window(started_at, result.duration_s, mic_wav, system_wav)
    meta = build_meta(
        {
            "embedding_model": "3dspeaker_campplus_sv_zh_en_16k",
            "embedding_model_sha256": _sha256(spk_model),
            "asr_model": "parakeet-tdt-0.6b-v3",
            "segmentation_model": "pyannote-segmentation-3.0",
        },
        embedding_dim=result.dim,
        meeting_id=meeting_id,
        started_at=started_iso,
        ended_at=ended_iso,
        duration_s=result.duration_s,
        cleaned=result.cleaned,
        cleanup_model=result.cleanup_model,
    )
    write_transcript(
        out / "transcript.json", meeting_id, result.duration_s, result.utterances,
        cleaned=result.cleaned,
    )
    write_embeddings(out / "embeddings.npz", result.turns, result.clusters, dim=result.dim)
    write_meta(out / "meta.json", meta)

    if bundle:
        from .output import bundle_dir, default_bundle_name

        bundle_path = out / default_bundle_name(meta)
        bundle_dir(out, bundle_path)
        print(f"wrote {bundle_path.name}")

    reporter.summary(summarize(result))
    print(f"wrote transcript.json, embeddings.npz, meta.json to {out}")
    return 0
