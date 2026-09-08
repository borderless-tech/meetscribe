"""End-to-end processing: audio tracks → transcript.json / embeddings.npz / meta.json (§4).

The two tracks are handled asymmetrically (§2.1): the mic track is the user (hard-labelled
``me``, no diarization), while the system track is diarized. Both are transcribed and embedded.
Stage objects are injected via :class:`Components` so this orchestration can be driven with real
sherpa models (the end-to-end smoke test) or fakes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .audio import load_wav_f32
from .backends import LocalBackend, TrackInput
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
    suggestions: list = field(default_factory=list)
    models_meta: dict = field(default_factory=dict)  # from the backend (asr/seg identity)


def _group_by_speaker(segments: list[DiarSegment]) -> dict[str, list[DiarSegment]]:
    out: dict[str, list[DiarSegment]] = {}
    for s in segments:
        out.setdefault(s.speaker, []).append(s)
    return out


def _prefix_turns(turns, prefix):
    return [(f"{prefix}_{i}", vec, spk) for i, (_, vec, spk) in enumerate(turns)]


def _load_track_16k(wav_path: str, reporter) -> np.ndarray:
    """Load a WAV and guarantee ``SAMPLE_RATE`` samples.

    Duration and the embedding slices are sample-count arithmetic on the assumption of
    16 kHz; a mis-rated array corrupts them SILENTLY with a remote backend (whose
    timestamps are true seconds, so the transcript looks perfect) — locally it at least
    self-announced as garbage ASR. Linear interpolation is plenty for the local stages;
    remote uploads always send the original file bytes untouched."""
    samples, rate = load_wav_f32(wav_path)
    if rate == SAMPLE_RATE:
        return samples
    reporter.warn(
        f"{Path(wav_path).name}: {rate} Hz input — resampling to {SAMPLE_RATE} Hz "
        "for the local stages (ASR/embeddings/duration)"
    )
    n_out = int(round(len(samples) * SAMPLE_RATE / rate))
    positions = np.arange(n_out, dtype=np.float64) * (rate / SAMPLE_RATE)
    return np.interp(
        positions, np.arange(len(samples), dtype=np.float64), samples
    ).astype(np.float32)


def process(
    mic_wav: str | None,
    system_wav: str | None,
    components: Components,
    reporter=None,
    glossary: list[str] | None = None,
    backend=None,
) -> Result:
    from .progress import NullReporter

    reporter = reporter or NullReporter()
    glossary = glossary or []
    if backend is None:
        backend = LocalBackend(components)
    turns: list = []
    clusters: list = []
    dim = getattr(components.embedder, "dim", EMBEDDING_DIM)
    duration = 0.0

    mic: TrackInput | None = None
    system: TrackInput | None = None
    if mic_wav:
        samples = _load_track_16k(mic_wav, reporter)
        duration = max(duration, len(samples) / SAMPLE_RATE)
        mic = TrackInput(mic_wav, samples)
    if system_wav:
        samples = _load_track_16k(system_wav, reporter)
        duration = max(duration, len(samples) / SAMPLE_RATE)
        system = TrackInput(system_wav, samples)

    tr = backend.transcribe(mic, system, glossary, reporter)

    # ---- mic track embeddings ------------------------------------------------------
    if mic is not None:
        # §2.6: embed the mic segments too → a clean "me" profile without clustering risk.
        # Turn vectors only for segments long enough to embed alone; the centroid uses
        # every segment (concatenation supplies the duration), so "me" has a vector
        # whenever it is in the transcript.
        me_segs = [DiarSegment(u.start, u.end, "me") for u in tr.mic_utts]
        with reporter.stage("embed (mic)"):
            me_long = filter_short(me_segs)
            total = len(me_long) + (1 if me_segs else 0)
            with reporter.track("embedding (mic)", total=total) as bar:
                turns += _prefix_turns(
                    embed_turns(
                        components.embedder, mic.samples, SAMPLE_RATE, me_long,
                        on_advance=bar.advance,
                    ),
                    "mic",
                )
                if me_segs:
                    clusters += cluster_centroids(
                        components.embedder, mic.samples, SAMPLE_RATE, {"me": me_segs},
                        on_advance=bar.advance,
                    )

    # ---- system track embeddings (from the backend's diarization) ------------------
    if system is not None:
        diar_spoken = tr.system_diar
        with reporter.stage("embed (system)"):
            sys_long = filter_short(diar_spoken)
            groups = _group_by_speaker(diar_spoken)
            with reporter.track(
                "embedding (system)", total=len(sys_long) + len(groups)
            ) as bar:
                turns += _prefix_turns(
                    embed_turns(
                        components.embedder, system.samples, SAMPLE_RATE, sys_long,
                        on_advance=bar.advance,
                    ),
                    "sys",
                )
                clusters += cluster_centroids(
                    components.embedder, system.samples, SAMPLE_RATE, groups,
                    on_advance=bar.advance,
                )

    utterances = coalesce_utterances(merge_tracks(tr.mic_utts, tr.system_utts))
    duration = max([duration] + [u.end for u in utterances])
    utterances, cleaned, cleanup_model, suggestions = apply_cleanup(
        utterances, components.cleaner, glossary, reporter
    )
    return Result(
        utterances, turns, clusters, dim, duration, cleaned, cleanup_model, suggestions,
        models_meta=tr.models_meta,
    )


def apply_cleanup(utterances, cleaner, glossary, reporter):
    """Repair only system-track utterances via the cleaner (echo-collapse + guarded span
    repair), which stashes the original in ``raw_text``; timings untouched. The mic ('me')
    track is the clean user audio and passes through. Returns ``(utterances, cleaned,
    cleanup_model)``. Shared by ``process()`` and the ``clean`` retrofit path."""
    sys_idx = [i for i, u in enumerate(utterances) if u.track == "system"]
    if not sys_idx:
        return utterances, False, None, []
    with reporter.stage("cleanup (repair)"):
        res = cleaner.clean([utterances[i] for i in sys_idx], glossary, reporter)
    for j, i in enumerate(sys_idx):
        utterances[i] = res.utterances[j]
    suggestions = list(getattr(res, "suggestions", []))
    for s in suggestions:  # remap local (system-list) segment index → transcript segment index
        s["segment"] = sys_idx[s["segment"]]
    if not res.active:
        return utterances, False, None, suggestions
    return utterances, True, getattr(cleaner, "model_info", None), suggestions


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


BACKENDS = ("local", "deepgram")


def config_error_message(e) -> str:
    """Render a :class:`config.ConfigError` for the exit-2 print. Always names the
    config file (the malformed-TOML message already carries path + line; the typed-
    value errors from the resolvers do not, so the path is prefixed here)."""
    from . import config as config_mod

    path = e.path or str(config_mod.config_path())
    msg = str(e)
    return msg if path in msg else f"config error in {path}: {msg}"


def resolve_backend(flag: str | None = None, env=None, cfg: dict | None = None) -> str:
    """STT backend name: flag > ``STT_BACKEND`` env > ``[stt].backend`` config >
    ``local`` (delegates to ``config.backend``; ``cfg=None`` loads the config file).
    Not validated here — ``run()`` rejects unknown names (exit 2) instead of silently
    falling back."""
    import os

    from . import config as config_mod

    env = os.environ if env is None else env
    cfg = config_mod.load() if cfg is None else cfg
    return config_mod.backend(flag, env, cfg).value


def check_backend(
    flag: str | None = None, env=None, cfg: dict | None = None
) -> tuple[str, str | None]:
    """Resolve the STT backend AND validate it is runnable *now*: a known name, and
    (for ``deepgram``) a present API key (``DEEPGRAM_API_KEY`` env or
    ``[deepgram].api_key`` config). Returns ``(name, error)`` where ``error`` is a
    printable message (exit-2 material) or ``None``. Shared by ``pipeline.run`` and
    ``record.run`` — the record flow must fail BEFORE ffmpeg starts, not after an
    hour-long meeting was recorded. NEVER a silent fallback to local (the user chose
    remote; degrading quietly would betray that)."""
    import os

    from . import config as config_mod

    env = os.environ if env is None else env
    cfg = config_mod.load() if cfg is None else cfg
    name = resolve_backend(flag, env, cfg)
    if name not in BACKENDS:
        return name, (
            f"unknown STT backend {name!r} "
            f"(--backend/STT_BACKEND must be one of: {', '.join(BACKENDS)})"
        )
    if name == "deepgram" and not config_mod.api_key(None, env, cfg).value:
        return name, (
            "the deepgram backend requires an API key — export DEEPGRAM_API_KEY or "
            f"set [deepgram].api_key in {config_mod.config_path()}, or use "
            "--backend local (or unset STT_BACKEND) to transcribe offline; "
            "there is no silent fallback"
        )
    return name, None


def resolve_language(flag: str | None = None, env=None, cfg: dict | None = None) -> str:
    """Remote-STT language: flag > ``STT_LANGUAGE`` env > ``[stt].language`` config >
    ``de`` (delegates to ``config.language``; the local backend is language-agnostic
    and ignores this)."""
    import os

    from . import config as config_mod

    env = os.environ if env is None else env
    cfg = config_mod.load() if cfg is None else cfg
    return config_mod.language(flag, env, cfg).value


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def build_components(models_dir: str, num_speakers: int = -1, cleanup: bool = True) -> Components:
    """Wire the real sherpa models. ``num_speakers`` is the expected speaker count on
    the system track (everyone except the user); -1 means threshold clustering.

    ``cleanup`` wires the LLM cleaner when the GGUF is present (a separate, opt-in flake
    output); if it's absent the cleaner degrades to a no-op at clean() time (warn + raw text),
    so a machine without the LLM model simply produces an uncleaned transcript."""
    from .asr import ParakeetRecognizer
    from .cleanup import ManagedSuggestCleaner, NullCleaner
    from .diarize import OfflineDiarizer
    from .embed import SpeakerEmbedder
    from .vad import SileroVad

    m = Path(models_dir)
    spk = str(m / "spk" / "model.onnx")
    cleaner = NullCleaner()
    if cleanup:
        # hunspell (dicts under hunspell/, in the models-llm output) drives echo-collapse +
        # broken-word flagging + suggestions; the LLM (llm/model.gguf) adds a contextual
        # candidate when present. Broken words are suggested for review, never auto-applied.
        dic = m / "hunspell"
        gguf = m / "llm" / "model.gguf"
        if dic.is_dir():
            cleaner = ManagedSuggestCleaner(
                str(gguf), str(dic),
                model_info=(
                    {"name": "Qwen2.5-7B-Instruct-Q4_K_M", "sha256": _sha256(str(gguf)),
                     "temperature": 0.0}
                    if gguf.exists() else None
                ),
                use_llm=gguf.exists(),
            )
    return Components(
        vad=SileroVad(str(m / "vad" / "silero_vad.onnx")),
        recognizer=ParakeetRecognizer(str(m / "asr")),
        diarizer=OfflineDiarizer(
            str(m / "seg" / "model.int8.onnx"), spk, num_clusters=num_speakers
        ),
        embedder=SpeakerEmbedder(spk),
        cleaner=cleaner,
    )


def build_embed_components(models_dir: str) -> Components:
    """Components for a remote backend: ONLY the local speaker embedder + ``NullCleaner``.

    Remote mode must not pay the local model costs — no Parakeet (~600 MB), no
    diarizer/VAD, no GGUF cleaner. Transcription and diarization happen remotely; the
    embedding pass stays local (vectors never leave the machine) and the transcript ships
    the service's text verbatim (``cleaned: false``)."""
    from .cleanup import NullCleaner
    from .embed import SpeakerEmbedder

    return Components(
        vad=None,
        recognizer=None,
        diarizer=None,
        embedder=SpeakerEmbedder(str(Path(models_dir) / "spk" / "model.onnx")),
        cleaner=NullCleaner(),
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
    bundle: bool | None = None,
    started_at=None,
    reporter=None,
    num_speakers: int = -1,
    cleanup: bool | None = None,
    backend: str | None = None,
    language: str | None = None,
) -> int:
    import os
    from datetime import datetime, timezone

    from . import config as config_mod

    if audio is None:
        print("process: no input given (expected a directory or a .wav file)")
        return 2

    models_dir = os.environ.get("MEETSCRIBE_MODELS")
    if not models_dir:
        print("MEETSCRIBE_MODELS is not set — run via `nix run` or the wrapper")
        return 2

    from .progress import NullReporter

    reporter = reporter or NullReporter()

    # flag > env > config > default (bundle/cleanup: None = "flag not given", contract
    # with the CLI). EVERY config-backed value resolves inside this guard — a
    # malformed config, bad backend name, missing key, or wrongly-typed value fails
    # fast (exit 2, never a traceback) — the record flow runs the same checks before
    # recording even starts.
    try:
        cfg = config_mod.load()
        for msg in config_mod.load_warnings(cfg):  # unknown keys, api_key perms
            reporter.warn(msg)
        backend_name, backend_err = check_backend(backend, cfg=cfg)
        bundle = config_mod.bundle(bundle, os.environ, cfg).value
        cleanup = config_mod.cleanup(cleanup, os.environ, cfg).value
        language = resolve_language(language, cfg=cfg)
        dg_key = config_mod.api_key(None, os.environ, cfg).value
        if backend_name == "deepgram":
            # An [deepgram].api_key_cmd resolves to an unexecuted marker — fetch
            # the material here (and only for the deepgram backend: local runs
            # must never execute a keyring command). Failure lands in the same
            # exit-2 guard as every other config problem.
            dg_key = config_mod.fetch_api_key(dg_key)
    except config_mod.ConfigError as e:
        print(config_error_message(e))
        return 2
    if backend_err:
        print(backend_err)
        return 2

    mic_wav, system_wav = resolve_inputs(audio)
    if not mic_wav and not system_wav:
        print(f"no mic.wav / system.wav found under {audio}")
        return 2

    out = Path(out_dir or audio)
    out.mkdir(parents=True, exist_ok=True)
    meeting_id = out.name if out.name else f"{datetime.now(timezone.utc):%Y-%m-%dT%H-%M-%S}"

    from . import glossary as glossary_mod

    backend_obj = None
    if backend_name == "deepgram":
        from .deepgram import DeepgramBackend, DeepgramConfig

        backend_obj = DeepgramBackend(
            # Both values were resolved inside the guarded block above;
            # check_backend guaranteed the key exists in one of the layers.
            DeepgramConfig(api_key=dg_key, language=language)
        )
        if num_speakers != -1:
            # Deepgram's diarizer takes no forced cluster count — say so instead of
            # silently discarding `--speakers N` / the participant-count answer.
            reporter.warn(
                "--speakers / the participant count only steers the LOCAL diarizer; "
                "the deepgram backend infers the speaker count itself — ignoring it"
            )
        # Remote mode loads ONLY the local speaker-embedding model — never Parakeet,
        # the diarizer/VAD, or the GGUF cleaner.
        with reporter.stage("loading models"):
            components = build_embed_components(models_dir)
    else:
        # Loading the ~600 MB Parakeet model takes several seconds — show a spinner so
        # the record→process transition isn't a silent gap.
        with reporter.stage("loading models"):
            components = build_components(models_dir, num_speakers)
    if not cleanup:  # --no-cleanup / [output].cleanup=false: force the no-op cleaner
        from .cleanup import NullCleaner

        components.cleaner = NullCleaner()
    glossary = glossary_mod.effective()
    from .deepgram import DeepgramError

    try:
        result = process(
            mic_wav, system_wav, components,
            reporter=reporter, glossary=glossary, backend=backend_obj,
        )
    except DeepgramError as e:
        # e.g. a present-but-invalid key (401) or exhausted retries — a clean exit 2
        # with the actionable message, not a traceback.
        print(str(e))
        return 2

    spk_model = str(Path(models_dir) / "spk" / "model.onnx")
    started_iso, ended_iso = _window(started_at, result.duration_s, mic_wav, system_wav)
    meta = build_meta(
        {
            # Embedding identity is added centrally here (embeddings are local in every
            # backend); the ASR/diarization identity comes from the backend's result.
            "embedding_model": "3dspeaker_campplus_sv_zh_en_16k",
            "embedding_model_sha256": _sha256(spk_model),
            **result.models_meta,
        },
        embedding_dim=result.dim,
        meeting_id=meeting_id,
        started_at=started_iso,
        ended_at=ended_iso,
        duration_s=result.duration_s,
        cleaned=result.cleaned,
        cleanup_model=result.cleanup_model,
        backend=backend_name,
    )
    write_transcript(
        out / "transcript.json", meeting_id, result.duration_s, result.utterances,
        cleaned=result.cleaned, suggestions=result.suggestions,
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


def clean_existing(audio_dir: str, out_dir: str | None = None, reporter=None) -> int:
    """Run the LLM cleanup over an EXISTING transcript, non-destructively.

    Reads ``<dir>/transcript.json`` (no audio, no re-ASR), cleans the system-track text, and
    writes ``<dir>-cleanup/`` with the cleaned transcript + copied embeddings + updated meta.
    The original directory is never modified — this is the A/B tool for old meetings."""
    import json
    import os
    import shutil

    from . import config as config_mod
    from . import glossary as glossary_mod
    from .output import read_transcript, write_meta, write_transcript
    from .progress import NullReporter

    reporter = reporter or NullReporter()
    # Acceptance: EVERY subcommand exits 2 on a broken config (clean doesn't use
    # config values today, but silently running against a broken file would hide it).
    try:
        config_mod.load()
    except config_mod.ConfigError as e:
        print(config_error_message(e))
        return 2
    src = Path(audio_dir)
    transcript = src / "transcript.json"
    if not transcript.exists():
        print(f"no transcript.json found in {src}")
        return 2

    models_dir = os.environ.get("MEETSCRIBE_MODELS")
    if not models_dir:
        print("MEETSCRIBE_MODELS is not set — run via `nix run` or the wrapper")
        return 2

    meeting_id, duration_s, utterances = read_transcript(transcript)
    with reporter.stage("loading models"):
        components = build_components(models_dir, -1)
    glossary = glossary_mod.effective()
    utterances, cleaned, cleanup_model, suggestions = apply_cleanup(
        list(utterances), components.cleaner, glossary, reporter
    )

    out = Path(out_dir) if out_dir else src.parent / f"{src.name}-cleanup"
    out.mkdir(parents=True, exist_ok=True)
    write_transcript(out / "transcript.json", meeting_id, duration_s, utterances,
                     cleaned=cleaned, suggestions=suggestions)
    if (src / "embeddings.npz").exists():  # unaffected by text cleanup → copy verbatim
        shutil.copy(src / "embeddings.npz", out / "embeddings.npz")
    meta = json.loads((src / "meta.json").read_text()) if (src / "meta.json").exists() else {}
    meta["cleaned"] = cleaned
    if cleanup_model is not None:
        meta["cleanup_model"] = cleanup_model
    write_meta(out / "meta.json", meta)

    print(f"wrote cleaned transcript to {out}")
    return 0
