"""Command-line entrypoint: ``record`` | ``process`` | ``doctor`` | ``config`` | …

Target UX (see what-we-build.md §1):

    meetscribe            # record (Ctrl-C stops) + process
    meetscribe process path/to/audio.wav
    meetscribe doctor     # preflight audio-setup checks

The bare invocation defaults to ``record``. Argument parsing lives here and is
kept side-effect free so it is unit-testable; the subcommands delegate to their
respective modules.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from . import __version__


def _add_backend_flags(p: argparse.ArgumentParser) -> None:
    """STT backend selection, shared by ``record`` and ``process``.

    Defaults are ``None`` (NOT "local"/"de") so the pipeline can apply the
    flag > env (``STT_BACKEND``/``STT_LANGUAGE``) > default precedence — a concrete
    parser default would shadow the env variables."""
    p.add_argument(
        "--backend", choices=("local", "deepgram"), default=None,
        help="transcription backend (default: $STT_BACKEND or local). 'deepgram' uploads "
             "the meeting AUDIO to the Deepgram API (requires DEEPGRAM_API_KEY); "
             "embeddings are always computed locally",
    )
    p.add_argument(
        "--language", default=None, metavar="LANG",
        help="spoken language for the remote backend, e.g. de/en/multi "
             "(default: $STT_LANGUAGE or de); ignored by the local backend",
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser. Pure — no side effects, for easy testing."""
    parser = argparse.ArgumentParser(
        prog="meetscribe",
        description="Record a meeting (mic + system audio) and produce a "
        "diarized transcript with speaker embeddings, fully offline on CPU.",
    )
    parser.add_argument("--version", action="version", version=f"meetscribe {__version__}")
    parser.add_argument(
        "--quiet", action="store_true", help="suppress the terminal UI (only errors)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="show extra per-stage detail"
    )

    sub = parser.add_subparsers(dest="command")

    p_record = sub.add_parser("record", help="record mic + system audio, then process")
    p_record.add_argument(
        "-o", "--out", default=None,
        help="output directory for artifacts (default: a new meetscribe-<timestamp>/ "
             "under the meetings dir — $XDG_DATA_HOME/meetscribe/meetings, or "
             "[storage].meetings_dir in the config)",
    )
    p_record.add_argument(
        "--bundle", action=argparse.BooleanOptionalAction, default=None,
        help="emit a single meeting-<id>.mscribe upload bundle (default: on, or "
             "[output].bundle in the config; --no-bundle to skip)",
    )
    p_record.add_argument(
        "--system-source", default=None, metavar="NAME",
        help="capture system audio from this exact source (e.g. a specific sink's .monitor); "
             "overrides the default-sink auto-detection",
    )
    p_record.add_argument(
        "--cleanup", action=argparse.BooleanOptionalAction, default=None,
        help="run the LLM transcript-cleanup pass (default: on, or [output].cleanup "
             "in the config; --no-cleanup emits raw ASR text)",
    )
    _add_backend_flags(p_record)

    p_process = sub.add_parser("process", help="process existing audio into artifacts")
    p_process.add_argument("audio", nargs="?", help="path to an existing recording to process")
    p_process.add_argument(
        "-o", "--out", default=None,
        help="output directory for artifacts",
    )
    p_process.add_argument(
        "--bundle", action=argparse.BooleanOptionalAction, default=None,
        help="emit a single meeting-<id>.mscribe upload bundle (default: on, or "
             "[output].bundle in the config; --no-bundle to skip)",
    )
    p_process.add_argument(
        "--speakers", type=int, default=None, metavar="N",
        help="number of people in the meeting, including you — same question the "
             "record flow asks (default: automatic threshold clustering); local "
             "backend only — deepgram infers the count itself (warned + ignored)",
    )
    p_process.add_argument(
        "--cleanup", action=argparse.BooleanOptionalAction, default=None,
        help="run the LLM transcript-cleanup pass (default: on, or [output].cleanup "
             "in the config; --no-cleanup emits raw ASR text)",
    )
    _add_backend_flags(p_process)

    p_bundle = sub.add_parser("bundle", help="zip an artifact directory into one .mscribe")
    p_bundle.add_argument(
        "dir", help="directory holding transcript.json / embeddings.npz / meta.json"
    )
    p_bundle.add_argument(
        "-o", "--out", default=None,
        help="output path (default: ./meeting-<id>.mscribe)",
    )

    p_clean = sub.add_parser(
        "clean", help="re-run LLM cleanup on an existing transcript (non-destructive)"
    )
    p_clean.add_argument("dir", help="directory holding transcript.json")
    p_clean.add_argument(
        "-o", "--out", default=None,
        help="output directory (default: <dir>-cleanup)",
    )

    sub.add_parser("doctor", help="check the audio setup is ready to record")

    p_config = sub.add_parser(
        "config", help="show effective configuration (init/path subactions)"
    )
    p_config.add_argument(
        "action", nargs="?", choices=("init", "path"), default=None,
        help="init: write a commented template config (refuses to overwrite); "
             "path: print the config file path; omit to show effective values",
    )

    return parser


def _run_config(action: str | None) -> int:
    """The ``config`` subcommand. Unlike record/process, stdout IS the data
    output here, so plain ``print`` is correct (and keeps it ``--quiet``-proof)."""
    import os

    from . import config as config_mod

    path = config_mod.config_path()

    if action == "path":
        print(path)
        return 0

    if action == "init":
        if path.exists():
            print(f"refusing to overwrite existing config: {path}", file=sys.stderr)
            return 1
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(config_mod.TEMPLATE, encoding="utf-8")
        path.chmod(0o600)  # the template has an api_key slot — private from day one
        print(f"wrote {path}")
        return 0

    # Bare `config`: the effective values table — key, value, origin — plus the
    # config path in use and whether it exists.
    env = os.environ
    try:
        cfg = config_mod.load(path)
        rows = [
            ("backend", config_mod.backend(None, env, cfg)),
            ("language", config_mod.language(None, env, cfg)),
            ("api_key", config_mod.api_key(None, env, cfg)),
            ("meetings_dir", config_mod.meetings_dir(None, env, cfg)),
            ("system_source", config_mod.system_source(None, env, cfg)),
            ("bundle", config_mod.bundle(None, env, cfg)),
            ("cleanup", config_mod.cleanup(None, env, cfg)),
        ]
    except config_mod.ConfigError as e:
        print(f"config: {path} (invalid)", file=sys.stderr)
        print(str(e), file=sys.stderr)
        return 2

    print(f"config: {path} ({'exists' if path.exists() else 'missing'})")
    for name, resolved in rows:
        value = resolved.value
        if name == "api_key" and value:
            value = f"{str(value)[:3]}…****"  # never print the secret itself
        if value is None:
            display = "(not set)"
        elif isinstance(value, bool):
            display = "true" if value else "false"
        else:
            display = str(value)
        print(f"{name:<14} {display}  ({resolved.origin})")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entrypoint. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Bare invocation defaults to the record→process flow.
    command = args.command or "record"

    from .progress import NullReporter, RichReporter

    reporter = NullReporter() if args.quiet else RichReporter(verbose=args.verbose)

    if command == "doctor":
        from . import doctor

        return doctor.run(reporter=reporter)
    if command == "record":
        from . import record

        return record.run(
            out_dir=getattr(args, "out", None),
            # Bare `meetscribe` never runs the subparser, so the fallbacks must
            # match the subparser defaults: None = "resolve from config".
            bundle=getattr(args, "bundle", None),
            system_source=getattr(args, "system_source", None),
            reporter=reporter,
            cleanup=getattr(args, "cleanup", None),
            backend=getattr(args, "backend", None),
            language=getattr(args, "language", None),
        )
    if command == "process":
        from . import pipeline

        from .record import speakers_from_count

        speakers = getattr(args, "speakers", None)
        return pipeline.run(
            audio=getattr(args, "audio", None),
            out_dir=getattr(args, "out", None),
            bundle=getattr(args, "bundle", None),
            reporter=reporter,
            num_speakers=-1 if speakers is None else speakers_from_count(speakers),
            cleanup=getattr(args, "cleanup", None),
            backend=getattr(args, "backend", None),
            language=getattr(args, "language", None),
        )
    if command == "clean":
        from . import pipeline

        return pipeline.clean_existing(
            audio_dir=args.dir, out_dir=getattr(args, "out", None), reporter=reporter,
        )
    if command == "config":
        return _run_config(getattr(args, "action", None))
    if command == "bundle":
        import json
        from pathlib import Path

        from . import config as config_mod
        from .output import bundle_dir, default_bundle_name

        # Acceptance: EVERY subcommand exits 2 on a broken config (bundle doesn't
        # use config values today, but a broken file must never pass silently).
        try:
            config_mod.load()
        except config_mod.ConfigError as e:
            from .pipeline import config_error_message

            print(config_error_message(e))
            return 2

        src = Path(args.dir)
        meta = json.loads((src / "meta.json").read_text())
        out = args.out or default_bundle_name(meta)
        bundle_dir(src, out)
        print(f"wrote {out}")
        return 0

    parser.error(f"unknown command: {command}")
    return 2  # unreachable; parser.error exits


if __name__ == "__main__":
    sys.exit(main())
