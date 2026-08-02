"""Command-line entrypoint: ``record`` | ``process`` | ``doctor``.

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
        help="output directory for artifacts (default: ./meetscribe-<timestamp>)",
    )

    p_process = sub.add_parser("process", help="process existing audio into artifacts")
    p_process.add_argument("audio", nargs="?", help="path to an existing recording to process")
    p_process.add_argument(
        "-o", "--out", default=None,
        help="output directory for artifacts",
    )

    sub.add_parser("doctor", help="check the audio setup is ready to record")

    return parser


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

        return doctor.run()
    if command == "record":
        from . import record

        return record.run(out_dir=getattr(args, "out", None), reporter=reporter)
    if command == "process":
        from . import pipeline

        return pipeline.run(
            audio=getattr(args, "audio", None),
            out_dir=getattr(args, "out", None),
            reporter=reporter,
        )

    parser.error(f"unknown command: {command}")
    return 2  # unreachable; parser.error exits


if __name__ == "__main__":
    sys.exit(main())
