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
    p_record.add_argument(
        "--upload", action=argparse.BooleanOptionalAction, default=None,
        help="upload the .mscribe bundle to borderless-knowledge after processing "
             "(default: off, or [bk].auto_upload in the config; --no-upload to skip)",
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
    p_process.add_argument(
        "--upload", action=argparse.BooleanOptionalAction, default=None,
        help="upload the .mscribe bundle to borderless-knowledge after processing "
             "(default: off, or [bk].auto_upload in the config; --no-upload to skip)",
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

    p_upload = sub.add_parser(
        "upload", help="upload a meeting's .mscribe bundle to borderless-knowledge"
    )
    p_upload.add_argument(
        "dir", help="meeting directory (bundled first if no .mscribe exists yet)"
    )

    p_status = sub.add_parser(
        "status", help="list local meetings and their bk upload state"
    )
    p_status.add_argument(
        "dir", nargs="?", default=None,
        help="directory to scan for meetings (default: the configured meetings dir)",
    )
    p_status.add_argument(
        "--offline", action="store_true",
        help="show cached workflow state only, never contact bk",
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
            ("bk_base_url", config_mod.bk_base_url(None, env, cfg)),
            ("bk_token", config_mod.bk_token(None, env, cfg)),
            ("bk_auto_upload", config_mod.bk_auto_upload(None, env, cfg)),
        ]
    except config_mod.ConfigError as e:
        print(f"config: {path} (invalid)", file=sys.stderr)
        print(str(e), file=sys.stderr)
        return 2

    secret_rows = {"api_key", "bk_token"}  # masked — the value itself never hits stdout
    print(f"config: {path} ({'exists' if path.exists() else 'missing'})")
    for name, resolved in rows:
        value = resolved.value
        if isinstance(value, config_mod.SecretCmd):
            # never execute here, don't echo the command — the marker's label names
            # the knob ("[deepgram].api_key_cmd" → "(via api_key_cmd)", etc.)
            value = f"(via {value.label.rsplit('.', 1)[-1]})"
        elif name in secret_rows and value:
            value = f"{str(value)[:3]}…****"  # never print the secret itself
        if value is None:
            display = "(not set)"
        elif isinstance(value, bool):
            display = "true" if value else "false"
        else:
            display = str(value)
        print(f"{name:<14} {display}  ({resolved.origin})")
    return 0


def _resolve_bk_client(env, cfg):
    """Resolve ``[bk]`` into a ready client (the module-level ``bk.BkClient`` seam —
    tests monkeypatch that one symbol). A ``token_cmd`` is executed HERE, because the
    secret material is actually needed now. Raises :class:`config.ConfigError` when
    bk is not configured; the caller decides what that means (exit 2 for ``upload``,
    a stale-state warning for ``status``)."""
    from . import bk as bk_mod
    from . import config as config_mod

    base_url = config_mod.bk_base_url(None, env, cfg).value
    if not base_url:
        raise config_mod.ConfigError(
            "bk is not configured — set [bk].base_url in "
            f"{config_mod.config_path()} (or MEETSCRIBE_BK_URL)"
        )
    token = config_mod.fetch_secret(config_mod.bk_token(None, env, cfg).value)
    if not token:
        raise config_mod.ConfigError(
            "bk needs a token — set [bk].token or [bk].token_cmd in "
            f"{config_mod.config_path()} (or MEETSCRIBE_BK_TOKEN)"
        )
    return bk_mod.BkClient(bk_mod.BkConfig(base_url, token))


def _run_upload(dir_arg: str) -> int:
    """The ``upload <dir>`` subcommand — the retry/backfill path for auto-upload.

    Unlike the record/process auto-upload (which warns and exits 0 because the
    artifacts are the task), here the upload IS the task: config problems exit 2,
    upload failures exit 1, with the actionable message on stdout."""
    import json
    import os
    from datetime import datetime
    from pathlib import Path

    from . import bk as bk_mod
    from . import config as config_mod
    from .output import bundle_dir, default_bundle_name
    from .pipeline import config_error_message

    try:
        cfg = config_mod.load()
        client = _resolve_bk_client(os.environ, cfg)
    except config_mod.ConfigError as e:
        print(config_error_message(e))
        return 2

    src = Path(dir_arg)
    try:
        meta = json.loads((src / "meta.json").read_text())
    except (OSError, ValueError) as e:
        print(f"upload: cannot read meta.json in {src}: {e}")
        return 1

    bundle_path = src / default_bundle_name(meta)
    if not bundle_path.exists():
        existing = sorted(src.glob("*.mscribe"))
        if existing:
            bundle_path = existing[0]
        else:
            bundle_dir(src, bundle_path)
            print(f"wrote {bundle_path.name}")

    try:
        caps = client.capabilities()
        message = bk_mod.preflight(caps, bundle_path.stat().st_size)
        if message:
            print(f"upload refused: {message}")
            return 1
        accepted = client.upload_bundle(bundle_path, meeting_id=meta["meeting_id"])
    except bk_mod.BkError as e:
        print(f"upload failed: {e}")
        return 1

    now = datetime.now().astimezone().isoformat()
    bk_mod.write_workflow_ref(
        src,
        {
            "workflow_id": accepted.get("workflow_id"),
            "state_url": accepted.get("state_url"),
            "web_url": accepted.get("web_url"),
            "uploaded_at": now,
            # a fresh 202 means the workflow just started (contract endpoint 3)
            "state": "processing",
            "checked_at": now,
        },
    )
    print(f"uploaded: workflow {accepted.get('workflow_id')}")
    print(accepted.get("web_url"))
    return 0


#: Workflow states that never change again — `status` shows them without asking bk.
_TERMINAL_STATES = frozenset({"done", "failed"})


def _run_status(dir_arg: str | None, offline: bool = False) -> int:
    """The ``status [dir] [--offline]`` subcommand: one line per local meeting.

    Stdout is the data output (plain prints); warnings go to stderr. A refresh
    failure degrades to the cached state + ``(stale)`` — never a non-zero exit,
    status is read-only reporting."""
    import json
    import os
    from datetime import datetime
    from pathlib import Path

    from . import bk as bk_mod
    from . import config as config_mod
    from .pipeline import config_error_message

    env = os.environ
    try:
        cfg = config_mod.load()
        root = Path(dir_arg) if dir_arg else config_mod.meetings_dir(None, env, cfg).value
    except config_mod.ConfigError as e:
        print(config_error_message(e))
        return 2

    if (root / "meta.json").exists():
        meetings = [root]
    elif root.is_dir():
        meetings = sorted(
            p for p in root.iterdir() if p.is_dir() and (p / "meta.json").exists()
        )
    else:
        meetings = []
    if not meetings:
        print(f"no meetings found in {root}")
        return 0

    # The client is built lazily (only when a non-terminal ref needs a refresh) and
    # once: an unconfigured/broken bk warns a single time, every refresh goes stale.
    client_slot: list = []

    def get_client():
        if not client_slot:
            try:
                client_slot.append(_resolve_bk_client(env, cfg))
            except config_mod.ConfigError as e:
                print(f"warning: cannot refresh from bk: {e}", file=sys.stderr)
                client_slot.append(None)
        return client_slot[0]

    for d in meetings:
        try:
            meta = json.loads((d / "meta.json").read_text())
        except (OSError, ValueError):
            print(f"{d.name}  (unreadable meta.json)")
            continue

        bundled = "bundled=yes" if any(d.glob("*.mscribe")) else "bundled=no"

        ref = bk_mod.read_workflow_ref(d)
        web_url = None
        if ref is None:
            upload_state = "not uploaded"
        else:
            state = str(ref.get("state") or "unknown")
            workflow_id = ref.get("workflow_id")
            if state in _TERMINAL_STATES or offline or not workflow_id:
                upload_state = state
            else:
                client = get_client()
                if client is None:
                    upload_state = f"{state} (stale)"
                else:
                    try:
                        wf = client.workflow(workflow_id)
                    except bk_mod.BkError as e:
                        print(
                            f"warning: could not refresh workflow {workflow_id}: {e}",
                            file=sys.stderr,
                        )
                        upload_state = f"{state} (stale)"
                    else:
                        state = str(wf.get("state") or state)
                        ref = {
                            **ref,
                            "state": state,
                            "web_url": wf.get("web_url") or ref.get("web_url"),
                            "checked_at": datetime.now().astimezone().isoformat(),
                        }
                        bk_mod.write_workflow_ref(d, ref)
                        upload_state = state
            # Contract forward-compat rule 1: on an unknown state the client MUST
            # fall back to showing web_url (bk may add workflow steps without
            # breaking older clients). Outside the self-explanatory processing/done
            # states the bk page is where the user can act — awaiting_review (the
            # 7-day gate; task rendering is Phase 4) and failed included.
            if state not in ("processing", "done"):
                web_url = ref.get("web_url")

        duration = meta.get("duration_s")
        dur = f"{duration:g}s" if isinstance(duration, (int, float)) else "?"
        line = (
            f"{meta.get('meeting_id', d.name)}  {meta.get('started_at', '?')}  {dur}  "
            f"{meta.get('backend', 'local')}  {bundled}  {upload_state}"
        )
        if web_url:
            line += f"  {web_url}"
        print(line)
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
            upload=getattr(args, "upload", None),
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
            upload=getattr(args, "upload", None),
        )
    if command == "clean":
        from . import pipeline

        return pipeline.clean_existing(
            audio_dir=args.dir, out_dir=getattr(args, "out", None), reporter=reporter,
        )
    if command == "config":
        return _run_config(getattr(args, "action", None))
    if command == "upload":
        return _run_upload(args.dir)
    if command == "status":
        return _run_status(
            getattr(args, "dir", None), offline=getattr(args, "offline", False)
        )
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
