"""Terminal UI seam. The pipeline stays pure; it calls an injected :class:`Reporter`.

``NullReporter`` (the default everywhere) is a silent no-op so tests run untouched and fast.
``RichReporter`` renders spinners / progress bars / tables via ``rich`` to stderr, and rich
auto-degrades to plain text on a non-TTY (pipes, CI). All UI goes to stderr so stdout stays clean.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from typing import ContextManager, Protocol


@dataclass
class Summary:
    duration_s: float
    n_segments: int
    # (speaker, talk_time_seconds, track)
    speakers: list[tuple[str, float, str]] = field(default_factory=list)


class TrackHandle(Protocol):
    def advance(self, n: int = 1) -> None: ...
    def __enter__(self) -> "TrackHandle": ...
    def __exit__(self, *exc) -> None: ...


class Reporter(Protocol):
    def stage(self, label: str) -> ContextManager[None]: ...
    def track(self, label: str, total: int) -> TrackHandle: ...
    def info(self, msg: str) -> None: ...
    def summary(self, summary: Summary) -> None: ...


# ---- Null (default) ------------------------------------------------------------------

class _NullTrack:
    def advance(self, n: int = 1) -> None:
        pass

    def __enter__(self) -> "_NullTrack":
        return self

    def __exit__(self, *exc) -> None:
        pass


class _NoopMeters:
    def update(self, channel: int, dbfs: float) -> None:
        pass

    def mark_stopping(self) -> None:
        pass

    def __enter__(self) -> "_NoopMeters":
        return self

    def __exit__(self, *exc) -> None:
        pass


class NullReporter:
    """Silent no-op reporter — the default injected into the pipeline and recorder."""

    def stage(self, label: str) -> ContextManager[None]:
        return contextlib.nullcontext()

    def track(self, label: str, total: int) -> _NullTrack:
        return _NullTrack()

    def info(self, msg: str) -> None:
        pass

    def summary(self, summary: Summary) -> None:
        pass

    def supports_meters(self) -> bool:
        return False

    def meters(self, labels: dict[int, str]) -> _NoopMeters:
        return _NoopMeters()


# ---- Rich ----------------------------------------------------------------------------

def _fmt_duration(seconds: float) -> str:
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {sec}s"
    if m:
        return f"{m}m {sec}s"
    return f"{sec}s"


def rms_to_meter(dbfs: float, width: int = 8, floor: float = -60.0) -> str:
    """Render a dBFS value as a fixed-width block bar (``floor``..0 dB → empty..full)."""
    if dbfs <= floor:
        frac = 0.0
    elif dbfs >= 0.0:
        frac = 1.0
    else:
        frac = (dbfs - floor) / (0.0 - floor)
    filled = round(frac * width)
    return "▇" * filled + "▁" * (width - filled)


class _RichTrack:
    def __init__(self, progress, task_id) -> None:
        self._progress = progress
        self._task_id = task_id

    def advance(self, n: int = 1) -> None:
        self._progress.advance(self._task_id, n)

    def __enter__(self) -> "_RichTrack":
        return self

    def __exit__(self, *exc) -> None:
        self._progress.stop()


class _CountTrack:
    """Non-TTY fallback: no live bar, advancing is a no-op (a start line was already printed)."""

    def advance(self, n: int = 1) -> None:
        pass

    def __enter__(self) -> "_CountTrack":
        return self

    def __exit__(self, *exc) -> None:
        pass


class RichReporter:
    def __init__(self, file=None, force_terminal=None, verbose: bool = False) -> None:
        from rich.console import Console

        if file is None:
            self.console = Console(stderr=True, force_terminal=force_terminal)
        else:
            self.console = Console(file=file, force_terminal=force_terminal)
        self.verbose = verbose

    @contextlib.contextmanager
    def stage(self, label: str):
        start = time.monotonic()
        if self.console.is_terminal:
            with self.console.status(f"{label}…", spinner="dots"):
                yield
        else:
            yield
        elapsed = time.monotonic() - start
        self.console.print(f"✓ {label}  [dim]{elapsed:.1f}s[/dim]")

    def track(self, label: str, total: int) -> TrackHandle:
        if self.console.is_terminal:
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                TextColumn,
                TimeRemainingColumn,
            )

            progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
                console=self.console,
            )
            progress.start()
            task_id = progress.add_task(label, total=total)
            return _RichTrack(progress, task_id)
        self.console.print(f"{label} (0/{total})")
        return _CountTrack()

    def info(self, msg: str) -> None:
        if self.verbose:
            self.console.print(f"[dim]{msg}[/dim]")

    def summary(self, summary: Summary) -> None:
        from rich.table import Table

        self.console.print(
            f"meeting  {_fmt_duration(summary.duration_s)} · "
            f"{summary.n_segments} segments · {len(summary.speakers)} speakers"
        )
        table = Table(show_header=True, header_style="bold")
        table.add_column("speaker")
        table.add_column("talk time", justify="right")
        table.add_column("track")
        for speaker, talk, track in summary.speakers:
            table.add_row(speaker, _fmt_duration(talk), track)
        self.console.print(table)

    def supports_meters(self) -> bool:
        return self.console.is_terminal

    @contextlib.contextmanager
    def meters(self, labels: dict[int, str]):
        """Live two-track level meters, refreshed in place. A track flatlined at silence for
        >3 s turns red — a dead track is visible during the meeting."""
        from rich.live import Live
        from rich.table import Table
        from rich.text import Text

        state = {ch: {"dbfs": float("-inf"), "active": time.monotonic()} for ch in labels}
        start = time.monotonic()
        flags = {"stopping": False}

        def render():
            grid = Table.grid(padding=(0, 1))
            el = int(time.monotonic() - start)
            if flags["stopping"]:
                grid.add_row(
                    Text("⏹ recording stopped — finalizing…", style="bold yellow")
                )
            else:
                grid.add_row(
                    Text(f"● recording {el // 60:02d}:{el % 60:02d}   (Ctrl-C to stop)", style="bold")
                )
            now = time.monotonic()
            for ch, label in labels.items():
                st = state[ch]
                db = st["dbfs"]
                silent = (now - st["active"]) > 3.0
                db_txt = " -inf" if db == float("-inf") else f"{db:5.0f} dB"
                grid.add_row(
                    Text(
                        f"  {label:7s} {rms_to_meter(db, 12)}  {db_txt}",
                        style="red" if silent else "green",
                    )
                )
            return grid

        if not self.console.is_terminal:
            yield _NoopMeters()
            return

        live = Live(render(), console=self.console, refresh_per_second=8, transient=True)
        live.start()

        class _Upd:
            def update(self, channel: int, dbfs: float) -> None:
                st = state.get(channel)
                if st is None:
                    return
                st["dbfs"] = dbfs
                if dbfs > -60.0:
                    st["active"] = time.monotonic()
                live.update(render())

            def mark_stopping(self) -> None:
                flags["stopping"] = True
                live.update(render())

        try:
            yield _Upd()
        finally:
            live.stop()
