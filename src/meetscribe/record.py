"""Dual-track recording — the ONLY platform-specific module (what-we-build.md §5).

Mic and system output are captured as two separate mono files (never mixed, never stereo). The
command-building and device-matching helpers are pure and unit-tested for both platforms; the
actual ffmpeg invocation differs per OS. Device indices are resolved by NAME at runtime — never
hard-coded, because avfoundation indices shift.
"""

from __future__ import annotations

import platform
import re
import signal
import subprocess
from pathlib import Path

import numpy as np

from .audio import load_wav_f32

# ---- macOS (avfoundation) --------------------------------------------------------------

_AVF_LINE = re.compile(r"\[(\d+)\]\s+(.+?)\s*$")


def parse_avfoundation_devices(stderr_text: str) -> list[tuple[int, str]]:
    """Parse the AUDIO device list from ``ffmpeg -f avfoundation -list_devices true -i ""``."""
    devices: list[tuple[int, str]] = []
    in_audio = False
    for line in stderr_text.splitlines():
        if "audio devices:" in line:
            in_audio = True
            continue
        if "video devices:" in line:
            in_audio = False
            continue
        if not in_audio:
            continue
        # strip the "[AVFoundation indev @ 0x..] " prefix, then match "[N] Name".
        tail = re.sub(r"^\[AVFoundation[^\]]*\]\s*", "", line)
        m = _AVF_LINE.match(tail)
        if m:
            devices.append((int(m.group(1)), m.group(2)))
    return devices


def match_device(devices: list[tuple[int, str]], name: str) -> int:
    for idx, dev_name in devices:
        if dev_name == name:
            return idx
    raise ValueError(f"audio device not found: {name}")


def build_macos_ffmpeg_cmd(
    aggregate_index: int, mic_out: str, system_out: str
) -> list[str]:
    """One avfoundation input against the aggregate device, split by channel with ``pan``.

    Convention for the ``meetscribe`` aggregate (BlackHole 2ch + Mic): channels 0/1 are the
    system output (downmixed to mono), channel 2 is the microphone.
    """
    return [
        "ffmpeg", "-y", "-hide_banner",
        "-f", "avfoundation", "-i", f":{aggregate_index}",
        "-filter_complex",
        "[0:a]pan=mono|c0=c2[mic];[0:a]pan=mono|c0=c0+c1[sys]",
        "-map", "[mic]", "-ar", "16000", "-ac", "1", mic_out,
        "-map", "[sys]", "-ar", "16000", "-ac", "1", system_out,
    ]


# ---- Linux (PulseAudio / PipeWire) -----------------------------------------------------

def parse_pw_sources(text: str) -> list[str]:
    """Extract source names from ``pactl list short sources`` (tab-separated)."""
    names: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) >= 2:
            names.append(fields[1])
    return names


def find_monitor_source(names: list[str]) -> str:
    for name in names:
        if name.endswith(".monitor"):
            return name
    raise ValueError("no PipeWire/Pulse monitor source found")


def build_linux_ffmpeg_cmd(
    mic_source: str, monitor_source: str, mic_out: str, system_out: str
) -> list[str]:
    """Two ``-f pulse`` inputs (mic + sink monitor) into two mono 16 kHz files."""
    return [
        "ffmpeg", "-y", "-hide_banner",
        "-f", "pulse", "-i", mic_source,
        "-f", "pulse", "-i", monitor_source,
        "-map", "0:a", "-ar", "16000", "-ac", "1", mic_out,
        "-map", "1:a", "-ar", "16000", "-ac", "1", system_out,
    ]


# ---- RMS / silence / graceful stop -----------------------------------------------------

def rms(samples: np.ndarray) -> float:
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))


def warn_if_silent(path: str | Path) -> str | None:
    samples, _ = load_wav_f32(path)
    if rms(samples) == 0.0:
        return f"track {path} is silent (RMS=0) — check the audio setup (see doctor)"
    return None


def stop_ffmpeg(proc) -> None:
    """Ask ffmpeg to quit cleanly by writing ``q`` to stdin (a hard kill corrupts WAV headers)."""
    try:
        if proc.stdin is not None:
            proc.stdin.write(b"q")
            proc.stdin.flush()
    except (BrokenPipeError, ValueError):
        pass
    try:
        proc.wait(timeout=10)
    except Exception:
        pass


# ---- Orchestration ---------------------------------------------------------------------

def _list_linux_sources() -> list[str]:
    """Best-effort source enumeration (pactl, else pw-dump)."""
    try:
        res = subprocess.run(
            ["pactl", "list", "short", "sources"],
            capture_output=True, text=True, timeout=5,
        )
        names = parse_pw_sources(res.stdout)
        if names:
            return names
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    try:
        import json

        res = subprocess.run(["pw-dump"], capture_output=True, text=True, timeout=5)
        names = []
        for node in json.loads(res.stdout):
            props = (node.get("info") or {}).get("props") or {}
            mclass = props.get("media.class")
            name = props.get("node.name")
            if not name:
                continue
            if mclass == "Audio/Source":
                names.append(name)
            elif mclass == "Audio/Sink":
                # a sink's monitor is exposed to Pulse/ffmpeg as "<sink>.monitor".
                names.append(f"{name}.monitor")
        return names
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return []


def record_tracks(
    mic_out: str, system_out: str, duration: float | None = None, reporter=None
) -> None:
    """Record both tracks until SIGINT (Ctrl-C), or for ``duration`` seconds if given."""
    system = platform.system()
    if system == "Darwin":
        listing = subprocess.run(
            ["ffmpeg", "-hide_banner", "-f", "avfoundation",
             "-list_devices", "true", "-i", ""],
            capture_output=True, text=True, timeout=10,
        ).stderr
        idx = match_device(parse_avfoundation_devices(listing), "meetscribe")
        cmd = build_macos_ffmpeg_cmd(idx, mic_out, system_out)
    else:
        sources = _list_linux_sources()
        monitor = find_monitor_source(sources)
        cmd = build_linux_ffmpeg_cmd("default", monitor, mic_out, system_out)

    if duration is not None:
        # Bound EACH input by placing -t before every -i (a single leading -t would only
        # limit the first input, letting the other track record until killed). Real recording
        # passes duration=None and stops both tracks cleanly via SIGINT → `q` on stdin.
        bounded: list[str] = []
        for tok in cmd:
            if tok == "-i":
                bounded += ["-t", str(duration)]
            bounded.append(tok)
        cmd = bounded

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    prev = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, lambda *_: stop_ffmpeg(proc))
    try:
        proc.wait()
    finally:
        signal.signal(signal.SIGINT, prev)


def run(out_dir: str | None = None, reporter=None) -> int:
    from datetime import datetime, timezone

    root = Path(out_dir or f"meetscribe-{datetime.now(timezone.utc):%Y-%m-%dT%H-%M-%S}")
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    mic_out = str(raw / "mic.wav")
    system_out = str(raw / "system.wav")

    print(f"Recording to {root} — press Ctrl-C to stop.")
    record_tracks(mic_out, system_out, reporter=reporter)

    for track in (mic_out, system_out):
        warning = warn_if_silent(track)
        if warning:
            print(f"⚠ {warning}")

    from . import pipeline

    return pipeline.run(audio=str(root), out_dir=str(root), reporter=reporter)
