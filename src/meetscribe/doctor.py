"""Preflight audio-setup checks (what-we-build.md §8).

The report formatting and pass/fail logic are pure and unit-tested. The platform probes — which
Nix cannot substitute for (BlackHole HAL plugin, the aggregate device, and crucially the
microphone RMS test-capture that catches macOS TCC silently recording silence) — live behind the
:class:`RealProbe` interface and are exercised by ``nix run .#doctor``.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass
class Check:
    name: str
    ok: bool
    hint: str | None = None


def format_report(checks: list[Check]) -> str:
    lines: list[str] = []
    for c in checks:
        if c.ok:
            lines.append(f"✓ {c.name}")
        else:
            lines.append(f"✗ {c.name}")
            if c.hint:
                lines.append(f"    → {c.hint}")
    return "\n".join(lines)


def checks_pass(checks: list[Check]) -> bool:
    return all(c.ok for c in checks)


def linux_monitor_check(sources: list[str], default_sink: str | None) -> Check:
    """Report the exact monitor source ``record.py`` would capture system audio from.

    Showing the *chosen* source (not merely "some monitor exists") makes a mis-selection —
    e.g. tapping a silent HDMI monitor while audio plays to a Bluetooth headset — obvious in
    the preflight, which is precisely the failure this surfaces.
    """
    from .record import find_monitor_source

    try:
        monitor = find_monitor_source(sources, default_sink)
    except ValueError:
        return Check(
            "System-audio source",
            False,
            "start PipeWire; a <sink>.monitor source is required",
        )
    return Check(f"System-audio source → {monitor}", True)


class Probe(Protocol):
    def checks(self) -> list[Check]: ...


def run(probe: Probe | None = None, reporter=None) -> int:
    from .progress import NullReporter

    reporter = reporter or NullReporter()
    probe = probe or RealProbe()
    # A spinner during the checks — the 1 s mic RMS test-capture otherwise looks frozen.
    with reporter.stage("running preflight checks"):
        checks = probe.checks()
    print(format_report(checks))
    return 0 if checks_pass(checks) else 1


def _rms(samples: np.ndarray) -> float:
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))


def rms_after_warmup(samples: np.ndarray, sample_rate: int, warmup_s: float = 1.5) -> float:
    """RMS of the capture *after* an initial warm-up window.

    A Bluetooth headset emits digital silence for up to ~1s while WirePlumber switches it from
    A2DP to HFP/HSP (the profile that has a microphone) the moment a capture stream opens —
    exactly what happens when doctor or record opens the mic. Averaging RMS over the whole clip
    would read that transition silence as a dead mic. Skip the warm-up and measure the tail; if
    the clip is shorter than the warm-up, measure all of it rather than discard everything.
    """
    start = int(warmup_s * sample_rate)
    tail = samples[start:] if start < len(samples) else samples
    return _rms(tail)


class RealProbe:
    """Real platform probes. Not unit-tested; run for real via ``nix run .#doctor``."""

    def checks(self) -> list[Check]:
        out = [self._ffmpeg(), self._models()]
        if platform.system() == "Darwin":
            out += self._macos_audio()
        else:
            out += self._linux_audio()
        out.append(self._mic_rms())
        return out

    def _ffmpeg(self) -> Check:
        path = shutil.which("ffmpeg")
        return Check(
            "ffmpeg", path is not None, None if path else "ffmpeg not found on PATH"
        )

    def _models(self) -> Check:
        root = os.environ.get("MEETSCRIBE_MODELS")
        ok = bool(root) and all(
            (Path(root) / sub).is_dir() for sub in ("asr", "seg", "spk", "vad")
        )
        return Check(
            "models (parakeet-tdt-0.6b-v3, cam++, seg-3.0, silero)",
            ok,
            None if ok else "MEETSCRIBE_MODELS is unset or incomplete",
        )

    def _linux_audio(self) -> list[Check]:
        # Report the exact source record.py would capture from — the default sink's monitor —
        # so a mis-selection is visible here rather than discovered as a silent system track.
        from .record import _default_sink, _list_linux_sources

        return [linux_monitor_check(_list_linux_sources(), _default_sink())]

    def _macos_audio(self) -> list[Check]:
        blackhole = Path(
            "/Library/Audio/Plug-Ins/HAL/BlackHole2ch.driver"
        ).exists()
        aggregate = self._macos_has_aggregate("meetscribe")
        return [
            Check(
                "BlackHole 2ch",
                blackhole,
                None if blackhole else "brew install blackhole-2ch",
            ),
            Check(
                'Aggregate device "meetscribe"',
                aggregate,
                None
                if aggregate
                else "Audio MIDI Setup → combine BlackHole 2ch + Mic, name it meetscribe",
            ),
        ]

    def _macos_has_aggregate(self, name: str) -> bool:
        try:
            res = subprocess.run(
                ["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return name in (res.stderr or "")
        except (FileNotFoundError, subprocess.SubprocessError):
            return False

    def _mic_rms(self) -> Check:
        """The most important check: a test capture whose RMS must be > 0.

        On macOS, without TCC permission ffmpeg records silence with no error at all. On Linux a
        Bluetooth headset spends the first ~1s switching A2DP→HFP (silence), so the capture runs
        long and the warm-up is skipped (see :func:`rms_after_warmup`).
        """
        try:
            rms = self._capture_rms()
        except Exception:
            rms = 0.0
        ok = rms > 0.0
        hint = None
        if not ok:
            if platform.system() == "Darwin":
                hint = (
                    "microphone captured silence — grant mic access to the terminal "
                    "(System Settings → Privacy → Microphone)"
                )
            else:
                hint = (
                    "microphone captured silence — check the default input isn't muted and "
                    "points at a working mic (wpctl / pavucontrol); a Bluetooth headset needs "
                    "bluetooth.autoswitch-to-headset-profile enabled to expose its mic"
                )
        return Check("Microphone capture (RMS > 0)", ok, hint)

    # Capture longer than the Bluetooth A2DP→HFP switch takes, so the warm-up window we discard
    # still leaves real signal to measure.
    _CAPTURE_S = 3
    _WARMUP_S = 1.5

    def _capture_rms(self) -> float:
        from .audio import load_wav_f32

        if platform.system() == "Darwin":
            fmt, src = "avfoundation", ":default"
        else:
            fmt, src = "pulse", "default"
        with tempfile.TemporaryDirectory() as td:
            wav = os.path.join(td, "probe.wav")
            subprocess.run(
                ["ffmpeg", "-y", "-hide_banner", "-f", fmt, "-i", src,
                 "-t", str(self._CAPTURE_S), "-ar", "16000", "-ac", "1", wav],
                capture_output=True,
                timeout=15,
            )
            if not os.path.exists(wav):
                return 0.0
            samples, sr = load_wav_f32(wav)
            return rms_after_warmup(samples, sr, self._WARMUP_S)
