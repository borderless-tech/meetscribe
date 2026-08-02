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


class Probe(Protocol):
    def checks(self) -> list[Check]: ...


def run(probe: Probe | None = None) -> int:
    probe = probe or RealProbe()
    checks = probe.checks()
    print(format_report(checks))
    return 0 if checks_pass(checks) else 1


def _rms(samples: np.ndarray) -> float:
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))


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
        # A PipeWire monitor source is what we record the far side from.
        has_monitor = False
        try:
            res = subprocess.run(
                ["pactl", "list", "short", "sources"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            has_monitor = ".monitor" in res.stdout
        except (FileNotFoundError, subprocess.SubprocessError):
            # pactl may be absent; fall back to checking a running PipeWire.
            has_monitor = shutil.which("pw-record") is not None
        return [
            Check(
                "PipeWire monitor source",
                has_monitor,
                None if has_monitor else "start PipeWire; a <sink>.monitor source is required",
            )
        ]

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
        """The most important check: 1 s test capture, RMS must be > 0.

        On macOS, without TCC permission ffmpeg records silence with no error at all.
        """
        try:
            rms = self._capture_rms()
        except Exception:
            rms = 0.0
        ok = rms > 0.0
        hint = None
        if not ok:
            hint = (
                "microphone captured silence — grant mic access to the terminal "
                "(macOS: System Settings → Privacy → Microphone)"
            )
        return Check("Microphone capture (RMS > 0)", ok, hint)

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
                 "-t", "1", "-ar", "16000", "-ac", "1", wav],
                capture_output=True,
                timeout=15,
            )
            if not os.path.exists(wav):
                return 0.0
            samples, _ = load_wav_f32(wav)
            return _rms(samples)
