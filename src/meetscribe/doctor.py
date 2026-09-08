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
import socket
import ssl
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol

import numpy as np


@dataclass
class Check:
    name: str
    ok: bool
    hint: str | None = None
    #: Advisory: rendered as ``!`` with its hint, but never fails the run.
    warn: bool = False


def format_report(checks: list[Check]) -> str:
    lines: list[str] = []
    for c in checks:
        if c.ok and not c.warn:
            lines.append(f"✓ {c.name}")
        else:
            lines.append(f"{'!' if c.warn else '✗'} {c.name}")
            if c.hint:
                lines.append(f"    → {c.hint}")
    return "\n".join(lines)


def checks_pass(checks: list[Check]) -> bool:
    return all(c.ok for c in checks)


def config_checks(path: str | os.PathLike | None = None) -> list[Check]:
    """Config-file health: path in use + existence, parse status, unknown-key and
    api_key-permission warnings.

    Returns Checks and never raises — a broken config is a red check, not an abort,
    so the audio checks that follow always run. A missing file is fine (all defaults).
    Malformed TOML (or an unreadable file) is red with the detail; wrongly-typed
    values — valid TOML that every record/process run would reject with exit 2 —
    are red too (the resolvers run here); unknown keys (typo detection) and a
    group/world-readable file with ``[deepgram].api_key`` set are warnings.
    """
    from . import config as config_mod

    p = Path(path) if path is not None else config_mod.config_path()
    if not p.exists():
        return [Check(f"Config {p} (not found — defaults apply)", True)]
    try:
        cfg = config_mod.load(p)
    except config_mod.ConfigError as e:
        detail = f"malformed TOML at line {e.line}" if e.line is not None else str(e)
        return [
            Check(
                f"Config {p}",
                False,
                f"{detail} — fix it or remove the file "
                "(meetscribe refuses to silently fall back to defaults)",
            )
        ]
    out = [Check(f"Config {p}", True)]
    try:
        config_mod.validate(cfg)
    except config_mod.ConfigError as e:
        out.append(
            Check(
                "Config values",
                False,
                f"{e} — every record/process run would abort on this (exit 2)",
            )
        )
    unknown = config_mod.unknown_keys(cfg)
    if unknown:
        out.append(
            Check(
                "Config keys",
                True,
                f"unknown key{'s' if len(unknown) > 1 else ''}: "
                f"{', '.join(unknown)} — typo? (ignored)",
                warn=True,
            )
        )
    if config_mod.insecure_api_key_perms(p):
        out.append(
            Check(
                "Config permissions",
                True,
                f"[deepgram].api_key is set but {p} is group/world-readable — "
                f"chmod 0600 {p}",
                warn=True,
            )
        )
    return out


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


# --- Remote-backend checks (STT_BACKEND=deepgram) --------------------------------------------
# The remote backend has no silent fallback to local, so a missing key or an unreachable API
# must surface in the preflight, not mid-recording. The reachability probe is a bare DNS+TCP+TLS
# handshake against the API endpoint — deliberately NOT an HTTP request, so it can never bill.

_DEEPGRAM_HOST = "api.deepgram.com"
_DEEPGRAM_PORT = 443

# Injected transport seam: callable(host, port) that raises on any DNS/TCP/TLS failure.
Connect = Callable[[str, int], None]


def _tls_connect(host: str, port: int, timeout_s: float = 5.0) -> None:
    ctx = ssl.create_default_context()
    with socket.create_connection((host, port), timeout=timeout_s) as raw:
        with ctx.wrap_socket(raw, server_hostname=host):
            pass


def deepgram_key_check(api_key: str | None) -> Check:
    from . import config as config_mod

    ok = bool(api_key and api_key.strip())
    return Check(
        "Deepgram API key set",
        ok,
        None
        if ok
        else (
            "export DEEPGRAM_API_KEY=<key> or set [deepgram].api_key in "
            f"{config_mod.config_path()} — the deepgram backend has no local fallback"
        ),
    )


def deepgram_reachability_check(connect: Connect | None = None) -> Check:
    connect = connect or _tls_connect
    name = f"Deepgram API reachable ({_DEEPGRAM_HOST}:{_DEEPGRAM_PORT})"
    try:
        connect(_DEEPGRAM_HOST, _DEEPGRAM_PORT)
    except Exception:
        return Check(
            name,
            False,
            f"cannot reach {_DEEPGRAM_HOST}:{_DEEPGRAM_PORT} — offline or DNS/proxy problem? "
            "the remote backend needs network; use STT_BACKEND=local to work offline",
        )
    return Check(name, True)


def remote_checks(
    env: Mapping[str, str] | None = None,
    connect: Connect | None = None,
    cfg: dict | None = None,
) -> list[Check]:
    """Extra checks when the STT backend resolves to ``deepgram``; empty for local.

    Doctor has no ``--backend`` flag, so resolution here is env > config > default —
    the pipeline's precedence chain minus the flag (``config.backend`` /
    ``config.api_key``), so a config-driven deepgram setup gets the same verdict
    doctor gives an env-driven one. A config the loader/resolvers reject falls back
    to env-only resolution here; ``config_checks`` already reports the file red.
    """
    from . import config as config_mod

    env = os.environ if env is None else env
    if cfg is None:
        try:
            cfg = config_mod.load()
        except config_mod.ConfigError:
            cfg = {}
    try:
        backend = config_mod.backend(None, env, cfg).value
        key = config_mod.api_key(None, env, cfg).value
    except config_mod.ConfigError:
        backend = config_mod.backend(None, env, {}).value
        key = config_mod.api_key(None, env, {}).value
    if backend != "deepgram":
        return []
    return [
        deepgram_key_check(key),
        deepgram_reachability_check(connect),
    ]


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
        # Config first — and config_checks never raises, so a broken config file
        # cannot abort the audio checks below.
        out = config_checks()
        out += [self._ffmpeg(), self._models()]
        if platform.system() == "Darwin":
            out += self._macos_audio()
        else:
            out += self._linux_audio()
        out.append(self._mic_rms())
        out += remote_checks()  # no-op unless the backend resolves to deepgram (env/config)
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
