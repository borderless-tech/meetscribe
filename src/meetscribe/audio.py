"""Audio helpers: ffmpeg resampling to 16 kHz mono, and loading WAV as float32.

The pipeline resamples the native 48 kHz recordings to 16 kHz mono (the rate every model
expects) before any ML stage runs.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def build_resample_cmd(src: str, dst: str, rate: int = 16000) -> list[str]:
    """ffmpeg command to resample ``src`` to mono ``rate`` Hz at ``dst`` (overwrites)."""
    return ["ffmpeg", "-y", "-i", src, "-ar", str(rate), "-ac", "1", dst]


def load_wav_f32(path: str | Path) -> tuple[np.ndarray, int]:
    """Load a 16-bit PCM mono WAV as a 1-D float32 array in [-1, 1) plus its sample rate."""
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        n_channels = w.getnchannels()
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if n_channels > 1:  # downmix defensively; the recorder writes mono
        samples = samples.reshape(-1, n_channels).mean(axis=1)
    return samples, rate
