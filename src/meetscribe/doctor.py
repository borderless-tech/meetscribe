"""Preflight audio-setup checks (see what-we-build.md §8).

Implemented in chunk C6. The check-result formatting is pure and unit-tested; the
platform probes (PipeWire monitor source, BlackHole HAL plugin, aggregate device,
1 s RMS test-capture) sit behind interfaces that are mocked in tests.
"""

from __future__ import annotations


def run() -> int:
    raise NotImplementedError("doctor is implemented in chunk C6")
