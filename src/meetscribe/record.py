"""Audio recording — the ONLY platform-specific module (see what-we-build.md §5).

Implemented in chunk C2. The pure command-building / device-matching helpers are
unit-tested for both macOS (avfoundation aggregate device + pan split) and Linux
(two PulseAudio inputs); the Linux path is additionally exercised for real.
"""

from __future__ import annotations


def run(out_dir: str | None = None) -> int:
    raise NotImplementedError("record is implemented in chunk C2")
