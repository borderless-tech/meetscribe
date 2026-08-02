"""meetscribe — local, offline dual-track meeting recorder + diarized transcriber.

See what-we-build.md for the architecture rationale. The package is deliberately
split so the only platform-specific module is :mod:`meetscribe.record`; everything
downstream operates on 16 kHz mono WAV files and is identical on Linux and macOS.
"""

__version__ = "0.1.0"
