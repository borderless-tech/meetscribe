"""meetscribe — local, offline dual-track meeting recorder + diarized transcriber.

See what-we-build.md for the architecture rationale. The package is deliberately
split so the only platform-specific module is :mod:`meetscribe.record`; everything
downstream operates on 16 kHz mono WAV files and is identical on Linux and macOS.
"""

# Kept in sync with pyproject.toml + uv.lock (enforced by tests/test_version.py);
# release process in CLAUDE.md "Versioning & releases".
__version__ = "0.2.0"
