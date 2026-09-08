"""Guards for the SemVer setup (CLAUDE.md "Versioning & releases").

The version is declared in three places — pyproject.toml, meetscribe.__version__, and the
project's own entry in uv.lock — and every released version needs a CHANGELOG entry. These
tests make forgetting any one of them a red suite instead of a silent drift.
"""

import re
import tomllib
from pathlib import Path

from meetscribe import __version__

ROOT = Path(__file__).resolve().parent.parent


def test_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__)


def test_version_matches_pyproject():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert pyproject["project"]["version"] == __version__


def test_version_matches_uv_lock():
    # uv.lock records the project's own version — regenerate with `uv lock` after a bump.
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    [pkg] = [p for p in lock["package"] if p["name"] == "meetscribe"]
    assert pkg["version"] == __version__


def test_changelog_has_entry_for_current_version():
    changelog = (ROOT / "CHANGELOG.md").read_text()
    assert f"## [{__version__}]" in changelog
