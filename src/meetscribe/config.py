"""TOML config + XDG paths + the single precedence rule: flag > env > config > default.

This module owns configuration for the whole CLI (see
``docs/plans/2026-09-08-phase1-config-design.md``). The file lives at
``$XDG_CONFIG_HOME/meetscribe/config.toml`` (``MEETSCRIBE_CONFIG`` overrides the path);
recordings default under ``$XDG_DATA_HOME/meetscribe/meetings``. Reading is stdlib
``tomllib`` only — we never *write* TOML, the ``config init`` template is the plain
string :data:`TEMPLATE`.

Every resolver returns a :class:`Resolved` ``(value, origin)`` so ``meetscribe config``
can show *why* each effective value is what it is; call sites that don't care use
``.value``. Empty strings in the config (the template ships ``""`` placeholders) count
as *unset* and fall through to the next layer. Booleans must be real TOML booleans —
a string like ``"false"`` raises :class:`ConfigError` instead of being truthy.

A malformed file raises :class:`ConfigError` carrying ``path`` and ``line``; callers
turn that into exit 2 (a typo'd config silently degrading to defaults is the worst
failure mode).
"""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path
from typing import Mapping, NamedTuple

TEMPLATE = """\
# meetscribe config — flags > environment > this file > built-in defaults.
# Everything ships commented out: absent keys use the built-in default
# (`meetscribe config` shows every effective value and where it came from).
# If you set [deepgram].api_key this file holds a secret — keep it 0600 and
# out of synced/public dotfile repos.

#[stt]
#backend = "local"        # "local" | "deepgram"      (flag --backend, env STT_BACKEND)
#language = "de"          # remote-STT language        (flag --language, env STT_LANGUAGE)

#[deepgram]
#api_key = ""             # env DEEPGRAM_API_KEY wins; keep this file 0600 when set
#api_key_cmd = ""         # shell command that prints the key (e.g. "pass show deepgram");
#                         # mutually exclusive with api_key, run only when the key is needed

#[storage]
#meetings_dir = ""        # where recordings land      (flag -o wins per run)
#                         # default: $XDG_DATA_HOME/meetscribe/meetings

#[record]
#system_source = ""       # fixed system-audio source  (flag --system-source wins)

#[output]
#bundle = true            # flag --bundle/--no-bundle wins
#cleanup = true           # flag --no-cleanup wins

#[bk]                     # reserved for Phase 2 (base_url, token, upload policy)
"""

#: Config schema v1 — section -> allowed keys. ``bk`` is reserved for Phase 2 and
#: deliberately absent here (its keys are never reported as unknown).
_SCHEMA: dict[str, frozenset[str]] = {
    "stt": frozenset({"backend", "language"}),
    "deepgram": frozenset({"api_key", "api_key_cmd"}),
    "storage": frozenset({"meetings_dir"}),
    "record": frozenset({"system_source"}),
    "output": frozenset({"bundle", "cleanup"}),
}

_RESERVED_SECTIONS = frozenset({"bk"})


class ConfigError(Exception):
    """A config problem the user must fix (exit-2 material): malformed TOML or a
    wrongly-typed value. Carries ``path`` and ``line`` when known so the message
    can point at the exact spot."""

    def __init__(
        self, message: str, *, path: str | os.PathLike | None = None,
        line: int | None = None,
    ) -> None:
        super().__init__(message)
        self.path = str(path) if path is not None else None
        self.line = line


class Resolved(NamedTuple):
    """A resolved setting plus where it came from (``flag``/``env``/``config``/
    ``default``) — the origin is what ``meetscribe config`` displays."""

    value: object
    origin: str


class ApiKeyCmd(NamedTuple):
    """Marker for a *lazily fetched* API key: ``[deepgram].api_key_cmd`` resolved
    but deliberately not executed. Only :func:`fetch_api_key` runs the command —
    at the moment the key material is actually needed, never during ``config``
    display, :func:`validate`, or doctor."""

    cmd: str


# --------------------------------------------------------------------- paths


def config_home() -> Path:
    """``$XDG_CONFIG_HOME``, else ``~/.config`` (empty env counts as unset)."""
    base = os.environ.get("XDG_CONFIG_HOME")
    return Path(base) if base else Path.home() / ".config"


def data_home() -> Path:
    """``$XDG_DATA_HOME``, else ``~/.local/share`` (empty env counts as unset)."""
    base = os.environ.get("XDG_DATA_HOME")
    return Path(base) if base else Path.home() / ".local" / "share"


def config_path() -> Path:
    """The config file path: ``MEETSCRIBE_CONFIG`` override, else
    ``config_home()/meetscribe/config.toml``."""
    override = os.environ.get("MEETSCRIBE_CONFIG")
    if override:
        return Path(override)
    return config_home() / "meetscribe" / "config.toml"


# -------------------------------------------------------------------- loading


def load(path: str | Path | None = None) -> dict:
    """Parse the config file (default: :func:`config_path`). Missing file → ``{}``
    (all defaults, silently fine). Malformed TOML → :class:`ConfigError` with
    file + line."""
    p = Path(path) if path is not None else config_path()
    try:
        text = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as e:
        # An unreadable file (permissions, a directory, …) is exit-2 material like
        # malformed TOML — never an uncaught traceback.
        raise ConfigError(f"cannot read config {p}: {e}", path=p) from e
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        m = re.search(r"at line (\d+)", str(e))
        line = int(m.group(1)) if m else None
        raise ConfigError(f"malformed config {p}: {e}", path=p, line=line) from e


def unknown_keys(cfg: dict) -> list[str]:
    """Dotted paths in ``cfg`` that are not in the v1 schema (typo detection for
    the load-time warning and ``doctor``). The reserved ``[bk]`` section is never
    reported. Order follows appearance in the file."""
    out: list[str] = []
    for key, val in cfg.items():
        if key in _RESERVED_SECTIONS:
            continue
        if key not in _SCHEMA or not isinstance(val, dict):
            out.append(key)
            continue
        for sub in val:
            if sub not in _SCHEMA[key]:
                out.append(f"{key}.{sub}")
    return out


def load_warnings(cfg: dict, path: str | Path | None = None) -> list[str]:
    """The design's load-time advisories, as printable messages (callers hand them
    to ``reporter.warn``): unknown sections/keys (typo detection) and a non-empty
    ``[deepgram].api_key`` in a group/world-readable file. Never raises, never
    echoes the key itself."""
    p = Path(path) if path is not None else config_path()
    msgs: list[str] = []
    unknown = unknown_keys(cfg)
    if unknown:
        msgs.append(
            f"config {p}: unknown key{'s' if len(unknown) > 1 else ''} "
            f"{', '.join(unknown)} — typo? (ignored)"
        )
    if insecure_api_key_perms(p):
        msgs.append(
            f"config {p}: [deepgram].api_key is set but the file is "
            f"group/world-readable — chmod 0600 {p}"
        )
    return msgs


def insecure_api_key_perms(path: str | Path) -> bool:
    """True iff the config at ``path`` sets ``[deepgram].api_key`` (non-empty) while
    the file is group- or world-accessible (mode has any of the 0o077 bits) — the
    warn-once secret-hygiene check."""
    p = Path(path)
    try:
        mode = p.stat().st_mode
    except OSError:
        return False
    try:
        key = _cfg_str(load(p), "deepgram", "api_key")
    except ConfigError:
        return False
    if not key:
        return False
    return bool(mode & 0o077)


# ------------------------------------------------------------------ resolvers
#
# Each resolver is (flag_value, env, cfg) -> Resolved and applies the one rule:
# flag > env > config > default. ``env`` is a plain Mapping so tests pass dicts.


def _cfg_str(cfg: dict, section: str, key: str) -> str | None:
    """``cfg[section][key]`` as a stripped string, or ``None`` when absent/empty.
    A non-string value is a :class:`ConfigError` (never silently coerced)."""
    sec = cfg.get(section)
    if not isinstance(sec, dict) or key not in sec:
        return None
    v = sec[key]
    if not isinstance(v, str):
        raise ConfigError(
            f"[{section}].{key} must be a string, got {type(v).__name__} ({v!r})"
        )
    return v.strip() or None


def _cfg_bool(cfg: dict, section: str, key: str) -> bool | None:
    """``cfg[section][key]`` as a real TOML boolean, or ``None`` when absent.
    Strings/ints raise :class:`ConfigError` — ``"false"`` must never be truthy."""
    sec = cfg.get(section)
    if not isinstance(sec, dict) or key not in sec:
        return None
    v = sec[key]
    if not isinstance(v, bool):
        raise ConfigError(
            f"[{section}].{key} must be a TOML boolean (true/false), "
            f"got {type(v).__name__} ({v!r})"
        )
    return v


def backend(flag_value: str | None, env: Mapping, cfg: dict) -> Resolved:
    """STT backend name, normalized (strip + lower). Default ``local``. Not
    validated here — ``pipeline.check_backend`` rejects unknown names."""
    if flag_value and flag_value.strip():
        return Resolved(flag_value.strip().lower(), "flag")
    e = (env.get("STT_BACKEND") or "").strip()
    if e:
        return Resolved(e.lower(), "env")
    c = _cfg_str(cfg, "stt", "backend")
    if c:
        return Resolved(c.lower(), "config")
    return Resolved("local", "default")


def language(flag_value: str | None, env: Mapping, cfg: dict) -> Resolved:
    """Remote-STT language (the local backend ignores it). Default ``de``."""
    if flag_value and flag_value.strip():
        return Resolved(flag_value.strip(), "flag")
    e = (env.get("STT_LANGUAGE") or "").strip()
    if e:
        return Resolved(e, "env")
    c = _cfg_str(cfg, "stt", "language")
    if c:
        return Resolved(c, "config")
    return Resolved("de", "default")


def api_key(flag_value: str | None, env: Mapping, cfg: dict) -> Resolved:
    """Deepgram API key: ``DEEPGRAM_API_KEY`` env > ``[deepgram].api_key`` >
    ``[deepgram].api_key_cmd`` (as an unexecuted :class:`ApiKeyCmd` marker) >
    ``None`` (there is no flag today; the parameter keeps the uniform shape).

    ``api_key`` and ``api_key_cmd`` both set is a :class:`ConfigError`: a
    leftover static key would silently shadow the keyring command — likely with
    a stale secret — so the ambiguity fails loudly."""
    if flag_value and flag_value.strip():
        return Resolved(flag_value.strip(), "flag")
    e = (env.get("DEEPGRAM_API_KEY") or "").strip()
    if e:
        return Resolved(e, "env")
    c = _cfg_str(cfg, "deepgram", "api_key")
    cmd = _cfg_str(cfg, "deepgram", "api_key_cmd")
    if c and cmd:
        raise ConfigError(
            "[deepgram].api_key and [deepgram].api_key_cmd are both set — "
            "they are mutually exclusive, remove one"
        )
    if c:
        return Resolved(c, "config")
    if cmd:
        return Resolved(ApiKeyCmd(cmd), "config")
    return Resolved(None, "default")


def fetch_api_key(value: object, timeout_s: float = 30.0) -> str | None:
    """Turn a resolved api-key value into key material. Plain strings and ``None``
    pass through; an :class:`ApiKeyCmd` is executed HERE and only here — via the
    shell (the command comes from the user's own 0600 config, same trust model as
    git config commands), stdout stripped. Failure, timeout, or empty output is a
    :class:`ConfigError` (exit-2 material with the command's stderr included)."""
    if not isinstance(value, ApiKeyCmd):
        return value  # type: ignore[return-value]
    import subprocess

    try:
        proc = subprocess.run(
            value.cmd, shell=True, capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        raise ConfigError(
            f"[deepgram].api_key_cmd timed out after {timeout_s:.0f}s: {value.cmd}"
        ) from None
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        raise ConfigError(
            f"[deepgram].api_key_cmd failed (exit {proc.returncode})"
            + (f": {stderr}" if stderr else "")
        )
    key = proc.stdout.strip()
    if not key:
        raise ConfigError(
            "[deepgram].api_key_cmd produced empty output — expected the key on stdout"
        )
    return key


def meetings_dir(
    flag_value: str | Path | None, env: Mapping, cfg: dict
) -> Resolved:
    """Where recordings land, as an expanduser'd :class:`Path`. Default
    ``data_home()/meetscribe/meetings``. No env layer is defined for this key."""
    if flag_value:
        return Resolved(Path(flag_value).expanduser(), "flag")
    c = _cfg_str(cfg, "storage", "meetings_dir")
    if c:
        return Resolved(Path(c).expanduser(), "config")
    return Resolved(data_home() / "meetscribe" / "meetings", "default")


def system_source(flag_value: str | None, env: Mapping, cfg: dict) -> Resolved:
    """Fixed system-audio source name; ``None`` means auto-detect. No env layer
    is defined for this key."""
    if flag_value and flag_value.strip():
        return Resolved(flag_value.strip(), "flag")
    c = _cfg_str(cfg, "record", "system_source")
    if c:
        return Resolved(c, "config")
    return Resolved(None, "default")


def bundle(flag_value: bool | None, env: Mapping, cfg: dict) -> Resolved:
    """Whether to write the ``.mscribe`` bundle. ``flag_value=None`` means the
    flag was not given. Default ``True``."""
    if flag_value is not None:
        return Resolved(bool(flag_value), "flag")
    c = _cfg_bool(cfg, "output", "bundle")
    if c is not None:
        return Resolved(c, "config")
    return Resolved(True, "default")


def cleanup(flag_value: bool | None, env: Mapping, cfg: dict) -> Resolved:
    """Whether to run the LLM cleanup pass. ``flag_value=None`` means the flag
    was not given. Default ``True``."""
    if flag_value is not None:
        return Resolved(bool(flag_value), "flag")
    c = _cfg_bool(cfg, "output", "cleanup")
    if c is not None:
        return Resolved(c, "config")
    return Resolved(True, "default")


def validate(cfg: dict) -> None:
    """Exercise every resolver against ``cfg`` (flag ``None``, empty env, so the
    config layer is actually read) and raise :class:`ConfigError` on the first
    wrongly-typed value. Valid TOML the resolvers reject is exit-2 material at
    run time; ``doctor`` calls this so such a config shows red in the preflight
    instead of a green check on a file that aborts every record/process run."""
    for resolver in (
        backend, language, api_key, meetings_dir, system_source, bundle, cleanup,
    ):
        resolver(None, {}, cfg)
