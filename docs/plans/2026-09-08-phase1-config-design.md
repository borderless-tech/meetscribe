# Phase 1 design: "installed" foundation — TOML config + XDG directories

Roadmap context: `docs/plans/2026-09-08-transition-roadmap.md` (P1). Goal: meetscribe feels
like an installed application — a human-editable config file in the standard place, data in
the standard place, and one precedence rule everywhere. Strictly no new dependencies
(`tomllib` is stdlib; we only ever *read* TOML — the `config init` template is a plain
string).

## Non-goals

- No `[bk]` section contents yet (Phase 2 owns it; the section name is reserved).
- No behavior change for anyone who uses flags/env today — new layers slot *underneath*.
- No Windows; Linux + macOS only, both via XDG conventions (precedent: the glossary already
  lives under `$XDG_CONFIG_HOME`/`~/.config` on both platforms — `glossary.py:15`).

## Files & locations

| What | Where | Fallback |
|---|---|---|
| Config | `$XDG_CONFIG_HOME/meetscribe/config.toml` | `~/.config/meetscribe/config.toml` |
| Recordings (new default) | `$XDG_DATA_HOME/meetscribe/meetings/` | `~/.local/share/meetscribe/meetings/` |
| Glossary (unchanged location) | `$XDG_CONFIG_HOME/meetscribe/glossary.txt` | `~/.config/meetscribe/glossary.txt` |

`MEETSCRIBE_CONFIG=<path>` overrides the config file path (tests, CI, power users).

## Config schema v1 (all keys optional)

```toml
# meetscribe config — flags > environment > this file > built-in defaults

[stt]
backend = "local"        # "local" | "deepgram"      (flag --backend, env STT_BACKEND)
language = "de"          # remote-STT language        (flag --language, env STT_LANGUAGE)

[deepgram]
api_key = ""             # env DEEPGRAM_API_KEY wins; keep this file 0600 when set
api_key_cmd = ""         # shell command printing the key; mutually exclusive with api_key,
                         # executed lazily only when the key is actually needed

[storage]
meetings_dir = ""        # where recordings land      (flag -o wins per run)
                         # default: $XDG_DATA_HOME/meetscribe/meetings

[record]
system_source = ""       # fixed system-audio source  (flag --system-source wins)

[output]
bundle = true            # flag --bundle/--no-bundle wins
cleanup = true           # flag --no-cleanup wins

# [bk]                   # reserved for Phase 2 (base_url, token, upload policy)
```

## Precedence & origins

One rule everywhere: **flag > env > config > default.** New module `config.py` owns it:

- `config_home()` / `data_home()` — XDG helpers (hoist the existing one from `glossary.py`).
- `load(path=None) -> dict` — parsed TOML; missing file → `{}`.
- `Resolved = (value, origin)` where origin ∈ `flag|env|config|default` — every resolver
  returns it, so `meetscribe config` can show *why* each value is what it is. Call sites
  that don't care use `.value`.
- Resolvers: `backend`, `language`, `api_key`, `meetings_dir`, `system_source`, `bundle`,
  `cleanup`. The existing `pipeline.resolve_backend`/`resolve_language` delegate here
  (their flag>env>default semantics gain the config layer in between).

Consequence for the CLI: `--bundle`/`--no-bundle` and `--no-cleanup` parser defaults become
`None` (like `--backend`/`--language` already are) so an unset flag is distinguishable from
an explicit one; resolution moves into `pipeline.run`/`record.run`. The bare-invocation
getattr fallbacks in `cli.main` become `None` accordingly (the guard tests move down a
layer: they now assert the *resolved* default, not the parser default).

## Error handling

- Missing config file: silently fine (all defaults).
- **Malformed TOML: exit 2 with file+line** — a typo'd config that silently degrades to
  defaults is the worst failure mode.
- Unknown sections/keys: `reporter.warn` (typo detection), then continue.
- `[deepgram].api_key` set while the file is group/world-readable: warn (once, at load).

## `meetscribe config` subcommand

- `meetscribe config` — effective values as a table: key, value (api_key masked), origin,
  plus the config path in use (and whether it exists). Read-only, prints to stdout (this IS
  the command's data output; the stdout-clean rule applies to record/process).
- `meetscribe config init` — writes the commented template above to the config path
  (refuses to overwrite; creates parent dir; `chmod 0600`).
- `meetscribe config path` — prints just the path (scripting).

## Behavior changes (the only two)

1. `meetscribe record` without `-o` writes to `<meetings_dir>/meetscribe-<timestamp>/`
   instead of `./meetscribe-<timestamp>/` (`record.py:462`). `-o` behaves exactly as
   before. This is the "stops littering the CWD" maturity win and warrants the minor bump.
2. `doctor` gains a config check: path, parse status, unknown-key warnings, api_key file
   permissions.

Everything else (bundle/cleanup/backend/language/system_source via config) only *adds* a
layer beneath existing flags/env — same defaults as today when no config exists.

## Testing rules

Tests never touch the real home: monkeypatch `XDG_CONFIG_HOME`, `XDG_DATA_HOME`,
`MEETSCRIBE_CONFIG` to tmp_path. Every resolver test asserts value AND origin. Suite stays
offline and sub-second.
