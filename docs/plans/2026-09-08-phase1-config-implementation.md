# Phase 1 implementation plan: config + XDG

Prereq reading: `2026-09-08-phase1-config-design.md` (the spec — schema, precedence,
error handling), `CLAUDE.md`. Branch: `feat/config-xdg`. Tests via `nix develop -c pytest
-q`; baseline **308 passed, 3 skipped** — green after every task. TDD throughout: failing
test first. No new dependencies (`tomllib` is stdlib, read-only; the init template is a
plain string constant). Tests must monkeypatch `XDG_CONFIG_HOME`/`XDG_DATA_HOME`/
`MEETSCRIBE_CONFIG` — never read the real home.

## Task 1 — `config.py` foundation (new files only)

Files: `src/meetscribe/config.py`, `tests/test_config.py`.

- `config_home()` / `data_home()`: XDG env with `~/.config` / `~/.local/share` fallbacks.
- `config_path()`: `MEETSCRIBE_CONFIG` override, else `config_home()/meetscribe/config.toml`.
- `load(path=None) -> dict`: missing → `{}`; malformed TOML → `ConfigError` carrying
  file+line (callers turn it into exit 2); returns parsed dict as-is.
- `unknown_keys(cfg) -> list[str]`: dotted paths not in the v1 schema (for warn + doctor).
- `Resolved` NamedTuple `(value, origin)`, origin ∈ `"flag"|"env"|"config"|"default"`.
- Resolvers (each `(flag_value, env: Mapping, cfg: dict) -> Resolved`): `backend`,
  `language`, `api_key`, `meetings_dir` (expanduser; default `data_home()/meetscribe/
  meetings`), `system_source`, `bundle`, `cleanup`. Booleans: config value must be a real
  TOML bool — a string is a `ConfigError`, not truthiness.
- `TEMPLATE` string constant = the commented example from the design doc, and
  `insecure_api_key_perms(path) -> bool` (api_key set + file mode has group/other bits).

TDD order: XDG helpers → path override → load/missing/malformed → each resolver
(value AND origin asserted, including flag-beats-env-beats-config-beats-default chains) →
unknown_keys → perms check.

## Task 2 — wire resolution into pipeline/record/glossary

Files: `src/meetscribe/pipeline.py`, `src/meetscribe/record.py`,
`src/meetscribe/glossary.py`, `tests/test_pipeline.py`, `tests/test_record.py`,
`tests/test_glossary.py` (whichever exists for glossary).

- `pipeline.resolve_backend`/`resolve_language`: gain the config layer (delegate to
  `config.py` resolvers; keep the public names — `check_backend`, doctor, and tests call
  them). `DEEPGRAM_API_KEY` resolution honors `[deepgram].api_key` beneath the env var —
  the fail-fast message mentions both ("export DEEPGRAM_API_KEY or set [deepgram].api_key
  in <config path>").
- `pipeline.run`/`record.run`: `bundle`/`cleanup` params accept `None` = "resolve it"
  (config → default true); explicit `True`/`False` (flag) wins. Malformed config →
  clean exit 2, message includes path+line.
- `record.run` default out dir: `<meetings_dir resolved>/meetscribe-<timestamp>/`
  (`record.py:462`); `-o` untouched; dir created with parents. `system_source=None` →
  resolve from config.
- `glossary.py`: hoist its XDG helper to use `config.config_home()` (behavior identical).

## Task 3 — CLI: flag defaults to None + `config` subcommand

Files: `src/meetscribe/cli.py`, `tests/test_cli.py`.

- `--bundle`/`--no-bundle` (record+process) and `--no-cleanup`: parser default `None`
  (BooleanOptionalAction already supports it; `--no-cleanup` becomes
  `--cleanup/--no-cleanup` BooleanOptionalAction with default None — keep `--no-cleanup`
  spelling working). `cli.main` getattr fallbacks change to `None`; the existing
  bundle-default guard tests move down a layer (assert resolution via monkeypatched
  `record.run`/`pipeline.run` receiving `None`, plus resolver tests in test_config.py
  already covering the default).
- New `config` subcommand: bare = table of effective values (key, value with api_key
  masked to `dg_…****`, origin) + config path + exists/valid status; `init` = write
  TEMPLATE (refuse overwrite, mkdir parents, chmod 0600, print path); `path` = print path
  only. Stdout is the data output here. No rich dependency in this path — plain print
  formatting is fine and keeps it `--quiet`-proof.

## Task 4 — doctor + docs

Files: `src/meetscribe/doctor.py`, `tests/test_doctor.py`, `CHANGELOG.md`, `CLAUDE.md`,
`README.md`.

- Doctor "config" check (matching existing check style): path in use + exists, parse
  ok/failed (failed = red with line), unknown keys (warn), api_key perms (warn 0600 advice).
  Runs before the audio checks; a broken config must not abort the other checks.
- CHANGELOG `[Unreleased]`: Added (config file, `meetscribe config` subcommand, XDG data
  dir) + **Changed** callout for the new default recordings location (the one behavior
  change). CLAUDE.md: new "Configuration" section (schema, precedence, file paths, the
  None-default flag pattern). README: short config section + `config init` quickstart.

## Sequencing

Task 1 alone first (foundation; new files only). Tasks 2/3/4 then run in parallel —
disjoint files (T2: pipeline/record/glossary, T3: cli, T4: doctor/docs). One integration
concern: T2 and T3 must agree that `None` means "resolve from config" for
bundle/cleanup — that contract is fixed HERE, both sides implement against it. Reviews
after: correctness-vs-design, invariants (no new deps, stdout clean, no real-home reads in
tests, offline suite), test quality; findings adversarially verified before fixing.

## Acceptance (whole phase)

- Fresh machine, no config: identical behavior to v0.2.1 except recordings land under
  `$XDG_DATA_HOME/meetscribe/meetings/`.
- `meetscribe config init` → edit backend to deepgram → `meetscribe config` shows
  `backend deepgram (config)`; `STT_BACKEND=local` flips it to `(env)`; `--backend
  deepgram` flips it to `(flag)`.
- Malformed config: every subcommand exits 2 with path+line — except `config path`
  (must keep printing the path so the user can locate the broken file) and `doctor`
  (renders it as a red check and exits 1, so the remaining preflight checks still run).
- Full suite green, offline, sub-second; `nix flake check` passes.
