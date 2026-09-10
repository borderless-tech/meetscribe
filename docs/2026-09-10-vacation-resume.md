# Resume context (written 2026-09-10, before vacation)

Claude Code session to resume directly: **`session_01Uv6snhiQdtPvuhqivaEKnp`**
(`claude --resume session_01Uv6snhiQdtPvuhqivaEKnp` in this directory; every commit from
that session also carries the id as a `Claude-Session:` trailer). Project memory is
persistent either way — a fresh session knows the key decisions too.

## The one-paragraph picture

meetscribe graduated from PoC to "installed app" and is mid-way through becoming the
capture front-end for **borderless-knowledge (bk)**, which owns calendar, storage, person
identities, and the async annotation workflow. The API contract v1 between the two repos
is **agreed and fixture-locked** (2026-09-09); bk's side is implemented. meetscribe uploads
bundles automatically; the interactive review loop is the next build.

## Where we stand (all on `main`, pushed; verify tags with `git push --tags`)

- **v0.3.0 released 2026-09-09** (`6210c0e`), suite 585 passed + `nix flake check` green:
  - **P1 — config/XDG**: `config.toml` under `$XDG_CONFIG_HOME/meetscribe/`, precedence
    flag > env > config > default with origin tracking, `meetscribe config|init|path`,
    recordings under `$XDG_DATA_HOME/meetscribe/meetings/`, lazy `api_key_cmd`/`token_cmd`
    secrets (`SecretCmd`, executed only when needed).
  - **P2 — bk client + auto-upload**: `bk.py` (stdlib urllib, capabilities preflight,
    Idempotency-Key upload, workflow GET), `[bk]` config (auto_upload strictly opt-in),
    `meetscribe upload <dir>` (retry path, exit 1 on failure) vs auto-upload (warn + hint,
    exit 0 — deliberate asymmetry), `meetscribe status [--offline]`, doctor bk checks,
    `bk-workflow.json` persisted per meeting.
  - Earlier same-week: **v0.2.1** = Deepgram remote backend (nova-3, `STT_BACKEND`,
    embeddings always local CAM++ so pgvector history stays comparable).
- **bk's canonical fixtures are vendored** (`tests/fixtures/bk/`, 13 files, from
  `../borderless-knowledge/fixtures/meetscribe-contract/`). The swap caught + fixed a real
  bug: exact-key asserts violated the contract's additive rule. bk already ships the
  additive `preselected` + `new_voice_model` fields (our requested nice-to-haves).
- Contract: `docs/meetscribe-bk-contract-v1.md` (**v1 AGREED**, changes only in lockstep
  with bk's copy). Key agreed semantics: ONE task per `awaiting_review`, meeting-mapping
  gate strictly before speaker-annotation (`?meeting_id=` skips it), format v2-only,
  50 MiB cap, idempotency dedupes live workflows only, 7-day gate timeout, voice match
  scoped by (model name, sha256) — never a hard reject.

## Next up (in order; user decision: these land as **0.3.x patches**, not minors)

1. **P4 — interactive review loop (CLI) → 0.3.1.** NOT STARTED, no design doc yet.
   Poll after upload → render the gates in the terminal (bk's fixtures for both task
   types are already vendored + tested) → one-key confirm of suggestions (use
   `preselected`!) → local audio snippet playback (ffplay; WAVs are on disk) →
   revision-guarded submit (409 → refetch) → resume from `bk-workflow.json` →
   `web_url` fallback for unknown task types. Design here the **Interaction seam**
   (prompt/select/confirm protocol next to `Reporter`) — the tray GUI reuses it.
   bk endpoints 4+5 are live; first real end-to-end possible immediately.
2. **P3 — calendar-aware recording → 0.3.2.** `GET /meetings?around=now` at record start
   (opportunistic, NEVER blocks recording), roster confirm → glossary/keyterms +
   `--speakers`, `?meeting_id=` on upload. Needs bk endpoint 2 — check whether their
   "finished" includes it (fixtures `meetings-around*.json` exist, so probably yes).
3. **P5 — tray GUI** (the 1.0-ish milestone). Entry ticket first: stop-event refactor in
   `record.run` (programmatic stop instead of SIGINT-only) — zero external deps, good
   gap-filler anytime.

## Loose ends / reminders

- **Tags**: `git push --tags` if v0.2.x/v0.3.0 aren't on origin yet (commits are pushed).
- The Deepgram paid e2e (`tests/test_deepgram_e2e.py`) has still never run against the
  real API (needs `DEEPGRAM_API_KEY` + `MEETSCRIBE_MODELS`, ~1 ct). Same for a first real
  `meetscribe upload` against a live bk — worth doing before building P4 on top.
- `.envrc` is git-ignored; `direnv` loads `use nix` (+ your `.env` if you added `dotenv`).
- Working agreements that keep biting when forgotten: everything runs via
  `nix develop -c pytest -q` (plain pytest isn't on PATH); no new runtime deps ever
  without discussion (uv2nix closure); tests never touch network or real home; TDD with
  ultracode workflows for each phase (design doc → implementation plan → workflow with
  review lenses + adversarial verification) worked well three times — keep the pattern.
