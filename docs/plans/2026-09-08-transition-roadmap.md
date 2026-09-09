# Transition roadmap: PoC CLI → installed app with bk integration + GUI

Big-picture sequencing only. **Each phase gets its own design + implementation plan before
any code** (drafted together, per phase — explicitly not part of this document). Companion
docs: `docs/meetscribe-bk-contract-v1.md` (the API contract),
`docs/2026-09-08-bk-handover-workflow-api.md` (what bk builds, in which order).

Starting point (v0.2.1): dual-backend CLI (local sherpa / Deepgram nova-3), `.mscribe`
bundle default-on, manual upload, glossary + keyterms, config via env vars only, recordings
land in CWD.

End state: an "installed" application — CLI for terminal users, tray GUI for everyone
else — that knows the current meeting from bk's calendar, records, transcribes, walks the
user through one small confirm/correct step, and lands the finished transcript in bk
automatically.

## Phase ordering and why

The UX magic is in the data flow, not the GUI — so the full loop is built and proven
**CLI-first** (cheap iteration, testable, terminal users get the payoff immediately), and
the GUI wraps proven flows at the end. Config comes first because every later phase needs a
home for its settings (bk URL, token, upload policy). bk implements the contract in
parallel from the handover; phases 2–4 each consume endpoints as they land (build order is
aligned in the handover).

```
P1 config/XDG ──► P2 bk client+upload ──► P3 calendar-aware recording ──► P4 review loop ──► P5 tray GUI
                        ▲                        ▲                              ▲
                bk: capabilities+upload      bk: meetings query        bk: workflows+tasks
```

## Phase 1 — "Installed" foundation (config + XDG)

TOML config at `$XDG_CONFIG_HOME/meetscribe/config.toml` (stdlib `tomllib`, zero new deps;
humans edit the file). Precedence everywhere: **flags > env > config > defaults** — extends
the existing `resolve_backend`/`resolve_language` pattern. Recordings default to
`$XDG_DATA_HOME/meetscribe/meetings/` (configurable; `-o` still wins). Glossary already
lives under the config dir — fold its path handling into the same module. Likely a
`meetscribe config` subcommand (show effective config + origins).
*Independent of bk. Small. Instant maturity win.*

## Phase 2 — bk client + auto-upload

`[bk]` config section (base URL, token). Capabilities preflight (format_version gate as a
clear pre-upload message). After bundling: upload, persist `workflow_id` + bk response in
the meeting dir. Auto-upload is an **explicit config opt-in** (privacy: with the deepgram
backend the pipeline is then cloud-in *and* cloud-out — both legs stated in config, neither
a surprise). A `meetscribe status` view over local meetings + their workflow states.
Client code tests against the shared contract fixtures, offline — same pattern as the
Deepgram fixtures.
*Needs: P1; bk endpoints 1+3. Already kills the manual-upload step on its own.*

## Phase 3 — Calendar-aware recording

At record start: opportunistic `GET /meetings?around=now` (short timeout, cached into the
meeting dir, **recording never blocks on bk**). Roster confirm prompt (CLI): attendee names
→ glossary/keyterms, count → `--speakers` hint, chosen `meeting_id` → attached to the later
upload. Reconciliation fallback at process/upload time via `meta.started_at` for recordings
made offline. Requires making all record-flow prompts non-interactive-capable (flags), which
P5 needs anyway.
*Needs: P2; bk endpoint 2.*

## Phase 4 — Interactive review loop (completes the vision, CLI-first)

Poll the workflow after upload; render `speaker_annotation` + `meeting_mapping` tasks in
the terminal (rich): per-speaker talk time + sample utterances from the local transcript,
**local audio snippet playback** (ffplay — the raw WAVs are on disk), bk's suggestions as
one-key confirmations. Submit with revision handling; resume from persisted `workflow_id`
("1 unfinished review"); `web_url` fallback for anything unknown. Architectural principle
to establish here: an injected **Interaction seam** (prompt/select/confirm protocol) next
to `Reporter`, so P5's GUI reuses the flow logic unchanged.
*Needs: P3; bk endpoints 4+5. After this phase, terminal users have the entire
"start → it knows the meeting → confirm → done, it's in bk" experience.*

## Phase 5 — Tray GUI

Entry ticket (small refactor, useful standalone): programmatic stop for `record.run`
(injected stop event instead of SIGINT-only). Then the tray MVP: idle/recording/processing
states, start/stop, roster + review dialogs implementing the P4 Interaction seam,
notifications. Toolkit decision in the phase plan (pystray vs PySide6 vs Tauri sidecar —
revisit against the DE reality, GNOME needs an extension for tray icons). Ships as a
separate flake app (`nix run .#tray`); the CLI remains a first-class, fully supported
frontend forever.
*Needs: P4 flows proven.*

## Backlog (deliberately not scheduled)

Distribution beyond `nix run` (flatpak?), desktop notifications from the CLI, multi-meeting
queue, surfacing per-word `language` tags, live streaming transcription (the deepgram-sdk
revisit trigger), merged-single-track cost mode.

## Cross-cutting rules

- Versioning (user decision 2026-09-09): P1+P2 ship as **0.3.0**; P3–P5 land as **0.3.x
  patch releases** (keeping the minor-bump count low during the bk-integration stretch).
  Changelog fed as features land either way.
- Contract/fixture lockstep discipline (see contract doc) from P2 on.
- Nothing in P2–P5 may degrade the offline/local-only path: bk unreachable and
  `backend=local` must always yield a complete local artifact set.
