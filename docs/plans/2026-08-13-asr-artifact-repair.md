# ASR artifact repair: detect-then-repair, not blind full-text rewrite

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan
> task-by-task. Strict TDD throughout: write the failing test, run it to confirm it fails, then
> implement minimally. Run `nix develop -c pytest -q` after every task; Tasks 1–7 must keep the
> whole suite green and <1 s (no model touched).

**Status:** design 2026-08-13. Supersedes the *strategy* of
`docs/plans/2026-08-12-transcript-cleanup-{design,implementation}.md` (glossary, name prompt,
`Cleaner` seam, guards, meta identity, `--no-cleanup`, `clean <dir>`, Nix pinning all stay — only
the LLM's *job* changes from "rewrite everything" to "repair flagged spans").

---

## Why

The 2026-08-12 pass sends **every** segment through a 7B GGUF and takes the model's rewrite. That
was aimed at the wrong defect.

- **The real defects are DECODER ARTIFACTS, not language errors.** On real captures the recurring
  damage is (a) *echoes / repetitions* — the decoder re-emits a word or short run ("the the the",
  "Borderless Borderless"), and (b) *broken / garbled words* — a single token mangled into a
  non-word ("Borderlestern", "verstehn Sie mich" fused, split fragments). The surrounding text is
  usually **clean German/English that needs no touching**.
- **Blind full-text LLM is slow.** ~19 min/meeting. It is decode-bound: llama.cpp CPU latency
  tracks **output** tokens, and we currently emit ~as many tokens as we feed in (~13k output
  tokens/meeting). Every clean sentence is regenerated for nothing.
- **Blind full-text LLM is unreliable.** Given a whole segment and told to "correct", a 7B model
  rewrites already-correct text, silently **translates** DE↔EN, and **invents** content. The
  `accept_candidate` ratio guards catch meltdowns but not these subtle, in-bounds corruptions.

**The pivot:** *detect artifacts deterministically, then repair only the flagged spans.* Most
segments are flagged in **zero** spans and are passed through byte-for-byte (no model call at
all). A flagged span emits only its few replacement tokens. Output tokens drop from **~13k → ~500**
per meeting, and since latency is output-bound, wall-clock collapses proportionally — while clean
text is now *structurally* untouchable because we never hand it to the model.

**What we keep** (all already built for 2026-08-12, reused unchanged):

- the injected **`Cleaner`** seam on `Components` (`NullCleaner` default → suite stays LLM-free);
- the persistent **glossary** (`glossary.py`) and the participant-name prompt;
- the anti-hallucination **guards** philosophy (empty / length-ratio), now joined by a
  phonetic/edit-distance guardrail;
- **meta identity** — every repairer model's name+SHA-256 travels into `meta.json`, `cleaned`
  flag + `format_version` bump; incomparable-model swaps stay structurally impossible;
- the **opt-in flag** (`--no-cleanup`) and the non-destructive `clean <dir>` subcommand;
- **Nix model-pinning** (`nix/models.nix`, `withLlm`) — the LLM stays a separate opt-in output.

Only `ManagedLlamaCleaner`'s *internal behaviour* changes (whole-segment rewrite → span repair);
its `clean(texts, glossary, reporter) -> CleanResult` signature and its place in the pipeline are
unchanged, so `pipeline.apply_cleanup`, `output.py`, and `meta.json` need no structural edits.

## The repair pipeline (per system-track segment)

```
raw text
  │
  ├─(1) echo collapse ────────── deterministic, gap≈0, no model. Removes echoes/repetitions.
  │
  ├─(2) broken-word flag ─────── bilingual hunspell de_DE ∪ en_US, minus glossary.
  │        for each word: broken ⇔ unknown to BOTH dicts AND not in glossary.
  │
  └─(3) span repair (flagged spans only)
           ├─ masked-LM fill-mask FIRST  (German BERT, ONE forward pass), candidates
           │    ranked by (4) phonetic/edit distance against the ASR surface form;
           └─ LLM span-fix FALLBACK      (multi-token / merge-garble the LM can't fill),
                the repaired token(s) then re-checked by (4).
  │
  └─(4) phonetic/Levenshtein guardrail — applied to WHICHEVER repairer produced the
         replacement: a candidate too far from the ASR form (in sound/edits) is rejected,
         keep raw. This is what stops the model from inventing a plausible-but-wrong word.
```

A segment with no flagged spans after (1)/(2) is returned unchanged and **never** reaches (3) —
that is where the token/latency win comes from.

### Constraint: production is offline / CPU / NO PyTorch

`what-we-build.md` and CLAUDE.md are explicit — no PyTorch, no CUDA, no HF token. The masked-LM in
stage (3) is a real integration task, **not** a `pip install transformers`:

- The German BERT (e.g. a `bert-base-german-cased`-class fill-mask model) must ship as **ONNX** and
  run under **`onnxruntime`** — the same native/offline stack sherpa already uses. Torch may be used
  **offline, once, at model-prep time** to export ONNX, but the exported artifact + its tokenizer
  (a `tokenizers`/vocab file, no torch at runtime) are what gets pinned in `nix/models.nix` and
  loaded in production. `onnxruntime` is already transitively present via sherpa; confirm it is a
  first-class pinned dep in `pyproject.toml` (`==`, never `>=`) if we import it directly.
- Fill-mask is **one forward pass** per flagged single-token span: replace the ASR token with
  `[MASK]`, run once, take the top-k candidates, rank by (4) against the ASR surface form. No
  autoregressive decode → far cheaper than the LLM and fully deterministic.
- **Primary span-repairer is TBD.** A masked-LM-vs-LLM benchmark is running **separately** to
  decide whether the fill-mask path or the LLM path is the default primary (the other becomes the
  fallback). This plan builds **both** behind the same span-repair interface so the benchmark just
  flips a default; do not block on its result. Stage-4 guardrail is shared regardless.

---

## Sequencing

Tasks 1–4 are pure/deterministic and fully verifiable in the fast suite with **no model** (hunspell
is a system lib but the flag logic is tested against an injected dictionary fake, and echo/phonetic
are stdlib-only). Task 5 rewires the LLM repairer behind the existing seam using a fake client.
Tasks 6–8 are the ML/Nix boundary (hunspell dicts, ONNX masked-LM, model pinning) — unit-tested
with fakes, real path only in the opt-in e2e. Ship 1–5 before the benchmark returns; 6–8 land the
real models.

---

### Task 1: `echo.py` — deterministic echo/repetition collapse

**Files:**
- Create/verify: `src/meetscribe/echo.py`  *(noted as DONE — if present, verify against these
  tests and skip to Task 2; if absent, build it here)*
- Test: `src/meetscribe/../tests/test_echo.py`  →  `tests/test_echo.py`

Pure, stdlib-only, operates on `Word` tuples so it composes with `align`/`merge` and keeps timings.
Collapse a run of consecutive words that are (case-insensitively, punctuation-stripped) identical
into a single word spanning the run — but **only** when the inter-word gap ≈ 0 (decoder echo, not a
real repeated word across a pause). Also collapse an immediately-repeated short n-gram (bigram /
trigram) under the same gap≈0 rule. Style: mirror `merge.coalesce_utterances` (frozen dataclasses,
`replace`, terse WHY docstring on the order-safety / gap rationale).

**Step 1 — failing tests** (write first, run, confirm fail):
```python
from meetscribe.echo import collapse_echoes
from meetscribe.types import Word

def _w(s, a, b): return Word(w=s, start=a, end=b)

def test_collapses_adjacent_identical_zero_gap():
    ws = [_w("the", 0.0, 0.2), _w("the", 0.20, 0.4), _w("cat", 0.4, 0.7)]
    out = collapse_echoes(ws)
    assert [w.w for w in out] == ["the", "cat"]
    assert out[0].start == 0.0 and out[0].end == 0.4   # span kept

def test_keeps_repeat_across_a_real_pause():
    ws = [_w("no", 0.0, 0.2), _w("no", 1.5, 1.7)]      # 1.3 s gap → genuine
    assert [w.w for w in collapse_echoes(ws)] == ["no", "no"]

def test_case_and_punct_insensitive():
    ws = [_w("Borderless", 0.0, 0.3), _w("borderless,", 0.30, 0.6)]
    assert [w.w for w in collapse_echoes(ws)] == ["Borderless"]  # first form kept

def test_collapses_repeated_bigram():
    ws = [_w("you", 0,0.2), _w("know", 0.2,0.4), _w("you", 0.40,0.6), _w("know", 0.6,0.8)]
    assert [w.w for w in collapse_echoes(ws)] == ["you", "know"]

def test_empty_and_singleton_are_noops():
    assert collapse_echoes([]) == []
    assert [w.w for w in collapse_echoes([_w("hi", 0,0.3)])] == ["hi"]
```

**Step 2 — run, expect FAIL** (`cannot import name 'collapse_echoes'`).
**Step 3 — implement `collapse_echoes(words, max_gap=0.15)`** minimally.
**Step 4 — PASS. Step 5 — commit** `feat: deterministic echo/repetition collapse (echo.py)`.

---

### Task 2: bilingual OOV broken-word flagger — pure logic

**Files:**
- Create: `src/meetscribe/artifacts.py`  *(flag detection; kept separate from `echo.py` so echo
  stays word-timing-only and artifacts owns the dictionary seam)*
- Test: `tests/test_artifacts.py`

A word is **broken ⇔ unknown to BOTH `de_DE` and `en_US` AND not in the glossary**. Requiring
unknown-to-both is the crux: legitimately code-switched meetings ("wir machen ein Standup") must
not flag `Standup` (known to `en_US`), and `Geschäftsführer` must not flag (known to `de_DE`).
Glossary terms (proper nouns hunspell has never heard of) are never flagged. This task is the
**pure** flag logic behind an injected `SpellChecker` protocol — no hunspell import yet (that is
Task 6). Return flagged **spans** (contiguous runs of broken words) as `(start_idx, end_idx)` word
indices so a merge-garble ("verstehnSie") flags as one span, not two.

**Step 1 — failing tests** (inject a fake bilingual checker):
```python
from meetscribe.artifacts import flag_broken_spans

class FakeDict:
    def __init__(self, known): self.known = {w.casefold() for w in known}
    def known_word(self, w): return w.casefold() in self.known

def test_flags_word_unknown_to_both():
    de = FakeDict(["der", "ist", "gut"]); en = FakeDict(["the", "is", "good"])
    words = ["der", "Borderlestern", "ist", "gut"]
    assert flag_broken_spans(words, de, en, glossary=[]) == [(1, 1)]

def test_known_to_either_dict_is_clean():
    de = FakeDict(["wir", "machen", "ein"]); en = FakeDict(["standup"])
    assert flag_broken_spans(["wir","machen","ein","Standup"], de, en, []) == []

def test_glossary_term_never_flagged():
    de = FakeDict(["die"]); en = FakeDict(["the"])
    assert flag_broken_spans(["die", "Borderless"], de, en, ["Borderless"]) == []

def test_contiguous_broken_run_is_one_span():
    de = FakeDict(["ich"]); en = FakeDict([])
    assert flag_broken_spans(["ich","xqz","wvu","ich"], de, en, []) == [(1, 2)]

def test_punctuation_and_case_stripped_before_lookup():
    de = FakeDict(["gut"]); en = FakeDict([])
    assert flag_broken_spans(["Gut,"], de, en, []) == []
```

**Step 2 — FAIL. Step 3 — implement** `SpellChecker` protocol (`known_word(str)->bool`),
`flag_broken_spans(words, de, en, glossary) -> list[tuple[int,int]]` with the same
casefold+punct-strip normalization `echo.py`/`glossary.py` use; glossary compared casefold.
**Step 4 — PASS. Step 5 — commit** `feat: bilingual OOV broken-word span flagger (artifacts.py)`.

---

### Task 3: `phonetic.py` — phonetic + Levenshtein distance guardrail

**Files:**
- Create: `src/meetscribe/phonetic.py`
- Test: `tests/test_phonetic.py`

Pure, stdlib-only, deterministic. Two primitives + one decision:
- `levenshtein(a, b) -> int` (classic DP);
- `phonetic_key(word) -> str` — a small, language-tolerant Soundex/metaphone-lite fold (collapse
  vowels, common DE↔EN digraph equivalences like `sch`≈`sh`, `v`≈`f`, `z`≈`ts`) so "Borderless" and
  "Borderlestern" share a key while "Borderless" and "Warehouse" do not;
- `is_plausible_repair(asr_form, candidate, *, max_edit_ratio=0.5) -> bool` — True iff the candidate
  is phonetically close to the ASR surface form **or** within an edit-distance ratio. This is the
  Stage-4 guardrail every repairer passes through; it is what stops a fill-mask top-1 or an LLM
  from swapping in a real-but-unrelated word.

**Step 1 — failing tests:**
```python
from meetscribe.phonetic import levenshtein, phonetic_key, is_plausible_repair

def test_levenshtein_basics():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3 and levenshtein("abc", "abc") == 0

def test_phonetic_key_groups_soundalikes():
    assert phonetic_key("Borderlestern") == phonetic_key("Borderless") \
        or is_plausible_repair("Borderlestern", "Borderless")

def test_plausible_accepts_near_and_rejects_far():
    assert is_plausible_repair("Borderlestern", "Borderless")   # near → accept
    assert not is_plausible_repair("Borderlestern", "Warehouse") # unrelated → reject
    assert not is_plausible_repair("ja", "absolutely")           # invention → reject
```

**Step 2 — FAIL. Step 3 — implement.** **Step 4 — PASS. Step 5 — commit**
`feat: phonetic + Levenshtein repair guardrail (phonetic.py)`.

---

### Task 4: span-repair interface + `MaskedLmRepairer`/`LlamaSpanRepairer` (fakes)

**Files:**
- Create: `src/meetscribe/repair.py`  *(the span-repairer seam; both strategies live here)*
- Test: `tests/test_repair.py`

Defines the interface the two repairers share and orchestrates Stage (3)+(4) for one segment:
```
class SpanRepairer(Protocol):
    def repair_span(self, words: list[str], span: tuple[int, int], glossary: list[str]) -> list[str] | None
```
- `MaskedLmRepairer(fill_mask_fn)` — masks the flagged token(s), takes top-k from an injected
  `fill_mask_fn(tokens, mask_idx) -> list[str]`, ranks by `phonetic.is_plausible_repair` against the
  ASR form, returns the best plausible candidate or `None` (→ caller falls back to LLM).
- `LlamaSpanRepairer(client)` — sends ONLY the flagged span + a tight window of context to the
  injected `LlamaClient.complete`, parses back the replacement token(s), then re-checks each against
  `phonetic.is_plausible_repair`. Reuses the existing `LlamaClient` protocol from `llama.py`.
- `repair_segment(words, spans, primary, fallback, glossary) -> list[Word]` — walks flagged spans,
  tries `primary.repair_span`, on `None` tries `fallback`, on `None` keeps raw; rebuilds `Word`
  tuples **reusing the ASR span's start/end** so timings never move (mirrors `align`/`merge`).

**Step 1 — failing tests** (all fakes, no model):
```python
from meetscribe.repair import MaskedLmRepairer, LlamaSpanRepairer, repair_segment

def test_masked_lm_picks_phonetically_closest_candidate():
    fm = lambda toks, i: ["Warehouse", "Borderless", "Firma"]   # top-k, wrong order
    r = MaskedLmRepairer(fm)
    assert r.repair_span(["die","Borderlestern","GmbH"], (1,1), []) == ["Borderless"]

def test_masked_lm_returns_none_when_nothing_plausible():
    r = MaskedLmRepairer(lambda t, i: ["Warehouse", "Elephant"])
    assert r.repair_span(["die","Borderlestern"], (1,1), []) is None

def test_llama_span_repairer_rechecks_with_guardrail():
    class C:  # returns an implausible invention → rejected → None
        def complete(self, prompt, max_tokens=None): return "absolutely"
    assert LlamaSpanRepairer(C()).repair_span(["ja","xqz"], (1,1), []) is None

def test_repair_segment_falls_back_and_preserves_timings():
    from meetscribe.types import Word
    words = [Word("die",0,.3), Word("Borderlestern",.3,.9), Word("GmbH",.9,1.2)]
    prim = MaskedLmRepairer(lambda t,i: ["Warehouse"])           # → None
    fb   = LlamaSpanRepairer(type("C",(),{"complete":lambda s,p,max_tokens=None:"Borderless"})())
    out = repair_segment(words, [(1,1)], prim, fb, [])
    assert [w.w for w in out] == ["die","Borderless","GmbH"]
    assert out[1].start == .3 and out[1].end == .9              # timings intact
```

**Step 2 — FAIL. Step 3 — implement.** **Step 4 — PASS. Step 5 — commit**
`feat: span-repair interface with masked-LM + LLM strategies (repair.py)`.

---

### Task 5: rewire `ManagedLlamaCleaner` to detect-then-repair

**Files:**
- Modify: `src/meetscribe/cleanup.py` (`LlamaCleaner`/`ManagedLlamaCleaner` internals)
- Test: `tests/test_cleanup.py`

Keep the public `Cleaner.clean(texts, glossary, reporter) -> CleanResult` signature and the
`model_info` attribute (meta identity) **unchanged** — only the per-segment work changes. New flow
per segment: `echo.collapse_echoes` → `artifacts.flag_broken_spans` → if no spans, pass through
(count as a no-op, NOT `cleaned`); else `repair.repair_segment(primary, fallback)`; count `cleaned`
only when a span actually changed. Old `build_prompt`/`token_budget`/whole-segment
`accept_candidate` become unused by the new path — delete the dead whole-segment prompt or repoint
`accept_candidate` reuse into `phonetic.is_plausible_repair`; do not leave two guard systems.

**Step 1 — failing tests:** a `FakeCleaner`-style setup with an injected fake spell checker +
fake fill-mask asserting: (a) a segment with no OOV words is returned **byte-identical** and the
model is **never called** (assert call count 0 — the latency claim); (b) a segment with one garbled
glossary term is repaired via the masked-LM path; (c) a merge-garble the fill-mask can't handle
falls through to the LLM; (d) `res.cleaned`/`res.kept_raw` counts reflect spans, not segments.
**Step 2–4** red→green. **Step 5 — commit** `feat: detect-then-repair cleanup (spans only, not full text)`.

---

### Task 6: hunspell dictionaries behind the `SpellChecker` seam

**Files:**
- Create: `src/meetscribe/spell.py`  (real `HunspellChecker` implementing `known_word`)
- Modify: `pyproject.toml` (pin the hunspell binding, `==` not `>=`)
- Modify: `nix/models.nix` **or** `nix/package.nix` (ship `hunspell` + `de_DE`/`en_US` dicts on the
  runtime PATH / as pinned inputs — a **NEW dependency**)
- Test: `tests/test_spell.py` (skips unless the dicts are present, like `test_e2e.py`)

`HunspellChecker(de_path, en_path)` wrapping the pinned dicts; construction failure → the caller
degrades to "flag nothing" (never crash a run for a missing dict). The pure flag logic (Task 2) is
already tested against the fake, so this task is a thin adapter + the Nix/dep wiring only. Pin the
`de_DE`/`en_US` dictionaries by hash in Nix exactly as the ASR/embedding models are pinned.

**Step 1 — failing test** (opt-in skip). **Step 3 — implement adapter + Nix/dep pins.**
**Step 5 — commit** `build: pin hunspell + de_DE/en_US dictionaries; feat: HunspellChecker`.

---

### Task 7: ONNX masked-LM (German BERT) under onnxruntime

**Files:**
- Create: `src/meetscribe/masklm.py` (loads the ONNX fill-mask + tokenizer, exposes `fill_mask_fn`)
- Modify: `nix/models.nix` (pin the exported ONNX + tokenizer under e.g. `mlm/`, `withLlm`-style
  opt-in output; record name+SHA-256 for `meta.json`)
- Modify: `pyproject.toml` if `onnxruntime`/`tokenizers` must be first-class (`==` pins)
- Test: `tests/test_masklm.py` (opt-in skip unless the ONNX model is present)

**Integration reality (the NO-PyTorch constraint):** the German BERT ships as **ONNX + tokenizer**,
run via **`onnxruntime`** (already in the sherpa stack) — torch is used only offline at export time,
never at runtime/CI. `fill_mask_fn(tokens, mask_idx) -> list[str]` runs **one** forward pass and
returns top-k for Task-4's `MaskedLmRepairer`. Document the export recipe (source model, opset,
how the SHA was obtained via `nix store prefetch-file`) alongside the pin, mirroring the GGUF note
in `nix/models.nix`.

**Step 1 — failing test** (opt-in skip). **Step 3 — implement loader + wire into
`build_components` so `ManagedLlamaCleaner` gets a real `MaskedLmRepairer` when the model is
present, else masked-LM path is skipped and only the LLM fallback / raw applies.**
**Step 5 — commit** `build: pin ONNX German masked-LM; feat: onnxruntime fill-mask repairer`.

---

### Task 8: e2e — artifact repair on a seeded transcript (opt-in)

**Files:**
- Modify: `tests/test_e2e.py` (or a new `tests/test_repair_e2e.py`), mirroring the
  `MEETSCRIBE_MODELS` skip guard.

Build a short transcript containing (a) an echo run, (b) a garbled glossary term, (c) a clean
code-switched sentence that must be left untouched. Run the real `clean_existing` path; assert: the
echo is collapsed, the garbled term is repaired to the glossary spelling, the clean sentence is
**byte-identical**, per-word timings are **byte-identical** to the input, and `meta.json` carries
the repairer model identity + `cleaned:true`. Commit `test: e2e artifact repair collapses echoes,
fixes garbles, preserves clean text + timings`.

---

## Notes for the executor

- DRY/YAGNI/strict TDD; commit per task. Suite stays green + <1 s for Tasks 1–5.
- **Never touch `words`/timings** — a repair reuses the flagged span's own start/end. `raw_text`
  keeps the original ASR text (unchanged mechanism from 2026-08-12).
- Only `track=='system'` is repaired; the mic (`me`) track passes through (as today).
- The **primary vs fallback** span-repairer default is set by the separate masked-LM-vs-LLM
  benchmark — Task 5 wires both and exposes the default as a single toggle; do not hard-decide it.
- Model identity (hunspell dicts, ONNX MLM, GGUF) MUST travel into `meta.json` — same
  incomparable-swap rule as embeddings. Pin every model by hash in Nix.
