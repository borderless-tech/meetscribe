# Handover: broken-word `suggestions` in `transcript.json` (for borderless-knowledge)

meetscribe now flags likely-broken ASR words and attaches **correction candidates** for a
human to accept during the existing annotation pass. It does **not** auto-apply them (acoustic
distance can't tell a correct fix from a plausible-wrong one — that's why a human decides).
This note is what borderless-knowledge needs to consume it.

## What changed in the artifact

`transcript.json` is now **`format_version: 2`** (see `meta.json`). Two additions:

1. Per segment: a new **`raw_text`** alongside `text`. `text` is echo-collapsed and otherwise
   the raw ASR words (broken words are left as-is); `raw_text` is the original ASR text before
   any cleanup. (Echo loops like `"und und und"` are already collapsed in `text`/`words`.)
2. A top-level **`suggestions`** array: one entry per flagged broken word.

```jsonc
{
  "meeting_id": "meetscribe-2026-08-12T12-00-55",
  "duration_s": 2576.95,
  "cleaned": true,                       // did a real cleanup pass run? (false → no suggestions)
  "suggestions": [
    {
      "segment": 12,                     // index into segments[]
      "word_index": 4,                   // index into segments[12].words[]
      "start": 207.8, "end": 208.3,      // the word's timestamps (redundant locator)
      "original": "geernt",              // the flagged (raw) word
      "candidates": ["geerntet", "gelernt", "geerbt", "gern"]  // ranked, best/most-sound-alike first
    }
  ],
  "segments": [
    {
      "start": 195.0, "end": 240.0, "speaker": "spk_1", "track": "system",
      "text": "…richtig geernt …",       // display this; still contains the raw broken word
      "raw_text": "…richtig geernt …",    // pre-cleanup original
      "words": [ { "w": "richtig", "start": 207.2, "end": 207.7 },
                 { "w": "geernt",  "start": 207.8, "end": 208.3 } ]
    }
  ]
}
```

## What borderless-knowledge should do

During the annotation pass (alongside speaker labelling), for each entry in `suggestions`:

1. **Locate the word.** `segments[suggestion.segment].words[suggestion.word_index]` — its `w`
   equals `suggestion.original`. (`start`/`end` are provided as a cross-check / for audio seek.)
2. **Offer the shortlist.** Show `candidates` (already ranked, best first) plus:
   - **Keep original** (the word wasn't broken — common; e.g. a name or English term), and
   - **Type your own** (the true word may not be in the list — see caveats).
3. **On accept**, apply the reviewer's choice to that word: set `words[word_index].w` to the
   chosen text and re-join the segment's `text`. Leave the word's `start`/`end` unchanged.
4. **On keep/skip**, leave the word as-is. Nothing is lost — `text`/`raw_text` already hold the
   raw word.

That's the whole contract. Accepted edits live in borderless-knowledge; meetscribe never mutates
the transcript beyond echo-collapse.

## Caveats worth coding for

- **Suggestions are optional and often absent.** `cleaned: false` (or an empty `suggestions`)
  means no cleanup ran (e.g. the LLM/dicts weren't present) — just show the transcript.
- **Not every flag is really broken.** Flagging over-produces on names/rare terms; the reviewer
  will "keep original" a lot. That's expected and cheap by design.
- **The right word may not be in `candidates`.** For badly-garbled audio neither hunspell nor the
  LLM can recover it — always allow free-text entry.
- **No numeric confidence.** Ranking is the only signal (position 0 = closest-sounding / most
  likely). If you want to sort/threshold, use list order, not a score (there isn't one).
- **Timestamps are stable.** `word_index` indexes the *post-echo-collapse* words that ship in the
  same file, so it always aligns with `segments[...].words`.
- **Model identity travels in `meta.json`** (`cleanup_model`: name + sha256) when `cleaned` — useful
  if you ever want to show/trust suggestions differently per model.
