# `.mscribe` format, version 2

Complete reader-side description of the meetscribe upload bundle at `format_version: 2`
(the version currently emitted). Intended for borderless-knowledge's ingest gate.

**v2 is strictly additive over v1** — no member, field, or naming was removed or renamed.
A v1 reader only fails at the version gate; the safe fix is to accept `format_version in {1, 2}`
and treat the v2 fields as optional.

## Container

A `.mscribe` file is a **plain zip** (deflate) named `meeting-<meeting_id>.mscribe`, holding
exactly three fixed-name members:

| member | content |
|---|---|
| `meta.json` | model identity + meeting metadata — **parse this first, gate on `format_version`** |
| `transcript.json` | diarized transcript with per-word timestamps |
| `embeddings.npz` | speaker vectors (numpy archive) |

## `meta.json`

```jsonc
{
  "embedding_model": "...",             // CAM++ model name
  "embedding_model_sha256": "...",      // pins the model → vectors from different models are incomparable
  "embedding_dim": 192,                 // sizes the npz vectors — read it, never assume (spec's 512 is stale)
  "asr_model": "...",
  "segmentation_model": "...",
  "sample_rate": 16000,
  "meetscribe_version": "...",
  "meeting_id": "meetscribe-2026-08-12T12-00-55",
  "started_at": "2026-08-12T12:00:55+02:00",  // tz-aware ISO 8601 WITH offset (calendar matching)
  "ended_at":   "2026-08-12T12:43:52+02:00",
  "duration_s": 2576.95,                // float seconds, like all times in the format
  "format_version": 2,

  // v2 additions:
  "cleaned": true,                      // did the LLM cleanup pass actually run?
  "cleanup_model": {                    // OPTIONAL — only when an LLM contributed; may be absent
    "name": "Qwen2.5-7B-Instruct-Q4_K_M",  // even with cleaned:true (hunspell-only suggest mode)
    "sha256": "...",
    "temperature": 0.0
  }
}
```

## `transcript.json`

```jsonc
{
  "meeting_id": "meetscribe-2026-08-12T12-00-55",
  "duration_s": 2576.95,
  "cleaned": true,                      // v2: mirrors meta.json
  "suggestions": [                      // v2: broken-word candidates for human review; often []
    {
      "segment": 12,                    // index into segments[]
      "word_index": 4,                  // index into segments[12].words[]
      "start": 207.8, "end": 208.3,     // the word's timestamps (redundant locator / audio seek)
      "original": "geernt",             // the flagged raw word (== words[word_index].w)
      "candidates": ["geerntet", "gelernt"]  // ranked, best first; no numeric confidence exists
    }
  ],
  "segments": [
    {
      "start": 195.0, "end": 240.0,     // float seconds
      "speaker": "me",                  // "me" (the recording user, mic track) or "spk_N"
      "track": "mic",                   // "mic" | "system"
      "text": "…",                      // display text (possibly cleaned; suggestions NOT applied)
      "raw_text": "…",                  // v2: verbatim ASR text before any cleanup; == text if none ran
      "words": [ { "w": "richtig", "start": 207.2, "end": 207.7 } ]  // timings are never rewritten
    }
  ]
}
```

Invariants: segments are time-sorted; mic-track segments are always `speaker: "me"` and are never
diarized; only system-track `text` is ever cleaned; `suggestions[].word_index` aligns with the
`words` arrays as shipped (post-echo-collapse). Consuming `suggestions` is described in
`2026-08-13-borderless-knowledge-handover.md`.

## `embeddings.npz`

Standard numpy `.npz` with five arrays (unchanged from v1):

| key | shape / dtype | meaning |
|---|---|---|
| `turn_ids` | (N,) str | one id per speaker turn |
| `turn_vectors` | (N, embedding_dim) float32 | per-turn speaker embeddings |
| `turn_speakers` | (N,) str | speaker label per turn (`spk_N` / `me`) |
| `cluster_ids` | (M,) str | one per speaker cluster |
| `cluster_vectors` | (M, embedding_dim) float32 | per-speaker centroid (from concatenated audio) |

Invariant: every speaker appearing in embeddings also appears in the transcript, and every
transcript speaker has a cluster vector. `embedding_dim` comes from `meta.json` (192 today).

## v1 → v2 reader checklist

1. Accept `format_version: 2` (additive bump — everything v1 expected is still there).
2. New per-segment `raw_text` (fallback: equals `text`).
3. New top-level `cleaned` in both `meta.json` and `transcript.json` (fallback: `false`).
4. New optional `meta.json.cleanup_model` (may be absent even when `cleaned: true`).
5. New top-level `transcript.json.suggestions` (fallback: `[]`).

Bump policy (bundle design doc): additive members/fields keep the version; renames/removals bump it.
