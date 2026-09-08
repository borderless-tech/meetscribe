"""Deepgram remote backend: ``POST /v1/listen`` (pre-recorded, nova family).

The remote counterpart to :class:`~meetscribe.backends.LocalBackend`: the mic track is
uploaded plain (all words are ``me``), the system track with ``diarize=true`` (utterance
speakers become ``spk_N`` and each utterance doubles as a :class:`DiarSegment` for the
*local* embedding pass — embeddings never leave the machine). Plain stdlib ``urllib``
only: no SDK, no new dependencies. The transport is a seam (``opener``) so every test
stays offline.
"""

from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .backends import BackendResult, TrackInput
from .types import DiarSegment, Utterance, Word

API_URL = "https://api.deepgram.com/v1/listen"

# nova-3 keyterm prompting limits (docs): at most 100 terms and 500 tokens in total.
MAX_KEYTERMS = 100
MAX_KEYTERM_TOKENS = 500

RETRY_ATTEMPTS = 3
BACKOFF_BASE_S = 1.0


class DeepgramError(RuntimeError):
    """A Deepgram request failed for good (auth, exhausted retries, bad response)."""


@dataclass(frozen=True)
class DeepgramConfig:
    api_key: str
    language: str = "de"  # STT_LANGUAGE / --language; "multi" = nova-3 code-switching
    model: str = "nova-3"
    timeout_s: float = 300.0


def build_query(config: DeepgramConfig, *, diarize: bool, keyterms: list[str]) -> str:
    """The /v1/listen query string; ``keyterm`` is repeated per term (nova-3 only)."""
    params: list[tuple[str, str]] = [
        ("model", config.model),
        ("language", config.language),
        ("smart_format", "true"),
        ("utterances", "true"),
    ]
    if diarize:
        params.append(("diarize", "true"))
    if config.model.startswith("nova-3"):  # keyterm prompting is a nova-3 feature
        params.extend(("keyterm", term) for term in keyterms)
    return urllib.parse.urlencode(params)


def build_url(config: DeepgramConfig, *, diarize: bool, keyterms: list[str]) -> str:
    return f"{API_URL}?{build_query(config, diarize=diarize, keyterms=keyterms)}"


def keyterms_from_glossary(glossary: list[str]) -> list[str]:
    """Glossary → keyterm list: dedupe case-insensitively keeping first occurrence and
    input order (the glossary is already curated — no re-ranking), capped at
    ``MAX_KEYTERMS`` terms and ``MAX_KEYTERM_TOKENS`` whitespace-tokens in total. A term
    that would blow the token budget is dropped; later, smaller terms may still fit."""
    seen: set[str] = set()
    out: list[str] = []
    tokens = 0
    for raw in glossary:
        term = raw.strip()
        if not term or term.lower() in seen:
            continue
        if len(out) >= MAX_KEYTERMS:
            break
        n_tokens = len(term.split())
        if tokens + n_tokens > MAX_KEYTERM_TOKENS:
            continue
        seen.add(term.lower())
        out.append(term)
        tokens += n_tokens
    return out


# ---- transport ----------------------------------------------------------------------


def _default_opener(request: urllib.request.Request, timeout: float):
    return urllib.request.urlopen(request, timeout=timeout)  # noqa: S310 — fixed https URL


def _error_detail(err: urllib.error.HTTPError) -> str:
    """Best-effort extraction of Deepgram's ``err_msg`` from an error body."""
    try:
        body = err.read().decode("utf-8", "replace")
    except Exception:
        return ""
    try:
        data = json.loads(body)
    except ValueError:
        return body.strip()
    if isinstance(data, dict):
        return str(data.get("err_msg") or body.strip())
    return body.strip()


class DeepgramClient:
    """Thin ``POST /v1/listen`` client over stdlib urllib.

    ``opener`` (``callable(Request, timeout) → response``) and ``sleep`` are seams so
    tests can script transport outcomes and observe backoff without real waiting.
    """

    def __init__(self, config: DeepgramConfig, opener=None, sleep=None):
        self.config = config
        self._opener = opener or _default_opener
        self._sleep = sleep or time.sleep

    def transcribe_file(self, wav_path: str, *, diarize: bool, keyterms: list[str]) -> dict:
        """Upload one WAV file, return the parsed response JSON.

        Retries (with exponential backoff) on 429/5xx and transport failures —
        URLError, plus the response-phase timeouts/resets/truncated reads that urllib
        does NOT wrap in URLError — up to ``RETRY_ATTEMPTS`` total attempts; auth
        failures and other 4xx are final."""
        url = build_url(self.config, diarize=diarize, keyterms=keyterms)
        body = Path(wav_path).read_bytes()
        last_error = "unknown error"
        for attempt in range(RETRY_ATTEMPTS):
            if attempt:
                self._sleep(BACKOFF_BASE_S * 2 ** (attempt - 1))
            request = urllib.request.Request(
                url,
                data=body,
                headers={
                    "Authorization": f"Token {self.config.api_key}",
                    "Content-Type": "audio/wav",
                },
                method="POST",
            )
            try:
                response = self._opener(request, self.config.timeout_s)
                raw = response.read()
            except urllib.error.HTTPError as e:
                detail = _error_detail(e)
                if e.code in (401, 403):
                    raise DeepgramError(
                        f"Deepgram rejected the request (HTTP {e.code}: {detail or e.reason})"
                        " — check DEEPGRAM_API_KEY"
                    ) from e
                if e.code == 429 or e.code >= 500:
                    last_error = f"HTTP {e.code}: {detail or e.reason}"
                    continue
                raise DeepgramError(
                    f"Deepgram request failed (HTTP {e.code}: {detail or e.reason})"
                ) from e
            except urllib.error.URLError as e:
                last_error = f"network error: {e.reason}"
                continue
            except (TimeoutError, OSError, http.client.HTTPException) as e:
                # urllib wraps only the request phase in URLError; a socket timeout or
                # reset while WAITING FOR or READING the response — the long phase here,
                # Deepgram processes the whole upload before answering — propagates raw
                # (TimeoutError/ConnectionResetError/IncompleteRead) and is just as
                # transient, so it retries too.
                last_error = f"network error: {str(e) or e.__class__.__name__}"
                continue
            try:
                return json.loads(raw.decode("utf-8", "replace"))
            except ValueError as e:
                snippet = raw[:120].decode("utf-8", "replace").strip()
                raise DeepgramError(
                    f"Deepgram returned a 200 response that is not JSON ({snippet!r})"
                    " — is a proxy or captive portal intercepting api.deepgram.com?"
                ) from e
        raise DeepgramError(
            f"Deepgram request failed after {RETRY_ATTEMPTS} attempts ({last_error})"
        )


# ---- response → Utterance / DiarSegment mapping -------------------------------------


def _require_utterances(response: dict) -> list[dict]:
    results = response.get("results")
    if not isinstance(results, dict) or "utterances" not in results:
        raise DeepgramError(
            "Deepgram response has no results.utterances — expected utterances=true output"
        )
    if not results.get("channels"):
        raise DeepgramError(
            "Deepgram response has no channels — the audio may be empty or unreadable"
        )
    return results["utterances"]


def _words(utt: dict) -> tuple[Word, ...]:
    return tuple(
        Word(w.get("punctuated_word") or w.get("word", ""), float(w["start"]), float(w["end"]))
        for w in utt.get("words", [])
    )


class DeepgramBackend:
    """Remote :class:`~meetscribe.backends.TranscriptionBackend`: mic uploaded plain
    (hard-labelled ``me``), system uploaded with diarization (``spk_N`` per utterance,
    plus one :class:`DiarSegment` per utterance for the local embedding pass)."""

    name = "deepgram"

    def __init__(self, config: DeepgramConfig, client=None):
        self.config = config
        self.client = client or DeepgramClient(config)

    def transcribe(
        self,
        mic: TrackInput | None,
        system: TrackInput | None,
        glossary: list[str],
        reporter,
    ) -> BackendResult:
        keyterms = keyterms_from_glossary(glossary)
        mic_utts: list[Utterance] = []
        system_utts: list[Utterance] = []
        system_diar: list[DiarSegment] = []
        request_ids: list[str] = []
        model_versions: dict = {}

        if mic is not None:
            with reporter.stage("transcribe (mic, deepgram)"):
                response = self.client.transcribe_file(
                    mic.wav_path, diarize=False, keyterms=keyterms
                )
            mic_utts = [
                Utterance(
                    float(u["start"]), float(u["end"]), "me", "mic",
                    u.get("transcript", ""), _words(u),
                )
                for u in _require_utterances(response)
            ]
            request_ids, model_versions = self._collect_meta(
                response, request_ids, model_versions
            )

        if system is not None:
            with reporter.stage("transcribe (system, deepgram)"):
                response = self.client.transcribe_file(
                    system.wav_path, diarize=True, keyterms=keyterms
                )
            for u in _require_utterances(response):
                speaker = f"spk_{int(u.get('speaker', 0))}"
                system_utts.append(
                    Utterance(
                        float(u["start"]), float(u["end"]), speaker, "system",
                        u.get("transcript", ""), _words(u),
                    )
                )
                system_diar.append(
                    DiarSegment(float(u["start"]), float(u["end"]), speaker)
                )
            # system preferred as the model-identity source (it carries diarization too)
            request_ids, model_versions = self._collect_meta(
                response, request_ids, model_versions, prefer=True
            )

        return BackendResult(
            mic_utts,
            system_utts,
            system_diar,
            models_meta={
                "asr_model": f"deepgram-{self.config.model}",
                "segmentation_model": "deepgram-diarizer",
                # the remote substitute for local SHA-256 pins: Deepgram's model_info
                # (name/version/uuid) + the request ids, straight from the response
                "backend_model_versions": model_versions,
                "request_ids": request_ids,
            },
        )

    @staticmethod
    def _collect_meta(response, request_ids, model_versions, prefer=False):
        metadata = response.get("metadata") or {}
        if metadata.get("request_id"):
            request_ids = request_ids + [metadata["request_id"]]
        if metadata.get("model_info") and (prefer or not model_versions):
            model_versions = metadata["model_info"]
        return request_ids, model_versions
