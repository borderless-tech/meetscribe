"""bk (borderless-knowledge) client: capabilities, bundle upload, workflow state.

Consumes contract v1 (``docs/meetscribe-bk-contract-v1.md``): after ``record``/
``process`` the ``.mscribe`` bundle is POSTed to bk, which answers with an async
workflow reference that is persisted next to the meeting artifacts
(``bk-workflow.json`` — NOT a bundle member). Plain stdlib ``urllib`` only, no new
dependencies; the transport is a seam (``opener``) so every test stays offline —
the exact same shape as :mod:`meetscribe.deepgram`.
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

from .output import FORMAT_VERSION

API_PREFIX = "/api/meetscribe/v1"  # the v1 IS contract_version 1 (new version = new path)
CONTRACT_VERSION = 1

RETRY_ATTEMPTS = 3
BACKOFF_BASE_S = 1.0

WORKFLOW_REF_NAME = "bk-workflow.json"


class BkError(RuntimeError):
    """A bk request failed for good (auth, exhausted retries, bad response).

    The message is always actionable — it names the knob to turn."""


@dataclass(frozen=True)
class BkConfig:
    base_url: str  # e.g. "https://bk.example.com"; a trailing slash is tolerated
    token: str
    timeout_s: float = 120.0  # uploads carry MBs on slow links


def build_url(config: BkConfig, path: str) -> str:
    """``<base_url>/api/meetscribe/v1<path>`` with any trailing slash stripped."""
    return f"{config.base_url.rstrip('/')}{API_PREFIX}{path}"


# ---- transport ----------------------------------------------------------------------


def _default_opener(request: urllib.request.Request, timeout: float):
    return urllib.request.urlopen(request, timeout=timeout)  # noqa: S310 — https base_url from config


def _error_detail(err: urllib.error.HTTPError) -> str:
    """Best-effort extraction of the contract's ``error.message`` from an error body."""
    try:
        body = err.read().decode("utf-8", "replace")
    except Exception:
        return ""
    try:
        data = json.loads(body)
    except ValueError:
        return body.strip()
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        return body.strip()
    return body.strip()


class BkClient:
    """Thin contract-v1 client over stdlib urllib.

    ``opener`` (``callable(Request, timeout) → response``) and ``sleep`` are seams so
    tests can script transport outcomes and observe backoff without real waiting.
    """

    def __init__(self, config: BkConfig, opener=None, sleep=None):
        self.config = config
        self._opener = opener or _default_opener
        self._sleep = sleep or time.sleep

    def capabilities(self) -> dict:
        """``GET /capabilities`` — the pre-upload contract handshake."""
        return self._request("GET", build_url(self.config, "/capabilities"))

    def upload_bundle(
        self, bundle_path, *, meeting_id: str, calendar_meeting_id: str | None = None
    ) -> dict:
        """``POST /bundles`` (zip body) → the 202 workflow reference.

        ``Idempotency-Key`` is the meeting id from ``meta.json``: re-uploading while a
        workflow is live returns the existing workflow; after ``done``/``failed`` bk
        starts a new one (contract carve-out) — so retries are always safe.
        ``calendar_meeting_id`` (``?meeting_id=``) is sent only when known (Phase 3)."""
        url = build_url(self.config, "/bundles")
        if calendar_meeting_id is not None:
            url += "?" + urllib.parse.urlencode({"meeting_id": calendar_meeting_id})
        return self._request(
            "POST",
            url,
            data=Path(bundle_path).read_bytes(),
            headers={
                "Content-Type": "application/zip",
                "Idempotency-Key": str(meeting_id),
            },
        )

    def workflow(self, workflow_id: str) -> dict:
        """``GET /workflows/{id}`` — the async workflow state (read-only in Phase 2)."""
        return self._request("GET", build_url(self.config, f"/workflows/{workflow_id}"))

    def _request(self, method: str, url: str, *, data=None, headers=None) -> dict:
        """One request with the deepgram.py retry discipline: ×3 with exponential
        backoff on 429/5xx and transport failures — URLError, plus the response-phase
        timeouts/resets/truncated reads that urllib does NOT wrap in URLError; 401/403
        (token) and 413 (bundle size) are final with an actionable message."""
        last_error = "unknown error"
        for attempt in range(RETRY_ATTEMPTS):
            if attempt:
                self._sleep(BACKOFF_BASE_S * 2 ** (attempt - 1))
            request = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Authorization": f"Bearer {self.config.token}",
                    **(headers or {}),
                },
                method=method,
            )
            try:
                response = self._opener(request, self.config.timeout_s)
                raw = response.read()
            except urllib.error.HTTPError as e:
                detail = _error_detail(e)
                if e.code in (401, 403):
                    raise BkError(
                        f"bk rejected the request (HTTP {e.code}: {detail or e.reason})"
                        " — check [bk].token / [bk].token_cmd (or MEETSCRIBE_BK_TOKEN)"
                    ) from e
                if e.code == 413:
                    raise BkError(
                        f"bk rejected the upload (HTTP 413: {detail or e.reason})"
                        " — the bundle exceeds bk's max_bundle_bytes"
                    ) from e
                if e.code == 429 or e.code >= 500:
                    last_error = f"HTTP {e.code}: {detail or e.reason}"
                    continue
                raise BkError(
                    f"bk request failed (HTTP {e.code}: {detail or e.reason})"
                ) from e
            except urllib.error.URLError as e:
                last_error = f"network error: {e.reason}"
                continue
            except (TimeoutError, OSError, http.client.HTTPException) as e:
                # urllib wraps only the request phase in URLError; a socket timeout or
                # reset while WAITING FOR or READING the response propagates raw
                # (TimeoutError/ConnectionResetError/IncompleteRead) and is just as
                # transient, so it retries too (same rationale as deepgram.py).
                last_error = f"network error: {str(e) or e.__class__.__name__}"
                continue
            try:
                return json.loads(raw.decode("utf-8", "replace"))
            except ValueError as e:
                snippet = raw[:120].decode("utf-8", "replace").strip()
                host = urllib.parse.urlsplit(url).netloc
                raise BkError(
                    f"bk returned a success response that is not JSON ({snippet!r})"
                    f" — is a proxy or captive portal intercepting {host}?"
                ) from e
        raise BkError(f"bk request failed after {RETRY_ATTEMPTS} attempts ({last_error})")


# ---- preflight ----------------------------------------------------------------------


def preflight(caps: dict, bundle_bytes: int) -> str | None:
    """Contract check before uploading; returns a human message (skip the upload,
    show this) or ``None`` when the upload may proceed. Never raises — the caller
    decides whether a failed preflight is a warning (auto-upload) or an error
    (`upload` subcommand)."""
    announced = caps.get("contract_version")
    supported = announced if isinstance(announced, list) else [announced]
    if CONTRACT_VERSION not in supported:
        return (
            f"bk speaks contract version {announced!r}, this meetscribe speaks"
            f" {CONTRACT_VERSION} — update meetscribe or bk"
        )
    format_versions = caps.get("mscribe_format_versions") or []
    if FORMAT_VERSION not in format_versions:
        return (
            f"bk cannot ingest bundle format_version {FORMAT_VERSION}"
            f" (it supports {format_versions}) — update bk or pin meetscribe"
        )
    max_bytes = caps.get("max_bundle_bytes")
    if max_bytes is not None and bundle_bytes > max_bytes:
        return (
            f"bundle is {bundle_bytes} bytes, over bk's max_bundle_bytes ({max_bytes})"
            " — the upload would be rejected"
        )
    return None


# ---- workflow reference persistence -------------------------------------------------
# `bk-workflow.json` lives NEXT TO the artifacts (never inside the bundle):
# {workflow_id, state_url, web_url, uploaded_at, state, checked_at}.


def write_workflow_ref(meeting_dir, ref: dict) -> Path:
    """Persist the workflow reference; simply overwrites (a re-upload after a terminal
    workflow legitimately starts a new one)."""
    path = Path(meeting_dir) / WORKFLOW_REF_NAME
    path.write_text(json.dumps(ref, indent=2) + "\n", encoding="utf-8")
    return path


def read_workflow_ref(meeting_dir) -> dict | None:
    """The persisted reference, or ``None`` when absent or corrupt — a broken ref
    file must degrade to "not uploaded", never crash ``status``."""
    path = Path(meeting_dir) / WORKFLOW_REF_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None
