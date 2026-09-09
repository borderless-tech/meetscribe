"""bk client: URL/auth, capabilities, preflight, upload, retries, workflow, ref file.

Everything here is offline — the transport seam (``opener``) and the ``sleep`` seam are
always faked; no test may touch the network. The fixtures under ``fixtures/bk/`` are
authored field-for-field from ``docs/meetscribe-bk-contract-v1.md`` (v1 AGREED) and are
meant to be drop-in replaceable by bk's canonical set; divergence then is a contract bug.
"""

import http.client
import io
import json
import urllib.error
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from meetscribe.bk import (
    WORKFLOW_REF_NAME,
    BkClient,
    BkConfig,
    BkError,
    build_url,
    preflight,
    read_workflow_ref,
    write_workflow_ref,
)
from meetscribe.output import FORMAT_VERSION

FIXTURES = Path(__file__).parent / "fixtures" / "bk"

CONFIG = BkConfig("https://bk.example.com", "tok-secret")


def _fixture(name):
    return json.loads((FIXTURES / name).read_text())


# ---- URL building + auth header -----------------------------------------------------


def test_build_url_prefixes_the_contract_base_path():
    assert (
        build_url(CONFIG, "/capabilities")
        == "https://bk.example.com/api/meetscribe/v1/capabilities"
    )


def test_build_url_strips_a_trailing_slash_on_base_url():
    config = BkConfig("https://bk.example.com/", "tok")
    assert (
        build_url(config, "/workflows/wf_01j9")
        == "https://bk.example.com/api/meetscribe/v1/workflows/wf_01j9"
    )


def test_config_default_timeout_carries_uploads_on_slow_links():
    assert CONFIG.timeout_s == 120.0


# ---- transport fakes (mirroring test_deepgram.py) -----------------------------------


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


def _http_error(code, body=None):
    if body is None:
        body = json.dumps(_fixture("error.json")).encode("utf-8")
    return urllib.error.HTTPError(
        "https://bk.example.com/api/meetscribe/v1/bundles", code, "err", {}, io.BytesIO(body)
    )


class SeqOpener:
    """Yields the scripted outcomes (exception → raised, response → returned) in order."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def __call__(self, request, timeout):
        self.calls.append((request, timeout))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _client(opener):
    sleeps = []
    return BkClient(CONFIG, opener=opener, sleep=sleeps.append), sleeps


# ---- capabilities -------------------------------------------------------------------


def test_capabilities_request_shape_and_parse():
    opener = SeqOpener([FakeResponse(_fixture("capabilities.json"))])
    client, _ = _client(opener)

    assert client.capabilities() == _fixture("capabilities.json")
    req, timeout = opener.calls[0]
    assert req.get_method() == "GET"
    assert req.full_url == "https://bk.example.com/api/meetscribe/v1/capabilities"
    assert req.get_header("Authorization") == "Bearer tok-secret"
    assert timeout == 120.0


def test_capabilities_fixture_matches_the_agreed_contract():
    # the agreed v1 semantics: bk is v2-only, and OUR format version must be in that list
    caps = _fixture("capabilities.json")
    assert caps["contract_version"] == 1
    assert caps["mscribe_format_versions"] == [2]
    assert FORMAT_VERSION in caps["mscribe_format_versions"]
    assert caps["max_bundle_bytes"] == 52428800


# ---- preflight verdicts -------------------------------------------------------------


def test_preflight_ok_returns_none():
    assert preflight(_fixture("capabilities.json"), 1024) is None


def test_preflight_rejects_unknown_contract_version():
    caps = {**_fixture("capabilities.json"), "contract_version": 2}
    message = preflight(caps, 1024)
    assert message is not None
    assert "contract" in message


def test_preflight_rejects_unlisted_format_version():
    caps = {**_fixture("capabilities.json"), "mscribe_format_versions": [1]}
    message = preflight(caps, 1024)
    assert message is not None
    assert f"format_version {FORMAT_VERSION}" in message
    assert "update bk or pin meetscribe" in message


def test_preflight_rejects_oversized_bundle():
    caps = _fixture("capabilities.json")
    message = preflight(caps, caps["max_bundle_bytes"] + 1)
    assert message is not None
    assert "max_bundle_bytes" in message


def test_preflight_bundle_at_the_cap_is_fine():
    caps = _fixture("capabilities.json")
    assert preflight(caps, caps["max_bundle_bytes"]) is None


def test_preflight_size_check_only_when_cap_present():
    caps = _fixture("capabilities.json")
    del caps["max_bundle_bytes"]
    assert preflight(caps, 10**12) is None


# ---- upload -------------------------------------------------------------------------


def _bundle(tmp_path):
    path = tmp_path / "meeting-20260909-100000.mscribe"
    path.write_bytes(b"PK\x03\x04fake-zip-bytes")
    return path


def test_upload_request_shape(tmp_path):
    opener = SeqOpener([FakeResponse(_fixture("upload_accepted.json"))])
    client, _ = _client(opener)

    out = client.upload_bundle(_bundle(tmp_path), meeting_id="20260909-100000")

    assert out == _fixture("upload_accepted.json")
    req, timeout = opener.calls[0]
    assert req.get_method() == "POST"
    assert req.full_url == "https://bk.example.com/api/meetscribe/v1/bundles"
    assert req.get_header("Authorization") == "Bearer tok-secret"
    assert req.get_header("Content-type") == "application/zip"
    assert req.get_header("Idempotency-key") == "20260909-100000"  # retry-safe upload
    assert req.data == b"PK\x03\x04fake-zip-bytes"
    assert timeout == 120.0


def test_upload_appends_meeting_id_query_only_when_given(tmp_path):
    opener = SeqOpener([FakeResponse(_fixture("upload_accepted.json"))])
    client, _ = _client(opener)

    client.upload_bundle(
        _bundle(tmp_path), meeting_id="20260909-100000", calendar_meeting_id="cal_evt_8f3a"
    )

    split = urlsplit(opener.calls[0][0].full_url)
    assert split.path == "/api/meetscribe/v1/bundles"
    assert parse_qs(split.query) == {"meeting_id": ["cal_evt_8f3a"]}


def test_upload_202_body_parses_to_the_workflow_reference(tmp_path):
    accepted = _fixture("upload_accepted.json")
    opener = SeqOpener([FakeResponse(accepted)])
    client, _ = _client(opener)

    out = client.upload_bundle(_bundle(tmp_path), meeting_id="m")

    assert out["workflow_id"] == "wf_01j9"
    assert out["state_url"] == "/api/meetscribe/v1/workflows/wf_01j9"
    assert out["web_url"] == "https://bk.example.com/workflows/wf_01j9"


# ---- retry / backoff ----------------------------------------------------------------


def test_upload_retries_on_429_then_succeeds(tmp_path):
    opener = SeqOpener([_http_error(429), FakeResponse(_fixture("upload_accepted.json"))])
    client, sleeps = _client(opener)

    out = client.upload_bundle(_bundle(tmp_path), meeting_id="m")

    assert out == _fixture("upload_accepted.json")
    assert len(opener.calls) == 2
    assert sleeps and all(s > 0 for s in sleeps)  # backed off before the retry


def test_upload_retries_on_urlerror_then_succeeds(tmp_path):
    opener = SeqOpener(
        [urllib.error.URLError("temporary name resolution failure"), FakeResponse({"ok": 1})]
    )
    client, _ = _client(opener)

    assert client.upload_bundle(_bundle(tmp_path), meeting_id="m") == {"ok": 1}
    assert len(opener.calls) == 2


def test_upload_retries_on_timeout_waiting_for_response(tmp_path):
    # urllib wraps only the request phase in URLError; a timeout while WAITING for the
    # response propagates raw and is just as transient (same handling as deepgram.py).
    opener = SeqOpener([TimeoutError("timed out"), FakeResponse({"ok": 1})])
    client, sleeps = _client(opener)

    assert client.upload_bundle(_bundle(tmp_path), meeting_id="m") == {"ok": 1}
    assert len(opener.calls) == 2
    assert sleeps and all(s > 0 for s in sleeps)


def test_upload_retries_on_connection_reset_while_reading(tmp_path):
    class BrokenReadResponse:
        def read(self):
            raise ConnectionResetError("reset by peer")

    opener = SeqOpener([BrokenReadResponse(), FakeResponse({"ok": 1})])
    client, _ = _client(opener)

    assert client.upload_bundle(_bundle(tmp_path), meeting_id="m") == {"ok": 1}
    assert len(opener.calls) == 2


def test_upload_retries_on_incomplete_read(tmp_path):
    # http.client.IncompleteRead is an HTTPException, not an OSError.
    opener = SeqOpener([http.client.IncompleteRead(b"partial"), FakeResponse({"ok": 1})])
    client, _ = _client(opener)

    assert client.upload_bundle(_bundle(tmp_path), meeting_id="m") == {"ok": 1}
    assert len(opener.calls) == 2


def test_upload_third_failure_raises_bk_error(tmp_path):
    opener = SeqOpener([_http_error(500), _http_error(503), urllib.error.URLError("down")])
    client, _ = _client(opener)

    with pytest.raises(BkError, match="3 attempts"):
        client.upload_bundle(_bundle(tmp_path), meeting_id="m")
    assert len(opener.calls) == 3


def test_upload_401_names_both_token_knobs_and_does_not_retry(tmp_path):
    opener = SeqOpener([_http_error(401)])
    client, sleeps = _client(opener)

    with pytest.raises(BkError, match="MEETSCRIBE_BK_TOKEN") as excinfo:
        client.upload_bundle(_bundle(tmp_path), meeting_id="m")
    assert "[bk].token" in str(excinfo.value)
    assert len(opener.calls) == 1
    assert sleeps == []


def test_upload_413_names_max_bundle_bytes_and_does_not_retry(tmp_path):
    opener = SeqOpener([_http_error(413)])
    client, sleeps = _client(opener)

    with pytest.raises(BkError, match="max_bundle_bytes"):
        client.upload_bundle(_bundle(tmp_path), meeting_id="m")
    assert len(opener.calls) == 1
    assert sleeps == []


def test_upload_non_retryable_status_surfaces_the_contract_error_message(tmp_path):
    # error.json is the contract's standard error shape; its .error.message must
    # surface in the raised BkError, not the raw body.
    opener = SeqOpener([_http_error(422)])
    client, _ = _client(opener)

    with pytest.raises(BkError, match="bundle format_version 3 is not supported"):
        client.upload_bundle(_bundle(tmp_path), meeting_id="m")
    assert len(opener.calls) == 1


def test_non_json_200_body_is_a_bk_error():
    # A captive portal / corporate proxy can return 200 with an HTML body; that must
    # be a clear BkError, not a JSONDecodeError traceback — and it is final.
    class HtmlResponse:
        def read(self):
            return b"<html>corporate proxy login</html>"

    opener = SeqOpener([HtmlResponse()])
    client, _ = _client(opener)

    with pytest.raises(BkError, match="not JSON"):
        client.capabilities()
    assert len(opener.calls) == 1


# ---- workflow GET -------------------------------------------------------------------


def test_workflow_request_shape_and_parse():
    opener = SeqOpener([FakeResponse(_fixture("workflow_processing.json"))])
    client, _ = _client(opener)

    out = client.workflow("wf_01j9")

    assert out == _fixture("workflow_processing.json")
    req, _ = opener.calls[0]
    assert req.get_method() == "GET"
    assert req.full_url == "https://bk.example.com/api/meetscribe/v1/workflows/wf_01j9"
    assert req.get_header("Authorization") == "Bearer tok-secret"


@pytest.mark.parametrize(
    "name, state",
    [
        ("workflow_processing.json", "processing"),
        ("workflow_awaiting_review.json", "awaiting_review"),
        ("workflow_done.json", "done"),
        ("workflow_failed.json", "failed"),
    ],
)
def test_workflow_fixtures_carry_web_url_in_every_state(name, state):
    # forward-compat rule 1: web_url is ALWAYS present — unknown states fall back to it
    wf = _fixture(name)
    assert wf["state"] == state
    assert wf["web_url"] == "https://bk.example.com/workflows/wf_01j9"
    assert wf["contract_version"] == 1


def test_awaiting_review_fixture_has_exactly_one_speaker_annotation_task():
    # the agreed single-gate semantics: at most ONE task, sequential gates only
    wf = _fixture("workflow_awaiting_review.json")
    assert len(wf["tasks"]) == 1
    task = wf["tasks"][0]
    assert task["type"] == "speaker_annotation"
    assert set(task) == {"task_id", "type", "revision", "submit_url", "web_url", "payload"}
    assert [s["label"] for s in task["payload"]["speakers"]] == ["me", "spk_0", "spk_1"]
    assert {p["person_id"] for p in task["payload"]["roster"]} == {"p_123", "p_456"}

    # Pin the payload interior too: Phase 4 task rendering consumes exactly these
    # fields, and a dropped-in canonical bk fixture that renames/drops one must go
    # red HERE (the fixtures' whole purpose is divergence detection).
    for speaker in task["payload"]["speakers"]:
        assert set(speaker) == {"label", "suggestions"}
        for s in speaker["suggestions"]:
            assert set(s) == {"person_id", "name", "confidence", "source"}
            assert s["source"] in {"owner", "voice_match", "roster"}  # the contract enum
    # the "me" owner suggestion and the contract's null-confidence roster suggestion
    me, spk_0, _ = task["payload"]["speakers"]
    assert [(s["source"], s["confidence"]) for s in me["suggestions"]] == [("owner", "high")]
    assert [(s["source"], s["confidence"]) for s in spk_0["suggestions"]] == [
        ("voice_match", "high"),
        ("roster", None),
    ]
    for person in task["payload"]["roster"]:
        assert set(person) == {"person_id", "email", "name"}
    assert {p["email"] for p in task["payload"]["roster"]} == {
        "anna@example.com",
        "ben@example.com",
    }


def test_failed_fixture_carries_the_standard_error_shape():
    wf = _fixture("workflow_failed.json")
    assert wf["tasks"] == []
    assert set(wf["error"]) == {"code", "message"}
    assert wf["error"]["code"] == "gate_timeout"


# ---- workflow reference file --------------------------------------------------------


def _ref():
    return {
        "workflow_id": "wf_01j9",
        "state_url": "/api/meetscribe/v1/workflows/wf_01j9",
        "web_url": "https://bk.example.com/workflows/wf_01j9",
        "uploaded_at": "2026-09-09T10:00:00+02:00",
        "state": "processing",
        "checked_at": "2026-09-09T10:00:00+02:00",
    }


def test_workflow_ref_roundtrip(tmp_path):
    path = write_workflow_ref(tmp_path, _ref())

    assert path == tmp_path / WORKFLOW_REF_NAME
    assert WORKFLOW_REF_NAME == "bk-workflow.json"
    assert read_workflow_ref(tmp_path) == _ref()
    # the file on disk is plain JSON (other tools may read it)
    assert json.loads(path.read_text()) == _ref()


def test_read_workflow_ref_absent_returns_none(tmp_path):
    assert read_workflow_ref(tmp_path) is None


def test_read_workflow_ref_tolerates_corrupt_json(tmp_path):
    # a corrupt ref (killed mid-write, disk hiccup) must not crash `status`
    (tmp_path / WORKFLOW_REF_NAME).write_text("{not json", encoding="utf-8")
    assert read_workflow_ref(tmp_path) is None


def test_read_workflow_ref_tolerates_non_dict_json(tmp_path):
    (tmp_path / WORKFLOW_REF_NAME).write_text("[1, 2]", encoding="utf-8")
    assert read_workflow_ref(tmp_path) is None


def test_write_workflow_ref_overwrites_existing(tmp_path):
    # re-upload after done/failed starts a NEW workflow — the ref is simply replaced
    write_workflow_ref(tmp_path, _ref())
    write_workflow_ref(tmp_path, {**_ref(), "workflow_id": "wf_02aa", "state": "done"})

    ref = read_workflow_ref(tmp_path)
    assert ref["workflow_id"] == "wf_02aa"
    assert ref["state"] == "done"
