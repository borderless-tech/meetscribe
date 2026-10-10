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
    CALENDAR_REF_NAME,
    WORKFLOW_REF_NAME,
    BkAuthError,
    BkClient,
    BkConfig,
    BkError,
    BkInvalidSubmission,
    BkRevisionConflict,
    build_url,
    preflight,
    read_calendar_ref,
    read_workflow_ref,
    write_calendar_ref,
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
    opener = SeqOpener([FakeResponse(_fixture("bundle-accepted.json"))])
    client, _ = _client(opener)

    out = client.upload_bundle(_bundle(tmp_path), meeting_id="20260909-100000")

    assert out == _fixture("bundle-accepted.json")
    req, timeout = opener.calls[0]
    assert req.get_method() == "POST"
    assert req.full_url == "https://bk.example.com/api/meetscribe/v1/bundles"
    assert req.get_header("Authorization") == "Bearer tok-secret"
    assert req.get_header("Content-type") == "application/zip"
    assert req.get_header("Idempotency-key") == "20260909-100000"  # retry-safe upload
    assert req.data == b"PK\x03\x04fake-zip-bytes"
    assert timeout == 120.0


def test_upload_appends_meeting_id_query_only_when_given(tmp_path):
    opener = SeqOpener([FakeResponse(_fixture("bundle-accepted.json"))])
    client, _ = _client(opener)

    client.upload_bundle(
        _bundle(tmp_path), meeting_id="20260909-100000", calendar_meeting_id="cal_evt_8f3a"
    )

    split = urlsplit(opener.calls[0][0].full_url)
    assert split.path == "/api/meetscribe/v1/bundles"
    assert parse_qs(split.query) == {"meeting_id": ["cal_evt_8f3a"]}


def test_upload_202_body_parses_to_the_workflow_reference(tmp_path):
    accepted = _fixture("bundle-accepted.json")
    opener = SeqOpener([FakeResponse(accepted)])
    client, _ = _client(opener)

    out = client.upload_bundle(_bundle(tmp_path), meeting_id="m")

    # ids are opaque (contract): assert self-consistency, not example values
    wf_id = out["workflow_id"]
    assert wf_id and isinstance(wf_id, str)
    assert out["state_url"] == f"/api/meetscribe/v1/workflows/{wf_id}"
    assert out["web_url"].startswith("https://")


# ---- retry / backoff ----------------------------------------------------------------


def test_upload_retries_on_429_then_succeeds(tmp_path):
    opener = SeqOpener([_http_error(429), FakeResponse(_fixture("bundle-accepted.json"))])
    client, sleeps = _client(opener)

    out = client.upload_bundle(_bundle(tmp_path), meeting_id="m")

    assert out == _fixture("bundle-accepted.json")
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
    # Typed as BkAuthError so callers can treat a bad token as FINAL (contract:
    # "401 → actionable client error, never a retry loop") — a generic BkError
    # handler would conflate it with a transient network blip and retry.
    opener = SeqOpener([_http_error(401)])
    client, sleeps = _client(opener)

    with pytest.raises(BkAuthError, match="MEETSCRIBE_BK_TOKEN") as excinfo:
        client.upload_bundle(_bundle(tmp_path), meeting_id="m")
    assert "[bk].token" in str(excinfo.value)
    assert len(opener.calls) == 1
    assert sleeps == []


def test_submit_task_401_is_a_typed_final_auth_error():
    opener = SeqOpener([_http_error(401)])
    client, sleeps = _client(opener)

    with pytest.raises(BkAuthError, match=r"\[bk\]\.token"):
        client.submit_task(_submit_url(), _fixture("task-submit-speaker-annotation.json"))
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

    with pytest.raises(BkError, match="unknown workflow"):
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
    opener = SeqOpener([FakeResponse(_fixture("workflow-processing.json"))])
    client, _ = _client(opener)

    out = client.workflow("wf_01j9")

    assert out == _fixture("workflow-processing.json")
    req, _ = opener.calls[0]
    assert req.get_method() == "GET"
    assert req.full_url == "https://bk.example.com/api/meetscribe/v1/workflows/wf_01j9"
    assert req.get_header("Authorization") == "Bearer tok-secret"


@pytest.mark.parametrize(
    "name, state",
    [
        ("workflow-processing.json", "processing"),
        ("workflow-awaiting-review-speaker-annotation.json", "awaiting_review"),
        ("workflow-done.json", "done"),
        ("workflow-failed.json", "failed"),
    ],
)
def test_workflow_fixtures_carry_web_url_in_every_state(name, state):
    # forward-compat rule 1: web_url is ALWAYS present — unknown states fall back to it.
    # The URL itself is opaque (bk serves /transcripts/... today) — shape only.
    wf = _fixture(name)
    assert wf["state"] == state
    assert wf["web_url"].startswith("https://")
    assert wf["contract_version"] == 1


def test_awaiting_review_fixture_has_exactly_one_speaker_annotation_task():
    # the agreed single-gate semantics: at most ONE task, sequential gates only
    wf = _fixture("workflow-awaiting-review-speaker-annotation.json")
    assert len(wf["tasks"]) == 1
    task = wf["tasks"][0]
    assert task["type"] == "speaker_annotation"
    # REQUIRED keys must be present; extra keys are fine — the contract's additive
    # rule obliges clients to ignore unknown fields, so these are subset asserts,
    # never equality (bk already ships additive `preselected`/`new_voice_model`).
    assert {"task_id", "type", "revision", "submit_url", "web_url", "payload"} <= set(task)
    assert task["submit_url"].endswith(f"/tasks/{task['task_id']}/submit")

    labels = [s["label"] for s in task["payload"]["speakers"]]
    assert len(labels) == len(set(labels))  # unique
    assert "me" in labels
    assert all(lbl == "me" or lbl.startswith("spk_") for lbl in labels)

    for speaker in task["payload"]["speakers"]:
        assert {"label", "suggestions"} <= set(speaker)
        for s in speaker["suggestions"]:
            assert {"person_id", "name", "confidence", "source"} <= set(s)
            assert s["source"] in {"owner", "voice_match", "roster"}  # the contract enum
            if "preselected" in s:  # additive (bk-implemented nice-to-have)
                assert isinstance(s["preselected"], bool)
    # "me" carries an owner-source suggestion (the uploading user)
    me = next(s for s in task["payload"]["speakers"] if s["label"] == "me")
    assert any(s["source"] == "owner" for s in me["suggestions"])

    roster_ids = set()
    for person in task["payload"]["roster"]:
        assert {"person_id", "email", "name"} <= set(person)
        assert person["person_id"]  # opaque but non-empty
        roster_ids.add(person["person_id"])
    assert len(roster_ids) == len(task["payload"]["roster"])  # unique
    # every roster-sourced suggestion refers back into the roster (cross-consistency)
    for speaker in task["payload"]["speakers"]:
        for s in speaker["suggestions"]:
            if s["source"] == "roster":
                assert s["person_id"] in roster_ids


def test_failed_fixture_carries_the_standard_error_shape():
    # code values are bk's (opaque examples) — the SHAPE is the contract
    wf = _fixture("workflow-failed.json")
    assert wf["tasks"] == []
    assert set(wf["error"]) == {"code", "message"}
    assert wf["error"]["code"] and isinstance(wf["error"]["code"], str)
    assert wf["error"]["message"] and isinstance(wf["error"]["message"], str)


# ---- task submit --------------------------------------------------------------------
# submit_url in the workflow fixtures is a HOST-RELATIVE path that already carries the
# contract prefix (contract: the client never builds task URLs itself), so the full URL
# is base_url + submit_url — not build_url().


def _submit_url():
    wf = _fixture("workflow-awaiting-review-speaker-annotation.json")
    return wf["tasks"][0]["submit_url"]


def test_submit_task_posts_the_body_and_returns_the_fresh_workflow_state():
    opener = SeqOpener([FakeResponse(_fixture("workflow-processing.json"))])
    client, _ = _client(opener)
    body = _fixture("task-submit-speaker-annotation.json")

    out = client.submit_task(_submit_url(), body)

    # 200 returns the FULL fresh workflow state — the engine feeds it straight back
    assert out == _fixture("workflow-processing.json")
    req, timeout = opener.calls[0]
    assert req.get_method() == "POST"
    assert req.full_url == "https://bk.example.com" + _submit_url()
    assert req.get_header("Authorization") == "Bearer tok-secret"
    assert req.get_header("Content-type") == "application/json"
    # the wire body, parsed back, IS the contract fixture (revision echo included)
    assert json.loads(req.data.decode("utf-8")) == body
    assert timeout == 120.0


def test_submit_task_409_is_a_typed_final_revision_conflict():
    # 409 means the task moved under us — a HUMAN must re-answer, so a transport
    # retry would be wrong: exactly one call, no backoff, typed exception.
    opener = SeqOpener([_http_error(409)])
    client, sleeps = _client(opener)

    with pytest.raises(BkRevisionConflict):
        client.submit_task(_submit_url(), _fixture("task-submit-speaker-annotation.json"))
    assert len(opener.calls) == 1
    assert sleeps == []


def test_submit_task_422_is_a_typed_final_invalid_submission_with_the_server_message():
    opener = SeqOpener([_http_error(422)])  # default body = error.json
    client, sleeps = _client(opener)

    with pytest.raises(BkInvalidSubmission, match="unknown workflow"):
        client.submit_task(_submit_url(), _fixture("task-submit-speaker-annotation.json"))
    assert len(opener.calls) == 1
    assert sleeps == []


def test_submit_task_typed_errors_are_bk_errors():
    # generic `except BkError` handlers (warn + retry hint) must still catch all three
    assert issubclass(BkRevisionConflict, BkError)
    assert issubclass(BkInvalidSubmission, BkError)
    assert issubclass(BkAuthError, BkError)


def test_submit_task_retries_on_500_then_succeeds():
    # 429/5xx keep the existing retry discipline — only 409/422 are final
    opener = SeqOpener([_http_error(500), FakeResponse(_fixture("workflow-processing.json"))])
    client, sleeps = _client(opener)

    out = client.submit_task(_submit_url(), _fixture("task-submit-speaker-annotation.json"))

    assert out == _fixture("workflow-processing.json")
    assert len(opener.calls) == 2
    assert sleeps and all(s > 0 for s in sleeps)


# ---- meetings around (calendar lookup) ----------------------------------------------
# Endpoint 2 of the contract: ONE opportunistic GET at record start. Single attempt and
# a short timeout — a slow bk must cost at most the timeout, once; the retry discipline
# exists for uploads, not here.


def test_meetings_around_request_shape_and_parse():
    opener = SeqOpener([FakeResponse(_fixture("meetings-around.json"))])
    client, _ = _client(opener)

    out = client.meetings_around("2026-09-08T09:00:25+02:00")

    assert out == _fixture("meetings-around.json")
    req, timeout = opener.calls[0]
    assert req.get_method() == "GET"
    split = urlsplit(req.full_url)
    assert split.scheme == "https" and split.netloc == "bk.example.com"
    assert split.path == "/api/meetscribe/v1/meetings"
    # the iso timestamp must survive urlencoding (`+02:00` would otherwise decode to a space)
    assert parse_qs(split.query) == {"around": ["2026-09-08T09:00:25+02:00"]}
    assert req.get_header("Authorization") == "Bearer tok-secret"


def test_meetings_around_empty_calendar_parses_to_empty_list():
    opener = SeqOpener([FakeResponse(_fixture("meetings-around-empty.json"))])
    client, _ = _client(opener)

    assert client.meetings_around("2026-09-08T09:00:25+02:00") == {"meetings": []}


def test_meetings_around_is_single_attempt():
    # contrast with the upload calls (3 attempts): a 500 here fails after exactly ONE
    # opener call and no backoff sleep — recording must never wait on bk.
    opener = SeqOpener([_http_error(500)])
    client, sleeps = _client(opener)

    with pytest.raises(BkError, match="1 attempt"):
        client.meetings_around("2026-09-08T09:00:25+02:00")
    assert len(opener.calls) == 1
    assert sleeps == []


def test_meetings_around_network_error_is_single_attempt_too():
    opener = SeqOpener([urllib.error.URLError("no route to host")])
    client, sleeps = _client(opener)

    with pytest.raises(BkError):
        client.meetings_around("2026-09-08T09:00:25+02:00")
    assert len(opener.calls) == 1
    assert sleeps == []


def test_meetings_around_defaults_to_a_3s_timeout():
    opener = SeqOpener([FakeResponse(_fixture("meetings-around-empty.json"))])
    client, _ = _client(opener)

    client.meetings_around("2026-09-08T09:00:25+02:00")

    assert opener.calls[0][1] == 3.0


def test_meetings_around_timeout_is_caller_overridable():
    opener = SeqOpener([FakeResponse(_fixture("meetings-around-empty.json"))])
    client, _ = _client(opener)

    client.meetings_around("2026-09-08T09:00:25+02:00", timeout_s=1.5)

    assert opener.calls[0][1] == 1.5


def test_meetings_around_401_is_a_typed_auth_error():
    # callers of the opportunistic path swallow it like any BkError, but the type
    # must stay consistent with the rest of the client
    opener = SeqOpener([_http_error(401)])
    client, _ = _client(opener)

    with pytest.raises(BkAuthError):
        client.meetings_around("2026-09-08T09:00:25+02:00")
    assert len(opener.calls) == 1


def test_other_calls_keep_the_config_timeout_and_retries(tmp_path):
    # the new _request knobs must not leak into the existing calls
    opener = SeqOpener([_http_error(500), FakeResponse(_fixture("bundle-accepted.json"))])
    client, _ = _client(opener)

    client.upload_bundle(_bundle(tmp_path), meeting_id="m")

    assert len(opener.calls) == 2
    assert [timeout for _, timeout in opener.calls] == [120.0, 120.0]


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


# ---- calendar reference file ---------------------------------------------------------
# `bk-calendar.json` is the cached record-start candidate list (+ later the picked id);
# same tolerant-reader rules as bk-workflow.json, and never a bundle member.


def _calendar_ref():
    return {
        "queried_at": "2026-09-08T09:00:25+02:00",
        "around": "2026-09-08T09:00:25+02:00",
        "meetings": _fixture("meetings-around.json")["meetings"],
    }


def test_calendar_ref_roundtrip(tmp_path):
    path = write_calendar_ref(tmp_path, _calendar_ref())

    assert path == tmp_path / CALENDAR_REF_NAME
    assert CALENDAR_REF_NAME == "bk-calendar.json"
    assert read_calendar_ref(tmp_path) == _calendar_ref()
    # the file on disk is plain JSON (other tools may read it)
    assert json.loads(path.read_text()) == _calendar_ref()


def test_read_calendar_ref_absent_returns_none(tmp_path):
    assert read_calendar_ref(tmp_path) is None


def test_read_calendar_ref_tolerates_corrupt_json(tmp_path):
    # a corrupt ref must degrade to "no calendar info", never crash the flow
    (tmp_path / CALENDAR_REF_NAME).write_text("{not json", encoding="utf-8")
    assert read_calendar_ref(tmp_path) is None


def test_read_calendar_ref_tolerates_non_dict_json(tmp_path):
    (tmp_path / CALENDAR_REF_NAME).write_text("[1, 2]", encoding="utf-8")
    assert read_calendar_ref(tmp_path) is None


def test_write_calendar_ref_overwrites_existing(tmp_path):
    # the post-recording pick rewrites the ref with picked_meeting_id added
    write_calendar_ref(tmp_path, _calendar_ref())
    write_calendar_ref(tmp_path, {**_calendar_ref(), "picked_meeting_id": "cal_evt_8f3a"})

    assert read_calendar_ref(tmp_path)["picked_meeting_id"] == "cal_evt_8f3a"
