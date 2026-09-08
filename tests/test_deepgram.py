"""Deepgram remote backend: URL/query building, keyterm caps, response mapping, retries.

Everything here is offline — the transport seam (``opener``) and the client seam
(``DeepgramBackend(client=...)``) are always faked; no test may touch the network.
"""

import http.client
import io
import json
import urllib.error
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest

from meetscribe.backends import BackendResult, TrackInput
from meetscribe.deepgram import (
    DeepgramBackend,
    DeepgramClient,
    DeepgramConfig,
    DeepgramError,
    build_url,
    keyterms_from_glossary,
)
from meetscribe.progress import NullReporter

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name):
    return json.loads((FIXTURES / name).read_text())


# ---- URL / query building -----------------------------------------------------------


def _qs(url):
    split = urlsplit(url)
    assert split.scheme == "https"
    assert split.netloc == "api.deepgram.com"
    assert split.path == "/v1/listen"
    return parse_qs(split.query, keep_blank_values=True)


def test_build_url_base_query():
    q = _qs(build_url(DeepgramConfig("key"), diarize=False, keyterms=[]))
    assert q["model"] == ["nova-3"]
    assert q["language"] == ["de"]
    assert q["smart_format"] == ["true"]
    assert q["utterances"] == ["true"]
    assert "diarize" not in q
    assert "keyterm" not in q


def test_build_url_diarize_for_system_track():
    q = _qs(build_url(DeepgramConfig("key"), diarize=True, keyterms=[]))
    assert q["diarize"] == ["true"]


def test_build_url_repeats_keyterm_in_order():
    q = _qs(
        build_url(
            DeepgramConfig("key"),
            diarize=False,
            keyterms=["Borderless", "pgvector", "sherpa onnx"],
        )
    )
    assert q["keyterm"] == ["Borderless", "pgvector", "sherpa onnx"]


def test_build_url_no_keyterm_for_non_nova3_models():
    # keyterm prompting is a nova-3 feature; other models must not receive the param
    q = _qs(
        build_url(DeepgramConfig("key", model="nova-2"), diarize=False, keyterms=["X"])
    )
    assert q["model"] == ["nova-2"]
    assert "keyterm" not in q


def test_build_url_language_override():
    q = _qs(build_url(DeepgramConfig("key", language="multi"), diarize=False, keyterms=[]))
    assert q["language"] == ["multi"]


# ---- keyterms_from_glossary ---------------------------------------------------------


def test_keyterms_dedupe_case_insensitively_keeping_first_and_order():
    assert keyterms_from_glossary(
        ["Borderless", "pgvector", "borderless", "Meetscribe"]
    ) == ["Borderless", "pgvector", "Meetscribe"]


def test_keyterms_strip_whitespace_and_drop_empty():
    assert keyterms_from_glossary(["  Borderless ", "", "   "]) == ["Borderless"]


def test_keyterms_cap_at_100_terms():
    terms = [f"term{i}" for i in range(150)]
    assert keyterms_from_glossary(terms) == terms[:100]


def test_keyterms_cap_at_500_tokens_total():
    # 7 whitespace-tokens each: 71 * 7 = 497 fits; the 72nd (504) and everything after
    # is equally sized, so all overflow is dropped.
    terms = [" ".join(f"w{i}_{j}" for j in range(7)) for i in range(80)]
    out = keyterms_from_glossary(terms)
    assert out == terms[:71]
    assert sum(len(t.split()) for t in out) <= 500


def test_keyterms_token_overflow_drops_only_the_overflowing_term():
    big = " ".join(f"w{i}" for i in range(500))  # 500 tokens: cannot fit next to anything
    # input order is kept; the oversized middle term is dropped, later terms still fit
    assert keyterms_from_glossary(["Borderless", big, "pgvector"]) == [
        "Borderless",
        "pgvector",
    ]


# ---- response → Utterance/DiarSegment mapping (fixtures) ----------------------------


class FakeClient:
    """Returns canned parsed-response dicts keyed by the ``diarize`` flag."""

    def __init__(self, responses):
        self.responses = dict(responses)  # {diarize_flag: response_dict}
        self.calls = []

    def transcribe_file(self, wav_path, *, diarize, keyterms):
        self.calls.append(
            {"wav_path": wav_path, "diarize": diarize, "keyterms": list(keyterms)}
        )
        return self.responses[diarize]


def _tracks():
    samples = np.zeros(16000, dtype=np.float32)
    return TrackInput("mic.wav", samples), TrackInput("system.wav", samples)


def _backend(client):
    return DeepgramBackend(DeepgramConfig("key"), client=client)


def test_backend_name():
    assert _backend(FakeClient({})).name == "deepgram"


def test_backend_maps_mic_utterances_to_me():
    mic, system = _tracks()
    client = FakeClient({False: _fixture("deepgram_mic.json")})
    res = _backend(client).transcribe(mic, None, [], NullReporter())

    assert isinstance(res, BackendResult)
    assert [(u.speaker, u.track, u.text) for u in res.mic_utts] == [
        ("me", "mic", "Okay, I will check the logs.")
    ]
    assert (res.mic_utts[0].start, res.mic_utts[0].end) == (0.1, 2.1)
    # punctuated_word preferred; "will" has none in the fixture → falls back to word
    assert [w.w for w in res.mic_utts[0].words] == [
        "Okay,", "I", "will", "check", "the", "logs.",
    ]
    assert res.mic_utts[0].words[0].start == 0.1
    assert res.mic_utts[0].words[-1].end == 2.1
    assert res.system_utts == []
    assert res.system_diar == []


def test_backend_maps_system_utterances_to_spk_n():
    mic, system = _tracks()
    client = FakeClient({True: _fixture("deepgram_system.json")})
    res = _backend(client).transcribe(None, system, [], NullReporter())

    # one Utterance per Deepgram utterance — the same-speaker pair with a <2 s gap is
    # NOT merged here (coalescing is the shared tail's job in pipeline.process)
    assert [(u.speaker, u.track, u.text) for u in res.system_utts] == [
        ("spk_0", "system", "Guten Morgen zusammen."),
        ("spk_0", "system", "Fangen wir mit dem Update an."),
        ("spk_1", "system", "Das Deployment ist durch."),
    ]
    assert [w.w for w in res.system_utts[2].words] == [
        "Das", "Deployment", "ist", "durch.",
    ]
    # one DiarSegment per utterance → feeds the local embedding pass
    assert [(s.start, s.end, s.speaker) for s in res.system_diar] == [
        (0.08, 1.9, "spk_0"),
        (2.4, 4.5, "spk_0"),
        (5.1, 7.2, "spk_1"),
    ]
    assert res.mic_utts == []


def test_backend_requests_diarize_only_for_system_track():
    mic, system = _tracks()
    client = FakeClient(
        {False: _fixture("deepgram_mic.json"), True: _fixture("deepgram_system.json")}
    )
    _backend(client).transcribe(mic, system, ["Borderless", "borderless"], NullReporter())

    assert [(c["wav_path"], c["diarize"]) for c in client.calls] == [
        ("mic.wav", False),
        ("system.wav", True),
    ]
    # glossary goes through keyterms_from_glossary (deduped) on both requests
    assert [c["keyterms"] for c in client.calls] == [["Borderless"], ["Borderless"]]


def test_backend_missing_utterances_is_a_clear_error():
    mic, _ = _tracks()
    client = FakeClient(
        {False: {"metadata": {}, "results": {"channels": [{"alternatives": []}]}}}
    )
    with pytest.raises(DeepgramError, match="utterances"):
        _backend(client).transcribe(mic, None, [], NullReporter())


def test_backend_empty_channels_is_a_clear_error():
    mic, _ = _tracks()
    client = FakeClient(
        {False: {"metadata": {}, "results": {"channels": [], "utterances": []}}}
    )
    with pytest.raises(DeepgramError, match="channel"):
        _backend(client).transcribe(mic, None, [], NullReporter())


# ---- retry / backoff (fake opener; never touches the network) -----------------------


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


def _http_error(code, body=b'{"err_code": "X", "err_msg": "boom"}'):
    return urllib.error.HTTPError(
        "https://api.deepgram.com/v1/listen", code, "err", {}, io.BytesIO(body)
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


def _client(tmp_path, opener):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFFfake-wav-bytes")
    sleeps = []
    client = DeepgramClient(DeepgramConfig("secret"), opener=opener, sleep=sleeps.append)
    return client, str(wav), sleeps


def test_client_request_shape(tmp_path):
    opener = SeqOpener([FakeResponse({"ok": True})])
    client, wav, _ = _client(tmp_path, opener)

    out = client.transcribe_file(wav, diarize=True, keyterms=["Borderless"])

    assert out == {"ok": True}
    req, timeout = opener.calls[0]
    assert req.get_method() == "POST"
    assert req.get_header("Authorization") == "Token secret"
    assert req.get_header("Content-type") == "audio/wav"
    assert req.data == b"RIFFfake-wav-bytes"
    assert timeout == 300.0
    q = _qs(req.full_url)
    assert q["diarize"] == ["true"]
    assert q["keyterm"] == ["Borderless"]


def test_client_retries_on_429_then_succeeds(tmp_path):
    opener = SeqOpener([_http_error(429), FakeResponse({"ok": True})])
    client, wav, sleeps = _client(tmp_path, opener)

    assert client.transcribe_file(wav, diarize=False, keyterms=[]) == {"ok": True}
    assert len(opener.calls) == 2
    assert sleeps and all(s > 0 for s in sleeps)  # backed off before the retry


def test_client_retries_on_urlerror_then_succeeds(tmp_path):
    opener = SeqOpener(
        [urllib.error.URLError("temporary name resolution failure"), FakeResponse({"ok": 1})]
    )
    client, wav, _ = _client(tmp_path, opener)

    assert client.transcribe_file(wav, diarize=False, keyterms=[]) == {"ok": 1}
    assert len(opener.calls) == 2


def test_client_401_names_the_env_var_and_does_not_retry(tmp_path):
    opener = SeqOpener([_http_error(401)])
    client, wav, sleeps = _client(tmp_path, opener)

    with pytest.raises(DeepgramError, match="DEEPGRAM_API_KEY"):
        client.transcribe_file(wav, diarize=False, keyterms=[])
    assert len(opener.calls) == 1
    assert sleeps == []


def test_client_third_failure_raises(tmp_path):
    opener = SeqOpener([_http_error(500), _http_error(503), urllib.error.URLError("down")])
    client, wav, _ = _client(tmp_path, opener)

    with pytest.raises(DeepgramError):
        client.transcribe_file(wav, diarize=False, keyterms=[])
    assert len(opener.calls) == 3


class BrokenReadResponse:
    """A 200 response whose body read fails mid-stream (socket reset/timeout)."""

    def __init__(self, exc):
        self._exc = exc

    def read(self):
        raise self._exc


def test_client_retries_on_timeout_waiting_for_response(tmp_path):
    # CPython wraps only h.request() failures in URLError; a socket timeout while
    # WAITING for the response — the long phase here, Deepgram processes the whole
    # upload before answering — is a raw TimeoutError and must retry too.
    opener = SeqOpener([TimeoutError("timed out"), FakeResponse({"ok": True})])
    client, wav, sleeps = _client(tmp_path, opener)

    assert client.transcribe_file(wav, diarize=False, keyterms=[]) == {"ok": True}
    assert len(opener.calls) == 2
    assert sleeps and all(s > 0 for s in sleeps)  # backed off before the retry


def test_client_retries_on_connection_reset_while_reading(tmp_path):
    # response.read() happens outside urllib's URLError wrap entirely.
    opener = SeqOpener(
        [BrokenReadResponse(ConnectionResetError("reset by peer")), FakeResponse({"ok": 1})]
    )
    client, wav, _ = _client(tmp_path, opener)

    assert client.transcribe_file(wav, diarize=False, keyterms=[]) == {"ok": 1}
    assert len(opener.calls) == 2


def test_client_retries_on_incomplete_read(tmp_path):
    # http.client.IncompleteRead is an HTTPException, not an OSError.
    opener = SeqOpener([http.client.IncompleteRead(b"partial"), FakeResponse({"ok": 1})])
    client, wav, _ = _client(tmp_path, opener)

    assert client.transcribe_file(wav, diarize=False, keyterms=[]) == {"ok": 1}
    assert len(opener.calls) == 2


def test_client_third_transport_failure_raises_deepgram_error(tmp_path):
    # Exhausted retries on response-phase failures end in DeepgramError, not a raw
    # TimeoutError/ConnectionResetError traceback.
    opener = SeqOpener(
        [TimeoutError("timed out"), ConnectionResetError("reset"), TimeoutError("timed out")]
    )
    client, wav, _ = _client(tmp_path, opener)

    with pytest.raises(DeepgramError, match="3 attempts"):
        client.transcribe_file(wav, diarize=False, keyterms=[])
    assert len(opener.calls) == 3


def test_client_non_json_200_body_is_a_deepgram_error(tmp_path):
    # A captive portal / corporate proxy can return 200 with an HTML body; that must
    # be a clear DeepgramError, not a JSONDecodeError traceback.
    class HtmlResponse:
        def read(self):
            return b"<html>corporate proxy login</html>"

    opener = SeqOpener([HtmlResponse()])
    client, wav, _ = _client(tmp_path, opener)

    with pytest.raises(DeepgramError, match="not JSON"):
        client.transcribe_file(wav, diarize=False, keyterms=[])
    assert len(opener.calls) == 1  # a bad body is final — retrying won't fix a proxy


def test_client_non_retryable_status_surfaces_server_message(tmp_path):
    opener = SeqOpener(
        [_http_error(400, b'{"err_code": "Bad Request", "err_msg": "bad container"}')]
    )
    client, wav, _ = _client(tmp_path, opener)

    with pytest.raises(DeepgramError, match="bad container"):
        client.transcribe_file(wav, diarize=False, keyterms=[])
    assert len(opener.calls) == 1


# ---- models_meta --------------------------------------------------------------------


def test_models_meta_from_system_response():
    mic, system = _tracks()
    client = FakeClient(
        {False: _fixture("deepgram_mic.json"), True: _fixture("deepgram_system.json")}
    )
    res = _backend(client).transcribe(mic, system, [], NullReporter())

    assert res.models_meta["asr_model"] == "deepgram-nova-3"
    assert res.models_meta["segmentation_model"] == "deepgram-diarizer"
    # the remote substitute for SHA pins: Deepgram's model_info, system track preferred
    # (the mic fixture carries a DIFFERENT model_info, so this assertion discriminates)
    assert res.models_meta["backend_model_versions"] == {
        "model-uuid-1": {
            "name": "2-general-nova",
            "version": "2026-01-01.0",
            "arch": "nova-3",
        }
    }
    assert res.models_meta["request_ids"] == ["mic-req-0002", "sys-req-0001"]


def test_models_meta_falls_back_to_mic_response():
    mic, _ = _tracks()
    client = FakeClient({False: _fixture("deepgram_mic.json")})
    res = _backend(client).transcribe(mic, None, [], NullReporter())

    assert res.models_meta["backend_model_versions"] == {
        "model-uuid-mic": {
            "name": "2-general-nova",
            "version": "2025-12-15.0",
            "arch": "nova-3",
        }
    }
    assert res.models_meta["request_ids"] == ["mic-req-0002"]
