"""Managed llama-server client: free-port finder, lifecycle, HTTP (transport mocked)."""

import pytest

from meetscribe.llama import LlamaClient, LlamaServer, NoFreePort, find_free_port


def test_find_free_port_returns_first_bindable():
    attempts = []

    def bind(port):
        attempts.append(port)
        return len(attempts) >= 3  # first two "in use", third free

    port = find_free_port(tries=5, _bind=bind, _randint=lambda lo, hi: 20000 + len(attempts))
    assert port == 20002
    assert len(attempts) == 3


def test_find_free_port_gives_up_after_tries():
    with pytest.raises(NoFreePort):
        find_free_port(tries=5, _bind=lambda p: False, _randint=lambda lo, hi: 30000)


def test_client_complete_parses_chat_response():
    seen = {}

    def post(url, payload):
        seen["url"] = url
        seen["payload"] = payload
        return {"choices": [{"message": {"content": "Borderless GmbH"}}]}

    client = LlamaClient("http://127.0.0.1:9999", _post=post)
    assert client.complete("fix this") == "Borderless GmbH"
    assert seen["url"].endswith("/v1/chat/completions")
    assert seen["payload"]["messages"][0]["content"] == "fix this"
    assert seen["payload"]["temperature"] == 0.0  # deterministic


def test_client_max_tokens_override():
    seen = {}

    def post(url, payload):
        seen["payload"] = payload
        return {"choices": [{"message": {"content": "ok"}}]}

    client = LlamaClient("http://127.0.0.1:9999", max_tokens=512, _post=post)
    client.complete("hi", max_tokens=64)          # per-call override
    assert seen["payload"]["max_tokens"] == 64
    client.complete("hi")                          # falls back to the default
    assert seen["payload"]["max_tokens"] == 512


def test_server_terminates_on_exception():
    class FakeProc:
        def __init__(self):
            self.terminated = False

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return None

    proc = FakeProc()
    server = LlamaServer(
        "model.gguf",
        _popen=lambda *a, **k: proc,
        _find_port=lambda: 12345,
        _wait_ready=lambda port: None,
    )
    with pytest.raises(RuntimeError):
        with server:
            raise RuntimeError("boom")
    assert proc.terminated  # __exit__ tears the server down even on error


def test_server_exposes_base_url():
    proc = type("P", (), {"terminate": lambda self: None, "wait": lambda self, timeout=None: 0,
                          "poll": lambda self: None})()
    with LlamaServer("m.gguf", _popen=lambda *a, **k: proc,
                     _find_port=lambda: 23456, _wait_ready=lambda port: None) as s:
        assert s.base_url == "http://127.0.0.1:23456"
