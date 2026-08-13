"""Managed ``llama-server`` client for the transcript-cleanup LLM (Qwen2.5-7B GGUF).

llama.cpp has no stdio request/response transport, so we run ``llama-server`` on a loopback
port and talk to it over HTTP. The server is spawned once (load 4.7 GB weights once), polled
until healthy, and always torn down via the context manager — the same lifecycle discipline
``record.py`` uses for ffmpeg. Port selection is robust to a busy machine: a random high
loopback port is bind-tested with a few retries before giving up.

Only the transport is here; the cleanup logic (prompting, guards) lives in ``cleanup.py`` and
talks to this via the tiny ``LlamaClient.complete`` seam, so it is unit-tested with a fake.
"""

from __future__ import annotations

import json
import socket
import subprocess
import time
import urllib.request

_HOST = "127.0.0.1"


class NoFreePort(Exception):
    pass


def _try_bind(port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((_HOST, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def find_free_port(low: int = 20000, high: int = 60000, tries: int = 5,
                   _bind=_try_bind, _randint=None) -> int:
    """Return a free loopback port, retrying ``tries`` times before :class:`NoFreePort`."""
    import random

    randint = _randint or random.randint
    for _ in range(tries):
        port = randint(low, high)
        if _bind(port):
            return port
    raise NoFreePort(f"no free loopback port in [{low},{high}] after {tries} tries")


class LlamaServer:
    """Context manager owning a ``llama-server`` subprocess on a free loopback port."""

    def __init__(self, model_path: str, threads: int = 4, ready_timeout: float = 300.0,
                 _popen=subprocess.Popen, _find_port=find_free_port, _wait_ready=None) -> None:
        self.model_path = model_path
        self.threads = threads  # pinned so greedy+seed determinism is reproducible
        self.ready_timeout = ready_timeout
        self._popen = _popen
        self._find_port = _find_port
        self._wait_ready = _wait_ready or self._default_wait_ready
        self.port: int | None = None
        self.proc = None

    @property
    def base_url(self) -> str:
        return f"http://{_HOST}:{self.port}"

    def __enter__(self) -> "LlamaServer":
        self.port = self._find_port()
        self.proc = self._popen(
            [
                "llama-server", "--host", _HOST, "--port", str(self.port),
                "-m", self.model_path, "--threads", str(self.threads),
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._wait_ready(self.port)
        return self

    def __exit__(self, *exc) -> None:
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=10)
            except Exception:
                pass

    def _default_wait_ready(self, port: int) -> None:
        deadline = time.monotonic() + self.ready_timeout
        url = f"http://{_HOST}:{port}/health"
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=5) as r:
                    if r.status == 200:
                        return
            except Exception:
                pass
            time.sleep(1.0)
        raise TimeoutError(f"llama-server not healthy within {self.ready_timeout}s")


class LlamaClient:
    """Talks to a running ``llama-server`` via the OpenAI-compatible chat endpoint.

    Greedy (temperature 0) + fixed seed for reproducibility. The prompt (built in
    ``cleanup.py``, already carrying the system instruction) is sent as a single user message —
    llama-server applies Qwen's ChatML template."""

    def __init__(self, base_url: str, temperature: float = 0.0, seed: int = 0,
                 max_tokens: int = 512, timeout: float = 60.0, _post=None) -> None:
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._post = _post or self._http_post

    def complete(self, prompt: str, max_tokens: int | None = None) -> str:
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        data = self._post(f"{self.base_url}/v1/chat/completions", payload)
        return data["choices"][0]["message"]["content"]

    def _http_post(self, url: str, payload: dict) -> dict:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())
