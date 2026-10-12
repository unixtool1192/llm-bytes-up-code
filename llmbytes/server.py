"""Minimal client for a running llama.cpp `llama-server` (standard library only)."""

from __future__ import annotations

import json
import urllib.request


class LlamaServer:
    def __init__(self, url: str = "http://127.0.0.1:8099", timeout: int = 300):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(
            self.url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=5) as r:
                return json.loads(r.read()).get("status") == "ok"
        except OSError:
            return False

    def tokenize(self, text: str, add_special: bool = False) -> list[int]:
        return self.post("/tokenize", {"content": text, "add_special": add_special})["tokens"]

    def complete(self, prompt: str, **params) -> dict:
        body = {"prompt": prompt, "cache_prompt": False}
        body.update(params)
        return self.post("/completion", body)
