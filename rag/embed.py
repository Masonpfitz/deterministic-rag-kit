"""Embedding providers. All use urllib, so there is nothing extra to install.

`ollama`  local, free, default (bge-m3).
`openai`  text-embedding-3-small; key comes from OPENAI_API_KEY in .env.
`fake`    deterministic hashed bag-of-words. For tests and timing only, never for real notes.
`none`    keyword search only.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.error
import urllib.request

from .config import Config


class EmbedError(RuntimeError):
    pass


def _post(url: str, payload: dict, timeout: float, headers: dict | None = None) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:200]
        raise EmbedError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise EmbedError(f"{url}: {exc}") from exc


class Embedder:
    name = "none"
    model = ""

    def embed(self, texts: list[str], timeout: float | None = None) -> list[list[float]]:
        raise EmbedError("embeddings are disabled (provider = none)")

    @property
    def signature(self) -> str:
        """Stored with the index; if it changes, every vector is recomputed."""
        return f"{self.name}:{self.model}"


class Ollama(Embedder):
    name = "ollama"

    def __init__(self, cfg: Config):
        e = cfg.section("embeddings")
        self.model, self.url = e["ollama_model"], e["ollama_url"].rstrip("/")
        self.keep_alive, self.index_timeout = e["ollama_keep_alive"], e["index_timeout_s"]
        self.query_timeout = e["query_timeout_s"]

    def embed(self, texts, timeout=None):
        data = _post(f"{self.url}/api/embed",
                     {"model": self.model, "input": texts, "keep_alive": self.keep_alive},
                     timeout or self.index_timeout)
        vecs = data.get("embeddings")
        if not vecs or len(vecs) != len(texts):
            raise EmbedError(f"Ollama returned {len(vecs or [])} vectors for {len(texts)} texts")
        return vecs


class OpenAI(Embedder):
    name = "openai"

    def __init__(self, cfg: Config):
        e = cfg.section("embeddings")
        self.model = e["openai_model"]
        self.index_timeout, self.query_timeout = e["index_timeout_s"], e["query_timeout_s"]

    def embed(self, texts, timeout=None):
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise EmbedError("OPENAI_API_KEY is not set (put it in the kit's .env file)")
        data = _post("https://api.openai.com/v1/embeddings",
                     {"model": self.model, "input": [t[:24000] for t in texts]},
                     timeout or self.index_timeout, {"Authorization": f"Bearer {key}"})
        rows = sorted(data["data"], key=lambda d: d["index"])
        return [r["embedding"] for r in rows]


class Fake(Embedder):
    name, model = "fake", "hash-1024"
    query_timeout = 1.0

    def embed(self, texts, timeout=None):
        out = []
        for t in texts:
            v = [0.0] * 1024
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                h = int.from_bytes(hashlib.md5(w.encode()).digest()[:4], "little")
                v[h % 1024] += 1.0 if (h >> 8) & 1 else -1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


def build(cfg: Config) -> Embedder:
    provider = cfg.get("embeddings", "provider")
    if provider == "ollama":
        return Ollama(cfg)
    if provider == "openai":
        return OpenAI(cfg)
    if provider == "fake":
        return Fake()
    if provider == "none":
        return Embedder()
    raise EmbedError(f"unknown embeddings provider {provider!r}; use ollama, openai or none")
