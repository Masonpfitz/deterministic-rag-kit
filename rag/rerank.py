"""Optional cross-encoder reranker (BAAI/bge-reranker-v2-m3 by default).

It needs `pip install sentence-transformers` (a multi-gigabyte torch download) and the
model itself (another ~2 GB on first use), which is why it is off by default.

A hook is a fresh process for every prompt, and loading the model takes seconds, so the
hook never loads it. `kit.py serve` keeps it loaded in a small local server (127.0.0.1
only); the hook calls that, and falls back to the cosine and word-overlap check when it is not running.
"""
from __future__ import annotations

import json
import os
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

from .config import Config

_model = None


def load_model(cfg: Config):
    global _model
    if _model is None:
        import torch  # noqa: F401
        from sentence_transformers import CrossEncoder

        device = "cpu"
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        _model = CrossEncoder(cfg.get("rerank", "model"), max_length=512, device=device)
        if device in ("mps", "cuda"):
            _model.model.half()
    return _model


def score_in_process(cfg: Config, query: str, passages: list[str]) -> list[float] | None:
    try:
        model = load_model(cfg)
        return [float(s) for s in model.predict([(query, p) for p in passages])]
    except Exception:  # noqa: BLE001 - missing package, no model, out of memory: fall back
        return None


TOKEN_FILE = "rerank.token"
TOKEN_HEADER = "X-Rerank-Token"


def token_path(cfg: Config):
    return cfg.state / TOKEN_FILE


def score_via_server(cfg: Config, query: str, passages: list[str]) -> list[float] | None:
    port = cfg.get("rerank", "port")
    try:
        token = token_path(cfg).read_text(encoding="utf-8").strip()
    except OSError:
        return None  # no server has started (or it has no token): use the fusion rule
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/score", method="POST",
        data=json.dumps({"query": query, "passages": passages}).encode(),
        headers={"Content-Type": "application/json", TOKEN_HEADER: token})
    try:
        with urllib.request.urlopen(req, timeout=2.5) as r:
            return json.loads(r.read())["scores"]
    except Exception:  # noqa: BLE001
        return None


def write_token(cfg: Config) -> str:
    """A fresh random token, readable only by you, that the kit's client sends with every request."""
    import secrets
    cfg.state.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(32)
    path = token_path(cfg)
    try:
        os.unlink(path)
    except OSError:
        pass
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(token)
    os.chmod(path, 0o600)
    return token


def build_server(cfg: Config, scorer=None, port: int | None = None) -> HTTPServer:
    """The local scoring server, bound to 127.0.0.1 only. It answers POST /score and nothing else, and
    refuses: a browser request (any Origin header), a Host that isn't loopback, a content type other than
    exactly application/json, a body over 1 MB, and any request without the random token."""
    import hmac
    token = write_token(cfg)
    scorer = scorer or (lambda q, ps: score_in_process(cfg, q, ps))
    listen = cfg.get("rerank", "port") if port is None else port

    class Handler(BaseHTTPRequestHandler):
        def _refuse(self, code: int) -> None:
            self.send_response(code)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self):  # noqa: N802
            if self.headers.get("Origin") is not None:  # browsers always send it on cross-site posts
                return self._refuse(403)
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
            if host not in ("127.0.0.1", "localhost", "::1"):
                return self._refuse(403)
            if not hmac.compare_digest(self.headers.get(TOKEN_HEADER, ""), token):
                return self._refuse(401)
            if self.path != "/score":
                return self._refuse(404)
            if self.headers.get_content_type() != "application/json":
                return self._refuse(415)
            try:
                length = int(self.headers.get("Content-Length", 0) or 0)
            except ValueError:
                return self._refuse(400)
            if not 0 < length <= 1_000_000:
                return self._refuse(413)
            try:
                body = json.loads(self.rfile.read(length))
                query, passages = str(body["query"]), [str(p) for p in body["passages"]]
            except (ValueError, KeyError, TypeError):
                return self._refuse(400)
            data = json.dumps({"scores": scorer(query, passages) or []}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):  # keep the terminal quiet
            pass

    return HTTPServer(("127.0.0.1", listen), Handler)


def serve(cfg: Config) -> None:
    load_model(cfg)
    server = build_server(cfg)
    print(f"reranker ready on http://127.0.0.1:{server.server_port} (Ctrl+C to stop)")
    server.serve_forever()
