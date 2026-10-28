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


def _hello_proof(token: str, nonce: str) -> str:
    import hashlib
    import hmac
    return hmac.new(token.encode("utf-8"), nonce.encode("utf-8"), hashlib.sha256).hexdigest()


def warn_once(cfg: Config, message: str) -> None:
    """Record why the reranker was skipped, once per distinct reason (not on every prompt)."""
    f = cfg.state / "rerank.warning"
    try:
        if f.exists() and f.read_text(encoding="utf-8") == message:
            return
        cfg.state.mkdir(parents=True, exist_ok=True)
        f.write_text(message, encoding="utf-8")
    except OSError:
        pass


def score_via_server(cfg: Config, query: str, passages: list[str]) -> list[float] | None:
    """Score passages through the local server, but only after the server proves it is ours. Another
    local process could be listening on the port while ours is down, so before any note text (or the
    token) is sent, the client sends a random nonce to /hello and checks the answer, which is an HMAC of
    the nonce under the secret token. A wrong or missing answer means no reranking, the same as when the
    server isn't running."""
    import hmac
    import secrets
    port = cfg.get("rerank", "port")
    try:
        token = token_path(cfg).read_text(encoding="utf-8").strip()
    except OSError:
        return None  # no server has started (or it has no token): use the fusion rule
    base = f"http://127.0.0.1:{port}"
    nonce = secrets.token_hex(16)
    try:
        hello = urllib.request.Request(base + "/hello", method="POST", data=json.dumps({"nonce": nonce}).encode(),
                                       headers={"Content-Type": "application/json"})  # no token, no notes
        with urllib.request.urlopen(hello, timeout=1.5) as r:
            proof = str(json.loads(r.read()).get("proof", ""))
    except Exception:  # noqa: BLE001 - nothing there, or something that doesn't speak our protocol
        return None
    if not hmac.compare_digest(proof, _hello_proof(token, nonce)):
        warn_once(cfg, f"Something other than the kit's reranker answered on 127.0.0.1:{port}. No notes were sent to it, "
                       "and reranking is off. Stop whatever uses that port or change [rerank] port in config.toml.")
        return None
    req = urllib.request.Request(
        base + "/score", method="POST",
        data=json.dumps({"query": query, "passages": passages}).encode(),
        headers={"Content-Type": "application/json", TOKEN_HEADER: token})
    try:
        with urllib.request.urlopen(req, timeout=2.5) as r:
            return json.loads(r.read())["scores"]
    except Exception:  # noqa: BLE001
        return None


SERVER_FILE = "rerank.server"


def status(cfg: Config) -> dict:
    """What `doctor` reports about the local reranker server: its recorded pid and port against the config."""
    info = {"token": token_path(cfg).exists(), "pid": None, "port": None, "alive": False,
            "config_port": cfg.get("rerank", "port")}
    try:
        rec = json.loads((cfg.state / SERVER_FILE).read_text(encoding="utf-8"))
        info["pid"], info["port"] = rec.get("pid"), rec.get("port")
    except (OSError, ValueError):
        return info
    if info["pid"]:
        from .index import _pid_alive
        info["alive"] = _pid_alive(int(info["pid"]))
    return info


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

        def _hello(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", 0) or 0)
                nonce = str(json.loads(self.rfile.read(length))["nonce"]) if 0 < length <= 1024 else ""
            except (ValueError, KeyError, TypeError):
                nonce = ""
            if not nonce or len(nonce) > 128:
                return self._refuse(400)
            data = json.dumps({"proof": _hello_proof(token, nonce)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):  # noqa: N802
            if self.headers.get("Origin") is not None:  # browsers always send it on cross-site posts
                return self._refuse(403)
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
            if host not in ("127.0.0.1", "localhost", "::1"):
                return self._refuse(403)
            if self.path == "/hello":  # identity proof; needs no token and reveals only an HMAC of the nonce
                return self._hello()
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

    server = HTTPServer(("127.0.0.1", listen), Handler)
    rec = cfg.state / SERVER_FILE  # next to the token, so `doctor` can tell a stale or mismatched server
    fd = os.open(rec, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "port": server.server_port}, f)
    os.chmod(rec, 0o600)
    return server


def serve(cfg: Config) -> None:
    load_model(cfg)
    server = build_server(cfg)
    print(f"reranker ready on http://127.0.0.1:{server.server_port} (Ctrl+C to stop)")
    server.serve_forever()
