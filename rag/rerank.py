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
AUTH_HEADERS = ("X-Rerank-Nonce", "X-Rerank-Time", "X-Rerank-Auth")
WINDOW_S = 30  # how far a request timestamp may be from the server clock


def token_path(cfg: Config):
    return cfg.state / TOKEN_FILE


def _mac(token: str, message: str) -> str:
    import hashlib
    import hmac
    return hmac.new(token.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()


def client_proof(token: str, nonce: str, port: int) -> str:
    """What the client proves first. Bound to the port it dialed, so a process that relays this to the
    real server on another port gets a proof the real server refuses."""
    return _mac(token, f"client|{nonce}|127.0.0.1:{port}")


def server_proof(token: str, nonce: str, port: int) -> str:
    return _mac(token, f"server|{nonce}|127.0.0.1:{port}")


def auth_headers(token: str, port: int, body: bytes, nonce: str | None = None, ts: int | None = None) -> dict:
    """Headers that authenticate one /score request without sending the token: a fresh nonce, the time, and
    an HMAC over nonce, time, port and the SHA-256 of the exact body."""
    import hashlib
    import secrets
    import time
    nonce = nonce or secrets.token_hex(16)
    ts = int(time.time()) if ts is None else ts
    digest = hashlib.sha256(body).hexdigest()
    mac = _mac(token, f"score|{nonce}|{ts}|127.0.0.1:{port}|{digest}")
    return {"X-Rerank-Nonce": nonce, "X-Rerank-Time": str(ts), "X-Rerank-Auth": mac}


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
    """Score passages through the local server, but only after a mutual, port-bound handshake. Another
    local process could be listening on the port while ours is down, or could relay to ours from another
    port. So: the client sends a nonce and a proof bound to the port it dialed (the server refuses a proof
    made for another port, so it is never a signing oracle for a relay), the server answers with its own
    proof bound to its port, and the client checks that against the port it dialed. Only then does it send
    /score, and /score carries a per-request HMAC, never the token. Any failure means no reranking, the
    same as when the server isn't running."""
    import hmac
    import secrets
    port = int(cfg.get("rerank", "port"))
    try:
        token = token_path(cfg).read_text(encoding="utf-8").strip()
    except OSError:
        return None  # no server has started (or it has no token): use the fusion rule
    base = f"http://127.0.0.1:{port}"
    nonce = secrets.token_hex(16)
    try:
        hello = urllib.request.Request(
            base + "/hello", method="POST",
            data=json.dumps({"nonce": nonce, "client_proof": client_proof(token, nonce, port)}).encode(),
            headers={"Content-Type": "application/json"})  # no token, no notes
        with urllib.request.urlopen(hello, timeout=1.5) as r:
            answer = str(json.loads(r.read()).get("proof", ""))
    except Exception:  # noqa: BLE001 - nothing there, or something that refused or doesn't speak our protocol
        return None
    if not hmac.compare_digest(answer, server_proof(token, nonce, port)):
        warn_once(cfg, f"Something other than the kit's reranker answered on 127.0.0.1:{port}. No notes were sent to it, "
                       "and reranking is off. Stop whatever uses that port or change [rerank] port in config.toml.")
        return None
    body = json.dumps({"query": query, "passages": passages}).encode()
    req = urllib.request.Request(base + "/score", method="POST", data=body,
                                 headers={"Content-Type": "application/json", **auth_headers(token, port, body)})
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
    """The local scoring server, bound to 127.0.0.1 only. POST /hello proves it is ours (and only to a
    client that proves it knows the token, for this port); POST /score needs a fresh per-request HMAC.
    It refuses: a browser request (any Origin header), a Host that isn't loopback, a content type other
    than exactly application/json, a body over 1 MB, a stale or replayed request, and anything without
    a valid proof. The token itself never crosses the wire."""
    import hashlib
    import hmac
    import threading
    import time
    token = write_token(cfg)
    scorer = scorer or (lambda q, ps: score_in_process(cfg, q, ps))
    listen = cfg.get("rerank", "port") if port is None else port
    seen: dict[str, float] = {}
    seen_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _refuse(self, code: int) -> None:
            self.send_response(code)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _reply(self, obj) -> None:
            data = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read(self, limit: int):
            try:
                length = int(self.headers.get("Content-Length", 0) or 0)
            except ValueError:
                return None
            return self.rfile.read(length) if 0 < length <= limit else None

        def _hello(self, my_port: int) -> None:
            raw = self._read(1024)
            try:
                body = json.loads(raw) if raw else {}
                nonce, theirs = str(body["nonce"]), str(body["client_proof"])
            except (ValueError, KeyError, TypeError):
                return self._refuse(400)
            if not nonce or len(nonce) > 128:
                return self._refuse(400)
            # The client's proof must be bound to THIS port. A relay that forwards a proof made for its own
            # port gets a refusal, so this server never signs for someone who doesn't know the token.
            if not hmac.compare_digest(theirs, client_proof(token, nonce, my_port)):
                return self._refuse(403)
            self._reply({"proof": server_proof(token, nonce, my_port)})

        def _authorized(self, my_port: int, raw: bytes) -> bool:
            nonce = self.headers.get(AUTH_HEADERS[0], "")
            ts_text = self.headers.get(AUTH_HEADERS[1], "")
            mac = self.headers.get(AUTH_HEADERS[2], "")
            try:
                ts = int(ts_text)
            except ValueError:
                return False
            now = time.time()
            if not nonce or len(nonce) > 128 or abs(now - ts) > WINDOW_S:
                return False
            want = _mac(token, f"score|{nonce}|{ts}|127.0.0.1:{my_port}|{hashlib.sha256(raw).hexdigest()}")
            if not hmac.compare_digest(mac, want):
                return False
            with seen_lock:  # a valid request can be used once
                for k in [k for k, exp in seen.items() if exp < now]:
                    del seen[k]
                if nonce in seen:
                    return False
                seen[nonce] = now + 2 * WINDOW_S
            return True

        def do_POST(self):  # noqa: N802
            if self.headers.get("Origin") is not None:  # browsers always send it on cross-site posts
                return self._refuse(403)
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
            if host not in ("127.0.0.1", "localhost", "::1"):
                return self._refuse(403)
            if self.headers.get_content_type() != "application/json":
                return self._refuse(415)
            my_port = self.server.server_port
            if self.path == "/hello":
                return self._hello(my_port)
            if self.path != "/score":
                return self._refuse(404)
            try:
                length = int(self.headers.get("Content-Length", 0) or 0)
            except ValueError:
                return self._refuse(400)
            if not 0 < length <= 1_000_000:
                return self._refuse(413)
            raw = self.rfile.read(length)
            if not self._authorized(my_port, raw):
                return self._refuse(401)
            try:
                body = json.loads(raw)
                query, passages = str(body["query"]), [str(p) for p in body["passages"]]
            except (ValueError, KeyError, TypeError):
                return self._refuse(400)
            self._reply({"scores": scorer(query, passages) or []})

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
