"""The prompt hook: reads the prompt, prints matching passages, always exits 0.

Claude Code (UserPromptSubmit) and Codex (UserPromptSubmit) both send JSON on stdin
with a `prompt` field and add whatever the hook prints to stdout as context. This
script fails open: if anything goes wrong it prints nothing, never exits 2, and gives
up after the configured time budget so it can never hold a prompt for long.
"""
from __future__ import annotations

import json
import sys
import threading
import time


def run(stdin_text: str, root=None, out=None, embedder=None, header=None) -> str:
    """Return the text to inject ('' for nothing). Importable so tests can call it."""
    try:
        payload = json.loads(stdin_text)
    except ValueError:
        return ""
    if not isinstance(payload, dict):
        return ""
    prompt = payload.get("prompt") or payload.get("user_prompt") or ""
    if not isinstance(prompt, str):
        return ""
    prompt = prompt.strip()
    from . import config as config_mod
    cfg = config_mod.load(root)
    if header:
        cfg = cfg.override(injection_header=header)
    if len(prompt) < cfg.get("retrieval", "min_prompt_chars"):
        return ""
    if prompt.startswith("<") or "[SYSTEM NOTIFICATION" in prompt:  # harness-made, not typed
        return ""
    box: dict = {}

    def work():
        try:
            from . import embed, index, search
            emb = embedder or embed.build(cfg)
            if not cfg.db_path.exists():
                return
            index_ok = True
            try:
                index.maybe_background_reindex(cfg)
            except Exception:  # noqa: BLE001
                index_ok = False
            res = search.retrieve(cfg, prompt, emb)
            box["block"] = res.block
            if cfg.get("retrieval", "rerank_enabled") and any("reranker unavailable" in w for w in res.warnings):
                _start_reranker(cfg)
            box["ok"] = index_ok
        except Exception:  # noqa: BLE001 - fail open on anything
            box["block"] = ""

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(cfg.get("hook", "budget_s"))
    return box.get("block", "") if not t.is_alive() else ""


def _start_reranker(cfg) -> None:
    import os
    import subprocess
    stamp = cfg.state / "rerank_start"
    try:
        if time.time() - stamp.stat().st_mtime < 300:
            return
    except OSError:
        pass
    stamp.write_text(str(time.time()))
    kw: dict = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if os.name == "nt":
        kw["creationflags"] = 0x00000008 | 0x00000200
    else:
        kw["start_new_session"] = True
    subprocess.Popen([sys.executable, str(cfg.root / "kit.py"), "serve"], **kw)


def main(argv: list[str] | None = None, header: str | None = None) -> int:
    """Read the prompt from stdin and print the passages. Both streams are handled as UTF-8 bytes so a
    Windows console code page can't make a non-ASCII note inject nothing."""
    import os
    try:
        raw = sys.stdin.buffer.read() if hasattr(sys.stdin, "buffer") else sys.stdin.read().encode("utf-8")
        block = run(raw.decode("utf-8", "replace"), header=header)
        if block:
            data = (block + "\n").encode("utf-8")
            out = getattr(sys.stdout, "buffer", None)
            if out is not None:
                out.write(data)
                out.flush()
            else:
                sys.stdout.write(data.decode("utf-8"))
                sys.stdout.flush()
    except BaseException:  # noqa: BLE001
        pass
    try:  # a search thread that timed out must not hold the interpreter open
        sys.stdout.flush()
    except Exception:  # noqa: BLE001
        pass
    if __name__ != "__main__" and not os.environ.get("RAG_HOOK_NO_EXIT"):
        os._exit(0)
    return 0
