"""A small MCP server (stdio, newline-delimited JSON-RPC 2.0) that exposes one tool,
`search_notes`, so an agent can search your notes when it decides to, instead of (or as
well as) having passages injected by the hook.

Only the standard library is used. The server implements the three requests a client
needs (initialize, tools/list, tools/call) plus ping, and ignores notifications. Anything
that goes wrong inside a search comes back as a tool error, never as a crash.
"""
from __future__ import annotations

import json
import sys
from typing import IO

PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "notes-search"
TOOL = {
    "name": "search_notes",
    "description": (
        "Search the user's personal notes (an indexed copy of their markdown vault) with keyword and "
        "semantic search. Returns the best matching sections with the file path of each note, marked "
        "strong or weak. Use it when a question might be answered by the user's own notes, projects, "
        "people, decisions or reference material. Open a returned path to read the whole note. A weak "
        "match often means the notes don't cover it."),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for, in plain words."},
            "limit": {"type": "integer", "description": "How many sections to return (1 to 10, default 5).",
                      "minimum": 1, "maximum": 10},
        },
        "required": ["query"],
    },
}


class Server:
    def __init__(self, cfg, embedder=None):
        self.cfg, self.embedder, self._db = cfg, embedder, None

    def _search(self, query: str, limit: int) -> str:
        from . import embed, index, search
        if self.embedder is None:
            self.embedder = embed.build(self.cfg)
        if not self.cfg.db_path.exists():
            return "The notes index doesn't exist yet. Run: python3 kit.py reindex"
        if self._db is None:
            self._db = index.connect(self.cfg, create=False)
        return search.tool_results(self.cfg, query, self.embedder, self._db, limit)

    def handle(self, msg: dict):
        """One JSON-RPC message in, one response dict out (or None for a notification)."""
        method, mid = msg.get("method"), msg.get("id")
        if mid is None:  # notifications/initialized and friends need no reply
            return None
        if method == "initialize":
            ver = (msg.get("params") or {}).get("protocolVersion") or PROTOCOL_VERSION
            return {"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": ver, "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": "1.0"}}}
        if method == "ping":
            return {"jsonrpc": "2.0", "id": mid, "result": {}}
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": mid, "result": {"tools": [TOOL]}}
        if method == "tools/call":
            params = msg.get("params") or {}
            if params.get("name") != TOOL["name"]:
                return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": f"unknown tool {params.get('name')!r}"}}
            args = params.get("arguments") or {}
            query = args.get("query")
            if not isinstance(query, str) or not query.strip():
                return self._result(mid, "Give a non-empty `query`.", error=True)
            try:
                limit = int(args.get("limit", 5))
                return self._result(mid, self._search(query, limit))
            except Exception as exc:  # noqa: BLE001 - report, never crash the server
                return self._result(mid, f"Search failed: {type(exc).__name__}: {exc}", error=True)
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}

    @staticmethod
    def _result(mid, text: str, error: bool = False) -> dict:
        return {"jsonrpc": "2.0", "id": mid, "result": {"content": [{"type": "text", "text": text}], "isError": error}}


def serve(cfg, stdin: IO[str] | None = None, stdout: IO[str] | None = None, embedder=None) -> None:
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    server = Server(cfg, embedder)
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            stdout.write(json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}) + "\n")
            stdout.flush()
            continue
        if not isinstance(msg, dict):
            continue
        resp = server.handle(msg)
        if resp is not None:
            stdout.write(json.dumps(resp) + "\n")
            stdout.flush()
