import io
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from conftest import KIT, finish, write_doc
from rag import embed, install, mcp_server

ROOT, OTHER = Path("/kits/one"), Path("/kits/two")


def rpc(server, method, params=None, mid=1):
    return server.handle({"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}})


@pytest.fixture
def server(cfg):
    write_doc(cfg, "projects", "permit", "Harbor permit", "## Inspection\n\nThe plumber is booked for the 14th of March.")
    write_doc(cfg, "projects", "budget", "Harbor budget", "Total budget 48000 with ten percent contingency.")
    finish(cfg)
    return mcp_server.Server(cfg, embed.build(cfg))


def test_handshake_and_tool_listing(server):
    init = rpc(server, "initialize", {"protocolVersion": "2025-03-26"})
    assert init["result"]["protocolVersion"] == "2025-03-26" and "tools" in init["result"]["capabilities"]
    tools = rpc(server, "tools/list")["result"]["tools"]
    assert [t["name"] for t in tools] == ["search_notes"] and tools[0]["inputSchema"]["required"] == ["query"]
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert rpc(server, "ping")["result"] == {}
    assert rpc(server, "nope")["error"]["code"] == -32601


def test_search_returns_paths_sections_and_strength(server):
    res = rpc(server, "tools/call", {"name": "search_notes", "arguments": {"query": "when is the plumber booked", "limit": 3}})
    text = res["result"]["content"][0]["text"]
    assert not res["result"]["isError"]
    assert "permit.md" in text and "section: Inspection" in text and "14th of March" in text
    assert "match: strong" in text or "match: weak" in text


def test_tool_errors_are_results_not_crashes(server, cfg):
    bad = rpc(server, "tools/call", {"name": "search_notes", "arguments": {"query": "  "}})
    assert bad["result"]["isError"]
    assert "error" in rpc(server, "tools/call", {"name": "other", "arguments": {}})

    class Boom(embed.Fake):
        def embed(self, texts, timeout=None):
            raise RuntimeError("boom")
    broken = mcp_server.Server(cfg, Boom())
    res = rpc(broken, "tools/call", {"name": "search_notes", "arguments": {"query": "plumber"}})
    assert res["result"]["isError"] and "boom" in res["result"]["content"][0]["text"]


def test_no_index_says_so(cfg):
    res = rpc(mcp_server.Server(cfg, embed.build(cfg)), "tools/call", {"name": "search_notes", "arguments": {"query": "plumber"}})
    assert "reindex" in res["result"]["content"][0]["text"]


def test_deprecated_notes_never_come_back(cfg):
    write_doc(cfg, "projects", "gone", "Old plan", "The kiln schedule quokka.", status="deprecated")
    write_doc(cfg, "projects", "live", "New plan", "The kiln schedule is Tuesday.")
    finish(cfg)
    s = mcp_server.Server(cfg, embed.build(cfg))
    text = rpc(s, "tools/call", {"name": "search_notes", "arguments": {"query": "kiln schedule quokka"}})["result"]["content"][0]["text"]
    assert "gone.md" not in text and "live.md" in text


def test_stdio_loop_handles_bad_lines(server):
    out = io.StringIO()
    lines = "\n".join([json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}), "not json", "",
                       json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})]) + "\n"
    mcp_server.serve(server.cfg, io.StringIO(lines), out, embedder=server.embedder)
    replies = [json.loads(x) for x in out.getvalue().splitlines()]
    assert replies[0]["result"]["tools"] and replies[1]["error"]["code"] == -32700 and len(replies) == 2


def test_real_subprocess_speaks_mcp(tmp_path):
    """The actual command an agent would launch, over real pipes. No index exists in a fresh
    checkout, so the search itself reports that; the protocol still works."""
    msgs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}]
    r = subprocess.run([sys.executable, str(KIT / "kit.py"), "mcp-serve", "--harness", "claude", "--drag-kit", "x"],
                       input="\n".join(json.dumps(m) for m in msgs) + "\n", capture_output=True, text=True, timeout=30,
                       env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"})
    replies = [json.loads(x) for x in r.stdout.splitlines()]
    assert r.returncode == 0 and replies[0]["result"]["serverInfo"]["name"] == "notes-search"
    assert replies[1]["result"]["tools"][0]["name"] == "search_notes"


# ------------------------------------------------------------------ Claude Code config (JSON)

def foreign_claude():
    return {"numStartups": 7, "projects": {"/x": {"mcpServers": {"p": {"command": "p"}}}},
            "mcpServers": {"other": {"command": "node", "args": ["srv.js"]}}}


def test_claude_mcp_merge_keeps_everything_and_backs_up(tmp_path):
    p = tmp_path / ".claude.json"
    p.write_text(json.dumps(foreign_claude()))
    res = install.install_claude_mcp(p, ROOT, "/venv/python")
    assert Path(res["backup"]).read_text() == json.dumps(foreign_claude())
    d = json.loads(p.read_text())
    assert d["numStartups"] == 7 and d["projects"] == foreign_claude()["projects"]
    assert d["mcpServers"]["other"] == foreign_claude()["mcpServers"]["other"]
    ours = d["mcpServers"]["notes"]
    assert ours["command"] == "/venv/python" and "mcp-serve" in ours["args"] and "--drag-kit" in ours["args"]


def test_claude_mcp_install_twice_and_name_collision(tmp_path):
    p = tmp_path / "m.json"
    install.install_claude_mcp(p, ROOT, "/v/python")
    assert install.install_claude_mcp(p, ROOT, "/v/python2")["replaced"]
    d = json.loads(p.read_text())
    assert list(d["mcpServers"]) == ["notes"] and d["mcpServers"]["notes"]["command"] == "/v/python2"
    res = install.install_claude_mcp(p, OTHER, "/v/python")  # a second kit must not clobber the first
    assert res["name"].startswith("notes-") and len(json.loads(p.read_text())["mcpServers"]) == 2


def test_claude_mcp_uninstall_removes_only_ours(tmp_path):
    p = tmp_path / ".claude.json"
    p.write_text(json.dumps(foreign_claude()))
    install.install_claude_mcp(p, ROOT, "/v/python")
    install.install_claude_mcp(p, OTHER, "/v/python")
    assert install.uninstall_claude_mcp(p, ROOT)["removed"] == 1
    d = json.loads(p.read_text())
    assert set(d["mcpServers"]) == {"other", f"notes-{install.kit_id(OTHER)}"}
    assert install.uninstall_claude_mcp(p, OTHER)["removed"] == 1
    assert json.loads(p.read_text()) == foreign_claude()


def test_claude_mcp_invalid_json_untouched(tmp_path):
    p = tmp_path / "m.json"
    p.write_text("{ nope")
    with pytest.raises(ValueError):
        install.install_claude_mcp(p, ROOT, "/v/python")
    assert p.read_text() == "{ nope"


# ------------------------------------------------------------------ Codex config (TOML)

CODEX = '''model = "gpt-5"

[mcp_servers.other]
command = "node"
args = ["srv.js"]

[mcp_servers.other.env]
KEY = "value"

[features]
hooks = true
'''


def test_codex_toml_block_is_appended_valid_and_removable(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(CODEX)
    res = install.install_codex_mcp(p, ROOT, "/venv/python")
    assert Path(res["backup"]).read_text() == CODEX
    d = tomllib.loads(p.read_text())
    assert d["model"] == "gpt-5" and d["mcp_servers"]["other"]["env"] == {"KEY": "value"} and d["features"] == {"hooks": True}
    ours = d["mcp_servers"]["notes"]
    assert ours["command"] == "/venv/python" and "mcp-serve" in ours["args"] and install.kit_id(ROOT) in ours["args"]
    assert install.uninstall_codex_mcp(p, ROOT)["removed"] == 1
    assert p.read_text().strip() == CODEX.strip() and tomllib.loads(p.read_text())["mcp_servers"].keys() == {"other"}


def test_codex_install_twice_is_one_block_and_collision_renames(tmp_path):
    p = tmp_path / "config.toml"
    install.install_codex_mcp(p, ROOT, "/v/python")
    assert install.install_codex_mcp(p, ROOT, "/v/python2")["replaced"]
    assert p.read_text().count(">>> drag-kit") == 1 and tomllib.loads(p.read_text())["mcp_servers"]["notes"]["command"] == "/v/python2"
    res = install.install_codex_mcp(p, OTHER, "/v/python")
    assert res["name"].startswith("notes-") and set(tomllib.loads(p.read_text())["mcp_servers"]) == {"notes", res["name"]}
    install.uninstall_codex_mcp(p, ROOT)
    assert set(tomllib.loads(p.read_text())["mcp_servers"]) == {res["name"]}


def test_codex_windows_paths_and_invalid_toml(tmp_path):
    p = tmp_path / "config.toml"
    install.install_codex_mcp(p, Path("C:/kits/a b"), "C:\\Users\\me\\kit\\.venv\\Scripts\\python.exe")
    assert tomllib.loads(p.read_text())["mcp_servers"]["notes"]["command"] == "C:\\Users\\me\\kit\\.venv\\Scripts\\python.exe"
    bad = tmp_path / "bad.toml"
    bad.write_text("this = [not valid")
    with pytest.raises(ValueError):
        install.install_codex_mcp(bad, ROOT, "/v/python")
    assert bad.read_text() == "this = [not valid" and not list(tmp_path.glob("bad.toml.bak*"))


def test_cli_modes_with_temp_files(tmp_path):
    """inject = hook only (the default), tool = MCP only, both = both; uninstall removes all of it."""
    from conftest import make_kit_copy
    kit = make_kit_copy(tmp_path)

    def run(*args):
        return subprocess.run([sys.executable, str(kit / "kit.py"), *args], capture_output=True, text=True, timeout=60,
                              env={"HOME": str(tmp_path / "home"), "PATH": "/usr/bin:/bin"})
    paths = ["--claude-settings", str(tmp_path / "s.json"), "--claude-mcp", str(tmp_path / "c.json"),
             "--codex-hooks", str(tmp_path / "h.json"), "--codex-config", str(tmp_path / "t.toml")]
    assert run("install", "--harness", "both", *paths).returncode == 0
    assert (tmp_path / "s.json").exists() and (tmp_path / "h.json").exists()
    assert not (tmp_path / "c.json").exists() and not (tmp_path / "t.toml").exists()  # default mode is inject
    assert run("install", "--harness", "both", "--mode", "tool", *paths).returncode == 0
    assert "mcpServers" in json.loads((tmp_path / "c.json").read_text()) and "mcp_servers" in tomllib.loads((tmp_path / "t.toml").read_text())
    r = run("uninstall", *paths)
    assert r.returncode == 0
    for f in ("s.json", "h.json", "c.json"):
        assert json.loads((tmp_path / f).read_text()) == {}
    assert tomllib.loads((tmp_path / "t.toml").read_text()) == {}
    assert not (tmp_path / "home").exists() or not any((tmp_path / "home").rglob("*.json"))
