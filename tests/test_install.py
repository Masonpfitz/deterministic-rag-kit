import json
from pathlib import Path

import pytest

from rag import install

ROOT = Path("/kits/one")
OTHER = Path("/kits/two")


def foreign():
    return {"model": "opus", "permissions": {"allow": ["Bash(ls)"]},
            "hooks": {"UserPromptSubmit": [{"hooks": [{"type": "command", "command": "echo mine", "timeout": 5}]}],
                      "Stop": [{"hooks": [{"type": "command", "command": "echo stop"}]}]}}


def test_install_into_missing_file(tmp_path):
    p = tmp_path / "s.json"
    res = install.install(p, ROOT, "/venv/python", "claude")
    assert res["backup"] is None
    data = json.loads(p.read_text())
    cmd = data["hooks"]["UserPromptSubmit"][0]["hooks"][0]
    assert cmd["timeout"] == 10 and "--drag-kit" in cmd["command"] and "kit.py" in cmd["command"]


def test_merge_keeps_everything_else_and_backs_up(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps(foreign()))
    res = install.install(p, ROOT, "/venv/python", "claude")
    assert Path(res["backup"]).read_text() == json.dumps(foreign())
    data = json.loads(p.read_text())
    assert data["model"] == "opus" and data["permissions"] == foreign()["permissions"]
    assert data["hooks"]["Stop"] == foreign()["hooks"]["Stop"]
    cmds = [h["command"] for g in data["hooks"]["UserPromptSubmit"] for h in g["hooks"]]
    assert "echo mine" in cmds and len(cmds) == 2


def test_install_twice_replaces_not_duplicates(tmp_path):
    p = tmp_path / "s.json"
    install.install(p, ROOT, "/venv/python", "claude")
    res = install.install(p, ROOT, "/venv/python2", "claude")
    assert res["replaced"]
    cmds = [h["command"] for g in json.loads(p.read_text())["hooks"]["UserPromptSubmit"] for h in g["hooks"]]
    assert len(cmds) == 1 and "python2" in cmds[0]


def test_uninstall_removes_only_our_entry(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps(foreign()))
    install.install(p, ROOT, "/venv/python", "claude")
    install.install(p, OTHER, "/venv/python", "claude")  # a second kit's hook must survive too
    res = install.uninstall(p, ROOT)
    assert res["removed"] == 1
    data = json.loads(p.read_text())
    cmds = [h["command"] for g in data["hooks"]["UserPromptSubmit"] for h in g["hooks"]]
    assert "echo mine" in cmds and any("two" in c for c in cmds) and not any("/one/" in c for c in cmds)
    assert data["hooks"]["Stop"] == foreign()["hooks"]["Stop"]


def test_uninstall_restores_original_when_it_was_the_only_entry(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"model": "opus"}))
    install.install(p, ROOT, "/venv/python", "codex")
    install.uninstall(p, ROOT)
    assert json.loads(p.read_text()) == {"model": "opus"}


def test_invalid_json_is_never_touched(tmp_path):
    p = tmp_path / "s.json"
    p.write_text("{ not json")
    with pytest.raises(ValueError):
        install.install(p, ROOT, "/venv/python", "claude")
    assert p.read_text() == "{ not json"
    assert list(tmp_path.iterdir()) == [p]


def test_uninstall_on_missing_file_is_a_noop(tmp_path):
    assert install.uninstall(tmp_path / "none.json", ROOT)["removed"] == 0


def test_codex_and_claude_use_the_same_event(tmp_path):
    p = tmp_path / "hooks.json"
    install.install(p, ROOT, "/venv/python", "codex")
    assert "UserPromptSubmit" in json.loads(p.read_text())["hooks"]
    assert install.is_installed(p, ROOT) and not install.is_installed(p, OTHER)


# ---------------------------------------------------------------- file permissions

import os
import stat
import sys

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")


def mode_of(p):
    return stat.S_IMODE(os.stat(p).st_mode)


@posix_only
@pytest.mark.parametrize("kind", ["hook", "claude_mcp", "codex_mcp"])
def test_private_files_stay_private_through_install_and_uninstall(tmp_path, kind):
    p = tmp_path / ("c.toml" if kind == "codex_mcp" else "c.json")
    p.write_text('model = "x"\n' if kind == "codex_mcp" else json.dumps({"model": "opus"}))
    os.chmod(p, 0o600)
    old = os.umask(0o022)
    try:
        inst = {"hook": lambda: install.install(p, ROOT, "/v/python", "claude"),
                "claude_mcp": lambda: install.install_claude_mcp(p, ROOT, "/v/python"),
                "codex_mcp": lambda: install.install_codex_mcp(p, ROOT, "/v/python")}[kind]
        uninst = {"hook": lambda: install.uninstall(p, ROOT),
                  "claude_mcp": lambda: install.uninstall_claude_mcp(p, ROOT),
                  "codex_mcp": lambda: install.uninstall_codex_mcp(p, ROOT)}[kind]
        res = inst()
        assert mode_of(p) == 0o600 and mode_of(res["backup"]) == 0o600
        res2 = uninst()
        assert mode_of(p) == 0o600 and mode_of(res2["backup"]) == 0o600
    finally:
        os.umask(old)
    assert not list(tmp_path.glob("*.tmp"))


@posix_only
def test_other_modes_are_preserved_and_new_files_are_private(tmp_path):
    p = tmp_path / "s.json"
    p.write_text("{}")
    os.chmod(p, 0o640)
    res = install.install(p, ROOT, "/v/python", "claude")
    assert mode_of(p) == 0o640 and mode_of(res["backup"]) == 0o640
    new = tmp_path / "sub" / "fresh.json"
    old = os.umask(0o000)  # even a permissive umask must not make a new settings file world-readable
    try:
        install.install(new, ROOT, "/v/python", "claude")
        install.install_claude_mcp(tmp_path / "mcp.json", ROOT, "/v/python")
        install.install_codex_mcp(tmp_path / "c.toml", ROOT, "/v/python")
    finally:
        os.umask(old)
    assert mode_of(new) == 0o600 and mode_of(tmp_path / "mcp.json") == 0o600 and mode_of(tmp_path / "c.toml") == 0o600


@posix_only
def test_a_leftover_predictable_tmp_file_is_not_used(tmp_path):
    p = tmp_path / "s.json"
    p.write_text("{}")
    os.chmod(p, 0o600)
    planted = tmp_path / "s.json.tmp"  # the name the old code would have written through
    planted.write_text("attacker content")
    os.chmod(planted, 0o666)
    install.install(p, ROOT, "/v/python", "claude")
    assert planted.read_text() == "attacker content" and mode_of(p) == 0o600
    assert "UserPromptSubmit" in json.loads(p.read_text())["hooks"]


def test_failed_write_leaves_the_original_and_no_temp_file(tmp_path, monkeypatch):
    p = tmp_path / "s.json"
    p.write_text('{"keep": true}')

    def boom(src, dst):
        raise OSError("disk full")
    monkeypatch.setattr(install.os, "replace", boom)
    with pytest.raises(OSError):
        install.atomic_write(p, "{}")
    assert p.read_text() == '{"keep": true}' and not [f for f in tmp_path.iterdir() if f.name.endswith(".tmp")]
