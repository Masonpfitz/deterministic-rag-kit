"""Add and remove this kit's prompt hook in Claude Code and Codex settings files.

Safety rules, each covered by a test:
  * the file is backed up (timestamped copy) before it is changed;
  * every other key and every other hook is left exactly as it was;
  * the kit's own entry is recognised by a marker on its command line, so installing
    twice replaces rather than duplicates, and uninstall removes only that entry;
  * a settings file that is not valid JSON is never touched.
Every function takes the settings path as an argument.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shlex
import subprocess
import time
from pathlib import Path

from .constants import HOOK_MARKER

EVENT = "UserPromptSubmit"


def kit_id(root: Path) -> str:
    """A stable id for this kit, kept in <root>/.kit-id so the marker on installed entries survives
    moving or renaming the folder. If the file can't be read or created (a read-only or
    nonexistent root), fall back to a hash of the path."""
    f = Path(root) / ".kit-id"
    try:
        v = f.read_text(encoding="utf-8").strip()
        if re.fullmatch(r"[0-9a-f]{8}", v):
            return v
    except OSError:
        pass
    try:
        v = secrets.token_hex(4)
        f.write_text(v + "\n", encoding="utf-8")
        return v
    except OSError:
        return hashlib.sha1(str(Path(root).resolve()).encode()).hexdigest()[:8]


INSTALLS_FILE = ".kit-installs.json"


def record_install(root: Path, kind: str, path: Path) -> None:
    """Remember every file we wrote into, so uninstall can clean all of them, project files included."""
    f = Path(root) / INSTALLS_FILE
    try:
        items = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        items = []
    entry = {"kind": kind, "path": str(Path(os.path.realpath(path)))}
    if entry not in items:
        items.append(entry)
    try:
        f.write_text(json.dumps(items, indent=1), encoding="utf-8")
    except OSError:
        pass


def recorded_installs(root: Path) -> list[dict]:
    try:
        return json.loads((Path(root) / INSTALLS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def hook_command(root: Path, python: str, harness: str) -> str:
    argv = [python, str(root / "kit.py"), "hook", "--harness", harness, HOOK_MARKER, kit_id(root)]
    return subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)


def _is_ours(entry: dict, root: Path) -> bool:
    cmd = str(entry.get("command", ""))
    return HOOK_MARKER in cmd and kit_id(root) in cmd


def _real(path: Path) -> Path:
    """Follow a symlinked settings file (stow, chezmoi, a dotfiles repo) so we write through it to
    the real file instead of replacing the link."""
    return Path(os.path.realpath(path))


def _load(path: Path) -> dict:
    path = _real(path)
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"{path} is not valid JSON ({exc}); leaving it alone") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object; leaving it alone")
    return data


def _mode(path: Path) -> int:
    """The file's permission bits, or 0600 for a file that doesn't exist yet. Settings files can
    hold credentials, so a new one starts private."""
    try:
        return os.stat(path).st_mode & 0o777
    except FileNotFoundError:
        return 0o600


def _backup(path: Path) -> Path | None:
    """A timestamped copy with the original's permissions, created with those bits from the start
    so it is never briefly more readable than the file it copies."""
    path = _real(path)
    if not path.exists():
        return None
    mode = _mode(path)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    n = 0
    while True:
        dest = path.with_name(f"{path.name}.bak-{stamp}" + (f"-{n}" if n else ""))
        try:
            fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
            break
        except FileExistsError:
            n += 1
    with os.fdopen(fd, "wb") as f:
        f.write(path.read_bytes())
        f.flush()
        os.fsync(f.fileno())
    os.chmod(dest, mode)  # the umask may have stripped bits at creation
    return dest


def atomic_write(path: Path, text: str) -> None:
    """Replace `path` without ever exposing a half-written or too-permissive file: write a uniquely
    named temp file in the same folder (mkstemp creates it 0600, so a pre-created or leftover
    predictable name can't be hijacked), apply the original's mode, fsync, then rename over it."""
    import tempfile
    path = _real(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = _mode(path)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _write(path: Path, data: dict) -> None:
    atomic_write(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def install(path: Path, root: Path, python: str, harness: str, timeout_s: int = 10) -> dict:
    """Merge the hook into `path`. Returns {backup, replaced}."""
    data = _load(path)
    hooks = data.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise ValueError(f"{path}: `hooks` is not an object; leaving it alone")
    groups = hooks.setdefault(EVENT, [])
    replaced = False
    emptied = []
    for g in groups:
        before = len(g.get("hooks", []))
        g["hooks"] = [h for h in g.get("hooks", []) if not _is_ours(h, root)]
        if len(g["hooks"]) != before:
            replaced = True
            if not g["hooks"]:
                emptied.append(g)  # a group that held only our entry
    groups[:] = [g for g in groups if not any(g is e for e in emptied)]
    groups.append({"hooks": [{"type": "command", "command": hook_command(root, python, harness),
                              "timeout": timeout_s}]})
    backup = _backup(path)
    _write(path, data)
    return {"backup": str(backup) if backup else None, "replaced": replaced}


def uninstall(path: Path, root: Path) -> dict:
    """Remove only this kit's entries. Returns {removed, backup}."""
    if not _real(path).exists():
        return {"removed": 0, "backup": None}
    data = _load(path)
    hooks = data.get("hooks")
    removed = 0
    if isinstance(hooks, dict) and isinstance(hooks.get(EVENT), list):
        groups = hooks[EVENT]
        emptied = []
        for g in groups:
            keep = [h for h in g.get("hooks", []) if not _is_ours(h, root)]
            if len(keep) != len(g.get("hooks", [])):
                removed += len(g.get("hooks", [])) - len(keep)
                if not keep:
                    emptied.append(g)
            g["hooks"] = keep
        hooks[EVENT] = [g for g in groups if not any(g is e for e in emptied)]
        if not hooks[EVENT]:
            del hooks[EVENT]
        if not hooks:
            del data["hooks"]
    if not removed:
        return {"removed": 0, "backup": None}
    backup = _backup(path)
    _write(path, data)
    return {"removed": removed, "backup": str(backup) if backup else None}


def is_installed(path: Path, root: Path) -> bool:
    try:
        data = _load(path)
    except ValueError:
        return False
    return any(_is_ours(h, root) for g in (data.get("hooks") or {}).get(EVENT, []) for h in g.get("hooks", []))


# ---------------------------------------------------------------------------- MCP server
# Claude Code: a JSON file with a top-level "mcpServers" object. User scope is ~/.claude.json,
# project scope is .mcp.json in the project. Codex: a [mcp_servers.<name>] table in
# ~/.codex/config.toml. The kit's entry is found by the marker on its command line (JSON)
# or by a marked comment block (TOML), so uninstall never touches anyone else's servers.

MCP_NAME = "notes"


def mcp_args(root: Path, harness: str) -> list[str]:
    return [str(root / "kit.py"), "mcp-serve", "--harness", harness, HOOK_MARKER, kit_id(root)]


def _is_ours_mcp(entry: object, root: Path) -> bool:
    return isinstance(entry, dict) and HOOK_MARKER in (entry.get("args") or []) and kit_id(root) in (entry.get("args") or [])


def install_claude_mcp(path: Path, root: Path, python: str) -> dict:
    data = _load(path)
    servers = data.setdefault("mcpServers", {})
    if not isinstance(servers, dict):
        raise ValueError(f"{path}: `mcpServers` is not an object; leaving it alone")
    replaced = False
    for k in [k for k, v in servers.items() if _is_ours_mcp(v, root)]:
        del servers[k]
        replaced = True
    name = MCP_NAME if MCP_NAME not in servers else f"{MCP_NAME}-{kit_id(root)}"
    servers[name] = {"command": python, "args": mcp_args(root, "claude")}
    backup = _backup(path)
    _write(path, data)
    return {"name": name, "backup": str(backup) if backup else None, "replaced": replaced}


def uninstall_claude_mcp(path: Path, root: Path) -> dict:
    if not _real(path).exists():
        return {"removed": 0, "backup": None}
    data = _load(path)
    servers = data.get("mcpServers")
    gone = [k for k, v in servers.items() if _is_ours_mcp(v, root)] if isinstance(servers, dict) else []
    if not gone:
        return {"removed": 0, "backup": None}
    for k in gone:
        del servers[k]
    if not servers:
        del data["mcpServers"]
    backup = _backup(path)
    _write(path, data)
    return {"removed": len(gone), "backup": str(backup) if backup else None}


def _toml_str(x: str) -> str:
    return json.dumps(x)  # a JSON string is a valid TOML basic string


def _block_markers(root: Path) -> tuple[str, str]:
    return f"# >>> drag-kit {kit_id(root)} (managed by the kit, remove with: python3 kit.py uninstall) >>>", \
           f"# <<< drag-kit {kit_id(root)} <<<"


def _strip_block(text: str, root: Path) -> tuple[str, bool]:
    start, end = _block_markers(root)
    if start not in text:
        return text, False
    a = text.index(start)
    try:
        b = text.index(end, a) + len(end)
    except ValueError as exc:
        raise ValueError("the kit's marked block in this file has lost its end marker (edited by hand?). "
                         "Remove the block that starts with the '>>> drag-kit' comment yourself.") from exc
    return (text[:a].rstrip("\n") + ("\n" if text[:a].strip() else "") + text[b:].lstrip("\n")), True


def install_codex_mcp(path: Path, root: Path, python: str) -> dict:
    import tomllib
    path = _real(path)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if text.strip():
        try:
            tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(f"{path} is not valid TOML ({exc}); leaving it alone") from exc
    text, replaced = _strip_block(text, root)
    existing = (tomllib.loads(text).get("mcp_servers") or {}) if text.strip() else {}
    name = MCP_NAME if MCP_NAME not in existing else f"{MCP_NAME}-{kit_id(root)}"
    start, end = _block_markers(root)
    args = ", ".join(_toml_str(a) for a in mcp_args(root, "codex"))
    block = f"{start}\n[mcp_servers.{name}]\ncommand = {_toml_str(python)}\nargs = [{args}]\n{end}\n"
    new = (text.rstrip("\n") + "\n\n" if text.strip() else "") + block
    tomllib.loads(new)  # never write something that doesn't parse
    backup = _backup(path)
    atomic_write(path, new)
    return {"name": name, "backup": str(backup) if backup else None, "replaced": replaced}


def uninstall_codex_mcp(path: Path, root: Path) -> dict:
    import tomllib
    path = _real(path)
    if not path.exists():
        return {"removed": 0, "backup": None}
    text = path.read_text(encoding="utf-8")
    new, found = _strip_block(text, root)
    if not found:
        return {"removed": 0, "backup": None}
    if new.strip():
        tomllib.loads(new)
    backup = _backup(path)
    atomic_write(path, new)
    return {"removed": 1, "backup": str(backup) if backup else None}
