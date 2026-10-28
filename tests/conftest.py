"""Shared helpers. Every test builds its own kit root in a temp folder with the fake
embedder, so nothing touches the real ~/.claude, ~/.codex, Ollama, or the network."""
import sys
from pathlib import Path

import pytest

KIT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KIT))

from rag import config, embed, index, lint  # noqa: E402
from rag.frontmatter import render  # noqa: E402


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    r = tmp_path / "kit"
    r.mkdir()
    (r / "config.toml").write_text('[embeddings]\nprovider = "fake"\n')
    return r


@pytest.fixture
def cfg(root):
    return config.load(root)


def write_doc(cfg, bundle, name, title, body, **meta):
    m = {"type": "Reference", "id": name, "title": title, "description": title, "status": "stable",
         "scope": "shared"}
    m.update(meta)
    p = cfg.knowledge / bundle / f"{name}.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render(m, body), encoding="utf-8")
    return p


def finish(cfg):
    """Write schema + indexes and reindex with the fake embedder."""
    from rag import lint as L
    bundles = {}
    for p in cfg.knowledge.glob("*/*.md"):
        from rag.frontmatter import parse
        try:
            d = parse(p, cfg.knowledge)
        except Exception:
            continue
        b = bundles.setdefault(p.parent.name, L.Bundle(p.parent.name, set(), d.scope if d.scope == "me" else None, False))
        b.types.add(d.type)
    L.write_schema(cfg.corpus / "schema.yaml", bundles)
    L.write_indexes(cfg.knowledge)
    return index.reindex(cfg, embed.build(cfg))


def make_kit_copy(tmp_path, with_venv=True):
    """A throwaway copy of the kit to run `kit.py` commands against. `uninstall` deletes .venv and
    .state next to kit.py, and `install` writes .kit-id there, so these commands must never run
    against the real checkout. The copy's .venv is a symlink to the current interpreter."""
    import shutil
    kit = tmp_path / "kitcopy"
    shutil.copytree(KIT, kit, ignore=shutil.ignore_patterns(".venv", ".git", "__pycache__", ".pytest_cache", "okf", ".state",
                                                            ".kit-id", ".kit-installs.json"))
    if with_venv:
        (kit / ".venv" / "bin").mkdir(parents=True)
        (kit / ".venv" / "bin" / "python").symlink_to(sys.executable)
    return kit
