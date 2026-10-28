"""Settings: defaults live in config.toml next to kit.py; a .env file holds API keys."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

KIT_ROOT = Path(__file__).resolve().parent.parent


def load_env(root: Path) -> None:
    """Read KEY=value lines from <kit>/.env into os.environ (real env vars win)."""
    env = root / ".env"
    if not env.is_file():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if v.strip():
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


@dataclass
class Config:
    root: Path
    raw: dict[str, Any] = field(default_factory=dict)

    def section(self, name: str) -> dict[str, Any]:
        return self.raw.get(name, {})

    def get(self, section: str, key: str) -> Any:
        return self.raw[section][key]

    @property
    def corpus(self) -> Path:
        p = Path(self.get("corpus", "dir")).expanduser()
        return p if p.is_absolute() else self.root / p

    @property
    def knowledge(self) -> Path:
        return self.corpus / "knowledge"

    @property
    def state(self) -> Path:
        """Disposable files: the index, the vector cache, locks. Safe to delete."""
        return self.root / ".state"

    @property
    def db_path(self) -> Path:
        return self.state / "index.sqlite"

    def override(self, **retrieval: Any) -> "Config":
        raw = {k: dict(v) for k, v in self.raw.items()}
        raw["retrieval"].update(retrieval)
        return Config(self.root, raw)


def load(root: Path | None = None, path: Path | None = None) -> Config:
    root = Path(root or KIT_ROOT)
    load_env(root)
    default = tomllib.loads((Path(__file__).parent / "defaults.toml").read_text(encoding="utf-8"))
    cfg_path = path or root / "config.toml"
    user = tomllib.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
    merged = {k: dict(v) for k, v in default.items()}
    for sec, vals in user.items():
        merged.setdefault(sec, {}).update(vals)
    return Config(root, merged)
