"""Read and write the YAML block at the top of a corpus document.

Required: type, id, title. status defaults to stable and scope to shared.
Only title and description are written for a reader; the rest steers retrieval.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .constants import SCOPES, STATUSES, TYPES

DELIM = re.compile(r"^---\s*$", re.M)


class FrontmatterError(ValueError):
    pass


@dataclass
class Document:
    id: str
    path: str  # relative to knowledge/, POSIX style
    type: str
    title: str
    description: str
    status: str
    scope: str
    stale_after: dt.datetime | None
    project: str | None
    tags: list[str]
    aliases: list[str]
    meta: dict[str, Any]
    body: str
    file_hash: str

    @property
    def indexable(self) -> bool:
        """Deprecated documents and navigation files (rag_index: false) never enter the index."""
        return self.status != "deprecated" and self.meta.get("rag_index", True) is not False

    @property
    def source_resource(self) -> str:
        for s in self.meta.get("sources") or []:
            if isinstance(s, dict) and s.get("resource"):
                return str(s["resource"])
        return ""


def to_dt(value: Any) -> dt.datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)
    if isinstance(value, dt.date):
        return dt.datetime.combine(value, dt.time.min, dt.timezone.utc)
    try:
        d = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise FrontmatterError(f"stale_after {value!r} is not a date") from exc
    return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)


def _strs(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, int, float)):
        return [str(value)]
    return [str(v) for v in value]


def split(raw: str) -> tuple[str, str]:
    """(yaml text, body). Raises when there is no frontmatter block."""
    if not raw.lstrip("﻿").startswith("---"):
        raise FrontmatterError("no YAML frontmatter block")
    parts = DELIM.split(raw.lstrip("﻿"), maxsplit=2)
    if len(parts) < 3:
        raise FrontmatterError("frontmatter block is not closed")
    return parts[1], parts[2]


def parse_text(raw: str, rel_path: str) -> Document:
    yaml_text, body = split(raw)
    try:
        meta = yaml.safe_load(yaml_text) or {}
    except yaml.YAMLError as exc:
        raise FrontmatterError(f"invalid YAML: {str(exc).splitlines()[0]}") from exc
    if not isinstance(meta, dict):
        raise FrontmatterError("frontmatter is not a mapping")
    for key in ("type", "id", "title"):
        if not meta.get(key):
            raise FrontmatterError(f"missing required field `{key}`")
    if meta["type"] not in TYPES:
        raise FrontmatterError(f"unknown type {meta['type']!r}; known: {', '.join(TYPES)}")
    status = meta.get("status", "stable")
    if status not in STATUSES:
        raise FrontmatterError(f"status {status!r} not one of {STATUSES}")
    scope = meta.get("scope", "shared")
    if scope not in SCOPES:
        raise FrontmatterError(f"scope {scope!r} not one of {SCOPES}")
    return Document(
        id=str(meta["id"]), path=rel_path, type=meta["type"], title=str(meta["title"]),
        description=str(meta.get("description") or ""), status=status, scope=scope,
        stale_after=to_dt(meta.get("stale_after")),
        project=str(meta["project"]) if meta.get("project") else None,
        tags=_strs(meta.get("tags")), aliases=_strs(meta.get("aliases")),
        meta=meta, body=body.strip("\n"),
        file_hash=hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20],
    )


def parse(path: Path, knowledge: Path) -> Document:
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise FrontmatterError(f"unreadable: {exc}") from exc
    try:
        return parse_text(raw, path.relative_to(knowledge).as_posix())
    except FrontmatterError as exc:
        raise FrontmatterError(f"{path.relative_to(knowledge).as_posix()}: {exc}") from exc


def render(meta: dict[str, Any], body: str) -> str:
    """A document as text. Key order is kept as given; strings are quoted by YAML only when needed."""
    block = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True, width=100).rstrip("\n")
    return f"---\n{block}\n---\n\n{body.strip(chr(10))}\n"
