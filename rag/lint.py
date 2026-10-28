"""The corpus linter: the gate every document passes before it is indexed.

It checks required fields, known type, bundle and scope fit against schema.yaml,
duplicate ids, duplicate anchors, dead links, a deprecated note without a successor,
a missing index.md, depth, and things shaped like credentials. `fix_anchors` is the one
repair it makes itself: it adds a frozen `{#anchor}` to every heading that lacks one.
"""
from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import sections
from .frontmatter import FrontmatterError, Document, parse, render, split

MAX_DEPTH = 3  # folders below knowledge/
SUPERSEDED = "## Superseded by"
SECRET = re.compile(
    r"(?<![A-Za-z0-9])("
    r"sk-(?:ant|proj|live|test|svcacct)-[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9]{20,}"
    r"|sk_(?:live|test)_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,}"
    r"|github_pat_[A-Za-z0-9_]{30,}|xox[abp]-[A-Za-z0-9-]{10,}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----|eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}"
    r"|AIza[0-9A-Za-z_-]{30,})"
)
# Frozen anchors are kebab-case words; "ask-before-you-draft" would otherwise look like sk-...
ANCHOR_TAG = re.compile(r"\{#[a-z0-9][a-z0-9-]*\}")


@dataclass
class Problem:
    path: str
    rule: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.rule}: {self.message}"


@dataclass
class Bundle:
    name: str
    types: set[str]
    scope: str | None
    flat: bool


def load_schema(path: Path) -> dict[str, Bundle]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {n: Bundle(n, set(s.get("types") or []), s.get("scope"), bool(s.get("flat")))
            for n, s in (raw.get("bundles") or {}).items()}


def write_schema(path: Path, bundles: dict[str, Bundle]) -> None:
    out = {"version": 1, "bundles": {}}
    for n, b in sorted(bundles.items()):
        spec: dict = {"types": sorted(b.types)}
        if b.scope:
            spec["scope"] = b.scope
        if b.flat:
            spec["flat"] = True
        out["bundles"][n] = spec
    path.write_text(yaml.safe_dump(out, sort_keys=False), encoding="utf-8")


def rendered_text(doc: Document) -> str:
    """Everything the document will say: its frontmatter (title, description, tags, aliases and the rest)
    and its body. The secret check runs over all of it, not only the body."""
    return yaml.safe_dump(doc.meta, allow_unicode=True) + "\n" + doc.body


def lint_document(doc: Document, bundles: dict[str, Bundle]) -> list[Problem]:
    parts = Path(doc.path).parts
    out: list[Problem] = []
    is_index = parts[-1] == "index.md"
    bundle = bundles.get(parts[0]) if len(parts) > 1 else None
    if bundles and len(parts) > 1 and bundle is None:
        out.append(Problem(doc.path, "bundle", f"{parts[0]!r} is not a bundle in schema.yaml"))
    if bundle and not is_index:
        if doc.type not in bundle.types:
            out.append(Problem(doc.path, "bundle-type",
                               f"type {doc.type} does not belong in {bundle.name} {sorted(bundle.types)}"))
        if doc.scope != (bundle.scope or "shared"):
            out.append(Problem(doc.path, "bundle-scope",
                               f"scope {doc.scope} but {bundle.name} is {bundle.scope or 'shared'}"))
    if bundle and bundle.flat and len(parts) > 2:
        out.append(Problem(doc.path, "flat", f"{bundle.name} holds no subfolders"))
    if len(parts) > MAX_DEPTH + 1:
        out.append(Problem(doc.path, "depth", f"deeper than {MAX_DEPTH} folders below knowledge/"))
    if doc.status == "deprecated":
        after = doc.body.split(SUPERSEDED, 1)
        if len(after) < 2 or not sections.LINK.search(after[1]):
            out.append(Problem(doc.path, "successor", "deprecated without a link under '## Superseded by'"))
    if SECRET.search(ANCHOR_TAG.sub("", rendered_text(doc))):
        out.append(Problem(doc.path, "secret-shape",
                           "contains something shaped like a credential (in the title, description, tags, aliases or body)"))
    seen: set[str] = set()
    for h in sections.headings(doc.body):
        if h.anchor and h.anchor in seen:
            out.append(Problem(doc.path, "duplicate-anchor", f"anchor {{#{h.anchor}}} used twice"))
        if h.anchor:
            seen.add(h.anchor)
    return out


def lint_corpus(knowledge: Path, bundles: dict[str, Bundle]) -> list[Problem]:
    docs: dict[str, Document] = {}
    problems: list[Problem] = []
    ids: dict[str, str] = {}
    for path in sorted(knowledge.rglob("*.md")):
        try:
            doc = parse(path, knowledge)
        except FrontmatterError as exc:
            problems.append(Problem(path.relative_to(knowledge).as_posix(), "frontmatter", str(exc).split(": ", 1)[-1]))
            continue
        docs[doc.path] = doc
        if doc.id in ids:
            problems.append(Problem(doc.path, "duplicate-id", f"id {doc.id!r} also used by {ids[doc.id]}"))
        ids.setdefault(doc.id, doc.path)
        problems.extend(lint_document(doc, bundles))
    for folder in {knowledge, *(p.parent for p in knowledge.rglob("*.md"))}:
        if not (folder / "index.md").exists():
            problems.append(Problem(folder.relative_to(knowledge).as_posix() or ".", "index", "folder has no index.md"))
    anchors = {p: set(sections.resolve_anchors(sections.headings(d.body))) for p, d in docs.items()}
    for doc in docs.values():
        for target, line in sections.links(doc.body):
            path, _, frag = target.partition("#")
            dest = posixpath.normpath(posixpath.join(posixpath.dirname(doc.path), path)) if path else doc.path
            if dest not in docs:
                problems.append(Problem(doc.path, "dead-link", f"line {line}: {target} does not exist"))
            elif frag and frag not in anchors[dest]:
                problems.append(Problem(doc.path, "dead-anchor", f"line {line}: no {{#{frag}}} in {dest}"))
    return problems


def fix_anchors(knowledge: Path) -> int:
    """Add missing heading anchors in place; the YAML block is written back untouched."""
    added = 0
    for path in sorted(knowledge.rglob("*.md")):
        raw = path.read_text(encoding="utf-8")
        try:
            yaml_text, body = split(raw)
        except FrontmatterError:
            continue
        fixed, n = sections.with_anchors(body)
        if n:
            path.write_text(f"---{yaml_text}---{fixed}", encoding="utf-8")
            added += n
    return added


def write_indexes(knowledge: Path) -> int:
    """Give every folder an index.md (rag_index: false, never retrieved). Returns files written."""
    written = 0
    folders = {knowledge}
    for p in knowledge.rglob("*"):
        if p.is_dir():
            folders.add(p)
    for folder in sorted(folders):
        idx = folder / "index.md"
        if idx.exists():
            continue
        rel = folder.relative_to(knowledge).as_posix()
        name = "knowledge" if rel == "." else rel
        meta = {"type": "Reference", "id": "index-" + (sections.slug(name) or "root"),
                "title": f"{name} index", "description": f"What lives under {name}.",
                "status": "stable", "rag_index": False}
        body = f"# {name}\n\nNavigation only. This file is never retrieved.\n"
        idx.write_text(render(meta, body), encoding="utf-8")
        written += 1
    return written
