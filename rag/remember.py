"""File one fix, decision, preference or fact into the corpus, searchable in seconds.

    python3 kit.py remember problem "Build fails on Node 22" --solution "pin to 20"
    python3 kit.py remember decision "Use Postgres" --why "we need transactions"
    python3 kit.py remember preference "No meetings before 10"
    python3 kit.py remember fact "The invoice portal moved to portal.example.com"

The record is written as a normal OKF document, goes through the same linter gate as a
migrated note, and is indexed immediately (one embedding call), not at the next reindex.
"""
from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path

from . import index as idx
from . import lint as linter
from . import sections
from .config import Config
from .embed import Embedder
from .frontmatter import parse_text, render

# kind -> (type, bundle, scope)
KINDS = {
    "problem": ("Fix", "lessons", "shared"),
    "decision": ("Decision", "lessons", "shared"),
    "preference": ("Memory", "me", "me"),
    "fact": ("Reference", "reference", "shared"),
    "note": ("Reference", "reference", "shared"),
}
RESERVED_BUNDLES = {b for _, b, _ in KINDS.values()}


def ensure_bundle(cfg: Config, typ: str, bundle: str, scope: str) -> None:
    schema = cfg.corpus / "schema.yaml"
    bundles = linter.load_schema(schema) if schema.exists() else {}
    b = bundles.get(bundle)
    if b is None:
        bundles[bundle] = linter.Bundle(bundle, {typ}, scope if scope != "shared" else None, bundle == "lessons")
    elif typ not in b.types:
        b.types.add(typ)
    else:
        return
    schema.parent.mkdir(parents=True, exist_ok=True)
    linter.write_schema(schema, bundles)


def build_body(kind: str, text: str, why=None, solution=None, root_cause=None, rejected=None) -> str:
    """Labelled paragraphs with no headings, so one record is one retrievable section
    and a symptom never gets separated from its fix."""
    label = {"problem": "Problem", "decision": "Decision", "preference": "Preference", "fact": "Fact"}.get(kind)
    parts = [f"{label}: {text}" if label else text]
    for name, val in (("Root cause", root_cause), ("Solution", solution), ("Why", why), ("Rejected", rejected)):
        if val:
            parts.append(f"{name}: {val}")
    return "\n\n".join(parts)


def remember(cfg: Config, kind: str, text: str, embedder: Embedder, area: str | None = None, why=None,
             solution=None, root_cause=None, rejected=None, now: dt.datetime | None = None) -> dict:
    if not text or not text.strip():
        return {"ok": False, "problems": ["There is nothing to remember: the text is empty."]}
    typ, bundle, scope = KINDS[kind]
    now = now or dt.datetime.now(dt.timezone.utc)
    title = text.strip().splitlines()[0][:120]
    digest = hashlib.sha1(f"{kind}\n{text}\n{now.isoformat()}".encode()).hexdigest()[:8]
    body = build_body(kind, text.strip(), why, solution, root_cause, rejected)
    meta = {
        "type": typ, "id": f"rem-{now:%Y%m%d%H%M%S}-{digest}", "title": title,
        "description": " ".join(body.replace("#", " ").split())[:200],
        "tags": [t for t in (kind, area) if t], "status": "stable", "scope": scope,
        "generated": {"by": "remember", "at": now.isoformat(timespec="seconds")},
    }
    if area:
        meta["project"] = sections.slug(area, 40)
    body, _ = sections.with_anchors(body)
    ensure_bundle(cfg, typ, bundle, scope)
    from .migrate import write_ignore
    write_ignore(cfg.corpus)
    # Gate: lint the new document in place before it is written for real.
    doc = parse_text(render(meta, body), f"{bundle}/{meta['id']}.md")
    bundles = linter.load_schema(cfg.corpus / "schema.yaml")
    problems = [str(x) for x in linter.lint_document(doc, bundles)]
    if problems:
        return {"ok": False, "problems": problems}
    path = cfg.knowledge / bundle / f"{now:%Y-%m-%d}-{sections.slug(title, 50)}-{digest}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(meta, body), encoding="utf-8")
    linter.write_indexes(cfg.knowledge)
    rep = idx.reindex(cfg, embedder, only=[path, *[i for i in (path.parent / "index.md", cfg.knowledge / "index.md") if i.exists()]],
                      wait_s=20)
    return {"ok": True, "path": str(path), "id": meta["id"], "report": rep.summary(), "errors": rep.errors}
