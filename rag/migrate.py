"""Turn a vault of notes into an OKF corpus, without touching the vault.

    plan    copy the notes to okf/raw/notes/ (evidence, never indexed) and write a
            proposal for every note: bundle, type, id, title, description, tags.
            The proposal is plain JSON in okf/staging/. Rules, not a model, make the
            first guess; your agent reviews it and corrects it with `set`.
    apply   write okf/knowledge/ from the plan, run the linter as a gate, and only if
            it is clean move the result into place. A note that fails the gate (for
            example, one containing something shaped like an API key) stays out of
            the corpus and is listed.
    revert  delete exactly the files `apply` created, so the corpus is back to what
            it was. Your vault was never touched, so there is nothing to restore there.
"""
from __future__ import annotations

import datetime as dt
import fnmatch
import hashlib
import json
import posixpath
import re
import shutil
import urllib.parse
from pathlib import Path

import yaml

from . import lint as linter
from . import sections
from .config import Config
from .constants import TYPES
from .frontmatter import render

PERSONAL_BUNDLES = {"journal", "daily", "dailies", "diary", "personal", "private", "me", "health", "dreams"}
DATE_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}")
INLINE_TAG = re.compile(r"(?<![\w#/&])#([A-Za-z][\w/-]*)")
WIKILINK = re.compile(r"(!?)\[\[([^\]|\n]+?)(?:\|([^\]\n]*))?\]\]")
MDLINK = re.compile(r"(!?)\[([^\]\n]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")

# First-guess rules, tried in this order. Each looks at the path, title and tags.
TYPE_RULES = [
    ("Meeting", r"meeting|standup|stand-up|1[:-]1|retro|minutes|\bsync\b|call with"),
    ("Decision", r"decision|\badr\b|\brfc\b|we chose|why we"),
    ("Fix", r"\bfix\b|\bfixed\b|\bbug\b|error|troubleshoot|incident|post-?mortem|workaround|not working"),
    ("SOP", r"how[ -]to|setup|set up|guide|procedure|checklist|runbook|\bsop\b|playbook|workflow|step[- ]by"),
    ("Policy", r"policy|guidelines|principles|code of conduct|\brules\b"),
]


def _rel(p: Path, base: Path) -> str:
    return p.relative_to(base).as_posix()


def walk_notes(vault: Path):
    """Markdown files in the vault, skipping dot-folders (.obsidian, .git, .trash)."""
    for p in sorted(vault.rglob("*.md")):
        rel = p.relative_to(vault).parts
        if any(part.startswith(".") for part in rel[:-1]) or p.is_symlink():
            continue
        yield p


def split_note(raw: str) -> tuple[dict, str]:
    """(frontmatter dict, body). A malformed frontmatter block is left in the body."""
    m = re.match(r"﻿?---[ \t]*\n(.*?)\n---[ \t]*(?:\n|$)", raw, re.S)
    if not m:
        return {}, raw
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        return {}, raw
    return (meta, raw[m.end():]) if isinstance(meta, dict) else ({}, raw)


def _list(v) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [t.strip().lstrip("#") for t in re.split(r"[,\n]", v) if t.strip()]
    return [str(t).strip().lstrip("#") for t in v if str(t).strip()]


def first_paragraph(body: str, limit: int = 200) -> str:
    para: list[str] = []
    for _, line in sections.prose_lines(body):
        s = line.strip()
        if not s:
            if para:
                break
            continue
        if s.startswith(("#", "|", ">", "---", "- [", "![", "<")) and not para:
            continue
        para.append(s)
    text = " ".join(para)
    text = re.sub(r"\[\[([^\]|]+)\|?([^\]]*)\]\]", lambda m: m.group(2) or m.group(1), text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[*_`]+", "", text).strip()
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "..."
    return text


def guess_type(rel: str, title: str, tags: list[str], bundle: str, body: str) -> tuple[str, list[str]]:
    hay = f"{rel} {title} {' '.join(tags)}".lower()
    hits = [t for t, rx in TYPE_RULES if re.search(rx, hay)]
    stem = Path(rel).stem.lower()
    if bundle in PERSONAL_BUNDLES or DATE_NAME.match(stem):
        if "Meeting" not in hits:
            return "Memory", []
    if hits:
        return hits[0], hits[1:]
    if stem in {"readme", "overview", "index", "brief", "spec", "plan"} and bundle not in PERSONAL_BUNDLES:
        return ("Client" if bundle in {"clients", "customers"} else "Project"), []
    return "Reference", []


def make_id(rel: str) -> str:
    stem = sections.slug(Path(rel).stem, 40) or "note"
    return f"{stem}-{hashlib.sha1(rel.encode()).hexdigest()[:6]}"


def inline_tags(body: str) -> list[str]:
    out = []
    for _, line in sections.prose_lines(body):
        if line.lstrip().startswith("#") and re.match(r"\s*#{1,6}\s", line):
            continue
        out += INLINE_TAG.findall(line)
    return out


def propose(rel: str, raw: str, mtime: float) -> dict:
    fm, body = split_note(raw)
    parts = Path(rel).parts
    bundle = sections.slug(parts[0], 40) if len(parts) > 1 else "notes"
    bundle = bundle or "notes"
    m = re.search(r"^#\s+(.+?)\s*$", body, re.M)
    title = str(fm.get("title") or (m.group(1) if m else "") or Path(rel).stem).strip()
    title = re.sub(r"\s*\{#[a-z0-9-]+\}$", "", title)
    tags = list(dict.fromkeys(_list(fm.get("tags")) + _list(fm.get("tag")) + inline_tags(body)))[:12]
    typ, others = guess_type(rel, title, tags, bundle, body)
    words = len(re.findall(r"\w+", body))
    flags = []
    if words == 0:
        flags.append("empty")
    if others:
        flags.append("type-conflict: also matches " + ", ".join(others))
    if len(parts) == 1:
        flags.append("root-level note, placed in 'notes'")
    return {
        "src": rel, "bundle": bundle, "type": typ, "id": make_id(rel), "title": title,
        "description": str(fm.get("description") or fm.get("summary") or "").strip() or first_paragraph(body),
        "tags": tags, "aliases": _list(fm.get("aliases")) + _list(fm.get("alias")),
        "project": sections.slug(parts[1], 40) if len(parts) > 2 else None,
        "scope": "me" if bundle in PERSONAL_BUNDLES else "shared",
        "words": words, "modified": dt.datetime.fromtimestamp(mtime, dt.timezone.utc).date().isoformat(),
        "flags": flags, "include": words > 0,
    }


def schema_from_plan(entries: list[dict]) -> dict[str, linter.Bundle]:
    bundles: dict[str, linter.Bundle] = {}
    for e in entries:
        if not e.get("include", True):
            continue
        b = bundles.setdefault(e["bundle"], linter.Bundle(e["bundle"], set(), e["scope"], False))
        b.types.add(e["type"])
    return bundles


IGNORE_TEXT = """# Written by the kit. ripgrep (which Claude Code's Grep and Glob use) and other tools that
# honor .ignore skip these folders, so an agent searching okf/ sees each note once, in
# knowledge/, instead of also finding the raw copies, the plan and the eval answer key.
raw/
staging/
eval/
reports/
.build/
migration.json
"""


MARKER = ".kit-corpus"
# The only things the kit ever creates inside the corpus folder. Removal deletes these and nothing else.
KNOWN_CHILDREN = ("knowledge", "raw", "staging", "eval", "reports", ".build", "schema.yaml", "migration.json",
                  ".ignore", MARKER)


def write_ignore(corpus: Path, vault: Path | None = None) -> None:
    """Keep evidence and bookkeeping out of search without moving it, and leave a marker saying
    this folder belongs to the kit (removal refuses folders without it). Idempotent; a .ignore
    you have edited is left alone."""
    corpus.mkdir(parents=True, exist_ok=True)
    f = corpus / ".ignore"
    if not f.exists():
        f.write_text(IGNORE_TEXT, encoding="utf-8")
    m = corpus / MARKER
    if not m.exists() or vault is not None:
        old = {}
        if m.exists():
            try:
                old = json.loads(m.read_text(encoding="utf-8"))
            except ValueError:
                old = {}
        old.update({"kit": "deterministic-rag-kit", "note": "Written by the kit. `kit.py uninstall --remove-corpus` only deletes folders that contain this file."})
        if vault is not None:
            old["vault"] = str(vault)
        m.write_text(json.dumps(old, indent=1), encoding="utf-8")


def unsafe_corpus_reason(corpus: Path, vault: Path | None) -> str | None:
    """Why this folder must not be the corpus, or None. The corpus is a folder the kit owns and
    may delete from, so it must be separate from your notes and must not be a place like your
    home folder."""
    corpus = corpus.resolve()
    home = Path.home().resolve()
    if corpus == home or corpus == Path(corpus.anchor) or len(corpus.parts) < 4:
        return f"{corpus} is too broad to be the corpus folder"
    if vault is not None:
        vault = vault.resolve()
        if corpus == vault:
            return "the corpus folder is the same folder as your notes"
        if vault in corpus.parents:
            return "the corpus folder is inside your notes folder"
        if corpus in vault.parents:
            return "the corpus folder contains your notes folder"
    return None


def check_removable(corpus: Path) -> Path:
    """Every reason to refuse deleting the corpus folder, checked before anything is deleted."""
    corpus = corpus.resolve()
    marker = corpus / MARKER
    if not marker.is_file():
        raise SystemExit(f"{corpus} has no {MARKER} file, so the kit won't delete it. If it really is the corpus, "
                         "remove its knowledge/, raw/ and staging/ folders yourself. Nothing was changed.")
    why = unsafe_corpus_reason(corpus, None)
    if why:
        raise SystemExit(f"Refusing to remove: {why}. Nothing was changed.")
    vaults = []
    try:
        vaults.append(json.loads(marker.read_text(encoding="utf-8")).get("vault"))
    except ValueError:
        pass
    for f in (corpus / "migration.json", corpus / "staging" / "migration-plan.json"):
        try:
            vaults.append(json.loads(f.read_text(encoding="utf-8")).get("vault"))
        except (OSError, ValueError):
            pass
    for v in filter(None, vaults):
        vp = Path(v).resolve()
        if vp == corpus or corpus in vp.parents or vp in corpus.parents:
            raise SystemExit(f"Refusing to remove: {corpus} overlaps your notes folder {vp}. Nothing was changed.")
    return corpus


def remove_corpus(corpus: Path) -> list[str]:
    """Delete the kit's own files from the corpus folder, then the folder itself only if that
    leaves it empty. Refuses (see check_removable) before deleting anything."""
    corpus = check_removable(corpus)
    removed = []
    for name in KNOWN_CHILDREN:
        child = corpus / name
        if child.is_symlink() or child.is_file():
            child.unlink()
            removed.append(name)
        elif child.is_dir():
            shutil.rmtree(child)
            removed.append(name)
    try:
        corpus.rmdir()
        removed.append(".")
    except OSError:
        pass  # something of yours is still in there; it stays
    return removed


def paths(cfg: Config) -> dict[str, Path]:
    c = cfg.corpus
    return {"corpus": c, "raw": c / "raw" / "notes", "staging": c / "staging",
            "plan": c / "staging" / "migration-plan.json", "manifest": c / "migration.json",
            "schema": c / "schema.yaml", "build": c / ".build"}


SAFE_NAME = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def load_plan(cfg: Config) -> dict:
    p = paths(cfg)["plan"]
    if not p.exists():
        raise SystemExit("No migration plan yet. Run: python3 kit.py migrate plan <your-vault>")
    plan = json.loads(p.read_text(encoding="utf-8"))
    # The plan is a file an agent or a person edits. Check the fields that become file paths.
    for e in plan.get("entries", []):
        if not SAFE_NAME.match(str(e.get("bundle", ""))):
            raise SystemExit(f"The plan has an unsafe bundle name {e.get('bundle')!r}; bundles are lowercase letters, "
                             "digits and hyphens.")
        if not re.fullmatch(r"[A-Za-z0-9._-]+", str(e.get("id", ""))):
            raise SystemExit(f"The plan has an unsafe id {e.get('id')!r}.")
        parts = Path(str(e.get("src", ""))).parts
        if not parts or ".." in parts or Path(str(e["src"])).is_absolute():
            raise SystemExit(f"The plan has an unsafe source path {e.get('src')!r}.")
    return plan


def save_plan(cfg: Config, plan: dict) -> None:
    p = paths(cfg)
    p["staging"].mkdir(parents=True, exist_ok=True)
    p["plan"].write_text(json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8")
    bundles = schema_from_plan(plan["entries"])
    if p["schema"].exists():  # keep the bundles `remember` created (lessons, me, reference)
        from .remember import RESERVED_BUNDLES
        for name, old in linter.load_schema(p["schema"]).items():
            if name in RESERVED_BUNDLES:
                cur = bundles.setdefault(name, old)
                cur.types |= old.types
    linter.write_schema(p["schema"], bundles)


def plan_vault(cfg: Config, vault: Path) -> dict:
    vault = vault.expanduser().resolve()
    p = paths(cfg)
    if not vault.is_dir():
        raise SystemExit(f"{vault} is not a folder")
    corpus = p["corpus"].resolve()
    why = unsafe_corpus_reason(corpus, vault)
    if why:
        raise SystemExit(f"Can't use {corpus} as the corpus folder: {why}. Pick a new, empty folder that is "
                         "separate from your notes (not inside it, and not a folder that contains it) and set "
                         "[corpus] dir in config.toml. The kit creates and deletes files there.")
    if corpus.exists() and any(corpus.iterdir()) and not (corpus / MARKER).exists():
        raise SystemExit(f"{corpus} already holds files that aren't the kit's. Use a new, empty folder for the "
                         "corpus, so the kit never writes into or deletes from a folder you use for something else.")
    entries, seen_ids = [], {}
    p["raw"].mkdir(parents=True, exist_ok=True)
    write_ignore(p["corpus"], vault)
    if not any(True for _ in walk_notes(vault)):
        raise SystemExit(f"No markdown notes found in {vault}. Check the path.")
    for note in walk_notes(vault):
        rel = _rel(note, vault)
        try:
            raw = note.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            entries.append({"src": rel, "include": False, "flags": ["unreadable"], "bundle": "notes",
                            "type": "Reference", "id": make_id(rel), "title": rel, "scope": "shared",
                            "words": 0, "tags": [], "description": ""})
            continue
        dest = p["raw"] / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(note, dest)
        e = propose(rel, raw, note.stat().st_mtime)
        if e["id"] in seen_ids:  # astronomically unlikely, but ids must be unique
            e["id"] = e["id"] + "-2"
        seen_ids[e["id"]] = rel
        entries.append(e)
    plan = {"version": 1, "vault": str(vault), "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "entries": entries}
    save_plan(cfg, plan)
    return plan


def summarize(plan: dict) -> str:
    es = plan["entries"]
    inc = [e for e in es if e.get("include", True)]
    out = [f"{len(es)} notes planned from {plan['vault']} ({len(inc)} included)"]
    by_bundle: dict[str, dict[str, int]] = {}
    for e in inc:
        by_bundle.setdefault(e["bundle"], {}).setdefault(e["type"], 0)
        by_bundle[e["bundle"]][e["type"]] += 1
    out.append("\nbundle (scope)            notes  types")
    for b, types in sorted(by_bundle.items(), key=lambda kv: -sum(kv[1].values())):
        scope = next(e["scope"] for e in inc if e["bundle"] == b)
        out.append(f"{b + ' (' + scope + ')':<25} {sum(types.values()):>5}  " +
                   ", ".join(f"{t} {n}" for t, n in sorted(types.items(), key=lambda kv: -kv[1])))
    flagged = [e for e in es if e.get("flags")]
    out.append(f"\n{len(flagged)} notes carry a flag worth a look (showing up to 15):")
    for e in flagged[:15]:
        out.append(f"  {e['src']} -> {e['bundle']}/{e['type']}  [{'; '.join(e['flags'])}]")
    return "\n".join(out)


def set_entries(cfg: Config, pattern: str, bundle: str | None = None, typ: str | None = None,
                scope: str | None = None, include: bool | None = None) -> int:
    """Edit the plan for every note whose vault path matches the glob (or folder prefix)."""
    if typ and typ not in TYPES:
        raise SystemExit(f"type must be one of {', '.join(TYPES)}")
    plan = load_plan(cfg)
    n = 0
    for e in plan["entries"]:
        if fnmatch.fnmatch(e["src"], pattern) or e["src"].startswith(pattern.rstrip("/") + "/"):
            if bundle:
                e["bundle"] = sections.slug(bundle, 40)
            if typ:
                e["type"] = typ
            if scope:
                e["scope"] = scope
            if include is not None:
                e["include"] = include
            e["flags"] = [f for f in e.get("flags", []) if not f.startswith("type-conflict")]
            n += 1
    save_plan(cfg, plan)
    return n


# ---------------------------------------------------------------- body conversion

class LinkMap:
    """Where each vault note ends up, so links can be rewritten."""

    def __init__(self, entries):
        self.by_rel = {e["src"]: e for e in entries if e.get("include", True)}
        self.by_name: dict[str, list[dict]] = {}
        for e in self.by_rel.values():
            self.by_name.setdefault(Path(e["src"]).stem.lower(), []).append(e)

    def dest(self, e) -> str:
        return f"{e['bundle']}/{e['id']}.md"

    def resolve_wiki(self, target: str):
        t = target.strip().split("#")[0].strip()
        if not t:
            return None
        if t.lower().endswith(".md"):
            t = t[:-3]
        hit = self.by_rel.get(t + ".md")
        if hit:
            return hit
        cands = self.by_name.get(posixpath.basename(t).lower(), [])
        if len(cands) == 1:
            return cands[0]
        if len(cands) > 1 and "/" in t:
            tail = ("/" + t.lower() + ".md")
            m = [c for c in cands if ("/" + c["src"].lower()).endswith(tail)]
            return m[0] if len(m) == 1 else None
        return None

    def resolve_rel(self, src: str, target: str):
        t = urllib.parse.unquote(target.split("#")[0])
        if not t:
            return None
        return self.by_rel.get(posixpath.normpath(posixpath.join(posixpath.dirname(src), t)))


def convert_body(entry: dict, body: str, lm: LinkMap) -> str:
    """Obsidian links become ordinary relative markdown links where the target is a
    note in this corpus, and plain text where it is not. Code fences are left alone."""
    here = lm.dest(entry)

    def link_to(target_entry, label):
        rel = posixpath.relpath(lm.dest(target_entry), posixpath.dirname(here))
        return f"[{label}]({urllib.parse.quote(rel)})"

    def wiki(m):
        embed, target, alias = m.group(1), m.group(2), m.group(3)
        label = (alias or "").strip() or target.split("#")[-1].strip() if target.startswith("#") else (alias or "").strip() or target.split("#")[0].strip()
        e = lm.resolve_wiki(target)
        if e is not None and e["src"] != entry["src"]:
            return link_to(e, label)
        return "" if embed else label

    def md(m):
        embed, label, target = m.group(1), m.group(2), m.group(3)
        if re.match(r"^(https?:|mailto:|tel:)", target) or target.startswith("#"):
            return m.group(0) if not embed else m.group(0)
        e = lm.resolve_rel(entry["src"], target)
        if e is not None and e["src"] != entry["src"]:
            return link_to(e, label or e["title"])
        return "" if embed else label

    out, fenced = [], False
    for line in body.split("\n"):
        if sections.FENCE.match(line):
            fenced = not fenced
            out.append(line)
            continue
        if not fenced:
            line = MDLINK.sub(md, line)  # first, so links written by the wikilink pass are not re-read
            line = WIKILINK.sub(wiki, line)
        out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------------- apply / revert

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:20]


def apply_plan(cfg: Config) -> dict:
    p = paths(cfg)
    plan = load_plan(cfg)
    if p["manifest"].exists():
        revert(cfg, keep_plan=True, quiet=True)
    entries = plan["entries"]
    lm = LinkMap(entries)
    build_k = p["build"] / "knowledge"
    if p["build"].exists():
        shutil.rmtree(p["build"])
    build_k.mkdir(parents=True)
    quarantined, written = [], []
    for e in entries:
        if not e.get("include", True):
            continue
        raw = (p["raw"] / e["src"]).read_text(encoding="utf-8")
        fm, body = split_note(raw)
        body = convert_body(e, body.strip("\n"), lm)
        body, _ = sections.with_anchors(body)
        meta = {"type": e["type"], "id": e["id"], "title": e["title"], "description": e["description"],
                "status": "stable", "scope": e["scope"]}
        if e.get("project"):
            meta["project"] = e["project"]
        if e.get("tags"):
            meta["tags"] = e["tags"]
        if e.get("aliases"):
            meta["aliases"] = e["aliases"]
        meta["modified"] = e["modified"]
        meta["generated"] = {"by": "import:migrate", "at": plan["created"]}
        meta["sources"] = [{"id": "original", "resource": f"raw/notes/{e['src']}",
                            "hash": hashlib.sha256((p["raw"] / e["src"]).read_bytes()).hexdigest()[:16]}]
        document = render(meta, body)
        if linter.SECRET.search(linter.ANCHOR_TAG.sub("", document)):  # the whole document, frontmatter included
            quarantined.append({"src": e["src"], "reason": "contains something shaped like a credential"})
            continue
        dest = build_k / e["bundle"] / f"{e['id']}.md"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(document, encoding="utf-8")
        written.append(dest)
    linter.write_indexes(build_k)
    bundles = linter.load_schema(p["schema"])
    problems = linter.lint_corpus(build_k, bundles)
    if problems:
        return {"ok": False, "problems": [str(x) for x in problems], "quarantined": quarantined,
                "build": str(p["build"])}
    created = []
    for f in sorted(build_k.rglob("*.md")):
        rel = f.relative_to(build_k)
        dest = cfg.knowledge / rel
        if f.name == "index.md" and dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(f), dest)
        created.append({"path": rel.as_posix(), "sha": _sha(dest)})
    shutil.rmtree(p["build"], ignore_errors=True)
    p["manifest"].write_text(json.dumps({"vault": plan["vault"], "created": created,
                                         "quarantined": quarantined}, indent=1), encoding="utf-8")
    return {"ok": True, "written": len(written), "files": len(created), "quarantined": quarantined}


def revert(cfg: Config, keep_plan: bool = False, quiet: bool = False) -> dict:
    """Delete exactly what `apply` created. A file you edited since is kept and listed."""
    p = paths(cfg)
    if not p["manifest"].exists():
        return {"removed": 0, "kept": [], "note": "nothing to revert"}
    man = json.loads(p["manifest"].read_text(encoding="utf-8"))
    removed, kept = 0, []
    for f in man["created"]:
        path = cfg.knowledge / f["path"]
        if not path.exists():
            continue
        if _sha(path) == f["sha"]:
            path.unlink()
            removed += 1
        else:
            kept.append(f["path"])
    for d in sorted((x for x in cfg.knowledge.rglob("*") if x.is_dir()), reverse=True):
        try:
            d.rmdir()
        except OSError:
            pass
    p["manifest"].unlink()
    if not keep_plan:
        shutil.rmtree(p["raw"], ignore_errors=True)
        for f in (p["plan"],):
            f.unlink(missing_ok=True)
    return {"removed": removed, "kept": kept}
