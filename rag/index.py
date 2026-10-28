"""The index: one SQLite file (documents, sections, an FTS5 table) plus a numpy vector cache.

Files in knowledge/ are the truth; everything in .state/ can be deleted and rebuilt.
Reindexing is incremental: a file is re-read only when its modification time or size
changed, and re-chunked only when its content hash changed. A section whose text did
not change keeps its stored embedding, so editing one note costs one embedding call.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import sections
from .config import Config
from .embed import EmbedError, Embedder, build as build_embedder
from .frontmatter import FrontmatterError, parse, parse_text

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY, path TEXT UNIQUE NOT NULL, type TEXT, title TEXT, title_norm TEXT,
  description TEXT, status TEXT, scope TEXT, stale_after REAL, project TEXT, tags TEXT,
  aliases TEXT, source TEXT, file_hash TEXT, mtime_ns INTEGER, size INTEGER, indexable INTEGER
);
CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT, doc_id TEXT NOT NULL, ord INTEGER, section_id TEXT,
  heading TEXT, heading_path TEXT, text TEXT, tokens INTEGER, text_hash TEXT, embedding BLOB,
  start_line INTEGER, end_line INTEGER
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS chunks_hash ON chunks(text_hash);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  title, heading, text, tokenize='porter unicode61');
"""

def connect(cfg: Config, create: bool = True) -> sqlite3.Connection:
    cfg.state.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(cfg.db_path, timeout=15)
    db.row_factory = sqlite3.Row
    if create:
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript(SCHEMA)
    return db


def get_meta(db, key, default=None):
    row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def set_meta(db, key, value):
    db.execute("INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
               (key, str(value)))


def embed_input(title: str, sec: sections.Section) -> str:
    """What gets embedded: the passage with its note title and heading path, so a
    section that says "it" still carries its subject."""
    head = f"{title} > {sec.heading_path}" if sec.heading_path else title
    return f"{head}\n\n{sec.text}"


def title_norm(text: str) -> str:
    return " ".join(sections.slug(text, 200).split("-"))


@dataclass
class Report:
    files: int = 0
    skipped: int = 0
    updated: int = 0
    removed: int = 0
    chunks_total: int = 0
    embedded: int = 0
    pending_embeddings: int = 0
    errors: list[str] = field(default_factory=list)
    seconds: float = 0.0

    def summary(self) -> str:
        s = (f"{self.files} files: {self.updated} updated, {self.skipped} unchanged, "
             f"{self.removed} removed; {self.chunks_total} sections, {self.embedded} embedded "
             f"this run, {self.pending_embeddings} still waiting for a vector; "
             f"{len(self.errors)} problems; {self.seconds:.1f}s")
        return s


def _pid_alive(pid: int) -> bool:
    """Is a process with this id running? Where that can't be asked (Windows without extras), say yes
    and let the age check decide."""
    if pid <= 0:
        return False
    if os.name == "nt":
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _lock_owner(path: Path) -> int:
    try:
        return int(path.read_text().strip() or 0)
    except (OSError, ValueError):
        return 0


@contextlib.contextmanager
def lock(cfg: Config, wait_s: float = 0.0):
    """One writer at a time. The lock file holds the owner's process id; a lock whose owner is gone is
    taken over at once, and a live owner keeps it (the owner touches the file as it works, and a lock
    untouched for 30 minutes is treated as dead whatever its pid says, for pid reuse)."""
    cfg.state.mkdir(parents=True, exist_ok=True)
    path = cfg.state / "reindex.lock"
    me = os.getpid()
    deadline = time.time() + wait_s
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(me).encode())
            os.close(fd)
            break
        except FileExistsError:
            owner = _lock_owner(path)
            try:
                stale = time.time() - path.stat().st_mtime > 1800
            except OSError:
                continue  # it vanished between our two looks; try again
            if stale or not _pid_alive(owner):
                try:
                    path.unlink()
                except OSError:
                    pass
                continue
            if time.time() >= deadline:
                raise BlockingIOError(f"another reindex is running (process {owner})")
            time.sleep(0.2)
    try:
        yield
    finally:
        if _lock_owner(path) == me:  # never delete a lock someone else took over
            with contextlib.suppress(OSError):
                path.unlink()


def touch_lock(cfg: Config) -> None:
    with contextlib.suppress(OSError):
        os.utime(cfg.state / "reindex.lock")


def _store_document(db, doc, mtime_ns, size, secs, vec_by_hash, cfg_title):
    db.execute("DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE doc_id=?)", (doc.id,))
    db.execute("DELETE FROM chunks WHERE doc_id=?", (doc.id,))
    stale = doc.stale_after.timestamp() if doc.stale_after else None
    db.execute(
        """INSERT INTO documents(id,path,type,title,title_norm,description,status,scope,stale_after,
             project,tags,aliases,source,file_hash,mtime_ns,size,indexable)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(id) DO UPDATE SET path=excluded.path,type=excluded.type,title=excluded.title,
             title_norm=excluded.title_norm,description=excluded.description,status=excluded.status,
             scope=excluded.scope,stale_after=excluded.stale_after,project=excluded.project,
             tags=excluded.tags,aliases=excluded.aliases,source=excluded.source,
             file_hash=excluded.file_hash,mtime_ns=excluded.mtime_ns,size=excluded.size,
             indexable=excluded.indexable""",
        (doc.id, doc.path, doc.type, doc.title, title_norm(doc.title), doc.description, doc.status,
         doc.scope, stale, doc.project, json.dumps(doc.tags), json.dumps(doc.aliases),
         doc.source_resource, doc.file_hash, mtime_ns, size, int(doc.indexable)))
    if not doc.indexable:
        return 0
    n = 0
    for s in secs:
        text_hash = hashlib.sha1(embed_input(doc.title, s).encode("utf-8")).hexdigest()
        cur = db.execute(
            """INSERT INTO chunks(doc_id,ord,section_id,heading,heading_path,text,tokens,text_hash,
                 embedding,start_line,end_line) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (doc.id, s.ord, s.section_id, s.heading, s.heading_path, s.text, s.tokens, text_hash,
             vec_by_hash.get(text_hash), s.start_line, s.end_line))
        db.execute("INSERT INTO chunks_fts(rowid,title,heading,text) VALUES(?,?,?,?)",
                   (cur.lastrowid, doc.title, s.heading_path or s.heading, s.text))
        n += 1
    return n


def _drop_document(db, doc_id):
    db.execute("DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE doc_id=?)", (doc_id,))
    db.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
    db.execute("DELETE FROM documents WHERE id=?", (doc_id,))


def reindex(cfg: Config, embedder: Embedder | None = None, force: bool = False,
            only: list[Path] | None = None, wait_s: float = 0.0, embed: bool = True, progress=None) -> Report:
    """Bring the index in line with knowledge/. `only` limits the scan to those files
    (used by `remember`, so a new note is searchable in seconds)."""
    t0 = time.time()
    embedder = embedder or build_embedder(cfg)
    rep = Report()
    knowledge = cfg.knowledge
    max_tokens = cfg.get("chunking", "max_section_tokens")
    with lock(cfg, wait_s):
        db = connect(cfg)
        dirty_vectors = False
        sig = embedder.signature
        if get_meta(db, "embedder") not in (None, sig):
            db.execute("UPDATE chunks SET embedding=NULL")
            dirty_vectors = True
        set_meta(db, "embedder", sig)
        existing = {r["path"]: r for r in db.execute(
            "SELECT id,path,file_hash,mtime_ns,size FROM documents")}
        if only is not None:
            files = [p for p in only if p.is_file()]
        else:
            files = sorted(knowledge.rglob("*.md")) if knowledge.is_dir() else []
        seen_paths = set()
        stored_ids: set[str] = set()
        on_disk = {p.relative_to(knowledge).as_posix() for p in files} | (set(existing) if only is not None else set())
        seen_ids = {r["id"]: r["path"] for r in existing.values()}
        for path in files:
            rel = path.relative_to(knowledge).as_posix()
            seen_paths.add(rel)
            rep.files += 1
            st = path.stat()
            old = existing.get(rel)
            if old and not force and old["mtime_ns"] == st.st_mtime_ns and old["size"] == st.st_size:
                rep.skipped += 1
                continue
            try:
                raw = path.read_text(encoding="utf-8")
                doc = parse_text(raw, rel)
            except (FrontmatterError, OSError, UnicodeDecodeError) as exc:
                rep.errors.append(f"{rel}: {exc}")
                continue
            if old and not force and old["file_hash"] == doc.file_hash:
                db.execute("UPDATE documents SET mtime_ns=?, size=? WHERE path=?", (st.st_mtime_ns, st.st_size, rel))
                rep.skipped += 1
                continue
            owner = seen_ids.get(doc.id)
            if owner and owner != rel and owner in on_disk:
                rep.errors.append(f"{rel}: duplicate id {doc.id!r} (also {owner}); not indexed")
                continue
            seen_ids[doc.id] = rel
            secs = sections.split_sections(doc.id, doc.body, max_tokens) if doc.indexable else []
            vec_by_hash = {r["text_hash"]: r["embedding"] for r in db.execute(
                "SELECT text_hash, embedding FROM chunks WHERE doc_id=? AND embedding IS NOT NULL", (doc.id,))}
            for sec in secs:  # a section moved from another note (or another id) keeps its vector too
                h = hashlib.sha1(embed_input(doc.title, sec).encode("utf-8")).hexdigest()
                if h not in vec_by_hash:
                    hit = db.execute("SELECT embedding FROM chunks WHERE text_hash=? AND embedding IS NOT NULL LIMIT 1", (h,)).fetchone()
                    if hit:
                        vec_by_hash[h] = hit[0]
            if old and old["id"] != doc.id:  # the file's id changed
                _drop_document(db, old["id"])
            _store_document(db, doc, st.st_mtime_ns, st.st_size, secs, vec_by_hash, doc.title)
            stored_ids.add(doc.id)
            rep.updated += 1
            dirty_vectors = True
        if only is None:
            for rel, row in existing.items():
                if rel not in seen_paths and row["id"] not in stored_ids:
                    _drop_document(db, row["id"])
                    rep.removed += 1
                    dirty_vectors = True
        db.commit()
        if embed and embedder.name != "none":
            n, errs = _embed_pending(db, embedder, cfg.get("embeddings", "batch_size"),
                                     cfg.get("embeddings", "concurrency"),
                                     progress=lambda d, t: (touch_lock(cfg), progress and progress(d, t)))
            rep.embedded = n
            rep.errors.extend(errs)
            dirty_vectors = dirty_vectors or n > 0
        if dirty_vectors or not (cfg.state / "vectors.npy").exists():
            build_vector_cache(cfg, db)
        rep.chunks_total = db.execute("SELECT count(*) FROM chunks").fetchone()[0]
        rep.pending_embeddings = db.execute("SELECT count(*) FROM chunks WHERE embedding IS NULL").fetchone()[0] \
            if embedder.name != "none" else 0
        set_meta(db, "last_reindex", time.time())
        db.commit()
        db.close()
    rep.seconds = time.time() - t0
    return rep


def _embed_pending(db, embedder, batch: int = 32, concurrency: int = 2, progress=None) -> tuple[int, list[str]]:
    """Embed every section that has no vector yet: batched requests, a few in flight at once.
    Only this thread touches the database; worker threads just call the embedding service.
    The first failed batch stops new batches from being sent; batches already finished are kept."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    rows = db.execute(
        """SELECT c.id, d.title, c.heading_path, c.text FROM chunks c JOIN documents d ON d.id=c.doc_id
           WHERE c.embedding IS NULL ORDER BY c.id""").fetchall()
    batch, concurrency = max(1, int(batch)), max(1, int(concurrency))
    parts = [rows[i:i + batch] for i in range(0, len(rows), batch)]
    done, errors, stop = 0, [], False

    def work(part):
        if stop:
            return part, None
        texts = [embed_input(r["title"], sections.Section("", "", r["heading_path"], r["text"], 0, 0, 0, 0))
                 for r in part]
        return part, embedder.embed(texts)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(work, p) for p in parts]
        for fut in as_completed(futures):
            try:
                part, vecs = fut.result()
            except EmbedError as exc:
                if not errors:
                    errors.append(f"embedding stopped after {done} sections: {exc}")
                stop = True
                continue
            if vecs is None:
                continue
            db.executemany("UPDATE chunks SET embedding=? WHERE id=?",
                           [(np.asarray(v, dtype=np.float32).tobytes(), r["id"]) for v, r in zip(vecs, part)])
            db.commit()
            done += len(part)
            if progress:
                progress(done, len(rows))
    return done, errors


def build_vector_cache(cfg: Config, db) -> int:
    """Write all stored embeddings as one normalized float32 matrix, atomically.
    Queries memory-map it, so the hook never loads blobs from SQLite."""
    n = db.execute("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL").fetchone()[0]
    ids_path, vec_path = cfg.state / "vector_ids.npy", cfg.state / "vectors.npy"
    if n == 0:
        for p in (ids_path, vec_path):
            with contextlib.suppress(OSError):
                p.unlink()
        return 0
    first = db.execute("SELECT embedding FROM chunks WHERE embedding IS NOT NULL LIMIT 1").fetchone()[0]
    dim = len(first) // 4
    mat = np.empty((n, dim), dtype=np.float32)
    ids = np.empty(n, dtype=np.int64)
    i = 0
    for r in db.execute("SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL ORDER BY id"):
        v = np.frombuffer(r["embedding"], dtype=np.float32)
        if len(v) != dim:
            continue
        mat[i], ids[i] = v / (np.linalg.norm(v) + 1e-12), r["id"]
        i += 1
    for arr, path in ((mat[:i], vec_path), (ids[:i], ids_path)):
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, arr)
        os.replace(tmp, path)
    return i


def maybe_background_reindex(cfg: Config) -> bool:
    """Called by the hook. If the last reindex is older than the configured interval,
    start one in a detached process and return at once. The stamp is written first so
    a burst of prompts starts at most one."""
    every = cfg.get("hook", "reindex_every_min") * 60
    stamp = cfg.state / "last_reindex_start"
    try:
        if time.time() - stamp.stat().st_mtime < every:
            return False
    except OSError:
        pass
    cfg.state.mkdir(parents=True, exist_ok=True)
    stamp.write_text(str(time.time()))
    cmd = [sys.executable, str(cfg.root / "kit.py"), "reindex", "--quiet"]
    kw: dict = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if os.name == "nt":
        kw["creationflags"] = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    subprocess.Popen(cmd, **kw)
    return True
