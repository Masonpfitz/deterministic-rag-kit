import os
import time

from conftest import finish, write_doc
from rag import embed, index


def count(db, sql):
    return db.execute(sql).fetchone()[0]


def test_unchanged_files_are_skipped(cfg):
    write_doc(cfg, "a", "one", "One", "alpha text")
    write_doc(cfg, "a", "two", "Two", "beta text")
    first = finish(cfg)
    assert first.updated >= 2
    second = index.reindex(cfg, embed.build(cfg))
    assert second.updated == 0 and second.embedded == 0


def test_edit_reembeds_only_the_changed_section(cfg):
    write_doc(cfg, "a", "one", "One", "## A\n\nalpha alpha\n\n## B\n\nbeta beta")
    finish(cfg)
    calls = []
    emb = embed.build(cfg)
    orig = emb.embed
    emb.embed = lambda texts, timeout=None: (calls.append(len(texts)), orig(texts))[1]
    write_doc(cfg, "a", "one", "One", "## A\n\nalpha alpha\n\n## B\n\nbeta CHANGED")
    rep = index.reindex(cfg, emb)
    assert rep.updated == 1 and sum(calls) == 1


def test_touch_without_content_change_does_not_rechunk(cfg):
    p = write_doc(cfg, "a", "one", "One", "alpha")
    finish(cfg)
    os.utime(p, (time.time() + 100, time.time() + 100))
    rep = index.reindex(cfg, embed.build(cfg))
    assert rep.updated == 0 and rep.embedded == 0


def test_deleted_file_leaves_the_index(cfg):
    p = write_doc(cfg, "a", "one", "One", "alpha")
    finish(cfg)
    p.unlink()
    rep = index.reindex(cfg, embed.build(cfg))
    db = index.connect(cfg, create=False)
    assert rep.removed == 1
    assert count(db, "select count(*) from documents where id='one'") == 0
    assert count(db, "select count(*) from chunks_fts") == count(db, "select count(*) from chunks")


def test_bad_file_is_reported_not_fatal(cfg):
    write_doc(cfg, "a", "one", "One", "alpha")
    (cfg.knowledge / "a" / "broken.md").write_text("no frontmatter here")
    rep = finish(cfg)
    assert any("broken.md" in e for e in rep.errors)
    db = index.connect(cfg, create=False)
    assert count(db, "select count(*) from documents where id='one'") == 1


def test_embedder_change_recomputes_vectors(cfg):
    write_doc(cfg, "a", "one", "One", "alpha")
    finish(cfg)
    db = index.connect(cfg, create=False)
    db.execute("UPDATE meta SET value='other:model' WHERE key='embedder'")
    db.commit()
    db.close()
    rep = index.reindex(cfg, embed.build(cfg))
    assert rep.embedded >= 1


def test_deprecated_is_recorded_but_not_chunked(cfg):
    write_doc(cfg, "a", "dead", "Dead", "text\n\n## Superseded by\n\n[x](one.md)", status="deprecated")
    finish(cfg)
    db = index.connect(cfg, create=False)
    assert count(db, "select count(*) from documents where id='dead'") == 1
    assert count(db, "select count(*) from chunks where doc_id='dead'") == 0


def test_second_writer_is_refused(cfg):
    with index.lock(cfg):
        try:
            with index.lock(cfg):
                assert False
        except BlockingIOError:
            pass


class CountingEmbedder(embed.Fake):
    """Records the size of every request and how many were in flight at once."""

    def __init__(self, delay=0.0, fail_on=None):
        import threading
        self.sizes, self.max_inflight, self._inflight, self._lock = [], 0, 0, threading.Lock()
        self.delay, self.fail_on, self.calls = delay, fail_on, 0

    def embed(self, texts, timeout=None):
        with self._lock:
            self._inflight += 1
            self.max_inflight = max(self.max_inflight, self._inflight)
            self.calls += 1
            n = self.calls
            self.sizes.append(len(texts))
        try:
            time.sleep(self.delay)
            if self.fail_on and n == self.fail_on:
                raise embed.EmbedError("boom")
            return super().embed(texts)
        finally:
            with self._lock:
                self._inflight -= 1


def _many(cfg, n=40):
    for i in range(n):
        write_doc(cfg, "a", f"doc{i}", f"Doc {i}", f"unique words number{i} alpha beta gamma")


def test_embeddings_are_batched(cfg):
    _many(cfg)
    emb = CountingEmbedder()
    (cfg.root / "config.toml").write_text('[embeddings]\nprovider = "fake"\nbatch_size = 16\nconcurrency = 1\n')
    from rag import config
    c = config.load(cfg.root)
    from conftest import finish
    from rag import lint as L
    L.write_indexes(c.knowledge)
    index.reindex(c, emb)
    assert max(emb.sizes) <= 16 and sum(emb.sizes) >= 40 and emb.calls < 40
    assert emb.max_inflight == 1


def test_batches_run_concurrently_up_to_the_limit_and_results_land_on_the_right_sections(cfg):
    _many(cfg, 60)
    emb = CountingEmbedder(delay=0.05)
    (cfg.root / "config.toml").write_text('[embeddings]\nprovider = "fake"\nbatch_size = 8\nconcurrency = 3\n')
    from rag import config, lint as L
    c = config.load(cfg.root)
    L.write_indexes(c.knowledge)
    index.reindex(c, emb)
    assert 2 <= emb.max_inflight <= 3
    # every stored vector must be the vector of its own text, whatever order batches finished in
    db = index.connect(c, create=False)
    import numpy as np
    fake = embed.Fake()
    for r in db.execute("SELECT c.text, c.heading_path, c.embedding, d.title FROM chunks c JOIN documents d ON d.id=c.doc_id LIMIT 20"):
        from rag import sections as S
        want = fake.embed([index.embed_input(r["title"], S.Section("", "", r["heading_path"], r["text"], 0, 0, 0, 0))])[0]
        assert np.allclose(np.frombuffer(r["embedding"], dtype=np.float32), np.asarray(want, dtype=np.float32), atol=1e-6)


def test_a_failed_batch_stops_new_ones_and_keeps_finished_work(cfg):
    _many(cfg, 40)
    emb = CountingEmbedder(fail_on=2)
    (cfg.root / "config.toml").write_text('[embeddings]\nprovider = "fake"\nbatch_size = 4\nconcurrency = 1\n')
    from rag import config, lint as L
    c = config.load(cfg.root)
    L.write_indexes(c.knowledge)
    rep = index.reindex(c, emb)
    assert any("embedding stopped" in e for e in rep.errors)
    assert 0 < rep.embedded < 40 and rep.pending_embeddings > 0
    emb2 = CountingEmbedder()
    rep2 = index.reindex(c, emb2)  # the next run finishes the rest, no re-embedding of finished sections
    assert rep2.pending_embeddings == 0 and rep2.embedded == rep.pending_embeddings
