import json
import subprocess
import sys
import time

import pytest

from conftest import KIT, finish, write_doc
from rag import embed, hook, index


@pytest.fixture
def ready(cfg, monkeypatch):
    monkeypatch.setattr(index, "maybe_background_reindex", lambda c: False)
    write_doc(cfg, "projects", "permit", "Harbor permit", "The plumber is booked for the 14th of March.")
    finish(cfg)
    return cfg


def payload(prompt):
    return json.dumps({"prompt": prompt, "session_id": "s", "cwd": "/tmp", "hook_event_name": "UserPromptSubmit"})


def test_injects_matching_passage(ready):
    out = hook.run(payload("when is the plumber booked for the harbor job"), ready.root)
    assert "<retrieved_notes" in out and "permit.md" in out


def test_codex_style_payload_field_works(ready):
    out = hook.run(json.dumps({"user_prompt": "when is the plumber booked for the harbor job"}), ready.root)
    assert "permit.md" in out


@pytest.mark.parametrize("bad", ["", "not json", "[]", "{}", '{"prompt": null}', '{"prompt": 5}'])
def test_garbage_input_injects_nothing(ready, bad):
    assert hook.run(bad, ready.root) == ""


def test_short_and_harness_prompts_are_skipped(ready):
    assert hook.run(payload("hi"), ready.root) == ""
    assert hook.run(payload("<task-notification>plumber harbor booked</task-notification>"), ready.root) == ""


def test_no_index_injects_nothing(cfg):
    assert hook.run(payload("when is the plumber booked for the harbor job"), cfg.root) == ""


def test_corrupt_database_fails_open(ready):
    ready.db_path.write_bytes(b"this is not a sqlite file" * 100)
    assert hook.run(payload("when is the plumber booked for the harbor job"), ready.root) == ""


def test_embedder_crash_fails_open(ready):
    class Boom(embed.Fake):
        def embed(self, texts, timeout=None):
            raise RuntimeError("boom")
    assert hook.run(payload("when is the plumber booked for the harbor job"), ready.root, embedder=Boom()) == ""


def test_embedder_outage_falls_back_to_keywords(ready):
    class Down(embed.Fake):
        def embed(self, texts, timeout=None):
            raise embed.EmbedError("connection refused")
    out = hook.run(payload("when is the plumber booked for the harbor job"), ready.root, embedder=Down())
    assert "permit.md" in out


def test_slow_embedder_is_cut_off_by_the_budget(ready):
    (ready.root / "config.toml").write_text('[embeddings]\nprovider = "fake"\n[hook]\nbudget_s = 0.4\n')

    class Slow(embed.Fake):
        def embed(self, texts, timeout=None):
            time.sleep(5)
    t = time.time()
    assert hook.run(payload("when is the plumber booked for the harbor job"), ready.root, embedder=Slow()) == ""
    assert time.time() - t < 2


def test_cli_hook_exits_zero_on_garbage_and_prints_nothing(tmp_path):
    r = subprocess.run([sys.executable, str(KIT / "kit.py"), "hook", "--harness", "claude", "--drag-kit", "abc"],
                       input="}}} definitely not json", capture_output=True, text=True, timeout=30,
                       env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0 and r.stdout == ""


def test_background_reindex_is_rate_limited(cfg, monkeypatch):
    spawned = []
    monkeypatch.setattr(index.subprocess, "Popen", lambda *a, **k: spawned.append(a))
    assert index.maybe_background_reindex(cfg) is True
    assert index.maybe_background_reindex(cfg) is False
    assert len(spawned) == 1


def test_hook_prints_the_pointer_list_when_enabled(ready):
    (ready.root / "config.toml").write_text(
        '[embeddings]\nprovider = "fake"\n[retrieval]\npointers_enabled = true\npointer_min_cosine = -1.0\n'
        "min_cosine = 0.99\nmin_term_overlap = 1.01\n")
    out = hook.run(payload("when is the plumber booked for the harbor job"), ready.root)
    assert "<possibly_relevant_notes" in out and "permit.md" in out and "<retrieved_notes" not in out
