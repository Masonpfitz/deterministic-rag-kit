"""Stage 1 tests. No model is ever called: the CLI runner is replaced with a fake."""
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

KIT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KIT))

import assess as A  # noqa: E402
import minitest as M  # noqa: E402

NOW = datetime(2026, 10, 15, 12, 0, 0)


def set_mtime(path: Path, dt: datetime):
    ts = dt.timestamp()
    os.utime(path, (ts, ts))


def tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        h.update(str(p.relative_to(root)).encode())
        if p.is_file():
            h.update(p.read_bytes())
            h.update(str(p.stat().st_mtime_ns).encode())
    return h.hexdigest()


@pytest.fixture
def vault(tmp_path):
    v = tmp_path / "vault"
    (v / "a").mkdir(parents=True)
    (v / ".obsidian").mkdir()
    (v / ".git").mkdir()
    (v / ".trash").mkdir()
    (v / "a" / "one.md").write_text(
        "---\ntags: [x, y]\n---\n# One\nLinks [[two]] and [site](http://e.com) #inline\n"
        "```\n[[ignored]] #ignored\n```\n"
    )
    (v / "a" / "two.md").write_text("# Two\nNo frontmatter here. " + "word " * 200)
    (v / "root.md").write_text("---\ntitle: Root\ntags:\n  - p\n  - q\n---\nbody")
    for hidden in (".obsidian/x.md", ".git/y.md", ".trash/z.md"):
        (v / hidden).write_text("hidden " * 50)
    set_mtime(v / "a" / "one.md", datetime(2026, 10, 2))
    set_mtime(v / "a" / "two.md", datetime(2026, 9, 20))
    set_mtime(v / "root.md", datetime(2024, 1, 1))  # older than 12 months
    return v


# ---------------------------------------------------------------- assess
def test_assess_counts_and_hidden_dirs_skipped(vault):
    s = A.assess(vault, now=NOW)
    assert s["notes"] == 3  # hidden dirs excluded
    assert s["frontmatter_notes"] == 2
    assert s["wikilinks"] == 1 and s["md_links"] == 1  # fenced [[ignored]] not counted
    assert s["tags_total"] == 2 + 1 + 2  # fm [x,y] + #inline + fm list p,q
    assert s["unreadable_files"] == 0


def test_month_bucketing(vault):
    s = A.assess(vault, now=NOW)
    months = {m["month"]: m["notes"] for m in s["months"]}
    assert list(months)[-1] == "2026-10" and len(months) == 12
    assert months["2026-10"] == 1 and months["2026-09"] == 1
    assert sum(months.values()) == 2  # the 2024 note falls outside the window


def test_assess_is_read_only(vault):
    before = tree_hash(vault)
    A.assess(vault, now=NOW)
    assert tree_hash(vault) == before


def test_unreadable_file_is_counted_not_fatal(vault, monkeypatch):
    orig = Path.read_text

    def boom(self, *a, **k):
        if self.name == "two.md":
            raise PermissionError("nope")
        return orig(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", boom)
    s = A.assess(vault, now=NOW)
    assert s["unreadable_files"] == 1 and s["notes"] == 2


def test_symlink_outside_vault_skipped(vault, tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text("secret " * 100)
    try:
        (vault / "link.md").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")
    assert A.assess(vault, now=NOW)["notes"] == 3


# ---------------------------------------------------------------- thresholds / verdict
def test_thresholds_schema():
    data = json.loads((KIT / "thresholds.json").read_text())
    entries = {k: v for k, v in data.items() if not k.startswith("_")}
    assert set(entries) == {"skip_retrieval_tokens", "recommended_tokens", "bm25_crossover_tokens"}
    for name, t in entries.items():
        assert set(t) >= {"value", "unit", "source", "status"}, name
        assert t["status"] in ("cited", "measured"), name
    assert entries["skip_retrieval_tokens"]["value"] == 200000
    assert entries["recommended_tokens"]["value"] == 5000000
    assert entries["bm25_crossover_tokens"]["value"] == 10000000
    assert "Anthropic" in entries["skip_retrieval_tokens"]["source"]


def fake_stats(notes, tokens, churn=0):
    return {"notes": notes, "est_tokens": tokens, "notes_touched_per_month_avg": churn}


def test_verdict_three_outcomes_and_labels():
    th = A.load_thresholds()
    skip = A.verdict(fake_stats(100, 50_000), th)
    mid = A.verdict(fake_stats(800, 1_500_000), th)
    big = A.verdict(fake_stats(5000, 12_000_000), th)
    assert (skip["outcome"], mid["outcome"], big["outcome"]) == ("skip", "optional", "recommended")
    for v in (skip, mid, big):
        assert set(v) == {"outcome", "headline", "reasons", "caveats"} and v["reasons"]
    assert "read your whole vault" in skip["headline"]
    assert "half the tokens" in mid["headline"] and "accuracy cost" in mid["headline"]
    assert "cost, speed or usage limits" in big["headline"] and "browsing was a few points better" in big["headline"]


def test_boundaries_follow_the_thresholds_by_tokens_only():
    th = A.load_thresholds()
    assert A.verdict(fake_stats(1, 199_999), th)["outcome"] == "skip"
    assert A.verdict(fake_stats(1, 200_000), th)["outcome"] == "optional"
    assert A.verdict(fake_stats(1, 4_999_999), th)["outcome"] == "optional"
    assert A.verdict(fake_stats(1, 5_000_000), th)["outcome"] == "recommended"
    assert A.verdict(fake_stats(100_000, 50_000), th)["outcome"] == "skip"  # many tiny notes still fit


def test_verdict_cites_sources_and_keeps_the_benchmark_in_one_place():
    th = A.load_thresholds()
    for tokens in (50_000, 1_500_000, 12_000_000):
        v = A.verdict(fake_stats(1, tokens), th)
        text = " ".join(v["reasons"] + v["caveats"])
        assert "PROVISIONAL" not in text and "PENDING" not in text
        assert sum(A.BENCHMARK["who"] in c for c in v["caveats"]) == 1
        assert A.BENCHMARK_NOTE in v["caveats"]
    skip_text = " ".join(A.verdict(fake_stats(1, 50_000), th)["reasons"])
    assert "cited" in skip_text and "skip_retrieval_tokens" in skip_text
    big = " ".join(A.verdict(fake_stats(1, 12_000_000), th)["reasons"]).lower()
    assert "keyword" in big and "embedding" in big and "both" in big
    assert "http" not in A.BENCHMARK_NOTE  # no link: the results aren't public yet


def test_verdict_follows_threshold_values():
    th = A.load_thresholds()
    assert A.verdict(fake_stats(400, 50_000), th)["outcome"] == "skip"
    th["skip_retrieval_tokens"] = dict(th["skip_retrieval_tokens"], value=10)
    assert A.verdict(fake_stats(400, 50_000), th)["outcome"] == "optional"


# ---------------------------------------------------------------- minitest: sample
def make_notes(v: Path, n=8):
    for i in range(n):
        p = v / f"f{i % 2}" / f"note{i}.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"# Note {i}\n\n" + f"alpha{i} beta gamma " * 60)


def test_sample_deterministic_with_seed(tmp_path):
    make_notes(tmp_path)
    (tmp_path / "short.md").write_text("too short")
    a = M.sample_notes(tmp_path, n=3, seed=42)
    b = M.sample_notes(tmp_path, n=3, seed=42)
    assert a == b and a["seed"] == 42
    assert all(x["words"] >= 120 and len(x["excerpt"]) <= 1500 for x in a["notes"])
    assert all(x["relpath"] != "short.md" for x in a["notes"])


def test_sample_records_random_seed_when_omitted(tmp_path):
    make_notes(tmp_path)
    assert isinstance(M.sample_notes(tmp_path, 2)["seed"], int)


# ---------------------------------------------------------------- minitest: grading
GOLD = {"id": "q", "question": "?", "gold": ["projects/Garden Redesign.md"], "expect_none": False}


def test_parse_answer_variants():
    plain = '{"answer": "x", "sources": ["a.md"]}'
    assert M.parse_answer(plain)["sources"] == ["a.md"]
    assert M.parse_answer("```json\n" + plain + "\n```")["answer"] == "x"
    assert M.parse_answer("Sure! here you go: " + plain + " hope it helps")["sources"] == ["a.md"]
    assert M.parse_answer("no json here") is None
    assert M.parse_answer("") is None


@pytest.mark.parametrize("cited", [
    "projects/Garden Redesign.md",
    "./projects/Garden Redesign.md",
    "projects/Garden Redesign",            # Obsidian: no extension
    "Garden Redesign",                     # bare note name
    "[[Garden Redesign]]",
    "[[Garden Redesign#Details|the garden]]",
    "/vault/root/projects/Garden Redesign.md",  # absolute path under the vault
    "projects\\Garden Redesign.md",
])
def test_grade_path_forms_hit(cited):
    parsed = {"answer": "6800", "sources": [cited]}
    assert M.grade(GOLD, parsed, vault="/vault/root")["hit"]


def test_grade_miss_and_unparseable():
    assert not M.grade(GOLD, {"answer": "", "sources": ["other.md"]})["hit"]
    assert not M.grade(GOLD, {"answer": "", "sources": []})["hit"]
    assert M.grade(GOLD, None)["reason"] == "unparseable"


def test_grade_expect_none():
    case = {"id": "n", "question": "?", "gold": [], "expect_none": True}
    assert M.grade(case, {"answer": "Here you go", "sources": []})["hit"]
    assert M.grade(case, {"answer": "I could not find that in the notes.", "sources": ["x.md"]})["hit"]
    assert not M.grade(case, {"answer": "It is 42.", "sources": ["x.md"]})["hit"]


# ---------------------------------------------------------------- minitest: keyword arm
@pytest.fixture
def small_vault(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    (v / "garden.md").write_text(
        "# Garden\n\n## Quote\nThe contractor quote was 6800 dollars including irrigation.\n\n"
        "## Plants\nMilkweed and coneflower along the fence.\n")
    (v / "car.md").write_text("---\ntitle: Car\n---\n# Car\nOil change at 48200 miles. Brake pads replaced.\n")
    (v / "meeting.md").write_text("# Meeting\n```\n# not a heading\n```\nDecision: postpone the redesign.\n")
    return v


def test_split_sections_headings_and_fences():
    secs = M.split_sections("# A\nintro\n## B\nbody\n```\n# fake\n```\n", "n.md")
    assert [s["heading"] for s in secs] == ["n > A", "n > A > B"]
    assert "# fake" in secs[1]["body"]


def test_fts_retrieval(small_vault, tmp_path):
    idx = M.KeywordIndex(small_vault, tmp_path)
    try:
        hits = idx.search("How much was the contractor quote for irrigation?")
        assert hits and hits[0]["relpath"] == "garden.md" and "Quote" in hits[0]["heading"]
        assert idx.search("brake pad replacement")[0]["relpath"] == "car.md"  # stemming
        assert idx.search("the of and") == []  # stopwords only -> nothing, no crash
        assert isinstance(idx.search('weird "quote" AND OR NEAR('), list)  # FTS syntax chars are safe
    finally:
        idx.close()


# ---------------------------------------------------------------- minitest: run with a fake runner
def fake_runner_factory(calls):
    def runner(argv, stdin_text, cwd, env, timeout):
        calls.append((argv, stdin_text, cwd))
        src = "garden.md" if "PASSAGE" in (stdin_text or "") else "./garden"
        out = {"result": "```json\n" + json.dumps({"answer": "6800", "sources": [src]}) + "\n```",
               "is_error": False, "total_cost_usd": 0.01,
               "usage": {"input_tokens": 3, "output_tokens": 5, "cache_read_input_tokens": 7,
                         "cache_creation_input_tokens": 11}}
        return M.RunResult(0, json.dumps(out), "", 1.5)
    return runner


def test_run_test_with_injected_runner(small_vault, tmp_path):
    q = tmp_path / "q.json"
    q.write_text(json.dumps({"version": 1, "cases": [
        {"id": "q1", "question": "contractor quote irrigation", "gold": ["garden.md"], "expect_none": False}]}))
    out = tmp_path / "res.json"
    calls = []
    doc = M.run_test(small_vault, q, "claude", out, runner=fake_runner_factory(calls), quiet=True)
    assert len(calls) == 2
    browse, keyword = calls
    assert browse[2] == small_vault.resolve() and "Read,Grep,Glob" in browse[0]
    assert "--safe-mode" in browse[0] and "--strict-mcp-config" in browse[0]
    assert keyword[0][keyword[0].index("--tools") + 1] == ""  # keyword arm gets no tools
    assert "PASSAGE" in keyword[1] and "garden.md" in keyword[1]
    arms = doc["cases"][0]["arms"]
    assert all(a["hit"] and a["status"] == "ok" for a in arms)
    assert arms[0]["cache_creation_tokens"] == 11 and arms[0]["cost_usd"] == 0.01
    assert doc["totals"]["browse"]["hits"] == 1
    assert json.loads(out.read_text())["harness"] == "claude"


def test_failures_are_recorded_not_fatal(small_vault, tmp_path):
    q = tmp_path / "q.json"
    q.write_text(json.dumps({"version": 1, "cases": [
        {"id": "q1", "question": "contractor quote", "gold": ["garden.md"]},
        {"id": "q2", "question": "oil change", "gold": ["car.md"]}]}))
    n = {"i": 0}

    def runner(argv, stdin_text, cwd, env, timeout):
        n["i"] += 1
        if n["i"] == 1:
            return M.RunResult(-1, "", "", 180.0, timed_out=True)
        if n["i"] == 2:
            raise RuntimeError("boom")
        if n["i"] == 3:
            return M.RunResult(0, json.dumps({"result": "not json at all", "is_error": False, "usage": {}}), "", 1)
        return M.RunResult(1, "", "auth failed", 1)

    doc = M.run_test(small_vault, q, "claude", tmp_path / "r.json", runner=runner, quiet=True)
    statuses = [a["status"] for c in doc["cases"] for a in c["arms"]]
    assert statuses == ["timeout", "error", "unparseable", "error"]
    assert doc["totals"]["browse"]["failures"] + doc["totals"]["keyword"]["failures"] == 4


def test_codex_parse_and_argv():
    h = M.CodexHarness()
    lines = "\n".join(json.dumps(e) for e in [
        {"type": "item.completed", "item": {"type": "agent_message", "text": '{"answer":"a","sources":[]}'}},
        {"type": "item.completed", "item": {"type": "command_execution"}},
        {"type": "turn.completed", "usage": {"input_tokens": 100, "cached_input_tokens": 40, "output_tokens": 5}},
    ])
    p = h.parse(lines)
    assert p["tool_calls"] == 1 and p["usage"]["cache_read_tokens"] == 40 and "answer" in p["text"]
    a = h.argv("browse", None, Path("/x"))
    assert "--ignore-user-config" in a and "read-only" in a and a[-1] == "-"


def test_codex_setup_cleans_up_auth(tmp_path, monkeypatch):
    home = tmp_path / "ch"
    home.mkdir()
    (home / "auth.json").write_text("{}")
    monkeypatch.setenv("CODEX_HOME", str(home))
    h = M.CodexHarness()
    with h.setup():
        tmp = h.home
        assert h.isolated and (tmp / "auth.json").exists() and h.env()["CODEX_HOME"] == str(tmp)
    assert not tmp.exists()


def test_questions_validation(tmp_path):
    p = tmp_path / "q.json"
    p.write_text(json.dumps({"version": 1, "cases": [{"id": "a", "question": "q", "gold": []}]}))
    with pytest.raises(ValueError):
        M.load_questions(p)
