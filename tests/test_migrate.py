import hashlib
from pathlib import Path

import pytest

from rag import embed, index, lint, migrate, search


def tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        h.update(p.relative_to(root).as_posix().encode())
        if p.is_file():
            h.update(p.read_bytes())
            h.update(str(p.stat().st_mtime_ns).encode())
    return h.hexdigest()


@pytest.fixture
def vault(tmp_path):
    v = tmp_path / "vault"
    (v / "Projects" / "Harbor").mkdir(parents=True)
    (v / "Journal").mkdir()
    (v / ".obsidian").mkdir()
    (v / ".obsidian" / "app.json").write_text("{}")
    (v / ".trash").mkdir()
    (v / ".trash" / "gone.md").write_text("deleted note")
    (v / "Projects/Harbor/permit.md").write_text(
        "---\ntags: [harbor]\n---\n# Permit timeline\n\nPermit HB-20931. See [[Harbor budget]] and "
        "[back](../Harbor/Harbor%20budget.md) and [x](missing.md).\n\n## Dates\n\nInspection March 14.\n")
    (v / "Projects/Harbor/Harbor budget.md").write_text("# Harbor budget\n\nTotal 48000. #money\n")
    (v / "Journal/2026-04-01.md").write_text("Quiet day with espresso. #personal\n")
    (v / "Journal/2026-04-02.md").write_text("Meeting notes with Sam about the roadmap.\n")
    (v / "root-note.md").write_text("A root level note about kiln firing.\n")
    (v / "empty.md").write_text("")
    (v / "secret.md").write_text("# Keys\n\nuse sk-" + "ant-api03-" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123 for the API\n")
    return v


def test_plan_apply_never_touches_the_vault(cfg, vault):
    before = tree_hash(vault)
    plan = migrate.plan_vault(cfg, vault)
    res = migrate.apply_plan(cfg)
    index.reindex(cfg, embed.build(cfg))
    assert tree_hash(vault) == before
    assert res["ok"]
    assert (cfg.corpus / "raw/notes/Projects/Harbor/permit.md").exists()
    srcs = {e["src"] for e in plan["entries"]}
    assert ".trash/gone.md" not in srcs and not any(s.startswith(".obsidian") for s in srcs)


def test_proposals(cfg, vault):
    plan = migrate.plan_vault(cfg, vault)
    by = {e["src"]: e for e in plan["entries"]}
    assert by["Projects/Harbor/permit.md"]["bundle"] == "projects"
    assert by["Projects/Harbor/permit.md"]["project"] == "harbor"
    assert by["Journal/2026-04-01.md"]["type"] == "Memory" and by["Journal/2026-04-01.md"]["scope"] == "me"
    assert by["root-note.md"]["bundle"] == "notes"
    assert by["Projects/Harbor/permit.md"]["title"] == "Permit timeline"
    assert "empty" in by["empty.md"]["flags"] and not by["empty.md"]["include"]


def test_gate_keeps_credentials_out_and_corpus_lints_clean(cfg, vault):
    migrate.plan_vault(cfg, vault)
    res = migrate.apply_plan(cfg)
    assert [q["src"] for q in res["quarantined"]] == ["secret.md"]
    assert lint.lint_corpus(cfg.knowledge, lint.load_schema(cfg.corpus / "schema.yaml")) == []
    assert not any("sk-ant" in p.read_text() for p in cfg.knowledge.rglob("*.md"))


def test_links_are_rewritten_or_flattened(cfg, vault):
    migrate.plan_vault(cfg, vault)
    migrate.apply_plan(cfg)
    text = next(cfg.knowledge.glob("projects/permit-*.md")).read_text()
    budget = next(cfg.knowledge.glob("projects/harbor-budget-*.md")).name
    assert f"(harbor-budget" in text and budget in text
    assert "[[" not in text and "](missing.md)" not in text and "[x]" not in text


def test_gate_failure_writes_nothing(cfg, vault):
    migrate.plan_vault(cfg, vault)
    plan = migrate.load_plan(cfg)
    plan["entries"][0]["type"] = "Nonsense"  # a typo the linter must catch
    migrate.save_plan(cfg, plan)
    res = migrate.apply_plan(cfg)
    assert not res["ok"] and res["problems"]
    assert not cfg.knowledge.exists() or not list(cfg.knowledge.rglob("*.md"))


def test_set_overrides_and_schema_follows(cfg, vault):
    migrate.plan_vault(cfg, vault)
    assert migrate.set_entries(cfg, "Journal/*", bundle="diary", typ="Memory", scope="me") == 2
    assert migrate.apply_plan(cfg)["ok"]
    assert list(cfg.knowledge.glob("diary/*.md"))
    assert "diary" in lint.load_schema(cfg.corpus / "schema.yaml")


def test_revert_removes_exactly_what_apply_created(cfg, vault):
    migrate.plan_vault(cfg, vault)
    migrate.apply_plan(cfg)
    keep = next(cfg.knowledge.glob("projects/harbor-budget-*.md"))
    keep.write_text(keep.read_text() + "\nmy own edit\n")
    mine = cfg.knowledge / "lessons" / "mine.md"
    mine.parent.mkdir(parents=True)
    mine.write_text("---\ntype: Fix\nid: m\ntitle: m\n---\nhi\n")
    res = migrate.revert(cfg)
    assert res["kept"] == [keep.relative_to(cfg.knowledge).as_posix()]
    assert keep.exists() and mine.exists()
    assert not list(cfg.knowledge.glob("projects/permit-*.md"))
    assert not (cfg.corpus / "raw/notes").exists()


def test_reapply_is_idempotent(cfg, vault):
    migrate.plan_vault(cfg, vault)
    migrate.apply_plan(cfg)
    first = sorted(p.name for p in cfg.knowledge.rglob("*.md"))
    migrate.apply_plan(cfg)
    assert sorted(p.name for p in cfg.knowledge.rglob("*.md")) == first


def test_corpus_inside_vault_is_refused(cfg, vault, monkeypatch):
    (cfg.root / "config.toml").write_text(f'[corpus]\ndir = "{vault / "okf"}"\n[embeddings]\nprovider = "fake"\n')
    from rag import config
    c2 = config.load(cfg.root)
    with pytest.raises(SystemExit):
        migrate.plan_vault(c2, vault)


def test_end_to_end_retrieval_after_migration(cfg, vault):
    migrate.plan_vault(cfg, vault)
    migrate.apply_plan(cfg)
    index.reindex(cfg, embed.build(cfg))
    res = search.retrieve(cfg, "harbor permit number HB-20931", embed.build(cfg))
    assert res.selected and res.selected[0].original == "Projects/Harbor/permit.md"


def test_search_tools_skip_the_raw_copies(cfg, vault):
    """An agent grepping okf/ should see each note once. The kit writes a .ignore that
    ripgrep honors, so raw/, staging/, eval/ and reports/ stay out of Grep and Glob."""
    import shutil
    import subprocess
    migrate.plan_vault(cfg, vault)
    migrate.apply_plan(cfg)
    ignore = (cfg.corpus / ".ignore").read_text()
    for d in ("raw/", "staging/", "eval/", "reports/"):
        assert d in ignore
    (cfg.corpus / "eval").mkdir(exist_ok=True)
    (cfg.corpus / "eval" / "golden.json").write_text('{"answer": "HB-20931"}')
    rg = shutil.which("rg")
    if not rg:
        pytest.skip("ripgrep not installed here; the .ignore contents were checked above")
    files = subprocess.run([rg, "--files", str(cfg.corpus)], capture_output=True, text=True).stdout
    assert "knowledge" in files and "/raw/" not in files and "migration-plan" not in files and "golden.json" not in files
    hits = subprocess.run([rg, "-l", "HB-20931", str(cfg.corpus)], capture_output=True, text=True).stdout.splitlines()
    assert hits and all("/knowledge/" in h for h in hits)


def test_ignore_file_is_not_overwritten_and_remember_writes_it(cfg):
    from rag import embed, remember
    (cfg.corpus).mkdir(parents=True)
    (cfg.corpus / ".ignore").write_text("my-own-rule/\n")
    migrate.write_ignore(cfg.corpus)
    assert (cfg.corpus / ".ignore").read_text() == "my-own-rule/\n"
    (cfg.corpus / ".ignore").unlink()
    remember.remember(cfg, "fact", "The portal moved", embed.build(cfg))
    assert "raw/" in (cfg.corpus / ".ignore").read_text()
