"""Both stages end to end on a generated fixture vault, with the fake embedder.
(Real Ollama runs are recorded in DEV-NOTES.md; tests must not need a running model.)"""
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import KIT
from rag import constants, embed, evalrun, index, migrate, search


def tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        h.update(p.relative_to(root).as_posix().encode())
        if p.is_file():
            h.update(p.read_bytes())
    return h.hexdigest()


@pytest.fixture
def fixture(tmp_path):
    out = tmp_path / "fx"
    subprocess.run([sys.executable, "-B", str(KIT / "fixtures" / "make_vault.py"), "--notes", "100",
                    "--seed", "7", "--out", str(out), "--quiet"], check=True)
    return out


def test_stage1_then_stage2_on_fixture(cfg, fixture):
    import importlib.util
    spec = importlib.util.spec_from_file_location("assess", KIT / "assess.py")
    assess = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(assess)
    vault = fixture / "vault"
    before = tree_hash(vault)

    stats = assess.assess(vault)
    verdict = assess.verdict(stats, assess.load_thresholds())
    assert stats["notes"] == 100 and verdict["outcome"] == "skip"  # 19k tokens fits in a prompt

    migrate.plan_vault(cfg, vault)
    res = migrate.apply_plan(cfg)
    assert res["ok"] and not res["quarantined"]
    rep = index.reindex(cfg, embed.build(cfg))
    assert not rep.errors and rep.chunks_total > 100

    cases = evalrun.load_cases(cfg, fixture / "golden.json")
    assert sum(c["kind"] == "offtopic" for c in cases) >= 40
    out = evalrun.evaluate(cfg, cases, embed.build(cfg))
    default = out["summary"]["hybrid (default)"]
    assert default["leaks"] == 0
    assert default["recall_at_4"] >= 0.2  # fake embedder: the keyword half does the work
    fi = default["false_injection"]
    assert fi["offtopic"] <= 0.10 and fi["no_answer"] <= 0.2  # chat, coding and git prompts get nothing

    r = search.retrieve(cfg, "What port does the Kestrel staging API listen on?", embed.build(cfg))
    assert r.selected and "8421" in r.block
    assert tree_hash(vault) == before, "the vault must never change"

    migrate.revert(cfg)
    assert not list(cfg.knowledge.rglob("*.md"))
    assert tree_hash(vault) == before


def test_skool_line_is_one_constant_shown_in_readme_and_report():
    assert constants.SKOOL_LINE in (KIT / "README.md").read_text()
    assert "free" not in constants.SKOOL_LINE.lower() and "$" not in constants.SKOOL_LINE
    assert "skool" not in (KIT / "START.md").read_text().lower(), "START.md must not tell the agent to promote"


def test_report_builds_with_toggle_and_skool_line(cfg, fixture):
    import argparse
    from rag import report
    migrate.plan_vault(cfg, fixture / "vault")
    migrate.apply_plan(cfg)
    index.reindex(cfg, embed.build(cfg))
    (cfg.corpus / "eval").mkdir(parents=True)
    (cfg.corpus / "eval" / "golden.json").write_text((fixture / "golden.json").read_text())
    page = report.build(cfg, argparse.Namespace(vault=str(fixture / "vault"), queries=None, before=None, after=None)).read_text()
    assert "id='theme'" in page and "prefers-color-scheme" in page and 'data-theme' in page
    assert page.rstrip().endswith("</html>") and constants.SKOOL_LINE.replace("'", "&#x27;") in page
    assert "<footer><p>" + constants.SKOOL_LINE.replace("'", "&#x27;") in page
    assert page.count("&lt;note ") >= 1  # the tuning view shows an injected block
    assert "http" not in page.replace("https://www.skool.com/ccu", "")  # self-contained, no external assets
