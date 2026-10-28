import datetime as dt

from conftest import finish, write_doc
from rag import embed, index, lint, remember, search
from rag.frontmatter import parse


def problems(cfg):
    return {p.rule for p in lint.lint_corpus(cfg.knowledge, lint.load_schema(cfg.corpus / "schema.yaml"))}


def test_clean_corpus_has_no_problems(cfg):
    write_doc(cfg, "a", "one", "One", "## H\n\ntext")
    finish(cfg)
    lint.fix_anchors(cfg.knowledge)
    assert problems(cfg) == set()


def test_each_rule_fires(cfg):
    write_doc(cfg, "a", "one", "One", "see [gone](nope.md) and [bad](one.md#nothing)")
    write_doc(cfg, "a", "dup", "Dup", "x")
    write_doc(cfg, "b", "dup2", "Dup2", "x", id="dup")
    write_doc(cfg, "a", "dead", "Dead", "no successor", status="deprecated")
    write_doc(cfg, "a", "key", "Key", "token sk-" + "ant-api03-" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123")
    write_doc(cfg, "a", "anchors", "Anchors", "## A {#x}\n\n## B {#x}\n")
    finish(cfg)
    (cfg.corpus / "schema.yaml").write_text("version: 1\nbundles:\n  a:\n    types: [Memory]\n")
    got = problems(cfg)
    assert {"dead-link", "dead-anchor", "duplicate-id", "successor", "secret-shape", "duplicate-anchor",
            "bundle-type", "bundle"} <= got


def test_fix_anchors_leaves_yaml_alone(cfg):
    p = write_doc(cfg, "a", "one", "One", "## Heading\n\nbody")
    before = p.read_text().split("---")[1]
    assert lint.fix_anchors(cfg.knowledge) == 1
    assert p.read_text().split("---")[1] == before and "{#heading}" in p.read_text()


def test_anchor_slug_does_not_look_like_a_key(cfg):
    write_doc(cfg, "a", "one", "One", "## Ask before you draft the whole long thing now {#ask-before-you-draft-the-whole-long-thing-now}\n\nx")
    finish(cfg)
    assert "secret-shape" not in problems(cfg)


def test_remember_is_searchable_immediately(cfg):
    write_doc(cfg, "reference", "seed", "Seed note", "unrelated seed content")
    finish(cfg)
    emb = embed.build(cfg)
    res = remember.remember(cfg, "problem", "Build fails with ERR_OSSL_EVP_UNSUPPORTED on Node 22",
                            emb, solution="Use NODE_OPTIONS=--openssl-legacy-provider", area="webapp")
    assert res["ok"], res
    out = search.retrieve(cfg, "ERR_OSSL_EVP_UNSUPPORTED build failing", emb)
    assert out.selected and "openssl-legacy-provider" in out.selected[0].text
    doc = parse(__import__("pathlib").Path(res["path"]), cfg.knowledge)
    assert doc.type == "Fix" and doc.meta["generated"]["by"] == "remember" and doc.project == "webapp"
    assert lint.lint_corpus(cfg.knowledge, lint.load_schema(cfg.corpus / "schema.yaml")) == []


def test_remember_refuses_a_credential(cfg):
    emb = embed.build(cfg)
    res = remember.remember(cfg, "fact", "key is sk-" + "ant-api03-" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123", emb)
    assert not res["ok"] and "credential" in res["problems"][0]
    assert not list(cfg.knowledge.rglob("rem-*")) and not list(cfg.knowledge.rglob("*-key-is*"))


def test_preference_goes_to_me_scope_and_respects_scope_filter(cfg):
    emb = embed.build(cfg)
    res = remember.remember(cfg, "preference", "I never take meetings before ten", emb, why="mornings are for deep work")
    assert res["ok"]
    doc = parse(__import__("pathlib").Path(res["path"]), cfg.knowledge)
    assert doc.scope == "me" and doc.type == "Memory"
    hit = search.retrieve(cfg, "do I take meetings before ten", emb)
    assert hit.selected
    hit2 = search.retrieve(cfg.override(scopes=["shared"]), "do I take meetings before ten", emb)
    assert not hit2.selected


def test_supersession_flow_keeps_old_note_out(cfg):
    write_doc(cfg, "reference", "price-old", "Price list", "The widget costs 10 dollars.",
              status="deprecated")
    write_doc(cfg, "reference", "price-new", "Price list", "The widget costs 12 dollars.")
    p = cfg.knowledge / "reference" / "price-old.md"
    p.write_text(p.read_text() + "\n## Superseded by\n\n[new](price-new.md)\n")
    finish(cfg)
    res = search.retrieve(cfg, "what does the widget cost in dollars", embed.build(cfg))
    assert [c.doc_id for c in res.selected] == ["price-new"]
    assert "successor" not in problems(cfg)


def test_stale_after_flips_with_time(cfg):
    soon = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=1)).isoformat()
    write_doc(cfg, "reference", "promo", "Promo code", "The promo code is SPRING26.", stale_after=soon)
    finish(cfg)
    emb = embed.build(cfg)
    assert search.retrieve(cfg, "what is the promo code", emb).selected
    import time
    time.sleep(1.2)
    assert not search.retrieve(cfg, "what is the promo code", emb).selected
