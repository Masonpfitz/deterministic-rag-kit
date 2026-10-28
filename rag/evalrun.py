"""Measure retrieval on questions whose answers you know, and ablate the optional parts.

The golden file (same format the stage 1 mini test uses):
    {"version": 1, "cases": [{"id": "g1", "question": "...", "gold": ["Projects/x.md"],
                              "expect_none": false}]}
`gold` lists the original note paths relative to your vault. A case with expect_none
is a question your notes do not answer: the right behavior is to inject nothing.

Candidates are gathered once per question and every variant is a cheap replay of
`select` over them, so an ablation costs one search per question, not one per variant.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
import statistics
import time

import numpy as np

from . import index as idx
from . import search
from .config import Config
from .embed import EmbedError, Embedder

# name -> (channels, config overrides, use reranker)
VARIANTS = {
    "keyword only": (("lexical",), {}, False),
    "semantic only": (("semantic",), {}, False),
    "hybrid, no thresholds": (("lexical", "semantic"), {"min_cosine": 0.0, "min_term_overlap": 0.0, "min_content_terms": 0}, False),
    "hybrid (default)": (("lexical", "semantic"), {"exact_enabled": False}, False),
    "hybrid + exact title": (("lexical", "semantic", "exact"), {"exact_enabled": True}, False),
    "hybrid + reranker": (("lexical", "semantic"), {}, True),
}


def load_golden(path) -> list[dict]:
    data = json.loads(open(path, encoding="utf-8").read())
    return data["cases"] if isinstance(data, dict) else data


def load_cases(cfg: Config, golden_path) -> list[dict]:
    """The golden questions, plus the shipped generic off-topic prompts (coding, git, chat,
    general knowledge: most of what people type into an agent) and, if present, the
    near-miss prompts in <corpus>/eval/nearmiss.json. Each unanswerable case carries a
    `kind` so false injection is reported separately per set."""
    cases = load_golden(golden_path)
    for c in cases:
        c["kind"] = "no_answer" if c.get("expect_none") else c.get("kind", "answerable")
    extra = [Path(__file__).parent / "offtopic.json", cfg.corpus / "eval" / "nearmiss.json"]
    for p in extra:
        if p.is_file():
            cases += load_golden(p)
    return cases


NONE_KINDS = ("no_answer", "offtopic", "nearmiss")


def gold_match(cand: search.Candidate, gold: set[str]) -> bool:
    return bool({cand.original, cand.path, cand.doc_id} & gold)


def _rate(rows):
    return sum(r["selected"] > 0 for r in rows) / len(rows)


def _pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))] if xs else 0.0


def leaks(db, cfg: Config, selected_doc_ids: list[str], now: float) -> int:
    """Selected passages that a hard filter should have kept out. Must always be 0."""
    if not selected_doc_ids:
        return 0
    el, params = search.eligible(cfg, now)
    q = ",".join("?" * len(selected_doc_ids))
    ok = db.execute(f"SELECT count(*) FROM documents d WHERE d.id IN ({q}) AND {el}", [*selected_doc_ids, *params]).fetchone()[0]
    return len(set(selected_doc_ids)) - ok


def evaluate(cfg: Config, cases: list[dict], embedder: Embedder, variants=None, rerank_fn=None,
             db=None) -> dict:
    variants = variants or {k: v for k, v in VARIANTS.items() if v[2] is False or rerank_fn}
    own = db is None
    db = db or idx.connect(cfg, create=False)
    now = time.time()
    rows = {name: [] for name in variants}
    for case in cases:
        q = case["question"]
        gold = set(case.get("gold") or [])
        none = bool(case.get("expect_none"))
        qvec = None
        if embedder.name != "none":
            try:
                v = np.asarray(embedder.embed([q])[0], dtype=np.float32)
                qvec = v / (np.linalg.norm(v) + 1e-12)
            except EmbedError:
                qvec = None
        gathered: dict = {}
        rr_scores = None
        for name, (channels, over, use_rr) in variants.items():
            vcfg = cfg.override(**over)
            key = channels
            if key not in gathered:
                gathered[key] = search.gather(vcfg, db, q, embedder, qvec=qvec, now=now, channels=channels)
            g = gathered[key]
            scores = None
            if use_rr:
                if rr_scores is None:
                    top = g.candidates[: cfg.get("retrieval", "rerank_top_n")]
                    rr_scores = rerank_fn(cfg, q, [c.text for c in top]) if top else []
                scores = rr_scores
            t = time.perf_counter()
            res = search.select(vcfg, g, scores)
            sel_ms = (time.perf_counter() - t) * 1000
            ranked_gold = [i for i, c in enumerate(g.candidates) if gold_match(c, gold)]
            sel_gold = [c for c in res.selected if gold_match(c, gold)]
            rows[name].append({
                "id": case["id"], "none": none, "kind": case.get("kind", ""),
                "selected": len(res.selected), "tokens": res.tokens,
                "hit": bool(sel_gold), "gold_in_top10": bool(ranked_gold and ranked_gold[0] < 10),
                "rr": (1.0 / (ranked_gold[0] + 1)) if ranked_gold else 0.0,
                "precision": (len(sel_gold) / len(res.selected)) if res.selected else None,
                "pointers": len(res.pointers),
                "gold_in_pointers": any(gold_match(c, gold) for c in res.pointers),
                "gold_docs_found": len({c.doc_id for c in sel_gold}), "gold_docs": len(gold),
                "leaks": leaks(db, cfg, [c.doc_id for c in res.selected], now),
                "select_ms": sel_ms,
            })
    summary = {}
    for name, rs in rows.items():
        ans = [r for r in rs if not r["none"]]
        non = [r for r in rs if r["none"]]
        precs = [r["precision"] for r in ans if r["precision"] is not None]
        summary[name] = {
            "recall_at_4": sum(r["hit"] for r in ans) / len(ans) if ans else None,
            "gold_in_top10": sum(r["gold_in_top10"] for r in ans) / len(ans) if ans else None,
            "mrr": statistics.mean(r["rr"] for r in ans) if ans else None,
            "precision": statistics.mean(precs) if precs else None,
            "abstain_accuracy": sum(r["selected"] == 0 for r in non) / len(non) if non else None,
            "pointer_recall": (sum(r["hit"] or r["gold_in_pointers"] for r in ans) / len(ans)) if ans else None,
            "pointer_rate": {k: sum(r["pointers"] > 0 for r in non if r["kind"] == k) / sum(r["kind"] == k for r in non)
                             for k in NONE_KINDS if any(r["kind"] == k for r in non)},
            "false_injection": {k: _rate([r for r in non if r["kind"] == k]) for k in NONE_KINDS
                                if any(r["kind"] == k for r in non)},
            "false_abstain": sum(r["selected"] == 0 for r in ans) / len(ans) if ans else None,
            "leaks": sum(r["leaks"] for r in rs),
            "tokens_mean": statistics.mean(r["tokens"] for r in rs) if rs else 0,
            "tokens_max": max((r["tokens"] for r in rs), default=0),
            "answerable": len(ans), "unanswerable": len(non),
        }
    if own:
        db.close()
    return {"summary": summary, "rows": rows}


def latency(cfg: Config, cases: list[dict], embedder: Embedder, rerank_fn=None, repeat: int = 1) -> dict:
    """End-to-end retrieve() time per question, embedding call included (what the hook pays)."""
    db = idx.connect(cfg, create=False)
    search.retrieve(cfg, cases[0]["question"], embedder, db=db, rerank=False)  # warm caches
    ms = []
    for _ in range(repeat):
        for c in cases:
            t = time.perf_counter()
            search.retrieve(cfg, c["question"], embedder, db=db, rerank=bool(rerank_fn), rerank_fn=rerank_fn)
            ms.append((time.perf_counter() - t) * 1000)
    db.close()
    return {"p50_ms": _pct(ms, 50), "p95_ms": _pct(ms, 95), "n": len(ms)}


def tune(cfg: Config, cases: list[dict], embedder: Embedder) -> list[dict]:
    """Sweep the no-reranker thresholds over the same stored candidates. Each row reports
    recall on answerable questions and false injection on each unanswerable set."""
    db = idx.connect(cfg, create=False)
    now = time.time()
    gathered = []
    for case in cases:
        qvec = None
        try:
            v = np.asarray(embedder.embed([case["question"]])[0], dtype=np.float32)
            qvec = v / (np.linalg.norm(v) + 1e-12)
        except EmbedError:
            pass
        gathered.append((case, search.gather(cfg, db, case["question"], embedder, qvec=qvec, now=now)))
    out = []
    for mc in (0.5, 0.55, 0.58, 0.6, 0.62, 0.65, 0.7):
        for mo in (0.5, 0.67, 0.8, 1.01):  # 1.01 turns the word-overlap route off
            for mt in (1, 2, 3):
                c2 = cfg.override(min_cosine=mc, min_term_overlap=mo, min_content_terms=mt)
                hit = a = 0
                fi: dict[str, list[int]] = {}
                for case, g in gathered:
                    res = search.select(c2, g)
                    if case.get("expect_none"):
                        row = fi.setdefault(case.get("kind", "no_answer"), [0, 0])
                        row[1] += 1
                        row[0] += bool(res.selected)
                    else:
                        a += 1
                        hit += any(gold_match(c, set(case.get("gold") or [])) for c in res.selected)
                rates = {k: v[0] / v[1] for k, v in fi.items()}
                out.append({"min_cosine": mc, "min_term_overlap": mo, "min_content_terms": mt,
                            "recall_at_4": hit / a if a else 0, "false_injection": rates,
                            "worst_fi": max(rates.values(), default=0)})
    out.sort(key=lambda r: (-r["recall_at_4"], r["worst_fi"]))
    db.close()
    return out


def sample_docs(cfg: Config, n: int = 25, seed: int = 7) -> list[dict]:
    """A seeded sample of indexed notes with excerpts, for the agent to write questions from."""
    db = idx.connect(cfg, create=False)
    rows = db.execute("""SELECT d.id, d.title, d.source, d.path, d.type,
                (SELECT text FROM chunks c WHERE c.doc_id=d.id ORDER BY c.ord LIMIT 1) AS excerpt
                FROM documents d WHERE d.indexable=1 ORDER BY d.id""").fetchall()
    rows = [r for r in rows if r["excerpt"] and len(r["excerpt"]) > 300]
    random.Random(seed).shuffle(rows)
    out = [{"original": (r["source"] or "")[len("raw/notes/"):] if (r["source"] or "").startswith("raw/notes/") else r["path"],
            "title": r["title"], "type": r["type"], "excerpt": r["excerpt"][:900]} for r in rows[:n]]
    db.close()
    return out


def format_table(summary: dict) -> str:
    def f(x, pct=True):
        return "  -  " if x is None else (f"{x * 100:4.0f}%" if pct else f"{x:5.2f}")
    head = (f"{'variant':<24}{'recall@4':>9}{'top10':>7}{'prec':>6}{'falseabs':>9}"
            f"{'FI no-ans':>10}{'FI offtop':>10}{'FI near':>9}{'tok avg':>8}{'leaks':>6}")
    lines = [head, "-" * len(head)]
    for name, s in summary.items():
        fi = s.get("false_injection", {})
        lines.append(f"{name:<24}{f(s['recall_at_4']):>9}{f(s['gold_in_top10']):>7}{f(s['precision']):>6}"
                     f"{f(s['false_abstain']):>9}{f(fi.get('no_answer')):>10}{f(fi.get('offtopic')):>10}"
                     f"{f(fi.get('nearmiss')):>9}{s['tokens_mean']:>8.0f}{s['leaks']:>6}")
    if any(s.get("pointer_rate") and any(s["pointer_rate"].values()) for s in summary.values()):
        for name, s in summary.items():
            pr = s["pointer_rate"]
            lines.append(f"  pointers [{name}]: recall incl. pointers {f(s['pointer_recall'])}; prompts with pointers "
                         f"no-ans {f(pr.get('no_answer'))} / off-topic {f(pr.get('offtopic'))} / near-miss {f(pr.get('nearmiss'))}")
    lines.append("FI = false injection: share of prompts the notes can't answer that still got passages injected.")
    return "\n".join(lines)
