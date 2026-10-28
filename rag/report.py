"""report.html: one self-contained page (no network, no scripts except the theme toggle).

Shows the vault stats and verdict from stage 1, the before/after test table, the eval
table, three sample queries with exactly what was injected for each (the tuning view),
the current settings, and how to uninstall.
"""
from __future__ import annotations

import html
import json
import time
from pathlib import Path

from . import embed, evalrun, index, search
from .config import Config
from .constants import SKOOL_LINE

CSS = """
:root{--bg:#fafaf8;--panel:#fff;--ink:#1c1c1a;--muted:#62625c;--line:#e3e2dc;--accent:#2f5d50;--hit:#1d7a4d;--miss:#a33a2a;--code:#f1f0ea}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#151614;--panel:#1d1e1b;--ink:#ecebe5;--muted:#a3a299;--line:#34352f;--accent:#7fbfa9;--hit:#6fcf9a;--miss:#f0907f;--code:#252622}}
:root[data-theme="dark"]{--bg:#151614;--panel:#1d1e1b;--ink:#ecebe5;--muted:#a3a299;--line:#34352f;--accent:#7fbfa9;--hit:#6fcf9a;--miss:#f0907f;--code:#252622}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:860px;margin:0 auto;padding:24px 16px 64px}
header{display:flex;justify-content:space-between;align-items:center;gap:12px}
h1{font-size:1.6rem;margin:.2em 0}h2{font-size:1.2rem;margin:2em 0 .5em;border-bottom:1px solid var(--line);padding-bottom:.3em}
button{font:inherit;background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:6px 12px;cursor:pointer}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0}
.verdict{border-left:4px solid var(--accent)}
.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.92rem}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
pre{background:var(--code);border:1px solid var(--line);border-radius:8px;padding:12px;overflow-x:auto;font-size:.82rem;white-space:pre-wrap;word-break:break-word}
code{background:var(--code);padding:1px 5px;border-radius:4px;font-size:.88em}
.muted{color:var(--muted)}.hit{color:var(--hit)}.miss{color:var(--miss)}
footer{margin-top:3em;padding-top:1em;border-top:1px solid var(--line)}
"""

JS = """
(function(){var r=document.documentElement,b=document.getElementById('theme');
function cur(){var t=r.getAttribute('data-theme');if(t)return t;return matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light'}
try{var s=localStorage.getItem('theme');if(s)r.setAttribute('data-theme',s)}catch(e){}
function label(){b.textContent=cur()==='dark'?'Light mode':'Dark mode'}
b.addEventListener('click',function(){var n=cur()==='dark'?'light':'dark';r.setAttribute('data-theme',n);try{localStorage.setItem('theme',n)}catch(e){}label()});label()})();
"""


def e(x) -> str:
    return html.escape(str(x))


def _pct(x):
    return "-" if x is None else f"{x * 100:.0f}%"


def _load(path):
    if path and Path(path).is_file():
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return None


def _test_table(before, after) -> str:
    arms = []
    for doc, label in ((before, "stage 1"), (after, "stage 2")):
        if not doc:
            continue
        for arm, t in doc["totals"].items():
            arms.append((f"{arm} ({label})", t))
    if not arms:
        return '<p class="muted">No mini-test results yet. Run the optional mini test (START.md, stage 1) and <code>python3 kit.py verify</code>.</p>'
    rows = ""
    for name, t in arms:
        tokens = (t.get("input_tokens") or 0) + (t.get("cache_read_tokens") or 0) + (t.get("cache_creation_tokens") or 0)
        rows += (f"<tr><td>{e(name)}</td><td>{t['hits']}/{t['runs']}</td><td>{t['wall_s']}s</td>"
                 f"<td>{tokens:,}</td><td>{t['failures']}</td></tr>")
    return ('<div class="table-wrap"><table><tr><th>how it answered</th><th>cited the right note</th>'
            '<th>total time</th><th>input tokens (all kinds)</th><th>failed runs</th></tr>' + rows + "</table></div>"
            '<p class="muted">Same questions each time. A hit means the right note was cited, not that the answer text was verified. '
            "Five questions is a small sample, so read it as a sanity check, not a benchmark.</p>")


def _eval_table(ev) -> str:
    if not ev:
        return '<p class="muted">No eval yet. Run <code>python3 kit.py eval</code>.</p>'
    rows = ""
    for name, s in ev["summary"].items():
        fi = s.get("false_injection", {})
        rows += (f"<tr><td>{e(name)}</td><td>{_pct(s['recall_at_4'])}</td><td>{_pct(s['precision'])}</td>"
                 f"<td>{_pct(fi.get('no_answer'))}</td><td>{_pct(fi.get('offtopic'))}</td><td>{_pct(fi.get('nearmiss'))}</td>"
                 f"<td>{s['tokens_mean']:.0f}</td><td>{s['leaks']}</td></tr>")
    lat = ev.get("latency")
    extra = f"<p class='muted'>End to end, one question through the whole pipeline: median {lat['p50_ms']:.0f} ms, 95th percentile {lat['p95_ms']:.0f} ms ({lat['n']} prompts, embedding call included).</p>" if lat else ""
    return ('<p>Recall is how often a passage from the right note was injected for a question your notes answer. '
            'False injection is how often something was injected for a prompt they do not answer: '
            '"no-answer" questions are ones that sound like your notes, "off-topic" prompts are coding, git, chat and general '
            'knowledge, and "near-miss" prompts reuse your notes\' vocabulary without being answered by them. Lower is better.</p>'
            '<div class="table-wrap"><table><tr><th>variant</th><th>recall</th><th>injected passages that were right</th>'
            '<th>false injection: no-answer</th><th>off-topic</th><th>near-miss</th>'
            '<th>avg tokens</th><th>leaks</th></tr>' + rows + "</table></div>" + extra)


def build(cfg: Config, args) -> Path:
    from . import config as config_mod
    assess_json = _load(cfg.corpus / "reports" / "assess.json")
    if args.vault:
        import importlib.util
        spec = importlib.util.spec_from_file_location("assess", Path(__file__).resolve().parent.parent / "assess.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        stats = mod.assess(Path(args.vault))
        assess_json = {"stats": stats, "verdict": mod.verdict(stats, mod.load_thresholds())}
    before, after = _load(args.before), _load(args.after)
    ev = _load(cfg.corpus / "eval" / "last.json")
    golden = cfg.corpus / "eval" / "golden.json"

    parts = []
    if assess_json:
        st, vd = assess_json["stats"], assess_json["verdict"]
        parts.append(f'<h2>Your vault</h2><div class="card"><b>{st["notes"]:,}</b> notes, <b>{st["words"]:,}</b> words, '
                     f'about <b>{st["est_tokens"]:,}</b> tokens (estimated as characters / 4). '
                     f'{st["notes_touched_per_month_avg"]} notes created or modified per month on average (by file date, a proxy). '
                     f'{st["links_per_note"]} links and {st["tags_per_note"]} tags per note.</div>')
        reasons = "".join(f"<li>{e(r)}</li>" for r in vd.get("reasons", []))
        cav = "".join(f"<li>{e(r)}</li>" for r in vd.get("caveats", []))
        parts.append(f'<h2>Verdict</h2><div class="card verdict"><b>{e(vd["headline"])}</b><ul>{reasons}</ul>'
                     f'<p class="muted">Caveats</p><ul class="muted">{cav}</ul></div>')
    parts.append(f"<h2>Before and after</h2>{_test_table(before, after)}")
    parts.append(f"<h2>Retrieval eval on your notes</h2>{_eval_table(ev)}")

    # Tuning view: three queries, exactly what would be injected.
    queries = list(args.queries or [])
    if not queries and golden.is_file():
        cases = evalrun.load_golden(golden)
        for kind in ("direct", "paraphrase"):
            q = next((c for c in cases if c.get("kind") == kind and not c.get("expect_none")), None)
            if q:
                queries.append(q["question"])
        none = next((c for c in cases if c.get("expect_none")), None)
        if none:
            queries.append(none["question"])
        if len(queries) < 3:
            queries += [c["question"] for c in cases if c["question"] not in queries][: 3 - len(queries)]
    queries = queries[:3]
    parts.append("<h2>What gets injected (the tuning view)</h2>"
                 "<p>Three real prompts and the exact text the hook adds above each one. "
                 "If a block is empty, the pipeline decided nothing in your notes answered it.</p>")
    if queries and cfg.db_path.exists():
        emb = embed.build(cfg)
        db = index.connect(cfg, create=False)
        for q in queries:
            res = search.retrieve(cfg, q, emb, db=db)
            cand = "".join(
                f"<tr><td>{'<b class=hit>injected</b>' if c.selected else 'left out'}</td><td>{c.fusion:.2f}</td>"
                f"<td>{'-' if c.cosine is None else f'{c.cosine:.2f}'}</td><td>{c.overlap:.2f}</td>"
                f"<td>{e(c.original or c.path)} &gt; {e(c.heading_path or 'top')}</td></tr>" for c in res.candidates[:6])
            parts.append(f'<div class="card"><b>{e(q)}</b><p class="muted">{len(res.selected)} passages, about {res.tokens} tokens'
                         f'{" - " + e(res.abstained) if res.abstained else ""}</p>'
                         f'<div class="table-wrap"><table><tr><th>decision</th><th>fused</th><th>cosine</th><th>word overlap</th><th>passage</th></tr>{cand}</table></div>'
                         f'<pre>{e(res.block) if res.block else "(nothing injected)"}</pre></div>')
        db.close()
    r = cfg.section("retrieval")
    knobs = "".join(f"<tr><td>{e(k)}</td><td>{e(v)}</td></tr>" for k, v in r.items())
    parts.append("<h2>The knobs</h2><p>Every one is in <code>config.toml</code> with a comment. Change a value, then "
                 f"run <code>python3 kit.py eval</code> to see what it did on your own questions.</p><div class='table-wrap'><table>{knobs}</table></div>")
    parts.append("<h2>Uninstall</h2><pre>python3 kit.py uninstall</pre>"
                 "<p>This removes only this kit's hook entries (your other hooks stay), the <code>.venv</code> folder and the index. "
                 "Your original notes are never modified. Your converted corpus (<code>okf/</code>) is kept unless you add "
                 "<code>--remove-corpus</code>. Settings files are backed up before every change.</p>")
    parts.append(f"<footer><p>{e(SKOOL_LINE)}</p></footer>")
    page = ("<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>Retrieval report</title><style>" + CSS + "</style></head><body><main><header><div><h1>Retrieval report</h1>"
            f"<span class='muted'>Generated {time.strftime('%Y-%m-%d %H:%M')}</span></div>"
            "<button id='theme' type='button'>Dark mode</button></header>" + "".join(parts) +
            "</main><script>" + JS + "</script></body></html>")
    out = cfg.root / "report.html"
    out.write_text(page, encoding="utf-8")
    return out
