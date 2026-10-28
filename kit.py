#!/usr/bin/env python3
"""Deterministic RAG kit: one entry point.

    python3 kit.py assess <vault>            stage 1: is this worth it? (read-only, no installs)
    python3 kit.py minitest ...              stage 1: optional 5-question test
    python3 kit.py setup                     stage 2: create .venv and install the two dependencies
    python3 kit.py migrate plan|show|set|apply|revert
    python3 kit.py reindex | search | remember | eval | lint | report
    python3 kit.py install | uninstall       the prompt hook and/or search tool (--mode inject|tool|both)
    python3 kit.py mcp-serve                 the search_notes MCP server (your agent starts it)

Stage 1 uses only the Python standard library. Stage 2 runs inside .venv in this folder.
"""
from __future__ import annotations

import sys

if sys.version_info < (3, 11):  # before anything else, so an old system Python gets a clear message
    if sys.argv[1:2] == ["hook"]:
        sys.exit(0)  # a hook must never complain or block a prompt
    sys.exit(f"This kit needs Python 3.11 or newer (this is {sys.version.split()[0]}). "
             "Install one from python.org or Homebrew and run it with that: python3.12 kit.py ...")

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

NO_DEPS = {"assess", "minitest", "setup", "doctor", "install", "uninstall", "mcp-serve"}


def venv_python() -> Path:
    return ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def in_venv() -> bool:
    """Is this interpreter the kit's own venv? (Its python is a symlink to the base python, so comparing
    resolved executable paths can't tell them apart; the prefix can.)"""
    try:
        return Path(sys.prefix).resolve() == (ROOT / ".venv").resolve()
    except OSError:
        return False


def ensure_deps(cmd: str) -> None:
    """Stage 2 commands need pyyaml and numpy. If this interpreter lacks them, re-run
    inside the kit's venv."""
    if cmd in NO_DEPS:
        return
    try:
        import numpy, yaml  # noqa: F401
        return
    except ImportError:
        pass
    py = venv_python()
    if not py.exists() or in_venv():
        sys.exit("Stage 2 isn't set up yet. Run:  python3 kit.py setup")
    sys.exit(subprocess.call([str(py), str(ROOT / "kit.py"), *sys.argv[1:]]))


def bootstrap_agent_command(cmd: str) -> None:
    """`hook` and `mcp-serve` are started by your agent with whatever Python the installer found. They
    need pyyaml and numpy, which live in .venv. If this interpreter lacks them, hand over to the venv's
    Python. If the venv is gone (uninstalled, deleted, moved), a hook exits 0 and says nothing, so it
    never disturbs a prompt; the MCP server says why on stderr."""
    py = venv_python()
    if py.exists() and not in_venv():
        argv = [str(py), str(ROOT / "kit.py"), *sys.argv[1:]]  # always run inside the kit's own venv
        if os.name == "nt":
            sys.exit(subprocess.call(argv))
        os.execv(str(py), argv)
    try:
        import numpy, yaml  # noqa: F401
        return
    except ImportError:
        pass
    if cmd == "mcp-serve":
        sys.exit("notes-search: the kit's .venv is missing. Run: python3 kit.py setup")
    sys.exit(0)


def cfg():
    from rag import config
    return config.load(ROOT)


# ------------------------------------------------------------------ commands

def cmd_setup(a):
    if sys.version_info < (3, 11):
        sys.exit(f"Python 3.11 or newer is required (this is {sys.version.split()[0]}).")
    import sqlite3
    try:
        sqlite3.connect(":memory:").execute("create virtual table t using fts5(x)")
    except sqlite3.Error:
        sys.exit("This Python's SQLite was built without FTS5, which the keyword index needs. "
                 "Use python.org or Homebrew Python.")
    py = venv_python()
    try:
        if not py.exists():
            print("Creating .venv ...")
            subprocess.check_call([sys.executable, "-m", "venv", str(ROOT / ".venv")])
        pkgs = ["pyyaml>=6", "numpy>=1.26"] + (["sentence-transformers>=3"] if a.with_reranker else [])
        print("Installing:", ", ".join(pkgs), "(into .venv only)")
        subprocess.check_call([str(py), "-m", "pip", "install", "-q", "--disable-pip-version-check", *pkgs])
    except subprocess.CalledProcessError as exc:
        sys.exit(f"Setup failed ({' '.join(map(str, exc.cmd[:4]))} ... exit {exc.returncode}). On Debian or Ubuntu, "
                 "creating the venv needs the python3-venv package (sudo apt install python3-venv). Otherwise check "
                 "your network connection and try again.")
    print("Done. Everything this kit installs lives in", ROOT / ".venv")
    cmd_doctor(argparse.Namespace())


def cmd_doctor(a):
    import urllib.request
    ok = True
    print(f"python {sys.version.split()[0]}; venv {'present' if venv_python().exists() else 'MISSING (run setup)'}")
    from rag import config, embed
    c = config.load(ROOT)
    prov = c.get("embeddings", "provider")
    print(f"embeddings provider: {prov}")
    if prov == "ollama":
        url = c.get("embeddings", "ollama_url")
        try:
            with urllib.request.urlopen(url + "/api/tags", timeout=3) as r:
                names = [m["name"] for m in json.loads(r.read()).get("models", [])]
            model = c.get("embeddings", "ollama_model")
            if any(n.split(":")[0] == model.split(":")[0] for n in names):
                print(f"ollama reachable, {model} installed")
            else:
                ok = False
                print(f"ollama is running but {model} is not pulled. Run: ollama pull {model}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"ollama not reachable at {url} ({exc}). Install it from ollama.com, or set "
                  "provider = \"openai\" or \"none\" in config.toml.")
    elif prov == "openai":
        print("OPENAI_API_KEY", "set" if os.environ.get("OPENAI_API_KEY") else "NOT set (put it in .env)")
    from rag import rerank
    rs = rerank.status(c)
    if rs["pid"] or rs["token"]:
        if rs["pid"] and not rs["alive"]:
            print(f"reranker: recorded server (pid {rs['pid']}) is not running; it will be started on demand")
        elif rs["pid"] and rs["port"] != rs["config_port"]:
            ok = False
            print(f"reranker: MISMATCH. The running server is on port {rs['port']} but config.toml says "
                  f"{rs['config_port']}, so the hook can't reach it. Restart it or fix [rerank] port.")
        elif rs["pid"]:
            print(f"reranker: server running (pid {rs['pid']}, port {rs['port']})")
    print("corpus:", c.corpus, "(exists)" if c.corpus.exists() else "(not created yet)")
    print("all checks passed" if ok else "some checks need attention")


def cmd_assess(a):
    sys.argv = ["assess.py", *a.rest]
    import runpy
    runpy.run_path(str(ROOT / "assess.py"), run_name="__main__")


def cmd_minitest(a):
    sys.argv = ["minitest.py", *a.rest]
    import runpy
    runpy.run_path(str(ROOT / "minitest.py"), run_name="__main__")


def cmd_migrate(a):
    from rag import migrate
    c = cfg()
    if a.action == "plan":
        plan = migrate.plan_vault(c, Path(a.arg))
        print(migrate.summarize(plan))
        print("\nNothing in your vault was changed. Review with `migrate show`, adjust with `migrate set`, "
              "then `migrate apply`.")
    elif a.action == "show":
        print(migrate.summarize(migrate.load_plan(c)))
    elif a.action == "set":
        n = migrate.set_entries(c, a.arg, a.bundle, a.type, a.scope, False if a.exclude else None)
        print(f"updated {n} notes")
    elif a.action == "apply":
        res = migrate.apply_plan(c)
        if not res["ok"]:
            print(f"The linter gate failed with {len(res['problems'])} problems; nothing was written to the corpus.")
            for p in res["problems"][:40]:
                print("  ", p)
            sys.exit(1)
        print(f"Wrote {res['written']} documents ({res['files']} files including indexes).")
        for q in res["quarantined"]:
            print(f"  kept out of the corpus: {q['src']} ({q['reason']})")
        print("Next: python3 kit.py reindex")
    elif a.action == "revert":
        res = migrate.revert(c)
        print(f"removed {res['removed']} files", f"; kept {len(res['kept'])} you edited: {res['kept']}" if res["kept"] else "")
        print("Run `python3 kit.py reindex` to drop them from the index.")


def cmd_lint(a):
    from rag import lint
    c = cfg()
    if a.fix:
        print("anchors added:", lint.fix_anchors(c.knowledge))
    bundles = lint.load_schema(c.corpus / "schema.yaml") if (c.corpus / "schema.yaml").exists() else {}
    problems = lint.lint_corpus(c.knowledge, bundles)
    for p in problems:
        print(p)
    print(f"{len(problems)} problem{'s' if len(problems) != 1 else ''}")
    sys.exit(1 if problems else 0)


def cmd_reindex(a):
    from rag import embed, index
    c = cfg()
    def progress(done, total):
        if not a.quiet:
            print(f"  embedded {done}/{total} sections", file=sys.stderr, flush=True)
    try:
        rep = index.reindex(c, embed.build(c), force=a.force, progress=progress)
    except BlockingIOError as exc:
        if not a.quiet:
            print(f"{exc}. Wait for it to finish, or see {c.state / 'reindex.lock'}.")
        return
    if not a.quiet:
        print(rep.summary())
        for e in rep.errors[:10]:
            print("  problem:", e)


def cmd_search(a):
    from rag import embed, index, search
    c = cfg()
    db = index.connect(c, create=False)
    emb = embed.build(c)
    rr = None
    if a.rerank:
        from rag import rerank
        rr = rerank.score_in_process  # loads the model in this process; the hook uses `serve` instead
    res = search.retrieve(c, a.query, emb, db=db, rerank=True if a.rerank else None, rerank_fn=rr)
    if a.json:
        print(json.dumps({"block": res.block, "tokens": res.tokens, "abstained": res.abstained,
                          "warnings": res.warnings, "timings": res.timings}, indent=1))
        return
    print(f"query: {a.query}\nnote: {res.note}")
    for w in res.warnings:
        print("warning:", w)
    print(f"\nTop candidates (of {len(res.candidates)}):")
    for c_ in res.candidates[:12]:
        mark = "INJECT" if c_.selected else "      "
        cos = f"{c_.cosine:.2f}" if c_.cosine is not None else " -  "
        print(f" {mark} fused {c_.fusion:.2f} cos {cos} ovl {c_.overlap:.2f} "
              f"{c_.original or c_.path} > {c_.heading_path or 'top'}"
              + (f"   [{c_.rejected}]" if c_.rejected else ""))
    print(f"\nInjected {len(res.selected)} passages, ~{res.tokens} tokens"
          + (f" ({res.abstained})" if res.abstained else "") + ":\n")
    print(res.block or "(nothing)")
    print("\ntimings (ms):", {k: round(v, 1) for k, v in res.timings.items()})


def cmd_remember(a):
    from rag import embed, remember
    c = cfg()
    res = remember.remember(c, a.kind, a.text, embed.build(c), area=a.area, why=a.why,
                            solution=a.solution, root_cause=a.root_cause, rejected=a.rejected)
    if not res["ok"]:
        print("Refused by the linter gate:")
        for p in res["problems"]:
            print("  ", p)
        sys.exit(1)
    print("stored:", res["path"])
    print(res["report"])
    for e in res["errors"]:
        print("  problem:", e)


def cmd_eval(a):
    from rag import embed, evalrun
    c = cfg()
    emb = embed.build(c)
    if a.action == "sample":
        print(json.dumps(evalrun.sample_docs(c, a.n, a.seed), indent=1, ensure_ascii=False))
        return
    golden = Path(a.golden) if a.golden else c.corpus / "eval" / "golden.json"
    cases = evalrun.load_cases(c, golden)
    if a.action == "tune":
        rows = evalrun.tune(c, cases, emb)
        pick = [r for r in rows if r["worst_fi"] <= a.max_fi][:10] or rows[:10]
        print(f"best recall with false injection <= {a.max_fi:.0%} on every unanswerable set "
              "(or the closest rows if none qualify):")
        print("min_cosine overlap  min_terms  recall@4   FI no-answer / off-topic / near-miss")
        for r in pick:
            fi = r["false_injection"]
            print(f"{r['min_cosine']:>10} {min(r['min_term_overlap'], 1.0):>7} {r['min_content_terms']:>10}  {r['recall_at_4']:>7.0%}   "
                  f"{fi.get('no_answer', 0):>6.0%} / {fi.get('offtopic', 0):>6.0%} / {fi.get('nearmiss', 0):>6.0%}")
        return
    rr = None
    if a.rerank:
        from rag import rerank
        rr = rerank.score_in_process
    out = evalrun.evaluate(c, cases, emb, rerank_fn=rr)
    print(evalrun.format_table(out["summary"]))
    lat = evalrun.latency(c, cases, emb, rerank_fn=rr)
    out["latency"] = lat
    print(f"\nend-to-end retrieve(): p50 {lat['p50_ms']:.0f} ms, p95 {lat['p95_ms']:.0f} ms over {lat['n']} questions")
    dest = c.corpus / "eval"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "last.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def cmd_verify(a):
    """Stage 2 check: rerun the stage 1 questions with the full hybrid retrieval injected."""
    import minitest
    from rag import embed, index, search
    c = cfg()
    emb, db = embed.build(c), index.connect(c, create=False)

    def hybrid(question):
        res = search.retrieve(c, question, emb, db=db)
        return res.block, [{"relpath": s.original or s.path, "heading": s.heading_path} for s in res.selected]

    minitest.run_test(Path(a.vault), Path(a.questions), a.harness, Path(a.out), a.model, a.timeout,
                      arms=("hybrid",), hybrid_fn=hybrid)


def _paths(a):
    home = Path.home()
    return {"claude_hook": Path(a.claude_settings or home / ".claude" / "settings.json"),
            "claude_mcp": Path(a.claude_mcp or home / ".claude.json"),
            "codex_hook": Path(a.codex_hooks or home / ".codex" / "hooks.json"),
            "codex_mcp": Path(a.codex_config or home / ".codex" / "config.toml")}


def detect_harnesses(a) -> list[str]:
    """Which tools to install for: what the person asked for, else whichever is installed.
    Never create ~/.codex for a Claude-only user, or ~/.claude for a Codex-only one."""
    if a.harness in ("claude", "codex"):
        return [a.harness]
    if a.harness == "both":
        return ["claude", "codex"]
    home = Path.home()
    found = []
    if a.claude_settings or a.claude_mcp or (home / ".claude").is_dir() or (home / ".claude.json").exists():
        found.append("claude")
    if a.codex_hooks or a.codex_config or (home / ".codex").is_dir():
        found.append("codex")
    if not found:
        sys.exit("Couldn't find Claude Code (~/.claude) or Codex (~/.codex) on this machine. "
                 "Run again with --harness claude or --harness codex.")
    return found


def cmd_install(a):
    """Modes: inject (the prompt hook, the default), tool (the agent calls search_notes itself), or both."""
    from rag import install
    if not venv_python().exists():
        sys.exit("Stage 2 isn't set up yet, so a hook installed now would do nothing. Run: python3 kit.py setup")
    py = sys.executable  # the interpreter that ran this command; kit.py hands over to .venv by itself
    p = _paths(a)
    harnesses = detect_harnesses(a)
    print("Installing for:", ", ".join(harnesses), "(mode: " + a.mode + ")")
    for harness in harnesses:
        if a.mode in ("inject", "both"):
            res = install.install(p[f"{harness}_hook"], ROOT, py, harness)
            install.record_install(ROOT, f"{harness}_hook", p[f"{harness}_hook"])
            print(f"{harness}: hook {'replaced' if res['replaced'] else 'added'} in {p[f'{harness}_hook']}")
            if res["backup"]:
                print(f"  backup: {res['backup']}")
        if a.mode in ("tool", "both"):
            fn = install.install_claude_mcp if harness == "claude" else install.install_codex_mcp
            res = fn(p[f"{harness}_mcp"], ROOT, py)
            install.record_install(ROOT, f"{harness}_mcp", p[f"{harness}_mcp"])
            print(f"{harness}: search_notes tool added as MCP server '{res['name']}' in {p[f'{harness}_mcp']}")
            if res["backup"]:
                print(f"  backup: {res['backup']}")
        if harness == "codex":
            print("  Codex asks you to trust a new hook once: open Codex and run /hooks." if a.mode != "tool"
                  else "  Restart Codex so it picks up the new MCP server.")
            print("  Note: the Codex setup follows `codex mcp add --help` and a working Codex config, "
                  "but hasn't been run live yet.")
        else:
            print("  Restart Claude Code so it loads the new settings." if a.mode != "inject" else
                  "  Hooks load when a session starts; open a new Claude Code session.")


def cmd_uninstall(a):
    import shutil
    from rag import config, install, migrate
    p = _paths(a)
    # Everything we ever installed into: the default files, any file named on the command line, and
    # every file recorded at install time (project-level settings included).
    targets: dict[tuple[str, str], Path] = {}
    for kind, path in p.items():
        targets[(kind, str(path))] = path
    for rec in install.recorded_installs(ROOT):
        targets[(rec["kind"], rec["path"])] = Path(rec["path"])
    corpus = config.load(ROOT).corpus
    if a.remove_corpus:  # check first, so a refusal leaves everything else as it was
        migrate.check_removable(corpus)
    kinds = {"claude_hook": install.uninstall, "codex_hook": install.uninstall,
             "claude_mcp": install.uninstall_claude_mcp, "codex_mcp": install.uninstall_codex_mcp}
    removed_any = []
    for (kind, _), path in sorted(targets.items()):
        fn = kinds[kind]
        res = fn(path, ROOT) if kind.endswith("hook") or kind.endswith("mcp") else None
        if res and res["removed"]:
            removed_any.append(str(path))
            print(f"removed {res['removed']} {'hook' if kind.endswith('hook') else 'MCP server'} entries from {path}"
                  + (f" (backup {res['backup']})" if res["backup"] else ""))
    extra = [str(Path(r["path"])) for r in install.recorded_installs(ROOT) if r["path"] not in {str(x) for x in p.values()}]
    if extra:
        print("Also installed into these files, now cleaned:", ", ".join(extra))
    print("Backups named *.bak-* were left next to each settings file; they can hold your settings, so delete them when you're sure.")
    for d in (ROOT / ".state", ROOT / ".venv"):
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            print("removed", d)
    for f in (ROOT / install.INSTALLS_FILE, ROOT / ".kit-id"):
        f.unlink(missing_ok=True)
    if a.remove_corpus:
        removed = migrate.remove_corpus(corpus)
        print("removed from the corpus folder:", ", ".join(removed) or "nothing")
    else:
        print("Your corpus (okf/) was kept. Your original notes were never touched.")


def cmd_mcp_serve(a):
    from rag import mcp_server
    mcp_server.serve(cfg())


def cmd_hook(a):
    from rag import hook
    sys.exit(hook.main(header=a.header))


def cmd_serve(a):
    from rag import rerank
    rerank.serve(cfg())


def cmd_report(a):
    from rag import report
    path = report.build(cfg(), a)
    print("wrote", path)


def main(argv=None):
    p = argparse.ArgumentParser(prog="kit.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup"); s.add_argument("--with-reranker", action="store_true"); s.set_defaults(f=cmd_setup)
    sub.add_parser("doctor").set_defaults(f=cmd_doctor)
    s = sub.add_parser("assess"); s.add_argument("rest", nargs=argparse.REMAINDER); s.set_defaults(f=cmd_assess)
    s = sub.add_parser("minitest"); s.add_argument("rest", nargs=argparse.REMAINDER); s.set_defaults(f=cmd_minitest)
    s = sub.add_parser("migrate")
    s.add_argument("action", choices=["plan", "show", "set", "apply", "revert"])
    s.add_argument("arg", nargs="?", help="plan: the vault folder. set: a note path, folder or glob")
    s.add_argument("--bundle"); s.add_argument("--type"); s.add_argument("--scope", choices=["shared", "me"])
    s.add_argument("--exclude", action="store_true"); s.set_defaults(f=cmd_migrate)
    s = sub.add_parser("lint"); s.add_argument("--fix", action="store_true"); s.set_defaults(f=cmd_lint)
    s = sub.add_parser("reindex"); s.add_argument("--force", action="store_true"); s.add_argument("--quiet", action="store_true"); s.set_defaults(f=cmd_reindex)
    s = sub.add_parser("search"); s.add_argument("query"); s.add_argument("--json", action="store_true")
    s.add_argument("--rerank", action="store_true"); s.set_defaults(f=cmd_search)
    s = sub.add_parser("remember"); s.add_argument("kind", choices=["problem", "decision", "preference", "fact", "note"])
    s.add_argument("text"); s.add_argument("--area"); s.add_argument("--why"); s.add_argument("--solution")
    s.add_argument("--root-cause"); s.add_argument("--rejected"); s.set_defaults(f=cmd_remember)
    s = sub.add_parser("eval"); s.add_argument("action", choices=["run", "tune", "sample"], nargs="?", default="run")
    s.add_argument("--golden"); s.add_argument("--rerank", action="store_true")
    s.add_argument("--max-fi", type=float, default=0.10); s.add_argument("--n", type=int, default=25); s.add_argument("--seed", type=int, default=7); s.set_defaults(f=cmd_eval)
    for name, f in (("install", cmd_install), ("uninstall", cmd_uninstall)):
        s = sub.add_parser(name)
        s.add_argument("--claude-settings", help="settings.json holding hooks (default ~/.claude/settings.json)")
        s.add_argument("--claude-mcp", help="file holding mcpServers (default ~/.claude.json; use .mcp.json for one project)")
        s.add_argument("--codex-hooks", help="hooks.json to edit (default ~/.codex/hooks.json)")
        s.add_argument("--codex-config", help="config.toml holding MCP servers (default ~/.codex/config.toml)")
        if name == "install":
            s.add_argument("--harness", choices=["claude", "codex", "both"], default=None,
                           help="default: whichever of Claude Code and Codex is installed")
            s.add_argument("--mode", choices=["inject", "tool", "both"], default="inject",
                           help="inject: hook puts passages in the prompt (default). tool: the agent calls search_notes. both.")
        else:
            s.add_argument("--remove-corpus", action="store_true")
        s.set_defaults(f=f)
    s = sub.add_parser("hook"); s.add_argument("--harness", default="claude"); s.add_argument("--drag-kit")
    s.add_argument("--header", choices=["verify", "plain"], help="override injection_header from config.toml"); s.set_defaults(f=cmd_hook)
    sub.add_parser("serve").set_defaults(f=cmd_serve)
    s = sub.add_parser("mcp-serve", help="run the search_notes MCP server on stdio (started by your agent)")
    s.add_argument("--harness", default="claude"); s.add_argument("--drag-kit"); s.set_defaults(f=cmd_mcp_serve)
    s = sub.add_parser("verify"); s.add_argument("vault"); s.add_argument("--questions", required=True)
    s.add_argument("--harness", choices=["claude", "codex"], required=True); s.add_argument("--out", required=True)
    s.add_argument("--model"); s.add_argument("--timeout", type=int, default=180); s.set_defaults(f=cmd_verify)
    s = sub.add_parser("report"); s.add_argument("--vault"); s.add_argument("--queries", nargs="*")
    s.add_argument("--before"); s.add_argument("--after"); s.set_defaults(f=cmd_report)
    a = p.parse_args(argv)
    if a.cmd in ("hook", "mcp-serve"):
        bootstrap_agent_command(a.cmd)
    else:
        ensure_deps(a.cmd)
    a.f(a)


if __name__ == "__main__":
    main()
