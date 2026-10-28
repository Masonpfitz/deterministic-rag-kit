#!/usr/bin/env python3
"""minitest.py - the OPTIONAL mini test: agent browsing vs. a keyword-retrieval prompt.

You do not need this to use the kit. It exists so you can check, on YOUR notes,
whether passing retrieved passages to the model finds the right note more often
than letting the agent browse. Five questions, two arms each, ten headless runs.

    python3 minitest.py estimate --harness claude
    python3 minitest.py sample   /path/to/vault --n 5 --out sample.json
        (your coding agent reads sample.json and writes questions.json)
    python3 minitest.py run      /path/to/vault --questions questions.json \
                                 --harness claude --out results.json

The two arms
    browse   The agent runs inside your vault with read-only tools only
             (Read, Grep, Glob for Claude Code; a read-only sandbox for Codex)
             and is told to answer from the vault.
    keyword  This script builds a throwaway SQLite FTS5 index (in a temp folder),
             splits notes at markdown headings, runs an OR query of the question's
             content words, ranks with bm25, and pastes the top 4 sections into the
             prompt. The model gets NO tools.

Both arms must answer in exactly this JSON shape, and are graded by code:
    {"answer": "...", "sources": ["relative/path.md", ...]}

What the grading measures, and what it does not
    A "hit" means a gold note path appears in the sources the model cited. It does
    NOT check that the answer text is correct. Read a few answers yourself.

expect_none cases (questions the vault cannot answer). A hit is:
    sources is empty, OR the answer text matches a "not found" phrase (see
    NOT_FOUND_RE). A confident answer with sources is a miss.

----------------------------------------------------------------------------
ISOLATION (verified empirically on 2026-10-27, see DEV-NOTES.md)
----------------------------------------------------------------------------
The runs must not trigger YOUR hooks, MCP servers, plugins, skills or CLAUDE.md,
or the numbers measure your setup instead of the method.

Claude Code 2.1.x argv (prompt goes in on stdin):
    claude -p --safe-mode --strict-mcp-config --disable-slash-commands
           --permission-mode dontAsk --no-session-persistence
           --output-format json --tools "Read,Grep,Glob"      (browse arm)
           --tools ""                                         (keyword arm)
  --safe-mode turns off hooks, MCP servers, plugins, skills, CLAUDE.md and custom
  commands but keeps OAuth login working. (--bare also isolates but needs an
  ANTHROPIC_API_KEY and ignores OAuth.) Canary: a project-level UserPromptSubmit
  hook and a CLAUDE.md in the working folder both fired without the flag and did
  not fire with it; the init event listed 0 MCP servers, 0 skills, 0 slash
  commands and only the tools we asked for.
  If ANTHROPIC_API_KEY is set in your environment, Claude Code bills that key
  instead of your subscription.

Codex CLI 0.164 argv (prompt on stdin, CODEX_HOME points at a temp folder):
    codex exec --json --ephemeral --skip-git-repo-check --sandbox read-only
               --ignore-user-config --ignore-rules -c project_doc_max_bytes=0
               -C <dir> -
  --ignore-user-config alone is NOT enough: a global AGENTS.md in your Codex home
  still reached the model in our canary. So we run with a temporary CODEX_HOME
  that holds only a COPY of your auth.json, and delete it afterwards. The auth
  file is never printed or logged. `-c project_doc_max_bytes=0` stops the vault's
  own AGENTS.md from loading. Codex has no "no tools" switch, so the keyword arm
  runs in an empty temp folder, read-only, and we count any tool calls it makes.
  If no auth.json is found we fall back to your real CODEX_HOME and print a
  warning, because isolation is then incomplete.

Standard library only. Python 3.11+.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from assess import iter_notes, split_frontmatter

TOP_K = 4                # sections pasted into the keyword-arm prompt
PASSAGE_CHARS = 1800     # cap per pasted section so one huge section cannot flood the prompt
MIN_WORDS_FOR_SAMPLE = 120
EXCERPT_CHARS = 1500
DEFAULT_TIMEOUT = 180

ANSWER_FORMAT = (
    'Reply with ONLY this JSON object and nothing else:\n'
    '{"answer": "<your answer>", "sources": ["<path/relative/to/vault.md>", ...]}\n'
    'List every note you relied on in "sources", as paths relative to the vault root. '
    'If the notes do not contain the answer, say so in "answer" and use "sources": [].'
)

# Phrases that count as "the vault does not have this" for expect_none cases.
NOT_FOUND_RE = re.compile(
    r"(not found|no (relevant |matching )?(information|notes?|answer|mention|record)|"
    r"(can(no|')t|could(n't| not)|unable to|did(n't| not)) (find|locate)|"
    r"(isn't|is not|aren't|are not|not) (covered|mentioned|present|discussed|in the (vault|notes))|"
    r"(don't|do not|doesn't|does not) (have|contain|say|mention|cover))",
    re.IGNORECASE,
)

STOPWORDS = set(
    "a an and are as at be but by can did do does for from has have how i if in into is it its "
    "me my of on or our so than that the their them then there these they this to was we were "
    "what when where which who whom why will with would you your about after also been being "
    "both each other some such than too very".split()
)


# ==========================================================================
# Estimates (printed BEFORE anything runs)
# ==========================================================================
# These are ESTIMATES from a handful of runs on one machine. `run` prints measured
# totals when it finishes; trust those over these.
ESTIMATES = {
    "claude": {
        "wall": ("a first measurement with a small model took about 13 s for 4 runs (2-5 s each). "
                 "Expect roughly 1 to 6 minutes for 10 runs, longer with a large model or a big vault, "
                 "since browse runs make several tool calls"),
        "usage": (
            "browse runs read files and re-send context each turn: expect on the order of "
            "10k-150k cached input tokens per run, growing with how much the agent reads; keyword "
            "runs about 1k-10k. Claude Code's reported 'input' counts only uncached tokens, so look at cache-read too. "
            "Dollar cost is reported by the CLI in the results file; on a subscription it "
            "counts against your usage limit instead of billing you."
        ),
    },
    "codex": {
        "wall": ("a first measurement took about 9 s for a browse run and 5 s for a keyword run on a 12-note vault. "
                 "Expect roughly 1 to 8 minutes for 10 runs, longer on a big vault"),
        "usage": (
            "Codex sends a large fixed prompt: about 25k input tokens for a keyword run and "
            "50k+ for a browse run (part of it cached; Codex's input count includes the cached part). Codex does not report a dollar cost; it "
            "counts against your ChatGPT plan or API usage."
        ),
    },
}


def estimate_text(harness: str, n_questions: int = 5) -> str:
    e = ESTIMATES[harness]
    return (
        f"ESTIMATE ({harness}, {n_questions} questions x 2 arms = {n_questions * 2} headless runs)\n"
        f"  Time:  {e['wall']}\n"
        f"  Usage: {e['usage']}\n"
        "  These are estimates until measured. They use YOUR own subscription or API usage.\n"
        "  The run prints measured totals when it finishes."
    )


# ==========================================================================
# Sampling
# ==========================================================================
def note_title(text: str, path: Path) -> str:
    fm, body = split_frontmatter(text)
    if fm:
        m = re.search(r"^title\s*:\s*(.+)$", fm, re.MULTILINE | re.IGNORECASE)
        if m:
            return m.group(1).strip().strip("'\"")
    m = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
    return m.group(1).strip() if m else path.stem


def sample_notes(vault: Path, n: int = 5, seed: int | None = None, min_words: int = MIN_WORDS_FOR_SAMPLE) -> dict:
    """Pick n random notes with enough text to ask a real question about.

    Candidates are sorted by path first, so the same seed gives the same sample
    on the same vault regardless of filesystem ordering.
    """
    vault = vault.expanduser().resolve()
    seed = seed if seed is not None else random.SystemRandom().randrange(2**31)
    cands = []
    for p in iter_notes(vault):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        words = len(text.split())
        if words >= min_words:
            cands.append((p.relative_to(vault).as_posix(), p, text, words))
    cands.sort(key=lambda c: c[0])
    rng = random.Random(seed)
    chosen = rng.sample(cands, min(n, len(cands)))
    return {
        "version": 1,
        "seed": seed,
        "vault": str(vault),
        "candidates": len(cands),
        "instructions_for_agent": (
            "Read each note below. For each, write ONE question that a person could ask "
            "without knowing the note exists, answerable from that note, and that does not "
            "quote the note's title. Output a questions file in exactly this shape: "
            '{"version":1,"cases":[{"id":"q1","question":"...","gold":["relative/path.md"],'
            '"expect_none":false}]}. Optionally add one case with "expect_none": true, "gold": [] '
            "about something the vault does not cover."
        ),
        "notes": [
            {
                "relpath": rel,
                "title": note_title(text, p),
                "excerpt": split_frontmatter(text)[1].strip()[:EXCERPT_CHARS],
                "words": words,
            }
            for rel, p, text, words in chosen
        ],
    }


# ==========================================================================
# Keyword arm: FTS5 index over heading-split sections
# ==========================================================================
def split_sections(text: str, relpath: str) -> list[dict]:
    """Split a note at markdown headings. Lines inside code fences are not headings.

    Each section keeps its heading trail so the pasted passage says where it came from.
    """
    _, body = split_frontmatter(text)
    title = Path(relpath).stem
    sections: list[dict] = []
    trail: list[tuple[int, str]] = []
    buf: list[str] = []
    in_fence = False
    cur_heading = title

    def flush():
        content = "\n".join(buf).strip()
        if content:
            sections.append({"relpath": relpath, "heading": cur_heading, "body": content})
        buf.clear()

    for line in body.split("\n"):
        if re.match(r"^\s{0,3}(```|~~~)", line):
            in_fence = not in_fence
        m = None if in_fence else re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if m:
            flush()
            level = len(m.group(1))
            trail = [t for t in trail if t[0] < level] + [(level, m.group(2))]
            cur_heading = title + " > " + " > ".join(t[1] for t in trail)
        else:
            buf.append(line)
    flush()
    return sections


def content_words(question: str) -> list[str]:
    seen, out = set(), []
    for w in re.findall(r"[A-Za-z0-9]+", question.lower()):
        if len(w) >= 3 and w not in STOPWORDS and w not in seen:
            seen.add(w)
            out.append(w)
    return out


class KeywordIndex:
    """Throwaway SQLite FTS5 index. Lives in a temp dir; the vault is only read."""

    def __init__(self, vault: Path, workdir: Path):
        self.db = sqlite3.connect(str(workdir / "index.sqlite"))
        try:
            # porter = light stemming ("meetings" matches "meeting"). unicode61 = sane tokenizing.
            self.db.execute(
                "CREATE VIRTUAL TABLE s USING fts5(relpath UNINDEXED, heading, body, "
                "tokenize='porter unicode61')"
            )
        except sqlite3.OperationalError as e:
            raise RuntimeError(
                "This Python's SQLite was built without FTS5, so the keyword arm cannot run "
                f"({e}). The official python.org builds and Homebrew Python include it."
            ) from e
        self.sections = 0
        for p in iter_notes(vault):
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = p.relative_to(vault).as_posix()
            for s in split_sections(text, rel):
                self.db.execute(
                    "INSERT INTO s(relpath, heading, body) VALUES (?,?,?)",
                    (s["relpath"], s["heading"], s["body"]),
                )
                self.sections += 1
        self.db.commit()

    def search(self, question: str, k: int = TOP_K) -> list[dict]:
        words = content_words(question)
        if not words:
            return []
        query = " OR ".join(f'"{w}"' for w in words)  # quoted so FTS syntax words are literal
        # bm25() is lower-is-better; weights: relpath 0 (unindexed), heading 2, body 1.
        rows = self.db.execute(
            "SELECT relpath, heading, body, bm25(s, 0.0, 2.0, 1.0) AS score FROM s "
            "WHERE s MATCH ? ORDER BY score LIMIT ?",
            (query, k),
        ).fetchall()
        return [{"relpath": r, "heading": h, "body": b[:PASSAGE_CHARS], "score": sc} for r, h, b, sc in rows]

    def close(self):
        self.db.close()


# ==========================================================================
# Prompts
# ==========================================================================
def browse_prompt(question: str) -> str:
    return (
        "You are answering a question from a folder of markdown notes. The current folder "
        "is the vault. Use only your read-only tools (list, search and read files) to find "
        "the answer. Do not use outside knowledge.\n\n"
        f"Question: {question}\n\n{ANSWER_FORMAT}"
    )


def hybrid_prompt(question: str, block: str) -> str:
    """Stage 2 verify: the same question, with the kit's full retrieval block injected."""
    return (
        "Answer the question using ONLY the retrieved notes below, which were found by the "
        "kit's hybrid retrieval. You have no tools. Do not use outside knowledge. The notes may "
        "not contain the answer. In `sources`, list the `original` path of each note you used.\n\n"
        f"{block or '(nothing was retrieved)'}\n\nQuestion: {question}\n\n{ANSWER_FORMAT}"
    )


def keyword_prompt(question: str, passages: list[dict]) -> str:
    if passages:
        blocks = "\n\n".join(
            f"--- PASSAGE {i} | path: {p['relpath']} | section: {p['heading']} ---\n{p['body']}"
            for i, p in enumerate(passages, 1)
        )
    else:
        blocks = "(no passages matched)"
    return (
        "Answer the question using ONLY the passages below, which were retrieved from the "
        "user's notes by keyword search. You have no tools. Do not use outside knowledge. "
        "The passages may not contain the answer.\n\n"
        f"{blocks}\n\nQuestion: {question}\n\n{ANSWER_FORMAT}"
    )


# ==========================================================================
# Grading
# ==========================================================================
def parse_answer(text: str) -> dict | None:
    """Pull {"answer","sources"} out of model output. Tolerates code fences and chatter."""
    if not text:
        return None
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, re.DOTALL | re.IGNORECASE)
    candidates = [t]
    if fence:
        candidates.insert(0, fence.group(1).strip())
    first, last = t.find("{"), t.rfind("}")
    if first != -1 and last > first:
        candidates.append(t[first:last + 1])
    for c in candidates:
        try:
            obj = json.loads(c)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict) and "answer" in obj:
            src = obj.get("sources", [])
            if isinstance(src, str):
                src = [src]
            if not isinstance(src, list):
                src = []
            return {"answer": str(obj.get("answer", "")), "sources": [str(s) for s in src if s]}
    return None


def normalize_path(p: str, vault: str | None = None) -> str:
    """Make a cited path comparable: forward slashes, no ./, no vault prefix,
    no [[ ]], no #heading or |alias, lowercase, and ".md" added if there is no extension
    (Obsidian links drop it)."""
    s = p.strip().replace("\\", "/")
    s = s.strip("[]").split("|")[0].split("#")[0].strip()
    if vault:
        v = vault.replace("\\", "/").rstrip("/") + "/"
        if s.startswith(v):
            s = s[len(v):]
    while s.startswith("./"):
        s = s[2:]
    s = s.lstrip("/")
    if s and not re.search(r"\.[A-Za-z0-9]{1,5}$", s):
        s += ".md"
    return s.casefold()


def path_matches(source: str, gold: str, vault: str | None = None) -> bool:
    """True if a cited source names the gold note.

    Exact relative path matches. A bare note name ("Meeting Notes") matches the gold
    file's basename, and a partial path ("sub/note.md") matches the tail of the gold
    path, the way Obsidian's shortest-path links work. If two folders hold notes with
    the same name, a bare name will count as a hit for either; that is a known
    generosity of this grader.
    """
    s, g = normalize_path(source, vault), normalize_path(gold)
    return bool(s) and (s == g or g.endswith("/" + s))


def grade(case: dict, parsed: dict | None, vault: str | None = None) -> dict:
    if parsed is None:
        return {"hit": False, "reason": "unparseable"}
    sources = [s for s in parsed["sources"] if normalize_path(s, vault)]
    if case.get("expect_none"):
        if not sources:
            return {"hit": True, "reason": "no sources cited"}
        if NOT_FOUND_RE.search(parsed["answer"]):
            return {"hit": True, "reason": "answer says not found"}
        return {"hit": False, "reason": "answered with sources when nothing was expected"}
    hit = any(path_matches(s, g, vault) for s in sources for g in case.get("gold", []))
    return {"hit": hit, "reason": "gold cited" if hit else "gold not cited"}


# ==========================================================================
# Running the CLIs (runner is injectable so tests never call a model)
# ==========================================================================
@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    wall_s: float
    timed_out: bool = False


Runner = Callable[[list, "str | None", Path, dict, int], RunResult]


def subprocess_runner(argv, stdin_text, cwd, env, timeout) -> RunResult:
    t0 = time.monotonic()
    try:
        p = subprocess.run(
            argv, input=stdin_text, cwd=str(cwd), env=env, capture_output=True,
            text=True, timeout=timeout, encoding="utf-8", errors="replace",
        )
        return RunResult(p.returncode, p.stdout, p.stderr, time.monotonic() - t0)
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        return RunResult(-1, out, "", time.monotonic() - t0, timed_out=True)
    except FileNotFoundError as e:
        return RunResult(127, "", str(e), time.monotonic() - t0)


def empty_usage() -> dict:
    return {"input_tokens": None, "output_tokens": None, "cache_read_tokens": None,
            "cache_creation_tokens": None, "cost_usd": None}


class ClaudeHarness:
    name = "claude"
    exe = "claude"

    def argv(self, arm: str, model: str | None, cwd: Path) -> list:
        a = [self.exe, "-p", "--safe-mode", "--strict-mcp-config", "--disable-slash-commands",
             "--permission-mode", "dontAsk", "--no-session-persistence", "--output-format", "json"]
        if model:
            a += ["--model", model]
        # Browse: read-only tools only. Keyword: no tools at all (empty string disables them).
        a += ["--tools", "Read,Grep,Glob" if arm == "browse" else ""]
        return a

    def env(self) -> dict:
        return {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}  # the model must not see it

    def cwd(self, arm: str, vault: Path, scratch: Path) -> Path:
        return vault if arm == "browse" else scratch

    def parse(self, stdout: str) -> dict:
        """-> {text, usage, error, tool_calls}"""
        try:
            d = json.loads(stdout)
        except (json.JSONDecodeError, ValueError):
            return {"text": "", "usage": empty_usage(), "error": "CLI output was not JSON", "tool_calls": None}
        u = d.get("usage") or {}
        usage = {
            "input_tokens": u.get("input_tokens"),
            "output_tokens": u.get("output_tokens"),
            "cache_read_tokens": u.get("cache_read_input_tokens"),
            "cache_creation_tokens": u.get("cache_creation_input_tokens"),
            "cost_usd": d.get("total_cost_usd"),
        }
        err = (d.get("result") or "CLI reported an error") if d.get("is_error") else None
        return {"text": d.get("result") or "", "usage": usage, "error": err, "tool_calls": None}

    def setup(self):  # no per-run state needed
        return _null_ctx()

    def warnings(self) -> list:
        w = []
        if os.environ.get("ANTHROPIC_API_KEY"):
            w.append("ANTHROPIC_API_KEY is set, so Claude Code will bill that API key rather than your subscription.")
        return w


class CodexHarness:
    name = "codex"
    exe = "codex"

    def __init__(self):
        self.home: Path | None = None
        self.isolated = False

    def argv(self, arm: str, model: str | None, cwd: Path) -> list:
        a = [self.exe, "exec", "--json", "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only",
             "--ignore-user-config", "--ignore-rules", "-c", "project_doc_max_bytes=0", "-C", str(cwd)]
        if model:
            a += ["-m", model]
        return a + ["-"]  # "-" = read the prompt from stdin

    def env(self) -> dict:
        e = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        if self.home:
            e["CODEX_HOME"] = str(self.home)
        return e

    def cwd(self, arm: str, vault: Path, scratch: Path) -> Path:
        return vault if arm == "browse" else scratch

    def parse(self, stdout: str) -> dict:
        text, usage, err, tools = "", empty_usage(), None, 0
        for line in stdout.splitlines():
            try:
                ev = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            t = ev.get("type")
            item = ev.get("item") or {}
            if t == "item.completed":
                if item.get("type") == "agent_message":
                    text = item.get("text", text)
                elif item.get("type") not in ("error", "reasoning", None):
                    tools += 1  # command execution, file reads, etc.
            elif t == "turn.completed":
                u = ev.get("usage") or {}
                usage["input_tokens"] = (usage["input_tokens"] or 0) + (u.get("input_tokens") or 0)
                usage["output_tokens"] = (usage["output_tokens"] or 0) + (u.get("output_tokens") or 0)
                usage["cache_read_tokens"] = (usage["cache_read_tokens"] or 0) + (u.get("cached_input_tokens") or 0)
            elif t in ("turn.failed", "error"):
                err = json.dumps(ev.get("error") or ev.get("message") or ev)[:300]
        return {"text": text, "usage": usage, "error": err, "tool_calls": tools}

    @contextmanager
    def setup(self):
        """Temp CODEX_HOME holding only a copy of auth.json. Deleted on exit."""
        src_home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
        auth = src_home / "auth.json"
        tmp = Path(tempfile.mkdtemp(prefix="rag-kit-codex-"))
        try:
            if auth.is_file():
                shutil.copyfile(auth, tmp / "auth.json")
                os.chmod(tmp / "auth.json", 0o600)
                self.home, self.isolated = tmp, True
            else:
                self.home, self.isolated = None, False
            yield
        finally:
            shutil.rmtree(tmp, ignore_errors=True)  # removes the auth copy too
            self.home = None

    def warnings(self) -> list:
        w = []
        if not self.isolated:
            w.append(
                "WARNING: no Codex auth.json found, so the run uses your real Codex home. Your own "
                "AGENTS.md, hooks and MCP servers may affect the results. Isolation is NOT complete."
            )
        w.append(
            "Codex has no 'no tools' switch. The keyword arm runs read-only in an empty folder "
            "and any tool calls are counted in the results file."
        )
        w.append("Codex does not report a dollar cost, so cost is left blank.")
        return w


@contextmanager
def _null_ctx():
    yield


HARNESSES = {"claude": ClaudeHarness, "codex": CodexHarness}


def run_arm(harness, arm: str, case: dict, vault: Path, scratch: Path, index: KeywordIndex | None,
            model: str | None, timeout: int, runner: Runner, hybrid_fn=None) -> dict:
    """Run one question through one arm. Never raises: failures become recorded results."""
    rec = {"arm": arm, "status": "ok", "hit": False, "reason": "", "wall_s": None, "answer": "",
           "sources": [], "error": None, "tool_calls": None, **empty_usage()}
    try:
        if arm == "browse":
            prompt = browse_prompt(case["question"])
            rec["retrieved"] = None
        elif arm == "hybrid":
            block, retrieved = hybrid_fn(case["question"])
            prompt = hybrid_prompt(case["question"], block)
            rec["retrieved"] = retrieved
        else:
            passages = index.search(case["question"])
            prompt = keyword_prompt(case["question"], passages)
            rec["retrieved"] = [{"relpath": p["relpath"], "heading": p["heading"]} for p in passages]
        cwd = harness.cwd(arm, vault, scratch)
        res = runner(harness.argv(arm, model, cwd), prompt, cwd, harness.env(), timeout)
        rec["wall_s"] = round(res.wall_s, 2)
        if res.timed_out:
            rec.update(status="timeout", reason="timeout", error=f"no answer within {timeout}s")
            return rec
        parsed_out = harness.parse(res.stdout)
        rec.update({k: parsed_out["usage"][k] for k in empty_usage()})
        rec["tool_calls"] = parsed_out["tool_calls"]
        if res.returncode != 0 or parsed_out["error"]:
            rec.update(status="error", reason="cli error",
                       error=(parsed_out["error"] or res.stderr.strip()[-300:] or f"exit code {res.returncode}"))
            return rec
        parsed = parse_answer(parsed_out["text"])
        g = grade(case, parsed, str(vault))
        rec.update(hit=g["hit"], reason=g["reason"])
        if parsed is None:
            rec.update(status="unparseable", error="could not find the JSON answer", answer=parsed_out["text"][:500])
        else:
            rec.update(answer=parsed["answer"], sources=parsed["sources"])
    except Exception as e:  # one bad run must not kill the other nine
        rec.update(status="error", reason="exception", error=f"{type(e).__name__}: {e}")
    return rec


def load_questions(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("version") != 1 or not isinstance(data.get("cases"), list) or not data["cases"]:
        raise ValueError('questions file must look like {"version":1,"cases":[{"id","question","gold","expect_none"}]}')
    for c in data["cases"]:
        if not c.get("id") or not c.get("question"):
            raise ValueError(f"every case needs an id and a question: {c}")
        c.setdefault("gold", [])
        c.setdefault("expect_none", False)
        if not c["expect_none"] and not c["gold"]:
            raise ValueError(f"case {c['id']} has no gold path and is not expect_none")
    return data["cases"]


def summarize(results: list[dict]) -> dict:
    """Per-arm totals over every recorded run (failures included, counted as misses)."""
    out = {}
    for arm in ("browse", "keyword", "hybrid"):
        rs = [r for c in results for r in c["arms"] if r["arm"] == arm]
        if not rs:
            continue
        def tot(k):
            vals = [r[k] for r in rs if r[k] is not None]
            return sum(vals) if vals else None
        out[arm] = {
            "runs": len(rs),
            "hits": sum(r["hit"] for r in rs),
            "failures": sum(r["status"] in ("timeout", "error", "unparseable") for r in rs),
            "wall_s": round(sum(r["wall_s"] or 0 for r in rs), 1),
            "input_tokens": tot("input_tokens"), "output_tokens": tot("output_tokens"),
            "cache_read_tokens": tot("cache_read_tokens"), "cache_creation_tokens": tot("cache_creation_tokens"),
            "cost_usd": (round(tot("cost_usd"), 4) if tot("cost_usd") is not None else None),
        }
    return out


def print_table(results: list[dict], totals: dict) -> None:
    f = lambda v: "-" if v is None else f"{v:,}"
    print(f"\n{'case':<8}{'arm':<9}{'result':<10}{'status':<12}{'secs':>7}{'input':>10}{'out tok':>9}{'cache rd':>10}{'cost $':>9}")
    for c in results:
        for r in c["arms"]:
            print(f"{c['id']:<8}{r['arm']:<9}{('HIT' if r['hit'] else 'miss'):<10}{r['status']:<12}"
                  f"{(r['wall_s'] if r['wall_s'] is not None else '-'):>7}{f(r['input_tokens']):>10}"
                  f"{f(r['output_tokens']):>9}{f(r['cache_read_tokens']):>10}"
                  f"{('-' if r['cost_usd'] is None else format(r['cost_usd'], '.4f')):>9}")
    print("\nTOTALS (failures and timeouts count as misses)")
    for arm, t in totals.items():
        print(f"  {arm:<8} {t['hits']}/{t['runs']} hits, {t['failures']} failed, {t['wall_s']}s, "
              f"in {f(t['input_tokens'])} / out {f(t['output_tokens'])} / cache-read {f(t['cache_read_tokens'])} tokens, "
              f"cost {'not reported' if t['cost_usd'] is None else '$' + format(t['cost_usd'], '.4f')}")
    print("\nA hit means a gold note was cited. It does not prove the answer text was right.")


def run_test(vault: Path, questions: Path, harness_name: str, out: Path, model: str | None = None,
             timeout: int = DEFAULT_TIMEOUT, runner: Runner = subprocess_runner, quiet: bool = False,
             arms: tuple = ("browse", "keyword"), hybrid_fn=None) -> dict:
    vault = vault.expanduser().resolve()
    cases = load_questions(questions)
    harness = HARNESSES[harness_name]()
    say = (lambda *a: None) if quiet else print
    if arms == ("browse", "keyword"):
        say(estimate_text(harness_name, len(cases)) + "\n")
    if runner is subprocess_runner and shutil.which(harness.exe) is None:
        raise SystemExit(f"error: `{harness.exe}` not found on PATH")

    t0 = time.monotonic()
    results = []
    with tempfile.TemporaryDirectory(prefix="rag-kit-mini-") as td, harness.setup():
        scratch = Path(td)
        for w in harness.warnings():
            say(w)
        (scratch / "idx").mkdir()
        index = KeywordIndex(vault, scratch / "idx") if "keyword" in arms else None
        if index:
            say(f"Indexed {index.sections} sections for the keyword arm.\n")
        try:
            for case in cases:
                done = []
                for arm in arms:
                    say(f"  {case['id']} / {arm} ...")
                    done.append(run_arm(harness, arm, case, vault, scratch, index, model, timeout, runner, hybrid_fn))
                results.append({"id": case["id"], "question": case["question"], "gold": case["gold"],
                                "expect_none": case["expect_none"], "arms": done})
        finally:
            if index:
                index.close()
        warnings = harness.warnings()
    totals = summarize(results)
    doc = {
        "version": 1, "harness": harness_name, "model": model or "(CLI default)",
        "vault": str(vault), "timeout_s": timeout,
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_wall_s": round(time.monotonic() - t0, 1),
        "warnings": warnings, "totals": totals, "cases": results,
    }
    Path(out).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    if not quiet:
        print_table(results, totals)
        print(f"\nMeasured wall time for the whole run: {doc['total_wall_s']}s. Wrote {out}")
    return doc


# ==========================================================================
# CLI
# ==========================================================================
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Optional mini test: agent browsing vs keyword retrieval.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("estimate", help="print expected time and usage before running")
    e.add_argument("--harness", choices=HARNESSES, required=True)

    s = sub.add_parser("sample", help="randomly sample notes for your agent to write questions about")
    s.add_argument("vault", type=Path)
    s.add_argument("--n", type=int, default=5)
    s.add_argument("--seed", type=int)
    s.add_argument("--min-words", type=int, default=MIN_WORDS_FOR_SAMPLE, help="skip notes shorter than this (default 120)")
    s.add_argument("--out", type=Path, required=True)

    r = sub.add_parser("run", help="run both arms for every question")
    r.add_argument("vault", type=Path)
    r.add_argument("--questions", type=Path, required=True)
    r.add_argument("--harness", choices=HARNESSES, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--model")
    r.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="seconds per run (default 180)")

    a = ap.parse_args(argv)
    if a.cmd == "estimate":
        print(estimate_text(a.harness))
        return 0
    if a.cmd == "sample":
        vault = a.vault.expanduser().resolve()
        out = a.out.expanduser().resolve()
        if vault in out.parents:
            ap.error("--out must be outside the vault")
        doc = sample_notes(vault, a.n, a.seed, a.min_words)
        if len(doc["notes"]) < a.n:
            print(f"note: only {len(doc['notes'])} notes have >= {a.min_words} words", file=sys.stderr)
        out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        print(f"Wrote {out} ({len(doc['notes'])} notes, seed {doc['seed']})")
        return 0
    run_test(a.vault, a.questions, a.harness, a.out, a.model, a.timeout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
