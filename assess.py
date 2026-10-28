#!/usr/bin/env python3
"""assess.py - measure a markdown vault and say whether retrieval is worth building.

Usage:
    python3 assess.py /path/to/vault [--json out.json] [--thresholds thresholds.json]

What it does: walks every *.md file under the vault, reads it, and counts things.
What it does NOT do: it never writes into the vault, never calls a model, never
touches the network. The only file it can write is the --json path you give it,
and it refuses to write that inside the vault.

Other code can import it:
    from assess import assess, verdict, load_thresholds
    stats = assess(Path("~/vault").expanduser())
    v = verdict(stats, load_thresholds())

Standard library only. Python 3.11+.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

KIT_DIR = Path(__file__).resolve().parent
DEFAULT_THRESHOLDS = KIT_DIR / "thresholds.json"

# Tokens are estimated as characters / 4. That is a rough English-text rule of
# thumb, not a real tokenizer count. Code, non-English text and heavy markup
# will be off, sometimes by 30% or more. Treat the number as an order of magnitude.
CHARS_PER_TOKEN = 4

WIKILINK_RE = re.compile(r"\[\[[^\]\n]+\]\]")
# Markdown links, excluding images (![alt](src)).
MDLINK_RE = re.compile(r"(?<!!)\[[^\]\n]*\]\([^)\n]+\)")
# Inline #tag: starts with a letter so "#1" and headings ("# Title") do not count.
# The lookbehind stops matches inside words, URLs (page#anchor) and nested tags.
INLINE_TAG_RE = re.compile(r"(?<![\w/&#\[(])#([A-Za-z][\w/-]*)")
INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
FENCE_RE = re.compile(r"^\s{0,3}(```|~~~)")


# --------------------------------------------------------------------------
# Walking the vault
# --------------------------------------------------------------------------
def iter_notes(vault: Path):
    """Yield the Path of each *.md file in the vault.

    Rules: skip any directory whose name starts with "." (.obsidian, .git,
    .trash ...). Do not descend into symlinked directories. Skip symlinked
    files that point outside the vault, so a stray link cannot pull in
    unrelated files.
    """
    vault = vault.resolve()
    for root, dirs, files in os.walk(vault, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(files):
            if not name.lower().endswith(".md") or name.startswith("."):
                continue
            p = Path(root) / name
            if p.is_symlink():
                try:
                    target = p.resolve()
                    target.relative_to(vault)
                except (OSError, ValueError, RuntimeError):
                    continue  # points outside the vault (or is broken)
            yield p


# --------------------------------------------------------------------------
# Parsing one note
# --------------------------------------------------------------------------
def split_frontmatter(text: str) -> tuple[str | None, str]:
    """Return (frontmatter_text_or_None, body). Frontmatter = leading --- block."""
    if not text.startswith("---"):
        return None, text
    lines = text.split("\n")
    if lines[0].strip() != "---":
        return None, text
    for i in range(1, len(lines)):
        if lines[i].strip() in ("---", "..."):
            return "\n".join(lines[1:i]), "\n".join(lines[i + 1:])
    return None, text  # opening --- never closed: not real frontmatter


def frontmatter_tags(fm: str) -> int:
    """Count tags declared in frontmatter. Handles the three common shapes:
    tags: [a, b]   |   tags: a, b   |   tags:\n  - a\n  - b
    """
    count = 0
    lines = fm.split("\n")
    for i, line in enumerate(lines):
        m = re.match(r"^(tags?)\s*:\s*(.*)$", line, re.IGNORECASE)
        if not m:
            continue
        rest = m.group(2).strip()
        if rest:
            rest = rest.strip("[]")
            count += len([t for t in re.split(r"[,\s]+", rest) if t.strip("'\"")])
        else:
            for nxt in lines[i + 1:]:
                lm = re.match(r"^\s*-\s*(\S.*)$", nxt)
                if lm:
                    count += 1
                elif nxt.strip() == "":
                    continue
                else:
                    break
        break  # only the first tags key counts
    return count


def analyze_note(text: str) -> dict:
    """Counts for one note's text. Code fences are ignored for links and tags."""
    fm, body = split_frontmatter(text)
    wikilinks = mdlinks = inline_tags = 0
    in_fence = False
    for line in body.split("\n"):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        clean = INLINE_CODE_RE.sub("", line)
        wikilinks += len(WIKILINK_RE.findall(clean))
        mdlinks += len(MDLINK_RE.findall(clean))
        inline_tags += len(INLINE_TAG_RE.findall(clean))
    return {
        "words": len(text.split()),
        "tokens": len(text) // CHARS_PER_TOKEN,
        "has_frontmatter": fm is not None,
        "wikilinks": wikilinks,
        "mdlinks": mdlinks,
        "tags": inline_tags + (frontmatter_tags(fm) if fm else 0),
    }


# --------------------------------------------------------------------------
# Whole-vault measurement
# --------------------------------------------------------------------------
def last_12_months(now: datetime) -> list[str]:
    """['YYYY-MM', ...] oldest first, ending with the current month."""
    y, m = now.year, now.month
    out = []
    for _ in range(12):
        out.append(f"{y:04d}-{m:02d}")
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    return out[::-1]


def assess(vault: Path, now: datetime | None = None) -> dict:
    """Measure the vault. Read-only. Returns a plain dict (JSON-serializable)."""
    vault = Path(vault).expanduser().resolve()
    if not vault.is_dir():
        raise NotADirectoryError(f"not a folder: {vault}")
    now = now or datetime.now()
    months = last_12_months(now)
    month_counts = {m: 0 for m in months}

    notes = words = tokens = wikilinks = mdlinks = tags = with_fm = unreadable = 0
    folder_tokens: dict[str, int] = {}
    folder_notes: dict[str, int] = {}
    biggest: list[dict] = []

    for path in iter_notes(vault):
        try:
            st = path.stat()
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            unreadable += 1  # counted, skipped, never fatal
            continue
        a = analyze_note(text)
        rel = path.relative_to(vault).as_posix()
        notes += 1
        words += a["words"]
        tokens += a["tokens"]
        wikilinks += a["wikilinks"]
        mdlinks += a["mdlinks"]
        tags += a["tags"]
        with_fm += a["has_frontmatter"]
        folder = path.parent.relative_to(vault).as_posix()
        folder = "(vault root)" if folder == "." else folder
        folder_tokens[folder] = folder_tokens.get(folder, 0) + a["tokens"]
        folder_notes[folder] = folder_notes.get(folder, 0) + 1
        biggest.append({"path": rel, "tokens": a["tokens"], "words": a["words"]})
        key = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m")
        if key in month_counts:
            month_counts[key] += 1

    biggest.sort(key=lambda d: (-d["tokens"], d["path"]))
    top_folders = sorted(folder_tokens.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
    per_note = (lambda n: n / notes) if notes else (lambda n: 0.0)

    return {
        "vault": str(vault),
        "notes": notes,
        "words": words,
        "est_tokens": tokens,
        "tokens_note": "Estimated as characters / 4. A rough guess, not a tokenizer count.",
        "months": [{"month": m, "notes": month_counts[m]} for m in months],
        "months_note": (
            "Based on file modification time (mtime). That is only a proxy for when you "
            "wrote or edited a note: copying, syncing or restoring files resets mtime."
        ),
        "notes_touched_per_month_avg": round(sum(month_counts.values()) / 12, 1),
        "wikilinks": wikilinks,
        "md_links": mdlinks,
        "links_per_note": round(per_note(wikilinks + mdlinks), 2),
        "tags_total": tags,
        "tags_per_note": round(per_note(tags), 2),
        "frontmatter_notes": with_fm,
        "frontmatter_pct": round(100 * per_note(with_fm), 1),
        "top_folders": [
            {"folder": f, "est_tokens": t, "notes": folder_notes[f]} for f, t in top_folders
        ],
        "largest_notes": biggest[:5],
        "unreadable_files": unreadable,
    }


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------
def load_thresholds(path: Path | None = None) -> dict:
    data = json.loads(Path(path or DEFAULT_THRESHOLDS).read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _fmt(n: float) -> str:
    return f"{int(n):,}"


def _cite(thresholds: dict, name: str, what: str) -> str:
    """One sentence fragment naming a threshold, its value and where it comes from."""
    t = thresholds[name]
    tag = "cited" if t["status"] == "cited" else "from the author's benchmark"
    return f"{what} ({name} = {_fmt(t['value'])} tokens; {tag})"


# The author's benchmark for the video. Everything about it that the verdict says comes from this one
# block, so updating the numbers means editing here and the one README paragraph. Results are not public yet.
BENCHMARK = {
    "who": "Mason's benchmark for the video",
    "setup": "Claude Sonnet, one fictional company's Obsidian-style vault, 36 questions x 3 runs per arm, 100 to 50,000 notes",
    "summary": (
        "An agent browsing the vault was the most accurate at every size (92%, 84%, 74%, 75% and 64% from 100 to "
        "50,000 notes). Plain injection of retrieved passages scored 77%, 72%, 69%, 63% and 59%, and injection with "
        "the verify header 77%, 70%, 72% and 59% from 1,000 notes up. A search tool the agent calls itself was no "
        "more accurate than browsing and was the slowest. Injection used 41 to 64% fewer tokens than browsing "
        "(27 to 50% with the verify header), was faster up to 10,000 notes, and no accuracy crossover appeared "
        "up to about 38M tokens."),
}
BENCHMARK_NOTE = f"{BENCHMARK['who']} ({BENCHMARK['setup']}): {BENCHMARK['summary']}"


def verdict(stats: dict, thresholds: dict) -> dict:
    """Return {outcome, headline, reasons, caveats}. The outcome depends on size in tokens only.

      skip         under ~200,000 tokens: the agent can read it all.
      optional     ~200,000 to ~5M tokens: retrieval saves roughly half the tokens per notes question, at a
                   small accuracy cost.
      recommended  above ~5M tokens: worth it if cost, speed or usage limits matter, since browsing used
                   33,000 to 51,000 tokens per question in the benchmark. Browsing was still a few points more accurate.
    No accuracy promise is made. Thresholds come from thresholds.json.
    """
    tokens = stats["est_tokens"]
    skip_tok = thresholds["skip_retrieval_tokens"]["value"]
    rec_tok = thresholds["recommended_tokens"]["value"]
    reasons: list[str] = []
    caveats: list[str] = [
        "Token counts are estimates (characters / 4), and file dates are a proxy.",
        "This verdict looks at size only. It does not measure how well your notes are written or how hard "
        "your questions are. The optional mini test does.",
    ]
    if tokens < skip_tok:
        outcome = "skip"
        headline = "You don't need this. Your agent can read your whole vault."
        reasons.append(
            f"Your vault is about {_fmt(tokens)} estimated tokens, "
            + _cite(thresholds, "skip_retrieval_tokens", "under the size where a knowledge base can be put in the prompt whole")
            + ".")
        reasons.append("At this size browsing with the agent's own search tools worked best in the benchmark below, "
                       "so retrieval would add machinery without adding accuracy.")
        caveats.append("Fitting in the context window does not make it free: a full-vault prompt costs tokens on every "
                       "call. If that bothers you, that is a cost question, not a quality one.")
    elif tokens < rec_tok:
        outcome = "optional"
        headline = "Optional. Retrieval would save roughly half the tokens per notes question, at a small accuracy cost."
        reasons.append(
            f"About {_fmt(tokens)} estimated tokens is over "
            + _cite(thresholds, "skip_retrieval_tokens", "the size where the whole vault fits in a prompt")
            + " and under " + _cite(thresholds, "recommended_tokens", "the size where the savings get large") + ".")
        reasons.append("In the benchmark below, injecting retrieved passages used 41 to 64% fewer tokens than browsing "
                       "at every size, and browsing was a few points more accurate (3 to 15 points, depending on size "
                       "and mode). If accuracy matters more to you than cost, keep browsing.")
    else:
        outcome = "recommended"
        headline = ("Worth it if cost, speed or usage limits matter. If accuracy matters most, "
                    "browsing was a few points better.")
        reasons.append(
            f"About {_fmt(tokens)} estimated tokens is at or past "
            + _cite(thresholds, "recommended_tokens", "the size where injection saved 17,000 to 27,000 tokens per notes question")
            + ", so injected retrieval saves a lot per question (roughly half in the benchmark below).")
        reasons.append("Browsing still scored higher in accuracy at every size the benchmark tested, up to about 38M "
                       "tokens, by 5 to 12 points at the largest sizes. If accuracy is the priority, consider `both` "
                       "mode (injection plus a search tool) or keeping browsing.")
        if tokens >= thresholds["bm25_crossover_tokens"]["value"]:
            reasons.append("At this scale keyword search matters most. One paper found plain keyword (BM25) retrieval "
                           "overtaking a file-browsing agent near 10M tokens, with embeddings alone doing no better, "
                           "which is why the kit always runs keyword search alongside embeddings.")
    caveats.append(BENCHMARK_NOTE)
    if outcome != "skip":
        caveats.append("No accuracy gain is promised. Size says retrieval may save cost; only your own questions "
                       "can say what it does to your answers.")
    return {"outcome": outcome, "headline": headline, "reasons": reasons, "caveats": caveats}


# --------------------------------------------------------------------------
# Terminal report
# --------------------------------------------------------------------------
def print_summary(stats: dict, v: dict) -> None:
    p = print
    p(f"Vault: {stats['vault']}")
    p(f"Notes: {stats['notes']:,}   Words: {stats['words']:,}   "
      f"Estimated tokens: {stats['est_tokens']:,} (characters / 4, an estimate)")
    if stats["unreadable_files"]:
        p(f"Unreadable files skipped: {stats['unreadable_files']}")
    p(f"Links per note: {stats['links_per_note']} "
      f"({stats['wikilinks']:,} wikilinks, {stats['md_links']:,} markdown links)")
    p(f"Tags per note: {stats['tags_per_note']}   Notes with frontmatter: {stats['frontmatter_pct']}%")
    p("")
    p("Notes created or modified per month (by file mtime; a proxy, copies reset it):")
    peak = max([m["notes"] for m in stats["months"]] + [1])
    for m in stats["months"]:
        p(f"  {m['month']}  {m['notes']:>5}  {'#' * round(30 * m['notes'] / peak)}")
    p("")
    p("Biggest folders by estimated tokens:")
    for f in stats["top_folders"]:
        p(f"  {f['est_tokens']:>10,}  {f['notes']:>5} notes  {f['folder']}")
    p("")
    p("Largest notes:")
    for n in stats["largest_notes"]:
        p(f"  {n['tokens']:>8,} tokens  {n['path']}")
    p("")
    p(f"VERDICT: {v['outcome']} - {v['headline']}")
    for r in v["reasons"]:
        p(f"  * {r}")
    p("Caveats:")
    for c in v["caveats"]:
        p(f"  - {c}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Read-only measurement of a markdown vault.")
    ap.add_argument("vault", type=Path)
    ap.add_argument("--json", type=Path, dest="json_out", help="also write stats + verdict here")
    ap.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    args = ap.parse_args(argv)

    vault = args.vault.expanduser().resolve()
    if args.json_out:
        out = args.json_out.expanduser().resolve()
        if out == vault or vault in out.parents:
            ap.error("--json must point outside the vault (this tool never writes into it)")
    try:
        stats = assess(vault)
    except NotADirectoryError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if stats["notes"] == 0:
        print(f"No markdown notes found in {vault}. Check the path: it should be the folder that holds your .md files.",
              file=sys.stderr)
        return 2
    v = verdict(stats, load_thresholds(args.thresholds))
    print_summary(stats, v)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({"stats": stats, "verdict": v}, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
