"""Sections: the unit of retrieval.

A document splits at its headings. Each heading carries a frozen anchor
(`## Setup {#setup}`), the section id is `<doc id>#<anchor>`, and text before the first
heading is `<doc id>#top`. A section over the size cap splits on paragraphs into
`<section id>~1`, `~2`. Renaming a heading keeps its anchor, so links to a section
keep working.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

CHARS_PER_TOKEN = 4
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
ANCHOR = re.compile(r"\s*\{#([a-z0-9][a-z0-9-]*)\}$")
FENCE = re.compile(r"^\s*(```|~~~)")
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)")
SKIP_TARGET = re.compile(r"^(https?:|mailto:|tel:|#)")


@dataclass
class Heading:
    line: int  # 1-based
    level: int
    text: str
    anchor: str | None


@dataclass
class Section:
    section_id: str
    heading: str  # innermost heading text ("" for the top section)
    heading_path: str  # "A > B > C"
    text: str
    tokens: int
    start_line: int
    end_line: int
    ord: int


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def prose_lines(body: str):
    """(line number, line) for every line outside a code fence."""
    fenced = False
    for n, line in enumerate(body.split("\n"), start=1):
        if FENCE.match(line):
            fenced = not fenced
            continue
        if not fenced:
            yield n, line


def headings(body: str) -> list[Heading]:
    out = []
    for n, line in prose_lines(body):
        m = HEADING.match(line)
        if not m:
            continue
        text, anchor = m.group(2), None
        a = ANCHOR.search(text)
        if a:
            text, anchor = text[: a.start()].rstrip(), a.group(1)
        out.append(Heading(n, len(m.group(1)), text, anchor))
    return out


def slug(text: str, limit: int = 60) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:limit].rstrip("-")


def resolve_anchors(heads: list[Heading]) -> list[str]:
    """The frozen anchor where there is one, else a slug, suffixed -2, -3 on a clash."""
    taken = {h.anchor for h in heads if h.anchor}
    out: list[str] = []
    for h in heads:
        if h.anchor:
            out.append(h.anchor)
            continue
        base = slug(h.text) or "section"
        cand, n = base, 1
        while cand in taken or cand in out:
            n += 1
            cand = f"{base}-{n}"
        out.append(cand)
    return out


def with_anchors(body: str) -> tuple[str, int]:
    """The body with every heading anchored, and how many anchors were added."""
    heads = headings(body)
    anchors = resolve_anchors(heads)
    lines = body.split("\n")
    added = 0
    for h, a in zip(heads, anchors):
        if h.anchor is None:
            lines[h.line - 1] = f"{lines[h.line - 1].rstrip()} {{#{a}}}"
            added += 1
    return "\n".join(lines), added


def _split_oversize(text: str, max_tokens: int) -> list[tuple[str, int]]:
    """Pack paragraphs (then sentences, then raw slices) into pieces of at most max_tokens.
    Returns (piece, line offset of the piece inside text)."""
    budget = max_tokens * CHARS_PER_TOKEN
    units: list[tuple[str, int]] = []
    line = 0
    for para in re.split(r"(\n\s*\n)", text):
        if para.strip():
            if len(para) <= budget:
                units.append((para, line))
            else:
                off = line
                for sent in re.split(r"(?<=[.!?])\s+(?=\S)", para):
                    while len(sent) > budget:  # an unbroken wall of text
                        units.append((sent[:budget], off))
                        sent = sent[budget:]
                    if sent.strip():
                        units.append((sent, off))
                    off += sent.count("\n")
        line += para.count("\n")
    pieces: list[tuple[str, int]] = []
    buf: list[str] = []
    start = 0
    size = 0
    for unit, off in units:
        if buf and size + len(unit) > budget:
            pieces.append(("\n\n".join(buf), start))
            buf, size = [], 0
        if not buf:
            start = off
        buf.append(unit.strip("\n"))
        size += len(unit)
    if buf:
        pieces.append(("\n\n".join(buf), start))
    return pieces


def split_sections(doc_id: str, body: str, max_tokens: int = 600) -> list[Section]:
    lines = body.split("\n")
    heads = headings(body)
    anchors = resolve_anchors(heads)
    spans: list[tuple[int, int, str, str, str]] = []  # start, end, anchor, heading, path
    first = heads[0].line if heads else len(lines) + 1
    if any(l.strip() for l in lines[: first - 1]):
        spans.append((1, first - 1, "top", "", ""))
    stack: list[tuple[int, str]] = []
    for i, (h, a) in enumerate(zip(heads, anchors)):
        while stack and stack[-1][0] >= h.level:
            stack.pop()
        stack.append((h.level, h.text))
        end = heads[i + 1].line - 1 if i + 1 < len(heads) else len(lines)
        spans.append((h.line, end, a, h.text, " > ".join(t for _, t in stack)))
    out: list[Section] = []
    for start, end, anchor, heading, path in spans:
        text = "\n".join(lines[start - 1 : end]).strip()
        if not text:
            continue
        # A section that is only its own heading line carries no information.
        if start <= len(lines) and HEADING.match(lines[start - 1]) and text == lines[start - 1].strip():
            continue
        sid = f"{doc_id}#{anchor}"
        if estimate_tokens(text) <= max_tokens:
            out.append(Section(sid, heading, path, text, estimate_tokens(text), start, end, len(out)))
            continue
        for n, (piece, off) in enumerate(_split_oversize(text, max_tokens), start=1):
            out.append(Section(f"{sid}~{n}", heading, path, piece, estimate_tokens(piece),
                               start + off, start + off + piece.count("\n"), len(out)))
    return out


def parent_id(section_id: str) -> str:
    """The section a slice (`~2`) belongs to."""
    return re.sub(r"~\d+$", "", section_id)


def links(body: str) -> list[tuple[str, int]]:
    """(target, line) for every local markdown link outside code fences."""
    out = []
    for n, line in prose_lines(body):
        for m in LINK.finditer(line):
            if not SKIP_TARGET.match(m.group(1)):
                out.append((m.group(1), n))
    return out
