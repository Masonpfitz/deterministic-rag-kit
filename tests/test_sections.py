from rag import sections


def test_anchors_are_added_once_and_frozen():
    body = "intro\n\n## Setup\n\ntext\n\n## Setup\n\nmore"
    fixed, n = sections.with_anchors(body)
    assert n == 2
    assert "## Setup {#setup}" in fixed and "## Setup {#setup-2}" in fixed
    again, n2 = sections.with_anchors(fixed)
    assert n2 == 0 and again == fixed


def test_renamed_heading_keeps_anchor_in_section_id():
    body = "## New name {#old-name}\n\ntext here"
    (s,) = sections.split_sections("doc", body)
    assert s.section_id == "doc#old-name" and s.heading == "New name"


def test_top_section_and_heading_path():
    body = "lead paragraph\n\n# A\n\nx\n\n## B\n\ny"
    ids = [s.section_id for s in sections.split_sections("d", body)]
    assert ids == ["d#top", "d#a", "d#b"]
    assert sections.split_sections("d", body)[2].heading_path == "A > B"


def test_headings_inside_code_fences_are_ignored():
    body = "```\n# not a heading\n```\n\n## Real\n\ntext"
    assert [h.text for h in sections.headings(body)] == ["Real"]


def test_oversize_section_splits_on_paragraphs_with_slice_ids():
    para = ("word " * 120).strip()  # ~150 tokens
    body = "## Big\n\n" + "\n\n".join([para] * 8)
    secs = sections.split_sections("d", body, max_tokens=300)
    assert len(secs) > 1
    assert all(s.section_id.startswith("d#big~") for s in secs)
    assert all(s.tokens <= 300 + 5 for s in secs)
    assert sections.parent_id(secs[0].section_id) == "d#big"


def test_unbroken_wall_of_text_still_splits():
    secs = sections.split_sections("d", "x" * 10000, max_tokens=200)
    assert len(secs) > 1 and all(s.tokens <= 200 for s in secs)


def test_heading_only_section_is_dropped():
    assert sections.split_sections("d", "## Empty {#empty}\n") == []


def test_links_skip_urls_and_code():
    body = "[a](x.md) [b](https://e.com) [c](#frag)\n```\n[d](y.md)\n```"
    assert [t for t, _ in sections.links(body)] == ["x.md"]
