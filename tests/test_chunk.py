from pathlib import Path

import pytest

from sbrag.chunk import chunk_note, embed_input, estimate_tokens, split_sections

FIXTURE = Path(__file__).parent / "fixtures" / "vault" / "projects" / "rag-project.md"


def body_of(path: Path) -> str:
    from sbrag.parse import split_frontmatter

    return split_frontmatter(path.read_text())[1]


def test_heading_paths_are_breadcrumbs():
    paths = [s.heading_path for s in split_sections(body_of(FIXTURE), "RAG Project")]
    assert paths == [
        "RAG Project",
        "RAG Project > Goals",
        "RAG Project > Stack > Storage",
        "RAG Project > Stack > Models",
        "RAG Project > Open Questions",
    ]


def test_hash_inside_code_fence_is_not_a_heading():
    sections = split_sections(body_of(FIXTURE), "RAG Project")
    models = next(s for s in sections if s.heading_path.endswith("Models"))
    assert "# not a heading" in models.text


def test_empty_sections_dropped_and_ordinals_sequential():
    chunks = chunk_note(body_of(FIXTURE), "RAG Project")
    assert not any(c.heading_path.endswith("Empty Section") for c in chunks)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_wikilinks_rendered_in_chunk_content():
    chunks = chunk_note(body_of(FIXTURE), "RAG Project")
    assert "built on Google Cloud" in chunks[0].content
    assert "[[" not in chunks[0].content


def test_text_before_first_heading_filed_under_title():
    chunks = chunk_note("intro text\n\n## A\n\nsection a", "My Note")
    assert chunks[0].heading_path == "My Note"
    assert chunks[1].heading_path == "A"


def long_section(n_paragraphs: int = 40) -> str:
    paras = [f"Paragraph {i}. " + " ".join(f"word{i}_{j}" for j in range(30))
             for i in range(n_paragraphs)]
    return "## Long\n\n" + "\n\n".join(paras) + "\n\n## Next\n\nshort tail section"


@pytest.mark.parametrize("target,overlap", [(500, 50), (200, 20), (120, 0)])
def test_long_section_respects_target(target, overlap):
    chunks = chunk_note(long_section(), "T", target=target, overlap=overlap)
    long_chunks = [c for c in chunks if c.heading_path == "Long"]
    assert len(long_chunks) > 1
    assert all(c.token_estimate <= target for c in chunks)


def test_overlap_within_section_but_not_across_headings():
    chunks = chunk_note(long_section(), "T", target=200, overlap=30)
    long_chunks = [c for c in chunks if c.heading_path == "Long"]
    for prev, nxt in zip(long_chunks, long_chunks[1:]):
        tail_word = prev.content.split()[-1]
        assert tail_word in nxt.content.split()[:20], "expected carried-over overlap"
    nxt_section = next(c for c in chunks if c.heading_path == "Next")
    assert nxt_section.content == "short tail section"


def test_no_text_lost_without_overlap():
    text = long_section()
    chunks = chunk_note(text, "T", target=150, overlap=0)
    joined = " ".join(c.content for c in chunks).split()
    for word in ("word0_0", "word17_29", "word39_29", "tail"):
        assert word in joined


def test_giant_table_split_by_rows():
    rows = "\n".join(f"| row {i} | " + "x" * 60 + " |" for i in range(100))
    chunks = chunk_note("## Table\n\n| a | b |\n|---|---|\n" + rows, "T", target=300, overlap=0)
    assert len(chunks) > 1
    assert all(c.token_estimate <= 300 for c in chunks)
    # rows are never cut mid-line
    for c in chunks:
        for line in c.content.splitlines():
            assert line.startswith("|") and line.endswith("|")


def test_unbreakable_run_is_hard_cut():
    chunks = chunk_note("## U\n\n" + "a" * 5000, "T", target=200, overlap=0)
    assert all(c.token_estimate <= 200 for c in chunks)
    assert sum(len(c.content) for c in chunks) == 5000


def test_overlap_must_be_smaller_than_target():
    with pytest.raises(ValueError):
        chunk_note("x", "T", target=50, overlap=50)


def test_embed_input_format():
    chunks = chunk_note("# Fitness Plan\n\n## Week 3\n\nsquat 5x5", "Fitness Plan")
    assert embed_input("Fitness Plan", chunks[0]) == (
        "title: Fitness Plan > Week 3 | text: squat 5x5"
    )
    chunks = chunk_note("## Week 3\n\nsquat", "Fitness Plan")
    assert embed_input("Fitness Plan", chunks[0]).startswith("title: Fitness Plan > Week 3 |")


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2
