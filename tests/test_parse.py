import datetime as dt

from sbrag.parse import (
    extract_links,
    normalize_tags,
    parse_note,
    render_wikilinks,
    split_frontmatter,
)


def test_frontmatter_split_and_typed_tags():
    note = parse_note("---\ntags: [Career, projects]\ncreated: 2026-08-15\n---\n# T\nbody",
                      "career/x.md")
    assert note.tags == ["career", "projects"]
    assert note.created_at == dt.datetime(2026, 8, 15, tzinfo=dt.UTC)
    assert note.date_source == "frontmatter"
    assert note.folder == "career"
    assert note.body.startswith("# T")
    # YAML dates must be JSON-serializable for the jsonb column
    assert note.frontmatter["created"] == "2026-08-15"


def test_tags_in_every_shape():
    assert normalize_tags("#journal, reflection") == ["journal", "reflection"]
    assert normalize_tags("a b a") == ["a", "b"]
    assert normalize_tags(["#X", "y"]) == ["x", "y"]
    assert normalize_tags(None) == []


def test_no_frontmatter():
    fm, body = split_frontmatter("# Just a body\n")
    assert fm == {}
    assert body == "# Just a body\n"


def test_malformed_yaml_does_not_crash():
    note = parse_note("---\ntags: [unclosed\n---\n\n# Broken\n\nBody.", "b.md")
    assert note.tags == []
    assert "Body." in note.body


def test_title_priority():
    assert parse_note("---\ntitle: FM Title\n---\n# H1", "a.md").title == "FM Title"
    assert parse_note("# Heading One\ntext", "a.md").title == "Heading One"
    assert parse_note("no heading", "dir/my-note_file.md").title == "my note file"


def test_root_note_has_empty_folder():
    assert parse_note("x", "index.md").folder == ""


def test_wikilinks():
    body = "See [[ops-101|Ops 101]], [[goals]], [[goals]] and ![[img.png]] and [[a#Section|alias]]."
    assert extract_links(body) == ["ops-101", "goals", "img.png", "a"]
    assert render_wikilinks(body) == "See Ops 101, goals, goals and img.png and alias."


def test_content_hash_changes_with_content():
    assert parse_note("a", "x.md").content_hash != parse_note("b", "x.md").content_hash
