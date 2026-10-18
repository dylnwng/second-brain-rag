"""Pipeline tests, run offline against MemoryStore + HashEmbedder."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from sbrag.config import Settings
from sbrag.embed import HashEmbedder
from sbrag.ingest import discover, dry_run, git_dates, load_notes, run
from sbrag.store import MemoryStore, note_id

FIXTURE = Path(__file__).parent / "fixtures" / "vault"
DIM = 64


def settings(vault: Path, **kw) -> Settings:
    base = dict(
        vault_path=vault, firestore_database="(default)", collection_prefix="",
        gcp_project=None,
        gcp_location="global", genai_backend="gemini_api", gemini_api_key=None,
        embed_model="hash-test", embed_dim=DIM,
        embed_price_per_1m_tokens=0.0, chunk_target_tokens=500, chunk_overlap_tokens=50,
        exclude_globs=["log.md"],
    )
    base.update(kw)
    return Settings(**base)


@pytest.fixture
def vault(tmp_path):
    v = tmp_path / "vault"
    shutil.copytree(FIXTURE, v)
    return v


def test_discover_skips_hidden_dirs_and_excluded_globs():
    rels = [p.relative_to(FIXTURE).as_posix() for p in discover(FIXTURE, ["log.md"])]
    assert rels == ["index.md", "journal/2026-09-01.md", "journal/broken-frontmatter.md",
                    "projects/rag-project.md"]


def test_dates_fall_back_frontmatter_then_git_then_mtime(vault):
    git = ["git", "-C", str(vault), "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(git[:3] + ["init", "-q"], check=True)
    subprocess.run(git + ["add", "index.md"], check=True)
    subprocess.run(git + ["commit", "-qm", "init", "--date=2026-01-02T03:04:05+00:00"],
                   check=True, env={**os.environ, "GIT_COMMITTER_DATE": "2026-01-02T03:04:05Z"})
    assert "index.md" in git_dates(vault)

    notes = {n.path: n for n in load_notes(vault, ["log.md"])}
    assert notes["projects/rag-project.md"].date_source == "frontmatter"
    assert notes["index.md"].date_source == "git"
    assert notes["index.md"].created_at.year == 2026 and notes["index.md"].created_at.month == 1
    assert notes["journal/2026-09-01.md"].date_source == "mtime"


def test_dry_run_counts():
    out = dry_run(settings(FIXTURE))
    assert out["notes"] == 4
    assert out["chunks"] >= 6
    assert out["notes_with_tags"] == 3


def test_second_run_embeds_nothing(vault):
    store, emb = MemoryStore(), HashEmbedder(DIM)
    first = run(settings(vault), store, emb)
    assert first.notes_changed == 4 and first.chunks_embedded > 0
    second = run(settings(vault), store, emb)
    assert second.notes_changed == 0
    assert second.chunks_embedded == 0
    assert second.chunks_total == first.chunks_total
    assert len(store.runs) == 2


def test_edit_reembeds_only_changed_chunks(vault):
    store, emb = MemoryStore(), HashEmbedder(DIM)
    first = run(settings(vault), store, emb)
    note = vault / "projects" / "rag-project.md"
    note.write_text(note.read_text().replace("About 500 tokens.", "About 400 tokens."))
    second = run(settings(vault), store, emb)
    assert second.notes_changed == 1
    assert second.chunks_embedded == 1  # only the "Open Questions" chunk changed
    assert second.chunks_total == first.chunks_total
    oq = [c for c in store.chunks.values() if c["heading_path"].endswith("Open Questions")]
    assert "400 tokens" in oq[0]["content"]


def test_deleted_note_removed_with_its_chunks(vault):
    store, emb = MemoryStore(), HashEmbedder(DIM)
    run(settings(vault), store, emb)
    (vault / "journal" / "2026-09-01.md").unlink()
    stats = run(settings(vault), store, emb)
    assert stats.notes_deleted == 1
    assert not any(c["note_path"] == "journal/2026-09-01.md" for c in store.chunks.values())
    assert note_id("journal/2026-09-01.md") not in store.notes


def test_chunker_setting_change_rechunks_but_reuses_identical_chunks(vault):
    store, emb = MemoryStore(), HashEmbedder(DIM)
    run(settings(vault), store, emb)
    # A different overlap changes the fingerprint, so every note is re-chunked, but all
    # fixture sections are shorter than the target, so chunk text is identical: reuse.
    stats = run(settings(vault, chunk_overlap_tokens=10), store, emb)
    assert stats.notes_changed == 4
    assert stats.chunks_embedded == 0
    assert stats.chunks_reused == stats.chunks_total


def test_chunk_and_note_metadata(vault):
    store = MemoryStore()
    run(settings(vault), store, HashEmbedder(DIM))
    note = store.notes[note_id("projects/rag-project.md")]
    assert note["tags"] == ["career", "projects"] and note["folder"] == "projects"
    assert note["links"] == ["google-cloud"]
    storage = next(c for c in store.chunks.values() if c["heading_path"].endswith("Storage"))
    assert storage["note_path"] == "projects/rag-project.md"
    assert storage["tags"] == ["career", "projects"]
    assert len(storage["embedding"]) == DIM


def test_search_returns_exact_match_first(vault):
    store, emb = MemoryStore(), HashEmbedder(DIM)
    run(settings(vault), store, emb)
    target = next(iter(store.chunks.values()))
    hits = store.search(target["embedding"], 5)
    assert len(hits) == 5
    assert hits[0].content == target["content"] and hits[0].distance < 1e-9
