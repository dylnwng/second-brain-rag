"""Ingestion pipeline: vault -> notes + chunks -> embeddings -> Firestore vector store.

Idempotent: a note is re-chunked only when its file hash or the pipeline fingerprint
(embedding model, dimension, chunker settings) changes, and a chunk is re-embedded only
when the note's stored chunks don't already have the same embed hash.

    python -m sbrag.ingest              # full run
    python -m sbrag.ingest --dry-run    # parse + chunk + estimate, no store, no API calls
"""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import hashlib
import json
import logging
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from sbrag.chunk import Chunk, chunk_note, embed_input, estimate_tokens
from sbrag.config import Settings
from sbrag.embed import Embedder
from sbrag.store import Store
from sbrag.parse import Note, parse_note

log = logging.getLogger("sbrag.ingest")

CHUNKER_VERSION = "1"  # bump when chunk.py output changes for the same input
SKIP_DIRS = {".obsidian", ".git", ".trash", ".claude", "node_modules"}


@dataclass
class RunStats:
    notes_seen: int = 0
    notes_changed: int = 0
    notes_unchanged: int = 0
    notes_deleted: int = 0
    chunks_total: int = 0
    chunks_embedded: int = 0
    chunks_reused: int = 0
    tokens_embedded: int = 0
    token_source: str = "none"  # none | api | estimate
    est_cost_usd: float = 0.0
    seconds: float = 0.0


# ---------------------------------------------------------------- discovery

def discover(vault: Path, exclude: list[str]) -> list[Path]:
    out = []
    for p in sorted(vault.rglob("*.md")):
        rel = p.relative_to(vault)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if any(fnmatch.fnmatch(rel.as_posix(), g) for g in exclude):
            continue
        out.append(p)
    return out


def git_dates(vault: Path) -> dict[str, tuple[dt.datetime, dt.datetime]]:
    """(first commit, last commit) per file from the vault's git history, read-only.
    Returns {} if the vault isn't a git repo."""
    try:
        out = subprocess.run(
            ["git", "-C", str(vault), "log", "--format=\x1e%aI", "--name-only",
             "--relative", "--", "."],
            capture_output=True, text=True, check=True, timeout=60,
        ).stdout
    except (subprocess.SubprocessError, FileNotFoundError):
        return {}
    dates: dict[str, tuple[dt.datetime, dt.datetime]] = {}
    for record in out.split("\x1e"):
        lines = [ln for ln in record.strip().splitlines() if ln.strip()]
        if not lines:
            continue
        when = dt.datetime.fromisoformat(lines[0])
        for path in lines[1:]:
            # log is newest-first: first sighting = last update, last sighting = creation
            last = dates[path][1] if path in dates else when
            dates[path] = (when, last)
    return dates


def load_notes(vault: Path, exclude: list[str]) -> list[Note]:
    gdates = git_dates(vault)
    notes = []
    for p in discover(vault, exclude):
        rel = p.relative_to(vault).as_posix()
        note = parse_note(p.read_text(encoding="utf-8", errors="replace"), rel)
        mtime = dt.datetime.fromtimestamp(p.stat().st_mtime, tz=dt.UTC)
        if note.created_at is None:
            if rel in gdates:
                note.created_at, note.date_source = gdates[rel][0], "git"
            else:
                note.created_at, note.date_source = mtime, "mtime"
        if note.updated_at is None:
            note.updated_at = max(gdates[rel][1], mtime) if rel in gdates else mtime
        notes.append(note)
    return notes


# ---------------------------------------------------------------- pipeline

def fingerprint(s: Settings, model: str) -> str:
    key = f"{model}|{s.embed_dim}|{s.chunk_target_tokens}|{s.chunk_overlap_tokens}|{CHUNKER_VERSION}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def embed_hash(model: str, dim: int, text: str) -> str:
    return hashlib.sha256(f"{model}|{dim}|{text}".encode()).hexdigest()


def chunk_all(notes: list[Note], s: Settings) -> dict[str, list[Chunk]]:
    return {n.path: chunk_note(n.body, n.title, s.chunk_target_tokens, s.chunk_overlap_tokens)
            for n in notes}


def run(s: Settings, store: Store, embedder: Embedder) -> RunStats:
    t0 = time.perf_counter()
    started = dt.datetime.now(dt.UTC)
    stats = RunStats()
    fp = fingerprint(s, embedder.model)

    notes = load_notes(s.vault_path, s.exclude_globs)
    stats.notes_seen = len(notes)
    stored = store.note_states()

    current = {n.path for n in notes}
    gone = sorted(set(stored) - current)
    changed = [n for n in notes if stored.get(n.path) != (n.content_hash, fp)]
    unchanged = [n for n in notes if stored.get(n.path) == (n.content_hash, fp)]
    stats.notes_changed, stats.notes_unchanged = len(changed), len(unchanged)
    stats.notes_deleted = len(gone)

    for path in gone:
        store.delete_note(path)
    # Metadata (dates from git) can move without the file changing; refresh it cheaply.
    for n in unchanged:
        store.update_note(n.path, {"created_at": n.created_at, "updated_at": n.updated_at,
                                   "date_source": n.date_source})

    estimated_tokens = 0
    for i, n in enumerate(changed, 1):
        chunks = chunk_note(n.body, n.title, s.chunk_target_tokens, s.chunk_overlap_tokens)
        inputs = [embed_input(n.title, c) for c in chunks]
        hashes = [embed_hash(embedder.model, embedder.dim, t) for t in inputs]
        vectors = store.note_vectors(n.path) if n.path in stored else {}

        # Reused = this note already had a vector for identical input, or an identical
        # chunk earlier in the note is being embedded anyway.
        todo: dict[str, str] = {}
        for h, t in zip(hashes, inputs, strict=True):
            if h in vectors or h in todo:
                stats.chunks_reused += 1
            else:
                todo[h] = t
        if todo:
            res = embedder.embed_documents(list(todo.values()))
            vectors.update(zip(todo.keys(), res.vectors, strict=True))
            stats.chunks_embedded += len(todo)
            stats.tokens_embedded += res.tokens
            estimated_tokens += sum(estimate_tokens(t) for t in todo.values())
            if not res.tokens_reported:
                stats.token_source = "estimate"
            elif stats.token_source == "none":
                stats.token_source = "api"

        store.replace_note(_note_doc(n, fp), [
            {"note_path": n.path, "ordinal": c.ordinal, "heading_path": c.heading_path,
             "content": c.content, "tags": n.tags, "token_estimate": c.token_estimate,
             "embed_hash": h, "embedding_model": embedder.model, "embedding": vectors[h]}
            for c, h in zip(chunks, hashes, strict=True)
        ])
        log.info("[%d/%d] %s: %d chunks, %d embedded", i, len(changed), n.path,
                 len(chunks), len(todo))

    if stats.token_source == "estimate":
        stats.tokens_embedded = estimated_tokens
    stats.chunks_total = store.count_chunks()
    stats.est_cost_usd = round(stats.tokens_embedded / 1e6 * s.embed_price_per_1m_tokens, 6)
    stats.seconds = round(time.perf_counter() - t0, 2)
    store.record_run({**asdict(stats), "started_at": started, "embedding_model": embedder.model,
                      "price_per_1m_tokens": s.embed_price_per_1m_tokens})
    return stats


def _note_doc(n: Note, fp: str) -> dict:
    return {"path": n.path, "title": n.title, "folder": n.folder, "tags": n.tags,
            "links": n.links, "frontmatter": n.frontmatter, "created_at": n.created_at,
            "updated_at": n.updated_at, "date_source": n.date_source,
            "content_hash": n.content_hash, "fingerprint": fp,
            "ingested_at": dt.datetime.now(dt.UTC)}


def dry_run(s: Settings) -> dict:
    """Parse and chunk without touching the store or any API. Token counts are estimates."""
    notes = load_notes(s.vault_path, s.exclude_globs)
    chunks = chunk_all(notes, s)
    sizes = sorted(c.token_estimate for cs in chunks.values() for c in cs)
    total = sum(estimate_tokens(embed_input(n.title, c)) for n in notes for c in chunks[n.path])
    return {
        "notes": len(notes),
        "chunks": len(sizes),
        "est_embed_tokens": total,
        "est_cost_usd_full_embed": round(total / 1e6 * s.embed_price_per_1m_tokens, 4),
        "chunk_tokens_p50": sizes[len(sizes) // 2] if sizes else 0,
        "chunk_tokens_max": sizes[-1] if sizes else 0,
        "notes_by_date_source": {src: sum(n.date_source == src for n in notes)
                                 for src in ("frontmatter", "git", "mtime")},
        "notes_with_tags": sum(bool(n.tags) for n in notes),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="parse + chunk only; no store/API")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    s = Settings.from_env()

    if args.dry_run:
        print(json.dumps(dry_run(s), indent=2))
        return

    from sbrag.embed import GenAIEmbedder, make_genai_client
    from sbrag.store import FirestoreStore

    store = FirestoreStore(s.gcp_project, s.firestore_database, s.collection_prefix)
    client = make_genai_client(s.genai_backend, api_key=s.gemini_api_key,
                               project=s.gcp_project, location=s.gcp_location)
    embedder = GenAIEmbedder(client, s.embed_model, s.embed_dim)
    stats = run(s, store, embedder)
    print(json.dumps(asdict(stats), indent=2))


if __name__ == "__main__":
    main()
