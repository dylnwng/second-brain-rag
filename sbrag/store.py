"""Vector store. `FirestoreStore` is the real one (Firestore native vector search);
`MemoryStore` has the same interface for offline tests.

Layout:
  notes/{note_id}     path, title, folder, tags, dates, content_hash, fingerprint
  chunks/{note_id}-{ordinal}
                      note_path, heading_path, tags, content, embed_hash, embedding (Vector)
  ingest_runs/{auto}  counts + cost of each ingestion run
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Protocol

NOTES, CHUNKS, RUNS = "notes", "chunks", "ingest_runs"
BATCH_LIMIT = 400  # Firestore allows 500 writes per batch; stay under it


def note_id(path: str) -> str:
    """Firestore IDs can't contain '/', so key notes by a hash of their path."""
    return hashlib.sha1(path.encode()).hexdigest()[:20]


def chunk_id(path: str, ordinal: int) -> str:
    return f"{note_id(path)}-{ordinal:04d}"


@dataclass
class Hit:
    note_path: str
    heading_path: str
    content: str
    tags: list[str]
    distance: float


class Store(Protocol):
    def note_states(self) -> dict[str, tuple[str, str]]: ...
    def note_vectors(self, path: str) -> dict[str, list[float]]: ...
    def replace_note(self, note: dict, chunks: list[dict]) -> None: ...
    def update_note(self, path: str, fields: dict) -> None: ...
    def delete_note(self, path: str) -> None: ...
    def count_chunks(self) -> int: ...
    def record_run(self, run: dict) -> None: ...
    def search(self, vector: list[float], k: int) -> list[Hit]: ...
    def all_chunks(self) -> list[dict]: ...


class FirestoreStore:
    def __init__(self, project: str | None, database: str = "(default)", prefix: str = ""):
        """prefix namespaces the collections (e.g. "demo_") so a second corpus can share
        the database without mixing with the first."""
        import json
        import os

        from google.cloud import firestore

        # On GCP (Cloud Run) credentials come from the attached service account via ADC.
        # Off GCP (Render) there's no metadata server, so a read-only service-account key
        # is passed as JSON in GOOGLE_CREDENTIALS_JSON (a host secret, never a file in git).
        creds = None
        if raw := os.environ.get("GOOGLE_CREDENTIALS_JSON"):
            from google.oauth2 import service_account

            info = json.loads(raw)
            creds = service_account.Credentials.from_service_account_info(info)
            project = project or info.get("project_id")
        self.db = firestore.Client(project=project, database=database, credentials=creds)
        self.notes_col, self.chunks_col, self.runs_col = (
            prefix + NOTES, prefix + CHUNKS, prefix + RUNS)

    def note_states(self):
        return {d.get("path"): (d.get("content_hash"), d.get("fingerprint"))
                for d in self.db.collection(self.notes_col).select(
                    ["path", "content_hash", "fingerprint"]).stream()}

    def note_vectors(self, path):
        q = (self.db.collection(self.chunks_col).where("note_path", "==", path)
             .select(["embed_hash", "embedding"]))
        return {d.get("embed_hash"): list(d.get("embedding")) for d in q.stream()}

    def _old_chunk_refs(self, path):
        q = self.db.collection(self.chunks_col).where("note_path", "==", path).select([])
        return [d.reference for d in q.stream()]

    def _commit(self, ops):
        """ops: list of (kind, ref, data). Splits across batches of BATCH_LIMIT."""
        for i in range(0, len(ops), BATCH_LIMIT):
            batch = self.db.batch()
            for kind, ref, data in ops[i:i + BATCH_LIMIT]:
                if kind == "set":
                    batch.set(ref, data)
                else:
                    batch.delete(ref)
            batch.commit()

    def replace_note(self, note, chunks):
        from google.cloud.firestore_v1.vector import Vector

        path = note["path"]
        new_ids = {chunk_id(path, c["ordinal"]) for c in chunks}
        ops = [("delete", r, None) for r in self._old_chunk_refs(path) if r.id not in new_ids]
        for c in chunks:
            data = {**c, "embedding": Vector(c["embedding"])}
            ops.append(("set", self.db.collection(self.chunks_col).document(chunk_id(path, c["ordinal"])),
                        data))
        # Note doc last: if a run dies mid-note, the stale hash forces a redo next time.
        ops.append(("set", self.db.collection(self.notes_col).document(note_id(path)), note))
        self._commit(ops)

    def update_note(self, path, fields):
        self.db.collection(self.notes_col).document(note_id(path)).update(fields)

    def delete_note(self, path):
        ops = [("delete", r, None) for r in self._old_chunk_refs(path)]
        ops.append(("delete", self.db.collection(self.notes_col).document(note_id(path)), None))
        self._commit(ops)

    def count_chunks(self):
        return int(self.db.collection(self.chunks_col).count().get()[0][0].value)

    def record_run(self, run):
        self.db.collection(self.runs_col).add(run)

    def all_chunks(self):
        """Chunk text + metadata without vectors (for the keyword baseline)."""
        fields = ["note_path", "ordinal", "heading_path", "content", "tags"]
        return [d.to_dict() for d in self.db.collection(self.chunks_col).select(fields).stream()]

    def search(self, vector, k):
        from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
        from google.cloud.firestore_v1.vector import Vector

        q = self.db.collection(self.chunks_col).find_nearest(
            vector_field="embedding", query_vector=Vector(vector),
            distance_measure=DistanceMeasure.COSINE, limit=k,
            distance_result_field="distance",
        )
        return [Hit(d.get("note_path"), d.get("heading_path"), d.get("content"),
                    d.get("tags") or [], d.get("distance")) for d in q.stream()]


class MemoryStore:
    def __init__(self):
        self.notes: dict[str, dict] = {}
        self.chunks: dict[str, dict] = {}
        self.runs: list[dict] = []

    def note_states(self):
        return {n["path"]: (n["content_hash"], n["fingerprint"]) for n in self.notes.values()}

    def note_vectors(self, path):
        return {c["embed_hash"]: c["embedding"] for c in self.chunks.values()
                if c["note_path"] == path}

    def replace_note(self, note, chunks):
        path = note["path"]
        for cid in [k for k, c in self.chunks.items() if c["note_path"] == path]:
            del self.chunks[cid]
        for c in chunks:
            self.chunks[chunk_id(path, c["ordinal"])] = dict(c)
        self.notes[note_id(path)] = dict(note)

    def update_note(self, path, fields):
        self.notes[note_id(path)].update(fields)

    def delete_note(self, path):
        for cid in [k for k, c in self.chunks.items() if c["note_path"] == path]:
            del self.chunks[cid]
        self.notes.pop(note_id(path), None)

    def count_chunks(self):
        return len(self.chunks)

    def record_run(self, run):
        self.runs.append(dict(run))

    def all_chunks(self):
        return [{k: v for k, v in c.items() if k != "embedding"} for c in self.chunks.values()]

    def search(self, vector, k):
        def cos_dist(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
            return 1 - dot / ((na * nb) or 1)

        scored = sorted(self.chunks.values(), key=lambda c: cos_dist(vector, c["embedding"]))
        return [Hit(c["note_path"], c["heading_path"], c["content"], c.get("tags", []),
                    cos_dist(vector, c["embedding"])) for c in scored[:k]]
