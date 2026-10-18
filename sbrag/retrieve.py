"""Retrievers. Both return (hits, timings_ms) so RAG and the eval treat them the same.

- VectorRetriever: embed the question, nearest neighbours in the store (the real system).
- KeywordRetriever: BM25 over the same chunks (the baseline). No embeddings involved.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from typing import Protocol

from sbrag.embed import Embedder
from sbrag.store import Hit, Store

TOKEN_RE = re.compile(r"[a-z0-9]+")
# Small English stoplist; question words matter little to BM25 and would otherwise dominate
# short queries ("what", "my", "did").
STOPWORDS = frozenset("""a an and are as at be by did do does for from how i in is it me my
of on or the to was were what when where which who why will with you your""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class Retriever(Protocol):
    name: str

    def retrieve(self, question: str, k: int) -> tuple[list[Hit], dict]: ...


def _ms(a: float, b: float) -> float:
    return round((b - a) * 1000, 1)


class VectorRetriever:
    name = "vector"

    def __init__(self, store: Store, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    def retrieve(self, question, k):
        t0 = time.perf_counter()
        qvec = self.embedder.embed_query(question)
        t1 = time.perf_counter()
        hits = self.store.search(qvec, k)
        t2 = time.perf_counter()
        return hits, {"embed": _ms(t0, t1), "retrieve": _ms(t1, t2)}


class KeywordRetriever:
    """Okapi BM25 over heading path + content, the same text the vector side embeds.
    Hit.distance is the negated BM25 score, so lower is better on both retrievers."""

    name = "keyword"

    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(f"{c['heading_path']} {c['content']}")) for c in chunks]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avgdl = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, question: str) -> list[float]:
        q = tokenize(question)
        out = []
        for d, dl in zip(self.docs, self.lens, strict=True):
            s = 0.0
            for t in q:
                tf = d.get(t)
                if tf:
                    norm = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                    s += self.idf[t] * tf * (self.k1 + 1) / norm
            out.append(s)
        return out

    def retrieve(self, question, k):
        t0 = time.perf_counter()
        scored = sorted(zip(self.scores(question), range(len(self.chunks))), reverse=True)
        hits = [Hit(self.chunks[i]["note_path"], self.chunks[i]["heading_path"],
                    self.chunks[i]["content"], self.chunks[i].get("tags") or [], -s)
                for s, i in scored[:k] if s > 0]
        return hits, {"embed": 0.0, "retrieve": _ms(t0, time.perf_counter())}
