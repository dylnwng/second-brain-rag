"""Embedding backends. `VertexEmbedder` is the real one; `HashEmbedder` is a deterministic
offline stand-in for tests and dry runs (it is not semantically meaningful)."""

from __future__ import annotations

import hashlib
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential

log = logging.getLogger(__name__)

# gemini-embedding-2 takes task instructions as a text prefix (documents use the
# "title: ... | text: ..." format built in chunk.embed_input).
QUERY_PREFIX = "task: search result | query: "


@dataclass
class EmbedResult:
    vectors: list[list[float]]
    tokens: int  # billed input tokens as reported by the API (0 if not reported)
    tokens_reported: bool


class Embedder(Protocol):
    model: str
    dim: int

    def embed_documents(self, texts: list[str]) -> EmbedResult: ...
    def embed_query(self, text: str) -> list[float]: ...


def _normalize(v: list[float]) -> list[float]:
    # Gemini embeddings are only unit-length at full dimensionality; truncated (MRL)
    # vectors need renormalizing so cosine and inner-product rankings agree.
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _is_retryable(exc: BaseException) -> bool:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return code in (429, 500, 502, 503, 504)


def make_genai_client(backend: str, *, api_key: str | None = None,
                      project: str | None = None, location: str = "global"):
    """`gemini_api`: AI Studio key, free tier available (free-tier inputs may be used by
    Google to improve its products). `vertex`: Vertex AI / Agent Platform, needs billing."""
    from google import genai  # imported lazily so offline tests don't need it configured

    from google.genai import types

    # Bounded timeout and SDK retries: callers do their own backoff (embeddings) or
    # fall back to another model (generation) instead of hanging for minutes.
    http = types.HttpOptions(timeout=60_000, retry_options=types.HttpRetryOptions(attempts=2))
    if backend == "gemini_api":
        if not api_key:
            raise SystemExit("GEMINI_API_KEY is not set (create one at aistudio.google.com)")
        return genai.Client(api_key=api_key, http_options=http)
    if backend == "vertex":
        if not project:
            raise SystemExit("GOOGLE_CLOUD_PROJECT is not set; can't call Vertex AI")
        return genai.Client(enterprise=True, project=project, location=location,
                            http_options=http)
    raise SystemExit(f"unknown GENAI_BACKEND {backend!r} (use gemini_api or vertex)")


class GenAIEmbedder:
    """Embeds through the google-genai SDK, whichever backend the client points at."""

    def __init__(self, client, model: str, dim: int, batch_size: int = 32, workers: int = 2):
        self.client = client
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self.workers = workers

    # Free-tier RPM limits are low, so back off patiently on 429 instead of failing a run.
    @retry(retry=retry_if_exception(_is_retryable), wait=wait_random_exponential(2, 60),
           stop=stop_after_attempt(8), reraise=True)
    def _call(self, contents):
        from google.genai import types

        if isinstance(contents, list):
            # A plain list of strings is read as the parts of ONE multimodal input by
            # gemini-embedding-2 (one vector back). One Content per text = one vector each.
            contents = [types.Content(parts=[types.Part(text=t)]) for t in contents]
        return self.client.models.embed_content(
            model=self.model, contents=contents,
            config=types.EmbedContentConfig(output_dimensionality=self.dim),
        )

    @staticmethod
    def _tokens(emb) -> int | None:
        stats = getattr(emb, "statistics", None)
        tokens = getattr(stats, "token_count", None) if stats else None
        return int(tokens) if tokens is not None else None

    def _embed_batch(self, texts: list[str]) -> list[tuple[list[float], int | None]]:
        resp = self._call(texts)
        if len(resp.embeddings) == len(texts):
            return [(_normalize(list(e.values)), self._tokens(e)) for e in resp.embeddings]
        # A multimodal model may fold a list into one embedding; fall back to one per call.
        log.warning("batch of %d returned %d embeddings; embedding one at a time",
                    len(texts), len(resp.embeddings))
        out = []
        for t in texts:
            e = self._call(t).embeddings[0]
            out.append((_normalize(list(e.values)), self._tokens(e)))
        return out

    def embed_documents(self, texts: list[str]) -> EmbedResult:
        batches = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            results = [r for batch in pool.map(self._embed_batch, batches) for r in batch]
        counts = [t for _, t in results]
        return EmbedResult(
            vectors=[v for v, _ in results],
            tokens=sum(c or 0 for c in counts),
            tokens_reported=all(c is not None for c in counts),
        )

    def embed_query(self, text: str) -> list[float]:
        return _normalize(list(self._call(QUERY_PREFIX + text).embeddings[0].values))


class HashEmbedder:
    """Deterministic pseudo-embeddings from SHA-256. Same text -> same vector."""

    model = "hash-test"

    def __init__(self, dim: int = 1536):
        self.dim = dim
        self.calls = 0

    def _vec(self, text: str) -> list[float]:
        out: list[float] = []
        seed = text.encode("utf-8")
        i = 0
        while len(out) < self.dim:
            h = hashlib.sha256(seed + i.to_bytes(4, "little")).digest()
            out.extend((b - 127.5) / 127.5 for b in h)
            i += 1
        return _normalize(out[: self.dim])

    def embed_documents(self, texts: list[str]) -> EmbedResult:
        self.calls += len(texts)
        return EmbedResult([self._vec(t) for t in texts], 0, False)

    def embed_query(self, text: str) -> list[float]:
        return self._vec(QUERY_PREFIX + text)
