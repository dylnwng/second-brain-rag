"""GenAIEmbedder against a fake google-genai client: batching, fallback, retry, normalization."""

import math
from types import SimpleNamespace as NS

import pytest

from sbrag.embed import QUERY_PREFIX, GenAIEmbedder, make_genai_client


class FakeModels:
    def __init__(self, fold_lists=False, fail_first=0, with_stats=True):
        self.calls = []
        self.fold_lists = fold_lists
        self.fail_first = fail_first
        self.with_stats = with_stats

    def _emb(self, text):
        stats = NS(token_count=len(text) // 4) if self.with_stats else None
        return NS(values=[3.0, 4.0] + [0.0] * 2, statistics=stats)  # norm 5 before normalizing

    def embed_content(self, model, contents, config):
        self.calls.append(contents)
        if self.fail_first:
            self.fail_first -= 1
            raise _Err429()
        items = ([c.parts[0].text for c in contents] if isinstance(contents, list)
                 else [contents])
        if self.fold_lists and isinstance(contents, list) and len(items) > 1:
            return NS(embeddings=[self._emb("".join(items))])
        return NS(embeddings=[self._emb(t) for t in items])


class _Err429(Exception):
    code = 429


def embedder(models, batch_size=3):
    return GenAIEmbedder(NS(models=models), "m", 4, batch_size=batch_size, workers=1)


def test_batches_requests_and_counts_tokens():
    models = FakeModels()
    res = embedder(models).embed_documents([f"text {i:03d}" for i in range(7)])
    assert len(res.vectors) == 7
    assert [len(c) for c in models.calls] == [3, 3, 1]
    assert res.tokens_reported and res.tokens == 7 * (len("text 000") // 4)


def test_vectors_are_unit_length():
    res = embedder(FakeModels()).embed_documents(["a"])
    assert math.isclose(math.sqrt(sum(x * x for x in res.vectors[0])), 1.0)


def test_falls_back_to_single_calls_if_batch_is_folded():
    models = FakeModels(fold_lists=True)
    res = embedder(models).embed_documents(["a", "b", "c"])
    assert len(res.vectors) == 3
    assert len(models.calls) == 4 and models.calls[1:] == ["a", "b", "c"]


def test_missing_token_stats_marked_unreported():
    res = embedder(FakeModels(with_stats=False)).embed_documents(["a", "b"])
    assert not res.tokens_reported and res.tokens == 0


def test_retries_on_429(monkeypatch):
    monkeypatch.setattr(GenAIEmbedder._call.retry, "sleep", lambda s: None)
    models = FakeModels(fail_first=2)
    res = embedder(models).embed_documents(["a"])
    assert len(res.vectors) == 1 and len(models.calls) == 3


def test_query_uses_retrieval_prefix():
    models = FakeModels()
    embedder(models).embed_query("what is VUT?")
    assert models.calls == [QUERY_PREFIX + "what is VUT?"]


def test_client_factory_validates_config():
    with pytest.raises(SystemExit, match="GEMINI_API_KEY"):
        make_genai_client("gemini_api", api_key=None)
    with pytest.raises(SystemExit, match="GOOGLE_CLOUD_PROJECT"):
        make_genai_client("vertex", project=None)
    with pytest.raises(SystemExit, match="unknown"):
        make_genai_client("openai")
