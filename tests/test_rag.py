"""RAG + API tests with a fake Gemini client, MemoryStore and HashEmbedder (offline)."""

import json
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

from sbrag.api import create_app
from sbrag.embed import HashEmbedder
from sbrag.retrieve import VectorRetriever
from sbrag.rag import RAG, REFUSAL, UpstreamUnavailable, cited_numbers, format_context
from sbrag.store import Hit, MemoryStore

DIM = 16


class APIErr(Exception):
    def __init__(self, code):
        super().__init__(f"HTTP {code}")
        self.code = code


class FakeModels:
    def __init__(self, payload, fail: dict | None = None):
        self.payload = payload
        self.fail = fail or {}  # model -> HTTP status to raise
        self.prompts = []
        self.models_called = []

    def generate_content(self, model, contents, config):
        self.models_called.append(model)
        if model in self.fail:
            raise APIErr(self.fail[model])
        self.prompts.append(contents)
        text = self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        usage = NS(prompt_token_count=900, candidates_token_count=40,
                   thoughts_token_count=25, total_token_count=965)
        return NS(text=text, usage_metadata=usage)


def store_with(n: int) -> MemoryStore:
    store, emb = MemoryStore(), HashEmbedder(DIM)
    for i in range(n):
        store.chunks[f"c{i}"] = {
            "note_path": f"notes/n{i}.md", "heading_path": f"Note {i} > Section",
            "content": f"fact number {i}", "tags": [], "embedding": emb._vec(f"fact {i}"),
        }
    return store


def rag(payload, n_chunks=6, k=5, chain=("fake-model",), fail=None):
    models = FakeModels(payload, fail)
    retriever = VectorRetriever(store_with(n_chunks), HashEmbedder(DIM))
    r = RAG(retriever, NS(models=models), list(chain), k=k)
    return r, models


def test_supported_answer_keeps_cited_sources_only():
    r, models = rag({"supported": True, "answer": "Fact A [1]. Fact B [2, 4]."})
    res = r.ask("what facts?")
    assert not res.refused and res.refusal_reason is None
    assert [s.n for s in res.sources] == [1, 2, 4]
    assert all(s.note_path.startswith("notes/") and s.heading for s in res.sources)
    assert len(res.retrieved) == 5
    assert res.tokens == {"prompt": 900, "output": 40, "thinking": 25, "total": 965}
    assert set(res.latency_ms) == {"embed", "retrieve", "generate", "total"}
    assert "Question: what facts?" in models.prompts[0]


@pytest.mark.parametrize("payload,reason", [
    ({"supported": False, "answer": REFUSAL}, "model"),
    ({"supported": True, "answer": REFUSAL}, "model"),
    ({"supported": True, "answer": "Confident claim with no citation."}, "no_citations"),
    ({"supported": True, "answer": "Claim [7]."}, "bad_citation"),
    ({"supported": True, "answer": "Claim [0]."}, "bad_citation"),
    ("not json at all", "unparseable"),
    ({"answer": "missing supported flag [1]"}, "unparseable"),
])
def test_refusals(payload, reason):
    res = rag(payload)[0].ask("q")
    assert res.refused and res.answer == REFUSAL and res.refusal_reason == reason
    assert res.sources == []
    assert res.tokens["total"] == 965  # the model was still called and billed


def test_empty_store_refuses_without_calling_model():
    r, models = rag({"supported": True, "answer": "x [1]"}, n_chunks=0)
    res = r.ask("q")
    assert res.refused and res.refusal_reason == "no_hits"
    assert models.prompts == [] and res.tokens["total"] == 0


def test_fewer_chunks_than_k_bounds_citations():
    res = rag({"supported": True, "answer": "x [3]"}, n_chunks=2)[0].ask("q")
    assert res.refusal_reason == "bad_citation"


def test_cited_numbers_parsing():
    assert cited_numbers("a [1] b [2, 3] c [ 4 ] [x]") == {1, 2, 3}
    assert cited_numbers("no cites") == set()


def test_context_is_numbered_with_path_and_heading():
    ctx = format_context([Hit("a.md", "A > B", "body", [], 0.1)])
    assert ctx == "[1] (note: a.md | section: A > B)\nbody"


def test_falls_back_to_next_model_on_overload():
    r, models = rag({"supported": True, "answer": "x [1]"}, chain=("a", "b", "c"),
                    fail={"a": 503, "b": 429})
    res = r.ask("q")
    assert models.models_called == ["a", "b", "c"] and res.model == "c" and not res.refused


def test_quota_exhausted_model_is_skipped_until_cooldown(monkeypatch):
    import sbrag.rag as rag_mod

    clock = [1000.0]
    monkeypatch.setattr(rag_mod.time, "monotonic", lambda: clock[0])
    r, models = rag({"supported": True, "answer": "x [1]"}, chain=("a", "b"), fail={"a": 429})
    r.ask("q1")
    r.ask("q2")
    assert models.models_called == ["a", "b", "b"]  # "a" skipped on the 2nd request
    clock[0] += rag_mod.QUOTA_COOLDOWN_S + 1
    r.ask("q3")
    assert models.models_called[-2:] == ["a", "b"]  # retried after the cooldown


def test_503_does_not_trigger_cooldown():
    r, models = rag({"supported": True, "answer": "x [1]"}, chain=("a", "b"), fail={"a": 503})
    r.ask("q1")
    r.ask("q2")
    assert models.models_called == ["a", "b", "a", "b"]


def test_non_retryable_error_is_not_swallowed():
    r, models = rag({"supported": True, "answer": "x [1]"}, chain=("a", "b"), fail={"a": 400})
    with pytest.raises(APIErr):
        r.ask("q")
    assert models.models_called == ["a"]


def test_all_models_down_raises_upstream_unavailable():
    r, _ = rag({"supported": True, "answer": "x [1]"}, chain=("a", "b"),
               fail={"a": 503, "b": 503})
    with pytest.raises(UpstreamUnavailable, match="a: 503; b: 503"):
        r.ask("q")


def test_model_is_none_when_no_model_called():
    assert rag({"supported": True, "answer": "x [1]"}, n_chunks=0)[0].ask("q").model is None


# ---------------------------------------------------------------- API

TOKEN = "test-token"
AUTH = {"authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def api():
    r, _ = rag({"supported": True, "answer": "Fact [1]."})
    with TestClient(create_app(lambda: r, ask_token=TOKEN, allow_no_auth=False), headers=AUTH) as c:
        yield c


def test_ask_endpoint(api):
    resp = api.post("/ask", json={"question": "what?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Fact [1]."
    assert body["sources"] == [{"n": 1, "note_path": body["sources"][0]["note_path"],
                                "heading": body["sources"][0]["heading"],
                                "distance": body["sources"][0]["distance"]}]
    assert body["tokens"]["total"] == 965 and body["latency_ms"]["total"] >= 0
    assert "retrieved" not in body


def test_ask_validates_input(api):
    assert api.post("/ask", json={"question": ""}).status_code == 422
    assert api.post("/ask", json={}).status_code == 422


def test_ask_returns_503_when_all_models_down():
    r, _ = rag({"supported": True, "answer": "x [1]"}, chain=("a",), fail={"a": 503})
    with TestClient(create_app(lambda: r, ask_token=TOKEN, allow_no_auth=False), headers=AUTH) as c:
        resp = c.post("/ask", json={"question": "q"})
    assert resp.status_code == 503 and "a: 503" in resp.json()["detail"]
    assert resp.headers["retry-after"] == "30"


def test_healthz(api):
    assert api.get("/healthz").json() == {"ok": True}


@pytest.mark.parametrize("headers", [{}, {"authorization": "Bearer wrong"},
                                     {"authorization": TOKEN}])
def test_ask_rejects_missing_or_wrong_token(headers):
    r, models = rag({"supported": True, "answer": "Fact [1]."})
    with TestClient(create_app(lambda: r, ask_token=TOKEN, allow_no_auth=False)) as c:
        resp = c.post("/ask", json={"question": "q"}, headers=headers)
    assert resp.status_code == 401 and models.prompts == []


def test_ask_fails_closed_without_configured_token():
    r, models = rag({"supported": True, "answer": "Fact [1]."})
    with TestClient(create_app(lambda: r, ask_token=None, allow_no_auth=False)) as c:
        resp = c.post("/ask", json={"question": "q"})
        assert c.get("/healthz").status_code == 200  # health stays open
    assert resp.status_code == 503 and "ASK_TOKEN" in resp.json()["detail"]
    assert models.prompts == []


def test_no_auth_only_with_explicit_opt_in():
    r, _ = rag({"supported": True, "answer": "Fact [1]."})
    with TestClient(create_app(lambda: r, ask_token=None, allow_no_auth=True)) as c:
        assert c.post("/ask", json={"question": "q"}).status_code == 200


def test_firestore_store_reads_credentials_from_env(monkeypatch):
    import sbrag.store as store_mod

    captured = {}

    class FakeClient:
        def __init__(self, project, database, credentials):
            captured.update(project=project, credentials=credentials)

    monkeypatch.setattr("google.cloud.firestore.Client", FakeClient)
    monkeypatch.setattr("google.oauth2.service_account.Credentials.from_service_account_info",
                        staticmethod(lambda info: ("creds-for", info["client_email"])))
    monkeypatch.setenv("GOOGLE_CREDENTIALS_JSON",
                       json.dumps({"project_id": "p1", "client_email": "sa@p1"}))
    store = store_mod.FirestoreStore(None, prefix="demo_")
    assert captured == {"project": "p1", "credentials": ("creds-for", "sa@p1")}
    assert (store.notes_col, store.chunks_col, store.runs_col) == (
        "demo_notes", "demo_chunks", "demo_ingest_runs")
