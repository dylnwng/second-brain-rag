"""Keyword baseline + eval harness, offline."""

import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from evals.run_eval import hit, load_golden, run, summarize
from sbrag.embed import HashEmbedder
from sbrag.rag import RAG, REFUSAL
from sbrag.retrieve import KeywordRetriever, VectorRetriever, tokenize
from sbrag.store import MemoryStore

GOLDEN = Path(__file__).resolve().parents[1] / "evals" / "golden.jsonl"

CHUNKS = [
    {"note_path": "q.md", "heading_path": "Queueing > VUT",
     "content": "The VUT equation: waiting time grows with variability and utilization."},
    {"note_path": "lean.md", "heading_path": "Lean > Muda",
     "content": "Seven wastes: overproduction, inventory, waiting, processing."},
    {"note_path": "gym.md", "heading_path": "Fitness Plan",
     "content": "Monday upper body, Tuesday lower body and plyometrics."},
]


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("What is my VUT-equation?") == ["vut", "equation"]


def test_bm25_ranks_keyword_match_first():
    hits, timing = KeywordRetriever(CHUNKS).retrieve("what are the seven wastes", 3)
    assert hits[0].note_path == "lean.md"
    assert timing["embed"] == 0.0


def test_bm25_returns_nothing_for_unknown_terms():
    assert KeywordRetriever(CHUNKS).retrieve("zebra quantum", 3)[0] == []


def test_bm25_scores_lower_is_better_like_vector_distance():
    hits, _ = KeywordRetriever(CHUNKS).retrieve("waiting utilization variability", 3)
    assert [h.distance for h in hits] == sorted(h.distance for h in hits)


def test_hit_is_note_level():
    assert hit(["a.md", "b.md"], ["b.md", "c.md"])
    assert not hit(["a.md"], ["c.md"])


GOLDEN_FILES = sorted((GOLDEN.parent).rglob("golden*.jsonl"))


@pytest.mark.parametrize("path", GOLDEN_FILES, ids=lambda p: str(p.relative_to(GOLDEN.parent)))
def test_golden_file_shape(path):
    rows = load_golden(path)
    assert len(rows) >= 5
    assert any(not r["answerable"] for r in rows), "need unanswerable questions"
    assert all(r["question"].endswith("?") for r in rows)
    assert all(r["expected_notes"] == [] for r in rows if not r["answerable"])


def test_load_golden_rejects_bad_rows(tmp_path):
    p = tmp_path / "g.jsonl"
    p.write_text(json.dumps({"id": "x", "question": "q?", "answerable": True,
                             "expected_notes": [], "reference": ""}) + "\n")
    with pytest.raises(SystemExit, match="expected_notes"):
        load_golden(p)


def _rag(retriever, answer):
    class Models:
        def generate_content(self, model, contents, config):
            return NS(text=json.dumps(answer), usage_metadata=NS(
                prompt_token_count=100, candidates_token_count=10,
                thoughts_token_count=5, total_token_count=115))
    return RAG(retriever, NS(models=Models()), ["m"], k=2)


def test_run_and_summarize_end_to_end_offline():
    store = MemoryStore()
    emb = HashEmbedder(16)
    for i, c in enumerate(CHUNKS):
        store.chunks[str(i)] = {**c, "tags": [], "embedding": emb._vec(c["content"])}
    retrievers = {"vector": VectorRetriever(store, emb), "keyword": KeywordRetriever(CHUNKS)}
    rags = {"vector": _rag(retrievers["vector"], {"supported": True, "answer": "x [1]"}),
            "keyword": _rag(retrievers["keyword"], {"supported": False, "answer": REFUSAL})}
    golden = [
        {"id": "a1", "question": "seven wastes?", "answerable": True,
         "expected_notes": ["lean.md"], "reference": ""},
        {"id": "u1", "question": "capital of Australia?", "answerable": False,
         "expected_notes": [], "reference": ""},
    ]
    per_q, answers = run(golden, retrievers, rags, k=2, log=lambda m: None)
    assert len(per_q) == 4 and len(answers) == 4

    kw = summarize([r for r in per_q if r["system"] == "keyword"], 2, False)
    assert kw["hits_at_2"] == "1/1"            # BM25 finds lean.md
    assert kw["correct_refusals"] == "1/1"     # fake model always refuses...
    assert kw["false_refusals"] == "1/1"       # ...which false_refusal exposes
    # BM25 has no terms for the out-of-scope question -> no hits -> refused without a
    # model call (0 tokens), so the per-request mean is (115 + 0) / 2.
    u1_kw = next(r for r in per_q if r["id"] == "u1" and r["system"] == "keyword")
    assert u1_kw["refusal_reason"] == "no_hits" and u1_kw["tokens"]["total"] == 0
    assert kw["mean_tokens_per_request"]["total"] == 57.5
    vec = summarize([r for r in per_q if r["system"] == "vector"], 2, False)
    assert vec["correct_refusals"] == "0/1" and vec["false_refusals"] == "0/1"
    assert vec["errors"] == 0


def test_retrieval_only_mode():
    retrievers = {"keyword": KeywordRetriever(CHUNKS)}
    golden = [{"id": "a1", "question": "VUT equation?", "answerable": True,
               "expected_notes": ["q.md"], "reference": ""}]
    per_q, answers = run(golden, retrievers, {}, k=1, retrieval_only=True, log=lambda m: None)
    assert per_q[0]["hit"] and answers == []
    assert summarize(per_q, 1, True)["hit_rate_at_1"] == 1.0


def test_report_renders_counts_next_to_rates():
    from evals.report import render

    sys_ = {"hit_rate_at_5": 0.867, "hits_at_5": "13/15", "correct_refusal_rate": 1.0,
            "correct_refusals": "5/5", "false_refusal_rate": 0.0, "false_refusals": "0/15",
            "mean_latency_ms": {"embed": 300.0, "retrieve": 400.0, "generate": 3000.0,
                                "total": 3700.0},
            "p50_total_latency_ms": 3500.0,
            "mean_tokens_per_request": {"prompt": 1000, "output": 50, "thinking": 50,
                                        "total": 1100.0},
            "requests": 20, "errors": 0}
    run_ = {"k": 5, "retrieval_only": False, "questions": 20, "answerable": 15,
            "unanswerable": 5, "golden_file": "golden.jsonl", "embed_model": "e",
            "gen_models": ["g"], "git_commit": "abc", "started_at": "t"}
    out = render({"run": run_, "systems": {"vector": sys_}})
    assert "| Retrieval hit rate @ 5 (answerable) | 87% (13/15) |" in out
    assert "| Correct refusal (unanswerable) | 100% (5/5) |" in out
    assert "| Mean latency, retrieval only (ms) | 700 |" in out
    assert "commit `abc`" in out


class _DownModels:
    """Every model returns 429, like an exhausted free-tier daily quota."""

    class Err(Exception):
        code = 429

    def generate_content(self, model, contents, config):
        raise self.Err()


def test_generation_outage_does_not_count_as_retrieval_miss():
    retrievers = {"keyword": KeywordRetriever(CHUNKS)}
    rags = {"keyword": RAG(retrievers["keyword"], NS(models=_DownModels()), ["a", "b"], k=2)}
    golden = [{"id": "a1", "question": "seven wastes?", "answerable": True,
               "expected_notes": ["lean.md"], "reference": ""}]
    per_q, _ = run(golden, retrievers, rags, k=2, log=lambda m: None)
    row = per_q[0]
    assert "a: 429; b: 429" in row["error"]
    assert row["hit"] is True and row["retrieved_notes"][0] == "lean.md"
    s = summarize(per_q, 2, False)
    assert s["hits_at_2"] == "1/1" and s["errors"] == 1 and s["false_refusals"] == "0/0"


def test_resume_keeps_done_rows_and_reruns_the_rest():
    retrievers = {"keyword": KeywordRetriever(CHUNKS)}
    calls = []

    class Models:
        def generate_content(self, model, contents, config):
            calls.append(contents)
            return NS(text=json.dumps({"supported": True, "answer": "x [1]"}),
                      usage_metadata=None)

    rags = {"keyword": RAG(retrievers["keyword"], NS(models=Models()), ["m"], k=2)}
    golden = [{"id": i, "question": "seven wastes?", "answerable": True,
               "expected_notes": ["lean.md"], "reference": ""} for i in ("a1", "a2")]
    done = {("a1", "keyword"): {"id": "a1", "system": "keyword", "kept": True}}
    per_q, answers = run(golden, retrievers, rags, k=2, log=lambda m: None, done=done)
    assert per_q[0] == done[("a1", "keyword")] and per_q[1]["id"] == "a2"
    assert len(calls) == 1 and [a["id"] for a in answers] == ["a2"]


def test_summary_export_drops_per_question_rows():
    from evals.report import render, summary

    results = {"run": {"k": 5, "retrieval_only": True, "questions": 1, "answerable": 1,
                       "unanswerable": 0, "golden_file": "g", "embed_model": "e",
                       "gen_models": [], "git_commit": "c", "started_at": "t"},
               "systems": {"v": {"hit_rate_at_5": 1.0, "hits_at_5": "1/1",
                                 "mean_retrieval_latency_ms": 1.0}},
               "per_question": [{"id": "q1", "retrieved_notes": ["private/diary.md"]}]}
    s = summary(results)
    assert "per_question" not in s and "private/diary.md" not in json.dumps(s)
    assert render(s) == render(results)
