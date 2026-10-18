"""Evaluate vector RAG against a keyword (BM25) baseline on evals/golden.jsonl.

Both systems share everything except retrieval: same chunks, same top-k, same Gemini
models, prompt and refusal rules. So differences come from retrieval alone.

    .venv/bin/python -m evals.run_eval                    # full run -> evals/results.json
    .venv/bin/python -m evals.run_eval --retrieval-only   # no generation calls

Metrics (per system):
  hit_rate_at_k      answerable questions where an expected note is among the top-k chunks
  correct_refusal    unanswerable questions answered exactly with the refusal message
  false_refusal      answerable questions wrongly refused (a system that always refuses
                     would ace correct_refusal; this keeps it honest)
  latency / tokens   means over requests that completed (errors are counted, not averaged)

Answers go to evals/answers.jsonl (gitignored: it quotes your private notes) for spot checks;
results.json holds only ids, note paths, flags and numbers.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_golden(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate ids in golden set")
    for r in rows:
        if r["answerable"] and not r["expected_notes"]:
            raise SystemExit(f"{r['id']}: answerable question needs expected_notes")
    return rows


def hit(retrieved_notes: list[str], expected: list[str]) -> bool:
    return bool(set(retrieved_notes) & set(expected))


def _mean(xs):
    return round(statistics.fmean(xs), 1) if xs else None


def summarize(rows: list[dict], k: int, retrieval_only: bool) -> dict:
    ans = [r for r in rows if r["answerable"]]
    hits = sum(r["hit"] for r in ans)
    out = {
        f"hit_rate_at_{k}": round(hits / len(ans), 3) if ans else None,
        f"hits_at_{k}": f"{hits}/{len(ans)}",
    }
    if retrieval_only:
        out["mean_retrieval_latency_ms"] = _mean([r["latency_ms"]["retrieve"] + r["latency_ms"]["embed"]
                                                  for r in rows])
        return out
    ok = [r for r in rows if not r.get("error")]
    unans = [r for r in ok if not r["answerable"]]
    ans_ok = [r for r in ok if r["answerable"]]
    correct_ref = sum(r["refused"] for r in unans)
    false_ref = sum(r["refused"] for r in ans_ok)
    out.update({
        "correct_refusal_rate": round(correct_ref / len(unans), 3) if unans else None,
        "correct_refusals": f"{correct_ref}/{len(unans)}",
        "false_refusal_rate": round(false_ref / len(ans_ok), 3) if ans_ok else None,
        "false_refusals": f"{false_ref}/{len(ans_ok)}",
        "mean_latency_ms": {stage: _mean([r["latency_ms"][stage] for r in ok])
                            for stage in ("embed", "retrieve", "generate", "total")},
        "p50_total_latency_ms": round(statistics.median([r["latency_ms"]["total"] for r in ok]), 1)
        if ok else None,
        "mean_tokens_per_request": {t: _mean([r["tokens"][t] for r in ok])
                                    for t in ("prompt", "output", "thinking", "total")},
        "models_used": {m: sum(r["model"] == m for r in ok) for m in {r["model"] for r in ok}},
        "requests": len(rows),
        "errors": len(rows) - len(ok),
    })
    return out


def git_commit() -> str | None:
    try:
        # -dirty marks results produced from uncommitted code (not reproducible from git).
        return subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True,
                              text=True, check=True, cwd=HERE.parent).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def build(args):
    """Real dependencies: Firestore chunks/vectors + Gemini (same config as the API)."""
    from sbrag.api import DEFAULT_GEN_MODELS
    from sbrag.config import Settings
    from sbrag.embed import GenAIEmbedder, make_genai_client
    from sbrag.rag import RAG
    from sbrag.retrieve import KeywordRetriever, VectorRetriever
    from sbrag.store import FirestoreStore

    s = Settings.from_env(require_vault=False)
    client = make_genai_client(s.genai_backend, api_key=s.gemini_api_key,
                               project=s.gcp_project, location=s.gcp_location)
    store = FirestoreStore(s.gcp_project, s.firestore_database, s.collection_prefix)
    models = [m.strip() for m in os.environ.get("GEN_MODELS", DEFAULT_GEN_MODELS).split(",")
              if m.strip()]
    retrievers = {
        "vector": VectorRetriever(store, GenAIEmbedder(client, s.embed_model, s.embed_dim)),
        "keyword": KeywordRetriever(store.all_chunks()),
    }
    rags = {name: RAG(r, client, models, k=args.k) for name, r in retrievers.items()}
    config = {"embed_model": s.embed_model, "embed_dim": s.embed_dim,
              "gen_models": models, "backend": s.genai_backend}
    return retrievers, rags, config


def run(golden, retrievers, rags, k, retrieval_only=False, pause=0.0, log=print, done=None):
    """done: {(id, system): row} of completed rows to keep instead of re-running."""
    from sbrag.rag import UpstreamUnavailable

    done = done or {}
    per_q, answers = [], []
    for g in golden:
        for name in retrievers:
            if (g["id"], name) in done:
                per_q.append(done[(g["id"], name)])
                continue
            row = {"id": g["id"], "system": name, "answerable": g["answerable"],
                   "at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}
            try:
                if retrieval_only:
                    hits, timing = retrievers[name].retrieve(g["question"], k)
                    notes = [h.note_path for h in hits]
                    row.update(latency_ms=timing)
                else:
                    res = rags[name].ask(g["question"])
                    notes = [s["note_path"] for s in res.retrieved]
                    row.update(refused=res.refused, refusal_reason=res.refusal_reason,
                               model=res.model, latency_ms=res.latency_ms, tokens=res.tokens,
                               cited_notes=sorted({s.note_path for s in res.sources}))
                    answers.append({"id": g["id"], "system": name, "question": g["question"],
                                    "answer": res.answer, "reference": g["reference"]})
            except UpstreamUnavailable as e:
                notes = e.retrieved  # retrieval worked; only generation failed
                row.update(error=str(e))
            row["retrieved_notes"] = list(dict.fromkeys(notes))  # dedupe, keep rank order
            row["hit"] = hit(notes, g["expected_notes"]) if g["answerable"] else None
            per_q.append(row)
            flag = "ERR" if row.get("error") else ("hit " if row["hit"] else
                                                   "miss" if g["answerable"] else "n/a ")
            extra = "" if retrieval_only or row.get("error") else \
                f" refused={row['refused']!s:5} {row['latency_ms']['total']:>7.0f}ms"
            log(f"{g['id']} {name:7s} {flag}{extra}")
            if pause:
                time.sleep(pause)
    return per_q, answers


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate vector RAG vs keyword baseline")
    ap.add_argument("--golden", type=Path, default=HERE / "golden.jsonl")
    ap.add_argument("--out", type=Path, default=HERE / "results.json")
    ap.add_argument("--answers", type=Path, default=HERE / "answers.jsonl")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--pause", type=float, default=1.0,
                    help="seconds between requests (free-tier rate limits)")
    ap.add_argument("--resume", action="store_true",
                    help="keep completed rows from --out; re-run only errored/missing ones")
    args = ap.parse_args(argv)

    golden = load_golden(args.golden)
    done, prev_answers = {}, []
    if args.resume and args.out.exists():
        prev = json.loads(args.out.read_text())
        if prev["run"]["golden_file"] != args.golden.name or prev["run"]["k"] != args.k:
            raise SystemExit("--resume: previous results used a different golden file or k")
        done = {(r["id"], r["system"]): r for r in prev["per_question"] if not r.get("error")}
        if args.answers.exists():
            prev_answers = [json.loads(x) for x in args.answers.read_text().splitlines() if x]
        print(f"resuming: {len(done)} completed rows kept", file=sys.stderr)
    retrievers, rags, config = build(args)
    started = dt.datetime.now(dt.UTC)
    per_q, answers = run(golden, retrievers, rags, args.k, args.retrieval_only, args.pause,
                         log=lambda m: print(m, file=sys.stderr, flush=True), done=done)
    kept = {(a["id"], a["system"]) for a in answers}
    answers = [a for a in prev_answers if (a["id"], a["system"]) in done
               and (a["id"], a["system"]) not in kept] + answers

    results = {
        "run": {"started_at": started.isoformat(timespec="seconds"), "resumed": bool(done),
                "finished_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                "git_commit": git_commit(), "golden_file": args.golden.name,
                "questions": len(golden),
                "answerable": sum(g["answerable"] for g in golden),
                "unanswerable": sum(not g["answerable"] for g in golden),
                "k": args.k, "retrieval_only": args.retrieval_only, **config},
        "systems": {name: summarize([r for r in per_q if r["system"] == name], args.k,
                                    args.retrieval_only) for name in retrievers},
        "per_question": per_q,
    }
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    if answers:
        args.answers.write_text("".join(json.dumps(a) + "\n" for a in answers))
    print(json.dumps(results["systems"], indent=2))


if __name__ == "__main__":
    main()
