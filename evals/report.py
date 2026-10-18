"""Render evals/results.json as a markdown table (paste into the README; never hand-edit
numbers).

    .venv/bin/python -m evals.report [evals/results.json]
    .venv/bin/python -m evals.report evals/results.json --export-summary evals/summary.json

--export-summary writes only run metadata and per-system totals: no questions, note paths
or answers. That's what can be published when the golden set itself is private.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt_rate(rate, count):
    return "n/a" if rate is None else f"{rate:.0%} ({count})"


def render(results: dict) -> str:
    run, systems = results["run"], results["systems"]
    k = run["k"]
    names = list(systems)
    lines = [
        "| Metric | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]

    def row(label, fn):
        lines.append(f"| {label} | " + " | ".join(fn(systems[n]) for n in names) + " |")

    row(f"Retrieval hit rate @ {k} (answerable)",
        lambda s: fmt_rate(s[f"hit_rate_at_{k}"], s[f"hits_at_{k}"]))
    if not run["retrieval_only"]:
        row("Correct refusal (unanswerable)",
            lambda s: fmt_rate(s["correct_refusal_rate"], s["correct_refusals"]))
        row("False refusal (answerable)",
            lambda s: fmt_rate(s["false_refusal_rate"], s["false_refusals"]))
        row("Mean latency, total (ms)", lambda s: f"{s['mean_latency_ms']['total']:,.0f}")
        row("Median latency, total (ms)", lambda s: f"{s['p50_total_latency_ms']:,.0f}")
        row("Mean latency, retrieval only (ms)",
            lambda s: f"{s['mean_latency_ms']['embed'] + s['mean_latency_ms']['retrieve']:,.0f}")
        row("Mean tokens / request (total)",
            lambda s: f"{s['mean_tokens_per_request']['total']:,.0f}")
        row("Errors (all models unavailable)", lambda s: f"{s['errors']}/{s['requests']}")
    else:
        row("Mean retrieval latency (ms)", lambda s: f"{s['mean_retrieval_latency_ms']:,.0f}")

    gen = ("retrieval only (no generation)" if run["retrieval_only"]
           else f"generation `{', '.join(run['gen_models'])}`")
    meta = (f"{run['questions']} questions ({run['answerable']} answerable, "
            f"{run['unanswerable']} unanswerable), `{run['golden_file']}`, k={k}, "
            f"embeddings `{run['embed_model']}`, {gen}, "
            f"commit `{run['git_commit']}`, run {run['started_at']}.")
    return "\n".join(lines) + "\n\n" + meta


def summary(results: dict) -> dict:
    """Totals only. Drops per_question rows (ids, retrieved/cited note paths)."""
    return {"run": results["run"], "systems": results["systems"],
            "note": "aggregate-only export; per-question rows withheld (private golden set)"}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Render eval results as a markdown table")
    ap.add_argument("path", nargs="?", type=Path, default=Path(__file__).parent / "results.json")
    ap.add_argument("--export-summary", type=Path, metavar="OUT")
    args = ap.parse_args(argv)
    results = json.loads(args.path.read_text())
    if args.export_summary:
        args.export_summary.write_text(json.dumps(summary(results), indent=2) + "\n")
    print(render(results))


if __name__ == "__main__":
    main()
