# Second Brain RAG

A question-answering API over my personal markdown knowledge base (an Obsidian vault of
class notes, recruiting prep and projects). Notes are split on headings, embedded with
Google's `gemini-embedding-2`, and stored in Firestore vector search. `POST /ask`
retrieves the 5 closest chunks and has Gemini answer **only** from them, citing the note
path and heading behind every claim. When the notes don't support an answer it says
`Not enough information in my notes.` instead of guessing. A 20-question eval compares
it against a keyword-search (BM25) baseline. Every number below comes from a script in this
repo, and a demo on public template notes lets anyone rerun the pipeline end to end. It runs entirely on free tiers: Gemini API free tier,
Firestore free quota, and Render's free plan for hosting. The image is Cloud Run-ready
once billing is on.

**Live:** https://second-brain-rag-m67t.onrender.com (`/ask` needs a bearer token; see
[Deploy](#deploy))

## Architecture

```mermaid
flowchart LR
    V[Markdown notes<br/>read-only vault] --> I[Ingest<br/>parse frontmatter,<br/>split on headings]
    I -->|chunks| E[Gemini embeddings<br/>gemini-embedding-2, 1536-d]
    E -->|vectors + metadata| F[(Firestore<br/>vector index)]
    Q[POST /ask] --> QE[Embed question]
    QE --> F
    F -->|top-5 chunks| G[Gemini<br/>answer only from chunks,<br/>cite n, or refuse]
    G --> R[answer + sources +<br/>latency + tokens]
    subgraph Render free tier, Cloud Run-ready image
        Q
        QE
        G
    end
```

Plain-text version: **notes → Gemini embeddings → Firestore vector search → Gemini (grounded
answer) → FastAPI in Docker on Render.**

Design choices worth knowing:

- **Chunks never cross headings.** Each chunk keeps `note path > heading path`, so a
  citation points at a section, not just a file. About 500 tokens per chunk, with a
  50-token overlap only inside a section.
- **Idempotent ingest.** A note is re-chunked only if its content hash or the pipeline
  settings (model, dimensions, chunk sizes) change. Unchanged chunks inside an edited note
  reuse their vectors.
- **Refusal is enforced, not just prompted.** An answer is kept only if the model says the
  excerpts support it **and** it cites at least one excerpt **and** every citation refers
  to a chunk that was actually retrieved. Anything else becomes the refusal, with a
  `refusal_reason`.
- **Fallback models.** On the free tier, `gemini-3.6-flash` and `gemini-3.5-flash` allow 20
  requests/day each, so generation falls through a model chain (`GEN_MODELS`). A model that
  returns 429 is skipped for 10 minutes, and every response reports which model answered.

## Evaluation

The baseline swaps vector retrieval for **BM25 keyword search over the same chunks**. Top-5,
model, prompt and refusal rules are identical, so differences come from retrieval alone.

```bash
.venv/bin/python -m evals.run_eval --golden <golden.jsonl> --out <results.json>
.venv/bin/python -m evals.report <results.json>        # markdown table
```

Metrics: **hit rate @ 5** (did a note containing the answer come back?), **correct refusal**
on questions the notes can't answer, **false refusal** on ones they can (so a system that
refuses everything can't look good), mean/median latency by stage, and tokens per request.

### On my real notes (golden set private)

20 questions: 15 answerable, each labeled with the note(s) that contain the answer, and 5
the notes can't answer (including one out-of-scope trivia question that tests the
no-outside-knowledge rule). The answerable questions are phrased the way you'd ask without
remembering a note's wording. The questions, answers and note paths are personal, so they
stay in a private working repo. Only per-system totals are published, exported with
`evals.report --export-summary` and rendered by the same script:

```bash
.venv/bin/python -m evals.report evals/summary.json
```

| Metric | vector | keyword |
|---|---|---|
| Retrieval hit rate @ 5 (answerable) | 100% (15/15) | 87% (13/15) |
| Correct refusal (unanswerable) | 100% (5/5) | 100% (5/5) |
| False refusal (answerable) | 0% (0/15) | 7% (1/15) |
| Mean latency, total (ms) | 2,439 | 1,403 |
| Median latency, total (ms) | 2,188 | 1,115 |
| Mean latency, retrieval only (ms) | 1,318 | 7 |
| Mean tokens / request (total) | 1,581 | 1,854 |
| Errors (all models unavailable) | 0/20 | 0/20 |

20 questions (15 answerable, 5 unanswerable), `golden.jsonl`, k=5, embeddings `gemini-embedding-2`, generation `gemini-3.5-flash-lite, gemini-3.1-flash-lite`, commit `3209058`, run 2026-10-18T17:04:03+00:00.

Asking the same 15 facts in the notes' own words doesn't separate the retrievers
(`evals.report evals/summary_literal.json`):

| Metric | vector | keyword |
|---|---|---|
| Retrieval hit rate @ 5 (answerable) | 100% (15/15) | 100% (15/15) |
| Mean retrieval latency (ms) | 1,371 | 8 |

20 questions (15 answerable, 5 unanswerable), `golden_literal.jsonl`, k=5, embeddings `gemini-embedding-2`, retrieval only (no generation), commit `d2fda88`, run 2026-10-18T17:54:06+00:00.

Commit hashes in these footers refer to the private working repo.

### Reproducible demo (public template notes)

`evals/demo/golden.jsonl` runs the same pipeline on
[second-brain-template](https://github.com/dylnwng/second-brain-template)'s fictional notes,
so anyone can rerun it end to end. The corpus is 10 notes and 35 chunks, so this is a smoke
test, not a benchmark.

```bash
VAULT_PATH=path/to/second-brain-template COLLECTION_PREFIX=demo_ \
  EXCLUDE_GLOBS=CLAUDE.md,README.md,wiki/log.md .venv/bin/python -m sbrag.ingest
COLLECTION_PREFIX=demo_ .venv/bin/python -m evals.run_eval --golden evals/demo/golden.jsonl \
  --out evals/demo/results.json --answers evals/demo/answers.jsonl
.venv/bin/python -m evals.report evals/demo/results.json
```

| Metric | vector | keyword |
|---|---|---|
| Retrieval hit rate @ 5 (answerable) | 100% (7/7) | 100% (7/7) |
| Correct refusal (unanswerable) | 100% (3/3) | 100% (3/3) |
| False refusal (answerable) | 0% (0/7) | 29% (2/7) |
| Mean latency, total (ms) | 2,708 | 1,766 |
| Median latency, total (ms) | 1,904 | 900 |
| Mean latency, retrieval only (ms) | 943 | 7 |
| Mean tokens / request (total) | 646 | 704 |
| Errors (all models unavailable) | 0/10 | 0/10 |

10 questions (7 answerable, 3 unanswerable), `golden.jsonl`, k=5, embeddings `gemini-embedding-2`, generation `gemini-3.5-flash-lite, gemini-3.1-flash-lite`, commit `266b3b1`, run 2026-10-18T18:19:00+00:00.

**How to read this honestly**

- **Vector search wins only when the wording differs.** On literal questions both
  retrievers find every note. On paraphrased ones BM25 misses 2 of 15.
- **n is small.** A 2-question gap out of 15 is a weak signal, not a proven improvement.
  Generation is also nondeterministic, so refusal counts can move by a question or two
  between identical runs.
- **Vector search costs latency:** about 1.3 s per request for the query embedding plus
  Firestore kNN, versus single-digit milliseconds for in-process BM25.
- **No answer-correctness metric.** In the private run, keyword search answered one
  paraphrased question by citing a different note than the one that holds the answer. Hit
  rate and refusal rate don't catch that, and this eval has no LLM judge or human grading
  of answer content.
- **Hit rate is note-level, which is lenient.** Both keyword false refusals in the demo are
  retrieval "hits": BM25 found the right note but not the section with the answer.
- **The tables' model differs from the deployed default.** Both runs used
  `gemini-3.5-flash-lite` for every answer because the Flash models were out of daily free
  quota. The deployed chain tries Flash first.
- **What went wrong along the way.** The first full run ran into the Flash models' daily
  quota partway through, and the harness counted those generation failures as retrieval
  misses. That bug is fixed and covered by a test. The first golden set reused the notes'
  exact words and couldn't tell the systems apart (the literal table).

Generated answers go to an `answers.jsonl` next to the results for spot checks. Those files
are gitignored because they quote the notes.

## How to run

Requires Python 3.12 and a Google Cloud project with Firestore. Embeddings and generation
use an [AI Studio](https://aistudio.google.com/apikey) key by default.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
cp .env.example .env    # set VAULT_PATH, GOOGLE_CLOUD_PROJECT, GEMINI_API_KEY, ASK_TOKEN

gcloud auth application-default login
gcloud firestore databases create --location=us-central1 --type=firestore-native
gcloud firestore indexes composite create --collection-group=chunks \
  --query-scope=COLLECTION --database='(default)' \
  --field-config='field-path=embedding,vector-config={"dimension":"1536","flat":"{}"}'

.venv/bin/python -m sbrag.ingest --dry-run   # parse + chunk only: counts, no API calls
.venv/bin/python -m sbrag.ingest             # embed + store; prints counts, tokens, cost
.venv/bin/uvicorn sbrag.api:app --port 8080
.venv/bin/pytest -q                          # 79 offline tests, no network
```

For my vault, `--dry-run` reports 42 notes and 404 chunks (median 196 tokens, max 496) and
about 101k embedding tokens (estimated: the Gemini API doesn't report embedding token
counts). On the free tier, the full embed cost $0.

```bash
curl -s localhost:8080/ask -H "authorization: Bearer $ASK_TOKEN" \
  -H 'content-type: application/json' \
  -d '{"question": "What are my near-term goals?"}'
```

The response has `answer`, `sources` (note path, heading, distance), `latency_ms` (embed,
retrieve, generate, total), `tokens` (prompt, output, thinking, total), `refused`,
`refusal_reason`, and the `model` that answered.

**Privacy note:** on the Gemini API free tier, Google may use submitted content to improve
its products. Set `GENAI_BACKEND=vertex` on a project with billing to keep notes out of
that.

## Deploy

One Docker image listens on `$PORT`, so it runs on Render now and Cloud Run later.
`render.yaml` is the Render Blueprint (free plan, `ohio` region, health check on
`/healthz`). Secrets are entered in Render's dashboard, never committed:

| Secret | What it is |
|---|---|
| `GEMINI_API_KEY` | AI Studio key |
| `GOOGLE_CREDENTIALS_JSON` | key for `sbrag-reader`, a service account whose only role is `roles/datastore.viewer` (reads Firestore; writes are denied) |
| `ASK_TOKEN` | bearer token `/ask` requires; if unset, `/ask` refuses to serve |

```bash
curl -s https://second-brain-rag-m67t.onrender.com/ask \
  -H "authorization: Bearer $ASK_TOKEN" -H 'content-type: application/json' \
  -d '{"question": "What are my near-term goals?"}'
```

The free instance sleeps after 15 idle minutes, so the first request after that waits for
a cold start (Render says about a minute).

### What the free host gives up versus Cloud Run

- **Keyless identity.** Cloud Run gets short-lived tokens for its service account from the
  metadata server. Off GCP, the app needs a long-lived key in an env var. Mitigation: the
  key's account is read-only.
- **Platform auth and secrets.** Cloud Run offers IAM-authenticated callers and Secret
  Manager with audit logs. Here, auth is a hand-rolled bearer token and secrets are Render
  env vars.
- **Observability.** Cloud Logging/Monitoring (and a path to Cloud Trace) versus Render's
  basic log stream.
- **Scaling.** Cloud Run autoscales and scales to zero with fast cold starts. Render free
  is one small instance that sleeps after 15 idle minutes.
- **Locality.** Cloud Run in us-central1 would sit next to Firestore. Render's Ohio region
  adds a cross-cloud hop to every Firestore call.

### Upgrade path: Cloud Run (needs billing; not run)

```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com
gcloud iam service-accounts create sbrag-run
gcloud projects add-iam-policy-binding $PROJECT --role=roles/datastore.viewer \
  --member=serviceAccount:sbrag-run@$PROJECT.iam.gserviceaccount.com
printf %s "$GEMINI_API_KEY" | gcloud secrets create gemini-api-key --data-file=-
printf %s "$ASK_TOKEN" | gcloud secrets create ask-token --data-file=-
gcloud secrets add-iam-policy-binding gemini-api-key --role=roles/secretmanager.secretAccessor \
  --member=serviceAccount:sbrag-run@$PROJECT.iam.gserviceaccount.com   # same for ask-token
gcloud run deploy second-brain-rag --source . --region us-central1 \
  --service-account sbrag-run@$PROJECT.iam.gserviceaccount.com \
  --set-secrets GEMINI_API_KEY=gemini-api-key:latest,ASK_TOKEN=ask-token:latest \
  --set-env-vars GOOGLE_CLOUD_PROJECT=$PROJECT --allow-unauthenticated
```

There's no key file on Cloud Run: with `GOOGLE_CREDENTIALS_JSON` unset, the Firestore
client uses the service's own identity.

## Tear it down

Everything here is on free tiers, so nothing bills while it's running. These steps remove
it completely.

1. **Render:** Dashboard → `second-brain-rag` service → Settings → *Delete Web Service*,
   then Blueprints → `second-brain-rag` → *Delete Blueprint*. On GitHub, Settings →
   Applications → Render → *Uninstall* (or remove the repo from it).
2. **Service-account key:** delete it on GCP, then locally:
   ```bash
   gcloud iam service-accounts keys list --iam-account=sbrag-reader@$PROJECT.iam.gserviceaccount.com
   gcloud iam service-accounts keys delete KEY_ID --iam-account=sbrag-reader@$PROJECT.iam.gserviceaccount.com
   rm ~/.config/sbrag/sbrag-reader-key.json
   ```
3. **Google Cloud project:** this removes Firestore data, the vector index and the service
   account in one step. Projects stay recoverable for 30 days.
   ```bash
   gcloud projects delete $PROJECT
   ```
4. **Gemini API key:** delete it at https://aistudio.google.com/apikey.
5. **Optional:** delete the GitHub repo(s).

## Layout

```
sbrag/        parse.py, chunk.py (heading-aware chunking), embed.py, store.py (Firestore +
              in-memory), retrieve.py (vector + BM25), rag.py (grounding, citations,
              refusal, model fallback), api.py (FastAPI), ingest.py, config.py
evals/        run_eval.py, report.py, summary*.json (totals only), demo/ (public golden
              set + results on the template notes)
tests/        offline tests: fake Gemini client, in-memory store, fixture vault
Dockerfile, render.yaml, .env.example
```
