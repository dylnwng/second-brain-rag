"""HTTP API.

    uvicorn sbrag.api:app --port 8080
    curl -s localhost:8080/ask -H "authorization: Bearer $ASK_TOKEN" \
         -H 'content-type: application/json' -d '{"question": "What are my near-term goals?"}'

/ask answers from personal notes and spends model quota, so it fails closed: it requires
`Authorization: Bearer $ASK_TOKEN`, and if ASK_TOKEN is unset it refuses to serve unless
ALLOW_NO_AUTH=1 (local dev only).
"""

from __future__ import annotations

import hmac
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from sbrag.rag import RAG, UpstreamUnavailable

# Free tier allows 20 requests/day/model on the Flash models (checked 2026-10-18), so
# fall through to Flash-Lite rather than going down for the day.
DEFAULT_GEN_MODELS = "gemini-3.6-flash,gemini-3.5-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class SourceOut(BaseModel):
    n: int
    note_path: str
    heading: str
    distance: float


class AskResponse(BaseModel):
    answer: str
    refused: bool
    refusal_reason: str | None
    sources: list[SourceOut]
    latency_ms: dict[str, float]
    tokens: dict[str, int]
    model: str | None


def build_rag() -> RAG:
    from sbrag.config import Settings
    from sbrag.embed import GenAIEmbedder, make_genai_client
    from sbrag.retrieve import VectorRetriever
    from sbrag.store import FirestoreStore

    s = Settings.from_env(require_vault=False)
    client = make_genai_client(s.genai_backend, api_key=s.gemini_api_key,
                               project=s.gcp_project, location=s.gcp_location)
    return RAG(
        retriever=VectorRetriever(FirestoreStore(s.gcp_project, s.firestore_database, s.collection_prefix),
                                  GenAIEmbedder(client, s.embed_model, s.embed_dim)),
        client=client,
        models=[m.strip() for m in os.environ.get("GEN_MODELS", DEFAULT_GEN_MODELS).split(",")
                if m.strip()],
        k=int(os.environ.get("TOP_K", "5")),
    )


def require_token(request: Request) -> None:
    token = request.app.state.ask_token
    if token is None:
        if request.app.state.allow_no_auth:
            return
        raise HTTPException(503, "server not configured: set ASK_TOKEN")
    given = request.headers.get("authorization", "")
    if not hmac.compare_digest(given.encode(), f"Bearer {token}".encode()):
        raise HTTPException(401, "missing or invalid bearer token",
                            headers={"WWW-Authenticate": "Bearer"})


_FROM_ENV = object()  # default: read ASK_TOKEN / ALLOW_NO_AUTH from the environment


def create_app(rag_factory=build_rag, ask_token=_FROM_ENV, allow_no_auth=_FROM_ENV) -> FastAPI:
    """Tests pass ask_token/allow_no_auth explicitly (None = no token configured)."""
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Resolve auth at startup, after .env is loaded: `app` is created at import time,
        # before load_dotenv runs, so reading the env there would miss local .env values.
        if ask_token is _FROM_ENV or allow_no_auth is _FROM_ENV:
            load_dotenv()
        app.state.ask_token = (os.environ.get("ASK_TOKEN") or None) if ask_token is _FROM_ENV \
            else ask_token
        app.state.allow_no_auth = (os.environ.get("ALLOW_NO_AUTH") == "1") \
            if allow_no_auth is _FROM_ENV else allow_no_auth
        app.state.rag = rag_factory()
        yield

    app = FastAPI(title="Second Brain RAG", lifespan=lifespan)

    @app.exception_handler(UpstreamUnavailable)
    def upstream_down(request: Request, exc: UpstreamUnavailable):
        return JSONResponse(status_code=503, content={"detail": str(exc)},
                            headers={"Retry-After": "30"})

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    # Sync handler: FastAPI runs it in a threadpool, so blocking SDK calls don't stall
    # the event loop.
    @app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_token)])
    def ask(body: AskRequest, request: Request):
        return request.app.state.rag.ask(body.question).to_dict()

    return app


app = create_app()
