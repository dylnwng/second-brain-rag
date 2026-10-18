"""Settings from environment (and .env for local dev). One place, no hidden defaults elsewhere."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


@dataclass(frozen=True)
class Settings:
    vault_path: Path | None
    firestore_database: str
    collection_prefix: str
    gcp_project: str | None
    gcp_location: str
    genai_backend: str  # gemini_api | vertex
    gemini_api_key: str | None
    embed_model: str
    embed_dim: int
    embed_price_per_1m_tokens: float
    chunk_target_tokens: int
    chunk_overlap_tokens: int
    exclude_globs: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls, require_vault: bool = True) -> Settings:
        """The API server only reads Firestore, so it can run without VAULT_PATH."""
        load_dotenv()
        vault = os.environ.get("VAULT_PATH")
        if require_vault and not vault:
            raise SystemExit("VAULT_PATH is not set (see .env.example)")
        return cls(
            vault_path=Path(vault).expanduser().resolve() if vault else None,
            firestore_database=os.environ.get("FIRESTORE_DATABASE", "(default)"),
            collection_prefix=os.environ.get("COLLECTION_PREFIX", ""),
            gcp_project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
            gcp_location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
            genai_backend=os.environ.get("GENAI_BACKEND", "gemini_api"),
            gemini_api_key=os.environ.get("GEMINI_API_KEY") or None,
            embed_model=os.environ.get("EMBED_MODEL", "gemini-embedding-2"),
            embed_dim=int(os.environ.get("EMBED_DIM", "1536")),
            embed_price_per_1m_tokens=float(os.environ.get("EMBED_PRICE_PER_1M_TOKENS", "0")),
            chunk_target_tokens=int(os.environ.get("CHUNK_TARGET_TOKENS", "500")),
            chunk_overlap_tokens=int(os.environ.get("CHUNK_OVERLAP_TOKENS", "50")),
            exclude_globs=_csv(os.environ.get("EXCLUDE_GLOBS", "")),
        )
