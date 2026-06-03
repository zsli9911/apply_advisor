"""Central configuration, loaded from environment / .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()  # read .env from the current working directory if present


def _get_bool(key: str, default: bool = False) -> bool:
    return os.getenv(key, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # OpenAI
    openai_api_key: str
    openai_base_url: str
    chat_model: str
    embedding_model: str
    embedding_dim: int

    # Database
    database_url: str

    # Agent
    max_agent_steps: int
    retrieval_top_k: int

    # Dev
    mock_llm: bool
    no_db: bool              # in-memory mode: load JSON instead of Postgres
    local_embeddings: bool   # compute embeddings locally (for chat providers w/o an embed API)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        chat_model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "1536")),
        database_url=os.getenv(
            "DATABASE_URL",
            "postgresql://advisor:advisor@localhost:5432/apply_advisor",
        ),
        max_agent_steps=int(os.getenv("MAX_AGENT_STEPS", "8")),
        retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "5")),
        mock_llm=_get_bool("MOCK_LLM", False),
        no_db=_get_bool("NO_DB", False) or _get_bool("IN_MEMORY", False),
        local_embeddings=_get_bool("LOCAL_EMBEDDINGS", False),
    )
