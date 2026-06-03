"""PostgreSQL + pgvector access helpers."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from .config import get_settings

# psycopg / pgvector are imported lazily inside the functions below so that the
# rest of the package (agent loop, tools, profile) can be imported and unit-
# tested without a database driver present.


@contextmanager
def get_conn() -> Iterator["Any"]:
    """Yield a psycopg connection with the pgvector type registered."""
    import psycopg
    from pgvector.psycopg import register_vector

    settings = get_settings()
    conn = psycopg.connect(settings.database_url)
    try:
        register_vector(conn)
        yield conn
    finally:
        conn.close()


def init_schema(schema_path: str) -> None:
    """Run the schema.sql file (idempotent)."""
    import psycopg

    with open(schema_path, "r", encoding="utf-8") as f:
        sql = f.read()
    settings = get_settings()
    # CREATE EXTENSION must run before we can register the vector type, so we
    # open a plain connection here rather than going through get_conn().
    with psycopg.connect(settings.database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()


def fetch_all(sql: str, params: tuple[Any, ...] | None = None) -> list[dict[str, Any]]:
    import psycopg

    with get_conn() as conn:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(sql, params or ())
            return cur.fetchall()


def execute(sql: str, params: tuple[Any, ...] | None = None) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
        conn.commit()
