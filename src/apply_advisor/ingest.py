"""Ingestion pipeline: normalize structured programs into 6 tables +
chunk/embed policy docs.

Reads data/programs.json where each record embeds a `university` block and
nested admission / language / document / source blocks, and writes them into
universities, programs, admission_requirements, language_requirements,
document_requirements and sources — deduplicating universities by name.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .config import get_settings
from .db import get_conn, init_schema
from .llm import LLMClient

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
SCHEMA_PATH = REPO_ROOT / "schema.sql"

# tables recreated on every full ingest (schema evolves; ingest is a rebuild)
_STRUCTURED_TABLES = [
    "sources", "document_requirements", "language_requirements",
    "admission_requirements", "programs", "universities",
    "policy_chunks", "policy_documents",
]

UNIVERSITY_COLUMNS = [
    "name", "country", "city", "university_type", "official_website",
    "ranking_qs", "ranking_times", "language_environment",
    "living_cost_min", "living_cost_max", "last_verified_at",
]
PROGRAM_COLUMNS = [
    "university_id", "name", "degree_type", "discipline", "sub_discipline",
    "teaching_language", "duration_months", "tuition_fee", "currency", "intake",
    "application_open_date", "application_deadline", "official_url",
    "application_platform", "program_status", "selectivity", "career_tags",
    "scholarship_available", "last_verified_at",
]
ADMISSION_COLUMNS = [
    "program_id", "minimum_grade", "grade_scale", "source_country",
    "requirement_type", "confidence", "recommended_grade", "accepted_degrees",
    "accepts_cross_discipline", "accepts_three_year_bachelor",
    "required_background", "prerequisite_courses", "work_experience_required",
    "portfolio_required", "gre_required", "gmat_required", "interview_required",
]
LANGUAGE_COLUMNS = [
    "program_id", "language", "test_type", "minimum_total", "minimum_listening",
    "minimum_reading", "minimum_writing", "minimum_speaking", "minimum_cefr",
    "waiver_available", "waiver_conditions",
]
DOCUMENT_COLUMNS = [
    "program_id", "document_type", "required", "conditions",
    "format_requirement", "translation_required", "certification_required",
]
SOURCE_COLUMNS = [
    "entity_type", "entity_id", "source_url", "source_title", "retrieved_at",
    "effective_date", "expiration_date", "source_type", "authority_level",
    "content_hash",
]


def _insert(cur, table: str, columns: list[str], record: dict, returning_id: bool = False):
    placeholders = ", ".join(["%s"] * len(columns))
    cols = ", ".join(columns)
    sql = f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"
    if returning_id:
        sql += " RETURNING id"
    cur.execute(sql, tuple(record.get(c) for c in columns))
    if returning_id:
        return cur.fetchone()[0]
    return None


def load_programs(path: Path | None = None) -> dict[str, int]:
    path = path or (DATA_DIR / "programs.json")
    records = json.loads(path.read_text(encoding="utf-8"))

    counts = {k: 0 for k in ("universities", "programs", "admission",
                             "language", "document", "sources")}
    uni_ids: dict[str, int] = {}

    with get_conn() as conn:
        with conn.cursor() as cur:
            for rec in records:
                # -- university (dedup by name) --------------------------------
                uni = rec["university"]
                uname = uni["name"]
                if uname not in uni_ids:
                    uni_ids[uname] = _insert(cur, "universities", UNIVERSITY_COLUMNS,
                                             uni, returning_id=True)
                    counts["universities"] += 1
                uid = uni_ids[uname]

                # -- program ---------------------------------------------------
                prog_rec = dict(rec, university_id=uid)
                pid = _insert(cur, "programs", PROGRAM_COLUMNS, prog_rec, returning_id=True)
                counts["programs"] += 1

                # -- admission (1 row) ----------------------------------------
                adm = dict(rec.get("admission", {}), program_id=pid)
                _insert(cur, "admission_requirements", ADMISSION_COLUMNS, adm)
                counts["admission"] += 1

                # -- languages -------------------------------------------------
                for lang in rec.get("languages", []):
                    _insert(cur, "language_requirements", LANGUAGE_COLUMNS,
                            dict(lang, program_id=pid))
                    counts["language"] += 1

                # -- documents -------------------------------------------------
                for doc in rec.get("documents", []):
                    _insert(cur, "document_requirements", DOCUMENT_COLUMNS,
                            dict(doc, program_id=pid))
                    counts["document"] += 1

                # -- sources (entity_id resolves to this program) --------------
                for src in rec.get("sources", []):
                    _insert(cur, "sources", SOURCE_COLUMNS,
                            dict(src, entity_id=pid))
                    counts["sources"] += 1
        conn.commit()
    return counts


# --------------------------------------------------------------------------- #
# Unstructured data: chunk + embed                                            #
# --------------------------------------------------------------------------- #
# heading keyword -> chunk_type
_CHUNK_TYPES = [
    ("admission", "academic_requirement"), ("academic requirement", "academic_requirement"),
    ("language", "language_requirement"), ("waiver", "language_requirement"),
    ("tuition", "tuition"), ("fee", "tuition"), ("scholarship", "scholarship"),
    ("document", "application_document"), ("material", "application_document"),
    ("deadline", "deadline"), ("visa", "visa"), ("curriculum", "curriculum"),
    ("course", "curriculum"), ("career", "career"), ("housing", "housing"),
    ("conversion", "credential_conversion"), ("recognition", "credential_conversion"),
]


def _chunk_type(heading_path: list[str]) -> str:
    joined = " ".join(heading_path).lower()
    for key, ctype in _CHUNK_TYPES:
        if key in joined:
            return ctype
    return "general"


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Split an optional leading '---' YAML-ish block into (metadata, body)."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    block, body = text[3:end], text[end + 4:]
    meta: dict[str, str] = {}
    for line in block.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, body.lstrip("\n")


def chunk_markdown(text: str, max_chars: int = 900) -> list[dict]:
    """Structure-aware split: track the full heading path (#, ##, ###) and tag
    each chunk with a semantic chunk_type. Long sections split on paragraphs.

    Returns dicts: {"heading_path": [...], "chunk_type": str, "content": str}.
    Also exposes ``title`` / ``section`` (path[0] / path[-1]) for back-compat.
    """
    _, body = parse_front_matter(text)
    chunks: list[dict] = []
    stack: list[tuple[int, str]] = []      # (level, heading text)
    buf: list[str] = []

    def flush() -> None:
        content = "\n".join(buf).strip()
        buf.clear()
        if not content:
            return
        path = [h for _, h in stack] or ["Overview"]
        for piece in _split_long(content, max_chars):
            chunks.append({
                "heading_path": path,
                "chunk_type": _chunk_type(path),
                "title": path[0],
                "section": path[-1],
                "content": piece,
            })

    for line in body.splitlines():
        m = re.match(r"^(#{1,6})\s+(.+)$", line)
        if m:
            flush()
            level, heading = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
        else:
            buf.append(line)
    flush()
    return chunks


def _split_long(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out: list[str] = []
    buf = ""
    for p in paragraphs:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = f"{buf}\n\n{p}" if buf else p
        else:
            if buf:
                out.append(buf)
            buf = p
    if buf:
        out.append(buf)
    return out


_DOC_META_COLUMNS = [
    "document_key", "country", "document_type", "source_url", "official",
    "retrieved_at", "effective_year", "language", "content_hash",
]


def load_policies(policies_dir: Path | None = None) -> int:
    policies_dir = policies_dir or (DATA_DIR / "policies")
    llm = LLMClient()
    files = sorted(policies_dir.glob("*.md"))

    # (source, doc_meta, chunk_dict) tuples in ingest order
    records: list[tuple[str, dict, dict]] = []
    for fp in files:
        text = fp.read_text(encoding="utf-8")
        meta, _body = parse_front_matter(text)
        doc_meta = {
            "document_key": fp.name,
            "country": meta.get("country"),
            "document_type": meta.get("document_type"),
            "source_url": meta.get("source_url"),
            "official": (meta.get("official", "true").lower() != "false"),
            "retrieved_at": meta.get("retrieved_at"),
            "effective_year": meta.get("effective_year"),
            "language": meta.get("language", "en"),
            "content_hash": hashlib.sha256(text.encode()).hexdigest()[:16],
        }
        for chunk in chunk_markdown(text):
            records.append((fp.name, doc_meta, chunk))

    if not records:
        return 0

    embeddings: list[list[float]] = []
    batch = 64
    contents = [c["content"] for _, _, c in records]
    for i in range(0, len(contents), batch):
        embeddings.extend(llm.embed(contents[i : i + batch]))

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE policy_chunks RESTART IDENTITY;")
            cur.execute("TRUNCATE policy_documents RESTART IDENTITY CASCADE;")
            doc_ids: dict[str, int] = {}
            per_doc_index: dict[str, int] = {}
            for (source, doc_meta, chunk), emb in zip(records, embeddings):
                if source not in doc_ids:
                    ph = ", ".join(["%s"] * len(_DOC_META_COLUMNS))
                    cur.execute(
                        f"INSERT INTO policy_documents ({', '.join(_DOC_META_COLUMNS)}) "
                        f"VALUES ({ph}) RETURNING id",
                        tuple(doc_meta.get(c) for c in _DOC_META_COLUMNS),
                    )
                    doc_ids[source] = cur.fetchone()[0]
                    per_doc_index[source] = 0
                idx = per_doc_index[source]
                per_doc_index[source] += 1
                cur.execute(
                    """INSERT INTO policy_chunks
                       (document_id, source, title, section, heading_path,
                        chunk_type, chunk_index, content, embedding)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (doc_ids[source], source, chunk["title"], chunk["section"],
                     json.dumps(chunk["heading_path"], ensure_ascii=False),
                     chunk["chunk_type"], idx, chunk["content"], emb),
                )
        conn.commit()
    return len(records)


def _drop_structured_tables() -> None:
    import psycopg
    with psycopg.connect(get_settings().database_url) as conn:
        with conn.cursor() as cur:
            for tbl in _STRUCTURED_TABLES:
                cur.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE;")
        conn.commit()


def run_full_ingest() -> dict[str, int]:
    # schema evolves and tables carry FKs; a full ingest rebuilds them cleanly.
    _drop_structured_tables()
    init_schema(str(SCHEMA_PATH))
    counts = load_programs()
    counts["policy_chunks"] = load_policies()
    return counts


if __name__ == "__main__":
    stats = run_full_ingest()
    print(
        f"Ingested {stats['programs']} programs across {stats['universities']} "
        f"universities, {stats['admission']} admission rows, {stats['language']} "
        f"language rows, {stats['document']} document rows, {stats['sources']} "
        f"sources, and {stats['policy_chunks']} policy chunks."
    )
