"""Hybrid retrieval over the normalized store.

The structured data lives in 6 normalized tables; the rule / ranking / material
engines want a single flat "program view". These functions run the joins and
assemble that view (mapping native fields back to the keys the engines expect),
plus expose provenance (sources) and document requirements for explainability.

- fetch_candidates()          -> SOFT filter for the recommendation engine
- search_universities()       -> direct hard SQL filters
- compare_programs()          -> fetch specific programs (by id or name)
- get_document_requirements() -> document requirements for a program
- get_program_sources()       -> provenance rows for a program
- search_policies()           -> hybrid similarity search over policy chunks
"""
from __future__ import annotations

from typing import Any, Optional

from .config import get_settings
from .db import get_conn
from .grades import to_gpa4
from .llm import LLMClient


def _no_db() -> bool:
    return get_settings().no_db

# Base join shared by all program-view queries.
_BASE_SELECT = """
    SELECT
        p.id                              AS id,
        p.university_id                   AS university_id,
        u.name                            AS name,          -- university name (legacy key)
        u.country                         AS country,
        u.city                            AS city,
        u.university_type                 AS university_type,
        u.official_website                AS university_website,
        u.ranking_qs                      AS ranking_qs,
        u.living_cost_min                 AS living_cost_min,
        u.living_cost_max                 AS living_cost_max,
        p.name                            AS program,        -- program name (legacy key)
        p.degree_type                     AS degree_level,   -- legacy key
        p.discipline                      AS field,          -- legacy key
        p.sub_discipline                  AS sub_discipline,
        p.teaching_language               AS language,       -- legacy key
        p.duration_months                 AS duration_months,
        p.tuition_fee                     AS tuition_eur_year,  -- legacy key (EUR)
        p.currency                        AS currency,
        p.intake                          AS intake,
        p.application_open_date           AS application_open,
        p.application_deadline            AS application_deadline,
        p.official_url                    AS official_url,
        p.application_platform            AS application_platform,
        p.program_status                  AS program_status,
        p.selectivity                     AS selectivity,
        p.career_tags                     AS career_tags,
        p.scholarship_available           AS scholarship,    -- legacy key
        p.last_verified_at                AS last_verified_at,
        a.minimum_grade                   AS min_grade_raw,
        a.grade_scale                     AS min_grade_scale,
        a.source_country                  AS min_grade_country,
        a.requirement_type                AS requirement_type,
        a.confidence                      AS requirement_confidence,
        a.recommended_grade               AS recommended_grade_raw,
        a.accepted_degrees                AS accepted_degrees,
        a.accepts_cross_discipline        AS accepts_cross_discipline,
        a.accepts_three_year_bachelor     AS accepts_three_year_bachelor,
        a.required_background             AS backgrounds_accepted,  -- legacy key
        a.gre_required                    AS gre_required,
        a.gmat_required                   AS gmat_required,
        a.interview_required              AS interview_required,
        a.work_experience_required        AS work_experience_required,
        a.portfolio_required              AS portfolio_required
    FROM programs p
    JOIN universities u          ON u.id = p.university_id
    LEFT JOIN admission_requirements a ON a.program_id = p.id
"""


def _attach_languages(cur, views: list[dict[str, Any]]) -> None:
    """Fold language_requirements rows into legacy ielts_min/toefl_min/french_level_min."""
    if not views:
        return
    ids = [v["id"] for v in views]
    cur.execute(
        """SELECT program_id, language, test_type, minimum_total, minimum_cefr,
                  waiver_available, waiver_conditions
           FROM language_requirements WHERE program_id = ANY(%s)""",
        (ids,),
    )
    by_prog: dict[int, list[dict[str, Any]]] = {}
    for row in cur.fetchall():
        by_prog.setdefault(row["program_id"], []).append(row)
    for v in views:
        v.setdefault("ielts_min", None)
        v.setdefault("toefl_min", None)
        v.setdefault("french_level_min", None)
        v["language_requirements"] = by_prog.get(v["id"], [])
        for lr in by_prog.get(v["id"], []):
            tt = (lr.get("test_type") or "").upper()
            if "IELTS" in tt:
                v["ielts_min"] = lr.get("minimum_total")
            elif "TOEFL" in tt:
                v["toefl_min"] = int(lr["minimum_total"]) if lr.get("minimum_total") else None
            if lr.get("minimum_cefr"):
                v["french_level_min"] = lr["minimum_cefr"]


def _finalize(views: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Derive the legacy convenience keys the engines read directly."""
    for v in views:
        # min_gpa on the 4.0 scale, converted from the native scale
        v["min_gpa"] = to_gpa4(v.get("min_grade_raw"), v.get("min_grade_scale"))
        v["recommended_gpa"] = to_gpa4(v.get("recommended_grade_raw"), v.get("min_grade_scale"))
        months = v.get("duration_months")
        v["duration_years"] = round(months / 12, 1) if months else None
        lo, hi = v.get("living_cost_min"), v.get("living_cost_max")
        v["living_cost_eur_year"] = int((lo + hi) / 2) if (lo and hi) else (lo or hi)
        v["website"] = v.get("official_url") or v.get("university_website")
    return views


def _load_views(cur, where: str, params: list[Any], order: str, limit: int) -> list[dict[str, Any]]:
    cur.execute(f"{_BASE_SELECT} {where} {order} LIMIT %s", tuple(params) + (limit,))
    views = [dict(r) for r in cur.fetchall()]
    _attach_languages(cur, views)
    return _finalize(views)


def _dict_cursor(conn):
    import psycopg
    return conn.cursor(row_factory=psycopg.rows.dict_row)


# --------------------------------------------------------------------------- #
# Candidate pull (soft filters) for the recommendation engine                 #
# --------------------------------------------------------------------------- #
def fetch_candidates(
    field: Optional[str] = None,
    language: Optional[str] = None,
    degree_level: Optional[str] = None,
    countries: Optional[list[str]] = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    if _no_db():
        from . import memstore
        return memstore.fetch_candidates(field, language, degree_level, countries, limit)
    clauses: list[str] = ["p.program_status = 'open'"]
    params: list[Any] = []
    if field:
        clauses.append("p.discipline ILIKE %s")
        params.append(f"%{field}%")
    if language:
        clauses.append("p.teaching_language = %s")
        params.append(language.lower())
    if degree_level:
        clauses.append("p.degree_type = %s")
        params.append(degree_level.lower())
    if countries:
        ors = " OR ".join(["u.country ILIKE %s"] * len(countries))
        clauses.append(f"({ors})")
        params.extend(f"%{c}%" for c in countries)
    where = "WHERE " + " AND ".join(clauses)
    with get_conn() as conn:
        with _dict_cursor(conn) as cur:
            return _load_views(cur, where, params, "ORDER BY p.id", limit)


# --------------------------------------------------------------------------- #
# Direct hard-filter search                                                   #
# --------------------------------------------------------------------------- #
def search_universities(
    field: Optional[str] = None,
    language: Optional[str] = None,
    degree_level: Optional[str] = None,
    country: Optional[str] = None,
    max_tuition_eur: Optional[int] = None,
    student_gpa: Optional[float] = None,
    student_ielts: Optional[float] = None,
    needs_scholarship: Optional[bool] = None,
    limit: int = 8,
) -> list[dict[str, Any]]:
    if _no_db():
        from . import memstore
        return memstore.search_universities(
            field, language, degree_level, country, max_tuition_eur,
            student_gpa, student_ielts, needs_scholarship, limit,
        )
    clauses: list[str] = ["p.program_status = 'open'"]
    params: list[Any] = []
    if field:
        clauses.append("p.discipline ILIKE %s")
        params.append(f"%{field}%")
    if language:
        clauses.append("p.teaching_language = %s")
        params.append(language.lower())
    if degree_level:
        clauses.append("p.degree_type = %s")
        params.append(degree_level.lower())
    if country:
        clauses.append("u.country ILIKE %s")
        params.append(f"%{country}%")
    if max_tuition_eur is not None:
        clauses.append("p.tuition_fee <= %s")
        params.append(max_tuition_eur)
    if needs_scholarship:
        clauses.append("p.scholarship_available = TRUE")
    # GPA/IELTS are on native scales; filter approximately in SQL then rely on
    # the rule engine for exact judgment. Only the clearly-4.0 rows filter here.
    if student_gpa is not None:
        clauses.append("(a.minimum_grade IS NULL OR a.grade_scale <> 4.0 OR a.minimum_grade <= %s)")
        params.append(student_gpa)
    where = "WHERE " + " AND ".join(clauses)
    with get_conn() as conn:
        with _dict_cursor(conn) as cur:
            rows = _load_views(cur, where, params, "ORDER BY p.tuition_fee ASC NULLS LAST, u.name", limit)
    if student_ielts is not None:
        rows = [r for r in rows if r.get("ielts_min") is None or r["ielts_min"] <= student_ielts]
    return rows


def compare_programs(identifiers: list[str]) -> list[dict[str, Any]]:
    if not identifiers:
        return []
    if _no_db():
        from . import memstore
        return memstore.compare_programs(identifiers)
    views: list[dict[str, Any]] = []
    seen: set[int] = set()
    with get_conn() as conn:
        with _dict_cursor(conn) as cur:
            for ident in identifiers:
                ident = str(ident).strip()
                if ident.isdigit():
                    where, params = "WHERE p.id = %s", [int(ident)]
                else:
                    where = "WHERE p.name ILIKE %s OR u.name ILIKE %s"
                    params = [f"%{ident}%", f"%{ident}%"]
                for v in _load_views(cur, where, params, "ORDER BY p.id", 3):
                    if v["id"] not in seen:
                        seen.add(v["id"])
                        views.append(v)
    return views


# --------------------------------------------------------------------------- #
# Provenance & documents                                            #
# --------------------------------------------------------------------------- #
def get_document_requirements(program_id: int) -> list[dict[str, Any]]:
    if _no_db():
        from . import memstore
        return memstore.get_document_requirements(program_id)
    with get_conn() as conn:
        with _dict_cursor(conn) as cur:
            cur.execute(
                """SELECT document_type, required, conditions, format_requirement,
                          translation_required, certification_required
                   FROM document_requirements WHERE program_id = %s ORDER BY id""",
                (program_id,),
            )
            return [dict(r) for r in cur.fetchall()]


def get_program_sources(program_id: int) -> list[dict[str, Any]]:
    """Return every source row tied to this program (program + admission etc.)."""
    if _no_db():
        from . import memstore
        return memstore.get_program_sources(program_id)
    with get_conn() as conn:
        with _dict_cursor(conn) as cur:
            cur.execute(
                """SELECT entity_type, source_url, source_title, retrieved_at,
                          effective_date, expiration_date, source_type,
                          authority_level, content_hash
                   FROM sources WHERE entity_id = %s ORDER BY id""",
                (program_id,),
            )
            return [dict(r) for r in cur.fetchall()]


# --------------------------------------------------------------------------- #
# Unstructured: semantic policy search                                        #
# --------------------------------------------------------------------------- #
def search_policies(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    if _no_db():
        from . import memstore
        return memstore.search_policies(query, top_k)
    from . import hybrid
    llm = LLMClient()
    q_emb = llm.embed_one(query)
    with get_conn() as conn:
        with _dict_cursor(conn) as cur:
            cur.execute(
                """
                SELECT c.id, c.source, c.title, c.section, c.heading_path,
                       c.chunk_type, c.content,
                       d.document_type, d.country, d.source_url, d.official,
                       d.effective_year,
                       1 - (c.embedding <=> %s::vector) AS similarity
                FROM policy_chunks c
                LEFT JOIN policy_documents d ON d.id = c.document_id
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
                """,
                (q_emb, q_emb, top_k * 3),
            )
            pool = [dict(r) for r in cur.fetchall()]
    return hybrid.combine(query, pool)[:top_k]
