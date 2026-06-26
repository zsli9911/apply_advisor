"""In-memory data store (NO_DB mode).

Loads data/programs.json and data/policies/*.md into memory and serves the SAME
retrieval interface as the Postgres path, so the advisor can answer questions
with only an LLM key — no database, no pgvector. Enabled by NO_DB=1 (or IN_MEMORY=1);
retrieval.py delegates here when that flag is set.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

from .grades import to_gpa4
from .ingest import chunk_markdown, parse_front_matter
from .llm import LLMClient

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


def _view_from_record(rec: dict, pid: int, uid: int) -> dict[str, Any]:
    """Flatten a programs.json record into the same 'program view' the SQL path assembles."""
    uni = rec["university"]
    adm = rec.get("admission", {})
    langs = rec.get("languages", [])
    v: dict[str, Any] = {
        "id": pid, "university_id": uid,
        "name": uni["name"], "country": uni.get("country"), "city": uni.get("city"),
        "university_type": uni.get("university_type"),
        "university_website": uni.get("official_website"),
        "ranking_qs": uni.get("ranking_qs"),
        "living_cost_min": uni.get("living_cost_min"),
        "living_cost_max": uni.get("living_cost_max"),
        "program": rec["name"], "degree_level": rec["degree_type"],
        "field": rec["discipline"], "sub_discipline": rec.get("sub_discipline"),
        "language": rec["teaching_language"], "duration_months": rec.get("duration_months"),
        "tuition_eur_year": rec.get("tuition_fee"), "currency": rec.get("currency"),
        "intake": rec.get("intake"), "application_open": rec.get("application_open_date"),
        "application_deadline": rec.get("application_deadline"),
        "official_url": rec.get("official_url"),
        "application_platform": rec.get("application_platform"),
        "program_status": rec.get("program_status", "open"),
        "selectivity": rec.get("selectivity", 3), "career_tags": rec.get("career_tags"),
        "scholarship": rec.get("scholarship_available", False),
        "last_verified_at": rec.get("last_verified_at"), "notes": rec.get("notes"),
        "min_grade_raw": adm.get("minimum_grade"), "min_grade_scale": adm.get("grade_scale"),
        "min_grade_country": adm.get("source_country"),
        "requirement_type": adm.get("requirement_type", "minimum"),
        "requirement_confidence": adm.get("confidence", "high"),
        "recommended_grade_raw": adm.get("recommended_grade"),
        "accepted_degrees": adm.get("accepted_degrees"),
        "accepts_cross_discipline": adm.get("accepts_cross_discipline"),
        "accepts_three_year_bachelor": adm.get("accepts_three_year_bachelor"),
        "backgrounds_accepted": adm.get("required_background", "any"),
        "gre_required": adm.get("gre_required", False),
        "gmat_required": adm.get("gmat_required", False),
        "interview_required": adm.get("interview_required", False),
        "work_experience_required": adm.get("work_experience_required", False),
        "portfolio_required": adm.get("portfolio_required", False),
        "ielts_min": None, "toefl_min": None, "french_level_min": None,
        "language_requirements": langs,
        "documents": rec.get("documents", []),
        "sources": rec.get("sources", []),
    }
    for lr in langs:
        tt = (lr.get("test_type") or "").upper()
        if "IELTS" in tt:
            v["ielts_min"] = lr.get("minimum_total")
        elif "TOEFL" in tt:
            v["toefl_min"] = int(lr["minimum_total"]) if lr.get("minimum_total") else None
        if lr.get("minimum_cefr"):
            v["french_level_min"] = lr["minimum_cefr"]
    v["min_gpa"] = to_gpa4(v["min_grade_raw"], v["min_grade_scale"])
    v["recommended_gpa"] = to_gpa4(v["recommended_grade_raw"], v["min_grade_scale"])
    m = v.get("duration_months")
    v["duration_years"] = round(m / 12, 1) if m else None
    lo, hi = v["living_cost_min"], v["living_cost_max"]
    v["living_cost_eur_year"] = int((lo + hi) / 2) if (lo and hi) else (lo or hi)
    v["website"] = v["official_url"] or v["university_website"]
    return v


@lru_cache(maxsize=1)
def _store() -> dict[str, Any]:
    """Build (once) the in-memory dataset: program views + embedded policy chunks."""
    records = json.loads((DATA_DIR / "programs.json").read_text(encoding="utf-8"))
    views: list[dict[str, Any]] = []
    uni_ids: dict[str, int] = {}
    for i, rec in enumerate(records, start=1):
        uname = rec["university"]["name"]
        uid = uni_ids.setdefault(uname, len(uni_ids) + 1)
        views.append(_view_from_record(rec, i, uid))

    llm = LLMClient()
    chunks: list[dict[str, Any]] = []
    for fp in sorted((DATA_DIR / "policies").glob("*.md")):
        text = fp.read_text(encoding="utf-8")
        meta, _ = parse_front_matter(text)
        for c in chunk_markdown(text):
            chunks.append({
                **c, "source": fp.name,
                "document_type": meta.get("document_type"),
                "country": meta.get("country"),
                "source_url": meta.get("source_url"),
                "official": (meta.get("official", "true").lower() != "false"),
                "effective_year": meta.get("effective_year"),
            })
    contents = [c["content"] for c in chunks]
    embs: list[list[float]] = []
    for i in range(0, len(contents), 64):
        embs.extend(llm.embed(contents[i : i + 64]))
    for c, e in zip(chunks, embs):
        c["embedding"] = e
    return {"views": views, "chunks": chunks}


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


# --------------------------------------------------------------------------- #
# Public interface (mirrors retrieval.py)                                     #
# --------------------------------------------------------------------------- #
def fetch_candidates(field=None, language=None, degree_level=None, countries=None, limit=50):
    out = []
    for v in _store()["views"]:
        if v["program_status"] != "open":
            continue
        if field and field.lower() not in (v["field"] or "").lower():
            continue
        if language and (v["language"] or "").lower() != language.lower():
            continue
        if degree_level and (v["degree_level"] or "").lower() != degree_level.lower():
            continue
        if countries and not any(c.lower() in (v["country"] or "").lower() for c in countries):
            continue
        out.append(dict(v))
        if len(out) >= limit:
            break
    return out


def search_universities(field=None, language=None, degree_level=None, country=None,
                        max_tuition_eur=None, student_gpa=None, student_ielts=None,
                        needs_scholarship=None, limit=8):
    rows = fetch_candidates(
        field=field, language=language, degree_level=degree_level,
        countries=[country] if country else None, limit=1000,
    )
    out = []
    for v in rows:
        if max_tuition_eur is not None and (v.get("tuition_eur_year") or 0) > max_tuition_eur:
            continue
        # only 4.0-scale minimums filter here; the rule engine judges the rest
        if student_gpa is not None and v.get("min_grade_scale") == 4.0 \
                and v.get("min_grade_raw") is not None and v["min_grade_raw"] > student_gpa:
            continue
        if student_ielts is not None and v.get("ielts_min") is not None \
                and v["ielts_min"] > student_ielts:
            continue
        if needs_scholarship and not v.get("scholarship"):
            continue
        out.append(v)
    out.sort(key=lambda v: (v.get("tuition_eur_year") is None, v.get("tuition_eur_year") or 0, v["name"]))
    return out[:limit]


def compare_programs(identifiers):
    if not identifiers:
        return []
    views = _store()["views"]
    picked, seen = [], set()
    for ident in identifiers:
        ident = str(ident).strip()
        for v in views:
            hit = (ident.isdigit() and v["id"] == int(ident)) or (
                not ident.isdigit() and (
                    ident.lower() in (v["program"] or "").lower()
                    or ident.lower() in (v["name"] or "").lower()
                )
            )
            if hit and v["id"] not in seen:
                seen.add(v["id"])
                picked.append(dict(v))
    return picked


def get_document_requirements(program_id):
    for v in _store()["views"]:
        if v["id"] == program_id:
            return [dict(d) for d in v.get("documents", [])]
    return []


def get_program_sources(program_id):
    for v in _store()["views"]:
        if v["id"] == program_id:
            return [dict(s) for s in v.get("sources", [])]
    return []


def search_policies(query, top_k=5):
    from . import hybrid
    chunks = _store()["chunks"]
    q = LLMClient().embed_one(query)
    pool = sorted(chunks, key=lambda c: -_cos(q, c["embedding"]))[: top_k * 3]
    rows = [{
        "id": None, "source": c["source"], "title": c["title"], "section": c["section"],
        "heading_path": c["heading_path"], "chunk_type": c["chunk_type"],
        "content": c["content"], "document_type": c.get("document_type"),
        "country": c.get("country"), "source_url": c.get("source_url"),
        "official": c.get("official"), "effective_year": c.get("effective_year"),
        "similarity": round(_cos(q, c["embedding"]), 3),
    } for c in pool]
    return hybrid.combine(query, rows)[:top_k]
