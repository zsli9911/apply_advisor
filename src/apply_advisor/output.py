"""Standard recommendation output, data-freshness checks and source-conflict
resolution. Pydantic validation is used when available but never required, so
the offline setup keeps working.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Optional

try:  # pydantic is optional
    from pydantic import BaseModel
    _HAS_PYDANTIC = True
except Exception:  # pragma: no cover
    _HAS_PYDANTIC = False

_CATEGORY = {"reach": "reach", "match": "match", "safety": "safer"}
_ELIG = {"eligible": "eligible", "conditional": "conditionally_eligible",
         "ineligible": "ineligible"}


# ---
def data_status(row: dict[str, Any], today: Optional[date] = None) -> str:
    today = today or date.today()
    exp = row.get("expiration_date")
    if exp:
        try:
            if date.fromisoformat(str(exp)) < today:
                return "expired"
        except ValueError:
            return "unknown"
    return "valid"


# ---
def detect_source_conflict(field: str, candidates: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Given several sources for one field, flag a conflict and resolve by
    authority then current year, else recommend manual confirmation."""
    values = [c.get("value") for c in candidates]
    if len({str(v) for v in values}) <= 1:
        return None

    def rank(c: dict[str, Any]) -> tuple:
        official = c.get("authority_level") == "official" or c.get("official") is True
        return (1 if official else 0, str(c.get("effective_year") or ""),
                str(c.get("retrieved_at") or ""))

    best = max(candidates, key=rank)
    resolvable = all(rank(best) > rank(o) for o in candidates if o is not best)
    return {
        "status": "source_conflict",
        "field": field,
        "values": values,
        "resolved_value": best.get("value") if resolvable else None,
        "recommended_action": (
            "Use the official / current-year source." if resolvable else "Contact admissions"
        ),
    }


# ---
def _strengths_risks_missing(rec) -> tuple[list[str], list[str], list[str]]:
    strengths, risks, missing = [], [], []
    for name, score in rec.scores.items():
        why = next((r.split(": ", 1)[1] for r in rec.reasons if r.startswith(name + ":")), name)
        if score >= 70:
            strengths.append(why)
    for w in rec.report.warnings:
        (missing if _looks_missing(w.reason) else risks).append(w.reason)
    if rec.program.get("min_gpa") is None:
        missing.append("Grade requirement not_publicly_specified.")
    return strengths, risks, missing


def _looks_missing(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in ("unknown", "not confirmed", "not published",
                                "inferred", "verify", "unrecognised"))


def _next_actions(recs) -> list[str]:
    actions: list[str] = []
    joined = " ".join(r.reason for rec in recs for r in rec.report.warnings).lower()
    if "ielts" in joined or "language" in joined or "pre-master" in joined:
        actions.append("Confirm whether a pre-sessional / language pathway is accepted.")
    if "not_publicly_specified" in " ".join(_next_missing(recs)).lower() or "equivalence" in joined:
        actions.append("Contact admissions about grade/credential equivalence.")
    actions.append("Prepare official transcripts and translations.")
    return actions


def _next_missing(recs) -> list[str]:
    return [m for rec in recs for m in _strengths_risks_missing(rec)[2]]


def _prog_sources(program: dict[str, Any]) -> list[dict[str, Any]]:
    srcs = program.get("sources") or []
    if srcs:
        return [{"title": s.get("source_title"), "url": s.get("source_url")} for s in srcs[:2]]
    return [{"title": "Official program page", "url": program.get("website")}]


def build_standard_output(profile, result, applied_constraints: dict[str, Any]) -> dict[str, Any]:
    from .grades import describe
    recs_out = []
    for rec in result.recommendations:
        p = rec.program
        strengths, risks, missing = _strengths_risks_missing(rec)
        recs_out.append({
            "program_id": p.get("id"),
            "university": p.get("name"),
            "program": p.get("program"),
            "category": _CATEGORY.get(rec.tier, rec.tier),
            "match_score": round(rec.total_score),
            "eligibility": _ELIG[rec.report.status],
            "estimated_range": rec.estimated_range,
            "strengths": strengths,
            "risks": risks,
            "missing_information": missing,
            "sources": _prog_sources(p),
        })
    out = {
        "profile_summary": {
            "education": " ".join(x for x in [profile.highest_degree, profile.major] if x) or None,
            "grade": describe(profile.gpa, profile.gpa_scale) if profile.gpa is not None else None,
            "english": (f"IELTS {profile.ielts}" if profile.ielts is not None
                        else f"TOEFL {profile.toefl}" if profile.toefl is not None else None),
        },
        "applied_constraints": applied_constraints,
        "recommendations": recs_out,
        "next_actions": _next_actions(result.recommendations),
    }
    return validate_standard_output(out)


# ---
if _HAS_PYDANTIC:
    class _Rec(BaseModel):
        program_id: Any
        university: Optional[str] = None
        program: Optional[str] = None
        category: str
        match_score: float
        eligibility: str
        strengths: list[str] = []
        risks: list[str] = []
        missing_information: list[str] = []
        sources: list[dict[str, Any]] = []

    class _Standard(BaseModel):
        profile_summary: dict[str, Any]
        applied_constraints: dict[str, Any]
        recommendations: list[_Rec]
        next_actions: list[str]


def validate_standard_output(out: dict[str, Any]) -> dict[str, Any]:
    """Validate the contract when pydantic is present; on failure fall back to
    the (already rule-built) dict rather than returning an error."""
    if _HAS_PYDANTIC:
        try:
            _Standard(**out)
            out["_validated"] = True
        except Exception as exc:  # pragma: no cover
            out["_validated"] = False
            out["_validation_error"] = str(exc)
    return out
