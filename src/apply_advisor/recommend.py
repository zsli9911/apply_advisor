"""Recommendation and ranking engine.

Pipeline:  candidates → rule engine (hard filters, traceable) → match scoring
→ reach/match/safety tiering → explained, ordered result.

Scores are heuristics over structured fields — deliberately simple, fully
inspectable, and explained in the output rather than hidden in the LLM.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _field
from datetime import date
from typing import Any, Optional

from .profile import UserProfile
from .rules import EligibilityReport, evaluate_program

REACH, MATCH, SAFETY = "reach", "match", "safety"   # reach / match / safety
TIER_ZH = {REACH: "冲刺", MATCH: "匹配", SAFETY: "保底"}


@dataclass
class Recommendation:
    program: dict[str, Any]
    report: EligibilityReport
    tier: str                        # reach | match | safety
    total_score: float               # 0–100
    scores: dict[str, float]         # per-dimension 0–100
    reasons: list[str] = _field(default_factory=list)
    success_note: str = ""
    estimated_range: str = "medium"  # qualitative band, NOT a fake precise probability

    def as_dict(self) -> dict[str, Any]:
        p = self.program
        return {
            "program_id": p.get("id"),
            "university": p.get("name"),
            "program": p.get("program"),
            "country": p.get("country"),
            "city": p.get("city"),
            "language": p.get("language"),
            "tuition_eur_year": p.get("tuition_eur_year"),
            "duration_years": p.get("duration_years"),
            "application_deadline": p.get("application_deadline"),
            "tier": self.tier,
            "tier_zh": TIER_ZH[self.tier],
            "total_score": round(self.total_score, 1),
            "scores": {k: round(v, 1) for k, v in self.scores.items()},
            "eligibility": self.report.status,
            "risk_level": self.tier,
            "estimated_range": self.estimated_range,
            "warnings": [r.reason for r in self.report.warnings],
            "reasons": self.reasons,
            "success_note": self.success_note,
        }


# --------------------------------------------------------------------------- #
# Scoring dimensions (each returns 0–100 and an explanation)                  #
# --------------------------------------------------------------------------- #
def _score_academic(p: UserProfile, prog: dict[str, Any]) -> tuple[float, str]:
    gpa = p.normalized_gpa()
    need = prog.get("min_gpa")
    if gpa is None or need is None:
        base = 60.0
        why = "academic fit unknown (missing GPA or program minimum) — neutral score"
    else:
        margin = gpa - need                      # -0.1 .. +1.0 typically
        base = max(0.0, min(100.0, 60 + margin * 80))
        why = f"GPA margin {margin:+.2f} vs program minimum {need}"
    bonus = p.experience_strength() * 5          # up to +15 for strong experience
    if bonus:
        why += f"; +{bonus} for internships/research/projects"
    return min(100.0, base + bonus), why


def _score_budget(p: UserProfile, prog: dict[str, Any]) -> tuple[float, str]:
    tuition = prog.get("tuition_eur_year")
    budget = p.budget_eur_year
    if tuition is None or budget is None or budget <= 0:
        return 60.0, "budget fit unknown — neutral score"
    ratio = tuition / budget
    if ratio <= 0.5:
        return 100.0, f"tuition {tuition} EUR is well under your {budget} budget"
    if ratio <= 1.0:
        return 100 - (ratio - 0.5) * 80, f"tuition {tuition} EUR fits your {budget} budget"
    if prog.get("scholarship"):
        return 40.0, f"tuition {tuition} EUR over budget but scholarships exist"
    return 10.0, f"tuition {tuition} EUR exceeds your {budget} budget"


def _score_career(p: UserProfile, prog: dict[str, Any]) -> tuple[float, str]:
    goal = (p.career_goal or "").lower()
    tags = (prog.get("career_tags") or "").lower()
    if not goal:
        return 60.0, "no career goal stated — neutral score"
    if not tags:
        return 60.0, "program career tags unknown — neutral score"
    goal_words = {w for w in goal.replace(",", " ").split() if len(w) > 2}
    tag_list = [t.strip() for t in tags.split(",")]
    hits = [t for t in tag_list if any(w in t for w in goal_words)]
    if hits:
        score = min(100.0, 60 + 20 * len(hits))
        return score, f"career goal '{p.career_goal}' matches program tags: {', '.join(hits)}"
    return 35.0, f"program tags ({tags}) don't obviously match your goal '{p.career_goal}'"


def _score_discipline(p: UserProfile, prog: dict[str, Any]) -> tuple[float, str]:
    """Field/discipline match: target field vs discipline / sub-discipline / tags."""
    want = (p.target_field or "").lower()
    disc = (prog.get("field") or "").lower()
    if not want:
        return 60.0, "no target field stated — neutral score"
    if disc and (disc in want or want in disc):
        return 90.0, f"program discipline '{disc}' matches your target field"
    hay = " ".join([disc, (prog.get("sub_discipline") or "").lower(),
                    (prog.get("career_tags") or "").lower()])
    words = {w for w in want.replace(",", " ").split() if len(w) > 2}
    hits = [w for w in words if w in hay]
    if hits:
        return min(90.0, 60 + 15 * len(hits)), f"field overlap on: {', '.join(hits)}"
    return 40.0, f"program discipline '{disc}' differs from your target '{p.target_field}'"


def _score_language(p: UserProfile, prog: dict[str, Any]) -> tuple[float, str]:
    lang = (prog.get("language") or "").lower()
    if lang == "english":
        need, have = prog.get("ielts_min"), p.ielts
        if need is None or have is None:
            return 60.0, "english test fit unknown — neutral score"
        margin = have - need
        return max(0.0, min(100.0, 70 + margin * 40)), f"IELTS margin {margin:+.1f} vs {need}"
    if lang == "french":
        from .rules import _cefr_at_least
        ok = _cefr_at_least(p.french_level, prog.get("french_level_min"))
        if ok is True:
            return 90.0, "French level meets the requirement"
        if ok is None:
            return 55.0, "French level unknown"
        return 30.0, "French level below the requirement"
    return 60.0, "language fit neutral"


def _score_location(p: UserProfile, prog: dict[str, Any]) -> tuple[float, str]:
    score, notes = 60.0, []
    country = (prog.get("country") or "").lower()
    city = (prog.get("city") or "").lower()
    if p.countries:
        if any(c.lower() in country or country in c.lower() for c in p.countries):
            score += 25
            notes.append("country matches your preference")
        else:
            score -= 35
            notes.append("country is outside your stated preferences")
    if p.cities:
        if any(c.lower() in city or city in c.lower() for c in p.cities):
            score += 15
            notes.append("city matches your preference")
    if p.preferred_language and (prog.get("language") or "").lower() == p.preferred_language:
        notes.append("taught in your preferred language")
    return max(0.0, min(100.0, score)), "; ".join(notes) or "no location preferences stated"


def _score_application(p: UserProfile, prog: dict[str, Any]) -> tuple[float, str]:
    """Application convenience: selectivity, portal simplicity, scholarship availability."""
    sel = prog.get("selectivity") or 3
    score = 60.0 + (3 - sel) * 8          # less selective → easier application
    notes = []
    plat = (prog.get("application_platform") or "").lower()
    if "portal" in plat or "studielink" in plat:
        score += 8
        notes.append("simple portal application")
    elif plat:
        notes.append(plat)
    if prog.get("scholarship"):
        notes.append("scholarship available")
    return max(0.0, min(100.0, score)), "; ".join(notes) or f"selectivity {sel}/5"


# layer-2 weights
WEIGHTS = {
    "academic": 0.30,      # academic fit
    "discipline": 0.20,    # discipline/course match
    "language": 0.15,      # language match
    "budget": 0.15,        # budget match
    "career": 0.10,        # career match
    "location": 0.05,      # location preference
    "application": 0.05,   # application convenience
}
_SCORERS = {
    "academic": _score_academic,
    "discipline": _score_discipline,
    "language": _score_language,
    "budget": _score_budget,
    "career": _score_career,
    "location": _score_location,
    "application": _score_application,
}


# --------------------------------------------------------------------------- #
# Tiering: reach / match / safety                                                   #
# --------------------------------------------------------------------------- #
# qualitative admission-likelihood band per tier (avoid fake precise probabilities)
_TIER_RANGE = {REACH: "low-medium", MATCH: "medium", SAFETY: "high"}


def _assign_tier(
    p: UserProfile, prog: dict[str, Any], report: EligibilityReport
) -> tuple[str, str, str]:
    """Classify by admission-probability signals → (tier, estimated_range, success_note)."""
    selectivity = prog.get("selectivity") or 3          # 1 (open) .. 5 (elite)
    gpa = p.normalized_gpa()
    need = prog.get("min_gpa")
    margin = (gpa - need) if (gpa is not None and need is not None) else None
    borderline = any(r.rule in ("gpa", "language") for r in report.warnings)

    if borderline or (margin is not None and margin < 0.15) or selectivity >= 5:
        return REACH, _TIER_RANGE[REACH], (
            "Admission is uncertain: you sit at or near the entry bar"
            + (f" and selectivity is high ({selectivity}/5)" if selectivity >= 4 else "")
            + ". Apply with a strong SOP and evidence of relevant experience."
        )
    if selectivity >= 4 or (margin is not None and margin < 0.4):
        return MATCH, _TIER_RANGE[MATCH], (
            "Solid chances: you clear the published requirements with a reasonable "
            f"margin against a selectivity of {selectivity}/5."
        )
    return SAFETY, _TIER_RANGE[SAFETY], (
        "High likelihood: you exceed the requirements comfortably and the program "
        f"is not highly selective ({selectivity}/5)."
    )


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class RecommendationResult:
    recommendations: list[Recommendation]
    excluded: list[EligibilityReport]        # ineligible programs + traceable reasons

    def as_dict(self) -> dict[str, Any]:
        return {
            "recommendations": [r.as_dict() for r in self.recommendations],
            "excluded": [
                {
                    "program_id": e.program_id,
                    "program": e.program_name,
                    "reasons": e.exclusion_reason(),
                }
                for e in self.excluded
            ],
        }


def recommend(
    profile: UserProfile,
    candidates: list[dict[str, Any]],
    today: Optional[date] = None,
    per_tier_limit: int = 4,
) -> RecommendationResult:
    """Evaluate, score, tier and rank candidate programs."""
    recs: list[Recommendation] = []
    excluded: list[EligibilityReport] = []

    for prog in candidates:
        report = evaluate_program(profile, prog, today=today)
        if report.status == "ineligible":
            excluded.append(report)
            continue

        dims: dict[str, float] = {}
        reasons: list[str] = []
        for name in WEIGHTS:
            score, why = _SCORERS[name](profile, prog)
            dims[name] = score
            reasons.append(f"{name}: {why}")

        total = sum(dims[k] * WEIGHTS[k] for k in WEIGHTS)
        tier, est_range, success_note = _assign_tier(profile, prog, report)
        recs.append(
            Recommendation(
                program=prog, report=report, tier=tier,
                total_score=total, scores=dims, reasons=reasons,
                success_note=success_note, estimated_range=est_range,
            )
        )

    return RecommendationResult(
        recommendations=_diversify(recs, per_tier_limit), excluded=excluded
    )


# --------------------------------------------------------------------------- #
# Layer 4: diversity re-rank                                       #
# --------------------------------------------------------------------------- #
def _cost_band(prog: dict[str, Any]) -> str:
    t = prog.get("tuition_eur_year") or 0
    return "low" if t < 5000 else "mid" if t < 15000 else "high"


def _diversify(recs: list[Recommendation], per_tier_limit: int) -> list[Recommendation]:
    """Order match→reach→safety while spreading country and cost band, so the top
    results aren't all one country/tier (aiming ~ match:reach:safety balance)."""
    by_tier: dict[str, list[Recommendation]] = {MATCH: [], REACH: [], SAFETY: []}
    for r in sorted(recs, key=lambda r: -r.total_score):
        by_tier[r.tier].append(r)

    ordered: list[Recommendation] = []
    for tier in (MATCH, REACH, SAFETY):
        picked: list[Recommendation] = []
        seen_country: dict[str, int] = {}
        seen_cost: set[str] = set()
        # first pass: prefer a fresh country + fresh cost band for variety
        for r in by_tier[tier]:
            c = (r.program.get("country") or "").lower()
            band = _cost_band(r.program)
            if seen_country.get(c, 0) >= 2 and (c and band in seen_cost):
                continue
            picked.append(r)
            seen_country[c] = seen_country.get(c, 0) + 1
            seen_cost.add(band)
            if len(picked) >= per_tier_limit:
                break
        # backfill if diversity constraints left us short of the tier limit
        if len(picked) < per_tier_limit:
            for r in by_tier[tier]:
                if r not in picked:
                    picked.append(r)
                    if len(picked) >= per_tier_limit:
                        break
        ordered.extend(picked)
    return ordered
