"""Rule engine for hard-eligibility judgment .

Every program is evaluated against the student's profile rule by rule; each
rule returns a structured verdict so the final decision is fully traceable —
which is what lets the system *record why a program was excluded* instead of
silently dropping it in SQL.

Verdicts:
    PASS    — requirement met
    WARN    — borderline / conditional / information missing
    FAIL    — hard requirement not met (program becomes ineligible)
"""
from __future__ import annotations

from dataclasses import dataclass, field as _field
from datetime import date
from typing import Any, Optional

from .profile import UserProfile

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"

# Margins that turn a hard miss into a "borderline" warning instead
GPA_BORDERLINE = 0.10        # on the 4.0 scale
IELTS_BORDERLINE = 0.5
TOEFL_BORDERLINE = 5

_CEFR_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2"]


@dataclass
class RuleResult:
    rule: str                # machine key, e.g. "gpa"
    verdict: str             # PASS | WARN | FAIL
    reason: str              # human-readable explanation


@dataclass
class EligibilityReport:
    program_id: Any
    program_name: str
    status: str                                  # eligible | conditional | ineligible
    results: list[RuleResult] = _field(default_factory=list)

    @property
    def failures(self) -> list[RuleResult]:
        return [r for r in self.results if r.verdict == FAIL]

    @property
    def warnings(self) -> list[RuleResult]:
        return [r for r in self.results if r.verdict == WARN]

    def exclusion_reason(self) -> str:
        return "; ".join(r.reason for r in self.failures) or ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "program": self.program_name,
            "status": self.status,
            "rules": [
                {"rule": r.rule, "verdict": r.verdict, "reason": r.reason}
                for r in self.results
            ],
        }


def _low_conf(p: UserProfile, *fields: str) -> bool:
    return any(p.is_low_confidence(f) for f in fields)


def _cefr_at_least(have: Optional[str], need: Optional[str]) -> Optional[bool]:
    if not need:
        return True
    if not have:
        return None
    try:
        return _CEFR_ORDER.index(have.upper()) >= _CEFR_ORDER.index(need.upper())
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# Individual rules — each takes (profile, program_row) and returns RuleResult #
# --------------------------------------------------------------------------- #
def rule_degree_level(p: UserProfile, prog: dict[str, Any]) -> Optional[RuleResult]:
    if not p.degree_level:
        return None
    if prog.get("degree_level") == p.degree_level:
        return RuleResult("degree_level", PASS, f"Program is a {p.degree_level} as requested.")
    return RuleResult(
        "degree_level", FAIL,
        f"Program level is {prog.get('degree_level')}, you asked for {p.degree_level}.",
    )


def rule_gpa(p: UserProfile, prog: dict[str, Any]) -> Optional[RuleResult]:
    from .grades import describe

    need = prog.get("min_gpa")            # already converted to the 4.0 scale
    if need is None:
        return RuleResult("gpa", PASS, "No explicit minimum GPA published.")

    # native-scale context for the explanation (e.g. "12/20 (≈GPA 3.0), France")
    raw, scale = prog.get("min_grade_raw"), prog.get("min_grade_scale")
    country = prog.get("min_grade_country")
    req_type = (prog.get("requirement_type") or "minimum").lower()
    confidence = (prog.get("requirement_confidence") or "high").lower()
    label = describe(raw, scale) if raw is not None else f"GPA {need}"
    if country:
        label += f", {country}"
    verb = "recommends" if req_type == "recommended" else "requires"

    have = p.normalized_gpa()
    if have is None:
        return RuleResult("gpa", WARN, f"Program {verb} {label} but your GPA is unknown.")
    if have >= need:
        return RuleResult("gpa", PASS, f"Your GPA {have} meets the {req_type} {label}.")

    gap = need - have
    # A 'recommended' grade is a guideline, not a hard gate → never a FAIL.
    if req_type == "recommended":
        return RuleResult(
            "gpa", WARN,
            f"Your GPA {have} is below the RECOMMENDED {label} (confidence: {confidence}); "
            "still worth applying, especially with strong experience.",
        )
    if gap <= GPA_BORDERLINE:
        return RuleResult(
            "gpa", WARN,
            f"Your GPA {have} is just below the minimum {label} — a long shot; strong "
            "experience or a good GRE may compensate.",
        )
    if confidence in ("low", "medium"):
        return RuleResult(
            "gpa", WARN,
            f"Your GPA {have} is below the stated {label}, but this threshold is "
            f"{confidence}-confidence — verify with admissions before ruling it out.",
        )
    # low-confidence / inferred student GPA must not hard-reject
    if _low_conf(p, "gpa", "gpa_scale"):
        return RuleResult(
            "gpa", WARN,
            f"Your GPA {have} is below {label}, but your GPA is inferred/low-confidence — "
            "confirm it before excluding this program.",
        )
    return RuleResult("gpa", FAIL, f"Your GPA {have} is below the minimum {label}.")


def _lang_fail(p: UserProfile, msg: str) -> RuleResult:
    """A language shortfall is only a hard FAIL when there is no pathway and the
    student's scores are trustworthy."""
    if p.accepts_pre_master:
        return RuleResult("language", WARN, msg + " A pre-master / language-class pathway could bridge this.")
    if _low_conf(p, "ielts", "toefl", "french_level"):
        return RuleResult("language", WARN, msg + " (Based on inferred/low-confidence scores — verify.)")
    return RuleResult("language", FAIL, msg)


def rule_language(p: UserProfile, prog: dict[str, Any]) -> Optional[RuleResult]:
    lang = (prog.get("language") or "").lower()
    if lang == "english":
        ielts_need, toefl_need = prog.get("ielts_min"), prog.get("toefl_min")
        if ielts_need is None and toefl_need is None:
            return RuleResult("language", PASS, "No English test minimum published.")
        if p.ielts is None and p.toefl is None:
            return RuleResult(
                "language", WARN,
                "English-taught program but no IELTS/TOEFL on file — you may qualify "
                "for a waiver (e.g. English-medium prior degree); check the policy KB.",
            )
        if p.ielts is not None and ielts_need is not None:
            if p.ielts >= ielts_need:
                return RuleResult("language", PASS, f"IELTS {p.ielts} ≥ required {ielts_need}.")
            if ielts_need - p.ielts <= IELTS_BORDERLINE:
                return RuleResult(
                    "language", WARN,
                    f"IELTS {p.ielts} is {ielts_need - p.ielts:.1f} below the required "
                    f"{ielts_need}; some schools offer conditional admission or accept a retake.",
                )
            return _lang_fail(p, f"IELTS {p.ielts} < required {ielts_need}.")
        if p.toefl is not None and toefl_need is not None:
            if p.toefl >= toefl_need:
                return RuleResult("language", PASS, f"TOEFL {p.toefl} ≥ required {toefl_need}.")
            if toefl_need - p.toefl <= TOEFL_BORDERLINE:
                return RuleResult(
                    "language", WARN,
                    f"TOEFL {p.toefl} is just below the required {toefl_need}.",
                )
            return _lang_fail(p, f"TOEFL {p.toefl} < required {toefl_need}.")
        # has one test, program only publishes the other
        return RuleResult(
            "language", WARN,
            "Your English test type differs from the one the program publishes; "
            "most schools accept equivalents — verify on the program page.",
        )
    if lang == "french":
        need = prog.get("french_level_min")
        ok = _cefr_at_least(p.french_level, need)
        if ok is True:
            return RuleResult("language", PASS, f"French level meets {need or 'requirement'}.")
        if ok is None:
            return RuleResult(
                "language", WARN,
                f"French-taught program requires CEFR {need or 'B2'}; your French level "
                "is unknown or unrecognised.",
            )
        return _lang_fail(p, f"French level {p.french_level} is below the required {need}.")
    return None


def rule_background(p: UserProfile, prog: dict[str, Any]) -> Optional[RuleResult]:
    """Academic-background requirement and cross-discipline acceptance."""
    accepted = (prog.get("backgrounds_accepted") or "any").lower()
    if accepted in ("any", ""):
        return RuleResult("background", PASS, "Program accepts any academic background.")
    if not p.major:
        return RuleResult(
            "background", WARN,
            f"Program expects a background in: {accepted}; your major is unknown.",
        )
    major = p.major.lower()
    accepted_list = [a.strip() for a in accepted.split(",")]
    if any(a in major or major in a for a in accepted_list):
        return RuleResult("background", PASS, f"Your major '{p.major}' matches the required background.")
    if prog.get("accepts_cross_discipline"):
        return RuleResult(
            "background", WARN,
            f"Your major '{p.major}' is outside the listed backgrounds ({accepted}) but the "
            "program accepts cross-discipline applicants — expect to justify it in your SOP.",
        )
    return RuleResult(
        "background", FAIL,
        f"Program requires a background in {accepted}; your major '{p.major}' does not "
        "match and cross-discipline applicants are not accepted.",
    )


def rule_three_year_bachelor(p: UserProfile, prog: dict[str, Any]) -> Optional[RuleResult]:
    if p.bachelor_years != 3 or p.degree_level not in (None, "master", "phd"):
        return None
    if prog.get("accepts_three_year_bachelor") is False:
        return RuleResult(
            "three_year_bachelor", FAIL,
            "Program requires a 4-year bachelor; yours is 3-year.",
        )
    if prog.get("accepts_three_year_bachelor") is None:
        return RuleResult(
            "three_year_bachelor", WARN,
            "3-year bachelor acceptance not confirmed for this program — verify with admissions.",
        )
    return RuleResult("three_year_bachelor", PASS, "Program accepts 3-year bachelor degrees.")


def rule_budget(p: UserProfile, prog: dict[str, Any]) -> Optional[RuleResult]:
    tuition = prog.get("tuition_eur_year")
    if tuition is None:
        return None
    if p.budget_eur_year is not None:
        if tuition > p.budget_eur_year:
            if prog.get("scholarship"):
                return RuleResult(
                    "budget", WARN,
                    f"Tuition {tuition} EUR/yr exceeds your {p.budget_eur_year} budget, but "
                    "scholarships are available — only viable with funding.",
                )
            return RuleResult(
                "budget", FAIL,
                f"Tuition {tuition} EUR/yr exceeds your budget of {p.budget_eur_year} EUR/yr.",
            )
    if p.total_budget_eur_year is not None:
        total = tuition + (prog.get("living_cost_eur_year") or 0)
        if total > p.total_budget_eur_year:
            return RuleResult(
                "budget", WARN,
                f"Tuition + living ≈ {total} EUR/yr exceeds your total budget "
                f"{p.total_budget_eur_year}; consider cheaper cities or scholarships.",
            )
    if p.budget_eur_year is None and p.total_budget_eur_year is None:
        return None
    return RuleResult("budget", PASS, f"Tuition {tuition} EUR/yr fits your budget.")


def rule_deadline(p: UserProfile, prog: dict[str, Any], today: Optional[date] = None) -> Optional[RuleResult]:
    raw = prog.get("application_deadline")
    if not raw:
        return None
    today = today or date.today()
    try:
        deadline = date.fromisoformat(str(raw).strip())
    except ValueError:
        return RuleResult("deadline", WARN, f"Deadline '{raw}' is not a parseable date — verify it.")
    if deadline < today:
        return RuleResult(
            "deadline", WARN,
            f"Deadline {deadline.isoformat()} has passed for this cycle — target the next intake.",
        )
    days = (deadline - today).days
    if days <= 30:
        return RuleResult("deadline", WARN, f"Deadline {deadline.isoformat()} is only {days} days away.")
    return RuleResult("deadline", PASS, f"Deadline {deadline.isoformat()} ({days} days from now).")


def rule_gre(p: UserProfile, prog: dict[str, Any]) -> Optional[RuleResult]:
    if not prog.get("gre_required"):
        return None
    if p.gre is None:
        return RuleResult("gre", WARN, "Program requires/recommends GRE and none is on file.")
    return RuleResult("gre", PASS, f"GRE {p.gre} on file.")


def rule_program_status(p: UserProfile, prog: dict[str, Any]) -> Optional[RuleResult]:
    status = (prog.get("program_status") or "open").lower()
    if status != "open":
        return RuleResult("program_status", FAIL, f"Program is '{status}' — not admitting this cycle.")
    return None


def rule_public_university(p: UserProfile, prog: dict[str, Any]) -> Optional[RuleResult]:
    if not p.requires_public_university:
        return None
    ut = (prog.get("university_type") or "").lower()
    if not ut or "public" in ut or ut == "university":
        return RuleResult("public_university", PASS, "Public institution, matches your constraint.")
    return RuleResult(
        "public_university", WARN,
        f"You asked for a public university; this is a {prog.get('university_type')}.",
    )


_ALL_RULES = [
    rule_program_status,
    rule_degree_level,
    rule_gpa,
    rule_language,
    rule_background,
    rule_three_year_bachelor,
    rule_public_university,
    rule_budget,
    rule_deadline,
    rule_gre,
]


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #
def evaluate_program(
    profile: UserProfile, program: dict[str, Any], today: Optional[date] = None
) -> EligibilityReport:
    """Run every applicable rule and aggregate a traceable verdict."""
    results: list[RuleResult] = []
    for rule in _ALL_RULES:
        if rule is rule_deadline:
            r = rule(profile, program, today)  # type: ignore[call-arg]
        else:
            r = rule(profile, program)
        if r is not None:
            results.append(r)

    if any(r.verdict == FAIL for r in results):
        status = "ineligible"
    elif any(r.verdict == WARN for r in results):
        status = "conditional"
    else:
        status = "eligible"

    return EligibilityReport(
        program_id=program.get("id"),
        program_name=f"{program.get('name')} — {program.get('program')}",
        status=status,
        results=results,
    )
