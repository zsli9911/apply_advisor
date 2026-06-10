"""User profile and conversation state.

The profile covers education (degree, school + tier, major, GPA on any common
scale, bachelor length), test scores (IELTS / TOEFL / French CEFR, GRE / GMAT),
experience (internships, research, projects), and preferences (budget, countries,
cities, career goal). It keeps an immutable snapshot of the original profile plus
a revision log, so recommendations can be re-run after constraints change without
losing where the user started.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, asdict, field as _field
from datetime import datetime
from typing import Any, Optional

# Fields that are lists of strings and should be *extended*, not replaced,
# when updated incrementally in conversation.
_APPEND_FIELDS = {"internships", "research", "projects", "notes"}

# provenance of a field value — higher wins on conflict
FIELD_SOURCES = ("user_explicit", "document_extracted", "model_inferred", "default_assumption")
_SOURCE_PRIORITY = {"user_explicit": 3, "document_extracted": 2, "model_inferred": 1, "default_assumption": 0}
# below this confidence, a value must not drive a hard rejection
CONF_THRESHOLD = 0.6


@dataclass
class UserProfile:
    # ---- Education -------------------------------------------------------
    highest_degree: Optional[str] = None        # high_school | bachelor | master
    school: Optional[str] = None                # current/previous institution
    school_tier: Optional[str] = None           # e.g. "985", "211", "double_non", "overseas"
    major: Optional[str] = None                 # current/previous major
    gpa: Optional[float] = None                 # raw value on gpa_scale
    gpa_scale: float = 4.0                      # 4.0 | 5.0 | 100
    bachelor_years: Optional[int] = None        # 3 or 4 (a 3-year bachelor matters in Europe)

    # ---- Language & test scores --------------------------------------------------
    ielts: Optional[float] = None
    toefl: Optional[int] = None
    french_level: Optional[str] = None          # CEFR, e.g. "B2"
    gre: Optional[int] = None                   # total (260–340)
    gmat: Optional[int] = None

    # ---- Experience ------------------------------------------------------------
    internships: list[str] = _field(default_factory=list)
    research: list[str] = _field(default_factory=list)
    projects: list[str] = _field(default_factory=list)
    publications: Optional[int] = None

    # ---- Target & preferences ---------------------------------------------------
    target_field: Optional[str] = None          # desired discipline
    degree_level: Optional[str] = None          # bachelor | master | phd (target)
    preferred_language: Optional[str] = None    # english | french
    budget_eur_year: Optional[int] = None       # tuition budget per year
    total_budget_eur_year: Optional[int] = None # tuition + living budget per year
    countries: list[str] = _field(default_factory=list)
    cities: list[str] = _field(default_factory=list)
    career_goal: Optional[str] = None           # e.g. "AI research", "quant finance"
    needs_scholarship: Optional[bool] = None
    intake: Optional[str] = None                # e.g. "2027 Fall"
    graduation_year: Optional[int] = None
    graduation_status: Optional[str] = None     # in_progress | completed
    self_funded: Optional[bool] = None
    requires_public_university: Optional[bool] = None
    accepts_pre_master: Optional[bool] = None    # ok with a pathway / language-class year
    notes: list[str] = _field(default_factory=list)

    # field-level provenance: field -> {value, source, confidence, updated_at}
    field_meta: dict[str, dict[str, Any]] = _field(default_factory=dict)

    # ------------------------------------------------------------------ ops
    def update(
        self, *, source: str = "user_explicit", confidence: float = 1.0, **kwargs: Any
    ) -> dict[str, Any]:
        """Apply non-None values; record provenance; return the changes made."""
        changes: dict[str, Any] = {}
        stamp = datetime.now().isoformat(timespec="seconds")
        for k, v in kwargs.items():
            if v is None or not hasattr(self, k):
                continue
            old = getattr(self, k)
            if k in _APPEND_FIELDS and isinstance(v, list):
                added = [x for x in v if x not in old]
                if added:
                    old.extend(added)
                    changes[k] = {"appended": added}
                    self._stamp(k, list(old), source, confidence, stamp)
            elif old != v:
                prev = self.field_meta.get(k)
                if prev and _SOURCE_PRIORITY.get(source, 0) < _SOURCE_PRIORITY.get(prev.get("source"), 0):
                    continue  # don't let a lower-priority source overwrite
                setattr(self, k, v)
                changes[k] = {"from": old, "to": v}
                self._stamp(k, v, source, confidence, stamp)
        return changes

    def missing_key_fields(self) -> list[str]:
        """Key fields still needed for a solid recommendation."""
        missing: list[str] = []
        if self.gpa is None:
            missing.append("gpa")
        if self.ielts is None and self.toefl is None and self.french_level is None:
            missing.append("language_score")
        if not self.target_field:
            missing.append("target_field")
        if not self.degree_level:
            missing.append("degree_level")
        if self.budget_eur_year is None and self.total_budget_eur_year is None:
            missing.append("budget")
        if not self.countries:
            missing.append("countries")
        return missing

    def _stamp(self, field: str, value: Any, source: str, confidence: float, at: str) -> None:
        self.field_meta[field] = {
            "value": value, "source": source, "confidence": confidence, "updated_at": at,
        }

    def is_low_confidence(self, field: str, threshold: float = CONF_THRESHOLD) -> bool:
        """True if a field was inferred/assumed or recorded below the threshold."""
        m = self.field_meta.get(field)
        if not m:
            return False
        return m.get("confidence", 1.0) < threshold or m.get("source") in (
            "model_inferred", "default_assumption"
        )

    # ------------------------------------------------------------ derived
    def normalized_gpa(self) -> Optional[float]:
        """GPA converted onto a 4.0 scale; see grades.to_gpa4."""
        from .grades import to_gpa4
        return to_gpa4(self.gpa, self.gpa_scale)

    def experience_strength(self) -> int:
        """Crude 0–3 signal used by the ranking engine."""
        n = len(self.internships) + len(self.research) + len(self.projects)
        n += (self.publications or 0)
        if n >= 5:
            return 3
        if n >= 3:
            return 2
        if n >= 1:
            return 1
        return 0

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v not in (None, [], "")}

    def summary(self) -> str:
        d = self.as_dict()
        if not d:
            return "(no profile information yet)"
        if self.gpa is not None:
            d["gpa_normalized_4.0"] = self.normalized_gpa()
        return "; ".join(f"{k}={v}" for k, v in d.items())


def validate_profile(profile: UserProfile) -> list[str]:
    """Return human-readable warnings about the profile."""
    warnings: list[str] = []

    from .grades import SUPPORTED_SCALES
    scale = profile.gpa_scale or 4.0
    if scale not in SUPPORTED_SCALES:
        warnings.append(
            f"GPA scale {scale} is not supported (use 4.0, 5.0, 20 or 100); "
            "treating the GPA as 4.0-scale."
        )
    if profile.gpa is not None and not (0.0 <= profile.gpa <= scale):
        warnings.append(
            f"GPA {profile.gpa} is outside the 0–{scale} range of the declared scale."
        )
    if profile.ielts is not None and not (0.0 <= profile.ielts <= 9.0):
        warnings.append(f"IELTS {profile.ielts} is outside the valid 0–9 range.")
    if profile.toefl is not None and not (0 <= profile.toefl <= 120):
        warnings.append(f"TOEFL {profile.toefl} is outside the valid 0–120 range.")
    if profile.gre is not None and not (260 <= profile.gre <= 340):
        warnings.append(f"GRE {profile.gre} is outside the valid 260–340 range.")

    # Language / test consistency
    if profile.preferred_language == "french" and profile.french_level is None:
        warnings.append(
            "You prefer French-taught programs but no French level is set; "
            "most French-taught master's require CEFR B2 or above."
        )
    if (
        profile.preferred_language == "english"
        and profile.ielts is None
        and profile.toefl is None
    ):
        warnings.append(
            "You prefer English-taught programs but no IELTS/TOEFL score is set; "
            "most require IELTS 6.5+ or TOEFL 80+. Some waivers exist — ask me "
            "about language-waiver policies."
        )

    if profile.bachelor_years == 3:
        warnings.append(
            "You hold/are completing a 3-year bachelor — some programs require a "
            "4-year degree; I will filter on this and can look up conversion policies."
        )

    # low-confidence key fields must not silently drive recommendations
    low = [f for f in ("gpa", "ielts", "toefl", "french_level")
           if profile.is_low_confidence(f)]
    if low:
        warnings.append(
            "These key fields are inferred/low-confidence and won't be used to hard-reject "
            f"programs — please confirm: {', '.join(low)}."
        )

    if profile.budget_eur_year is not None and profile.budget_eur_year < 0:
        warnings.append("Budget cannot be negative.")
    if (
        profile.total_budget_eur_year is not None
        and profile.budget_eur_year is not None
        and profile.total_budget_eur_year < profile.budget_eur_year
    ):
        warnings.append("Total budget is lower than the tuition-only budget — check the numbers.")

    return warnings


@dataclass
class ConversationState:
    """Tracks the dialogue, the evolving profile, and provenance of facts.

    `original_profile` freezes the profile the first time it is
    populated; `revisions` records every later change so recommendations can
    be re-run and still explained against where the student started.
    """
    profile: UserProfile = _field(default_factory=UserProfile)
    original_profile: Optional[dict[str, Any]] = None
    revisions: list[dict[str, Any]] = _field(default_factory=list)

    messages: list[dict[str, Any]] = _field(default_factory=list)
    # source tracking: every fact the agent surfaces can be traced here
    sources: list[dict[str, Any]] = _field(default_factory=list)
    steps: list[str] = _field(default_factory=list)  # agent trace for explainability

    def record_profile_update(self, changes: dict[str, Any]) -> None:
        if not changes:
            return
        if self.original_profile is None:
            # freeze the very first known state (i.e. after the first update)
            self.original_profile = copy.deepcopy(self.profile.as_dict())
        else:
            self.revisions.append({"turn": len(self.messages), "changes": changes})

    # fact classification: structured/retrieved facts vs rule inference
    _FACT_TYPE = {
        "university": "verified_fact", "policy": "verified_fact",
        "checklist": "verified_fact", "plan": "verified_fact",
        "provenance": "verified_fact", "rule_check": "rule_based_inference",
    }

    def add_source(self, kind: str, ref: str, detail: str) -> None:
        self.sources.append({
            "kind": kind, "ref": ref, "detail": detail,
            "fact_type": self._FACT_TYPE.get(kind, "model_interpretation"),
        })

    def add_step(self, text: str) -> None:
        self.steps.append(text)
