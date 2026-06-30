"""Application materials and planning: checklists + deadline-ordered plans.

- build_checklist(): a personalized material list for one program, with
  per-item status seeded from the student profile so MISSING items are
  flagged, plus recommendation-letter / motivation-letter notes.
- build_application_plan(): a cross-program timeline ordered by deadline,
  with suggested milestones counted back from each deadline.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Optional

from .profile import UserProfile

# item statuses
MISSING, TODO, IN_PROGRESS, DONE = "missing", "todo", "in_progress", "done"


def _infer_status(profile: UserProfile, program: dict[str, Any], doc_type: str) -> str:
    """Status a document item starts in, inferred from the profile."""
    if doc_type in ("language_certificate",):
        lang = (program.get("language") or "").lower()
        if lang == "english":
            return DONE if (profile.ielts is not None or profile.toefl is not None) else MISSING
        if lang == "french":
            return DONE if profile.french_level else MISSING
    if doc_type in ("gre_report",):
        return DONE if profile.gre is not None else MISSING
    return TODO


def _doc_note(profile: UserProfile, program: dict[str, Any], doc: dict[str, Any]) -> str:
    """Attach a profile-specific note to a DB document requirement."""
    dt = doc.get("document_type")
    parts = [doc.get("conditions") or ""]
    if dt == "degree_certificate" and profile.bachelor_years == 3:
        parts.append("You hold a 3-year bachelor — include the ENIC-NARIC equivalence note.")
    if dt == "motivation_letter" and _is_cross_discipline(profile, program):
        parts.append("Cross-discipline: explicitly justify the switch with projects/coursework.")
    if dt == "language_certificate" and _infer_status(profile, program, dt) == MISSING:
        lang = (program.get("language") or "").lower()
        if lang == "english":
            parts.append("No English test on file — book one or check waiver eligibility.")
        else:
            parts.append("No French certificate on file.")
    if doc.get("translation_required"):
        parts.append("Certified translation required.")
    return " ".join(s for s in parts if s).strip()


def build_checklist(
    profile: UserProfile,
    program: dict[str, Any],
    doc_requirements: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Personalized checklist: required items + status inferred from profile.

    When `doc_requirements` (rows from document_requirements) is provided, the
    checklist is built from the DATABASE material list; otherwise it falls back
    to the built-in default set. Either way, item status is seeded from the
    profile so missing items surface.
    """
    if doc_requirements:
        items = []
        for doc in doc_requirements:
            dt = doc["document_type"]
            items.append({
                "key": dt,
                "item": dt.replace("_", " ").capitalize()
                + (f" — {doc['format_requirement']}" if doc.get("format_requirement") else ""),
                "required": doc.get("required", True),
                "status": _infer_status(profile, program, dt),
                "note": _doc_note(profile, program, doc),
            })
        missing = [i["item"] for i in items if i["status"] == MISSING]
        return {
            "program_id": program.get("id"),
            "program": f"{program.get('name')} — {program.get('program')}",
            "country": program.get("country"),
            "application_deadline": program.get("application_deadline"),
            "application_platform": program.get("application_platform"),
            "estimated_annual_cost_eur": (program.get("tuition_eur_year") or 0)
            + (program.get("living_cost_eur_year") or 0),
            "source": "database",
            "items": items,
            "missing_items": missing,
        }

    return _build_checklist_default(profile, program)


def _build_checklist_default(profile: UserProfile, program: dict[str, Any]) -> dict[str, Any]:
    """Fallback checklist when no DB document requirements are available."""
    items: list[dict[str, Any]] = []

    def add(key: str, label: str, status: str = TODO, note: str = "") -> None:
        items.append({"key": key, "item": label, "status": status, "note": note})

    add("passport", "Valid passport covering the full stay")
    add(
        "transcripts",
        "Academic transcripts + diploma (certified translation if not in English/French)",
        note=(
            "3-year bachelor: attach a degree-equivalence note (see credential "
            "conversion policy)." if profile.bachelor_years == 3 else ""
        ),
    )
    add("cv", "CV / résumé (1–2 pages, education + projects + internships)")
    add(
        "motivation_letter",
        f"Motivation letter tailored to {program.get('program')} at {program.get('name')}",
        note=(
            "You are applying cross-discipline — explicitly justify the switch "
            "with projects/coursework." if _is_cross_discipline(profile, program) else ""
        ),
    )
    add(
        "references",
        "Two academic or professional reference letters",
        note="Ask referees 4–6 weeks before the deadline; brief them with your CV and SOP.",
    )

    # ---- language proof, status from profile -----------------------------
    lang = (program.get("language") or "").lower()
    if lang == "english":
        req = []
        if program.get("ielts_min"):
            req.append(f"IELTS ≥ {program['ielts_min']}")
        if program.get("toefl_min"):
            req.append(f"TOEFL iBT ≥ {program['toefl_min']}")
        label = "English proficiency proof (" + " or ".join(req or ["IELTS/TOEFL"]) + ")"
        if profile.ielts is not None or profile.toefl is not None:
            add("language_test", label, status=DONE, note="Score already on file — attach the report.")
        else:
            add(
                "language_test", label, status=MISSING,
                note="No English test on file — book one, or check waiver eligibility "
                     "(English-medium prior degree) in the policy KB.",
            )
    elif lang == "french":
        lvl = program.get("french_level_min") or "B2"
        label = f"French proficiency proof (DELF/DALF or TCF, level {lvl}+)"
        if profile.french_level:
            add("language_test", label, status=DONE, note=f"Level {profile.french_level} on file.")
        else:
            add("language_test", label, status=MISSING, note="No French certificate on file.")

    if program.get("gre_required"):
        status = DONE if profile.gre is not None else MISSING
        add("gre", "GRE score report", status=status)

    # ---- country-specific procedure --------------------------------------
    if (program.get("country") or "").lower() == "france":
        add("campus_france", "Campus France 'Études en France' online application "
                             "(if your country is connected)")
        add("visa", "Long-stay student visa (VLS-TS): proof of funds (~615 EUR/month), "
                    "accommodation proof, admission letter")
        add("visa_validation", "Validate the VLS-TS online within 3 months of arrival")

    if program.get("scholarship") and (profile.needs_scholarship or profile.budget_eur_year is not None):
        add("scholarship", "Scholarship application (Eiffel / institutional grants — "
                           "separate, usually earlier deadline)")

    missing = [i["item"] for i in items if i["status"] == MISSING]
    return {
        "program_id": program.get("id"),
        "program": f"{program.get('name')} — {program.get('program')}",
        "country": program.get("country"),
        "application_deadline": program.get("application_deadline"),
        "estimated_annual_cost_eur": (program.get("tuition_eur_year") or 0)
        + (program.get("living_cost_eur_year") or 0),
        "items": items,
        "missing_items": missing,
    }


def _is_cross_discipline(profile: UserProfile, program: dict[str, Any]) -> bool:
    accepted = (program.get("backgrounds_accepted") or "any").lower()
    if accepted in ("any", "") or not profile.major:
        return False
    major = profile.major.lower()
    return not any(
        a.strip() in major or major in a.strip() for a in accepted.split(",")
    )


# --------------------------------------------------------------------------- #
# Cross-program application plan / timeline                                   #
# --------------------------------------------------------------------------- #
_MILESTONES = [                       # (days before deadline, task)
    (70, "Ask referees for recommendation letters"),
    (56, "Draft motivation letter / SOP"),
    (42, "Book & take language test (if still missing)"),
    (28, "Collect transcripts, translations, and equivalence documents"),
    (14, "Finalize and proofread all documents"),
    (7, "Submit the application (leave buffer for platform issues)"),
]


def build_application_plan(
    programs: list[dict[str, Any]], today: Optional[date] = None
) -> dict[str, Any]:
    """Timeline across target programs, ordered by nearest deadline."""
    today = today or date.today()
    entries: list[dict[str, Any]] = []
    for prog in programs:
        raw = prog.get("application_deadline")
        deadline: Optional[date] = None
        if raw:
            try:
                deadline = date.fromisoformat(str(raw).strip())
            except ValueError:
                deadline = None
        entry: dict[str, Any] = {
            "program_id": prog.get("id"),
            "program": f"{prog.get('name')} — {prog.get('program')}",
            "deadline": raw,
        }
        if deadline is None:
            entry["note"] = "Deadline unknown/unparseable — verify on the program page."
        elif deadline < today:
            entry["note"] = "Deadline has passed for this cycle; target the next intake."
        else:
            entry["days_left"] = (deadline - today).days
            entry["milestones"] = [
                {
                    "date": (deadline - timedelta(days=d)).isoformat(),
                    "task": task,
                    "overdue": (deadline - timedelta(days=d)) < today,
                }
                for d, task in _MILESTONES
            ]
        entries.append(entry)

    def sort_key(e: dict[str, Any]) -> tuple[int, str]:
        return (0 if "days_left" in e else 1, str(e.get("deadline") or "9999"))

    entries.sort(key=sort_key)
    return {"generated_on": today.isoformat(), "programs": entries}
