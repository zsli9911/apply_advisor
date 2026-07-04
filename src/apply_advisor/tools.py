"""Tool definitions and dispatch for the agent.

Each tool has an OpenAI function schema and a Python implementation. The
dispatcher records provenance into the workflow state and returns a compact
JSON string that goes back to the LLM as the tool result.

Tools:
    update_profile            record/update background (+ snapshot & revision log)
    recommend_programs        rule engine + scoring + reach/match/safety tiering
    check_eligibility         per-program rule report
    search_universities       direct hard-filter search
    compare_programs          side-by-side comparison
    search_policies           hybrid policy retrieval
    generate_checklist        personalized material checklist
    build_application_plan    deadline-ordered plan
    exclude_program           record an exclusion + reason
    set_application_status    application-status tracking
    update_material_status    material-status tracking
    get_workflow_summary      profile / revisions / exclusions / progress
"""
from __future__ import annotations

import json
from typing import Any

from . import retrieval
from .materials import build_application_plan, build_checklist
from .recommend import recommend
from .profile import FIELD_SOURCES
from .workflow import (
    APPLICATION_STATUSES,
    MATERIAL_STATUSES,
    WorkflowState,
)


def _j(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


# --------------------------------------------------------------------------- #
# OpenAI tool schemas                                                         #
# --------------------------------------------------------------------------- #
_PROFILE_PROPERTIES: dict[str, Any] = {
    "highest_degree": {"type": "string", "enum": ["high_school", "bachelor", "master"]},
    "school": {"type": "string", "description": "Current/previous institution"},
    "school_tier": {"type": "string", "description": "e.g. 985, 211, double_non, overseas"},
    "major": {"type": "string", "description": "Current/previous major"},
    "gpa": {"type": "number", "description": "GPA on the declared gpa_scale"},
    "gpa_scale": {"type": "number", "enum": [4.0, 5.0, 100], "description": "Scale the GPA is on"},
    "bachelor_years": {"type": "integer", "description": "3 or 4 (length of bachelor degree)"},
    "ielts": {"type": "number"},
    "toefl": {"type": "integer"},
    "french_level": {"type": "string", "description": "CEFR level, e.g. B2"},
    "gre": {"type": "integer"},
    "gmat": {"type": "integer"},
    "internships": {"type": "array", "items": {"type": "string"}},
    "research": {"type": "array", "items": {"type": "string"}},
    "projects": {"type": "array", "items": {"type": "string"}},
    "publications": {"type": "integer"},
    "target_field": {"type": "string", "description": "Desired discipline"},
    "degree_level": {"type": "string", "enum": ["bachelor", "master", "phd"]},
    "preferred_language": {"type": "string", "enum": ["english", "french"]},
    "budget_eur_year": {"type": "integer", "description": "Tuition budget EUR/year"},
    "total_budget_eur_year": {"type": "integer", "description": "Tuition+living budget EUR/year"},
    "countries": {"type": "array", "items": {"type": "string"}},
    "cities": {"type": "array", "items": {"type": "string"}},
    "career_goal": {"type": "string", "description": "e.g. 'AI research', 'quant finance'"},
    "needs_scholarship": {"type": "boolean"},
    "intake": {"type": "string", "description": "target intake, e.g. '2027 Fall'"},
    "graduation_year": {"type": "integer"},
    "graduation_status": {"type": "string", "enum": ["in_progress", "completed"]},
    "self_funded": {"type": "boolean"},
    "requires_public_university": {"type": "boolean"},
    "accepts_pre_master": {"type": "boolean",
                           "description": "ok with a pathway / language-class year to meet requirements"},
    "source": {"type": "string", "enum": list(FIELD_SOURCES),
               "description": "provenance of these values; default user_explicit. "
                              "Use model_inferred when you inferred them rather than "
                              "the student stating them."},
    "confidence": {"type": "number", "description": "0–1 confidence in these values (default 1.0)"},
}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "update_profile",
            "description": (
                "Record or update structured facts about the student's background: "
                "education (degree, school, tier, major, GPA + scale, 3/4-year bachelor), "
                "language & test scores, experience (internships/research/projects), and "
                "preferences (field, budget, countries, cities, career goal). Call this "
                "whenever the user reveals or CHANGES such facts — changes are logged and "
                "the original profile is preserved."
            ),
            "parameters": {"type": "object", "properties": _PROFILE_PROPERTIES},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_programs",
            "description": (
                "The MAIN recommendation tool. Pulls candidate programs, runs the hard-"
                "eligibility RULE ENGINE (GPA, language, background/cross-discipline, "
                "3-year bachelor, budget, deadline), scores each eligible program on "
                "academic/budget/career/preference fit, and classifies them into "
                "reach/match/safety with explanations. Ineligible "
                "programs are returned with traceable exclusion reasons and remembered. "
                "Uses the stored profile; optional args override the search scope."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "field": {"type": "string", "description": "Override the discipline to search"},
                    "language": {"type": "string", "enum": ["english", "french"]},
                    "degree_level": {"type": "string", "enum": ["bachelor", "master", "phd"]},
                    "countries": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_eligibility",
            "description": (
                "Run the hard-condition rule engine on ONE specific program and return "
                "the per-rule verdict (PASS/WARN/FAIL with reasons). Use when the student "
                "asks 'can I get into X?' or 'why was X excluded?'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "program_identifier": {"type": "string", "description": "Program id, name, or keyword"},
                },
                "required": ["program_identifier"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_universities",
            "description": (
                "Direct structured search over the program database with explicit hard "
                "filters (field/language/level/country/tuition/GPA/IELTS). Use for factual "
                "'what programs exist under X' questions; prefer recommend_programs for "
                "personalized advice."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "field": {"type": "string"},
                    "language": {"type": "string", "enum": ["english", "french"]},
                    "degree_level": {"type": "string", "enum": ["bachelor", "master", "phd"]},
                    "country": {"type": "string"},
                    "max_tuition_eur": {"type": "integer"},
                    "student_gpa": {"type": "number"},
                    "student_ielts": {"type": "number"},
                    "needs_scholarship": {"type": "boolean"},
                    "limit": {"type": "integer", "default": 8},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_programs",
            "description": (
                "Fetch full details for specific programs to compare them side by side. "
                "Pass program ids (numbers) or names/keywords."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "identifiers": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["identifiers"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_policies",
            "description": (
                "Hybrid search (vector + keyword + metadata + source authority) over the "
                "policy knowledge base: application requirements, credential/degree "
                "conversion (3-year bachelor recognition), language waivers, visas, Campus "
                "France, scholarships, housing. Prefer passing SEVERAL focused sub-queries "
                "in `queries` (e.g. entry requirements, IELTS acceptance, grade-scale "
                "recognition) rather than one long sentence."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "single query (or use `queries`)"},
                    "queries": {"type": "array", "items": {"type": "string"},
                                "description": "multiple focused sub-queries"},
                    "document_types": {"type": "array", "items": {"type": "string"},
                                       "description": "restrict to these document types, e.g. visa_policy"},
                    "top_k": {"type": "integer", "default": 5},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_checklist",
            "description": (
                "Generate a PERSONALIZED application-material checklist for one program: "
                "required items with status inferred from the profile (missing language "
                "test, GRE, etc.), recommendation/motivation-letter notes, cross-discipline "
                "and 3-year-bachelor caveats, and country-specific steps. Also registers "
                "the program in the application tracker."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "program_identifier": {"type": "string", "description": "Program id, name, or keyword"},
                },
                "required": ["program_identifier"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "build_application_plan",
            "description": (
                "Build a deadline-ordered application timeline across programs (tracked "
                "applications by default, or pass identifiers), with backdated milestones "
                "(referees, SOP draft, language test, submission buffer)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "identifiers": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Optional program ids/names; defaults to tracked applications",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "exclude_program",
            "description": (
                "Record that the student rejects a program, WITH the reason. Excluded "
                "programs stay excluded in future recommendations until the student "
                "changes their mind."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "program_identifier": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["program_identifier", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_application_status",
            "description": (
                "Update the application-pipeline status of a tracked program. "
                f"Statuses: {', '.join(APPLICATION_STATUSES)}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "program_identifier": {"type": "string"},
                    "status": {"type": "string", "enum": APPLICATION_STATUSES},
                },
                "required": ["program_identifier", "status"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_material_status",
            "description": (
                "Update one material item's status for a tracked program (e.g. mark the "
                f"IELTS report as done). Statuses: {', '.join(MATERIAL_STATUSES)}. "
                "Item keys come from generate_checklist (e.g. 'language_test', 'cv', "
                "'motivation_letter', 'references', 'transcripts')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "program_identifier": {"type": "string"},
                    "item_key": {"type": "string"},
                    "status": {"type": "string", "enum": MATERIAL_STATUSES},
                },
                "required": ["program_identifier", "item_key", "status"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sources",
            "description": (
                "Return the official SOURCES behind a program's facts: "
                "source URL, title, retrieved/effective/expiration dates, authority "
                "level, and the grade requirement with its native scale, source "
                "country, requirement type (minimum vs recommended) and confidence. "
                "Use when the student asks 'where does this come from / is this "
                "official / how current is it'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "program_identifier": {"type": "string"},
                },
                "required": ["program_identifier"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_workflow_summary",
            "description": (
                "Return the full decision-workflow state: current profile, ORIGINAL "
                "profile snapshot, revision log, excluded programs with reasons, and the "
                "application pipeline with material statuses. Use when the student asks "
                "'where are we?' or before re-planning."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# --------------------------------------------------------------------------- #
# Implementations                                                             #
# --------------------------------------------------------------------------- #
def _resolve_program(identifier: str) -> dict[str, Any] | None:
    progs = retrieval.compare_programs([identifier])
    return progs[0] if progs else None


def _tool_update_profile(args: dict[str, Any], wf: WorkflowState) -> str:
    conv = wf.conversation
    source = args.pop("source", "user_explicit")
    confidence = args.pop("confidence", 1.0)
    changes = conv.profile.update(source=source, confidence=confidence, **args)
    conv.record_profile_update(changes)
    return _j({
        "ok": True,
        "changes": changes,
        "profile": conv.profile.as_dict(),
        "gpa_normalized_4.0": conv.profile.normalized_gpa(),
        "revisions_recorded": len(conv.revisions),
    })


def _tool_recommend_programs(args: dict[str, Any], wf: WorkflowState) -> str:
    p = wf.conversation.profile
    candidates = retrieval.fetch_candidates(
        field=args.get("field") or p.target_field,
        language=args.get("language") or p.preferred_language,
        degree_level=args.get("degree_level") or p.degree_level,
        countries=args.get("countries") or (p.countries or None),
    )
    # honour user-made exclusions across turns
    user_excluded = [
        c for c in candidates if wf.is_user_excluded(c.get("id"))
    ]
    candidates = [c for c in candidates if not wf.is_user_excluded(c.get("id"))]

    result = recommend(p, candidates)
    out = result.as_dict()
    from . import output
    out["standard_output"] = output.build_standard_output(
        p, result,
        applied_constraints={
            "countries": args.get("countries") or p.countries or None,
            "fields": [args.get("field") or p.target_field] if (args.get("field") or p.target_field) else None,
            "max_tuition": p.budget_eur_year,
            "teaching_language": args.get("language") or p.preferred_language,
        },
    )
    wf.last_recommendation = out["standard_output"]
    out["skipped_user_excluded"] = [
        {"program_id": c.get("id"),
         "program": f"{c.get('name')} — {c.get('program')}",
         "reason": wf.excluded[str(c.get("id"))]["reason"]}
        for c in user_excluded
    ]

    # refresh rule-based exclusion memory against the CURRENT profile
    wf.clear_rule_exclusions()
    wf.record_rule_exclusions(out["excluded"])

    for r in result.recommendations:
        prog = r.program
        wf.conversation.add_source(
            kind="university",
            ref=f"{prog['name']} — {prog['program']} (id {prog['id']})",
            detail=f"{r.tier} ({r.total_score:.0f} pts), {prog['country']}, "
                   f"tuition {prog.get('tuition_eur_year')} EUR/yr",
        )
    if not result.recommendations:
        out["note"] = (
            "No eligible programs after applying the hard rules. See 'excluded' for the "
            "reason each candidate failed; consider relaxing budget, language, or field."
        )
    return _j(out)


def _tool_check_eligibility(args: dict[str, Any], wf: WorkflowState) -> str:
    prog = _resolve_program(args["program_identifier"])
    if not prog:
        return _j({"error": f"No program found for '{args['program_identifier']}'."})
    from .rules import evaluate_program
    report = evaluate_program(wf.conversation.profile, prog)
    wf.conversation.add_source(
        kind="rule_check",
        ref=f"{prog['name']} — {prog['program']} (id {prog['id']})",
        detail=f"eligibility: {report.status}",
    )
    _result = {"PASS": "met", "WARN": "conditional", "FAIL": "not_met"}
    _status = {"eligible": "eligible", "conditional": "conditionally_eligible",
               "ineligible": "ineligible"}
    checks = [
        {"condition": r.rule, "result": _result[r.verdict], "reason": r.reason,
         "possible_solution": r.reason if r.verdict != "PASS" else None}
        for r in report.results
    ]
    return _j({
        "program_id": prog["id"],
        "program": f"{prog['name']} — {prog['program']}",
        "status": _status[report.status],
        "checks": checks,
    })


def _tool_search_universities(args: dict[str, Any], wf: WorkflowState) -> str:
    rows = retrieval.search_universities(**args)
    for r in rows:
        wf.conversation.add_source(
            kind="university",
            ref=f"{r['name']} — {r['program']} (id {r['id']})",
            detail=f"{r['country']}, {r['language']}, tuition {r['tuition_eur_year']} EUR/yr",
        )
    if not rows:
        # anomaly fallback: explain why nothing matched
        return _j({
            "results": [],
            "note": (
                "No programs matched all constraints. Consider relaxing budget, "
                "language, or field, or improving language scores."
            ),
        })
    return _j({"results": rows, "count": len(rows)})


def _tool_compare_programs(args: dict[str, Any], wf: WorkflowState) -> str:
    rows = retrieval.compare_programs(args.get("identifiers", []))
    for r in rows:
        wf.conversation.add_source(
            kind="university",
            ref=f"{r['name']} — {r['program']} (id {r['id']})",
            detail="comparison",
        )
    return _j({"results": rows, "count": len(rows)})


def _tool_search_policies(args: dict[str, Any], wf: WorkflowState) -> str:
    top_k = int(args.get("top_k", 5))
    queries = args.get("queries") or ([args["query"]] if args.get("query") else [])
    if not queries:
        return _j({"error": "Provide `query` or `queries`."})
    doc_types = {d.lower() for d in (args.get("document_types") or [])}

    def _run(qs: list[str], types: set[str]) -> dict[str, dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for q in qs:
            for r in retrieval.search_policies(q, top_k=top_k):
                if types and (r.get("document_type") or "").lower() not in types:
                    continue
                key = f"{r['source']}|{r.get('section')}|{r['content'][:40]}"
                if key not in merged or r.get("retrieval_score", 0) > merged[key].get("retrieval_score", 0):
                    merged[key] = r
        return merged

    # fallback ladder: filtered → drop the type filter → give up honestly
    merged = _run(queries, doc_types)
    if not merged and doc_types:
        merged = _run(queries, set())
    if not merged:
        return _j({"status": "could_not_confirm", "subqueries": queries,
                   "note": "No matching policy text found; confirm with the official source."})
    rows = sorted(merged.values(), key=lambda r: -(r.get("retrieval_score") or 0))[:top_k]

    for r in rows:
        official = " (official)" if r.get("official") else ""
        wf.conversation.add_source(
            kind="policy",
            ref=f"{r['source']} › {' › '.join(_heading_path(r))}",
            detail=f"{r.get('chunk_type', 'general')}{official}, "
                   f"score={r.get('retrieval_score')}",
        )
    slim = [
        {
            "source": r["source"],
            "heading_path": _heading_path(r),
            "chunk_type": r.get("chunk_type"),
            "document_type": r.get("document_type"),
            "country": r.get("country"),
            "official": r.get("official"),
            "effective_year": r.get("effective_year"),
            "source_url": r.get("source_url"),
            "content": r["content"],
            "retrieval_score": r.get("retrieval_score"),
            "scores": {"vector": round(float(r.get("similarity") or 0), 3),
                       "keyword": r.get("keyword_score"),
                       "metadata": r.get("metadata_match"),
                       "authority": r.get("source_authority")},
        }
        for r in rows
    ]
    return _j({"results": slim, "count": len(slim), "subqueries": queries})


def _heading_path(row: dict[str, Any]) -> list[str]:
    hp = row.get("heading_path")
    if isinstance(hp, str):
        try:
            return json.loads(hp)
        except json.JSONDecodeError:
            return [hp]
    if isinstance(hp, list):
        return hp
    return [row.get("section")] if row.get("section") else []


def _tool_generate_checklist(args: dict[str, Any], wf: WorkflowState) -> str:
    prog = _resolve_program(args["program_identifier"])
    if not prog:
        return _j({"error": f"No program found for '{args['program_identifier']}'."})
    try:
        doc_reqs = retrieval.get_document_requirements(prog["id"])
    except Exception:
        doc_reqs = None
    checklist = build_checklist(wf.conversation.profile, prog, doc_requirements=doc_reqs)
    # register in the tracker and seed material statuses
    wf.track_application(prog["id"], checklist["program"])
    wf.seed_materials(prog["id"], checklist["items"])
    # spec-shaped grouping: completed / missing / conditional / deadlines
    items = checklist["items"]
    checklist["completed"] = [i["item"] for i in items if i["status"] == "done"]
    checklist["missing"] = [i["item"] for i in items if i["status"] == "missing"]
    checklist["conditional"] = [i["item"] for i in items if not i.get("required", True)]
    checklist["deadlines"] = [{"program": checklist["program"],
                               "application_deadline": checklist.get("application_deadline")}]
    wf.conversation.add_source(
        kind="checklist",
        ref=f"{prog['name']} — {prog['program']} (id {prog['id']})",
        detail=f"{len(items)} items, {len(checklist['missing'])} missing",
    )
    return _j(checklist)


def _tool_build_application_plan(args: dict[str, Any], wf: WorkflowState) -> str:
    idents = args.get("identifiers")
    if idents:
        progs = retrieval.compare_programs(idents)
    else:
        tracked_ids = list(wf.applications.keys())
        if not tracked_ids:
            return _j({
                "error": "No tracked applications yet. Generate a checklist for target "
                         "programs first, or pass identifiers explicitly."
            })
        progs = retrieval.compare_programs(tracked_ids)
    plan = build_application_plan(progs)
    for prog in progs:
        wf.track_application(prog["id"], f"{prog['name']} — {prog['program']}")
    wf.conversation.add_source(
        kind="plan", ref=f"{len(progs)} programs", detail="application timeline"
    )
    return _j(plan)


def _tool_exclude_program(args: dict[str, Any], wf: WorkflowState) -> str:
    prog = _resolve_program(args["program_identifier"])
    if not prog:
        return _j({"error": f"No program found for '{args['program_identifier']}'."})
    label = f"{prog['name']} — {prog['program']}"
    wf.exclude(prog["id"], label, args["reason"], by="user")
    return _j({
        "ok": True,
        "excluded": label,
        "reason": args["reason"],
        "note": "This program will be skipped in future recommendations.",
    })


def _tool_set_application_status(args: dict[str, Any], wf: WorkflowState) -> str:
    prog = _resolve_program(args["program_identifier"])
    if not prog:
        return _j({"error": f"No program found for '{args['program_identifier']}'."})
    wf.track_application(prog["id"], f"{prog['name']} — {prog['program']}")
    err = wf.set_status(prog["id"], args["status"])
    if err:
        return _j({"error": err})
    return _j({"ok": True, "program": f"{prog['name']} — {prog['program']}",
               "status": args["status"]})


def _tool_update_material_status(args: dict[str, Any], wf: WorkflowState) -> str:
    prog = _resolve_program(args["program_identifier"])
    if not prog:
        return _j({"error": f"No program found for '{args['program_identifier']}'."})
    wf.track_application(prog["id"], f"{prog['name']} — {prog['program']}")
    err = wf.set_material_status(prog["id"], args["item_key"], args["status"])
    if err:
        return _j({"error": err})
    return _j({"ok": True, "program": f"{prog['name']} — {prog['program']}",
               "item": args["item_key"], "status": args["status"]})


def _tool_get_sources(args: dict[str, Any], wf: WorkflowState) -> str:
    prog = _resolve_program(args["program_identifier"])
    if not prog:
        return _j({"error": f"No program found for '{args['program_identifier']}'."})
    from . import output
    sources = retrieval.get_program_sources(prog["id"])
    for s in sources:
        s["data_status"] = output.data_status(s)
    wf.conversation.add_source(
        kind="provenance",
        ref=f"{prog['name']} — {prog['program']} (id {prog['id']})",
        detail=f"{len(sources)} official source(s)",
    )
    return _j({
        "program": f"{prog['name']} — {prog['program']}",
        "last_verified_at": str(prog.get("last_verified_at")),
        "grade_requirement": {
            "raw": prog.get("min_grade_raw"),
            "scale": prog.get("min_grade_scale"),
            "source_country": prog.get("min_grade_country"),
            "type": prog.get("requirement_type"),
            "confidence": prog.get("requirement_confidence"),
            "normalized_gpa_4.0": prog.get("min_gpa"),
        },
        "sources": sources,
    })


def _tool_get_workflow_summary(args: dict[str, Any], wf: WorkflowState) -> str:
    return _j(wf.summary())


_DISPATCH = {
    "update_profile": _tool_update_profile,
    "recommend_programs": _tool_recommend_programs,
    "check_eligibility": _tool_check_eligibility,
    "search_universities": _tool_search_universities,
    "compare_programs": _tool_compare_programs,
    "search_policies": _tool_search_policies,
    "generate_checklist": _tool_generate_checklist,
    "build_application_plan": _tool_build_application_plan,
    "exclude_program": _tool_exclude_program,
    "set_application_status": _tool_set_application_status,
    "update_material_status": _tool_update_material_status,
    "get_sources": _tool_get_sources,
    "get_workflow_summary": _tool_get_workflow_summary,
}


def execute_tool(name: str, args: dict[str, Any], wf: WorkflowState) -> str:
    fn = _DISPATCH.get(name)
    if fn is None:
        return _j({"error": f"Unknown tool '{name}'."})
    try:
        return fn(args, wf)
    except Exception as exc:  # anomaly fallback
        return _j({"error": f"Tool '{name}' failed: {exc}", "args": args})
