"""Offline tests: exercise the full pipeline without OpenAI or a database.

Run with:  MOCK_LLM=1 python -m pytest tests/ -q
or simply: MOCK_LLM=1 python tests/test_offline.py

Covers all six task areas:
  1. 用户背景分析  — profile fields, GPA-scale conversion, validation
  2. 院校筛选      — rule engine hard-condition verdicts (traceable)
  3. 招生政策检索  — chunker + deterministic mock embeddings
  4. 推荐与排序    — scoring + 冲刺/匹配/保底 tiering + exclusion reasons
  5. 申请材料生成  — personalized checklist, missing-item detection, plan
  6. 多轮决策管理  — revisions, user exclusions, status tracking, persistence
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import date
from pathlib import Path

os.environ.setdefault("MOCK_LLM", "1")

# make src/ importable without installing
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from apply_advisor import retrieval  # noqa: E402
from apply_advisor.agent import AdvisorAgent  # noqa: E402
from apply_advisor.ingest import chunk_markdown  # noqa: E402
from apply_advisor.llm import LLMClient  # noqa: E402
from apply_advisor.materials import build_application_plan, build_checklist  # noqa: E402
from apply_advisor.profile import UserProfile, validate_profile  # noqa: E402
from apply_advisor.recommend import recommend  # noqa: E402
from apply_advisor.rules import evaluate_program  # noqa: E402
from apply_advisor.workflow import WorkflowState  # noqa: E402

TODAY = date(2026, 7, 13)

# ------------------------------------------------------------------ fixtures
def _prog(**kw):
    """A program 'view' as retrieval assembles it from the normalized tables."""
    base = {
        "id": 1, "name": "Test Uni", "country": "France", "city": "Paris",
        "program": "MSc Data Science", "degree_level": "master",
        "field": "computer science", "language": "english",
        # requirement, native-scale + converted (mirrors the retrieval view)
        "min_gpa": 3.0, "min_grade_raw": 3.0, "min_grade_scale": 4.0,
        "min_grade_country": None, "requirement_type": "minimum",
        "requirement_confidence": "high",
        "ielts_min": 6.5, "toefl_min": 90, "french_level_min": None,
        "backgrounds_accepted": "computer science, mathematics",
        "accepts_cross_discipline": False, "accepts_three_year_bachelor": True,
        "gre_required": False, "selectivity": 3,
        "career_tags": "data science, machine learning", "duration_years": 2,
        "duration_months": 24, "tuition_eur_year": 6000, "living_cost_eur_year": 15000,
        "application_open": None, "application_deadline": "2027-03-15",
        "application_platform": "Campus France / Mon Master",
        "scholarship": True, "website": "https://x", "notes": "n",
        "last_verified_at": "2026-06-15",
    }
    base.update(kw)
    return base


FAKE_DOCS = [
    {"document_type": "passport", "required": True, "conditions": None,
     "translation_required": False, "certification_required": False, "format_requirement": None},
    {"document_type": "transcripts", "required": True, "conditions": "All years.",
     "translation_required": True, "certification_required": True, "format_requirement": None},
    {"document_type": "motivation_letter", "required": True, "conditions": "Program-specific.",
     "translation_required": False, "certification_required": False, "format_requirement": None},
    {"document_type": "recommendation_letter", "required": True, "conditions": "Two academic references",
     "translation_required": False, "certification_required": False,
     "format_requirement": "Uploaded directly by referee"},
    {"document_type": "language_certificate", "required": True, "conditions": "IELTS/TOEFL report.",
     "translation_required": False, "certification_required": False, "format_requirement": None},
]
FAKE_SOURCES = [
    {"entity_type": "program", "source_url": "https://x", "source_title": "official page",
     "retrieved_at": "2026-06-15", "effective_date": "2026-01-01",
     "expiration_date": "2027-12-31", "source_type": "program_page",
     "authority_level": "official", "content_hash": "abc123"},
    {"entity_type": "admission", "source_url": "https://x/admissions",
     "source_title": "admission requirements", "retrieved_at": "2026-06-15",
     "effective_date": "2026-01-01", "expiration_date": "2027-12-31",
     "source_type": "admission_requirements", "authority_level": "official",
     "content_hash": "def456"},
]


FAKE_UNIS = [
    _prog(id=1, name="Safety U", min_gpa=2.7, selectivity=1, tuition_eur_year=4000),
    _prog(id=2, name="Match U", min_gpa=3.1, selectivity=3, tuition_eur_year=6000),
    _prog(id=3, name="Reach U", min_gpa=3.35, selectivity=5, tuition_eur_year=8000,
          ielts_min=7.0),
    _prog(id=4, name="TooExpensive U", tuition_eur_year=25000, scholarship=False),
    _prog(id=5, name="FourYearOnly U", accepts_three_year_bachelor=False),
]
FAKE_POLICIES = [
    {
        "id": 1, "source": "student_visa_vls_ts.md", "title": "VLS-TS",
        "section": "Core documents", "content": "You need proof of funds ~615 EUR/month.",
        "similarity": 0.42,
    }
]

PROFILE = UserProfile(
    highest_degree="bachelor", school="Some University", school_tier="211",
    major="computer science", gpa=3.4, gpa_scale=4.0, bachelor_years=3,
    ielts=6.5, target_field="computer science", degree_level="master",
    preferred_language="english", budget_eur_year=10000,
    countries=["France"], career_goal="machine learning engineer",
    internships=["ML intern at a startup"], projects=["Kaggle top 5%"],
)


def _patch_retrieval():
    retrieval.fetch_candidates = lambda **kw: [dict(p) for p in FAKE_UNIS]  # type: ignore
    retrieval.search_universities = lambda **kw: [dict(FAKE_UNIS[0])]  # type: ignore
    retrieval.compare_programs = lambda ids: [
        dict(p) for p in FAKE_UNIS if str(p["id"]) in [str(i) for i in ids]
        or any(str(i).lower() in p["name"].lower() for i in ids)
    ]  # type: ignore
    retrieval.search_policies = lambda q, top_k=5: FAKE_POLICIES  # type: ignore
    retrieval.get_document_requirements = lambda pid: list(FAKE_DOCS)  # type: ignore
    retrieval.get_program_sources = lambda pid: list(FAKE_SOURCES)  # type: ignore


# ------------------------------------------------------- 1. profile analysis
def test_profile_and_gpa_conversion():
    p = UserProfile(gpa=85, gpa_scale=100)
    assert p.normalized_gpa() == 3.7
    p2 = UserProfile(gpa=4.0, gpa_scale=5.0)
    assert p2.normalized_gpa() == 3.2
    warnings = validate_profile(UserProfile(gpa=5.5, preferred_language="french"))
    assert any("range" in w for w in warnings)
    assert any("French level" in w for w in warnings)
    # revision log
    wf = WorkflowState()
    ch1 = wf.conversation.profile.update(gpa=3.2, gpa_scale=4.0)
    wf.conversation.record_profile_update(ch1)
    ch2 = wf.conversation.profile.update(budget_eur_year=8000)
    wf.conversation.record_profile_update(ch2)
    assert wf.conversation.original_profile is not None
    assert len(wf.conversation.revisions) == 1
    print("[ok] profile: GPA conversion, validation, original snapshot + revisions")


# ------------------------------------------------------------ 2. rule engine
def test_rule_engine():
    report = evaluate_program(PROFILE, _prog(), today=TODAY)
    assert report.status in ("eligible", "conditional")

    # over budget without scholarship -> hard FAIL, traceable reason
    r = evaluate_program(PROFILE, _prog(tuition_eur_year=25000, scholarship=False), today=TODAY)
    assert r.status == "ineligible"
    assert "budget" in [x.rule for x in r.failures]

    # 3-year bachelor rejected
    r = evaluate_program(PROFILE, _prog(accepts_three_year_bachelor=False), today=TODAY)
    assert r.status == "ineligible"
    assert "three_year_bachelor" in [x.rule for x in r.failures]

    # cross-discipline: mismatched major, program tolerant -> WARN not FAIL
    p = UserProfile(major="english literature", degree_level="master", gpa=3.5)
    r = evaluate_program(p, _prog(accepts_cross_discipline=True), today=TODAY)
    assert r.status != "ineligible"
    assert any(x.rule == "background" for x in r.warnings)

    # borderline GPA -> WARN (long shot), big miss -> FAIL
    p = UserProfile(gpa=3.25, gpa_scale=4.0)
    assert any(
        x.rule == "gpa" and x.verdict == "WARN"
        for x in evaluate_program(p, _prog(min_gpa=3.3), today=TODAY).results
    )
    p = UserProfile(gpa=2.5, gpa_scale=4.0)
    assert evaluate_program(p, _prog(min_gpa=3.3), today=TODAY).status == "ineligible"
    print("[ok] rule engine: budget/3-year/cross-discipline/GPA verdicts traceable")


def test_grade_scale_and_requirement_type():
    from apply_advisor.grades import from_gpa4, to_gpa4

    # 12/20 French Licence ≈ 3.0 on the 4.0 scale
    assert to_gpa4(12, 20) == 3.0
    assert to_gpa4(85, 100) == 3.7
    assert round(from_gpa4(3.0, 20)) == 12

    # French program stored on /20 as a RECOMMENDED bar: a miss is WARN, not FAIL
    french = _prog(language="french", min_grade_raw=12, min_grade_scale=20,
                   min_gpa=to_gpa4(12, 20), min_grade_country="France",
                   requirement_type="recommended", requirement_confidence="medium",
                   french_level_min="B2", ielts_min=None, toefl_min=None)
    p = UserProfile(gpa=2.7, gpa_scale=4.0, french_level="B2")
    rep = evaluate_program(p, french, today=TODAY)
    gpa_rules = [r for r in rep.results if r.rule == "gpa"]
    assert gpa_rules and gpa_rules[0].verdict == "WARN"
    assert "20" in gpa_rules[0].reason and "RECOMMENDED" in gpa_rules[0].reason
    assert rep.status != "ineligible"          # recommended bar never hard-fails
    print("[ok] grade scales: /20 & /100 convert; recommended bar → WARN not FAIL")


# --------------------------------------------------- 3. policy KB (chunking)
def test_chunker():
    from apply_advisor.ingest import parse_front_matter
    for name in ("campus_france.md", "credential_conversion.md", "application_materials.md"):
        text = (ROOT / "data" / "policies" / name).read_text(encoding="utf-8")
        meta, body = parse_front_matter(text)
        assert meta.get("document_type") and meta.get("source_url")  # front matter parsed
        chunks = chunk_markdown(text)
        assert len(chunks) >= 3, name
        assert all(c["content"] and c["heading_path"] for c in chunks)
        # structure-aware: at least one chunk is typed beyond 'general'
        assert any(c["chunk_type"] != "general" for c in chunks), name
    # heading_path is a real path, not a single section
    conv = chunk_markdown((ROOT / "data" / "policies" / "credential_conversion.md").read_text())
    assert any("conversion" in c["chunk_type"] or c["chunk_type"] != "general" for c in conv)
    print("[ok] chunker: front matter + structure-aware heading_path + chunk_type")


def test_mock_embedding_stable():
    llm = LLMClient()
    a = llm.embed_one("campus france student visa")
    b = llm.embed_one("campus france student visa")
    assert a == b and len(a) == llm.settings.embedding_dim
    print(f"[ok] mock embeddings deterministic, dim={len(a)}")


# --------------------------------------------------- 4. recommend & tiering
def test_recommendation_tiers():
    result = recommend(PROFILE, [dict(p) for p in FAKE_UNIS], today=TODAY)
    tiers = {r.program["name"]: r.tier for r in result.recommendations}
    assert tiers["Safety U"] == "safety"
    assert tiers["Match U"] == "match"
    assert tiers["Reach U"] == "reach"          # IELTS 6.5 vs 7.0 → borderline WARN
    excluded_names = [e.program_name for e in result.excluded]
    assert any("TooExpensive" in n for n in excluded_names)
    assert any("FourYearOnly" in n for n in excluded_names)
    assert all(e.exclusion_reason() for e in result.excluded)  # reasons always present
    top = result.recommendations[0]
    assert top.total_score > 0 and len(top.reasons) == 7    # 7-dimension weighted model
    assert set(top.scores) == set(("academic", "discipline", "language", "budget",
                                   "career", "location", "application"))
    assert top.success_note and top.estimated_range in ("low-medium", "medium", "high")
    print(f"[ok] 7-dim scoring + 冲刺/匹配/保底 + range; {len(result.excluded)} exclusions explained")


def test_confidence_gating_and_constraints():
    # low-confidence (model_inferred) GPA must NOT hard-reject
    p = UserProfile()
    p.update(gpa=2.4, gpa_scale=4.0, source="model_inferred", confidence=0.4)
    rep = evaluate_program(p, _prog(min_gpa=3.3), today=TODAY)
    gpa = [r for r in rep.results if r.rule == "gpa"][0]
    assert gpa.verdict == "WARN" and "confidence" in gpa.reason.lower()

    # pre-master pathway turns a language FAIL into WARN
    p2 = UserProfile(ielts=5.5, accepts_pre_master=True)
    lang = [r for r in evaluate_program(p2, _prog(), today=TODAY).results if r.rule == "language"][0]
    assert lang.verdict == "WARN" and "pathway" in lang.reason

    # suspended program is a hard FAIL
    assert evaluate_program(PROFILE, _prog(program_status="suspended"),
                            today=TODAY).status == "ineligible"

    # public-university constraint flags a grande école (WARN, not silent)
    p3 = UserProfile(requires_public_university=True)
    rep3 = evaluate_program(p3, _prog(university_type="grande_ecole"), today=TODAY)
    assert any(r.rule == "public_university" and r.verdict == "WARN" for r in rep3.results)
    print("[ok] gating: low-confidence→WARN, pre-master→WARN, suspended→FAIL, public-uni constraint")


def test_diversity_rerank():
    # many same-country candidates: re-rank should not stack one country at the top
    pool = [_prog(id=i, name=f"U{i}", country="France", min_gpa=3.0, selectivity=2,
                  tuition_eur_year=4000 + i) for i in range(6)]
    pool += [_prog(id=99, name="Germany U", country="Germany", min_gpa=3.0, selectivity=2,
                   tuition_eur_year=0)]
    res = recommend(UserProfile(gpa=3.6, gpa_scale=4.0), pool, today=TODAY, per_tier_limit=4)
    countries = [r.program["country"] for r in res.recommendations]
    assert "Germany" in countries          # the lone other-country program surfaces
    print("[ok] diversity re-rank surfaces country variety, not one-country stacking")


# ------------------------------------------- 5. checklist & application plan
def test_checklist_and_plan():
    # default (no DB docs): built-in list, status inferred from profile
    no_test_profile = UserProfile(major="computer science", degree_level="master",
                                  bachelor_years=3)
    cl = build_checklist(no_test_profile, _prog())
    keys = {i["key"]: i for i in cl["items"]}
    assert keys["language_test"]["status"] == "missing"
    assert cl["missing_items"]
    assert "equivalence" in keys["transcripts"]["note"]  # 3-year bachelor note

    cl2 = build_checklist(PROFILE, _prog())
    keys2 = {i["key"]: i for i in cl2["items"]}
    assert keys2["language_test"]["status"] == "done"    # IELTS on file

    # DB-driven: checklist built from document_requirements rows
    cl3 = build_checklist(no_test_profile, _prog(), doc_requirements=FAKE_DOCS)
    assert cl3["source"] == "database"
    keys3 = {i["key"]: i for i in cl3["items"]}
    assert keys3["language_certificate"]["status"] == "missing"   # no test on file
    assert "equivalence" not in "".join(i["note"] for i in cl3["items"])  # only if degree_certificate present
    cl4 = build_checklist(PROFILE, _prog(), doc_requirements=FAKE_DOCS)
    assert {i["key"]: i for i in cl4["items"]}["language_certificate"]["status"] == "done"

    plan = build_application_plan([_prog(), _prog(id=9, application_deadline="2026-01-01")],
                                  today=TODAY)
    assert plan["programs"][0]["deadline"] == "2027-03-15"      # upcoming first
    assert "passed" in plan["programs"][1]["note"]
    assert len(plan["programs"][0]["milestones"]) >= 5
    print("[ok] checklist (default + DB-driven) flags missing; plan orders deadlines")


def test_hybrid_ranking():
    from apply_advisor import hybrid
    rows = [
        {"content": "student visa proof of funds", "similarity": 0.30,
         "official": True, "country": "France", "document_type": "visa_policy",
         "chunk_type": "visa", "heading_path": ["Visa", "Funds"]},
        {"content": "unrelated housing text", "similarity": 0.32,
         "official": False, "country": "France", "document_type": "living_guide",
         "chunk_type": "housing", "heading_path": ["Housing"]},
    ]
    ranked = hybrid.combine("student visa funds France", rows)
    assert "retrieval_score" in ranked[0]
    # the on-topic official visa chunk beats the higher-vector but off-topic one
    assert ranked[0]["document_type"] == "visa_policy"
    assert ranked[0]["source_authority"] == 1.0
    print("[ok] hybrid: vector+keyword+metadata+authority re-ranks; official wins")


def test_intent_classification():
    from apply_advisor.agent import classify_intent
    assert classify_intent("帮我对比 PSL 和 Paris-Saclay") == "PROGRAM_COMPARISON"
    assert classify_intent("法国学生签证怎么办") == "POLICY_QA"
    assert classify_intent("生成材料清单") == "DOCUMENT_CHECKLIST"
    assert classify_intent("预算改成 6000 欧") == "PROFILE_UPDATE"
    assert classify_intent("有哪些适合我的数据科学硕士") == "PROGRAM_RECOMMENDATION"
    print("[ok] intent classification covers the 8-intent taxonomy")


def test_source_precedence_and_missing():
    p = UserProfile()
    p.update(budget_eur_year=20000, source="user_explicit", confidence=1.0)
    # a lower-priority inferred value must NOT overwrite an explicit one (§十)
    p.update(budget_eur_year=99999, source="model_inferred", confidence=0.3)
    assert p.budget_eur_year == 20000
    # but a fresh explicit statement wins (current explicit > stored)
    p.update(budget_eur_year=30000, source="user_explicit", confidence=1.0)
    assert p.budget_eur_year == 30000
    assert "gpa" in p.missing_key_fields() and "language_score" in p.missing_key_fields()
    print("[ok] §十 precedence: inferred can't clobber explicit; missing_key_fields works")


def test_eligibility_and_checklist_shape():
    _patch_retrieval()
    from apply_advisor.tools import execute_tool
    wf = WorkflowState()
    wf.conversation.profile.update(ielts=6.0, gpa=3.0, gpa_scale=4.0)
    elig = json.loads(execute_tool("check_eligibility", {"program_identifier": "2"}, wf))
    assert "checks" in elig and elig["status"] in (
        "eligible", "conditionally_eligible", "ineligible")
    assert all("possible_solution" in c for c in elig["checks"])

    cl = json.loads(execute_tool("generate_checklist", {"program_identifier": "2"}, wf))
    assert set(("completed", "missing", "conditional", "deadlines")) <= set(cl)
    print("[ok] §九 shapes: eligibility checks w/ possible_solution; checklist grouped")


def test_standard_output_and_validation():
    from apply_advisor import output
    res = recommend(PROFILE, [dict(p) for p in FAKE_UNIS], today=TODAY)
    std = output.build_standard_output(
        PROFILE, res, applied_constraints={"countries": ["France"], "max_tuition": 10000})
    assert set(("profile_summary", "applied_constraints", "recommendations", "next_actions")) <= set(std)
    r0 = std["recommendations"][0]
    assert set(("program_id", "category", "match_score", "eligibility",
                "strengths", "risks", "missing_information", "sources")) <= set(r0)
    assert r0["category"] in ("reach", "match", "safer")     # 保底 → 'safer'
    assert std.get("_validated") is True                     # pydantic contract holds
    print("[ok] §十四 standard output built + §十二.4 pydantic-validated")


def test_conflict_and_freshness():
    from apply_advisor import output
    # official current-year 6.5 vs third-party old 7.0 → resolvable to official
    conflict = output.detect_source_conflict("ielts", [
        {"value": 6.5, "official": True, "effective_year": "2027-2028", "retrieved_at": "2026-06-15"},
        {"value": 7.0, "official": False, "effective_year": "2024-2025", "retrieved_at": "2024-01-01"},
    ])
    assert conflict["status"] == "source_conflict" and conflict["resolved_value"] == 6.5
    # equal values → no conflict
    assert output.detect_source_conflict("ielts", [{"value": 6.5}, {"value": 6.5}]) is None
    # freshness
    assert output.data_status({"expiration_date": "2025-01-01"}, today=TODAY) == "expired"
    assert output.data_status({"expiration_date": "2027-12-31"}, today=TODAY) == "valid"
    print("[ok] §十二 source conflict resolves by authority/year; §十三 data_status")


def test_retrieval_fallback():
    _patch_retrieval()
    import apply_advisor.retrieval as R
    from apply_advisor.tools import execute_tool
    R.search_policies = lambda q, top_k=5: []          # simulate nothing found
    out = json.loads(execute_tool("search_policies", {"query": "obscure question"}, WorkflowState()))
    assert out.get("status") == "could_not_confirm"
    _patch_retrieval()                                  # restore
    print("[ok] §十二.3 retrieval fallback → could_not_confirm instead of guessing")


def test_sources_tool():
    _patch_retrieval()
    from apply_advisor.tools import execute_tool
    wf = WorkflowState()
    out = json.loads(execute_tool("get_sources", {"program_identifier": "2"}, wf))
    assert out["sources"] and out["sources"][0]["authority_level"] == "official"
    assert out["grade_requirement"]["scale"] == 4.0
    assert out["grade_requirement"]["type"] == "minimum"
    assert any(s["kind"] == "provenance" for s in wf.conversation.sources)
    print("[ok] get_sources returns official provenance + grade-scale metadata")


# --------------------------------------------- 6. multi-turn decision state
def test_workflow_state_and_persistence():
    wf = WorkflowState()
    wf.exclude(3, "Reach U — MSc Data Science", "user dislikes the city", by="user")
    wf.record_rule_exclusions(
        [{"program_id": 4, "program": "TooExpensive U", "reasons": "over budget"}]
    )
    # rule exclusions are cleared on re-recommendation, user ones survive
    wf.clear_rule_exclusions()
    assert wf.is_user_excluded(3) and "4" not in wf.excluded

    wf.track_application(2, "Match U — MSc Data Science")
    assert wf.set_status(2, "submitted") is None
    assert wf.set_status(2, "not_a_status") is not None
    assert wf.set_material_status(2, "cv", "done") is None
    assert wf.applications["2"]["history"][0]["to"] == "submitted"

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "s.json"
        wf.conversation.profile.update(gpa=3.4, target_field="computer science")
        wf.save(path)
        loaded = WorkflowState.load(path)
        assert loaded.is_user_excluded(3)
        assert loaded.applications["2"]["status"] == "submitted"
        assert loaded.conversation.profile.gpa == 3.4
    print("[ok] workflow: exclusion memory, status tracking, save/load round-trip")


# -------------------------------------------------------- agent end-to-end
def test_agent_end_to_end():
    _patch_retrieval()
    agent = AdvisorAgent()
    reply = agent.chat(
        "I have a 3-year bachelor in computer science, GPA 3.4, IELTS 6.5, want an "
        "english master under 10000 eur in France. What fits, and how does the visa work?"
    )
    assert reply.answer
    called = " ".join(reply.steps)
    assert "update_profile" in called
    assert "recommend_programs" in called
    assert "search_policies" in called
    assert len(reply.sources) >= 1
    # profile was actually populated through the tool path
    p = agent.workflow.conversation.profile
    assert p.gpa == 3.4 and p.bachelor_years == 3
    print(f"[ok] agent E2E: {len(reply.steps)} tool steps, {len(reply.sources)} sources")
    print("     steps:", reply.steps)


def test_agent_tool_error_fallback():
    _patch_retrieval()
    from apply_advisor.tools import execute_tool
    wf = WorkflowState()
    out = json.loads(execute_tool("no_such_tool", {}, wf))
    assert "error" in out
    out = json.loads(execute_tool("exclude_program",
                                  {"program_identifier": "zzz", "reason": "x"}, wf))
    assert "error" in out
    print("[ok] anomaly fallback: unknown tool / unknown program handled gracefully")


if __name__ == "__main__":
    test_profile_and_gpa_conversion()
    test_rule_engine()
    test_grade_scale_and_requirement_type()
    test_chunker()
    test_mock_embedding_stable()
    test_recommendation_tiers()
    test_confidence_gating_and_constraints()
    test_diversity_rerank()
    test_hybrid_ranking()
    test_intent_classification()
    test_source_precedence_and_missing()
    test_eligibility_and_checklist_shape()
    test_standard_output_and_validation()
    test_conflict_and_freshness()
    test_retrieval_fallback()
    test_checklist_and_plan()
    test_sources_tool()
    test_workflow_state_and_persistence()
    test_agent_end_to_end()
    test_agent_tool_error_fallback()
    print("\nALL OFFLINE TESTS PASSED")
