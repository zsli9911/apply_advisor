"""Recommendation evaluation harness (§十七).

Runs a fixed set of canonical cases against the REAL in-memory data
(data/programs.json) and checks recommendation-quality metrics — hard-condition
errors, ineligible leakage, diversity, tier classification, multi-turn
consistency, and source-conflict handling. Re-run after any change to prompts,
chunking, or scoring rules.

    MOCK_LLM=1 NO_DB=1 python tests/eval_cases.py
"""
from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

os.environ.setdefault("MOCK_LLM", "1")
os.environ.setdefault("NO_DB", "1")          # use JSON data, no database
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from apply_advisor import output, retrieval          # noqa: E402
from apply_advisor.profile import UserProfile         # noqa: E402
from apply_advisor.recommend import recommend         # noqa: E402

TODAY = date(2026, 7, 13)


def _run(profile: UserProfile):
    cands = retrieval.fetch_candidates(
        field=profile.target_field, language=profile.preferred_language,
        degree_level=profile.degree_level, countries=profile.countries or None,
    )
    return recommend(profile, cands, today=TODAY)


CASES = [
    ("低成绩、较高预算、接受商学院",
     UserProfile(gpa=2.8, gpa_scale=4.0, ielts=6.5, target_field="business",
                 degree_level="master", preferred_language="english", budget_eur_year=30000)),
    ("高成绩、低预算、仅接受公立大学",
     UserProfile(gpa=3.8, gpa_scale=4.0, ielts=7.0, target_field="computer science",
                 degree_level="master", preferred_language="english", budget_eur_year=6000,
                 requires_public_university=True)),
    ("跨专业申请数据科学",
     UserProfile(gpa=3.3, gpa_scale=4.0, ielts=6.5, major="economics",
                 target_field="computer science", degree_level="master",
                 preferred_language="english", budget_eur_year=15000)),
    ("语言暂时不达标(接受语言班)",
     UserProfile(gpa=3.3, gpa_scale=4.0, ielts=5.5, accepts_pre_master=True,
                 target_field="computer science", degree_level="master",
                 preferred_language="english", budget_eur_year=15000)),
    ("三年制本科 + 中途修改国家和预算",
     UserProfile(gpa=3.2, gpa_scale=4.0, ielts=6.5, bachelor_years=3,
                 target_field="computer science", degree_level="master",
                 preferred_language="english", budget_eur_year=8000, countries=["France"])),
]


def evaluate() -> int:
    hard_errors = 0
    print(f"{'case':44} {'#rec':>4} {'#excl':>5} {'countries':>9} {'tiers'}")
    print("-" * 90)
    for name, profile in CASES:
        res = _run(profile)
        recs = res.recommendations
        # metric: no recommended program may be ineligible (硬条件错误率)
        bad = [r for r in recs if r.report.status == "ineligible"]
        hard_errors += len(bad)
        countries = {r.program.get("country") for r in recs}
        tiers = sorted({r.tier for r in recs})
        print(f"{name:44.44} {len(recs):>4} {len(res.excluded):>5} {len(countries):>9} {tiers}")
        assert not bad, f"{name}: ineligible program recommended!"

    # case 5: multi-turn consistency — change country + budget, re-recommend
    name, profile = CASES[-1]
    profile.update(countries=["Germany"], budget_eur_year=0)  # free tuition only
    res2 = _run(profile)
    de = [r for r in res2.recommendations if r.program.get("country") == "Germany"]
    print("-" * 90)
    print(f"multi-turn: after switching to Germany + free-tuition, "
          f"{len(de)}/{len(res2.recommendations)} recs are in Germany")
    assert all(r.program.get("country") == "Germany" for r in res2.recommendations), \
        "multi-turn: recommendations ignored the country change"

    # case 6: source conflict (官网 6.5 vs 第三方旧 PDF 7.0)
    conflict = output.detect_source_conflict("ielts", [
        {"value": 6.5, "official": True, "effective_year": "2027-2028"},
        {"value": 7.0, "official": False, "effective_year": "2024-2025"},
    ])
    assert conflict and conflict["resolved_value"] == 6.5
    print(f"conflict case: {conflict['status']} → resolved {conflict['resolved_value']} "
          f"({conflict['recommended_action']})")

    print("\nHARD-CONDITION ERRORS:", hard_errors)
    print("RESULT:", "PASS ✅" if hard_errors == 0 else "FAIL ❌")
    return 0 if hard_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(evaluate())
