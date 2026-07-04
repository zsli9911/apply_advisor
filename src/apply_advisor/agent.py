"""The agentic tool-calling loop.

The agent plans which tools to call, in what order, based on the student's
question and the persistent workflow state — profile analysis, rule-engine
recommendation, policy retrieval, checklist/plan generation, and application
tracking — then explains the result with source tracking and validation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .config import get_settings
from .llm import LLMClient
from .profile import validate_profile
from .tools import TOOL_SCHEMAS, execute_tool
from .workflow import WorkflowState

SYSTEM_PROMPT = """You are "Apply Advisor", an expert assistant for students applying to \
master's programs in Europe, the UK, and Canada. You run a complete, traceable advising \
workflow over six tasks:

1. BACKGROUND ANALYSIS (用户背景分析) — whenever the student reveals or changes facts \
(degree, school + tier, major, GPA + its scale, 3/4-year bachelor, IELTS/TOEFL/French/GRE, \
internships/research/projects, budget, countries, cities, career goal), call `update_profile`. \
Changes are logged and the original profile is preserved automatically.

2. PROGRAM RECOMMENDATION (推荐与排序) — for "which schools should I apply to" questions, \
call `recommend_programs`. It runs the hard-condition rule engine, scores 7 weighted \
dimensions, and classifies reach(冲刺)/match(匹配)/safety(保底); its `standard_output` gives \
each program a category, match_score, eligibility, strengths, risks, missing_information and \
sources, plus overall next_actions — present those. Explain WHY each fits, mention notable \
exclusions with reasons, and suggest a balanced portfolio across tiers.

3. FILTERING & FACTS (院校筛选) — use `search_universities` for factual "what exists" \
queries, `compare_programs` for side-by-side comparisons, `check_eligibility` when the \
student asks about ONE specific program or why something was excluded (it returns per-check \
results with a possible_solution for anything not met).

4. POLICY QUESTIONS (招生政策检索) — use `search_policies` for anything procedural: official \
requirements, credential/degree conversion (学历换算, 3-year bachelor recognition), language \
waivers, visas, Campus France, scholarships, housing. For "where does this figure come from / \
is it official / how current is it", use `get_sources` — it returns the official source URL, \
retrieval date, authority level, and whether a grade bar is a hard MINIMUM or a RECOMMENDED \
guideline (with its native scale, e.g. 12/20 for France, and a confidence level).

5. MATERIALS & PLANNING (申请材料与计划) — use `generate_checklist` for personalized \
material lists (it flags MISSING items against the profile), and `build_application_plan` \
for a deadline-ordered timeline with milestones.

6. MULTI-TURN DECISIONS (多轮决策管理) — when the student changes budget/country/field, \
update the profile then RE-RUN `recommend_programs`; prior user exclusions are respected. \
Use `exclude_program` (with the reason) when the student rejects an option, \
`set_application_status` / `update_material_status` to track progress, and \
`get_workflow_summary` when asked "where are we".

Fallback & conflict handling: if `search_policies` returns status could_not_confirm, tell the \
student it isn't published rather than guessing. If two sources disagree, surface both and \
recommend confirming with admissions. Treat a null requirement as not_publicly_specified.

Rules:
- Ground every concrete claim in tool results. Do NOT invent tuition numbers, deadlines,
  visa amounts, or admission thresholds — retrieve them.
- Distinguish fact types: verified_fact (from structured data / official sources),
  rule_based_inference (rule-engine verdicts), model_interpretation (your reasoning),
  and unknown (not published). Never present an inference or a guess as a verified fact.
- If constraints conflict or key facts are missing (GPA scale, language scores), say so and
  ask for or suggest the fix. Missing language test → point at waiver policies.
- Structure each recommendation as: 为什么推荐 / 为什么可能适合 / 存在哪些风险 /
  哪些信息还需确认 / 对应官方来源. Group programs by 冲刺/匹配/保底 with a one-line why.
- Policy documents are illustrative; remind the student to confirm official figures with
  official sources.
- Answer in the same language the student writes in (English or Chinese).
"""

# intent classification
INTENTS = [
    "PROGRAM_COMPARISON", "DOCUMENT_CHECKLIST", "APPLICATION_PLANNING",
    "POLICY_QA", "PROFILE_UPDATE", "FOLLOW_UP_REFINEMENT",
    "PROGRAM_RECOMMENDATION", "PROFILE_ANALYSIS",
]
_INTENT_CUES = [
    ("PROGRAM_COMPARISON", ("对比", "比较", "compare", " vs ", "哪个更")),
    ("DOCUMENT_CHECKLIST", ("材料", "清单", "checklist", "推荐信", "文书", "documents")),
    ("APPLICATION_PLANNING", ("计划", "时间线", "timeline", "截止", "deadline", "规划", "什么时候")),
    ("POLICY_QA", ("签证", "visa", "政策", "豁免", "换算", "campus france", "奖学金",
                   "scholarship", "住宿", "保险", "policy")),
    ("PROFILE_UPDATE", ("改成", "改为", "更新", "预算改", "换成", "change my", "update my")),
    ("FOLLOW_UP_REFINEMENT", ("排除", "exclude", "不想", "去掉", "换个", "instead")),
    ("DOCUMENT_CHECKLIST", ("需要准备", "要交什么")),
    ("PROGRAM_RECOMMENDATION", ("推荐", "申请哪些", "有哪些", "适合", "recommend",
                                "which program", "哪些项目")),
]


def classify_intent(text: str) -> str:
    t = (text or "").lower()
    for intent, cues in _INTENT_CUES:
        if any(c.lower() in t for c in cues):
            return intent
    return "PROGRAM_RECOMMENDATION"


@dataclass
class AgentReply:
    answer: str
    sources: list[dict[str, Any]]
    steps: list[str]
    profile_warnings: list[str]
    intent: str = "PROGRAM_RECOMMENDATION"
    missing_fields: list[str] = None  # type: ignore[assignment]


class AdvisorAgent:
    def __init__(self, workflow: WorkflowState | None = None) -> None:
        self.settings = get_settings()
        self.llm = LLMClient()
        self.workflow = workflow or WorkflowState()

    @property
    def state(self):  # convenience alias for the conversation sub-state
        return self.workflow.conversation

    def chat(self, user_message: str) -> AgentReply:
        state = self.state
        # reset per-turn provenance and trace, keep persistent profile + history
        turn_sources_start = len(state.sources)
        state.steps = []
        self.workflow.last_recommendation = None
        intent = classify_intent(user_message)
        state.add_step(f"intent={intent}")
        state.messages.append({"role": "user", "content": user_message})

        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += state.messages

        final_answer = ""
        for _step in range(self.settings.max_agent_steps):
            resp = self.llm.chat(messages, tools=TOOL_SCHEMAS)

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": resp.get("content"),
            }
            if resp.get("tool_calls"):
                assistant_msg["tool_calls"] = resp["tool_calls"]
            messages.append(assistant_msg)
            state.messages.append(assistant_msg)

            tool_calls = resp.get("tool_calls")
            if not tool_calls:
                final_answer = resp.get("content") or ""
                break

            for tc in tool_calls:
                name = tc["function"]["name"]
                raw_args = tc["function"].get("arguments") or "{}"
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    args = {}
                state.add_step(f"call {name}({json.dumps(args, ensure_ascii=False)})")
                result = execute_tool(name, args, self.workflow)
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": name,
                    "content": result,
                }
                messages.append(tool_msg)
                state.messages.append(tool_msg)
        else:
            # loop exhausted without a final content answer
            final_answer = (
                "I gathered information but reached the step limit before finishing. "
                "Here is what I found so far — please narrow your question and I'll continue."
            )

        warnings = validate_profile(state.profile)
        turn_sources = state.sources[turn_sources_start:]
        return AgentReply(
            answer=final_answer,
            sources=turn_sources,
            steps=list(state.steps),
            profile_warnings=warnings,
            intent=intent,
            missing_fields=state.profile.missing_key_fields(),
        )

    def chat_stream(self, user_message: str):
        """Streaming variant: yields SSE-ready events — intent, per-tool step,
        final-answer deltas, then a terminal 'done' payload (sources, recommendations)."""
        state = self.state
        turn_sources_start = len(state.sources)
        state.steps = []
        self.workflow.last_recommendation = None
        intent = classify_intent(user_message)
        state.add_step(f"intent={intent}")
        yield {"type": "intent", "intent": intent}
        state.messages.append({"role": "user", "content": user_message})

        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += state.messages

        final_answer = ""
        for _step in range(self.settings.max_agent_steps):
            assembled: dict[str, Any] = {}
            for ev in self.llm.chat_stream(messages, tools=TOOL_SCHEMAS):
                if ev["type"] == "delta":
                    yield ev
                else:
                    assembled = ev["message"]

            assistant_msg: dict[str, Any] = {"role": "assistant", "content": assembled.get("content")}
            if assembled.get("tool_calls"):
                assistant_msg["tool_calls"] = assembled["tool_calls"]
            messages.append(assistant_msg)
            state.messages.append(assistant_msg)

            tool_calls = assembled.get("tool_calls")
            if not tool_calls:
                final_answer = assembled.get("content") or ""
                break

            for tc in tool_calls:
                name = tc["function"]["name"]
                raw_args = tc["function"].get("arguments") or "{}"
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    args = {}
                state.add_step(f"call {name}({json.dumps(args, ensure_ascii=False)})")
                yield {"type": "step", "step": name}
                result = execute_tool(name, args, self.workflow)
                state.messages.append({"role": "tool", "tool_call_id": tc["id"],
                                       "name": name, "content": result})
                messages.append(state.messages[-1])
        else:
            final_answer = (
                "I gathered information but reached the step limit before finishing. "
                "Here is what I found so far — please narrow your question and I'll continue."
            )
            yield {"type": "delta", "text": final_answer}

        yield {
            "type": "done",
            "answer": final_answer,
            "intent": intent,
            "missing_fields": state.profile.missing_key_fields(),
            "profile_warnings": validate_profile(state.profile),
            "sources": state.sources[turn_sources_start:],
            "steps": list(state.steps),
            "recommendations": self.workflow.last_recommendation,
        }
