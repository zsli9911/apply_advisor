"""Thin wrapper around the OpenAI API for chat + embeddings.

Supports a deterministic ``MOCK_LLM`` mode so the whole pipeline (ingestion,
retrieval, agent loop) can be exercised offline without spending credits.
The chat() method always returns a *normalised* assistant-message dict:

    {
        "role": "assistant",
        "content": str | None,
        "tool_calls": [
            {"id": str, "type": "function",
             "function": {"name": str, "arguments": "<json string>"}},
            ...
        ] | None,
    }

which is exactly the shape OpenAI accepts back in the ``messages`` list.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from .config import get_settings


class LLMClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = None
        if not self.settings.mock_llm:
            from openai import OpenAI  # imported lazily so mock mode needs no key

            self._client = OpenAI(
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_base_url,
            )

    # ------------------------------------------------------------------ chat
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        if self.settings.mock_llm:
            return _mock_chat(messages, tools)
        return self._chat_openai(messages, tools, temperature)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _chat_openai(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.settings.chat_model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        resp = self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        out: dict[str, Any] = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            out["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        return out

    # ------------------------------------------------------------ streaming
    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
    ):
        """Yield {"type":"delta","text"} for content pieces, then a final
        {"type":"message","message":<assistant dict>}. Tool-call turns emit no
        deltas (content is None), so streaming deltas only ever shows the answer."""
        if self.settings.mock_llm:
            msg = _mock_chat(messages, tools)
            if msg.get("content"):
                yield {"type": "delta", "text": msg["content"]}
            yield {"type": "message", "message": msg}
            return

        kwargs: dict[str, Any] = {
            "model": self.settings.chat_model, "messages": messages,
            "temperature": temperature, "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        stream = self._client.chat.completions.create(**kwargs)
        content: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if getattr(delta, "content", None):
                content.append(delta.content)
                yield {"type": "delta", "text": delta.content}
            for tc in (getattr(delta, "tool_calls", None) or []):
                slot = calls.setdefault(tc.index, {"id": None, "name": "", "args": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["name"] += tc.function.name
                if tc.function and tc.function.arguments:
                    slot["args"] += tc.function.arguments
        msg: dict[str, Any] = {"role": "assistant", "content": "".join(content) or None}
        if calls:
            msg["tool_calls"] = [
                {"id": s["id"] or f"call_{i}", "type": "function",
                 "function": {"name": s["name"], "arguments": s["args"] or "{}"}}
                for i, s in sorted(calls.items())
            ]
        yield {"type": "message", "message": msg}

    # ------------------------------------------------------------- embeddings
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def embed(self, texts: list[str]) -> list[list[float]]:
        # local embeddings for mock mode, or when the chat provider has no embed API
        if self.settings.mock_llm or self.settings.local_embeddings:
            return [_mock_embedding(t, self.settings.embedding_dim) for t in texts]
        resp = self._client.embeddings.create(
            model=self.settings.embedding_model,
            input=texts,
        )
        return [d.embedding for d in resp.data]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


# --------------------------------------------------------------------------- #
# Mock implementations                                                        #
# --------------------------------------------------------------------------- #
def _mock_embedding(text: str, dim: int) -> list[float]:
    """Deterministic pseudo-embedding: hash tokens into a fixed-size vector.

    Not semantically meaningful, but stable and normalised, so cosine search
    still returns consistent, testable results offline.
    """
    vec = [0.0] * dim
    for token in re.findall(r"\w+", text.lower()):
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        idx = h % dim
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _mock_chat(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
) -> dict[str, Any]:
    """A scripted 'agent' good enough to drive the loop offline.

    Strategy: record the profile, then run the rule-engine recommendation,
    then a policy search, then produce a final text answer once all tool
    results are present.
    """
    tool_names = {t["function"]["name"] for t in tools} if tools else set()
    tool_results = [m for m in messages if m.get("role") == "tool"]
    called = {m.get("name") for m in tool_results}
    user_text = " ".join(
        m.get("content", "") for m in messages if m.get("role") == "user"
    )

    if "update_profile" in tool_names and "update_profile" not in called:
        args = _mock_parse_profile_args(user_text)
        if args:
            return _tool_call_message("update_profile", args)

    if "recommend_programs" in tool_names and "recommend_programs" not in called:
        return _tool_call_message("recommend_programs", {})

    if "search_universities" in tool_names and "search_universities" not in called \
            and "recommend_programs" not in tool_names:
        args = _mock_parse_university_args(user_text)
        return _tool_call_message("search_universities", args)

    if "search_policies" in tool_names and "search_policies" not in called:
        return _tool_call_message("search_policies", {"query": user_text[:200] or "student visa"})

    # Otherwise: synthesise a final answer from whatever tool output we saw.
    summary_bits = []
    for tr in tool_results:
        summary_bits.append(f"[{tr.get('name')}] {str(tr.get('content'))[:400]}")
    joined = "\n".join(summary_bits)
    content = (
        "[MOCK ANSWER] Based on the retrieved data below, here is a summary of matching "
        "programs and relevant policy notes. (Set MOCK_LLM=0 and provide an OPENAI_API_KEY "
        "for a real, reasoned answer.)\n\n" + joined
    )
    return {"role": "assistant", "content": content, "tool_calls": None}


def _mock_parse_profile_args(text: str) -> dict[str, Any]:
    """Extract profile facts from free text for the offline scripted agent."""
    uni_args = _mock_parse_university_args(text)
    args: dict[str, Any] = {}
    if "field" in uni_args:
        args["target_field"] = uni_args["field"]
    if "language" in uni_args:
        args["preferred_language"] = uni_args["language"]
    if "gpa" in uni_args:
        args["gpa"] = uni_args["gpa"]
    if "max_tuition_eur" in uni_args:
        args["budget_eur_year"] = uni_args["max_tuition_eur"]
    t = text.lower()
    if "master" in t or "msc" in t:
        args["degree_level"] = "master"
    m = re.search(r"ielts\s*(\d(?:\.\d)?)", t)
    if m:
        args["ielts"] = float(m.group(1))
    if "three-year" in t or "3-year" in t or "三年制" in t:
        args["bachelor_years"] = 3
    return args


def _mock_parse_university_args(text: str) -> dict[str, Any]:
    t = text.lower()
    args: dict[str, Any] = {}
    field_map = {
        "computer science": ["computer", "cs", "data science", "ai", "artificial", "cyber", "informatics"],
        "business": ["business", "management", "finance", "mba", "mim"],
        "engineering": ["engineering", "mechanical", "electrical"],
        "mathematics": ["math", "mathematics"],
        "biology": ["biology", "bio"],
        "chemistry": ["chemistry", "chem"],
        "social science": ["international relations", "political", "social"],
    }
    for field, kws in field_map.items():
        if any(k in t for k in kws):
            args["field"] = field
            break
    if "french" in t and "english" not in t:
        args["language"] = "french"
    elif "english" in t:
        args["language"] = "english"
    m = re.search(r"(\d\.\d)\s*gpa|gpa\s*(\d\.\d)", t)
    if m:
        args["gpa"] = float(m.group(1) or m.group(2))
    m = re.search(r"(\d{4,6})\s*(?:eur|euro|€)", t)
    if m:
        args["max_tuition_eur"] = int(m.group(1))
    return args


_MOCK_ID_COUNTER = {"n": 0}


def _tool_call_message(name: str, args: dict[str, Any]) -> dict[str, Any]:
    _MOCK_ID_COUNTER["n"] += 1
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"mock_call_{_MOCK_ID_COUNTER['n']}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ],
    }
