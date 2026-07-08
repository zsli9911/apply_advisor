"""Web chat UI for the advisor (FastAPI + one self-contained page).

Run:  PYTHONPATH=src python -m apply_advisor.webapp    # http://127.0.0.1:8000
Honours .env exactly like the CLI (NO_DB / MOCK_LLM / provider keys), so in
no-db mode it needs no database. Each browser tab is one in-memory session.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import json

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from .agent import AdvisorAgent
from .config import get_settings
from .workflow import WorkflowState

app = FastAPI(title="Apply Advisor")
_WEB = Path(__file__).parent / "web"
_SESSIONS: dict[str, AdvisorAgent] = {}


class ChatIn(BaseModel):
    session_id: str
    message: str


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (_WEB / "index.html").read_text(encoding="utf-8")


@app.get("/api/config")
def config() -> dict[str, Any]:
    s = get_settings()
    return {"mode": "MOCK" if s.mock_llm else s.chat_model, "no_db": s.no_db}


def _agent(session_id: str) -> AdvisorAgent:
    agent = _SESSIONS.get(session_id)
    if agent is None:
        agent = AdvisorAgent(workflow=WorkflowState())
        _SESSIONS[session_id] = agent
    return agent


@app.post("/api/chat")
def chat(inp: ChatIn) -> dict[str, Any]:
    agent = _agent(inp.session_id)
    reply = agent.chat(inp.message)
    return {
        "answer": reply.answer,
        "intent": reply.intent,
        "missing_fields": reply.missing_fields or [],
        "profile_warnings": reply.profile_warnings,
        "sources": reply.sources,
        "steps": reply.steps,
        "recommendations": agent.workflow.last_recommendation,
    }


@app.post("/api/chat/stream")
def chat_stream(inp: ChatIn) -> StreamingResponse:
    agent = _agent(inp.session_id)

    def gen():
        for ev in agent.chat_stream(inp.message):
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/state/{session_id}")
def state(session_id: str) -> dict[str, Any]:
    agent = _SESSIONS.get(session_id)
    return agent.workflow.summary() if agent else {"profile": {}}


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
