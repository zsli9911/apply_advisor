"""Interactive command-line interface for the Apply Advisor agent.

Usage:
    python -m apply_advisor.cli --ingest              # build the knowledge base
    python -m apply_advisor.cli                        # start interactive chat
    python -m apply_advisor.cli --verbose              # also show the agent's tool trace
    python -m apply_advisor.cli --session my_plan      # persist & resume the workflow

In-chat commands:
    /profile   show the current profile (+ original snapshot & revisions)
    /state     show excluded programs and the application pipeline
    /save      save the session now (also happens automatically on exit)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .agent import AdvisorAgent
from .config import get_settings
from .workflow import WorkflowState

console = Console()
SESSIONS_DIR = Path("sessions")


def do_ingest() -> None:
    from .ingest import run_full_ingest

    console.print("[bold cyan]Building knowledge base...[/bold cyan]")
    stats = run_full_ingest()
    console.print(
        f"[green]Done.[/green] Loaded [bold]{stats['programs']}[/bold] programs across "
        f"[bold]{stats['universities']}[/bold] universities "
        f"([bold]{stats['admission']}[/bold] admission, [bold]{stats['language']}[/bold] language, "
        f"[bold]{stats['document']}[/bold] document, [bold]{stats['sources']}[/bold] source rows) "
        f"and [bold]{stats['policy_chunks']}[/bold] policy chunks."
    )


def _render_reply(reply, verbose: bool) -> None:
    console.print(Panel(Markdown(reply.answer or "(no answer)"), title="Advisor", border_style="cyan"))

    if verbose:
        console.print(f"[dim]intent: {reply.intent}[/dim]")
    if reply.missing_fields:
        console.print(f"[yellow]Still need:[/yellow] {', '.join(reply.missing_fields)}")

    if reply.profile_warnings:
        console.print("[yellow]⚠ Profile checks:[/yellow]")
        for w in reply.profile_warnings:
            console.print(f"  • {w}")

    if reply.sources:
        console.print("[dim]Sources:[/dim]")
        for s in reply.sources:
            console.print(f"  [dim]- ({s['kind']}) {s['ref']} — {s['detail']}[/dim]")

    if verbose and reply.steps:
        console.print("[dim]Agent trace:[/dim]")
        for st in reply.steps:
            console.print(f"  [dim]· {st}[/dim]")


def _print_json_panel(title: str, data) -> None:
    console.print(
        Panel(json.dumps(data, ensure_ascii=False, indent=2, default=str),
              title=title, border_style="magenta")
    )


def _handle_command(cmd: str, workflow: WorkflowState, session_path: Path | None) -> bool:
    """Return True if `cmd` was an in-chat command and was handled."""
    conv = workflow.conversation
    if cmd == "/profile":
        _print_json_panel("Current profile", conv.profile.as_dict())
        if conv.original_profile:
            _print_json_panel("Original profile (frozen)", conv.original_profile)
        if conv.revisions:
            _print_json_panel("Revisions", conv.revisions)
        return True
    if cmd == "/state":
        _print_json_panel("Excluded programs", workflow.excluded or "(none)")
        _print_json_panel("Application pipeline", workflow.applications or "(none)")
        return True
    if cmd == "/save":
        if session_path is None:
            console.print("[yellow]No --session name given; nothing to save to.[/yellow]")
        else:
            workflow.save(session_path)
            console.print(f"[green]Saved to {session_path}[/green]")
        return True
    return False


def interactive(verbose: bool, workflow: WorkflowState, session_path: Path | None) -> None:
    settings = get_settings()
    mode = "MOCK" if settings.mock_llm else settings.chat_model
    session_note = f"   |   session: [cyan]{session_path}[/cyan]" if session_path else ""
    console.print(
        Panel.fit(
            "[bold]Apply Advisor[/bold] — French/European study-abroad AI agent\n"
            f"LLM: [cyan]{mode}[/cyan]   |   type [bold]exit[/bold] to quit, "
            "[bold]/profile /state /save[/bold] for the workflow" + session_note,
            border_style="magenta",
        )
    )
    agent = AdvisorAgent(workflow=workflow)
    try:
        while True:
            try:
                user = console.input("[bold green]you>[/bold green] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]bye[/dim]")
                return
            if not user:
                continue
            if user.lower() in {"exit", "quit", ":q"}:
                console.print("[dim]bye[/dim]")
                return
            if _handle_command(user, workflow, session_path):
                continue
            with console.status("[dim]thinking...[/dim]"):
                reply = agent.chat(user)
            _render_reply(reply, verbose)
    finally:
        if session_path is not None:
            workflow.save(session_path)
            console.print(f"[dim]session saved to {session_path}[/dim]")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply Advisor agentic RAG CLI")
    parser.add_argument("--ingest", action="store_true", help="build/refresh the knowledge base")
    parser.add_argument("--verbose", action="store_true", help="show the agent's tool trace")
    parser.add_argument("--ask", type=str, default=None, help="ask a single question and exit")
    parser.add_argument(
        "--session", type=str, default=None,
        help="session name; loads/saves workflow state under ./sessions/<name>.json",
    )
    parser.add_argument(
        "--no-db", action="store_true",
        help="in-memory mode: read data from JSON instead of Postgres (no database needed)",
    )
    args = parser.parse_args(argv)

    # Must set before the first get_settings() call so the flag is picked up.
    if args.no_db:
        os.environ["NO_DB"] = "1"

    if args.ingest:
        if args.no_db:
            console.print("[yellow]--no-db mode reads data straight from JSON; "
                          "no ingest needed. Skipping.[/yellow]")
        else:
            do_ingest()
        if args.ask is None:
            return 0

    session_path: Path | None = None
    workflow = WorkflowState()
    if args.session:
        session_path = SESSIONS_DIR / f"{args.session}.json"
        if session_path.exists():
            workflow = WorkflowState.load(session_path)
            console.print(f"[dim]resumed session from {session_path}[/dim]")

    if args.ask is not None:
        agent = AdvisorAgent(workflow=workflow)
        reply = agent.chat(args.ask)
        _render_reply(reply, args.verbose)
        if session_path is not None:
            workflow.save(session_path)
        return 0

    interactive(args.verbose, workflow, session_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
