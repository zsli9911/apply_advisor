"""Traceable, multi-turn decision workflow.

Owns everything that must survive across turns and across constraint changes:
- the profile + its original snapshot + revision log (in ConversationState)
- excluded programs and WHY they were excluded (by rules or by the user)
- the application pipeline: per-program status + per-material status
- JSON persistence so a session can be saved and resumed
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field as _field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .profile import ConversationState, UserProfile

APPLICATION_STATUSES = [
    "shortlisted", "preparing", "submitted", "interview",
    "admitted", "rejected", "declined", "withdrawn",
]
MATERIAL_STATUSES = ["missing", "todo", "in_progress", "done"]


@dataclass
class WorkflowState:
    """Wraps ConversationState with exclusion memory and application tracking."""
    conversation: ConversationState = _field(default_factory=ConversationState)
    # program_id (as str) -> {"program", "reason", "by": "rules"|"user", "at"}
    excluded: dict[str, dict[str, Any]] = _field(default_factory=dict)
    # program_id (as str) -> {"program", "status", "materials": {key: status}, "history": [...]}
    applications: dict[str, dict[str, Any]] = _field(default_factory=dict)
    # transient: the most recent recommend_programs standard_output (for the UI)
    last_recommendation: Optional[dict[str, Any]] = None

    # ------------------------------------------------------------ excludes
    def exclude(self, program_id: Any, program: str, reason: str, by: str = "user") -> None:
        self.excluded[str(program_id)] = {
            "program": program,
            "reason": reason,
            "by": by,
            "at": datetime.now().isoformat(timespec="seconds"),
        }

    def record_rule_exclusions(self, reports: list[dict[str, Any]]) -> None:
        """Remember programs the rule engine rejected, with reasons."""
        for r in reports:
            pid = str(r.get("program_id"))
            # user decisions take precedence over rule verdicts; and rules may
            # re-include a program if the profile changed, so only overwrite
            # rule-based entries.
            if pid in self.excluded and self.excluded[pid]["by"] == "user":
                continue
            self.excluded[pid] = {
                "program": r.get("program"),
                "reason": r.get("reasons"),
                "by": "rules",
                "at": datetime.now().isoformat(timespec="seconds"),
            }

    def clear_rule_exclusions(self) -> None:
        """Called before re-recommendation so rule verdicts reflect the CURRENT profile."""
        self.excluded = {
            k: v for k, v in self.excluded.items() if v.get("by") == "user"
        }

    def is_user_excluded(self, program_id: Any) -> bool:
        e = self.excluded.get(str(program_id))
        return bool(e and e.get("by") == "user")

    # -------------------------------------------------------- applications
    def track_application(self, program_id: Any, program: str, status: str = "shortlisted") -> dict[str, Any]:
        pid = str(program_id)
        if pid not in self.applications:
            self.applications[pid] = {
                "program": program,
                "status": status,
                "materials": {},
                "history": [],
            }
        return self.applications[pid]

    def set_status(self, program_id: Any, status: str) -> Optional[str]:
        if status not in APPLICATION_STATUSES:
            return f"Unknown status '{status}'. Valid: {', '.join(APPLICATION_STATUSES)}"
        pid = str(program_id)
        if pid not in self.applications:
            return f"Program {program_id} is not tracked yet — shortlist it first."
        app = self.applications[pid]
        app["history"].append(
            {"from": app["status"], "to": status,
             "at": datetime.now().isoformat(timespec="seconds")}
        )
        app["status"] = status
        return None

    def set_material_status(self, program_id: Any, item_key: str, status: str) -> Optional[str]:
        if status not in MATERIAL_STATUSES:
            return f"Unknown material status '{status}'. Valid: {', '.join(MATERIAL_STATUSES)}"
        pid = str(program_id)
        if pid not in self.applications:
            return f"Program {program_id} is not tracked yet — shortlist it first."
        self.applications[pid]["materials"][item_key] = status
        return None

    def seed_materials(self, program_id: Any, checklist_items: list[dict[str, Any]]) -> None:
        pid = str(program_id)
        if pid not in self.applications:
            return
        mats = self.applications[pid]["materials"]
        for item in checklist_items:
            mats.setdefault(item["key"], item["status"])

    # ------------------------------------------------------------- summary
    def conversation_summary(self) -> list[str]:
        """Short decision summary: revisions + user exclusions."""
        lines: list[str] = []
        for rev in self.conversation.revisions:
            for field, chg in rev.get("changes", {}).items():
                if "to" in chg:
                    lines.append(f"changed {field}: {chg.get('from')} → {chg['to']}")
        for e in self.excluded.values():
            if e.get("by") == "user":
                lines.append(f"excluded {e['program']} ({e['reason']})")
        return lines

    def summary(self) -> dict[str, Any]:
        conv = self.conversation
        return {
            "profile": conv.profile.as_dict(),
            "original_profile": conv.original_profile,
            "revisions": conv.revisions,
            "conversation_summary": self.conversation_summary(),
            "excluded_programs": self.excluded,
            "applications": self.applications,
        }

    # --------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "profile": self.conversation.profile.as_dict(),
            "profile_field_meta": self.conversation.profile.field_meta,
            "original_profile": self.conversation.original_profile,
            "revisions": self.conversation.revisions,
            "messages": self.conversation.messages,
            "excluded": self.excluded,
            "applications": self.applications,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "WorkflowState":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        profile = UserProfile()
        profile.update(**data.get("profile", {}))
        profile.field_meta = data.get("profile_field_meta", {})
        conv = ConversationState(
            profile=profile,
            original_profile=data.get("original_profile"),
            revisions=data.get("revisions", []),
            messages=data.get("messages", []),
        )
        return cls(
            conversation=conv,
            excluded=data.get("excluded", {}),
            applications=data.get("applications", {}),
        )
