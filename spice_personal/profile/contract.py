from __future__ import annotations

from pathlib import Path
from typing import Any


PROFILE_SCHEMA_VERSION = "spice_personal.profile.v1"
PROFILE_DEFAULT_ID = "personal.default.v1"
PROFILE_RELATIVE_PATH = Path("config/personal.profile.json")

EXECUTOR_MODE_MOCK = "mock"
EXECUTOR_MODE_CLI = "cli"
EXECUTOR_MODE_SDEP = "sdep"
VALID_EXECUTOR_MODES = (
    EXECUTOR_MODE_MOCK,
    EXECUTOR_MODE_CLI,
    EXECUTOR_MODE_SDEP,
)

CATEGORY_EXTERNAL_EVIDENCE = "external.evidence"
CATEGORY_EXTERNAL_SYSTEM = "external.system"
CATEGORY_EXTERNAL_COMMUNICATE = "external.communicate"
CATEGORY_EXTERNAL_SCHEDULE = "external.schedule"
CATEGORY_EXTERNAL_MANAGE_TASK = "external.manage_task"

P0_CATEGORIES = (
    CATEGORY_EXTERNAL_EVIDENCE,
    CATEGORY_EXTERNAL_SYSTEM,
    CATEGORY_EXTERNAL_COMMUNICATE,
    CATEGORY_EXTERNAL_SCHEDULE,
)
P1_CATEGORIES = (
    CATEGORY_EXTERNAL_MANAGE_TASK,
)
ALL_CATEGORIES = P0_CATEGORIES + P1_CATEGORIES


def default_profile_payload() -> dict[str, Any]:
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "profile_id": PROFILE_DEFAULT_ID,
        "executor_mode": EXECUTOR_MODE_MOCK,
        "category_routes": {
            CATEGORY_EXTERNAL_EVIDENCE: {
                "enabled": True,
                "operation_name": "personal.gather_evidence",
                "target": {
                    "kind": "external.service",
                    "id": "research",
                },
                "required_capabilities": ["personal.gather_evidence"],
            },
            CATEGORY_EXTERNAL_SYSTEM: {
                "enabled": True,
                "operation_name": "personal.system",
                "target": {
                    "kind": "external.service",
                    "id": "system",
                },
                "required_capabilities": ["personal.system"],
            },
            CATEGORY_EXTERNAL_COMMUNICATE: {
                "enabled": True,
                "operation_name": "personal.communicate",
                "target": {
                    "kind": "external.service",
                    "id": "communication",
                },
                "required_capabilities": ["personal.communicate"],
            },
            CATEGORY_EXTERNAL_SCHEDULE: {
                "enabled": True,
                "operation_name": "personal.schedule",
                "target": {
                    "kind": "external.service",
                    "id": "calendar",
                },
                "required_capabilities": ["personal.schedule"],
            },
            CATEGORY_EXTERNAL_MANAGE_TASK: {
                "enabled": False,
                "operation_name": "personal.manage_task",
                "target": {
                    "kind": "external.service",
                    "id": "task_manager",
                },
                "required_capabilities": ["personal.manage_task"],
            },
        },
    }


def normalize_category(value: Any) -> str:
    token = str(value or "").strip().lower()
    if token in ALL_CATEGORIES:
        return token
    return ""


def infer_category_from_selected_action(selected_action: str) -> str:
    token = str(selected_action or "").strip().lower()
    if not token:
        return CATEGORY_EXTERNAL_SYSTEM
    if "gather_evidence" in token or "evidence" in token or "research" in token or "search" in token:
        return CATEGORY_EXTERNAL_EVIDENCE
    if "schedule" in token or "calendar" in token:
        return CATEGORY_EXTERNAL_SCHEDULE
    if "communicate" in token or "message" in token or "notify" in token:
        return CATEGORY_EXTERNAL_COMMUNICATE
    if "task" in token or "ticket" in token:
        return CATEGORY_EXTERNAL_MANAGE_TASK
    return CATEGORY_EXTERNAL_SYSTEM


def ensure_minimum_execution_brief(
    brief: Any,
    *,
    selected_action: str,
    suggestion_text: str,
) -> dict[str, Any]:
    payload = dict(brief) if isinstance(brief, dict) else {}
    category = normalize_category(payload.get("category"))
    if not category:
        category = infer_category_from_selected_action(selected_action)

    goal = _as_text(payload.get("goal"))
    if not goal:
        goal = _as_text(suggestion_text)
    if not goal:
        goal = f"Execute requested action: {selected_action or 'action'}"

    success_criteria = _normalize_success_criteria(payload.get("success_criteria"))
    if not success_criteria:
        success_criteria = [
            {
                "id": "execution.completed",
                "description": "Requested action completed successfully.",
            }
        ]

    normalized: dict[str, Any] = {
        "category": category,
        "goal": goal,
        "success_criteria": success_criteria,
    }

    optional_fields = (
        "inputs",
        "constraints",
        "expected_output",
        "risk_level",
        "dry_run_preferred",
        "timeout_seconds",
        "idempotency_hint",
    )
    for key in optional_fields:
        if key not in payload:
            continue
        normalized[key] = payload[key]
    return normalized


def _normalize_success_criteria(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            description = _as_text(item.get("description"))
            if not description:
                continue
            entry = dict(item)
            if not _as_text(entry.get("id")):
                entry["id"] = f"criterion.{len(normalized) + 1}"
            normalized.append(entry)
            continue
        if isinstance(item, str):
            description = item.strip()
            if not description:
                continue
            normalized.append(
                {
                    "id": f"criterion.{len(normalized) + 1}",
                    "description": description,
                }
            )
    return normalized


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""
