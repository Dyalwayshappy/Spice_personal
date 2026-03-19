from __future__ import annotations

import unittest

from spice.protocols import Decision, ExecutionIntent, ExecutionResult
from spice_personal.execution.execution_intent_v1 import (
    ERROR_CAPABILITY_MISSING,
    ERROR_COMMUNICATE_SIDE_EFFECT_BLOCKED,
    ERROR_EXECUTION_FALLBACK_APPLIED,
    ERROR_FALLBACK_CLI_UNAVAILABLE,
    ERROR_MANAGE_TASK_DISABLED,
    ERROR_OPERATION_CATEGORY_MISMATCH,
    ERROR_SCHEMA_VERSION_INVALID,
    ERROR_SUPPORT_LEVEL_MISMATCH,
    ERROR_SYSTEM_BOUNDARY_MISSING,
    PEIV1RouteContext,
    ensure_minimal_execution_result_output,
    normalize_execution_intent_v1,
    validate_execution_intent_v1,
)


class ExecutionIntentV1Tests(unittest.TestCase):
    def test_normalizer_derives_evidence_search_queries_and_clamps_parameters(self) -> None:
        payload = _base_payload("external.evidence")
        payload["input"] = {
            "evidence_plan": [
                {"fact": "Verify attrition trend for option A.", "why": "It impacts downside execution risk."},
                {"fact": "Compare manager coaching outcomes for option B.", "why": "It impacts promotion path odds."},
                {"fact": "Validate workload scope for each option.", "why": "It impacts sustainability of delivery."},
            ],
        }
        payload["parameters"] = {
            "search_depth": "very_deep",
            "max_results": 999,
            "require_source_citations": False,
        }

        normalized = normalize_execution_intent_v1(payload)

        self.assertTrue(
            any(issue.code == "pei.v1.evidence.search_queries.missing" for issue in normalized.issues)
        )
        self.assertEqual(normalized.payload["parameters"].get("search_depth"), "focused")
        self.assertEqual(normalized.payload["parameters"].get("max_results"), 10)
        self.assertTrue(bool(normalized.payload["parameters"].get("require_source_citations")))
        queries = normalized.payload["input"].get("search_queries")
        self.assertIsInstance(queries, list)
        self.assertEqual(len(queries), 3)

    def test_validator_rejects_schema_and_support_level_mismatch(self) -> None:
        payload = _base_payload("external.system")
        payload["schema_version"] = "v2"
        payload["support_level"] = "limited"

        validation = validate_execution_intent_v1(payload)
        codes = {issue.code for issue in validation.errors}

        self.assertIn(ERROR_SCHEMA_VERSION_INVALID, codes)
        self.assertIn(ERROR_SUPPORT_LEVEL_MISMATCH, codes)

    def test_validator_rejects_operation_category_mismatch(self) -> None:
        payload = _base_payload("external.schedule")
        payload["operation"]["name"] = "personal.system"

        validation = validate_execution_intent_v1(payload)
        codes = {issue.code for issue in validation.errors}

        self.assertIn(ERROR_OPERATION_CATEGORY_MISMATCH, codes)

    def test_validator_degrades_limited_communicate_side_effect_request(self) -> None:
        payload = _base_payload("external.communicate")
        payload["operation"]["dry_run"] = False
        payload["parameters"]["delivery_mode"] = "send"
        payload["input"] = {
            "communication_goal": "Notify stakeholders",
            "audience": "all",
            "direct_send": True,
        }

        validation = validate_execution_intent_v1(payload)

        self.assertFalse(validation.errors)
        self.assertTrue(
            any(issue.code == ERROR_COMMUNICATE_SIDE_EFFECT_BLOCKED for issue in validation.degradations)
        )
        self.assertTrue(bool(validation.payload["operation"].get("dry_run")))
        self.assertEqual(validation.payload["parameters"].get("delivery_mode"), "draft_only")

    def test_validator_rejects_manage_task_when_disabled(self) -> None:
        payload = _base_payload("external.manage_task")

        validation = validate_execution_intent_v1(payload)
        codes = {issue.code for issue in validation.errors}

        self.assertIn(ERROR_MANAGE_TASK_DISABLED, codes)

    def test_route_precheck_rejects_when_capability_missing_without_fallback(self) -> None:
        payload = _base_payload("external.system")
        route_context = PEIV1RouteContext(
            category_route={
                "enabled": True,
                "operation_name": "personal.system",
                "required_capabilities": ["personal.system"],
            },
            fallback_route=None,
            profile_mode="sdep",
            available_capabilities=("personal.gather_evidence",),
            fallback_applied=False,
            fallback_available=False,
        )

        validation = validate_execution_intent_v1(payload, route_context=route_context)
        codes = {issue.code for issue in validation.errors}

        self.assertIn(ERROR_CAPABILITY_MISSING, codes)

    def test_route_precheck_uses_fallback_when_available_and_applied(self) -> None:
        payload = _base_payload("external.system")
        route_context = PEIV1RouteContext(
            category_route={
                "enabled": True,
                "operation_name": "personal.system",
                "required_capabilities": ["personal.system"],
                "fallback_cli": {"operation_name": "personal.system"},
            },
            fallback_route={"operation_name": "personal.system"},
            profile_mode="sdep",
            available_capabilities=(),
            fallback_applied=True,
            fallback_available=True,
        )

        validation = validate_execution_intent_v1(payload, route_context=route_context)

        self.assertFalse(validation.errors)
        self.assertTrue(any(issue.code == ERROR_EXECUTION_FALLBACK_APPLIED for issue in validation.infos))

    def test_route_precheck_rejects_when_fallback_unavailable(self) -> None:
        payload = _base_payload("external.system")
        route_context = PEIV1RouteContext(
            category_route={
                "enabled": True,
                "operation_name": "personal.system",
                "required_capabilities": ["personal.system"],
                "fallback_cli": {"operation_name": "personal.system"},
            },
            fallback_route={"operation_name": "personal.system"},
            profile_mode="sdep",
            available_capabilities=(),
            fallback_applied=False,
            fallback_available=False,
        )

        validation = validate_execution_intent_v1(payload, route_context=route_context)
        codes = {issue.code for issue in validation.errors}

        self.assertIn(ERROR_FALLBACK_CLI_UNAVAILABLE, codes)

    def test_system_unbounded_scope_forces_pending_with_dry_run(self) -> None:
        payload = _base_payload("external.system")
        payload["input"] = {
            "task": "Inspect repo status and report findings",
            "scope": "global",
        }

        validation = validate_execution_intent_v1(payload)

        self.assertFalse(validation.errors)
        self.assertTrue(validation.pending_confirmation)
        self.assertTrue(
            any(issue.code == ERROR_SYSTEM_BOUNDARY_MISSING for issue in validation.degradations)
        )
        self.assertTrue(bool(validation.payload["operation"].get("dry_run")))

    def test_result_minimal_backflow_structure_is_filled(self) -> None:
        intent = ExecutionIntent(
            id="intent-1",
            intent_type="personal.assistant.execution",
            status="planned",
            operation={"name": "personal.system", "mode": "sync", "dry_run": False},
            target={"kind": "external.service", "id": "system"},
            input_payload={"task": "inspect", "scope": "workspace"},
            parameters={},
            constraints=[],
            success_criteria=[{"id": "execution.completed", "description": "done"}],
            failure_policy={"strategy": "fail_fast", "max_retries": 0},
            provenance={"decision_id": "dec-1"},
        )
        decision = Decision(
            id="dec-1",
            decision_type="personal.assistant.llm",
            status="proposed",
            selected_action="personal.assistant.suggest",
            refs=["state-1"],
        )
        result = ExecutionResult(
            id="result-1",
            status="success",
            output={
                "actions": [{"step": "collect status"}],
                "evidence": [{"claim": "workspace clean"}],
            },
            refs=["exec-ref-1"],
        )

        ensure_minimal_execution_result_output(
            result,
            intent=intent,
            decision=decision,
            category="external.system",
        )

        output = result.output
        self.assertTrue(str(output.get("summary", "")).strip())
        self.assertIsInstance(output.get("items"), list)
        self.assertIsInstance(output.get("evidence_items"), list)
        self.assertEqual(output.get("status"), "success")
        self.assertIsInstance(output.get("source_refs"), list)
        self.assertIn("result-1", output.get("source_refs", []))
        self.assertIsInstance(output.get("confidence"), float)


def _base_payload(category: str) -> dict[str, object]:
    support = {
        "external.evidence": "full",
        "external.system": "full",
        "external.communicate": "limited",
        "external.schedule": "limited",
        "external.manage_task": "disabled",
    }[category]
    operation = {
        "external.evidence": "personal.gather_evidence",
        "external.system": "personal.system",
        "external.communicate": "personal.communicate",
        "external.schedule": "personal.schedule",
        "external.manage_task": "personal.manage_task",
    }[category]
    input_payload: dict[str, object]
    if category == "external.evidence":
        input_payload = {
            "evidence_plan": [
                {"fact": "fact 1", "why": "why 1"},
                {"fact": "fact 2", "why": "why 2"},
                {"fact": "fact 3", "why": "why 3"},
            ],
            "search_queries": ["fact 1"],
        }
    elif category == "external.system":
        input_payload = {"task": "Inspect workspace", "scope": "workspace"}
    elif category == "external.communicate":
        input_payload = {"communication_goal": "Notify user", "audience": "stakeholder"}
    elif category == "external.schedule":
        input_payload = {"scheduling_goal": "Find a meeting slot", "time_window": "this week"}
    else:
        input_payload = {"task_goal": "manage tasks"}

    return {
        "schema_version": "v1",
        "category": category,
        "support_level": support,
        "goal": "Execute action safely",
        "operation": {"name": operation, "mode": "sync", "dry_run": False},
        "target": {"kind": "external.service", "id": "test"},
        "input": input_payload,
        "parameters": (
            {"require_source_citations": True}
            if category == "external.evidence"
            else {}
        ),
        "constraints": [],
        "success_criteria": (
            [{"id": "evidence.collected", "description": "evidence collected"}]
            if category == "external.evidence"
            else [{"id": "execution.completed", "description": "completed"}]
        ),
        "failure_policy": {"strategy": "fail_fast", "max_retries": 0},
        "provenance": {
            "decision_id": "dec-1",
            "selected_action": "personal.assistant.suggest",
            "source_domain": "personal.assistant",
            "source_turn_id": "turn-1",
        },
    }


if __name__ == "__main__":
    unittest.main()
