from __future__ import annotations

import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from uuid import uuid4

from spice.protocols import Decision, ExecutionIntent, ExecutionResult
from spice_personal.app import personal as personal_app
from spice_personal.executors.factory import PersonalExecutorConfig
from spice_personal.profile.contract import CATEGORY_EXTERNAL_SYSTEM


REPO_ROOT = Path(__file__).resolve().parents[2]


class _FailingExecutor:
    def execute(self, intent: ExecutionIntent) -> ExecutionResult:
        return ExecutionResult(
            id=f"result-{uuid4().hex}",
            result_type="personal.assistant.evidence_applied",
            status="failed",
            executor="failing-executor",
            output={"operation": intent.operation.get("name", "")},
            error="simulated execution failure",
            refs=[intent.id],
        )


class _FailingSDEPExecutor:
    def execute(self, intent: ExecutionIntent) -> ExecutionResult:
        return ExecutionResult(
            id=f"result-{uuid4().hex}",
            result_type="sdep.execute_result",
            status="failed",
            executor="failing-sdep-executor",
            output={},
            error="wrapper failed",
            refs=[intent.id],
            attributes={
                "sdep": {
                    "response": {
                        "error": {
                            "code": "transport.runtime",
                            "message": "Claude Code print mode failed",
                            "details": {
                                "subtype": "error_max_turns",
                                "stop_reason": "max_turns",
                                "permission_denials": [
                                    {"tool_name": "Bash", "tool_use_id": "tooluse-1"}
                                ],
                            },
                        }
                    }
                }
            },
        )


class ActionProposalExecutionTests(unittest.TestCase):
    def test_confirmed_action_proposal_executes_and_records_success_metadata(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            workspace = Path(tmp_dir) / "personal_workspace"
            runtime = self._build_action_proposal_runtime(workspace)

            turn = personal_app._run_advisory_turn(
                runtime,
                question="Should I apply to the role now?",
                source="tests.personal.action_proposal",
                model=None,
                choice_resolver=lambda **_: personal_app.ADOPTION_STATUS_ADOPTED,
            )

            self.assertEqual(turn.outcome.metadata.get("result_kind"), personal_app.RESULT_KIND_ACTION_PROPOSAL)
            self.assertEqual(
                turn.outcome.metadata.get("decision_adoption_status"),
                personal_app.ADOPTION_STATUS_ADOPTED,
            )
            self.assertEqual(
                turn.outcome.metadata.get("execution_status"),
                personal_app.EXECUTION_STATUS_SUCCESS,
            )
            self.assertEqual(
                turn.reflection.metadata.get("execution_status"),
                personal_app.EXECUTION_STATUS_SUCCESS,
            )
            self.assertEqual(
                turn.outcome.outcome_type,
                personal_app.PERSONAL_OUTCOME_ADVICE_RECORDED,
            )
            self.assertEqual(turn.outcome.attributes.get("suggestion_text"), "proposed action")
            self.assertTrue(self._history_contains(runtime, ExecutionIntent))
            self.assertTrue(self._history_contains(runtime, ExecutionResult))

            current = turn.world_state.entities.get(personal_app.PERSONAL_ENTITY_ID, {})
            self.assertIsInstance(current, dict)
            self.assertEqual(current.get("latest_question"), "Should I apply to the role now?")
            self.assertEqual(current.get("latest_suggestion"), "proposed action")
            self.assertEqual(current.get("confidence"), 0.85)
            self.assertEqual(current.get("urgency"), "high")
            feedback_payload = json.loads(str(current.get("last_feedback", "")))
            self.assertEqual(
                feedback_payload.get("latest_result_kind"),
                personal_app.RESULT_KIND_ACTION_PROPOSAL,
            )
            self.assertEqual(
                feedback_payload.get("latest_decision_adoption_status"),
                personal_app.ADOPTION_STATUS_ADOPTED,
            )

    def test_declined_action_proposal_skips_execution_and_records_not_requested(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            workspace = Path(tmp_dir) / "personal_workspace"
            runtime = self._build_action_proposal_runtime(workspace)

            turn = personal_app._run_advisory_turn(
                runtime,
                question="Should I make this irreversible decision?",
                source="tests.personal.action_proposal",
                model=None,
                choice_resolver=lambda **_: personal_app.ADOPTION_STATUS_DECLINED,
            )

            self.assertEqual(turn.outcome.metadata.get("result_kind"), personal_app.RESULT_KIND_ACTION_PROPOSAL)
            self.assertEqual(
                turn.outcome.metadata.get("decision_adoption_status"),
                personal_app.ADOPTION_STATUS_DECLINED,
            )
            self.assertEqual(
                turn.outcome.metadata.get("execution_status"),
                personal_app.EXECUTION_STATUS_NOT_REQUESTED,
            )
            self.assertEqual(
                turn.reflection.metadata.get("execution_status"),
                personal_app.EXECUTION_STATUS_NOT_REQUESTED,
            )
            self.assertFalse(self._history_contains(runtime, ExecutionIntent))
            self.assertFalse(self._history_contains(runtime, ExecutionResult))

    def test_non_interactive_action_proposal_defaults_to_pending_without_execution(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            workspace = Path(tmp_dir) / "personal_workspace"
            runtime = self._build_action_proposal_runtime(workspace)

            turn = personal_app._run_advisory_turn(
                runtime,
                question="Should I send the final resignation now?",
                source="tests.personal.action_proposal",
                model=None,
                choice_resolver=None,
            )

            self.assertEqual(turn.outcome.metadata.get("result_kind"), personal_app.RESULT_KIND_ACTION_PROPOSAL)
            self.assertEqual(
                turn.outcome.metadata.get("decision_adoption_status"),
                personal_app.ADOPTION_STATUS_PENDING,
            )
            self.assertEqual(
                turn.outcome.metadata.get("execution_status"),
                personal_app.EXECUTION_STATUS_NOT_REQUESTED,
            )
            self.assertFalse(self._history_contains(runtime, ExecutionIntent))
            self.assertFalse(self._history_contains(runtime, ExecutionResult))

    def test_confirmed_action_proposal_failed_execution_records_failed_metadata(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            workspace = Path(tmp_dir) / "personal_workspace"
            runtime = self._build_action_proposal_runtime(workspace)
            runtime.executor = _FailingExecutor()

            turn = personal_app._run_advisory_turn(
                runtime,
                question="Should I trigger the migration now?",
                source="tests.personal.action_proposal",
                model=None,
                choice_resolver=lambda **_: personal_app.ADOPTION_STATUS_ADOPTED,
            )

            self.assertEqual(
                turn.outcome.metadata.get("decision_adoption_status"),
                personal_app.ADOPTION_STATUS_ADOPTED,
            )
            self.assertEqual(
                turn.outcome.metadata.get("execution_status"),
                personal_app.EXECUTION_STATUS_FAILED,
            )
            self.assertEqual(
                turn.reflection.metadata.get("execution_status"),
                personal_app.EXECUTION_STATUS_FAILED,
            )
            self.assertEqual(
                turn.outcome.outcome_type,
                personal_app.PERSONAL_OUTCOME_ADVICE_RECORDED,
            )
            current = turn.world_state.entities.get(personal_app.PERSONAL_ENTITY_ID, {})
            self.assertIsInstance(current, dict)
            self.assertEqual(current.get("latest_question"), "Should I trigger the migration now?")
            self.assertEqual(current.get("latest_suggestion"), "proposed action")
            self.assertEqual(current.get("confidence"), 0.85)
            self.assertEqual(current.get("urgency"), "high")
            self.assertTrue(self._history_contains(runtime, ExecutionIntent))
            self.assertTrue(self._history_contains(runtime, ExecutionResult))

    def test_session_choice_for_action_proposal_uses_confirmation_prompt(self) -> None:
        input_stream = io.StringIO("y\n")
        output_stream = io.StringIO()
        advice = personal_app.PersonalAdvice(
            selected_action=personal_app.PERSONAL_ACTION_SUGGEST,
            suggestion="proposed action",
            urgency="normal",
            confidence=0.9,
        )

        status = personal_app._resolve_session_choice(
            advice=advice,
            result_kind=personal_app.RESULT_KIND_ACTION_PROPOSAL,
            input_stream=input_stream,
            output_stream=output_stream,
        )

        self.assertEqual(status, personal_app.ADOPTION_STATUS_ADOPTED)
        rendered = output_stream.getvalue()
        self.assertIn("advisor> proposed action\n", rendered)
        self.assertIn("confirm execution? [y/N] ", rendered)

    def test_confirmed_action_proposal_route_missing_blocks_execution(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            workspace = Path(tmp_dir) / "personal_workspace"
            runtime = self._build_action_proposal_runtime(workspace)
            profile = {
                "schema_version": "spice_personal.profile.v1",
                "profile_id": "test",
                "executor_mode": "cli",
                "category_routes": {},
            }

            turn = personal_app._run_advisory_turn(
                runtime,
                question="Should I run this automated change?",
                source="tests.personal.action_proposal",
                model=None,
                profile=profile,
                executor_config=PersonalExecutorConfig(mode="mock"),
                choice_resolver=lambda **_: personal_app.ADOPTION_STATUS_ADOPTED,
            )

            self.assertEqual(
                turn.outcome.metadata.get("execution_status"),
                personal_app.EXECUTION_STATUS_FAILED,
            )
            debug = turn.outcome.metadata.get("execution_debug")
            self.assertIsInstance(debug, dict)
            self.assertEqual(debug.get("preflight_allow_execution"), False)
            self.assertEqual(debug.get("route_enabled"), False)
            self.assertFalse(self._history_contains(runtime, ExecutionResult))

    def test_confirmed_action_proposal_failed_sdep_execution_exposes_wrapper_debug_fields(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            workspace = Path(tmp_dir) / "personal_workspace"
            runtime = self._build_action_proposal_runtime(workspace)
            runtime.executor = _FailingSDEPExecutor()
            profile = {
                "schema_version": "spice_personal.profile.v1",
                "profile_id": "test",
                "executor_mode": "sdep",
                "category_routes": {
                    CATEGORY_EXTERNAL_SYSTEM: {
                        "enabled": True,
                        "operation_name": "personal.system",
                        "target": {"kind": "external.service", "id": "system"},
                        "required_capabilities": ["personal.system"],
                    }
                },
            }

            turn = personal_app._run_advisory_turn(
                runtime,
                question="Should I execute this external change?",
                source="tests.personal.action_proposal",
                model=None,
                profile=profile,
                executor_config=PersonalExecutorConfig(mode="sdep"),
                available_capabilities=("personal.system",),
                choice_resolver=lambda **_: personal_app.ADOPTION_STATUS_ADOPTED,
            )

            self.assertEqual(
                turn.outcome.metadata.get("execution_status"),
                personal_app.EXECUTION_STATUS_FAILED,
            )
            debug = turn.outcome.metadata.get("execution_debug")
            self.assertIsInstance(debug, dict)
            self.assertEqual(debug.get("preflight_allow_execution"), True)
            self.assertEqual(debug.get("route_enabled"), True)
            self.assertEqual(debug.get("route_fallback_applied"), False)
            self.assertEqual(debug.get("wrapper_error_code"), "transport.runtime")
            self.assertEqual(debug.get("wrapper_subtype"), "error_max_turns")
            self.assertEqual(debug.get("wrapper_stop_reason"), "max_turns")
            self.assertTrue(bool(debug.get("wrapper_permission_denials")))

    def test_emit_execution_failure_debug_renders_preflight_route_and_wrapper_details(self) -> None:
        output_stream = io.StringIO()
        personal_app._emit_execution_failure_debug(
            output_stream=output_stream,
            execution_debug={
                "preflight_allow_execution": False,
                "preflight_pending_confirmation": False,
                "preflight_errors": ["pei.v1.route.missing: Route missing"],
                "route_enabled": False,
                "route_fallback_applied": True,
                "wrapper_error_code": "transport.runtime",
                "wrapper_error_message": "Claude Code print mode failed",
                "wrapper_subtype": "error_max_turns",
                "wrapper_stop_reason": "max_turns",
                "wrapper_permission_denials": [{"tool_name": "Bash"}],
            },
        )
        rendered = output_stream.getvalue()
        self.assertIn("execution_debug.preflight_allow_execution=false", rendered)
        self.assertIn("execution_debug.preflight_errors=pei.v1.route.missing: Route missing", rendered)
        self.assertIn("execution_debug.route_enabled=false", rendered)
        self.assertIn("execution_debug.route_fallback_applied=true", rendered)
        self.assertIn("execution_debug.wrapper_error_code=transport.runtime", rendered)
        self.assertIn("execution_debug.wrapper_subtype=error_max_turns", rendered)
        self.assertIn("execution_debug.wrapper_stop_reason=max_turns", rendered)
        self.assertIn("execution_debug.wrapper_permission_denials=", rendered)

    def test_confirmed_action_proposal_with_unbounded_scope_degrades_to_pending(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            workspace = Path(tmp_dir) / "personal_workspace"
            runtime = self._build_action_proposal_runtime(
                workspace,
                execution_brief={
                    "category": CATEGORY_EXTERNAL_SYSTEM,
                    "goal": "Run system operation with broad scope.",
                    "inputs": {"task": "Inspect and report", "scope": "global"},
                    "success_criteria": [
                        {"id": "execution.completed", "description": "operation prepared safely"}
                    ],
                },
            )
            profile = {
                "schema_version": "spice_personal.profile.v1",
                "profile_id": "test",
                "executor_mode": "mock",
                "category_routes": {
                    CATEGORY_EXTERNAL_SYSTEM: {
                        "enabled": True,
                        "operation_name": "personal.system",
                        "target": {"kind": "external.service", "id": "system"},
                        "required_capabilities": ["personal.system"],
                    }
                },
            }

            turn = personal_app._run_advisory_turn(
                runtime,
                question="Should I run this broad automation now?",
                source="tests.personal.action_proposal",
                model=None,
                profile=profile,
                executor_config=PersonalExecutorConfig(mode="mock"),
                choice_resolver=lambda **_: personal_app.ADOPTION_STATUS_ADOPTED,
            )

            self.assertEqual(
                turn.outcome.metadata.get("execution_status"),
                personal_app.EXECUTION_STATUS_NOT_REQUESTED,
            )
            self.assertFalse(self._history_contains(runtime, ExecutionResult))

    def test_sdep_missing_capability_with_unavailable_cli_fallback_fails_preflight(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            workspace = Path(tmp_dir) / "personal_workspace"
            runtime = self._build_action_proposal_runtime(workspace)
            profile = {
                "schema_version": "spice_personal.profile.v1",
                "profile_id": "test",
                "executor_mode": "sdep",
                "category_routes": {
                    CATEGORY_EXTERNAL_SYSTEM: {
                        "enabled": True,
                        "operation_name": "personal.system",
                        "target": {"kind": "external.service", "id": "system"},
                        "required_capabilities": ["personal.system"],
                        "fallback_cli": {
                            "operation_name": "personal.system",
                            "target": {"kind": "external.service", "id": "system"},
                        },
                    }
                },
            }

            turn = personal_app._run_advisory_turn(
                runtime,
                question="Should I execute this now?",
                source="tests.personal.action_proposal",
                model=None,
                profile=profile,
                executor_config=PersonalExecutorConfig(mode="sdep"),
                available_capabilities=(),
                choice_resolver=lambda **_: personal_app.ADOPTION_STATUS_ADOPTED,
            )

            self.assertEqual(
                turn.outcome.metadata.get("execution_status"),
                personal_app.EXECUTION_STATUS_FAILED,
            )
            self.assertFalse(self._history_contains(runtime, ExecutionResult))

    def test_apply_profile_to_intent_maps_category_and_execution_brief(self) -> None:
        intent = ExecutionIntent(
            id=f"intent-{uuid4().hex}",
            intent_type="personal.assistant.placeholder",
            status="planned",
            objective={"id": "obj-1", "description": "placeholder"},
            executor_type="agent",
            target={"kind": "placeholder", "id": "placeholder"},
            operation={"name": "noop", "mode": "sync", "dry_run": False},
            input_payload={"preexisting": True},
            parameters={"existing_param": 1},
            constraints=[{"name": "existing.constraint", "kind": "guard", "params": {}}],
            success_criteria=[{"id": "existing.success", "description": "existing"}],
            failure_policy={"strategy": "fail_fast", "max_retries": 0},
            refs=[],
            provenance={},
        )
        decision = Decision(
            id=f"dec-{uuid4().hex}",
            decision_type="personal.assistant.llm",
            status="proposed",
            selected_action=personal_app.PERSONAL_ACTION_SUGGEST,
            refs=[],
            attributes={
                "result_kind": personal_app.RESULT_KIND_ACTION_PROPOSAL,
                "execution_brief": {
                    "category": CATEGORY_EXTERNAL_SYSTEM,
                    "goal": "Run a system operation safely.",
                    "inputs": {"command": "echo hello"},
                    "constraints": [
                        {"name": "brief.constraint", "kind": "policy", "params": {"risk": "low"}}
                    ],
                    "success_criteria": [
                        {"id": "brief.success", "description": "command completes"}
                    ],
                    "risk_level": "low",
                },
            },
        )
        profile = {
            "schema_version": "spice_personal.profile.v1",
            "profile_id": "test",
            "executor_mode": "cli",
            "category_routes": {
                CATEGORY_EXTERNAL_SYSTEM: {
                    "enabled": True,
                    "operation_name": "personal.system",
                    "target": {"kind": "external.service", "id": "system"},
                    "input_defaults": {"from_profile": True},
                    "parameter_defaults": {"timeout_seconds": 12},
                    "guardrails": {"force_dry_run": True},
                }
            },
        }

        resolved_mode = personal_app._apply_profile_to_intent(
            intent,
            decision=decision,
            profile=profile,
            available_capabilities=(),
        )

        self.assertEqual(resolved_mode, "cli")
        self.assertEqual(intent.operation.get("name"), "personal.system")
        self.assertTrue(bool(intent.operation.get("dry_run")))
        self.assertEqual(intent.target.get("kind"), "external.service")
        self.assertEqual(intent.target.get("id"), "system")
        execution_brief = intent.input_payload.get("execution_brief", {})
        self.assertIsInstance(execution_brief, dict)
        self.assertEqual(execution_brief.get("category"), CATEGORY_EXTERNAL_SYSTEM)
        self.assertEqual(execution_brief.get("goal"), "Run a system operation safely.")
        self.assertEqual(intent.parameters.get("risk_level"), "low")
        self.assertTrue(
            any(entry.get("id") == "brief.success" for entry in intent.success_criteria if isinstance(entry, dict))
        )
        self.assertTrue(
            any(entry.get("name") == "brief.constraint" for entry in intent.constraints if isinstance(entry, dict))
        )

    def _build_action_proposal_runtime(
        self,
        workspace: Path,
        *,
        execution_brief: dict[str, object] | None = None,
    ):
        personal_app.ensure_personal_workspace(workspace)
        runtime = personal_app._build_personal_runtime(
            workspace,
            model=None,
            executor_config=PersonalExecutorConfig(mode="mock"),
        )
        runtime.decision_policy = None

        def _decide(_self, state, *, decision_context=None):
            del decision_context
            return Decision(
                id=f"dec-{uuid4().hex}",
                decision_type="personal.assistant.llm",
                status="proposed",
                selected_action=personal_app.PERSONAL_ACTION_SUGGEST,
                refs=[state.id],
                attributes={
                    "result_kind": personal_app.RESULT_KIND_ACTION_PROPOSAL,
                    "suggestion_text": "proposed action",
                    "confidence": 0.85,
                    "urgency": "high",
                    **(
                        {"execution_brief": dict(execution_brief)}
                        if isinstance(execution_brief, dict)
                        else {}
                    ),
                },
            )

        runtime.domain_pack.decide = types.MethodType(_decide, runtime.domain_pack)
        return runtime

    @staticmethod
    def _history_contains(runtime, record_type: type) -> bool:
        return any(isinstance(record, record_type) for record in runtime.state_store.history)


if __name__ == "__main__":
    unittest.main()
