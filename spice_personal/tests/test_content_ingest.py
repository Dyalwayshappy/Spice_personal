from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from spice.protocols import Decision, ExecutionIntent, Observation
from spice_personal.app import personal as personal_app
from spice_personal.executors.factory import PersonalExecutorConfig


REPO_ROOT = Path(__file__).resolve().parents[2]


class _NoExecuteExecutor:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, intent):
        self.calls += 1
        raise AssertionError(f"Executor should not be called during ingest-only flow: {intent.id}")


class ContentIngestTests(unittest.TestCase):
    def test_context_text_flows_into_state_before_decision(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            workspace = Path(tmp_dir) / "personal_workspace"
            personal_app.ensure_personal_workspace(workspace)
            runtime = personal_app._build_personal_runtime(
                workspace,
                model=None,
                executor_config=PersonalExecutorConfig(mode="mock"),
            )
            runtime.executor = _NoExecuteExecutor()
            payloads = personal_app._build_context_ingest_payloads(
                context_text="Project brief: optimize onboarding path.",
                context_file=None,
            )

            turn = personal_app._run_advisory_turn(
                runtime,
                question="What should I do next?",
                source="tests.personal.context_ingest",
                model=None,
                context_ingests=payloads,
            )

            self.assertEqual(len(payloads), 1)
            self.assertNotIn("evidence_summary", payloads[0])
            current = turn.world_state.entities.get(personal_app.PERSONAL_ENTITY_ID, {})
            self.assertIsInstance(current, dict)
            summary = str(current.get("evidence_summary", ""))
            self.assertIn("[context_text]", summary)
            self.assertIn("Project brief: optimize onboarding path.", summary)
            self.assertTrue(
                any(
                    isinstance(record, Observation)
                    and record.observation_type == personal_app.PERSONAL_OBSERVATION_CONTEXT_INGEST
                    for record in runtime.state_store.history
                )
            )
            self.assertEqual(runtime.executor.calls, 0)

    def test_context_file_flows_into_state(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            workspace = Path(tmp_dir) / "personal_workspace"
            note_file = Path(tmp_dir) / "brief.md"
            note_file.write_text("Roadmap note: prioritize profile validation UX.", encoding="utf-8")
            personal_app.ensure_personal_workspace(workspace)
            runtime = personal_app._build_personal_runtime(
                workspace,
                model=None,
                executor_config=PersonalExecutorConfig(mode="mock"),
            )
            payloads = personal_app._build_context_ingest_payloads(
                context_text=None,
                context_file=note_file,
            )

            turn = personal_app._run_advisory_turn(
                runtime,
                question="How should I sequence milestones?",
                source="tests.personal.context_ingest",
                model=None,
                context_ingests=payloads,
            )

            self.assertEqual(len(payloads), 1)
            self.assertNotIn("evidence_summary", payloads[0])
            current = turn.world_state.entities.get(personal_app.PERSONAL_ENTITY_ID, {})
            self.assertIsInstance(current, dict)
            summary = str(current.get("evidence_summary", ""))
            self.assertIn(
                "Roadmap note: prioritize profile validation UX.",
                summary,
            )
            self.assertIn(f"[context_file: {note_file.resolve()}]", summary)

    def test_context_text_and_file_can_coexist(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            workspace = Path(tmp_dir) / "personal_workspace"
            note_file = Path(tmp_dir) / "project_brief.md"
            note_file.write_text("File context: prioritize profile validation.", encoding="utf-8")
            personal_app.ensure_personal_workspace(workspace)
            runtime = personal_app._build_personal_runtime(
                workspace,
                model=None,
                executor_config=PersonalExecutorConfig(mode="mock"),
            )
            payloads = personal_app._build_context_ingest_payloads(
                context_text="Text context: focus on urgent next step.",
                context_file=note_file,
            )

            turn = personal_app._run_advisory_turn(
                runtime,
                question="What should I do next?",
                source="tests.personal.context_ingest",
                model=None,
                context_ingests=payloads,
            )

            self.assertEqual(len(payloads), 2)
            history = runtime.state_store.history
            ingest_observations = [
                record
                for record in history
                if isinstance(record, Observation)
                and record.observation_type == personal_app.PERSONAL_OBSERVATION_CONTEXT_INGEST
            ]
            self.assertEqual(len(ingest_observations), 2)
            sources = {str(obs.attributes.get("source_type")) for obs in ingest_observations}
            self.assertEqual(sources, {"context_text", "context_file"})
            self.assertTrue(
                all("evidence_summary" not in obs.attributes for obs in ingest_observations)
            )
            current = turn.world_state.entities.get(personal_app.PERSONAL_ENTITY_ID, {})
            self.assertIsInstance(current, dict)
            summary = str(current.get("evidence_summary", ""))
            text_header = "[context_text]"
            file_header = f"[context_file: {note_file.resolve()}]"
            self.assertIn(text_header, summary)
            self.assertIn("Text context: focus on urgent next step.", summary)
            self.assertIn(file_header, summary)
            self.assertIn("File context: prioritize profile validation.", summary)
            self.assertLess(summary.find(text_header), summary.find(file_header))
            self.assertTrue(turn.outcome.id)

    def test_invalid_file_type_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            sample = Path(tmp_dir) / "payload.exe"
            sample.write_text("not allowed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported file type"):
                personal_app.validate_personal_context_inputs(
                    context_text=None,
                    context_file=sample,
                )

    def test_oversized_file_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            sample = Path(tmp_dir) / "large.txt"
            sample.write_text("a" * (personal_app.MAX_CONTEXT_FILE_BYTES + 1), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "too large"):
                personal_app.validate_personal_context_inputs(
                    context_text=None,
                    context_file=sample,
                )

    def test_prepare_evidence_intent_enriches_payload_for_external_collection(self) -> None:
        decision = Decision(
            id="dec-evidence",
            decision_type="personal.assistant.llm",
            status="proposed",
            selected_action=personal_app.PERSONAL_ACTION_GATHER_EVIDENCE,
            attributes={
                "evidence_plan": [
                    {
                        "fact": "Verify team attrition trend for each option.",
                        "why": "Execution stability affects long-term promotion probability.",
                    },
                    {
                        "fact": "Validate manager coaching record from direct sources.",
                        "why": "Coaching quality changes management-path readiness.",
                    },
                    {
                        "fact": "Compare downside workload scenarios over 12 months.",
                        "why": "Sustained overload undermines leadership progression.",
                    },
                ]
            },
        )
        intent = ExecutionIntent(
            id="intent-evidence",
            intent_type="personal.assistant.execution",
            status="planned",
            executor_type="external-agent",
            operation={"name": "personal.gather_evidence", "mode": "sync", "dry_run": False},
            input_payload={},
            parameters={},
        )

        personal_app._prepare_evidence_intent(
            intent,
            decision=decision,
            question="Should I choose offer A or offer B?",
            profile={},
            executor_config=None,
            available_capabilities=(),
        )

        self.assertIn("evidence_plan", intent.input_payload)
        self.assertIn("search_queries", intent.input_payload)
        self.assertEqual(len(intent.input_payload.get("evidence_plan", [])), 3)
        self.assertEqual(len(intent.input_payload.get("search_queries", [])), 3)
        self.assertIn("execution_brief", intent.input_payload)
        self.assertTrue(bool(intent.parameters.get("require_source_citations")))


if __name__ == "__main__":
    unittest.main()
