from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from spice_personal.app import personal as personal_app
from spice_personal.executors.factory import PersonalExecutorConfig


REPO_ROOT = Path(__file__).resolve().parents[2]


class SuggestionAdoptionMetadataTests(unittest.TestCase):
    def test_suggestion_adopted_records_outcome_and_reflection_metadata(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            workspace = Path(tmp_dir) / "personal_workspace"
            personal_app.ensure_personal_workspace(workspace)

            runtime = personal_app._build_personal_runtime(
                workspace,
                model=None,
                executor_config=PersonalExecutorConfig(mode="mock"),
            )
            turn = personal_app._run_advisory_turn(
                runtime,
                question="Should I switch teams?",
                source="tests.personal.metadata",
                model=None,
                choice_resolver=lambda **_: personal_app.ADOPTION_STATUS_ADOPTED,
            )

            self.assertEqual(turn.outcome.metadata.get("result_kind"), personal_app.RESULT_KIND_SUGGESTION)
            self.assertEqual(
                turn.outcome.metadata.get("decision_adoption_status"),
                personal_app.ADOPTION_STATUS_ADOPTED,
            )
            self.assertEqual(
                turn.outcome.metadata.get("execution_status"),
                personal_app.EXECUTION_STATUS_NOT_REQUESTED,
            )
            self.assertEqual(
                turn.reflection.metadata.get("decision_adoption_status"),
                personal_app.ADOPTION_STATUS_ADOPTED,
            )
            current = turn.world_state.entities.get(personal_app.PERSONAL_ENTITY_ID, {})
            self.assertIsInstance(current, dict)
            feedback_payload = json.loads(str(current.get("last_feedback", "")))
            self.assertEqual(
                feedback_payload.get("latest_decision_adoption_status"),
                personal_app.ADOPTION_STATUS_ADOPTED,
            )
            self.assertEqual(
                feedback_payload.get("latest_result_kind"),
                personal_app.RESULT_KIND_SUGGESTION,
            )

    def test_suggestion_declined_records_outcome_and_reflection_metadata(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            workspace = Path(tmp_dir) / "personal_workspace"
            personal_app.ensure_personal_workspace(workspace)

            runtime = personal_app._build_personal_runtime(
                workspace,
                model=None,
                executor_config=PersonalExecutorConfig(mode="mock"),
            )
            turn = personal_app._run_advisory_turn(
                runtime,
                question="Should I relocate for work?",
                source="tests.personal.metadata",
                model=None,
                choice_resolver=lambda **_: personal_app.ADOPTION_STATUS_DECLINED,
            )

            self.assertEqual(turn.outcome.metadata.get("result_kind"), personal_app.RESULT_KIND_SUGGESTION)
            self.assertEqual(
                turn.outcome.metadata.get("decision_adoption_status"),
                personal_app.ADOPTION_STATUS_DECLINED,
            )
            self.assertEqual(
                turn.outcome.metadata.get("execution_status"),
                personal_app.EXECUTION_STATUS_NOT_REQUESTED,
            )
            self.assertEqual(
                turn.reflection.metadata.get("decision_adoption_status"),
                personal_app.ADOPTION_STATUS_DECLINED,
            )
            current = turn.world_state.entities.get(personal_app.PERSONAL_ENTITY_ID, {})
            self.assertIsInstance(current, dict)
            feedback_payload = json.loads(str(current.get("last_feedback", "")))
            self.assertEqual(
                feedback_payload.get("latest_decision_adoption_status"),
                personal_app.ADOPTION_STATUS_DECLINED,
            )
            self.assertEqual(
                feedback_payload.get("latest_result_kind"),
                personal_app.RESULT_KIND_SUGGESTION,
            )

    def test_suggestion_without_explicit_choice_records_pending_metadata(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            workspace = Path(tmp_dir) / "personal_workspace"
            personal_app.ensure_personal_workspace(workspace)

            runtime = personal_app._build_personal_runtime(
                workspace,
                model=None,
                executor_config=PersonalExecutorConfig(mode="mock"),
            )
            turn = personal_app._run_advisory_turn(
                runtime,
                question="Should I switch projects?",
                source="tests.personal.metadata",
                model=None,
                choice_resolver=None,
            )

            self.assertEqual(turn.outcome.metadata.get("result_kind"), personal_app.RESULT_KIND_SUGGESTION)
            self.assertEqual(
                turn.outcome.metadata.get("decision_adoption_status"),
                personal_app.ADOPTION_STATUS_PENDING,
            )
            self.assertEqual(
                turn.outcome.metadata.get("execution_status"),
                personal_app.EXECUTION_STATUS_NOT_REQUESTED,
            )
            self.assertEqual(
                turn.reflection.metadata.get("decision_adoption_status"),
                personal_app.ADOPTION_STATUS_PENDING,
            )
            current = turn.world_state.entities.get(personal_app.PERSONAL_ENTITY_ID, {})
            self.assertIsInstance(current, dict)
            feedback_payload = json.loads(str(current.get("last_feedback", "")))
            self.assertEqual(
                feedback_payload.get("latest_decision_adoption_status"),
                personal_app.ADOPTION_STATUS_PENDING,
            )


if __name__ == "__main__":
    unittest.main()
