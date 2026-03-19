from __future__ import annotations

import unittest

from spice.protocols import Decision, ExecutionIntent, ExecutionResult
from spice_personal.execution.evidence_round import (
    PERSONAL_ACTION_GATHER_EVIDENCE,
    PERSONAL_OBSERVATION_EVIDENCE_CHECKLIST_PREPARED,
    normalize_execution_result_to_evidence_observation,
    run_mock_evidence_round,
)


class EvidenceRoundTests(unittest.TestCase):
    def test_normalization_outputs_structured_evidence_items(self) -> None:
        decision = Decision(
            id="dec-evidence-1",
            decision_type="personal.assistant.llm",
            status="proposed",
            selected_action=PERSONAL_ACTION_GATHER_EVIDENCE,
        )
        intent = ExecutionIntent(
            id="intent-evidence-1",
            intent_type="personal.assistant.execution",
            status="planned",
            operation={"name": "personal.gather_evidence", "mode": "sync", "dry_run": False},
            input_payload={},
            parameters={},
        )
        result = ExecutionResult(
            id="result-evidence-1",
            status="success",
            executor="external-agent",
            output={
                "summary": "Evidence snapshot from external sources.",
                "items": [
                    {
                        "claim": "Team A attrition reached 28% in the last 12 months.",
                        "source": "Company annual filing",
                        "title": "2025 annual report",
                        "url": "https://example.test/report",
                        "published_at": "2025-12-01",
                        "reliability": 0.90,
                    },
                    {
                        "text": "Mentorship quality has mixed signals.",
                    },
                ],
            },
        )

        observation = normalize_execution_result_to_evidence_observation(
            decision=decision,
            intent=intent,
            execution_result=result,
            execution_outcome=None,
            source="tests.evidence",
        )

        items = observation.attributes.get("evidence_items")
        self.assertIsInstance(items, list)
        self.assertEqual(len(items), 2)
        first = items[0]
        self.assertEqual(first.get("claim"), "Team A attrition reached 28% in the last 12 months.")
        self.assertEqual(first.get("source"), "Company annual filing")
        self.assertEqual(first.get("title"), "2025 annual report")
        self.assertEqual(first.get("url"), "https://example.test/report")
        self.assertEqual(first.get("published_at"), "2025-12-01")
        self.assertEqual(first.get("date"), "2025-12-01")
        self.assertGreaterEqual(float(first.get("reliability", 0.0)), 0.85)
        self.assertGreaterEqual(float(first.get("confidence", 0.0)), 0.85)
        second = items[1]
        self.assertLessEqual(float(second.get("reliability", 1.0)), 0.35)
        self.assertLessEqual(float(second.get("confidence", 1.0)), 0.35)
        self.assertEqual(observation.attributes.get("evidence_mode"), "external_execution")
        self.assertEqual(observation.attributes.get("evidence_source_coverage"), 0.5)

    def test_mock_evidence_round_produces_manual_checklist_observation(self) -> None:
        decision = Decision(
            id="dec-evidence-2",
            decision_type="personal.assistant.llm",
            status="proposed",
            selected_action=PERSONAL_ACTION_GATHER_EVIDENCE,
            attributes={
                "evidence_plan": [
                    {
                        "fact": "Verify attrition trend for each option over 12 months.",
                        "why": "Attrition changes execution risk ranking.",
                    },
                    {
                        "fact": "Validate manager coaching outcomes from direct references.",
                        "why": "Coaching quality changes management-path probability.",
                    },
                    {
                        "fact": "Confirm first-6-month scope ownership expectations.",
                        "why": "Ownership scope affects promotion readiness.",
                    },
                ]
            },
        )

        round_result = run_mock_evidence_round(
            decision=decision,
            source="tests.evidence.manual",
        )

        self.assertTrue(round_result.requested)
        self.assertIsNotNone(round_result.evidence_observation)
        observation = round_result.evidence_observation
        self.assertEqual(
            observation.observation_type,
            PERSONAL_OBSERVATION_EVIDENCE_CHECKLIST_PREPARED,
        )
        self.assertEqual(observation.attributes.get("evidence_mode"), "manual_checklist")
        items = observation.attributes.get("evidence_items")
        self.assertIsInstance(items, list)
        self.assertEqual(len(items), 3)
        for item in items:
            self.assertTrue(str(item.get("claim", "")).strip())
            self.assertEqual(str(item.get("source", "")), "")
            self.assertLessEqual(float(item.get("confidence", 1.0)), 0.35)
        self.assertLessEqual(float(observation.attributes.get("evidence_confidence", 1.0)), 0.25)

    def test_mock_evidence_round_filters_internal_runtime_checklist_items(self) -> None:
        decision = Decision(
            id="dec-evidence-3",
            decision_type="personal.assistant.llm",
            status="proposed",
            selected_action=PERSONAL_ACTION_GATHER_EVIDENCE,
            attributes={
                "evidence_plan": [
                    {
                        "fact": "Verify question_received signal aligns with selected_action hypothesis.",
                        "why": "It validates session protocol consistency.",
                    },
                    {
                        "fact": "Cross-check checklist timestamp against current state.",
                        "why": "It ensures worldstate readiness.",
                    },
                    {
                        "fact": "Confirm active session risk_budget constraints.",
                        "why": "It checks complexity requirements.",
                    },
                ]
            },
        )

        round_result = run_mock_evidence_round(
            decision=decision,
            source="tests.evidence.manual",
        )

        self.assertTrue(round_result.requested)
        self.assertIsNotNone(round_result.evidence_observation)
        observation = round_result.evidence_observation
        items = observation.attributes.get("evidence_items")
        self.assertIsInstance(items, list)
        self.assertEqual(len(items), 3)
        combined_claims = " ".join(str(item.get("claim", "")) for item in items).lower()
        self.assertNotIn("question_received", combined_claims)
        self.assertNotIn("signal", combined_claims)
        self.assertIn("attrition", combined_claims)


if __name__ == "__main__":
    unittest.main()
