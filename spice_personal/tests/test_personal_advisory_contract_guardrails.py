from __future__ import annotations

import unittest

from spice_personal.advisory.personal_advisory import (
    _evaluate_action_entry_assessment,
    _extract_clarifying_questions,
    _extract_defer_plan,
    _extract_evidence_plan,
    _suggest_generic_reasons,
)


class PersonalAdvisoryContractGuardrailsTests(unittest.TestCase):
    def test_missing_action_fields_are_not_silently_filled(self) -> None:
        self.assertEqual(
            _extract_clarifying_questions({}, question="Should I choose offer A or B?"),
            (),
        )
        self.assertEqual(
            _extract_evidence_plan({}, question="Should I choose offer A or B?"),
            (),
        )
        self.assertEqual(_extract_defer_plan({}), {})

    def test_generic_suggest_detection_checks_multiple_signals(self) -> None:
        reasons = _suggest_generic_reasons(
            {
                "suggestion_text": "Talk to a friend before deciding.",
                "benefits": ["Get perspective."],
                "risks": ["Could still feel uncertain."],
                "key_assumptions": ["A conversation will help."],
                "first_step_24h": "Reach out soon.",
                "stop_loss_trigger": "",
                "change_mind_condition": "",
            },
            question="I have offer A vs offer B and want management growth in 3 years.",
        )
        self.assertIn("generic_suggestion_text", reasons)
        self.assertIn("generic_missing_time_action", reasons)
        self.assertIn("generic_missing_stop_loss", reasons)
        self.assertIn("generic_missing_change_mind", reasons)

    def test_specific_suggest_passes_generic_checks(self) -> None:
        reasons = _suggest_generic_reasons(
            {
                "suggestion_text": (
                    "Option B better fits your 3-year management goal because verified mentorship "
                    "quality is higher and attrition risk is lower."
                ),
                "benefits": ["Mentor support increases management readiness."],
                "risks": ["Short-term salary upside is lower than Option A."],
                "key_assumptions": ["Mentorship remains stable for 12 months."],
                "first_step_24h": "Within 24h, confirm first-quarter ownership scope with Option B manager.",
                "stop_loss_trigger": "If ownership scope is unclear by week 4, reopen Option A.",
                "change_mind_condition": "Switch if verified stability and promotion-path data favor Option A.",
            },
            question="I have offer A vs offer B and want management growth in 3 years.",
        )
        self.assertEqual(reasons, [])

    def test_structured_report_counts_as_tradeoff_signal(self) -> None:
        reasons = _suggest_generic_reasons(
            {
                "suggestion_text": (
                    "Option B is recommended because it better aligns with your 3-year management goal."
                ),
                "benefits": [],
                "risks": [],
                "key_assumptions": [],
                "first_step_24h": "",
                "stop_loss_trigger": "",
                "change_mind_condition": "",
                "decision_brain_report": {
                    "options": [
                        {
                            "label": "Option A",
                            "benefits": ["Higher compensation in the short term."],
                            "risks": ["Manager volatility can block management development."],
                            "key_assumptions": ["Turnover stabilizes quickly."],
                            "first_step_24h": "Within 24h, verify manager stability signals.",
                            "stop_loss_trigger": "Reassess if manager turnover remains high in first 90 days.",
                        },
                        {
                            "label": "Option B",
                            "benefits": ["Mentorship and stable org accelerate management readiness."],
                            "risks": ["Lower salary in the short term."],
                            "key_assumptions": ["Mentor commitment is real and sustained."],
                            "first_step_24h": "Within 24h, confirm mentorship cadence and cross-team scope.",
                            "stop_loss_trigger": "Reassess if no ownership exposure appears by month 6.",
                        },
                        {
                            "label": "Option C",
                            "benefits": ["Preserves negotiating leverage."],
                            "risks": ["Negotiation may fail or slow decision velocity."],
                            "key_assumptions": ["Employer remains flexible on package terms."],
                            "first_step_24h": "Within 24h, present a bounded counter-offer package.",
                            "stop_loss_trigger": "Reassess if critical terms remain unchanged after negotiation round 1.",
                        },
                    ],
                    "recommended_option_label": "Option B",
                    "recommendation_reason": "Best balance of management growth path and stability.",
                    "what_would_change_my_mind": "Switch if Option A proves stable with concrete management path evidence.",
                },
            },
            question="I have offer A vs offer B and want management growth in 3 years.",
        )
        self.assertNotIn("generic_missing_tradeoff", reasons)
        self.assertNotIn("generic_missing_time_action", reasons)

    def test_gather_evidence_rejects_internal_runtime_semantics(self) -> None:
        assessment = _evaluate_action_entry_assessment(
            action="personal.assistant.gather_evidence",
            advisory={
                "confidence": 0.72,
                "evidence_plan": [
                    {
                        "fact": "Verify question_received signal content aligns with selected_action hypothesis.",
                        "why": "It validates protocol consistency before recommendation ranking.",
                    },
                    {
                        "fact": "Cross-check evidence_checklist_prepared timestamp with current session state.",
                        "why": "It avoids stale runtime state before ranking alternatives.",
                    },
                    {
                        "fact": "Confirm worldstate risk_budget alignment for active session.",
                        "why": "It ensures the current state passes complexity requirements.",
                    },
                ],
            },
            question=(
                "I have offer A and offer B. Goal is management in 3 years and risk tolerance is medium."
            ),
        )
        self.assertFalse(bool(assessment.get("passes")))
        reasons = assessment.get("reasons")
        self.assertIsInstance(reasons, list)
        self.assertIn("evidence_internal_runtime_semantics", reasons)

    def test_gather_evidence_requires_question_entity_binding(self) -> None:
        assessment = _evaluate_action_entry_assessment(
            action="personal.assistant.gather_evidence",
            advisory={
                "confidence": 0.72,
                "evidence_plan": [
                    {
                        "fact": "Confirm whether public macro indicators changed this quarter.",
                        "why": "It could change recommendation ranking.",
                    },
                    {
                        "fact": "Compare average market sentiment reports from last month.",
                        "why": "It may affect option ranking confidence.",
                    },
                    {
                        "fact": "Validate industry benchmark volatility assumptions.",
                        "why": "It impacts recommendation direction.",
                    },
                ],
            },
            question=(
                "I have offer A and offer B. Goal is management in 3 years and risk tolerance is medium."
            ),
        )
        self.assertFalse(bool(assessment.get("passes")))
        reasons = assessment.get("reasons")
        self.assertIsInstance(reasons, list)
        self.assertIn("evidence_missing_user_entity_binding", reasons)

    def test_gather_evidence_accepts_real_world_offer_facts(self) -> None:
        assessment = _evaluate_action_entry_assessment(
            action="personal.assistant.gather_evidence",
            advisory={
                "confidence": 0.72,
                "evidence_plan": [
                    {
                        "fact": "Confirm Offer A team manager turnover count in the past 12 months.",
                        "why": "It can change risk ranking between Offer A and Offer B.",
                    },
                    {
                        "fact": "Verify Offer B mentorship commitments in writing from the hiring manager.",
                        "why": "It changes management-path probability and recommendation direction.",
                    },
                    {
                        "fact": "Compare Offer A and Offer B guaranteed cash compensation and variable components.",
                        "why": "It can reorder recommendation when downside tolerance is medium.",
                    },
                ],
            },
            question=(
                "I have offer A and offer B. Goal is management in 3 years and risk tolerance is medium."
            ),
        )
        self.assertTrue(bool(assessment.get("passes")))


if __name__ == "__main__":
    unittest.main()
