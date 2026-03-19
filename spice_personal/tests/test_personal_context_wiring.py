from __future__ import annotations

import unittest

from spice.decision import DecisionObjective
from spice.protocols import Decision, WorldState
from spice_personal.advisory.personal_advisory import PersonalLLMDecisionPolicy


class _CaptureDecisionAdapter:
    def __init__(self) -> None:
        self.last_context: dict[str, object] | None = None

    def propose(
        self,
        state: WorldState,
        *,
        context: dict[str, object] | None = None,
        max_candidates: int | None = None,
    ) -> list[Decision]:
        del state, max_candidates
        self.last_context = dict(context) if isinstance(context, dict) else None
        return [
            Decision(
                id="dec-context-1",
                decision_type="personal.assistant.llm",
                status="proposed",
                selected_action="personal.assistant.suggest",
                attributes={"score": 0.82, "confidence": 0.88, "urgency": "high"},
            )
        ]


class _CaptureSimulationAdapter:
    def __init__(self) -> None:
        self.last_contexts: list[dict[str, object]] = []

    def simulate(
        self,
        state: WorldState,
        *,
        decision: Decision | None = None,
        context: dict[str, object] | None = None,
    ) -> dict[str, object]:
        del state, decision
        self.last_contexts.append(dict(context) if isinstance(context, dict) else {})
        return {
            "score": 0.93,
            "confidence": 0.88,
            "urgency": "high",
            "suggestion_text": (
                "Option B is the better recommendation because it preserves management-growth momentum "
                "while reducing team-volatility downside compared with Option A."
            ),
            "benefits": ["Stronger mentorship and lower execution volatility for management growth."],
            "risks": ["Lower short-term salary compared with Option A."],
            "key_assumptions": ["Mentorship quality remains high during the first 12 months."],
            "first_step_24h": (
                "Within 24h, confirm first-quarter ownership scope and mentorship cadence for Offer B."
            ),
            "stop_loss_trigger": (
                "Reopen options if the promised ownership scope is still undefined by week 4."
            ),
            "change_mind_condition": (
                "Switch to Option A if verified stability and management-path evidence materially improve."
            ),
            "simulation_rationale": "context_wiring_test",
        }


class PersonalContextWiringTests(unittest.TestCase):
    def test_propose_passes_latest_question_and_profile_into_decision_context(self) -> None:
        decision_adapter = _CaptureDecisionAdapter()
        simulation_adapter = _CaptureSimulationAdapter()
        policy = PersonalLLMDecisionPolicy(
            decision_adapter=decision_adapter,
            simulation_adapter=simulation_adapter,
            allowed_actions=(
                "personal.assistant.suggest",
                "personal.assistant.ask_clarify",
                "personal.assistant.defer",
            ),
            strict_model=True,
        )

        state = _state_with_question()
        _ = policy.propose(state, context={"trace_id": "ctx-123"})
        context = decision_adapter.last_context
        self.assertIsInstance(context, dict)
        assert isinstance(context, dict)
        self.assertEqual(context.get("trace_id"), "ctx-123")
        self.assertEqual(
            context.get("latest_question"),
            "I have offer A and offer B; goal is management in 3 years with medium risk tolerance.",
        )
        self.assertIsInstance(context.get("question_profile"), dict)
        self.assertIsInstance(context.get("personal_entity"), dict)

    def test_select_passes_latest_question_and_profile_into_simulation_context(self) -> None:
        decision_adapter = _CaptureDecisionAdapter()
        simulation_adapter = _CaptureSimulationAdapter()
        policy = PersonalLLMDecisionPolicy(
            decision_adapter=decision_adapter,
            simulation_adapter=simulation_adapter,
            allowed_actions=("personal.assistant.suggest",),
            strict_model=True,
        )

        state = _state_with_question()
        candidates = policy.propose(state, context=None)
        selected = policy.select(candidates, DecisionObjective(), constraints=[])

        self.assertEqual(selected.selected_action, "personal.assistant.suggest")
        self.assertTrue(simulation_adapter.last_contexts)
        context = simulation_adapter.last_contexts[0]
        self.assertEqual(
            context.get("latest_question"),
            "I have offer A and offer B; goal is management in 3 years with medium risk tolerance.",
        )
        self.assertIsInstance(context.get("question_profile"), dict)
        personal_entity = context.get("personal_entity")
        self.assertIsInstance(personal_entity, dict)
        assert isinstance(personal_entity, dict)
        self.assertEqual(
            personal_entity.get("evidence_summary"),
            "Offer A has better pay, Offer B has stronger manager stability.",
        )


def _state_with_question() -> WorldState:
    return WorldState(
        id="state-context-wiring",
        entities={
            "personal.assistant.current": {
                "status": "ready",
                "latest_question": (
                    "I have offer A and offer B; goal is management in 3 years with medium risk tolerance."
                ),
                "evidence_summary": "Offer A has better pay, Offer B has stronger manager stability.",
                "urgency": "high",
                "confidence": 0.5,
            }
        },
    )


if __name__ == "__main__":
    unittest.main()
