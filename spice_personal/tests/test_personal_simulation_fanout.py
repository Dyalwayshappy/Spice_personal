from __future__ import annotations

import unittest

from spice.decision import DecisionObjective
from spice.protocols import Decision, WorldState
from spice_personal.advisory.personal_advisory import (
    PERSONAL_SIMULATION_FANOUT_LIMIT_DEFAULT,
    PersonalLLMDecisionPolicy,
)
from spice_personal.wrappers.errors import WrapperIntegrationError


class _StubDecisionAdapter:
    def __init__(self, proposals: list[Decision]) -> None:
        self._proposals = list(proposals)

    def propose(
        self,
        state: WorldState,
        *,
        context: dict[str, object] | None = None,
        max_candidates: int | None = None,
    ) -> list[Decision]:
        del state, context
        proposals = list(self._proposals)
        if max_candidates is not None and max_candidates > 0:
            proposals = proposals[:max_candidates]
        return proposals


class _StubSimulationAdapter:
    def __init__(self, artifacts_by_action: dict[str, dict[str, object]]) -> None:
        self._artifacts_by_action = dict(artifacts_by_action)
        self.calls: list[str] = []

    def simulate(
        self,
        state: WorldState,
        *,
        decision: Decision | None = None,
        context: dict[str, object] | None = None,
    ) -> dict[str, object]:
        del state, context
        action = decision.selected_action if decision is not None and decision.selected_action else ""
        self.calls.append(action)
        payload = self._artifacts_by_action.get(action, {})
        return dict(payload) if isinstance(payload, dict) else {}


class PersonalSimulationFanoutTests(unittest.TestCase):
    def test_multiple_candidates_are_still_proposed(self) -> None:
        policy, _ = _build_policy(
            strict_model=False,
            simulation_fanout_limit=PERSONAL_SIMULATION_FANOUT_LIMIT_DEFAULT,
            artifacts_by_action={
                "personal.assistant.suggest": _valid_artifact(
                    "Option B better balances team stability, mentorship quality, and 3-year management growth.",
                    score=0.10,
                    action="personal.assistant.suggest",
                ),
            },
        )
        candidates = policy.propose(_world_state(), context=None)
        self.assertEqual(len(candidates), 3)
        self.assertEqual(
            [candidate.action for candidate in candidates],
            [
                "personal.assistant.suggest",
                "personal.assistant.ask_clarify",
                "personal.assistant.defer",
            ],
        )

    def test_select_simulates_only_top1_when_limit_is_1(self) -> None:
        policy, simulation_adapter = _build_policy(
            strict_model=False,
            simulation_fanout_limit=1,
            artifacts_by_action={
                "personal.assistant.suggest": _valid_artifact(
                    "Option B should lead because manager quality and team stability are stronger for management growth.",
                    score=0.15,
                    action="personal.assistant.suggest",
                ),
                "personal.assistant.ask_clarify": _valid_artifact(
                    "top-2 simulated",
                    score=0.99,
                    action="personal.assistant.ask_clarify",
                ),
                "personal.assistant.defer": _valid_artifact(
                    "top-3 simulated",
                    score=0.95,
                    action="personal.assistant.defer",
                ),
            },
        )
        candidates = policy.propose(_world_state(), context=None)
        selected = policy.select(candidates, DecisionObjective(), constraints=[])

        self.assertEqual(simulation_adapter.calls, ["personal.assistant.suggest"])
        self.assertEqual(selected.attributes.get("selected_candidate_id"), candidates[0].id)
        self.assertIn(
            "Option B should lead because manager quality and team stability are stronger for management growth.",
            str(selected.attributes.get("suggestion_text")),
        )

    def test_select_exposes_three_options_and_one_recommended(self) -> None:
        policy, _ = _build_policy(
            strict_model=False,
            simulation_fanout_limit=3,
            artifacts_by_action={
                "personal.assistant.suggest": _valid_artifact(
                    "Option A improves short-term compensation, but Option B better supports your management path.",
                    score=0.55,
                    action="personal.assistant.suggest",
                ),
                "personal.assistant.ask_clarify": _valid_artifact(
                    "top-2 simulated",
                    score=0.91,
                    action="personal.assistant.ask_clarify",
                ),
                "personal.assistant.defer": _valid_artifact(
                    "top-3 simulated",
                    score=0.72,
                    action="personal.assistant.defer",
                ),
            },
        )
        candidates = policy.propose(_world_state(), context=None)
        selected = policy.select(candidates, DecisionObjective(), constraints=[])

        options = selected.attributes.get("decision_options")
        self.assertIsInstance(options, list)
        self.assertEqual(len(options), 3)
        self.assertEqual(options[0].get("candidate_id"), candidates[1].id)
        self.assertEqual(options[0].get("action"), "personal.assistant.ask_clarify")
        self.assertEqual(options[0].get("suggestion_text"), "top-2 simulated")
        self.assertIn("benefits", options[0])
        self.assertIn("risks", options[0])
        self.assertIn("key_assumptions", options[0])
        self.assertIn("first_step_24h", options[0])
        self.assertIn("stop_loss_trigger", options[0])
        self.assertIn("change_mind_condition", options[0])
        self.assertEqual(selected.attributes.get("recommended_option_id"), candidates[1].id)
        self.assertEqual(selected.attributes.get("selected_candidate_id"), candidates[1].id)
        self.assertEqual(selected.attributes.get("suggestion_text"), "top-2 simulated")

    def test_select_avoids_incomplete_suggest_contract_when_alternative_is_available(self) -> None:
        policy, _ = _build_policy(
            strict_model=False,
            simulation_fanout_limit=3,
            artifacts_by_action={
                "personal.assistant.suggest": _valid_artifact(
                    "Reach out to a trusted friend before deciding.",
                    score=0.97,
                    action="personal.assistant.suggest",
                ),
                "personal.assistant.ask_clarify": _valid_artifact(
                    "clarify before recommending",
                    score=0.91,
                    action="personal.assistant.ask_clarify",
                ),
                "personal.assistant.defer": _valid_artifact(
                    "defer for one checkpoint",
                    score=0.85,
                    action="personal.assistant.defer",
                ),
            },
        )
        candidates = policy.propose(_world_state(), context=None)
        selected = policy.select(candidates, DecisionObjective(), constraints=[])

        self.assertEqual(selected.selected_action, "personal.assistant.ask_clarify")
        self.assertEqual(selected.attributes.get("recommended_option_id"), candidates[1].id)
        options = selected.attributes.get("decision_options")
        self.assertIsInstance(options, list)
        self.assertTrue(options)
        self.assertEqual(options[0].get("candidate_id"), selected.attributes.get("recommended_option_id"))

    def test_non_strict_select_marks_degraded_when_all_candidates_fail_entry_contract(self) -> None:
        policy, _ = _build_policy(
            strict_model=False,
            simulation_fanout_limit=3,
            artifacts_by_action={
                "personal.assistant.suggest": _valid_artifact(
                    "Talk to a friend before you decide.",
                    score=0.98,
                    action="personal.assistant.suggest",
                ),
                "personal.assistant.ask_clarify": {
                    "score": 0.90,
                    "confidence": 0.82,
                    "urgency": "high",
                    "suggestion_text": "Ask a question before deciding.",
                },
                "personal.assistant.defer": {
                    "score": 0.89,
                    "confidence": 0.70,
                    "urgency": "normal",
                    "suggestion_text": "Defer for now.",
                },
            },
        )
        candidates = policy.propose(_world_state(), context=None)
        selected = policy.select(candidates, DecisionObjective(), constraints=[])

        self.assertTrue(bool(selected.attributes.get("advisory_degraded")))
        degraded_reason = str(selected.attributes.get("degraded_reason", ""))
        self.assertIn("entry_contract_failed", degraded_reason)
        self.assertIsInstance(selected.attributes.get("decision_options"), list)
        self.assertTrue(str(selected.attributes.get("recommended_option_id", "")).strip())

    def test_strict_mode_with_fanout_limit_remains_strict_on_invalid_simulation(self) -> None:
        policy, simulation_adapter = _build_policy(
            strict_model=True,
            simulation_fanout_limit=1,
            artifacts_by_action={
                "personal.assistant.suggest": {"score": 0.42, "confidence": 0.5, "urgency": "normal"},
                "personal.assistant.ask_clarify": _valid_artifact(
                    "should not run",
                    score=0.99,
                    action="personal.assistant.ask_clarify",
                ),
            },
        )
        candidates = policy.propose(_world_state(), context=None)

        with self.assertRaises(WrapperIntegrationError) as raised:
            policy.select(candidates, DecisionObjective(), constraints=[])

        self.assertEqual(raised.exception.info.code, "model.response_validity")
        self.assertEqual(raised.exception.info.stage, "simulation_advise")
        self.assertEqual(simulation_adapter.calls, ["personal.assistant.suggest"])

    def test_complete_question_prefers_suggest_over_ask_clarify(self) -> None:
        policy, _ = _build_policy(
            strict_model=False,
            simulation_fanout_limit=3,
            artifacts_by_action={
                "personal.assistant.suggest": _valid_artifact(
                    (
                        "Option B is the stronger recommendation because it better aligns with the "
                        "3-year management goal while keeping team-volatility risk lower than Option A."
                    ),
                    score=0.88,
                    action="personal.assistant.suggest",
                ),
                "personal.assistant.ask_clarify": _valid_artifact(
                    "ask one more question before deciding",
                    score=0.95,
                    action="personal.assistant.ask_clarify",
                ),
                "personal.assistant.defer": _valid_artifact(
                    "defer briefly",
                    score=0.40,
                    action="personal.assistant.defer",
                ),
            },
        )
        candidates = policy.propose(
            _world_state(
                latest_question=(
                    "I have offer A and offer B. My goal is management in 3 years, risk tolerance is medium, "
                    "and minimum monthly cash flow is 50k."
                )
            ),
            context=None,
        )
        selected = policy.select(candidates, DecisionObjective(), constraints=[])

        self.assertEqual(selected.selected_action, "personal.assistant.suggest")
        self.assertEqual(selected.attributes.get("recommended_option_id"), candidates[0].id)

    def test_suggest_action_specific_payload_is_accepted(self) -> None:
        policy, _ = _build_policy(
            strict_model=False,
            simulation_fanout_limit=3,
            artifacts_by_action={
                "personal.assistant.suggest": {
                    "score": 0.86,
                    "confidence": 0.84,
                    "urgency": "normal",
                    "suggestion_text": (
                        "Option B better matches your management goal and medium risk preference than Option A."
                    ),
                    "action_specific": {
                        "benefits": [
                            "Mentorship and cross-team exposure increase management readiness.",
                        ],
                        "risks": [
                            "Short-term compensation is lower than Option A.",
                        ],
                        "key_assumptions": [
                            "Mentor commitments will be delivered within six months.",
                        ],
                        "first_step_24h": (
                            "Within 24h, confirm ownership scope and mentorship cadence for Option B."
                        ),
                        "stop_loss_trigger": (
                            "Reassess if ownership exposure is still unclear by week 4."
                        ),
                        "change_mind_condition": (
                            "Switch if Option A proves stable with explicit management path."
                        ),
                    },
                },
                "personal.assistant.ask_clarify": _valid_artifact(
                    "clarify before recommending",
                    score=0.50,
                    action="personal.assistant.ask_clarify",
                ),
                "personal.assistant.defer": _valid_artifact(
                    "defer briefly",
                    score=0.30,
                    action="personal.assistant.defer",
                ),
            },
        )
        candidates = policy.propose(
            _world_state(
                latest_question=(
                    "I have offer A and offer B. Goal is management in 3 years, medium risk tolerance, minimum cash flow 50k."
                )
            ),
            context=None,
        )
        selected = policy.select(candidates, DecisionObjective(), constraints=[])

        self.assertEqual(selected.selected_action, "personal.assistant.suggest")
        self.assertFalse(bool(selected.attributes.get("advisory_degraded")))
        self.assertTrue(str(selected.attributes.get("first_step_24h", "")).strip())
        self.assertTrue(str(selected.attributes.get("stop_loss_trigger", "")).strip())


def _build_policy(
    *,
    strict_model: bool,
    simulation_fanout_limit: int,
    artifacts_by_action: dict[str, dict[str, object]],
) -> tuple[PersonalLLMDecisionPolicy, _StubSimulationAdapter]:
    proposals = [
        Decision(
            id="dec-1",
            decision_type="personal.assistant.llm",
            status="proposed",
            selected_action="personal.assistant.suggest",
            attributes={"score": 0.10, "confidence": 0.50},
        ),
        Decision(
            id="dec-2",
            decision_type="personal.assistant.llm",
            status="proposed",
            selected_action="personal.assistant.ask_clarify",
            attributes={"score": 0.20, "confidence": 0.40},
        ),
        Decision(
            id="dec-3",
            decision_type="personal.assistant.llm",
            status="proposed",
            selected_action="personal.assistant.defer",
            attributes={"score": 0.30, "confidence": 0.30},
        ),
    ]
    simulation_adapter = _StubSimulationAdapter(artifacts_by_action)
    policy = PersonalLLMDecisionPolicy(
        decision_adapter=_StubDecisionAdapter(proposals),
        simulation_adapter=simulation_adapter,
        allowed_actions=(
            "personal.assistant.suggest",
            "personal.assistant.ask_clarify",
            "personal.assistant.defer",
        ),
        strict_model=strict_model,
        simulation_fanout_limit=simulation_fanout_limit,
    )
    return policy, simulation_adapter


def _world_state(*, latest_question: str = "") -> WorldState:
    entities: dict[str, object] = {}
    if latest_question:
        entities = {
            "personal.assistant.current": {
                "status": "ready",
                "latest_question": latest_question,
                "urgency": "normal",
                "confidence": 0.0,
            }
        }
    return WorldState(id="state-personal-fanout", entities=entities)


def _valid_artifact(
    suggestion_text: str,
    *,
    score: float,
    action: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "score": score,
        "confidence": 0.8,
        "urgency": "normal",
        "suggestion_text": suggestion_text,
        "simulation_rationale": "test",
    }
    if action == "personal.assistant.ask_clarify":
        payload["clarifying_questions"] = [
            {
                "question": "What is your top non-negotiable?",
                "why": "It can change option ranking directly.",
            },
            {
                "question": "How much downside can you tolerate in 12 months?",
                "why": "It can reorder high-upside and stable options.",
            },
            {
                "question": "What 3-year outcome matters most?",
                "why": "It can change which option aligns with your target.",
            },
        ]
        return payload
    if action == "personal.assistant.defer":
        payload["defer_plan"] = {
            "revisit_at": "7 days",
            "monitor_signal": "new verified signal on team stability",
            "resume_trigger": "resume when a verified signal arrives",
        }
        return payload
    if action == "personal.assistant.gather_evidence":
        payload["evidence_plan"] = [
            {
                "fact": "Verify attrition trend for each option.",
                "why": "Attrition changes execution risk ranking.",
            },
            {
                "fact": "Validate manager mentorship track record.",
                "why": "Mentorship quality changes promotion-path odds.",
            },
            {
                "fact": "Confirm scope ownership expectations in first 6 months.",
                "why": "Ownership scope changes management readiness.",
            },
        ]
        return payload
    payload["benefits"] = ["Keeps momentum while preserving a clear objective fit."]
    payload["risks"] = ["Could miss upside if market conditions change rapidly."]
    payload["key_assumptions"] = ["Decision constraints remain stable for the next quarter."]
    payload["first_step_24h"] = "Within 24h, confirm role scope and manager expectations for both options."
    payload["stop_loss_trigger"] = "If scope clarity remains missing after one week, pause and re-evaluate."
    payload["change_mind_condition"] = "Switch if verified stability and growth signals materially reverse."
    return payload


if __name__ == "__main__":
    unittest.main()
