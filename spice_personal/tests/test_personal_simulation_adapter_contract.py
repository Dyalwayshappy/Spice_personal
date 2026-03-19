from __future__ import annotations

import unittest

from spice.protocols import WorldState
from spice_personal.advisory.personal_simulation_adapter import _build_personal_prompt


class PersonalSimulationAdapterContractTests(unittest.TestCase):
    def test_personal_prompt_keeps_action_specific_contract(self) -> None:
        prompt = _build_personal_prompt(
            state=WorldState(id="state-personal-sim-contract"),
            decision=None,
            intent=None,
            context={"domain": "personal.assistant"},
        )

        self.assertIn("You are SPICE Personal Advisor. Speak directly to the user.", prompt)
        self.assertIn("Action-specific requirements:", prompt)
        self.assertIn("personal.assistant.suggest", prompt)
        self.assertIn("personal.assistant.ask_clarify", prompt)
        self.assertIn("personal.assistant.gather_evidence", prompt)
        self.assertIn("personal.assistant.defer", prompt)
        self.assertIn("decision_brain_report object with exactly 3 options", prompt)
        self.assertIn("Forbidden evidence subjects", prompt)
        self.assertIn("externally verifiable", prompt)

    def test_personal_prompt_includes_personal_entity_snapshot(self) -> None:
        prompt = _build_personal_prompt(
            state=WorldState(
                id="state-personal-sim-entity",
                entities={
                    "personal.assistant.current": {
                        "latest_question": "I have offer A and offer B, which one should I pick?",
                        "evidence_summary": "Offer A has higher pay; Offer B has better stability.",
                        "urgency": "high",
                    }
                },
            ),
            decision=None,
            intent=None,
            context={"domain": "personal.assistant"},
        )

        self.assertIn('"entities": {', prompt)
        self.assertIn('"personal.assistant.current"', prompt)
        self.assertIn('"latest_question": "I have offer A and offer B, which one should I pick?"', prompt)
        self.assertIn('"evidence_summary": "Offer A has higher pay; Offer B has better stability."', prompt)


if __name__ == "__main__":
    unittest.main()
