from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from spice.decision import DecisionObjective
from spice.llm.adapters import LLMSimulationAdapter
from spice.llm.core import LLMClient, LLMModelConfig, LLMRouter, LLMTaskHook, ProviderRegistry
from spice.llm.providers import DeterministicLLMProvider
from spice.protocols import Decision, WorldState
from spice_personal.advisory import personal_advisory as advisory
from spice_personal.advisory.personal_advisory import PersonalLLMDecisionPolicy
from spice_personal.wrappers.errors import WrapperIntegrationError


class _StubDecisionAdapter:
    def propose(
        self,
        state: WorldState,
        *,
        context: dict[str, object] | None = None,
        max_candidates: int | None = None,
    ) -> list[Decision]:
        del state, context, max_candidates
        return [
            Decision(
                id="dec-sim-debug-1",
                decision_type="personal.assistant.llm",
                status="proposed",
                selected_action="personal.assistant.suggest",
                attributes={"score": 0.5, "confidence": 0.5},
            )
        ]


class PersonalSimulationDebugArtifactTests(unittest.TestCase):
    def test_simulation_response_validity_failure_writes_debug_artifact(self) -> None:
        simulation_adapter = _build_simulation_adapter(
            timeout_seconds=12.5,
            simulation_response=json.dumps(
                {
                    "score": 0.42,
                    "confidence": 0.62,
                    "urgency": "normal",
                },
                ensure_ascii=True,
            ),
        )
        policy = PersonalLLMDecisionPolicy(
            decision_adapter=_StubDecisionAdapter(),
            simulation_adapter=simulation_adapter,
            allowed_actions=("personal.assistant.suggest",),
            strict_model=True,
            simulation_fanout_limit=1,
        )
        state = WorldState(id="state-sim-debug")

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifacts_dir = Path(tmp_dir)
            previous_debug_dir = advisory.PERSONAL_DEBUG_MODEL_IO_DIR
            previous_env = os.environ.get(advisory.PERSONAL_DEBUG_MODEL_IO_ENV)
            advisory.PERSONAL_DEBUG_MODEL_IO_DIR = artifacts_dir
            os.environ[advisory.PERSONAL_DEBUG_MODEL_IO_ENV] = "1"
            try:
                candidates = policy.propose(state, context=None)
                with self.assertRaises(WrapperIntegrationError) as raised:
                    policy.select(candidates, DecisionObjective(), constraints=[])
            finally:
                advisory.PERSONAL_DEBUG_MODEL_IO_DIR = previous_debug_dir
                if previous_env is None:
                    os.environ.pop(advisory.PERSONAL_DEBUG_MODEL_IO_ENV, None)
                else:
                    os.environ[advisory.PERSONAL_DEBUG_MODEL_IO_ENV] = previous_env

            self.assertEqual(raised.exception.info.code, "model.response_validity")
            self.assertEqual(raised.exception.info.stage, "simulation_advise")

            files = sorted(artifacts_dir.glob("model_debug_*.json"))
            self.assertEqual(len(files), 1)
            payload = json.loads(files[0].read_text(encoding="utf-8"))
            self.assertEqual(payload.get("stage"), "simulation_advise")
            self.assertIn('"score": 0.42', payload.get("raw_stdout", ""))
            self.assertEqual(payload.get("raw_stderr"), "")
            self.assertEqual(payload.get("candidate_id"), "dec-sim-debug-1")
            self.assertEqual(payload.get("selected_action"), "personal.assistant.suggest")
            self.assertEqual(payload.get("timeout_seconds"), 12.5)


def _build_simulation_adapter(
    *,
    timeout_seconds: float,
    simulation_response: str,
) -> LLMSimulationAdapter:
    provider = DeterministicLLMProvider(
        responses={LLMTaskHook.SIMULATION_ADVISE: simulation_response}
    )
    registry = ProviderRegistry.empty().register(provider)
    cfg = LLMModelConfig(
        provider_id="deterministic",
        model_id="deterministic.sim.debug",
        timeout_sec=timeout_seconds,
    )
    router = LLMRouter(
        hook_defaults={LLMTaskHook.SIMULATION_ADVISE: cfg}
    )
    return LLMSimulationAdapter(client=LLMClient(registry=registry, router=router))


if __name__ == "__main__":
    unittest.main()
