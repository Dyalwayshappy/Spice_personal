from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from spice.llm.core import LLMClient, LLMModelConfigOverride, LLMRequest, LLMTaskHook
from spice.llm.simulation import SimulationModel
from spice.llm.util import extract_first_json_object, strip_markdown_fences
from spice.protocols import Decision, ExecutionIntent, WorldState

_MODEL_STDOUT_ATTR = "_spice_model_stdout"
_MODEL_STDERR_ATTR = "_spice_model_stderr"


@dataclass(slots=True)
class PersonalLLMSimulationAdapter(SimulationModel):
    client: LLMClient
    model_override: LLMModelConfigOverride | None = None
    _last_model_stdout: str = field(default="", init=False, repr=False)
    _last_model_stderr: str = field(default="", init=False, repr=False)
    _last_timeout_seconds: float | None = field(default=None, init=False, repr=False)

    def simulate(
        self,
        state: WorldState,
        *,
        decision: Decision | None = None,
        intent: ExecutionIntent | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._last_model_stdout = ""
        self._last_model_stderr = ""
        self._last_timeout_seconds = None
        request = LLMRequest(
            task_hook=LLMTaskHook.SIMULATION_ADVISE,
            domain=_domain_from_context(context),
            input_text=_build_personal_prompt(
                state=state,
                decision=decision,
                intent=intent,
                context=context,
            ),
            response_format_hint="json_object",
            metadata={"state_id": state.id},
        )
        self._last_timeout_seconds = _resolve_timeout_seconds(
            client=self.client,
            request=request,
            model_override=self.model_override,
        )
        response = self.client.generate(request, model_override=self.model_override)
        self._last_model_stdout, self._last_model_stderr = _extract_model_io_from_response(response)
        try:
            payload = _parse_json_object(response.output_text)
            return payload
        except Exception as exc:
            _attach_model_io(
                exc,
                stdout=self._last_model_stdout,
                stderr=self._last_model_stderr,
            )
            raise


def _build_personal_prompt(
    *,
    state: WorldState,
    decision: Decision | None,
    intent: ExecutionIntent | None,
    context: dict[str, Any] | None,
) -> str:
    payload = {
        "state": {
            "id": state.id,
            "status": state.status,
            "resources": state.resources,
            "signals": state.signals,
            "risks": state.risks,
            "entities": _state_entities_snapshot(state),
        },
        "decision": _decision_payload(decision),
        "intent": _intent_payload(intent),
        "context": context or {},
    }
    return (
        "You are SPICE Personal Advisor. Speak directly to the user.\n"
        "You are SPICE Personal Decision Brain.\n"
        "You are not a system evaluator.\n"
        "Task: using the JSON input below, generate an option-level decision artifact for candidate evaluation before execution.\n"
        "This is the simulation advice stage for candidate evaluation before execution.\n"
        "This is not generic coaching. Ground output in the user's concrete tradeoffs, constraints, and goals.\n"
        "Use the user's dominant language from latest_question when possible.\n"
        "JSON object only.\n"
        "Required top-level fields: suggestion_text (string), score (number), confidence (number), urgency (string).\n"
        "Normalize score to [0.0, 1.0].\n"
        "suggestion_text must open with a clear recommendation and one-sentence reason tied to user goals/constraints.\n"
        "suggestion_text should read like real decision reasoning, not a template or field dump.\n"
        "Forbidden in suggestion_text: response/system/model/prompt/instruction/policy/process commentary.\n"
        "Forbidden in suggestion_text: generic template advice with no concrete action.\n"
        "Forbidden in suggestion_text: ask user to talk to friend/family/colleague for generic perspective.\n"
        "Action-specific requirements:\n"
        "- For personal.assistant.suggest: include benefits (non-empty list), risks (non-empty list), key_assumptions (non-empty list), first_step_24h (non-empty string), stop_loss_trigger (non-empty string), change_mind_condition (non-empty string).\n"
        "- For personal.assistant.suggest: include decision_brain_report object with exactly 3 options when alternatives are present.\n"
        "- For each decision_brain_report option include: label, option_rank (1-based), option_positioning (one-sentence judgement), benefits, risks, key_assumptions, first_step_24h, stop_loss_trigger, change_mind_condition.\n"
        "- decision_brain_report must include recommended_option_label, recommendation_reason, what_would_change_my_mind.\n"
        "- In options, avoid homogeneous bullet dumps; keep each option internally coherent as a natural tradeoff block.\n"
        "- For personal.assistant.ask_clarify: include clarifying_questions as exactly 3 items; each item must contain non-empty question and non-empty why explaining how the answer could change option ranking.\n"
        "- For personal.assistant.gather_evidence: include evidence_plan as exactly 3 items.\n"
        "- For each evidence_plan item: fact must be real-world and externally verifiable, tied to concrete entities from the user's question (for example A/B offers, salary, manager, mentor, team stability, attrition, promotion path, cashflow).\n"
        "- For each evidence_plan item: why must explain how this fact would change option ranking or recommendation direction.\n"
        "- For personal.assistant.gather_evidence: do NOT output runtime self-check items.\n"
        "- Forbidden evidence subjects: signal, observation, session, state, worldstate, timestamp, checklist status, selected_action, question_received, protocol validation, complexity requirements.\n"
        "- If an evidence item does not reference user-question entities, it is invalid.\n"
        "- For personal.assistant.defer: include defer_plan object with revisit_at, monitor_signal, and resume_trigger.\n"
        "When user context includes alternatives (for example A/B), explicitly reference those alternatives in reasoning.\n"
        "No markdown.\n"
        "No prose outside the JSON object.\n"
        "Non-JSON output is invalid.\n"
        "Missing required fields means the response is invalid.\n"
        + json.dumps(payload, ensure_ascii=True, sort_keys=True)
    )


def _decision_payload(decision: Decision | None) -> dict[str, Any] | None:
    if decision is None:
        return None
    return {
        "id": decision.id,
        "decision_type": decision.decision_type,
        "status": decision.status,
        "selected_action": decision.selected_action,
        "refs": decision.refs,
        "metadata": decision.metadata,
        "attributes": decision.attributes,
    }


def _intent_payload(intent: ExecutionIntent | None) -> dict[str, Any] | None:
    if intent is None:
        return None
    return {
        "id": intent.id,
        "intent_type": intent.intent_type,
        "status": intent.status,
        "executor_type": intent.executor_type,
        "target": intent.target,
        "operation": intent.operation,
        "input_payload": intent.input_payload,
        "parameters": intent.parameters,
        "provenance": intent.provenance,
        "refs": intent.refs,
    }


def _domain_from_context(context: dict[str, Any] | None) -> str | None:
    if context is None:
        return None
    value = context.get("domain")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _parse_json_object(text: str) -> dict[str, Any]:
    normalized = strip_markdown_fences(text)
    candidate = extract_first_json_object(normalized)
    if candidate is None:
        raise ValueError("No JSON object found in response.")
    payload = json.loads(candidate)
    if not isinstance(payload, dict):
        raise ValueError("Simulation adapter expected JSON object response.")
    return payload


def _extract_model_io_from_response(response: Any) -> tuple[str, str]:
    raw_payload = response.raw_payload if isinstance(getattr(response, "raw_payload", None), dict) else {}
    stdout = raw_payload.get("stdout")
    stderr = raw_payload.get("stderr")
    normalized_stdout = stdout if isinstance(stdout, str) else response.output_text
    normalized_stderr = stderr if isinstance(stderr, str) else ""
    return normalized_stdout, normalized_stderr


def _attach_model_io(exc: Exception, *, stdout: str, stderr: str) -> None:
    existing_stdout = getattr(exc, _MODEL_STDOUT_ATTR, "")
    existing_stderr = getattr(exc, _MODEL_STDERR_ATTR, "")
    try:
        if not isinstance(existing_stdout, str) or not existing_stdout:
            setattr(exc, _MODEL_STDOUT_ATTR, stdout if isinstance(stdout, str) else "")
        if not isinstance(existing_stderr, str) or not existing_stderr:
            setattr(exc, _MODEL_STDERR_ATTR, stderr if isinstance(stderr, str) else "")
    except Exception:
        return


def _resolve_timeout_seconds(
    *,
    client: LLMClient,
    request: LLMRequest,
    model_override: LLMModelConfigOverride | None,
) -> float | None:
    try:
        model_config = client.resolve_model_config(
            request.task_hook,
            domain=request.domain,
            model_override=model_override,
        )
    except Exception:
        return None

    timeout = getattr(model_config, "timeout_sec", None)
    if isinstance(timeout, (int, float)):
        return float(timeout)
    return None


def _state_entities_snapshot(state: WorldState) -> dict[str, Any]:
    entities = state.entities if isinstance(state.entities, dict) else {}
    personal_entity = entities.get("personal.assistant.current")
    if not isinstance(personal_entity, dict):
        return {}
    snapshot: dict[str, Any] = {}
    for key in (
        "status",
        "latest_question",
        "latest_suggestion",
        "evidence_summary",
        "urgency",
        "confidence",
        "last_feedback",
    ):
        if key in personal_entity:
            snapshot[key] = personal_entity.get(key)
    return {"personal.assistant.current": snapshot}
