from __future__ import annotations

"""Personal advisory LLM orchestration.

This module keeps personal-mode LLM decision/simulation wiring thin:
- proposal normalization
- action filtering
- simulation-based scoring and field normalization

Deterministic provider responses in this module are an explicit dev/test stub
for local runs and CI. They are not product-grade advisory intelligence.
"""

import json
import os
import re
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from spice.decision import (
    CandidateDecision,
    DecisionObjective,
    DecisionPolicy,
    PolicyIdentity,
    SafetyConstraint,
)
from spice.llm.adapters import LLMDecisionAdapter
from spice.llm.core import (
    LLMClient,
    LLMModelConfig,
    LLMModelConfigOverride,
    LLMRouter,
    LLMTaskHook,
    ProviderRegistry,
)
from spice.llm.simulation import SimulationModel
from spice.llm.providers import DeterministicLLMProvider, SubprocessLLMProvider
from spice.protocols import Decision, WorldState
from spice_personal.advisory.personal_simulation_adapter import (
    PersonalLLMSimulationAdapter,
)
from spice_personal.profile.contract import ensure_minimum_execution_brief
from spice_personal.wrappers.errors import (
    model_response_validity_error,
    model_unsupported_capability_error,
    wrap_model_exception,
)


PERSONAL_MODEL_ENV = "SPICE_PERSONAL_MODEL"
PERSONAL_DEBUG_MODEL_IO_ENV = "SPICE_PERSONAL_DEBUG_MODEL_IO"
PERSONAL_DEBUG_MODEL_IO_DIR = Path(".spice/personal/artifacts")
_MODEL_STDOUT_ATTR = "_spice_model_stdout"
_MODEL_STDERR_ATTR = "_spice_model_stderr"
PERSONAL_SIMULATION_FANOUT_LIMIT_DEFAULT = 3
PERSONAL_ACTION_ALIAS_MAP = {
    "personal.assistant.answer": "personal.assistant.suggest",
    "personal.assistant.question_refinement": "personal.assistant.ask_clarify",
    "personal.assistant.retrieval_task": "personal.assistant.gather_evidence",
}
RESULT_KIND_SUGGESTION = "suggestion"
RESULT_KIND_ACTION_PROPOSAL = "action_proposal"
PERSONAL_ACTION_SUGGEST = "personal.assistant.suggest"
PERSONAL_ACTION_ASK_CLARIFY = "personal.assistant.ask_clarify"
PERSONAL_ACTION_DEFER = "personal.assistant.defer"
PERSONAL_ACTION_GATHER_EVIDENCE = "personal.assistant.gather_evidence"
PERSONAL_ADVISORY_ATTRIBUTE_KEYS = (
    "suggestion_text",
    "confidence",
    "urgency",
    "score",
    "simulation_rationale",
    "result_kind",
    "execution_brief",
    "benefits",
    "risks",
    "key_assumptions",
    "first_step_24h",
    "stop_loss_trigger",
    "change_mind_condition",
    "decision_brain_report",
    "recommended_option_label",
    "recommended_option_rank",
    "recommendation_reason",
    "what_would_change_my_mind",
    "clarifying_questions",
    "evidence_plan",
    "defer_plan",
)
ACTION_ENTRY_MIN_CONFIDENCE = {
    PERSONAL_ACTION_SUGGEST: 0.72,
    PERSONAL_ACTION_ASK_CLARIFY: 0.35,
    PERSONAL_ACTION_GATHER_EVIDENCE: 0.35,
    PERSONAL_ACTION_DEFER: 0.25,
}
ACTION_ENTRY_MAX_CONFIDENCE = {
    PERSONAL_ACTION_ASK_CLARIFY: 0.92,
    PERSONAL_ACTION_GATHER_EVIDENCE: 0.88,
    PERSONAL_ACTION_DEFER: 0.75,
}
ACTION_ENTRY_SWITCH_SCORE_GAP = {
    PERSONAL_ACTION_SUGGEST: 0.12,
    PERSONAL_ACTION_ASK_CLARIFY: 0.08,
    PERSONAL_ACTION_GATHER_EVIDENCE: 0.08,
    PERSONAL_ACTION_DEFER: 0.08,
}
SUGGEST_REQUIRED_TEXT_FIELDS = (
    "suggestion_text",
    "first_step_24h",
    "stop_loss_trigger",
    "change_mind_condition",
)
SUGGEST_REQUIRED_LIST_FIELDS = (
    "benefits",
    "risks",
    "key_assumptions",
)
GENERIC_SUGGESTION_PHRASES = (
    "reach out to a friend",
    "reach out to a trusted friend",
    "ask them for their thoughts",
    "clarify your next steps",
    "draft a short, specific question",
    "talk to a friend or colleague",
    "friend or family",
    "friend or family member",
    "share your goals with a friend",
    "ask a friend",
    "ask your family",
    "trusted person",
)
GENERIC_SUGGESTION_WEAK_PATTERNS = (
    "take a moment to",
    "talk to",
    "speak with",
    "share this with",
    "get perspective from",
    "reflect on your priorities",
    "clarify your priorities",
)
DECISION_SPECIFIC_SIGNAL_TOKENS = (
    "option",
    "offer",
    "salary",
    "team",
    "manager",
    "mentor",
    "risk",
    "evidence",
    "assumption",
    "24h",
    "24-hour",
    "first_step",
    "stop_loss",
    "tradeoff",
    "promotion",
    "management",
    "a/b",
    "薪资",
    "团队",
    "导师",
    "管理",
    "风险",
    "证据",
    "假设",
    "止损",
)
TRADEOFF_SIGNAL_TOKENS = (
    "tradeoff",
    "trade-off",
    "risk",
    "benefit",
    "upside",
    "downside",
    "opportunity cost",
    "cost",
    "volatility",
    "stability",
    "uncertainty",
    "收益",
    "风险",
    "机会成本",
    "稳定",
    "不稳定",
)
TIME_ANCHOR_TOKENS = (
    "today",
    "tonight",
    "tomorrow",
    "24h",
    "24-hour",
    "24 hour",
    "within",
    "by ",
    "before ",
    "next day",
    "next week",
    "this week",
    "day",
    "week",
    "hour",
    "today",
    "今天",
    "明天",
    "24小时",
    "本周",
    "周内",
)
ACTION_VERB_TOKENS = (
    "confirm",
    "verify",
    "check",
    "ask",
    "book",
    "schedule",
    "draft",
    "compare",
    "collect",
    "send",
    "align",
    "negotiate",
    "call",
    "message",
    "确认",
    "核实",
    "检查",
    "询问",
    "预约",
    "安排",
    "拟定",
    "比较",
    "收集",
    "沟通",
    "谈判",
)
EVIDENCE_INTERNAL_RUNTIME_TOKENS = (
    "signal-",
    "obs-",
    "observation count",
    "session state",
    "state snapshot",
    "worldstate",
    "timestamp against",
    "checklist status",
    "selected_action",
    "question_received",
    "protocol validation",
    "complexity requirement",
    "complexity requirements",
    "risk_budget",
    "active session",
    "current state",
    "hypothesis",
    "prepared evidence",
)
EVIDENCE_REAL_WORLD_ACTION_TOKENS = (
    "verify",
    "confirm",
    "check",
    "request",
    "obtain",
    "compare",
    "validate",
    "collect",
    "review",
    "ask",
    "contact",
    "call",
    "email",
    "interview",
    "核实",
    "确认",
    "比较",
    "收集",
    "调研",
    "联系",
    "询问",
    "邮件",
)
EVIDENCE_REAL_WORLD_OBJECT_TOKENS = (
    "offer",
    "option",
    "salary",
    "compensation",
    "cashflow",
    "cash flow",
    "manager",
    "mentor",
    "team",
    "attrition",
    "turnover",
    "promotion",
    "management",
    "scope",
    "workload",
    "equity",
    "bonus",
    "contract",
    "offer letter",
    "reference",
    "薪资",
    "现金流",
    "团队",
    "经理",
    "导师",
    "离职",
    "流失",
    "晋升",
    "管理",
    "职责",
    "股权",
    "奖金",
    "offer",
)
EVIDENCE_RANKING_IMPACT_TOKENS = (
    "change",
    "changes",
    "affect",
    "affects",
    "impact",
    "impacts",
    "rank",
    "ranking",
    "reorder",
    "recommend",
    "recommendation",
    "choice",
    "tradeoff",
    "risk",
    "probability",
    "odds",
    "priority",
    "decision",
    "readiness",
    "trajectory",
    "leverage",
    "path",
    "改变",
    "影响",
    "排序",
    "推荐",
    "取舍",
    "风险",
    "概率",
    "决策",
)
QUESTION_ENTITY_STOPWORDS = {
    "what",
    "should",
    "could",
    "would",
    "please",
    "help",
    "choose",
    "have",
    "with",
    "this",
    "that",
    "from",
    "your",
    "about",
    "into",
    "after",
    "before",
    "goal",
    "target",
    "risk",
    "tolerance",
    "appetite",
    "medium",
    "high",
    "low",
    "year",
    "years",
    "month",
    "months",
    "and",
    "for",
}
PERSONAL_ACTION_SELECTION_RULES = {
    PERSONAL_ACTION_SUGGEST: (
        "Use when core decision information is sufficient to provide immediately actionable options."
    ),
    PERSONAL_ACTION_ASK_CLARIFY: (
        "Use when key preference/constraint slots are missing and user answers can change option ranking."
    ),
    PERSONAL_ACTION_GATHER_EVIDENCE: (
        "Use when missing facts are externally verifiable and should be collected before confident recommendation."
    ),
    PERSONAL_ACTION_DEFER: (
        "Use when decision is high-irreversibility and confidence remains low after clarify/evidence paths."
    ),
}


@dataclass(slots=True)
class PersonalLLMDecisionPolicy(DecisionPolicy):
    """DecisionPolicy that delegates personal recommendations to LLM adapters."""

    decision_adapter: LLMDecisionAdapter
    simulation_adapter: SimulationModel
    allowed_actions: tuple[str, ...]
    domain: str = "personal.assistant"
    max_candidates: int = 3
    simulation_fanout_limit: int = PERSONAL_SIMULATION_FANOUT_LIMIT_DEFAULT
    strict_model: bool = False
    identity: PolicyIdentity = field(
        default_factory=lambda: PolicyIdentity.create(
            policy_name="personal.assistant.llm_policy",
            policy_version="0.1",
            implementation_fingerprint="phase2_5",
        )
    )
    _last_state: WorldState | None = field(default=None, init=False, repr=False)
    _last_candidate_decisions: dict[str, Decision] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def propose(self, state: WorldState, context: Any) -> list[CandidateDecision]:
        self._last_state = state
        self._last_candidate_decisions = {}
        decision_context_payload = dict(context) if isinstance(context, dict) else {}
        question_context = _build_question_context_for_model(state)
        decision_context_payload.update(
            {
                "domain": self.domain,
                "stage": "personal_decision",
                "allowed_actions": list(self.allowed_actions),
                "required_output_mode": "decision_brain",
                "action_selection_rules": PERSONAL_ACTION_SELECTION_RULES,
                "selection_objective": (
                    "Prefer the action that most improves decision quality under current uncertainty."
                ),
                "selection_rationale": (
                    "suggest if recommendation-ready now; ask_clarify if user preferences/constraints can change ranking; "
                    "gather_evidence if missing facts are externally verifiable; defer only for high-irreversibility decisions "
                    "where safe recommendation is not yet possible."
                ),
            }
        )
        if question_context:
            decision_context_payload.update(question_context)
        try:
            proposals = self.decision_adapter.propose(
                state,
                context=decision_context_payload,
                max_candidates=self.max_candidates,
            )
        except Exception as exc:
            if self.strict_model:
                wrapped = wrap_model_exception(exc, stage="decision_propose")
                if wrapped.info.code == "model.response_validity":
                    compatibility = _collect_decision_compatibility(
                        adapter=self.decision_adapter,
                        action_normalization_events=[],
                    )
                    _maybe_write_model_debug_artifact(
                        stage="decision_propose",
                        parsed_failure_reason=wrapped.info.message,
                        exc=exc,
                        adapter=self.decision_adapter,
                        compatibility=compatibility,
                    )
                raise wrapped from exc
            return []

        candidates: list[CandidateDecision] = []
        invalid_actions: set[str] = set()
        action_normalization_events: list[dict[str, Any]] = []
        for proposal in proposals:
            normalized = _normalize_decision(proposal, state=state, domain=self.domain)
            original_action = normalized.selected_action or ""
            mapped_action = _normalize_personal_decision_action(
                original_action,
                domain=self.domain,
            )
            alias_mapping_used = mapped_action != original_action
            normalized.selected_action = mapped_action or original_action or None
            action = normalized.selected_action or ""
            action_normalization_events.append(
                {
                    "original_action": original_action,
                    "normalized_action": action,
                    "alias_mapping_used": alias_mapping_used,
                }
            )
            if action not in self.allowed_actions:
                if action:
                    invalid_actions.add(action)
                continue

            candidate_id = normalized.id or f"dec-{uuid4().hex}"
            normalized.id = candidate_id
            self._last_candidate_decisions[candidate_id] = normalized
            score = _as_float(normalized.attributes.get("score"), 0.0)
            confidence = _as_float(normalized.attributes.get("confidence"), 0.0)
            risk = _as_float(normalized.attributes.get("risk"), 0.0)
            params_raw = normalized.attributes.get("params")
            params = dict(params_raw) if isinstance(params_raw, dict) else {}
            candidates.append(
                CandidateDecision(
                    id=candidate_id,
                    action=action,
                    params=params,
                    score_total=score,
                    score_breakdown={"proposal": score},
                    risk=risk,
                    confidence=confidence,
                )
            )
        if self.strict_model and not candidates:
            if invalid_actions:
                proposed = ", ".join(sorted(invalid_actions))
                allowed = ", ".join(sorted(self.allowed_actions))
                raise model_unsupported_capability_error(
                    (
                        "Model proposed unsupported action(s): "
                        f"{proposed}. Allowed actions: {allowed}."
                    ),
                    stage="decision_propose",
                )
            failure_reason = "Model returned no usable decision proposals."
            compatibility = _collect_decision_compatibility(
                adapter=self.decision_adapter,
                action_normalization_events=action_normalization_events,
            )
            _maybe_write_model_debug_artifact(
                stage="decision_propose",
                parsed_failure_reason=failure_reason,
                exc=None,
                adapter=self.decision_adapter,
                compatibility=compatibility,
            )
            raise model_response_validity_error(
                failure_reason,
                stage="decision_propose",
            )
        compatibility = _collect_decision_compatibility(
            adapter=self.decision_adapter,
            action_normalization_events=action_normalization_events,
        )
        if _compatibility_used(compatibility):
            _maybe_write_model_debug_artifact(
                stage="decision_propose",
                parsed_failure_reason="compatibility_applied",
                exc=None,
                adapter=self.decision_adapter,
                compatibility=compatibility,
            )
        return candidates

    def select(
        self,
        candidates: list[CandidateDecision],
        objective: DecisionObjective,
        constraints: list[SafetyConstraint],
    ) -> Decision:
        if not candidates:
            if self.strict_model:
                raise model_response_validity_error(
                    "No candidates available after model proposal stage.",
                    stage="decision_select",
                )
            default_action = self.allowed_actions[0] if self.allowed_actions else "personal.assistant.suggest"
            return self._degraded_decision(
                selected_action=default_action,
                reason="no_candidates_from_llm",
            )

        if not any(candidate.id in self._last_candidate_decisions for candidate in candidates):
            if self.strict_model:
                raise model_response_validity_error(
                    "Runtime fallback candidates were injected for a configured model.",
                    stage="decision_select",
                )
            # Runtime injected fallback candidates (for example domain fallback).
            # Keep degraded behavior explicit and minimal.
            return self._degraded_decision(
                selected_action=candidates[0].action,
                reason="runtime_domain_fallback_candidates",
            )

        risk_budget = objective.risk_budget
        risk_filtered = [candidate for candidate in candidates if candidate.risk <= risk_budget]
        eligible = risk_filtered or candidates
        simulation_candidates = _limit_simulation_candidates(
            eligible,
            fanout_limit=self.simulation_fanout_limit,
        )

        best_candidate = eligible[0]
        best_seed_decision = self._seed_decision_for_candidate(eligible[0])
        best_advisory = _normalize_advisory_attributes(
            selected_action=best_seed_decision.selected_action or eligible[0].action,
            score=eligible[0].score_total,
            confidence=eligible[0].confidence,
            simulation_rationale="candidate_default",
            result_kind=_extract_result_kind(best_seed_decision, {}),
            execution_brief=_extract_execution_brief(best_seed_decision, {}),
            benefits=(),
            risks=(),
            key_assumptions=(),
            first_step_24h="",
            stop_loss_trigger="",
            change_mind_condition="",
            decision_brain_report={},
            clarifying_questions=(),
            evidence_plan=(),
            defer_plan={},
        )
        best_artifact: dict[str, Any] = {}
        decision_options: list[dict[str, Any]] = []
        advisory_by_candidate_id: dict[str, dict[str, Any]] = {}
        artifact_by_candidate_id: dict[str, dict[str, Any]] = {}
        latest_question = _state_latest_question(self._last_state)

        for candidate in simulation_candidates:
            seed_decision = self._seed_decision_for_candidate(candidate)
            artifact = self._simulate_candidate(seed_decision, objective=objective, constraints=constraints)
            selected_action = seed_decision.selected_action or candidate.action
            extracted_suggestion = _extract_suggestion(seed_decision, artifact)
            advisory = _normalize_advisory_attributes(
                selected_action=selected_action,
                suggestion_text=extracted_suggestion,
                confidence=_extract_confidence(seed_decision, artifact, default=candidate.confidence),
                urgency=_extract_urgency(seed_decision, artifact),
                score=_as_float(artifact.get("score"), candidate.score_total),
                simulation_rationale=_extract_rationale(artifact),
                result_kind=_extract_result_kind(seed_decision, artifact),
                execution_brief=_extract_execution_brief(seed_decision, artifact),
                benefits=_extract_option_list(
                    artifact,
                    keys=("benefits", "upside", "pros"),
                ),
                risks=_extract_option_list(
                    artifact,
                    keys=("risks", "downside", "cons"),
                ),
                key_assumptions=_extract_option_list(
                    artifact,
                    keys=("key_assumptions", "assumptions"),
                ),
                first_step_24h=_extract_first_step_24h(artifact),
                stop_loss_trigger=_extract_stop_loss_trigger(artifact),
                change_mind_condition=_extract_change_mind_condition(artifact),
                decision_brain_report=(
                    _extract_decision_brain_report(artifact)
                    if selected_action == PERSONAL_ACTION_SUGGEST
                    else {}
                ),
                clarifying_questions=(
                    _extract_clarifying_questions(artifact, question=latest_question)
                    if selected_action == PERSONAL_ACTION_ASK_CLARIFY
                    else ()
                ),
                evidence_plan=(
                    _extract_evidence_plan(artifact, question=latest_question)
                    if selected_action == PERSONAL_ACTION_GATHER_EVIDENCE
                    else ()
                ),
                defer_plan=(
                    _extract_defer_plan(artifact)
                    if selected_action == PERSONAL_ACTION_DEFER
                    else {}
                ),
            )
            entry_assessment = _evaluate_action_entry_assessment(
                action=selected_action,
                advisory=advisory,
                question=latest_question,
            )
            advisory["action_entry_assessment"] = entry_assessment
            decision_options.append(
                _build_decision_option_payload(
                    candidate=candidate,
                    advisory=advisory,
                )
            )
            advisory_by_candidate_id[candidate.id] = advisory
            artifact_by_candidate_id[candidate.id] = artifact

            if advisory["score"] > best_advisory["score"]:
                best_candidate = candidate
                best_advisory = advisory
                best_artifact = artifact

        ranked_options = sorted(
            decision_options,
            key=lambda option: (
                1 if bool(option.get("entry_contract_passed")) else 0,
                _as_float(option.get("score"), 0.0),
                _as_float(option.get("confidence"), 0.0),
            ),
            reverse=True,
        )[:3]
        best_candidate, best_advisory = _apply_action_entry_thresholds(
            best_candidate=best_candidate,
            best_advisory=best_advisory,
            ranked_options=ranked_options,
            advisory_by_candidate_id=advisory_by_candidate_id,
            eligible=eligible,
            question=latest_question,
        )
        best_artifact = artifact_by_candidate_id.get(best_candidate.id, best_artifact)
        best_entry_assessment = _entry_assessment_from_advisory(best_advisory)
        if self.strict_model and not _entry_assessment_passes(best_entry_assessment):
            reasons = best_entry_assessment.get("reasons")
            reason_text = ", ".join(reasons) if isinstance(reasons, list) else ""
            failure_reason = (
                "Model simulation failed action entry contract."
                if not reason_text
                else f"Model simulation failed action entry contract: {reason_text}"
            )
            _maybe_write_model_debug_artifact(
                stage="simulation_advise",
                parsed_failure_reason=failure_reason,
                exc=None,
                adapter=self.simulation_adapter,
                candidate_id=best_candidate.id,
                selected_action=best_candidate.action,
                timeout_seconds=_resolve_simulation_timeout_seconds(
                    adapter=self.simulation_adapter,
                    domain=self.domain,
                ),
            )
            raise model_response_validity_error(
                failure_reason,
                stage="simulation_advise",
            )
        if not _entry_assessment_passes(best_entry_assessment):
            reasons = best_entry_assessment.get("reasons")
            reason_tokens = (
                [str(item).strip() for item in reasons if str(item).strip()]
                if isinstance(reasons, list)
                else []
            )
            degraded_reason = (
                "entry_contract_failed"
                if not reason_tokens
                else "entry_contract_failed:" + ",".join(reason_tokens)
            )
            selected_action = best_candidate.action
            degraded = self._degraded_decision(
                selected_action=selected_action,
                reason=degraded_reason,
            )
            ranked_with_recommended = _prioritize_recommended_option(
                ranked_options,
                recommended_candidate_id=best_candidate.id,
            )
            ranked_with_recommended = _annotate_option_labels_and_ranks(
                ranked_with_recommended
            )
            degraded.attributes["selected_candidate_id"] = best_candidate.id
            degraded.attributes["recommended_option_id"] = best_candidate.id
            degraded.attributes["decision_options"] = ranked_with_recommended
            degraded.attributes["recommended_option_label"] = _recommended_option_label_from_options(
                ranked_with_recommended,
                recommended_candidate_id=best_candidate.id,
            )
            degraded.attributes["entry_contract_reasons"] = reason_tokens
            if best_artifact:
                degraded.metadata["simulation"] = dict(best_artifact)
            return degraded
        selected = self._seed_decision_for_candidate(best_candidate)
        if selected.selected_action is None:
            selected.selected_action = best_candidate.action
        ranked_options = _prioritize_recommended_option(
            ranked_options,
            recommended_candidate_id=best_candidate.id,
        )
        ranked_options = _annotate_option_labels_and_ranks(ranked_options)
        selected.attributes["selected_candidate_id"] = best_candidate.id
        selected.attributes["all_candidates"] = [
            {
                "id": candidate.id,
                "action": candidate.action,
                "score_total": candidate.score_total,
                "risk": candidate.risk,
                "confidence": candidate.confidence,
            }
            for candidate in candidates
        ]
        selected.attributes["decision_options"] = ranked_options
        selected.attributes["recommended_option_id"] = best_candidate.id
        selected.attributes["recommended_option_label"] = _recommended_option_label_from_options(
            ranked_options,
            recommended_candidate_id=best_candidate.id,
        )
        selected.attributes["advisory_degraded"] = False
        selected.attributes.update(best_advisory)
        if best_artifact:
            selected.metadata["simulation"] = dict(best_artifact)
        return selected

    def _simulate_candidate(
        self,
        decision: Decision,
        *,
        objective: DecisionObjective,
        constraints: list[SafetyConstraint],
    ) -> dict[str, Any]:
        if self._last_state is None:
            if self.strict_model:
                raise model_response_validity_error(
                    "Internal state was missing before simulation stage.",
                    stage="simulation_advise",
                )
            return {}

        timeout_seconds = _resolve_simulation_timeout_seconds(
            adapter=self.simulation_adapter,
            domain=self.domain,
        )
        question_context = _build_question_context_for_model(self._last_state)
        simulation_context_payload: dict[str, Any] = {
            "domain": self.domain,
            "stage": "personal_simulation",
            "objective": {
                "risk_budget": objective.risk_budget,
            },
            "constraints": [
                {
                    "name": constraint.name,
                    "kind": constraint.kind,
                    "params": constraint.params,
                }
                for constraint in constraints
            ],
        }
        if question_context:
            simulation_context_payload.update(question_context)
        try:
            artifact = self.simulation_adapter.simulate(
                self._last_state,
                decision=decision,
                context=simulation_context_payload,
            )
        except Exception as exc:
            if self.strict_model:
                wrapped = wrap_model_exception(exc, stage="simulation_advise")
                _maybe_write_model_debug_artifact(
                    stage="simulation_advise",
                    parsed_failure_reason=wrapped.info.message,
                    exc=exc,
                    adapter=self.simulation_adapter,
                    candidate_id=decision.id,
                    selected_action=decision.selected_action or "",
                    timeout_seconds=timeout_seconds,
                )
                raise wrapped from exc
            return {}
        normalized = dict(artifact) if isinstance(artifact, dict) else {}
        if self.strict_model and not normalized:
            failure_reason = "Model simulation returned an empty artifact."
            _maybe_write_model_debug_artifact(
                stage="simulation_advise",
                parsed_failure_reason=failure_reason,
                exc=None,
                adapter=self.simulation_adapter,
                candidate_id=decision.id,
                selected_action=decision.selected_action or "",
                timeout_seconds=timeout_seconds,
            )
            raise model_response_validity_error(
                failure_reason,
                stage="simulation_advise",
            )
        return normalized

    def _seed_decision_for_candidate(self, candidate: CandidateDecision) -> Decision:
        proposal = self._last_candidate_decisions.get(candidate.id)
        if proposal is not None:
            return _copy_decision(proposal, state=self._last_state, domain=self.domain)

        refs = [self._last_state.id] if self._last_state is not None else []
        return Decision(
            id=f"dec-{uuid4().hex}",
            decision_type=f"{self.domain}.llm",
            status="proposed",
            selected_action=candidate.action,
            refs=refs,
            attributes={},
        )

    def _degraded_decision(self, *, selected_action: str, reason: str) -> Decision:
        refs = [self._last_state.id] if self._last_state is not None else []
        attributes = _normalize_advisory_attributes(
            confidence=0.0,
            urgency="",
            score=0.0,
            simulation_rationale=f"degraded:{reason}",
        )
        attributes.update(
            {
                "advisory_degraded": True,
                "degraded_reason": reason,
            }
        )
        return Decision(
            id=f"dec-{uuid4().hex}",
            decision_type=f"{self.domain}.llm",
            status="proposed",
            selected_action=selected_action,
            refs=refs,
            attributes=attributes,
        )


def build_personal_llm_decision_policy(
    *,
    model: str | None,
    domain: str,
    allowed_actions: tuple[str, ...],
    simulation_fanout_limit: int = PERSONAL_SIMULATION_FANOUT_LIMIT_DEFAULT,
    strict_model: bool = False,
) -> PersonalLLMDecisionPolicy | None:
    if not allowed_actions:
        return None

    try:
        client = _build_personal_llm_client(allowed_actions=allowed_actions)
        model_override = _resolve_personal_model_override(model=model)
        return PersonalLLMDecisionPolicy(
            decision_adapter=LLMDecisionAdapter(
                client=client,
                model_override=model_override,
            ),
            simulation_adapter=PersonalLLMSimulationAdapter(
                client=client,
                model_override=model_override,
            ),
            allowed_actions=tuple(allowed_actions),
            domain=domain,
            simulation_fanout_limit=simulation_fanout_limit,
            strict_model=strict_model,
        )
    except Exception as exc:
        if strict_model:
            raise wrap_model_exception(exc, stage="client_init") from exc
        return None


def _build_personal_llm_client(*, allowed_actions: tuple[str, ...]) -> LLMClient:
    # Deterministic provider here is a dev/test scaffold only.
    # Product onboarding/usage should rely on configured real model commands.
    decision_default = LLMModelConfig(
        provider_id="deterministic",
        model_id="deterministic.personal.stub.v1",
        temperature=0.0,
        max_tokens=1200,
        timeout_sec=120.0,
        response_format_hint="json_array",
    )
    simulation_default = replace(
        decision_default,
        response_format_hint="json_object",
    )
    router = LLMRouter(
        global_default=decision_default,
        hook_defaults={
            LLMTaskHook.DECISION_PROPOSE: decision_default,
            LLMTaskHook.SIMULATION_ADVISE: simulation_default,
        },
    )
    stub_provider = DeterministicLLMProvider(
        responses={
            LLMTaskHook.DECISION_PROPOSE: _stub_llm_personal_decision_response(
                allowed_actions=allowed_actions
            ),
            LLMTaskHook.SIMULATION_ADVISE: _stub_llm_personal_simulation_response(),
        }
    )
    registry = (
        ProviderRegistry.empty()
        .register(stub_provider)
        .register(SubprocessLLMProvider())
    )
    return LLMClient(registry=registry, router=router)


def _resolve_personal_model_override(model: str | None) -> LLMModelConfigOverride | None:
    # "deterministic" is kept for tests/dev workflows only.
    # It is not a standard product onboarding path.
    raw = model if model is not None else os.environ.get(PERSONAL_MODEL_ENV)
    if raw is None:
        return None
    token = raw.strip()
    if not token:
        return None
    if token.lower() == "deterministic":
        return LLMModelConfigOverride(
            provider_id="deterministic",
            model_id="deterministic.personal.stub.v1",
        )
    return LLMModelConfigOverride(
        provider_id="subprocess",
        model_id=token,
    )


def _stub_llm_personal_decision_response(*, allowed_actions: tuple[str, ...]) -> str:
    action = allowed_actions[0] if allowed_actions else "personal.assistant.suggest"
    payload = [
        {
            "decision_type": "personal.assistant.llm",
            "status": "proposed",
            "selected_action": action,
            "attributes": {
                "confidence": 0.65,
                "urgency": "normal",
            },
        }
    ]
    return json.dumps(payload, ensure_ascii=True)


def _stub_llm_personal_simulation_response() -> str:
    payload = {
        "score": 0.78,
        "confidence": 0.78,
        "urgency": "normal",
        "suggestion_text": (
            "Stub LLM advisory suggestion (dev/test): Option B is lower-regret now because it "
            "balances growth and team stability better than Option A."
        ),
        "benefits": [
            "Keeps optionality while preserving momentum.",
        ],
        "risks": [
            "May delay upside if the window closes quickly.",
        ],
        "key_assumptions": [
            "The opportunity remains available for at least two more weeks.",
        ],
        "first_step_24h": "Within 24h, confirm manager expectations and first-quarter scope for both Option A and Option B.",
        "stop_loss_trigger": "Pause and revisit if one option introduces new irreversible downside.",
        "change_mind_condition": "Switch recommendation if verified team-stability evidence contradicts current assumptions.",
        "simulation_rationale": "stub_llm_default_personal_simulation",
        "result_kind": RESULT_KIND_SUGGESTION,
    }
    return json.dumps(payload, ensure_ascii=True)


def _normalize_advisory_attributes(
    *,
    selected_action: str = "",
    suggestion_text: str = "",
    confidence: float = 0.0,
    urgency: str = "",
    score: float = 0.0,
    simulation_rationale: str = "",
    result_kind: str = RESULT_KIND_SUGGESTION,
    execution_brief: Any = None,
    benefits: tuple[str, ...] | list[str] = (),
    risks: tuple[str, ...] | list[str] = (),
    key_assumptions: tuple[str, ...] | list[str] = (),
    first_step_24h: str = "",
    stop_loss_trigger: str = "",
    change_mind_condition: str = "",
    decision_brain_report: dict[str, Any] | None = None,
    clarifying_questions: tuple[dict[str, str], ...] | list[dict[str, str]] = (),
    evidence_plan: tuple[dict[str, str], ...] | list[dict[str, str]] = (),
    defer_plan: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Return a stable personal advisory attributes contract."""

    normalized_result_kind = _normalize_result_kind(result_kind)
    normalized_execution_brief: dict[str, Any] = {}
    if normalized_result_kind == RESULT_KIND_ACTION_PROPOSAL:
        normalized_execution_brief = ensure_minimum_execution_brief(
            execution_brief,
            selected_action=selected_action,
            suggestion_text=suggestion_text,
        )
    normalized_defer_plan: dict[str, str] = {}
    if isinstance(defer_plan, dict):
        revisit_at = _as_text(defer_plan.get("revisit_at"))
        monitor = _as_text(defer_plan.get("monitor_signal"))
        trigger = _as_text(defer_plan.get("resume_trigger"))
        if revisit_at:
            normalized_defer_plan["revisit_at"] = revisit_at
        if monitor:
            normalized_defer_plan["monitor_signal"] = monitor
        if trigger:
            normalized_defer_plan["resume_trigger"] = trigger
    normalized_decision_brain_report = _normalize_decision_brain_report(decision_brain_report)
    recommended_option_label = _as_text(normalized_decision_brain_report.get("recommended_option_label"))
    recommended_option_rank = _as_int(
        normalized_decision_brain_report.get("recommended_option_rank"),
        0,
    )
    recommendation_reason = _as_text(normalized_decision_brain_report.get("recommendation_reason"))
    if not recommendation_reason:
        recommendation_reason = _as_text(suggestion_text)
    what_would_change_my_mind = _as_text(
        normalized_decision_brain_report.get("what_would_change_my_mind")
    )
    if not what_would_change_my_mind:
        what_would_change_my_mind = _as_text(change_mind_condition)
    return {
        "suggestion_text": _as_text(suggestion_text),
        "confidence": _as_float(confidence, 0.0),
        "urgency": _as_text(urgency),
        "score": _as_float(score, 0.0),
        "simulation_rationale": _as_text(simulation_rationale),
        "result_kind": normalized_result_kind,
        "execution_brief": normalized_execution_brief,
        "benefits": _normalize_text_list(benefits),
        "risks": _normalize_text_list(risks),
        "key_assumptions": _normalize_text_list(key_assumptions),
        "first_step_24h": _as_text(first_step_24h),
        "stop_loss_trigger": _as_text(stop_loss_trigger),
        "change_mind_condition": _as_text(change_mind_condition),
        "decision_brain_report": normalized_decision_brain_report,
        "recommended_option_label": recommended_option_label,
        "recommended_option_rank": recommended_option_rank,
        "recommendation_reason": recommendation_reason,
        "what_would_change_my_mind": what_would_change_my_mind,
        "clarifying_questions": _normalize_question_list(clarifying_questions),
        "evidence_plan": _normalize_evidence_plan(evidence_plan),
        "defer_plan": normalized_defer_plan,
    }


def _normalize_decision(decision: Decision, *, state: WorldState, domain: str) -> Decision:
    decision_id = (
        decision.id.strip()
        if isinstance(decision.id, str) and decision.id.strip()
        else f"dec-{uuid4().hex}"
    )
    decision_type = (
        decision.decision_type.strip()
        if isinstance(decision.decision_type, str) and decision.decision_type.strip()
        else f"{domain}.llm"
    )
    status = (
        decision.status.strip()
        if isinstance(decision.status, str) and decision.status.strip()
        else "proposed"
    )
    refs = [ref for ref in decision.refs if isinstance(ref, str)]
    if state.id not in refs:
        refs.append(state.id)
    return Decision(
        id=decision_id,
        decision_type=decision_type,
        status=status,
        selected_action=decision.selected_action,
        refs=refs,
        metadata=dict(decision.metadata),
        attributes=dict(decision.attributes),
    )


def _copy_decision(decision: Decision, *, state: WorldState | None, domain: str) -> Decision:
    refs = [ref for ref in decision.refs if isinstance(ref, str)]
    if state is not None and state.id not in refs:
        refs.append(state.id)
    decision_id = (
        decision.id
        if isinstance(decision.id, str) and decision.id.strip()
        else f"dec-{uuid4().hex}"
    )
    decision_type = (
        decision.decision_type
        if isinstance(decision.decision_type, str) and decision.decision_type.strip()
        else f"{domain}.llm"
    )
    status = (
        decision.status
        if isinstance(decision.status, str) and decision.status.strip()
        else "proposed"
    )
    return Decision(
        id=decision_id,
        decision_type=decision_type,
        status=status,
        selected_action=decision.selected_action,
        refs=refs,
        metadata=dict(decision.metadata),
        attributes=dict(decision.attributes),
    )


def _extract_suggestion(decision: Decision, artifact: dict[str, Any]) -> str:
    fields = (
        artifact.get("suggestion_text"),
        artifact.get("suggestion"),
        artifact.get("advice"),
        decision.attributes.get("suggestion_text"),
    )
    for value in fields:
        text = _as_text(value)
        if text:
            return text
    return ""


def _state_latest_question(state: WorldState | None) -> str:
    if state is None:
        return ""
    entity = state.entities.get("personal.assistant.current")
    if not isinstance(entity, dict):
        return ""
    return _as_text(entity.get("latest_question"))


def _state_personal_entity(state: WorldState | None) -> dict[str, Any]:
    if state is None:
        return {}
    entity = state.entities.get("personal.assistant.current")
    if isinstance(entity, dict):
        return dict(entity)
    return {}


def _build_question_context_for_model(state: WorldState | None) -> dict[str, Any]:
    entity = _state_personal_entity(state)
    latest_question = _as_text(entity.get("latest_question"))
    if not latest_question:
        return {}
    evidence_summary = _as_text(entity.get("evidence_summary"))
    profile = _build_question_profile(latest_question)
    complete_context = _question_profile_is_complete(profile)
    entity_snapshot = {
        "latest_question": latest_question,
        "status": _as_text(entity.get("status")),
        "urgency": _as_text(entity.get("urgency")),
        "confidence": _as_float(entity.get("confidence"), 0.0),
        "evidence_summary": evidence_summary,
    }
    return {
        "latest_question": latest_question,
        "question_profile": profile,
        "personal_entity": entity_snapshot,
        "decision_readiness": "high" if complete_context else "partial",
        "action_bias": (
            "prefer personal.assistant.suggest unless high-impact slots are missing"
            if complete_context
            else "ask_clarify or gather_evidence only when missing slots materially change ranking"
        ),
    }


def _build_question_profile(question: str) -> dict[str, Any]:
    normalized = _as_text(question)
    lowered = normalized.lower()

    alternative_labels = _extract_alternative_labels(normalized)
    has_alternatives = bool(alternative_labels) or any(
        token in lowered
        for token in (
            "offer",
            "option",
            "a/b",
            "方案",
            "选项",
        )
    )
    has_goal = any(
        token in lowered
        for token in (
            "goal",
            "target",
            "management",
            "promotion",
            "目标",
            "管理",
            "晋升",
        )
    )
    has_risk_preference = any(
        token in lowered
        for token in (
            "risk",
            "risk tolerance",
            "risk appetite",
            "风险",
        )
    )
    has_constraints = bool(_extract_hard_constraints(normalized))
    has_time_horizon = any(
        token in lowered
        for token in (
            "3 year",
            "3-year",
            "3年",
            "6 month",
            "6-month",
            "6个月",
            "24h",
            "24-hour",
            "12 month",
            "12-month",
            "12个月",
        )
    )
    missing_slots: list[str] = []
    if not has_alternatives:
        missing_slots.append("alternatives")
    if not has_goal:
        missing_slots.append("goal")
    if not has_risk_preference:
        missing_slots.append("risk_preference")
    if not has_constraints:
        missing_slots.append("hard_constraints")
    readiness_total = 5
    readiness_hits = sum(
        [
            1 if has_alternatives else 0,
            1 if has_goal else 0,
            1 if has_risk_preference else 0,
            1 if has_constraints else 0,
            1 if has_time_horizon else 0,
        ]
    )
    readiness_score = round(readiness_hits / readiness_total, 2)
    return {
        "alternatives_detected": has_alternatives,
        "alternative_labels": alternative_labels,
        "goal_detected": has_goal,
        "risk_preference_detected": has_risk_preference,
        "hard_constraints_detected": has_constraints,
        "hard_constraints": _extract_hard_constraints(normalized),
        "time_horizon_detected": has_time_horizon,
        "decision_readiness_score": readiness_score,
        "missing_slots": missing_slots,
    }


def _extract_alternative_labels(question: str) -> list[str]:
    token = _as_text(question)
    if not token:
        return []
    labels: list[str] = []
    for match in re.findall(r"(?<![A-Za-z0-9])([A-Ca-c])(?![A-Za-z0-9])", token):
        label = match.upper()
        if label not in labels:
            labels.append(label)
    return labels


def _extract_hard_constraints(question: str) -> list[str]:
    token = _as_text(question)
    if not token:
        return []
    constraints: list[str] = []
    pattern = re.compile(
        r"(最低现金流[^,，。；;\n]*|minimum[^,.;\n]*cash flow[^,.;\n]*|cash flow requirement[^,.;\n]*)",
        flags=re.IGNORECASE,
    )
    for match in pattern.findall(token):
        normalized = _as_text(match)
        if normalized and normalized not in constraints:
            constraints.append(normalized)
    return constraints


def _extract_option_list(
    artifact: dict[str, Any],
    *,
    keys: tuple[str, ...],
) -> tuple[str, ...]:
    action_specific = _artifact_action_specific(artifact)
    for key in keys:
        if key not in artifact:
            continue
        normalized = _normalize_text_list(artifact.get(key))
        if normalized:
            return tuple(normalized)
    for key in keys:
        if key not in action_specific:
            continue
        normalized = _normalize_text_list(action_specific.get(key))
        if normalized:
            return tuple(normalized)
    return ()


def _extract_first_step_24h(artifact: dict[str, Any]) -> str:
    action_specific = _artifact_action_specific(artifact)
    fields = (
        artifact.get("first_step_24h"),
        artifact.get("first_step"),
        artifact.get("next_step"),
        artifact.get("immediate_action"),
        action_specific.get("first_step_24h"),
        action_specific.get("first_step"),
        action_specific.get("next_step"),
        action_specific.get("immediate_action"),
    )
    for value in fields:
        text = _as_text(value)
        if text:
            return text
    return ""


def _extract_stop_loss_trigger(artifact: dict[str, Any]) -> str:
    action_specific = _artifact_action_specific(artifact)
    fields = (
        artifact.get("stop_loss_trigger"),
        artifact.get("stop_loss"),
        artifact.get("stop_condition"),
        action_specific.get("stop_loss_trigger"),
        action_specific.get("stop_loss"),
        action_specific.get("stop_condition"),
    )
    for value in fields:
        text = _as_text(value)
        if text:
            return text
    return ""


def _extract_change_mind_condition(artifact: dict[str, Any]) -> str:
    action_specific = _artifact_action_specific(artifact)
    fields = (
        artifact.get("change_mind_condition"),
        artifact.get("what_would_change_my_mind"),
        artifact.get("reconsider_if"),
        action_specific.get("change_mind_condition"),
        action_specific.get("what_would_change_my_mind"),
        action_specific.get("reconsider_if"),
    )
    for value in fields:
        text = _as_text(value)
        if text:
            return text
    return ""


def _extract_decision_brain_report(artifact: dict[str, Any]) -> dict[str, Any]:
    action_specific = _artifact_action_specific(artifact)
    candidates = (
        artifact.get("decision_brain_report"),
        artifact.get("decision_report"),
        artifact.get("report"),
        action_specific.get("decision_brain_report"),
        action_specific.get("decision_report"),
        action_specific.get("report"),
        {"options": artifact.get("options")} if "options" in artifact else None,
        {"options": action_specific.get("options")} if "options" in action_specific else None,
    )
    for value in candidates:
        normalized = _normalize_decision_brain_report(value)
        if normalized:
            return normalized
    return {}


def _artifact_action_specific(artifact: dict[str, Any]) -> dict[str, Any]:
    payload = artifact.get("action_specific")
    if isinstance(payload, dict):
        return payload
    return {}


def _normalize_decision_brain_report(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    options = _normalize_decision_brain_report_options(value.get("options"))
    if len(options) < 2:
        return {}

    recommended_option_label = _as_text(
        value.get("recommended_option_label")
        or value.get("recommended_option")
    )
    recommendation_payload = value.get("recommendation")
    if isinstance(recommendation_payload, dict):
        if not recommended_option_label:
            recommended_option_label = _as_text(
                recommendation_payload.get("option_label")
                or recommendation_payload.get("recommended_option_label")
            )
    if not recommended_option_label:
        recommended_option_label = _as_text(options[0].get("label"))
    if recommended_option_label:
        available_labels = {_as_text(option.get("label")) for option in options}
        if recommended_option_label not in available_labels:
            recommended_option_label = _as_text(options[0].get("label"))
    recommended_option_rank = _recommended_option_rank(
        options,
        recommended_option_label=recommended_option_label,
        raw_rank=value.get("recommended_option_rank") if isinstance(value, dict) else None,
    )

    recommendation_reason = _as_text(
        value.get("recommendation_reason")
        or value.get("recommended_reason")
    )
    if isinstance(recommendation_payload, dict) and not recommendation_reason:
        recommendation_reason = _as_text(
            recommendation_payload.get("reason")
            or recommendation_payload.get("rationale")
        )
    what_would_change_my_mind = _as_text(
        value.get("what_would_change_my_mind")
        or value.get("change_mind_condition")
    )
    if isinstance(recommendation_payload, dict) and not what_would_change_my_mind:
        what_would_change_my_mind = _as_text(
            recommendation_payload.get("what_would_change_my_mind")
            or recommendation_payload.get("change_mind_condition")
        )

    return {
        "options": options[:3],
        "recommended_option_label": recommended_option_label,
        "recommended_option_rank": recommended_option_rank,
        "recommendation_reason": recommendation_reason,
        "what_would_change_my_mind": what_would_change_my_mind,
    }


def _normalize_decision_brain_report_options(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if len(normalized) >= 3:
            break
        option = _normalize_decision_brain_report_option(item, index=index)
        if option:
            normalized.append(option)
    return normalized


def _normalize_decision_brain_report_option(item: Any, *, index: int) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    label = _report_option_label(index=index, raw_label=item.get("label") or item.get("title") or item.get("name"))
    option_rank = _report_option_rank(index=index, raw_rank=item.get("option_rank") or item.get("rank"))
    benefits = _normalize_text_list(item.get("benefits") or item.get("upside"))
    risks = _normalize_text_list(item.get("risks") or item.get("downside"))
    key_assumptions = _normalize_text_list(item.get("key_assumptions") or item.get("assumptions"))
    first_step_24h = _as_text(item.get("first_step_24h") or item.get("first_step"))
    stop_loss_trigger = _as_text(item.get("stop_loss_trigger") or item.get("stop_loss"))
    option_positioning = _as_text(
        item.get("option_positioning")
        or item.get("positioning")
        or item.get("judgement")
        or item.get("judgment")
    )
    suggestion_text = _as_text(item.get("suggestion_text") or item.get("summary"))
    if not suggestion_text:
        suggestion_text = option_positioning
    change_mind_condition = _as_text(
        item.get("change_mind_condition") or item.get("what_would_change_my_mind")
    )
    if not benefits and not risks and not key_assumptions and not suggestion_text and not option_positioning:
        return {}
    return {
        "option_id": f"option-{option_rank}",
        "option_rank": option_rank,
        "option_label": label,
        "label": label,
        "option_positioning": option_positioning,
        "suggestion_text": suggestion_text,
        "benefits": benefits,
        "risks": risks,
        "key_assumptions": key_assumptions,
        "first_step_24h": first_step_24h,
        "stop_loss_trigger": stop_loss_trigger,
        "change_mind_condition": change_mind_condition,
    }


def _report_option_label(*, index: int, raw_label: Any) -> str:
    token = _as_text(raw_label)
    if token:
        return token
    label = chr(ord("A") + min(max(index, 0), 25))
    return f"Option {label}"


def _report_option_rank(*, index: int, raw_rank: Any) -> int:
    rank = _as_int(raw_rank, 0)
    if rank > 0:
        return rank
    return index + 1


def _recommended_option_rank(
    options: list[dict[str, Any]],
    *,
    recommended_option_label: str,
    raw_rank: Any,
) -> int:
    rank = _as_int(raw_rank, 0)
    if rank > 0:
        return rank
    token = _as_text(recommended_option_label)
    if token:
        for option in options:
            if _as_text(option.get("label")) != token:
                continue
            option_rank = _as_int(option.get("option_rank"), 0)
            if option_rank > 0:
                return option_rank
            break
    if options:
        first_rank = _as_int(options[0].get("option_rank"), 0)
        if first_rank > 0:
            return first_rank
    return 1


def _decision_brain_report_contract_complete(value: Any) -> bool:
    report = _normalize_decision_brain_report(value)
    if not report:
        return False
    options = report.get("options")
    if not isinstance(options, list) or len(options) != 3:
        return False
    for item in options:
        if not isinstance(item, dict):
            return False
        if not _as_text(item.get("label")):
            return False
        if not _normalize_text_list(item.get("benefits")):
            return False
        if not _normalize_text_list(item.get("risks")):
            return False
        if not _normalize_text_list(item.get("key_assumptions")):
            return False
        if not _as_text(item.get("first_step_24h")):
            return False
        if not _as_text(item.get("stop_loss_trigger")):
            return False
    if not _as_text(report.get("recommended_option_label")):
        return False
    if not _as_text(report.get("recommendation_reason")):
        return False
    if not _as_text(report.get("what_would_change_my_mind")):
        return False
    return True


def _decision_brain_report_text(value: Any) -> str:
    report = _normalize_decision_brain_report(value)
    if not report:
        return ""
    options = report.get("options")
    if not isinstance(options, list):
        options = []
    segments: list[str] = []
    for option in options:
        if not isinstance(option, dict):
            continue
        segments.append(_as_text(option.get("option_label")))
        segments.append(str(_as_int(option.get("option_rank"), 0)))
        segments.append(_as_text(option.get("label")))
        segments.append(_as_text(option.get("option_positioning")))
        segments.append(_as_text(option.get("suggestion_text")))
        segments.append(_join_text_list(_normalize_text_list(option.get("benefits"))))
        segments.append(_join_text_list(_normalize_text_list(option.get("risks"))))
        segments.append(_join_text_list(_normalize_text_list(option.get("key_assumptions"))))
        segments.append(_as_text(option.get("first_step_24h")))
        segments.append(_as_text(option.get("stop_loss_trigger")))
        segments.append(_as_text(option.get("change_mind_condition")))
    segments.append(_as_text(report.get("recommended_option_label")))
    segments.append(str(_as_int(report.get("recommended_option_rank"), 0)))
    segments.append(_as_text(report.get("recommendation_reason")))
    segments.append(_as_text(report.get("what_would_change_my_mind")))
    return " ".join([segment for segment in segments if segment]).strip()


def _extract_clarifying_questions(
    artifact: dict[str, Any],
    *,
    question: str,
) -> tuple[dict[str, str], ...]:
    del question
    fields = (
        artifact.get("clarifying_questions"),
        artifact.get("questions_to_clarify"),
        artifact.get("clarify_questions"),
    )
    for value in fields:
        normalized = _normalize_question_list(value)
        if normalized:
            return tuple(normalized[:3])
    return ()


def _extract_evidence_plan(
    artifact: dict[str, Any],
    *,
    question: str,
) -> tuple[dict[str, str], ...]:
    del question
    fields = (
        artifact.get("evidence_plan"),
        artifact.get("evidence_needed"),
        artifact.get("evidence_tasks"),
    )
    for value in fields:
        normalized = _normalize_evidence_plan(value)
        if normalized:
            return tuple(normalized[:3])
    return ()


def _extract_defer_plan(artifact: dict[str, Any]) -> dict[str, str]:
    payload = artifact.get("defer_plan")
    if isinstance(payload, dict):
        revisit_at = _as_text(payload.get("revisit_at"))
        monitor_signal = _as_text(payload.get("monitor_signal"))
        resume_trigger = _as_text(payload.get("resume_trigger"))
        normalized: dict[str, str] = {}
        if revisit_at:
            normalized["revisit_at"] = revisit_at
        if monitor_signal:
            normalized["monitor_signal"] = monitor_signal
        if resume_trigger:
            normalized["resume_trigger"] = resume_trigger
        if normalized:
            return normalized

    revisit_at = _as_text(artifact.get("revisit_at"))
    monitor_signal = _as_text(artifact.get("monitor_signal"))
    resume_trigger = _as_text(artifact.get("resume_trigger"))
    normalized: dict[str, str] = {}
    if revisit_at:
        normalized["revisit_at"] = revisit_at
    if monitor_signal:
        normalized["monitor_signal"] = monitor_signal
    if resume_trigger:
        normalized["resume_trigger"] = resume_trigger
    return normalized


def _default_clarifying_questions(question: str) -> tuple[dict[str, str], ...]:
    topic = _as_text(question) or "this decision"
    return (
        {
            "question": "What is the one non-negotiable requirement for your next role?",
            "why": f"It separates must-have constraints from preferences for {topic}.",
        },
        {
            "question": "How much short-term uncertainty can you tolerate in the next 12 months?",
            "why": "It decides whether a higher-upside but unstable option is acceptable.",
        },
        {
            "question": "What measurable outcome defines success for this decision in 3 years?",
            "why": "It anchors the recommendation to a concrete long-term objective.",
        },
    )


def _default_evidence_plan(question: str) -> tuple[dict[str, str], ...]:
    topic = _as_text(question) or "this decision"
    return (
        {
            "fact": "Verify team stability and recent attrition trends for each option.",
            "why": f"It reduces uncertainty on execution risk for {topic}.",
        },
        {
            "fact": "Confirm manager quality and mentorship track record from direct sources.",
            "why": "It impacts growth probability over the next 2-3 years.",
        },
        {
            "fact": "Compare realistic workload and downside scenarios over 12 months.",
            "why": "It tests whether the choice is sustainable under stress.",
        },
    )


def _apply_action_entry_thresholds(
    *,
    best_candidate: CandidateDecision,
    best_advisory: dict[str, Any],
    ranked_options: list[dict[str, Any]],
    advisory_by_candidate_id: dict[str, dict[str, Any]],
    eligible: list[CandidateDecision],
    question: str,
) -> tuple[CandidateDecision, dict[str, Any]]:
    best_score = _as_float(best_advisory.get("score"), best_candidate.score_total)
    best_entry = _entry_assessment_from_advisory(best_advisory)
    if _entry_assessment_passes(best_entry):
        return best_candidate, best_advisory

    fallback = _select_entry_passing_option(
        actions=_fallback_action_order(best_candidate.action),
        best_score=best_score,
        ranked_options=ranked_options,
        advisory_by_candidate_id=advisory_by_candidate_id,
        eligible=eligible,
        enforce_score_gap=True,
    )
    if fallback is not None:
        return fallback

    fallback = _select_entry_passing_option(
        actions=_fallback_action_order(best_candidate.action),
        best_score=best_score,
        ranked_options=ranked_options,
        advisory_by_candidate_id=advisory_by_candidate_id,
        eligible=eligible,
        enforce_score_gap=False,
    )
    if fallback is not None:
        return fallback
    return best_candidate, best_advisory


def _fallback_action_order(action: str) -> tuple[str, ...]:
    token = _as_text(action)
    if token == PERSONAL_ACTION_SUGGEST:
        return (
            PERSONAL_ACTION_ASK_CLARIFY,
            PERSONAL_ACTION_GATHER_EVIDENCE,
            PERSONAL_ACTION_DEFER,
        )
    if token == PERSONAL_ACTION_ASK_CLARIFY:
        return (
            PERSONAL_ACTION_SUGGEST,
            PERSONAL_ACTION_GATHER_EVIDENCE,
            PERSONAL_ACTION_DEFER,
        )
    if token == PERSONAL_ACTION_GATHER_EVIDENCE:
        return (
            PERSONAL_ACTION_ASK_CLARIFY,
            PERSONAL_ACTION_SUGGEST,
            PERSONAL_ACTION_DEFER,
        )
    if token == PERSONAL_ACTION_DEFER:
        return (
            PERSONAL_ACTION_ASK_CLARIFY,
            PERSONAL_ACTION_GATHER_EVIDENCE,
            PERSONAL_ACTION_SUGGEST,
        )
    return (
        PERSONAL_ACTION_ASK_CLARIFY,
        PERSONAL_ACTION_GATHER_EVIDENCE,
        PERSONAL_ACTION_DEFER,
        PERSONAL_ACTION_SUGGEST,
    )


def _select_entry_passing_option(
    *,
    actions: tuple[str, ...],
    best_score: float,
    ranked_options: list[dict[str, Any]],
    advisory_by_candidate_id: dict[str, dict[str, Any]],
    eligible: list[CandidateDecision],
    enforce_score_gap: bool,
) -> tuple[CandidateDecision, dict[str, Any]] | None:
    for target_action in actions:
        max_gap = _as_float(ACTION_ENTRY_SWITCH_SCORE_GAP.get(target_action), 0.0)
        for option in ranked_options:
            action = _as_text(option.get("action"))
            if action != target_action:
                continue

            option_score = _as_float(option.get("score"), 0.0)
            if enforce_score_gap and option_score + max_gap < best_score:
                continue

            candidate_id = _as_text(option.get("candidate_id"))
            if not candidate_id:
                continue
            candidate = _candidate_by_id(candidate_id, eligible=eligible)
            if candidate is None:
                continue
            advisory = advisory_by_candidate_id.get(candidate.id)
            if not isinstance(advisory, dict):
                continue
            assessment = _entry_assessment_from_advisory(advisory)
            if not _entry_assessment_passes(assessment):
                continue
            return candidate, advisory
    return None


def _candidate_by_id(candidate_id: str, *, eligible: list[CandidateDecision]) -> CandidateDecision | None:
    for candidate in eligible:
        if candidate.id == candidate_id:
            return candidate
    return None


def _evaluate_action_entry_assessment(
    *,
    action: str,
    advisory: dict[str, Any],
    question: str,
) -> dict[str, Any]:
    normalized_action = _as_text(action)
    reasons: list[str] = []
    confidence = _as_float(advisory.get("confidence"), 0.0)
    question_present = bool(_as_text(question))
    question_profile = _build_question_profile(question)
    complete_context = _question_profile_is_complete(question_profile)

    min_confidence = _as_float(
        ACTION_ENTRY_MIN_CONFIDENCE.get(normalized_action),
        0.0,
    )
    if normalized_action == PERSONAL_ACTION_SUGGEST and complete_context:
        min_confidence = min(min_confidence, 0.60)
    if confidence < min_confidence:
        reasons.append("confidence_below_min_threshold")

    max_confidence = ACTION_ENTRY_MAX_CONFIDENCE.get(normalized_action)
    if isinstance(max_confidence, (float, int)) and confidence > float(max_confidence):
        reasons.append("confidence_above_action_threshold")

    if normalized_action == PERSONAL_ACTION_SUGGEST:
        if not _suggest_contract_complete(advisory):
            reasons.append("suggest_contract_incomplete")
        generic_reasons = _suggest_generic_reasons(
            advisory,
            question=question,
        )
        reasons.extend(generic_reasons)
    elif normalized_action == PERSONAL_ACTION_ASK_CLARIFY:
        if not _clarifying_contract_complete(advisory):
            reasons.append("clarifying_questions_incomplete")
        if complete_context and not _question_indicates_explicit_uncertainty(question):
            reasons.append("clarify_not_necessary_with_complete_context")
    elif normalized_action == PERSONAL_ACTION_GATHER_EVIDENCE:
        if not _evidence_contract_complete(advisory):
            reasons.append("evidence_plan_incomplete")
        else:
            reasons.extend(
                _evidence_semantic_reasons(
                    advisory,
                    question=question,
                )
            )
    elif normalized_action == PERSONAL_ACTION_DEFER:
        if not _defer_contract_complete(advisory):
            reasons.append("defer_plan_incomplete")

    return {
        "action": normalized_action,
        "passes": not reasons,
        "reasons": reasons,
        "confidence": confidence,
        "question_present": question_present,
        "question_profile": question_profile,
    }


def _entry_assessment_from_advisory(advisory: dict[str, Any]) -> dict[str, Any]:
    payload = advisory.get("action_entry_assessment")
    return dict(payload) if isinstance(payload, dict) else {}


def _entry_assessment_passes(assessment: dict[str, Any]) -> bool:
    return bool(assessment.get("passes"))


def _suggest_contract_complete(advisory: dict[str, Any]) -> bool:
    complete_fields = True
    for key in SUGGEST_REQUIRED_TEXT_FIELDS:
        if not _as_text(advisory.get(key)):
            complete_fields = False
            break
    if complete_fields:
        for key in SUGGEST_REQUIRED_LIST_FIELDS:
            if not _normalize_text_list(advisory.get(key)):
                complete_fields = False
                break
    if complete_fields:
        return True
    return _decision_brain_report_contract_complete(advisory.get("decision_brain_report"))


def _clarifying_contract_complete(advisory: dict[str, Any]) -> bool:
    questions = advisory.get("clarifying_questions")
    if not isinstance(questions, list):
        return False
    if len(questions) != 3:
        return False
    for item in questions:
        if not isinstance(item, dict):
            return False
        if not _as_text(item.get("question")):
            return False
        if not _as_text(item.get("why")):
            return False
    return True


def _evidence_contract_complete(advisory: dict[str, Any]) -> bool:
    plan = advisory.get("evidence_plan")
    if not isinstance(plan, list):
        return False
    if len(plan) != 3:
        return False
    for item in plan:
        if not isinstance(item, dict):
            return False
        if not _as_text(item.get("fact")):
            return False
        if not _as_text(item.get("why")):
            return False
    return True


def _evidence_semantic_reasons(advisory: dict[str, Any], *, question: str) -> list[str]:
    plan = advisory.get("evidence_plan")
    if not isinstance(plan, list):
        return ["evidence_plan_incomplete"]

    reasons: list[str] = []
    user_entities = _question_entity_tokens(question)

    for item in plan:
        if not isinstance(item, dict):
            continue
        fact = _as_text(item.get("fact"))
        why = _as_text(item.get("why"))
        combined = f"{fact} {why}".strip()
        if _contains_internal_runtime_evidence_term(combined):
            reasons.append("evidence_internal_runtime_semantics")
            continue
        if not _references_user_entities(fact, question=question, user_entities=user_entities):
            reasons.append("evidence_missing_user_entity_binding")
        if not _has_real_world_evidence_signal(fact):
            reasons.append("evidence_not_externally_verifiable")
        if not _why_explains_recommendation_impact(
            why,
            question=question,
            user_entities=user_entities,
        ):
            reasons.append("evidence_why_missing_ranking_impact")

    return list(dict.fromkeys(reasons))


def _contains_internal_runtime_evidence_term(text: str) -> bool:
    lowered = _as_text(text).lower()
    if not lowered:
        return False
    for marker in EVIDENCE_INTERNAL_RUNTIME_TOKENS:
        token = _as_text(marker).lower()
        if not token:
            continue
        if " " in token or "_" in token or "-" in token:
            if token in lowered:
                return True
            continue
        if re.search(rf"\b{re.escape(token)}\b", lowered):
            return True
    return False


def _question_entity_tokens(question: str) -> tuple[str, ...]:
    lowered_question = _as_text(question).lower()
    if not lowered_question:
        return ()

    tokens: list[str] = []
    for marker in EVIDENCE_REAL_WORLD_OBJECT_TOKENS:
        normalized = _as_text(marker).lower()
        if normalized and normalized in lowered_question:
            tokens.append(normalized)

    for marker in _question_signal_tokens(lowered_question):
        token = _as_text(marker).lower()
        if not token:
            continue
        if token in QUESTION_ENTITY_STOPWORDS:
            continue
        if token:
            tokens.append(token)

    for label in _extract_alternative_labels(question):
        normalized = _as_text(label).lower()
        if not normalized:
            continue
        tokens.extend(
            [
                f"option {normalized}",
                f"offer {normalized}",
                f"方案{normalized}",
                f"选项{normalized}",
            ]
        )

    return tuple(dict.fromkeys(tokens))


def _references_user_entities(
    text: str,
    *,
    question: str,
    user_entities: tuple[str, ...] | None = None,
) -> bool:
    lowered = _as_text(text).lower()
    if not lowered:
        return False
    entities = user_entities if isinstance(user_entities, tuple) else _question_entity_tokens(question)
    for entity in entities:
        token = _as_text(entity).lower()
        if token and token in lowered:
            return True
    if not entities:
        return False
    return any(token in lowered for token in EVIDENCE_REAL_WORLD_OBJECT_TOKENS)


def _has_real_world_evidence_signal(text: str) -> bool:
    lowered = _as_text(text).lower()
    if not lowered:
        return False
    if _contains_internal_runtime_evidence_term(lowered):
        return False
    has_action = any(token in lowered for token in EVIDENCE_REAL_WORLD_ACTION_TOKENS)
    has_object = any(token in lowered for token in EVIDENCE_REAL_WORLD_OBJECT_TOKENS)
    return has_action and has_object


def _why_explains_recommendation_impact(
    why: str,
    *,
    question: str,
    user_entities: tuple[str, ...] | None = None,
) -> bool:
    del question, user_entities
    lowered = _as_text(why).lower()
    if not lowered:
        return False
    if _contains_internal_runtime_evidence_term(lowered):
        return False
    has_impact_signal = any(token in lowered for token in EVIDENCE_RANKING_IMPACT_TOKENS)
    return has_impact_signal


def _defer_contract_complete(advisory: dict[str, Any]) -> bool:
    payload = advisory.get("defer_plan")
    if not isinstance(payload, dict):
        return False
    return bool(
        _as_text(payload.get("revisit_at"))
        and _as_text(payload.get("monitor_signal"))
        and _as_text(payload.get("resume_trigger"))
    )


def _is_generic_suggestion_text(text: str, *, question: str) -> bool:
    token = _as_text(text).lower()
    if not token:
        return True
    for phrase in GENERIC_SUGGESTION_PHRASES:
        if phrase in token:
            return True
    for pattern in GENERIC_SUGGESTION_WEAK_PATTERNS:
        if pattern in token:
            if not _has_decision_specific_signal(token, question=question):
                return True
    if not _has_decision_specific_signal(token, question=question):
        # Short, context-free text is usually generic in personal decision setting.
        if len(token) < 100:
            return True
    return False


def _suggest_generic_reasons(advisory: dict[str, Any], *, question: str) -> list[str]:
    reasons: list[str] = []
    suggestion_text = _as_text(advisory.get("suggestion_text"))
    decision_brain_report = advisory.get("decision_brain_report")
    has_structured_report = _decision_brain_report_contract_complete(decision_brain_report)
    if _is_generic_suggestion_text(suggestion_text, question=question):
        reasons.append("generic_suggestion_text")

    evidence_text = " ".join(
        [
            suggestion_text,
            _join_text_list(_normalize_text_list(advisory.get("benefits"))),
            _join_text_list(_normalize_text_list(advisory.get("risks"))),
            _join_text_list(_normalize_text_list(advisory.get("key_assumptions"))),
            _as_text(advisory.get("first_step_24h")),
            _as_text(advisory.get("stop_loss_trigger")),
            _as_text(advisory.get("change_mind_condition")),
            _decision_brain_report_text(decision_brain_report),
        ]
    ).strip()

    if not _has_decision_specific_signal(evidence_text, question=question):
        reasons.append("generic_missing_decision_object")
    if not _has_tradeoff_signal(advisory):
        reasons.append("generic_missing_tradeoff")

    first_step = _as_text(advisory.get("first_step_24h"))
    if has_structured_report and not _has_time_action_signal(first_step):
        first_step = _decision_brain_report_text(decision_brain_report)
    if not _has_time_action_signal(first_step):
        reasons.append("generic_missing_time_action")

    if not _as_text(advisory.get("stop_loss_trigger")) and not has_structured_report:
        reasons.append("generic_missing_stop_loss")
    if not _as_text(advisory.get("change_mind_condition")) and not has_structured_report:
        reasons.append("generic_missing_change_mind")

    if (
        len(suggestion_text) < 120
        and not _has_decision_specific_signal(suggestion_text, question=question)
        and not _has_decision_specific_signal(evidence_text, question=question)
    ):
        reasons.append("generic_context_free_short")
    return reasons


def _question_profile_is_complete(profile: dict[str, Any]) -> bool:
    if not isinstance(profile, dict):
        return False
    missing_slots = profile.get("missing_slots")
    if isinstance(missing_slots, list) and any(_as_text(item) for item in missing_slots):
        return False
    readiness = _as_float(profile.get("decision_readiness_score"), 0.0)
    return readiness >= 0.80


def _question_indicates_explicit_uncertainty(question: str) -> bool:
    token = _as_text(question).lower()
    if not token:
        return False
    hints = (
        "not sure",
        "unclear",
        "uncertain",
        "need more info",
        "maybe",
        "不知道",
        "不确定",
        "不清楚",
    )
    return any(hint in token for hint in hints)


def _has_tradeoff_signal(advisory: dict[str, Any]) -> bool:
    report = _normalize_decision_brain_report(advisory.get("decision_brain_report"))
    options = report.get("options") if isinstance(report, dict) else None
    if isinstance(options, list):
        for item in options:
            if not isinstance(item, dict):
                continue
            if _normalize_text_list(item.get("benefits")) and _normalize_text_list(item.get("risks")):
                return True
    segments = [
        _as_text(advisory.get("suggestion_text")),
        _join_text_list(_normalize_text_list(advisory.get("benefits"))),
        _join_text_list(_normalize_text_list(advisory.get("risks"))),
        _join_text_list(_normalize_text_list(advisory.get("key_assumptions"))),
        _decision_brain_report_text(report),
    ]
    merged = " ".join([segment for segment in segments if segment]).lower()
    if not merged:
        return False
    has_benefits = bool(_normalize_text_list(advisory.get("benefits")))
    has_risks = bool(_normalize_text_list(advisory.get("risks")))
    if has_benefits and has_risks:
        return True
    for marker in TRADEOFF_SIGNAL_TOKENS:
        token = _as_text(marker).lower()
        if token and token in merged:
            return True
    return False


def _has_time_action_signal(text: str) -> bool:
    token = _as_text(text).lower()
    if not token:
        return False
    has_time_anchor = any(marker in token for marker in TIME_ANCHOR_TOKENS)
    has_action_verb = any(marker in token for marker in ACTION_VERB_TOKENS)
    has_decision_object = _has_decision_specific_signal(token, question="")
    return has_action_verb and (has_time_anchor or has_decision_object)


def _has_decision_specific_signal(text: str, *, question: str) -> bool:
    lowered_text = _as_text(text).lower()
    lowered_question = _as_text(question).lower()
    for marker in DECISION_SPECIFIC_SIGNAL_TOKENS:
        normalized_marker = _as_text(marker).lower()
        if not normalized_marker:
            continue
        if normalized_marker in lowered_text:
            return True
    # Encourage grounding by reusing concrete question tokens.
    for raw in _question_signal_tokens(lowered_question):
        if raw and raw in lowered_text:
            return True
    return False


def _question_signal_tokens(question: str) -> tuple[str, ...]:
    token = _as_text(question).lower()
    if not token:
        return ()
    separators = "\n\t,.;:!?()[]{}<>/\\|\"'"
    for sep in separators:
        token = token.replace(sep, " ")
    parts = [part.strip() for part in token.split(" ") if part.strip()]
    filtered: list[str] = []
    for part in parts:
        if len(part) < 3:
            continue
        if part in {"what", "should", "could", "would", "please", "help", "choose"}:
            continue
        filtered.append(part)
    return tuple(filtered[:8])


def _normalize_text_list(value: Any) -> list[str]:
    if isinstance(value, str):
        token = value.strip()
        return [token] if token else []
    if not isinstance(value, (list, tuple)):
        return []
    normalized: list[str] = []
    for item in value:
        token = _as_text(item)
        if not token:
            continue
        normalized.append(token)
    return normalized


def _normalize_question_list(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, (list, tuple)):
        return []
    normalized: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, str):
            question = _as_text(item)
            if not question:
                continue
            normalized.append(
                {
                    "question": question,
                    "why": "",
                }
            )
            continue
        if not isinstance(item, dict):
            continue
        question = _as_text(item.get("question")) or _as_text(item.get("q"))
        why = _as_text(item.get("why")) or _as_text(item.get("reason"))
        if not question:
            continue
        normalized.append({"question": question, "why": why})
    return normalized


def _normalize_evidence_plan(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, (list, tuple)):
        return []
    normalized: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, str):
            fact = _as_text(item)
            if not fact:
                continue
            normalized.append(
                {
                    "fact": fact,
                    "why": "",
                }
            )
            continue
        if not isinstance(item, dict):
            continue
        fact = _as_text(item.get("fact")) or _as_text(item.get("item")) or _as_text(item.get("question"))
        why = _as_text(item.get("why")) or _as_text(item.get("reason"))
        if not fact:
            continue
        normalized.append({"fact": fact, "why": why})
    return normalized


def _join_text_list(values: list[str]) -> str:
    if not values:
        return ""
    return " ".join([_as_text(value) for value in values if _as_text(value)])


def _prioritize_recommended_option(
    options: list[dict[str, Any]],
    *,
    recommended_candidate_id: str,
) -> list[dict[str, Any]]:
    token = _as_text(recommended_candidate_id)
    if not token:
        return list(options)
    matching: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    for option in options:
        candidate_id = _as_text(option.get("candidate_id"))
        if candidate_id == token:
            matching.append(option)
        else:
            remaining.append(option)
    if not matching:
        return list(options)
    return matching + remaining


def _annotate_option_labels_and_ranks(options: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, option in enumerate(options, start=1):
        if not isinstance(option, dict):
            continue
        entry = dict(option)
        entry["option_rank"] = index
        label = _as_text(entry.get("option_label"))
        if not label:
            label = _rank_to_option_label(index)
        entry["option_label"] = label
        option_id = _as_text(entry.get("option_id"))
        if not option_id:
            option_id = _as_text(entry.get("candidate_id"))
        if option_id:
            entry["option_id"] = option_id
        normalized.append(entry)
    return normalized


def _recommended_option_label_from_options(
    options: list[dict[str, Any]],
    *,
    recommended_candidate_id: str,
) -> str:
    token = _as_text(recommended_candidate_id)
    for option in options:
        if not isinstance(option, dict):
            continue
        if _as_text(option.get("candidate_id")) != token:
            continue
        label = _as_text(option.get("option_label")) or _as_text(option.get("label"))
        if label:
            return label
        break
    if options:
        first = options[0]
        if isinstance(first, dict):
            return _as_text(first.get("option_label")) or _as_text(first.get("label"))
    return ""


def _rank_to_option_label(rank: int) -> str:
    normalized = max(1, rank)
    label = chr(ord("A") + min(normalized - 1, 25))
    return f"Option {label}"


def _build_decision_option_payload(
    *,
    candidate: CandidateDecision,
    advisory: dict[str, Any],
) -> dict[str, Any]:
    execution_brief = advisory.get("execution_brief")
    normalized_execution_brief = (
        dict(execution_brief)
        if isinstance(execution_brief, dict)
        else {}
    )
    entry_assessment = _entry_assessment_from_advisory(advisory)
    entry_reasons_raw = entry_assessment.get("reasons")
    entry_reasons = (
        [str(item).strip() for item in entry_reasons_raw if str(item).strip()]
        if isinstance(entry_reasons_raw, list)
        else []
    )
    return {
        "candidate_id": candidate.id,
        "option_id": candidate.id,
        "option_label": "",
        "option_rank": 0,
        "action": candidate.action,
        "score": _as_float(advisory.get("score"), candidate.score_total),
        "confidence": _as_float(advisory.get("confidence"), candidate.confidence),
        "urgency": _as_text(advisory.get("urgency")),
        "suggestion_text": _as_text(advisory.get("suggestion_text")),
        "simulation_rationale": _as_text(advisory.get("simulation_rationale")),
        "result_kind": _as_text(advisory.get("result_kind")),
        "risk": _as_float(candidate.risk, 0.0),
        "execution_brief": normalized_execution_brief,
        "benefits": _normalize_text_list(advisory.get("benefits")),
        "risks": _normalize_text_list(advisory.get("risks")),
        "key_assumptions": _normalize_text_list(advisory.get("key_assumptions")),
        "first_step_24h": _as_text(advisory.get("first_step_24h")),
        "stop_loss_trigger": _as_text(advisory.get("stop_loss_trigger")),
        "change_mind_condition": _as_text(advisory.get("change_mind_condition")),
        "recommendation_reason": _as_text(advisory.get("recommendation_reason")),
        "what_would_change_my_mind": _as_text(advisory.get("what_would_change_my_mind")),
        "decision_brain_report": _normalize_decision_brain_report(advisory.get("decision_brain_report")),
        "entry_contract_passed": bool(entry_assessment.get("passes")),
        "entry_contract_reasons": entry_reasons,
    }


def _extract_urgency(decision: Decision, artifact: dict[str, Any]) -> str:
    fields = (
        artifact.get("urgency"),
        decision.attributes.get("urgency"),
    )
    for value in fields:
        text = _as_text(value)
        if text:
            return text
    return ""


def _extract_confidence(decision: Decision, artifact: dict[str, Any], *, default: float) -> float:
    if "confidence" in artifact:
        return _as_float(artifact.get("confidence"), default)
    if "confidence" in decision.attributes:
        return _as_float(decision.attributes.get("confidence"), default)
    return default


def _extract_rationale(artifact: dict[str, Any]) -> str:
    fields = (
        artifact.get("simulation_rationale"),
        artifact.get("rationale"),
        artifact.get("summary"),
    )
    for value in fields:
        text = _as_text(value)
        if text:
            return text
    return ""


def _extract_result_kind(decision: Decision, artifact: dict[str, Any]) -> str:
    candidates = (
        artifact.get("result_kind"),
        decision.attributes.get("result_kind"),
    )
    for value in candidates:
        token = _normalize_result_kind(value)
        if token:
            return token
    return RESULT_KIND_SUGGESTION


def _extract_execution_brief(decision: Decision, artifact: dict[str, Any]) -> dict[str, Any]:
    fields = (
        artifact.get("execution_brief"),
        decision.attributes.get("execution_brief"),
    )
    for value in fields:
        if isinstance(value, dict):
            return dict(value)
    return {}


def _normalize_result_kind(value: Any) -> str:
    token = _as_text(value).lower()
    if token == RESULT_KIND_ACTION_PROPOSAL:
        return RESULT_KIND_ACTION_PROPOSAL
    return RESULT_KIND_SUGGESTION


def _limit_simulation_candidates(
    candidates: list[CandidateDecision],
    *,
    fanout_limit: int,
) -> list[CandidateDecision]:
    if not candidates:
        return []
    try:
        normalized_limit = int(fanout_limit)
    except Exception:
        normalized_limit = PERSONAL_SIMULATION_FANOUT_LIMIT_DEFAULT
    if normalized_limit <= 0:
        return list(candidates)
    return list(candidates[:normalized_limit])


def _normalize_personal_decision_action(action: str, *, domain: str) -> str:
    if domain.strip() != "personal.assistant":
        return action
    return PERSONAL_ACTION_ALIAS_MAP.get(action, action)


def _collect_decision_compatibility(
    *,
    adapter: LLMDecisionAdapter | None,
    action_normalization_events: list[dict[str, Any]],
) -> dict[str, Any]:
    field_fallback_used = bool(getattr(adapter, "_last_field_fallback_used", False))

    raw_field_fallback_events = getattr(adapter, "_last_field_fallback_events", [])
    field_fallback_events: list[dict[str, Any]] = []
    if isinstance(raw_field_fallback_events, list):
        for entry in raw_field_fallback_events:
            if not isinstance(entry, dict):
                continue
            normalized_entry = {
                "field_fallback_used": bool(entry.get("field_fallback_used")),
                "selected_action_fallback_used": bool(entry.get("selected_action_fallback_used")),
                "decision_type_fallback_used": bool(entry.get("decision_type_fallback_used")),
                "original_selected_action": _as_text(entry.get("original_selected_action")),
                "fallback_action": _as_text(entry.get("fallback_action")),
                "resolved_selected_action": _as_text(entry.get("resolved_selected_action")),
                "original_decision_type": _as_text(entry.get("original_decision_type")),
                "fallback_type": _as_text(entry.get("fallback_type")),
                "resolved_decision_type": _as_text(entry.get("resolved_decision_type")),
            }
            field_fallback_events.append(normalized_entry)

    action_events: list[dict[str, Any]] = []
    for entry in action_normalization_events:
        if not isinstance(entry, dict):
            continue
        action_events.append(
            {
                "original_action": _as_text(entry.get("original_action")),
                "normalized_action": _as_text(entry.get("normalized_action")),
                "alias_mapping_used": bool(entry.get("alias_mapping_used")),
            }
        )

    alias_mapping_used = any(bool(entry.get("alias_mapping_used")) for entry in action_events)
    return {
        "field_fallback_used": field_fallback_used,
        "alias_mapping_used": alias_mapping_used,
        "action_normalization_events": action_events,
        "field_fallback_events": field_fallback_events,
    }


def _compatibility_used(compatibility: dict[str, Any]) -> bool:
    if not isinstance(compatibility, dict):
        return False
    if bool(compatibility.get("field_fallback_used")):
        return True
    if bool(compatibility.get("alias_mapping_used")):
        return True
    return False


def _maybe_write_model_debug_artifact(
    *,
    stage: str,
    parsed_failure_reason: str,
    exc: Exception | None,
    adapter: Any,
    compatibility: dict[str, Any] | None = None,
    candidate_id: str = "",
    selected_action: str = "",
    timeout_seconds: float | None = None,
) -> None:
    if os.environ.get(PERSONAL_DEBUG_MODEL_IO_ENV) != "1":
        return

    timestamp = datetime.utcnow()
    stdout, stderr = _extract_model_io_for_debug(exc=exc, adapter=adapter)
    payload = {
        "timestamp": timestamp.isoformat(timespec="microseconds") + "Z",
        "stage": stage.strip() if isinstance(stage, str) else "",
        "parsed_failure_reason": parsed_failure_reason.strip()
        if isinstance(parsed_failure_reason, str)
        else "",
        "raw_stdout": stdout,
        "raw_stderr": stderr,
        "candidate_id": candidate_id.strip() if isinstance(candidate_id, str) else "",
        "selected_action": selected_action.strip() if isinstance(selected_action, str) else "",
        "timeout_seconds": timeout_seconds if isinstance(timeout_seconds, (int, float)) else None,
        "field_fallback_used": bool((compatibility or {}).get("field_fallback_used")),
        "alias_mapping_used": bool((compatibility or {}).get("alias_mapping_used")),
        "action_normalization_events": list((compatibility or {}).get("action_normalization_events", [])),
        "field_fallback_events": list((compatibility or {}).get("field_fallback_events", [])),
    }

    try:
        PERSONAL_DEBUG_MODEL_IO_DIR.mkdir(parents=True, exist_ok=True)
        output_path = PERSONAL_DEBUG_MODEL_IO_DIR / (
            f"model_debug_{timestamp.strftime('%Y%m%dT%H%M%S%fZ')}.json"
        )
        output_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception:
        return


def _extract_model_io_for_debug(
    *,
    exc: Exception | None,
    adapter: Any,
) -> tuple[str, str]:
    for current in _iter_exception_chain(exc):
        stdout = getattr(current, _MODEL_STDOUT_ATTR, None)
        stderr = getattr(current, _MODEL_STDERR_ATTR, None)
        if isinstance(stdout, str) or isinstance(stderr, str):
            return (
                stdout if isinstance(stdout, str) else "",
                stderr if isinstance(stderr, str) else "",
            )

    if adapter is not None:
        adapter_stdout = getattr(adapter, "_last_model_stdout", "")
        adapter_stderr = getattr(adapter, "_last_model_stderr", "")
        return (
            adapter_stdout if isinstance(adapter_stdout, str) else "",
            adapter_stderr if isinstance(adapter_stderr, str) else "",
        )
    return "", ""


def _resolve_simulation_timeout_seconds(
    *,
    adapter: Any,
    domain: str,
) -> float | None:
    if adapter is None:
        return None

    adapter_timeout = getattr(adapter, "_last_timeout_seconds", None)
    if isinstance(adapter_timeout, (int, float)):
        return float(adapter_timeout)

    client = getattr(adapter, "client", None)
    if client is None:
        return None

    try:
        model_config = client.resolve_model_config(
            LLMTaskHook.SIMULATION_ADVISE,
            domain=domain,
            model_override=getattr(adapter, "model_override", None),
        )
    except Exception:
        return None

    timeout = getattr(model_config, "timeout_sec", None)
    if isinstance(timeout, (int, float)):
        return float(timeout)
    return None


def _iter_exception_chain(exc: Exception | None) -> list[Exception]:
    if exc is None:
        return []
    chain: list[Exception] = []
    current: Exception | None = exc
    seen: set[int] = set()
    while isinstance(current, Exception):
        current_id = id(current)
        if current_id in seen:
            break
        seen.add(current_id)
        chain.append(current)
        next_exc = current.__cause__
        if isinstance(next_exc, Exception):
            current = next_exc
            continue
        next_exc = current.__context__
        if isinstance(next_exc, Exception):
            current = next_exc
            continue
        break
    return chain


def _as_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""
