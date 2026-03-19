from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable
from uuid import uuid4

from spice.core import SpiceRuntime
from spice.protocols import Decision, ExecutionIntent, ExecutionResult, Observation, Outcome
from spice_personal.execution.execution_intent_v1 import ensure_minimal_execution_result_output


PERSONAL_ACTION_GATHER_EVIDENCE = "personal.assistant.gather_evidence"
PERSONAL_OBSERVATION_EVIDENCE_RECEIVED = "personal.assistant.evidence_received"
PERSONAL_OBSERVATION_EVIDENCE_CHECKLIST_PREPARED = "personal.assistant.evidence_checklist_prepared"
INTERNAL_RUNTIME_EVIDENCE_MARKERS = (
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
    "risk_budget",
    "active session",
    "current state",
    "hypothesis",
    "prepared evidence",
)


@dataclass(slots=True, frozen=True)
class EvidenceExecutionPolicy:
    max_evidence_rounds: int = 1
    timeout_seconds: float = 20.0
    max_output_chars: int = 4000
    max_summary_chars: int = 600
    max_item_chars: int = 240
    max_items: int = 6
    max_source_refs: int = 10
    allowed_operation_prefixes: tuple[str, ...] = (
        "personal.gather_evidence",
        "read",
        "query",
        "inspect",
        "search",
        "script.small",
    )


DEFAULT_EVIDENCE_POLICY = EvidenceExecutionPolicy()


@dataclass(slots=True)
class EvidenceRoundResult:
    requested: bool
    notice: str | None = None
    evidence_observation: Observation | None = None
    execution_intent: ExecutionIntent | None = None
    execution_result: ExecutionResult | None = None
    execution_outcome: Outcome | None = None


def run_mock_evidence_round(
    *,
    decision: Decision,
    source: str,
    policy: EvidenceExecutionPolicy = DEFAULT_EVIDENCE_POLICY,
    observation_type: str = PERSONAL_OBSERVATION_EVIDENCE_CHECKLIST_PREPARED,
    notice: str | None = None,
) -> EvidenceRoundResult:
    if not should_gather_evidence(decision):
        return EvidenceRoundResult(requested=False)

    evidence_plan = _extract_evidence_plan_from_decision(decision, max_items=policy.max_items)
    evidence_plan = _sanitize_manual_evidence_plan(evidence_plan)
    if not evidence_plan:
        evidence_plan = _default_manual_evidence_plan()
    observation = normalize_execution_result_to_evidence_observation(
        decision=decision,
        intent=None,
        execution_result=None,
        execution_outcome=None,
        source=source,
        policy=policy,
        observation_type=observation_type,
        error="manual_evidence_required",
        evidence_plan=evidence_plan,
        manual_round=True,
    )
    manual_notice = (
        notice
        if isinstance(notice, str) and notice.strip()
        else (
            "Evidence action selected, but external evidence agent is not configured (executor=mock). "
            "Prepared a manual evidence checklist and re-evaluated once."
        )
    )
    return EvidenceRoundResult(
        requested=True,
        notice=manual_notice,
        evidence_observation=observation,
        execution_intent=None,
        execution_result=None,
        execution_outcome=None,
    )


def should_gather_evidence(
    decision: Decision,
    *,
    evidence_action: str = PERSONAL_ACTION_GATHER_EVIDENCE,
) -> bool:
    selected_action = decision.selected_action or ""
    return selected_action.strip() == evidence_action


def is_operation_allowed(
    operation_name: str,
    *,
    policy: EvidenceExecutionPolicy = DEFAULT_EVIDENCE_POLICY,
) -> bool:
    normalized = operation_name.strip().lower()
    if not normalized:
        return False

    for prefix in policy.allowed_operation_prefixes:
        token = prefix.strip().lower()
        if not token:
            continue
        if normalized == token:
            return True
        if normalized.startswith(token + "."):
            return True
        if normalized.startswith(token + "_"):
            return True
        if normalized.startswith(token + "-"):
            return True
    return False


def run_bounded_evidence_round(
    runtime: SpiceRuntime,
    *,
    decision: Decision,
    source: str,
    policy: EvidenceExecutionPolicy = DEFAULT_EVIDENCE_POLICY,
    observation_type: str = PERSONAL_OBSERVATION_EVIDENCE_RECEIVED,
    prepare_intent: Callable[[ExecutionIntent], None] | None = None,
) -> EvidenceRoundResult:
    if not should_gather_evidence(decision):
        return EvidenceRoundResult(requested=False)

    notice = "Gathering one bounded evidence snapshot before final advice."
    intent: ExecutionIntent | None = None
    execution_result: ExecutionResult | None = None
    execution_outcome: Outcome | None = None
    normalization_error: str | None = None

    try:
        intent = runtime.plan_execution(decision)
    except Exception as exc:
        observation = normalize_execution_result_to_evidence_observation(
            decision=decision,
            intent=None,
            execution_result=None,
            execution_outcome=None,
            source=source,
            policy=policy,
            observation_type=observation_type,
            error=f"planning_failed: {exc}",
        )
        return EvidenceRoundResult(
            requested=True,
            notice=notice,
            evidence_observation=observation,
            execution_intent=None,
            execution_result=None,
            execution_outcome=None,
        )

    if prepare_intent is not None:
        try:
            prepare_intent(intent)
        except Exception as exc:
            observation = normalize_execution_result_to_evidence_observation(
                decision=decision,
                intent=intent,
                execution_result=None,
                execution_outcome=None,
                source=source,
                policy=policy,
                observation_type=observation_type,
                error=f"intent_preparation_failed: {exc}",
            )
            return EvidenceRoundResult(
                requested=True,
                notice=notice,
                evidence_observation=observation,
                execution_intent=intent,
                execution_result=None,
                execution_outcome=None,
            )

    _enforce_intent_guardrails(intent, policy=policy)
    operation_name = _operation_name(intent)
    if not is_operation_allowed(operation_name, policy=policy):
        observation = normalize_execution_result_to_evidence_observation(
            decision=decision,
            intent=intent,
            execution_result=None,
            execution_outcome=None,
            source=source,
            policy=policy,
            observation_type=observation_type,
            error=f"operation_not_allowed: {operation_name}",
        )
        return EvidenceRoundResult(
            requested=True,
            notice=notice,
            evidence_observation=observation,
            execution_intent=intent,
            execution_result=None,
            execution_outcome=None,
        )

    try:
        execution_result = runtime.execute(intent)
        ensure_minimal_execution_result_output(
            execution_result,
            intent=intent,
            decision=decision,
            category="external.evidence",
        )
    except Exception as exc:
        normalization_error = f"execution_failed: {exc}"
    else:
        try:
            execution_outcome = runtime.process_execution_result(
                execution_result,
                decision=decision,
                intent=intent,
            )
        except Exception as exc:
            normalization_error = _join_errors(
                normalization_error,
                f"result_interpretation_failed: {exc}",
            )

    timed_out = _execution_timed_out(execution_result)
    if timed_out:
        normalization_error = _join_errors(normalization_error, "timeout")
    if execution_result is not None and execution_result.error:
        normalization_error = _join_errors(normalization_error, execution_result.error)

    observation = normalize_execution_result_to_evidence_observation(
        decision=decision,
        intent=intent,
        execution_result=execution_result,
        execution_outcome=execution_outcome,
        source=source,
        policy=policy,
        observation_type=observation_type,
        error=normalization_error,
        timed_out=timed_out,
    )

    return EvidenceRoundResult(
        requested=True,
        notice=notice,
        evidence_observation=observation,
        execution_intent=intent,
        execution_result=execution_result,
        execution_outcome=execution_outcome,
    )


def normalize_execution_result_to_evidence_observation(
    *,
    decision: Decision,
    intent: ExecutionIntent | None,
    execution_result: ExecutionResult | None,
    execution_outcome: Outcome | None,
    source: str,
    policy: EvidenceExecutionPolicy = DEFAULT_EVIDENCE_POLICY,
    observation_type: str = PERSONAL_OBSERVATION_EVIDENCE_RECEIVED,
    error: str | None = None,
    timed_out: bool = False,
    evidence_plan: list[dict[str, str]] | None = None,
    manual_round: bool = False,
) -> Observation:
    operation_name = _operation_name(intent)
    source_refs = _build_source_refs(
        decision=decision,
        intent=intent,
        execution_result=execution_result,
        execution_outcome=execution_outcome,
        max_refs=policy.max_source_refs,
    )
    summary = _build_summary(
        execution_result=execution_result,
        error=error,
        policy=policy,
    )
    evidence_items = _build_evidence_items(
        execution_result=execution_result,
        policy=policy,
        evidence_plan=evidence_plan,
    )
    if manual_round:
        if evidence_items:
            summary = (
                f"Manual evidence checklist prepared ({len(evidence_items)} items). "
                "Awaiting externally verifiable sources."
            )
        else:
            summary = "Manual evidence checklist pending; provide externally verifiable sources."
    confidence = _estimate_evidence_confidence(
        execution_result=execution_result,
        summary=summary,
        has_error=bool(error),
        timed_out=timed_out,
        evidence_items=evidence_items,
    )

    attributes: dict[str, Any] = {
        "evidence_summary": summary,
        "source_refs": source_refs,
        "execution_id": execution_result.id if execution_result is not None else "",
        "intent_id": intent.id if intent is not None else "",
        "evidence_confidence": confidence,
        "evidence_mode": "manual_checklist" if manual_round else "external_execution",
    }

    if evidence_items:
        attributes["evidence_items"] = evidence_items
        attributes["evidence_item_count"] = len(evidence_items)
        attributes["evidence_source_coverage"] = _evidence_source_coverage(evidence_items)
    if execution_result is not None:
        attributes["execution_status"] = execution_result.status
        if operation_name:
            attributes["operation_name"] = operation_name
        executor_name = _as_text(execution_result.executor)
        if executor_name:
            attributes["executor"] = executor_name
    if error:
        attributes["error"] = _truncate_text(error, policy.max_item_chars)

    metadata: dict[str, Any] = {
        "mode": "personal_advisor",
        "max_evidence_rounds": policy.max_evidence_rounds,
        "timeout_seconds": policy.timeout_seconds,
        "max_output_chars": policy.max_output_chars,
    }
    if timed_out:
        metadata["timed_out"] = True
    if manual_round:
        metadata["manual_round"] = True

    return Observation(
        id=f"obs-{uuid4().hex}",
        observation_type=observation_type,
        source=source,
        attributes=attributes,
        metadata=metadata,
    )


def _enforce_intent_guardrails(
    intent: ExecutionIntent,
    *,
    policy: EvidenceExecutionPolicy,
) -> None:
    parameters = dict(intent.parameters) if isinstance(intent.parameters, dict) else {}
    timeout_value = _as_float(parameters.get("timeout_seconds"), policy.timeout_seconds)
    if timeout_value <= 0:
        timeout_value = policy.timeout_seconds
    parameters["timeout_seconds"] = min(timeout_value, policy.timeout_seconds)

    max_output_raw = int(_as_float(parameters.get("max_output_chars"), float(policy.max_output_chars)))
    if max_output_raw <= 0:
        max_output_raw = policy.max_output_chars
    parameters["max_output_chars"] = min(max_output_raw, policy.max_output_chars)
    intent.parameters = parameters

    constraints = [entry for entry in intent.constraints if isinstance(entry, dict)]
    constraints = _upsert_constraint(
        constraints,
        name="evidence.max_rounds",
        kind="count_limit",
        params={"max_rounds": policy.max_evidence_rounds},
    )
    constraints = _upsert_constraint(
        constraints,
        name="evidence.timeout_seconds",
        kind="time_budget",
        params={"max_seconds": policy.timeout_seconds},
    )
    constraints = _upsert_constraint(
        constraints,
        name="evidence.max_output_chars",
        kind="size_cap",
        params={"max_chars": policy.max_output_chars},
    )
    intent.constraints = constraints

    failure_policy = dict(intent.failure_policy) if isinstance(intent.failure_policy, dict) else {}
    failure_policy["strategy"] = "fail_fast"
    failure_policy["max_retries"] = 0
    intent.failure_policy = failure_policy


def _upsert_constraint(
    constraints: list[dict[str, Any]],
    *,
    name: str,
    kind: str,
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    remaining = [
        entry
        for entry in constraints
        if _as_text(entry.get("name")) != name
    ]
    remaining.append(
        {
            "name": name,
            "kind": kind,
            "params": dict(params),
        }
    )
    return remaining


def _operation_name(intent: ExecutionIntent | None) -> str:
    if intent is None:
        return ""
    operation = intent.operation if isinstance(intent.operation, dict) else {}
    name = _as_text(operation.get("name"))
    if name:
        return name
    return _as_text(intent.intent_type)


def _execution_timed_out(result: ExecutionResult | None) -> bool:
    if result is None:
        return False

    if "timeout" in _as_text(result.error).lower():
        return True
    if "timed out" in _as_text(result.error).lower():
        return True

    attributes = result.attributes if isinstance(result.attributes, dict) else {}
    cli_payload = attributes.get("cli_adapter")
    if isinstance(cli_payload, dict):
        capture = cli_payload.get("capture")
        if isinstance(capture, dict) and bool(capture.get("timed_out")):
            return True

    sdep_payload = attributes.get("sdep")
    if isinstance(sdep_payload, dict):
        response = sdep_payload.get("response")
        if isinstance(response, dict):
            status = _as_text(response.get("status")).lower()
            if status == "timeout":
                return True

    return False


def _build_summary(
    *,
    execution_result: ExecutionResult | None,
    error: str | None,
    policy: EvidenceExecutionPolicy,
) -> str:
    if execution_result is None:
        if error:
            return _truncate_text(
                f"Evidence attempt unavailable: {error}",
                policy.max_summary_chars,
            )
        return "Evidence attempt completed without execution output."

    output = execution_result.output if isinstance(execution_result.output, dict) else {}
    for key in ("evidence_summary", "summary", "text", "message"):
        value = output.get(key)
        text = _value_to_text(value, max_chars=policy.max_output_chars)
        if text:
            return _truncate_text(text, policy.max_summary_chars)

    if output:
        serialized = _value_to_text(output, max_chars=policy.max_output_chars)
        if serialized:
            return _truncate_text(
                f"Evidence snapshot: {serialized}",
                policy.max_summary_chars,
            )

    status = _as_text(execution_result.status) or "unknown"
    if error:
        return _truncate_text(
            f"Evidence attempt status={status}: {error}",
            policy.max_summary_chars,
        )
    return _truncate_text(
        f"Evidence attempt status={status}.",
        policy.max_summary_chars,
    )


def _build_source_refs(
    *,
    decision: Decision,
    intent: ExecutionIntent | None,
    execution_result: ExecutionResult | None,
    execution_outcome: Outcome | None,
    max_refs: int,
) -> list[str]:
    refs: list[str] = []
    candidates: list[Any] = [decision.id]
    candidates.extend(decision.refs)
    if intent is not None:
        candidates.append(intent.id)
        candidates.extend(intent.refs)
    if execution_result is not None:
        candidates.append(execution_result.id)
        candidates.extend(execution_result.refs)
    if execution_outcome is not None:
        candidates.append(execution_outcome.id)
        candidates.extend(execution_outcome.refs)

    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        token = candidate.strip()
        if not token:
            continue
        refs.append(token)

    deduped = list(dict.fromkeys(refs))
    return deduped[:max(1, max_refs)]


def _build_evidence_items(
    *,
    execution_result: ExecutionResult | None,
    policy: EvidenceExecutionPolicy,
    evidence_plan: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    output = execution_result.output if isinstance(getattr(execution_result, "output", None), dict) else {}
    candidates: list[Any] = []
    for key in ("evidence_items", "items", "results", "hits"):
        value = output.get(key)
        if isinstance(value, list):
            candidates = value
            break
    if not candidates and output and execution_result is not None:
        candidates = [output]
    if not candidates and isinstance(evidence_plan, list):
        candidates = list(evidence_plan)

    items: list[dict[str, Any]] = []
    for index, value in enumerate(candidates[: policy.max_items]):
        item = _normalize_evidence_item(
            value,
            index=index + 1,
            policy=policy,
            from_manual_plan=(
                execution_result is None
                and isinstance(evidence_plan, list)
                and value in evidence_plan
            ),
        )
        if not item:
            continue
        items.append(item)
    return items


def _estimate_evidence_confidence(
    *,
    execution_result: ExecutionResult | None,
    summary: str,
    has_error: bool,
    timed_out: bool,
    evidence_items: list[dict[str, Any]] | None = None,
) -> float:
    if execution_result is None:
        base = 0.10
    else:
        status = _as_text(execution_result.status).lower()
        if status == "success":
            base = 0.78
        elif status == "failed":
            base = 0.18
        else:
            base = 0.40

    normalized_items = evidence_items if isinstance(evidence_items, list) else []
    if normalized_items:
        item_confidences: list[float] = []
        for item in normalized_items:
            if not isinstance(item, dict):
                continue
            item_confidences.append(_as_float(item.get("confidence"), 0.0))
        if item_confidences:
            avg_item_conf = sum(item_confidences) / len(item_confidences)
            base = (base * 0.35) + (avg_item_conf * 0.65)
        coverage = _evidence_source_coverage(normalized_items)
        if coverage <= 0.0:
            base = min(base, 0.30)
        elif coverage < 0.5:
            base = min(base, 0.45)
    elif execution_result is not None:
        base = min(base, 0.30)

    if has_error:
        base = min(base, 0.25)
    if timed_out:
        base = min(base, 0.15)
    if not summary.strip():
        base = min(base, 0.20)
    return round(max(0.0, min(1.0, base)), 2)


def _extract_evidence_plan_from_decision(
    decision: Decision,
    *,
    max_items: int,
) -> list[dict[str, str]]:
    attributes = decision.attributes if isinstance(decision.attributes, dict) else {}
    payload = attributes.get("evidence_plan")
    if not isinstance(payload, list):
        return []
    normalized: list[dict[str, str]] = []
    for item in payload[: max(1, max_items)]:
        if isinstance(item, str):
            fact = _as_text(item)
            if not fact:
                continue
            normalized.append({"fact": fact, "why": ""})
            continue
        if not isinstance(item, dict):
            continue
        fact = _as_text(item.get("fact")) or _as_text(item.get("item")) or _as_text(item.get("question"))
        why = _as_text(item.get("why")) or _as_text(item.get("reason"))
        if not fact:
            continue
        normalized.append({"fact": fact, "why": why})
    return normalized


def _sanitize_manual_evidence_plan(plan: list[dict[str, str]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for item in plan[:3]:
        if not isinstance(item, dict):
            continue
        fact = _as_text(item.get("fact"))
        why = _as_text(item.get("why"))
        if not fact or not why:
            continue
        if _contains_internal_runtime_evidence_marker(f"{fact} {why}"):
            continue
        normalized.append({"fact": fact, "why": why})
    if len(normalized) != 3:
        return []
    return normalized


def _default_manual_evidence_plan() -> list[dict[str, str]]:
    return [
        {
            "fact": "Verify team stability and attrition trends for each option over the last 12 months.",
            "why": "It can change the risk ranking between options.",
        },
        {
            "fact": "Confirm manager coaching quality and mentorship outcomes from direct references.",
            "why": "It changes the probability of management-path growth.",
        },
        {
            "fact": "Confirm first-6-month scope ownership and promotion-path expectations in writing.",
            "why": "It can reorder recommendations when growth leverage differs.",
        },
    ]


def _contains_internal_runtime_evidence_marker(text: str) -> bool:
    lowered = _as_text(text).lower()
    if not lowered:
        return False
    for marker in INTERNAL_RUNTIME_EVIDENCE_MARKERS:
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


def _normalize_evidence_item(
    value: Any,
    *,
    index: int,
    policy: EvidenceExecutionPolicy,
    from_manual_plan: bool,
) -> dict[str, Any]:
    claim = ""
    why = ""
    source = ""
    title = ""
    url = ""
    published_at = ""
    date = ""
    reliability_value: float | None = None
    confidence_value: float | None = None

    if isinstance(value, str):
        claim = _truncate_text(_as_text(value), policy.max_item_chars)
    elif isinstance(value, dict):
        claim = _truncate_text(
            _as_text(
                value.get("claim")
                or value.get("fact")
                or value.get("text")
                or value.get("summary")
                or value.get("message")
                or value.get("item")
            ),
            policy.max_item_chars,
        )
        why = _truncate_text(
            _as_text(value.get("why") or value.get("reason") or value.get("rationale")),
            policy.max_item_chars,
        )
        source = _truncate_text(
            _as_text(value.get("source") or value.get("publisher") or value.get("origin")),
            policy.max_item_chars,
        )
        title = _truncate_text(
            _as_text(value.get("title") or value.get("source_title") or value.get("name")),
            policy.max_item_chars,
        )
        url = _truncate_text(
            _as_text(value.get("url") or value.get("link") or value.get("source_url")),
            policy.max_output_chars,
        )
        published_at = _truncate_text(
            _as_text(value.get("published_at") or value.get("published") or value.get("timestamp")),
            policy.max_item_chars,
        )
        date = _truncate_text(_as_text(value.get("date")), policy.max_item_chars)
        reliability_value = _parse_reliability(value.get("reliability"))
        confidence_value = _parse_reliability(value.get("confidence"))

        source_payload = value.get("source")
        if isinstance(source_payload, dict):
            source = source or _truncate_text(
                _as_text(
                    source_payload.get("name")
                    or source_payload.get("source")
                    or source_payload.get("publisher")
                    or source_payload.get("site")
                ),
                policy.max_item_chars,
            )
            title = title or _truncate_text(
                _as_text(source_payload.get("title") or source_payload.get("name")),
                policy.max_item_chars,
            )
            url = url or _truncate_text(
                _as_text(source_payload.get("url") or source_payload.get("link")),
                policy.max_output_chars,
            )
            published_at = published_at or _truncate_text(
                _as_text(source_payload.get("published_at") or source_payload.get("date")),
                policy.max_item_chars,
            )

    if not claim:
        return {}
    if not date:
        date = published_at

    reliability = _resolve_item_reliability(
        explicit=reliability_value,
        source=source,
        title=title,
        url=url,
        published_at=published_at,
        from_manual_plan=from_manual_plan,
    )
    confidence = _resolve_item_confidence(
        explicit=confidence_value,
        reliability=reliability,
        source=source,
        url=url,
    )

    item: dict[str, Any] = {
        "id": f"item-{index}",
        "claim": claim,
        "source": source,
        "title": title,
        "url": url,
        "published_at": published_at,
        "date": date,
        "reliability": reliability,
        "confidence": confidence,
    }
    if why:
        item["why"] = why
    return item


def _resolve_item_reliability(
    *,
    explicit: float | None,
    source: str,
    title: str,
    url: str,
    published_at: str,
    from_manual_plan: bool,
) -> float:
    if isinstance(explicit, float):
        resolved = max(0.0, min(1.0, explicit))
    else:
        has_source = bool(_as_text(source))
        has_title = bool(_as_text(title))
        has_url = bool(_as_text(url))
        has_date = bool(_as_text(published_at))
        if has_source and has_url:
            resolved = 0.80
        elif has_source or has_title:
            resolved = 0.62
        elif has_url:
            resolved = 0.55
        else:
            resolved = 0.25
        if has_date:
            resolved = min(0.95, resolved + 0.05)
    if from_manual_plan and not (_as_text(source) or _as_text(url)):
        resolved = min(resolved, 0.20)
    return round(max(0.0, min(1.0, resolved)), 2)


def _resolve_item_confidence(
    *,
    explicit: float | None,
    reliability: float,
    source: str,
    url: str,
) -> float:
    if isinstance(explicit, float):
        resolved = max(0.0, min(1.0, explicit))
    else:
        resolved = reliability
    if not (_as_text(source) or _as_text(url)):
        resolved = min(resolved, 0.35)
    return round(max(0.0, min(1.0, resolved)), 2)


def _parse_reliability(value: Any) -> float | None:
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if not token:
            return None
        if token in {"high", "strong", "credible"}:
            return 0.85
        if token in {"medium", "moderate"}:
            return 0.60
        if token in {"low", "weak", "uncertain"}:
            return 0.35
        try:
            return float(token)
        except ValueError:
            return None
    return None


def _evidence_source_coverage(evidence_items: list[dict[str, Any]]) -> float:
    if not evidence_items:
        return 0.0
    sourced = 0
    for item in evidence_items:
        if not isinstance(item, dict):
            continue
        if _as_text(item.get("source")) or _as_text(item.get("url")):
            sourced += 1
    return round(sourced / max(1, len(evidence_items)), 2)


def _value_to_text(value: Any, *, max_chars: int) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""

    try:
        serialized = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    except TypeError:
        serialized = str(value)
    return _truncate_text(serialized, max_chars)


def _truncate_text(text: str, max_chars: int) -> str:
    cleaned = text.strip()
    if max_chars <= 0:
        return ""
    if len(cleaned) <= max_chars:
        return cleaned
    if max_chars <= 3:
        return cleaned[:max_chars]
    return cleaned[: max_chars - 3].rstrip() + "..."


def _join_errors(first: str | None, second: str | None) -> str:
    left = _as_text(first)
    right = _as_text(second)
    if left and right:
        return f"{left}; {right}"
    if left:
        return left
    return right


def _as_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""
