from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict

from spice.protocols import Decision, ExecutionIntent, ExecutionResult
from spice_personal.profile.contract import (
    CATEGORY_EXTERNAL_COMMUNICATE,
    CATEGORY_EXTERNAL_EVIDENCE,
    CATEGORY_EXTERNAL_MANAGE_TASK,
    CATEGORY_EXTERNAL_SCHEDULE,
    CATEGORY_EXTERNAL_SYSTEM,
)


EXECUTION_INTENT_V1_SCHEMA_VERSION = "v1"
EXECUTION_INTENT_V1_SOURCE_DOMAIN = "personal.assistant"

SUPPORT_LEVEL_FULL = "full"
SUPPORT_LEVEL_LIMITED = "limited"
SUPPORT_LEVEL_DISABLED = "disabled"

VALID_SUPPORT_LEVELS = frozenset(
    {
        SUPPORT_LEVEL_FULL,
        SUPPORT_LEVEL_LIMITED,
        SUPPORT_LEVEL_DISABLED,
    }
)

CATEGORY_SUPPORT_LEVEL_MAP = {
    CATEGORY_EXTERNAL_EVIDENCE: SUPPORT_LEVEL_FULL,
    CATEGORY_EXTERNAL_SYSTEM: SUPPORT_LEVEL_FULL,
    CATEGORY_EXTERNAL_COMMUNICATE: SUPPORT_LEVEL_LIMITED,
    CATEGORY_EXTERNAL_SCHEDULE: SUPPORT_LEVEL_LIMITED,
    CATEGORY_EXTERNAL_MANAGE_TASK: SUPPORT_LEVEL_DISABLED,
}

CATEGORY_CANONICAL_OPERATION_MAP = {
    CATEGORY_EXTERNAL_EVIDENCE: "personal.gather_evidence",
    CATEGORY_EXTERNAL_SYSTEM: "personal.system",
    CATEGORY_EXTERNAL_COMMUNICATE: "personal.communicate",
    CATEGORY_EXTERNAL_SCHEDULE: "personal.schedule",
    CATEGORY_EXTERNAL_MANAGE_TASK: "personal.manage_task",
}

VALID_FAILURE_STRATEGIES = frozenset({"fail_fast", "retry_once"})
VALID_OPERATION_MODES = frozenset({"sync"})
VALID_EVIDENCE_SEARCH_DEPTH = frozenset({"focused", "standard"})

DEFAULT_EVIDENCE_SEARCH_DEPTH = "focused"
DEFAULT_EVIDENCE_MAX_RESULTS = 5
DEFAULT_SYSTEM_TIMEOUT_SECONDS = 30
DEFAULT_SYSTEM_MAX_OUTPUT_CHARS = 4000

ERROR_SCHEMA_VERSION_INVALID = "pei.v1.schema_version.invalid"
ERROR_CATEGORY_INVALID = "pei.v1.category.invalid"
ERROR_SUPPORT_LEVEL_INVALID = "pei.v1.support_level.invalid"
ERROR_SUPPORT_LEVEL_MISMATCH = "pei.v1.support_level.mismatch"
ERROR_GOAL_MISSING = "pei.v1.goal.missing"
ERROR_OPERATION_INVALID = "pei.v1.operation.invalid"
ERROR_OPERATION_CATEGORY_MISMATCH = "pei.v1.operation.category_mismatch"
ERROR_TARGET_INVALID = "pei.v1.target.invalid"
ERROR_INPUT_INVALID = "pei.v1.input.invalid"
ERROR_SUCCESS_CRITERIA_MISSING = "pei.v1.success_criteria.missing"
ERROR_FAILURE_POLICY_INVALID = "pei.v1.failure_policy.invalid"
ERROR_PROVENANCE_INVALID = "pei.v1.provenance.invalid"
ERROR_EVIDENCE_PLAN_INVALID = "pei.v1.evidence.plan.invalid"
ERROR_EVIDENCE_SEARCH_QUERIES_MISSING = "pei.v1.evidence.search_queries.missing"
ERROR_EVIDENCE_CITATIONS_REQUIRED = "pei.v1.evidence.citations.required"
ERROR_SYSTEM_BOUNDARY_MISSING = "pei.v1.system.boundary.missing"
ERROR_SYSTEM_DESTRUCTIVE_UNCONFIRMED = "pei.v1.system.destructive_unconfirmed"
ERROR_COMMUNICATE_SIDE_EFFECT_BLOCKED = "pei.v1.communicate.side_effect_blocked"
ERROR_SCHEDULE_SIDE_EFFECT_BLOCKED = "pei.v1.schedule.side_effect_blocked"
ERROR_MANAGE_TASK_DISABLED = "pei.v1.manage_task.disabled"
ERROR_ROUTE_DISABLED = "pei.v1.route.disabled"
ERROR_ROUTE_MISSING = "pei.v1.route.missing"
ERROR_CAPABILITY_MISSING = "pei.v1.capability.missing"
ERROR_FALLBACK_CLI_UNAVAILABLE = "pei.v1.fallback.cli.unavailable"
ERROR_EXECUTION_PARTIAL_FAILURE = "pei.v1.execution.partial_failure"
ERROR_EXECUTION_TIMEOUT = "pei.v1.execution.timeout"
ERROR_EXECUTION_FAILED = "pei.v1.execution.failed"
ERROR_EXECUTION_RESULT_INVALID = "pei.v1.execution.result_invalid"
ERROR_EXECUTION_FALLBACK_APPLIED = "pei.v1.execution.fallback.applied"

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

DESTRUCTIVE_TASK_MARKERS = (
    "delete",
    "destroy",
    "drop table",
    "drop database",
    "truncate",
    "rm -rf",
    "remove all",
    "shutdown",
    "format disk",
    "wipe",
)

UNBOUNDED_SCOPE_MARKERS = frozenset(
    {
        "*",
        "all",
        "any",
        "global",
        "everywhere",
        "unbounded",
        "unspecified",
    }
)

DIRECT_SEND_MARKERS = frozenset({"send", "direct_send", "live_send", "deliver_now"})
CALENDAR_WRITE_MARKERS = frozenset({"calendar_write", "write_calendar", "create_event", "update_event"})


class ExecutionIntentV1Payload(TypedDict):
    schema_version: str
    category: str
    support_level: str
    goal: str
    operation: dict[str, Any]
    target: dict[str, Any]
    input: dict[str, Any]
    parameters: dict[str, Any]
    constraints: list[dict[str, Any]]
    success_criteria: list[dict[str, Any]]
    failure_policy: dict[str, Any]
    provenance: dict[str, Any]


@dataclass(slots=True, frozen=True)
class PEIV1Issue:
    code: str
    message: str
    field_path: str = ""


@dataclass(slots=True)
class PEIV1NormalizationResult:
    payload: ExecutionIntentV1Payload
    issues: list[PEIV1Issue] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class PEIV1RouteContext:
    category_route: dict[str, Any] | None = None
    fallback_route: dict[str, Any] | None = None
    profile_mode: str = ""
    available_capabilities: tuple[str, ...] = ()
    fallback_applied: bool = False
    fallback_available: bool = False


@dataclass(slots=True)
class PEIV1ValidationResult:
    payload: ExecutionIntentV1Payload
    errors: list[PEIV1Issue] = field(default_factory=list)
    degradations: list[PEIV1Issue] = field(default_factory=list)
    infos: list[PEIV1Issue] = field(default_factory=list)
    pending_confirmation: bool = False

    @property
    def allow_execution(self) -> bool:
        return not self.errors and not self.pending_confirmation


def build_execution_intent_v1_payload(
    intent: ExecutionIntent,
    *,
    decision: Decision | None = None,
) -> ExecutionIntentV1Payload:
    input_payload = _as_dict(intent.input_payload)
    brief = _as_dict(input_payload.get("execution_brief"))
    brief_inputs = _as_dict(brief.get("inputs"))

    normalized_input = dict(input_payload)
    normalized_input.pop("execution_brief", None)
    for key, value in brief_inputs.items():
        normalized_input.setdefault(str(key), value)

    parameters = _as_dict(intent.parameters)
    support_level = (
        _as_text(parameters.get("support_level"))
        or _as_text(input_payload.get("support_level"))
        or _as_text(brief.get("support_level"))
    )
    goal = _as_text(brief.get("goal")) or _as_text(_as_dict(intent.objective).get("description"))

    provenance = _as_dict(intent.provenance)
    if decision is not None:
        if not _as_text(provenance.get("decision_id")):
            provenance["decision_id"] = _as_text(decision.id)
        if not _as_text(provenance.get("selected_action")):
            provenance["selected_action"] = _as_text(decision.selected_action)
        if not _as_text(provenance.get("source_turn_id")):
            source_turn_id = _as_text(_as_dict(decision.metadata).get("source_turn_id"))
            provenance["source_turn_id"] = source_turn_id or _as_text(decision.id)
    if not _as_text(provenance.get("source_domain")):
        provenance["source_domain"] = EXECUTION_INTENT_V1_SOURCE_DOMAIN

    payload: ExecutionIntentV1Payload = {
        "schema_version": _as_text(brief.get("schema_version"))
        or _as_text(input_payload.get("schema_version"))
        or EXECUTION_INTENT_V1_SCHEMA_VERSION,
        "category": _as_text(brief.get("category")) or _as_text(input_payload.get("category")),
        "support_level": support_level,
        "goal": goal,
        "operation": _as_dict(intent.operation),
        "target": _as_dict(intent.target),
        "input": normalized_input,
        "parameters": parameters,
        "constraints": _as_list_of_dict(intent.constraints),
        "success_criteria": _as_list_of_dict(intent.success_criteria),
        "failure_policy": _as_dict(intent.failure_policy),
        "provenance": provenance,
    }
    return payload


def apply_execution_intent_v1_payload(
    intent: ExecutionIntent,
    payload: ExecutionIntentV1Payload,
) -> None:
    intent.operation = _as_dict(payload.get("operation"))
    intent.target = _as_dict(payload.get("target"))
    input_payload = _as_dict(payload.get("input"))
    existing_input = _as_dict(intent.input_payload)
    existing_brief = _as_dict(existing_input.get("execution_brief"))
    if existing_brief:
        input_payload["execution_brief"] = existing_brief
    intent.input_payload = input_payload
    intent.parameters = _as_dict(payload.get("parameters"))
    intent.constraints = _as_list_of_dict(payload.get("constraints"))
    intent.success_criteria = _as_list_of_dict(payload.get("success_criteria"))
    intent.failure_policy = _as_dict(payload.get("failure_policy"))
    intent.provenance = _as_dict(payload.get("provenance"))


def normalize_execution_intent_v1(
    payload: dict[str, Any],
) -> PEIV1NormalizationResult:
    normalized = _coerce_payload(payload)
    issues: list[PEIV1Issue] = []

    category = _as_text(normalized.get("category"))
    operation = _as_dict(normalized.get("operation"))
    parameters = _as_dict(normalized.get("parameters"))
    input_payload = _as_dict(normalized.get("input"))
    failure_policy = _as_dict(normalized.get("failure_policy"))

    mode = _as_text(operation.get("mode")).lower()
    if not mode:
        operation["mode"] = "sync"
    else:
        operation["mode"] = mode
    operation["dry_run"] = _as_bool(operation.get("dry_run"))
    normalized["operation"] = operation

    strategy = _as_text(failure_policy.get("strategy")).lower()
    if strategy not in VALID_FAILURE_STRATEGIES:
        failure_policy["strategy"] = "fail_fast"
    else:
        failure_policy["strategy"] = strategy
    failure_policy["max_retries"] = _clamp_int(
        failure_policy.get("max_retries"),
        default=0,
        min_value=0,
        max_value=1,
    )
    normalized["failure_policy"] = failure_policy

    if category == CATEGORY_EXTERNAL_EVIDENCE:
        parameters["require_source_citations"] = True

        search_depth = _as_text(parameters.get("search_depth")).lower()
        if search_depth not in VALID_EVIDENCE_SEARCH_DEPTH:
            parameters["search_depth"] = DEFAULT_EVIDENCE_SEARCH_DEPTH
        else:
            parameters["search_depth"] = search_depth

        parameters["max_results"] = _clamp_int(
            parameters.get("max_results"),
            default=DEFAULT_EVIDENCE_MAX_RESULTS,
            min_value=1,
            max_value=10,
        )

        search_queries = _normalize_string_list(input_payload.get("search_queries"))
        if not search_queries:
            derived_queries = _derive_search_queries_from_evidence_plan(input_payload.get("evidence_plan"))
            if derived_queries:
                input_payload["search_queries"] = derived_queries
                issues.append(
                    PEIV1Issue(
                        code=ERROR_EVIDENCE_SEARCH_QUERIES_MISSING,
                        message="Derived search_queries from evidence_plan facts.",
                        field_path="input.search_queries",
                    )
                )
        else:
            input_payload["search_queries"] = search_queries

    if category == CATEGORY_EXTERNAL_SYSTEM:
        parameters["timeout_seconds"] = _clamp_int(
            parameters.get("timeout_seconds"),
            default=DEFAULT_SYSTEM_TIMEOUT_SECONDS,
            min_value=1,
            max_value=120,
        )
        if "max_output_chars" in parameters:
            parameters["max_output_chars"] = _clamp_int(
                parameters.get("max_output_chars"),
                default=DEFAULT_SYSTEM_MAX_OUTPUT_CHARS,
                min_value=128,
                max_value=50000,
            )

    if category == CATEGORY_EXTERNAL_COMMUNICATE:
        requested_send = _requests_communicate_side_effect(input_payload, parameters, operation)
        operation["dry_run"] = True
        parameters["delivery_mode"] = "draft_only"
        if requested_send:
            issues.append(
                PEIV1Issue(
                    code=ERROR_COMMUNICATE_SIDE_EFFECT_BLOCKED,
                    message="Limited communicate request degraded to draft_only.",
                    field_path="operation.dry_run",
                )
            )

    if category == CATEGORY_EXTERNAL_SCHEDULE:
        requested_write = _requests_schedule_side_effect(input_payload, parameters, operation)
        operation["dry_run"] = True
        parameters["calendar_write"] = False
        if requested_write:
            issues.append(
                PEIV1Issue(
                    code=ERROR_SCHEDULE_SIDE_EFFECT_BLOCKED,
                    message="Limited schedule request degraded to candidate slots and draft invite text.",
                    field_path="operation.dry_run",
                )
            )

    normalized["operation"] = operation
    normalized["parameters"] = parameters
    normalized["input"] = input_payload
    return PEIV1NormalizationResult(
        payload=normalized,
        issues=issues,
    )


def validate_execution_intent_v1(
    payload: dict[str, Any],
    *,
    route_context: PEIV1RouteContext | None = None,
) -> PEIV1ValidationResult:
    normalized = _coerce_payload(payload)
    errors: list[PEIV1Issue] = []
    degradations: list[PEIV1Issue] = []
    infos: list[PEIV1Issue] = []
    pending_confirmation = False

    schema_version = _as_text(normalized.get("schema_version"))
    category = _as_text(normalized.get("category"))
    support_level = _as_text(normalized.get("support_level"))
    goal = _as_text(normalized.get("goal"))
    operation = _as_dict(normalized.get("operation"))
    target = _as_dict(normalized.get("target"))
    input_payload = _as_dict(normalized.get("input"))
    parameters = _as_dict(normalized.get("parameters"))
    constraints = _as_list_of_dict(normalized.get("constraints"))
    success_criteria = _as_list_of_dict(normalized.get("success_criteria"))
    failure_policy = _as_dict(normalized.get("failure_policy"))
    provenance = _as_dict(normalized.get("provenance"))

    if schema_version != EXECUTION_INTENT_V1_SCHEMA_VERSION:
        errors.append(
            PEIV1Issue(
                code=ERROR_SCHEMA_VERSION_INVALID,
                message=f"schema_version must equal {EXECUTION_INTENT_V1_SCHEMA_VERSION!r}.",
                field_path="schema_version",
            )
        )

    if category not in CATEGORY_SUPPORT_LEVEL_MAP:
        errors.append(
            PEIV1Issue(
                code=ERROR_CATEGORY_INVALID,
                message="category is not in the v1 whitelist.",
                field_path="category",
            )
        )

    if support_level not in VALID_SUPPORT_LEVELS:
        errors.append(
            PEIV1Issue(
                code=ERROR_SUPPORT_LEVEL_INVALID,
                message="support_level must be full/limited/disabled.",
                field_path="support_level",
            )
        )
    elif category in CATEGORY_SUPPORT_LEVEL_MAP and support_level != CATEGORY_SUPPORT_LEVEL_MAP.get(category):
        errors.append(
            PEIV1Issue(
                code=ERROR_SUPPORT_LEVEL_MISMATCH,
                message="support_level does not match frozen category mapping.",
                field_path="support_level",
            )
        )

    if not goal:
        errors.append(
            PEIV1Issue(
                code=ERROR_GOAL_MISSING,
                message="goal must be non-empty.",
                field_path="goal",
            )
        )

    if not isinstance(operation, dict):
        errors.append(
            PEIV1Issue(
                code=ERROR_OPERATION_INVALID,
                message="operation must be an object.",
                field_path="operation",
            )
        )
    else:
        operation_name = _as_text(operation.get("name"))
        operation_mode = _as_text(operation.get("mode")).lower()
        if not operation_name or operation_mode not in VALID_OPERATION_MODES:
            errors.append(
                PEIV1Issue(
                    code=ERROR_OPERATION_INVALID,
                    message="operation.name must be non-empty and operation.mode must be sync.",
                    field_path="operation",
                )
            )
        elif category in CATEGORY_CANONICAL_OPERATION_MAP and operation_name != CATEGORY_CANONICAL_OPERATION_MAP[category]:
            errors.append(
                PEIV1Issue(
                    code=ERROR_OPERATION_CATEGORY_MISMATCH,
                    message="operation.name does not match category canonical operation.",
                    field_path="operation.name",
                )
            )

    if not _as_text(target.get("kind")) or not _as_text(target.get("id")):
        errors.append(
            PEIV1Issue(
                code=ERROR_TARGET_INVALID,
                message="target.kind and target.id are required.",
                field_path="target",
            )
        )

    if not isinstance(input_payload, dict) or not input_payload:
        errors.append(
            PEIV1Issue(
                code=ERROR_INPUT_INVALID,
                message="input must be a non-empty object.",
                field_path="input",
            )
        )

    if not _has_valid_success_criteria(success_criteria):
        errors.append(
            PEIV1Issue(
                code=ERROR_SUCCESS_CRITERIA_MISSING,
                message="success_criteria must contain at least one complete entry.",
                field_path="success_criteria",
            )
        )

    failure_strategy = _as_text(failure_policy.get("strategy")).lower()
    max_retries = failure_policy.get("max_retries")
    if failure_strategy not in VALID_FAILURE_STRATEGIES or not isinstance(max_retries, int) or max_retries not in {0, 1}:
        errors.append(
            PEIV1Issue(
                code=ERROR_FAILURE_POLICY_INVALID,
                message="failure_policy must use fail_fast/retry_once and max_retries 0..1.",
                field_path="failure_policy",
            )
        )

    if (
        not _as_text(provenance.get("decision_id"))
        or not _as_text(provenance.get("selected_action"))
        or _as_text(provenance.get("source_domain")) != EXECUTION_INTENT_V1_SOURCE_DOMAIN
        or not _as_text(provenance.get("source_turn_id"))
    ):
        errors.append(
            PEIV1Issue(
                code=ERROR_PROVENANCE_INVALID,
                message="provenance must include decision_id, selected_action, source_domain, source_turn_id.",
                field_path="provenance",
            )
        )

    if category == CATEGORY_EXTERNAL_EVIDENCE:
        if not _is_valid_evidence_plan(input_payload.get("evidence_plan")):
            errors.append(
                PEIV1Issue(
                    code=ERROR_EVIDENCE_PLAN_INVALID,
                    message="evidence_plan must contain exactly 3 non-empty fact/why items.",
                    field_path="input.evidence_plan",
                )
            )
        search_queries = _normalize_string_list(input_payload.get("search_queries"))
        if not search_queries:
            errors.append(
                PEIV1Issue(
                    code=ERROR_EVIDENCE_SEARCH_QUERIES_MISSING,
                    message="search_queries must contain at least one query.",
                    field_path="input.search_queries",
                )
            )
        if parameters.get("require_source_citations") is not True:
            errors.append(
                PEIV1Issue(
                    code=ERROR_EVIDENCE_CITATIONS_REQUIRED,
                    message="require_source_citations must be true for external.evidence.",
                    field_path="parameters.require_source_citations",
                )
            )
        if not _contains_evidence_collected_criterion(success_criteria):
            errors.append(
                PEIV1Issue(
                    code=ERROR_SUCCESS_CRITERIA_MISSING,
                    message="external.evidence must include success criterion evidence.collected.",
                    field_path="success_criteria",
                )
            )

    if category == CATEGORY_EXTERNAL_SYSTEM:
        if not _as_text(input_payload.get("task")) or not _has_non_empty_scope(input_payload.get("scope")):
            errors.append(
                PEIV1Issue(
                    code=ERROR_SYSTEM_BOUNDARY_MISSING,
                    message="external.system requires input.task and input.scope.",
                    field_path="input",
                )
            )
        scope = input_payload.get("scope")
        if _scope_is_unbounded(scope):
            operation["dry_run"] = True
            normalized["operation"] = operation
            pending_confirmation = True
            degradations.append(
                PEIV1Issue(
                    code=ERROR_SYSTEM_BOUNDARY_MISSING,
                    message="System scope is unbounded; forcing dry_run and pending confirmation.",
                    field_path="input.scope",
                )
            )
        if _looks_destructive_task(input_payload.get("task")) and not _has_confirmation_constraint(constraints):
            errors.append(
                PEIV1Issue(
                    code=ERROR_SYSTEM_DESTRUCTIVE_UNCONFIRMED,
                    message="Destructive system task requires explicit confirmation constraint.",
                    field_path="constraints",
                )
            )

    if category == CATEGORY_EXTERNAL_COMMUNICATE and _requests_communicate_side_effect(
        input_payload,
        parameters,
        operation,
    ):
        operation["dry_run"] = True
        normalized["operation"] = operation
        parameters["delivery_mode"] = "draft_only"
        normalized["parameters"] = parameters
        degradations.append(
            PEIV1Issue(
                code=ERROR_COMMUNICATE_SIDE_EFFECT_BLOCKED,
                message="Limited communicate side effects blocked; downgraded to draft_only.",
                field_path="operation.dry_run",
            )
        )

    if category == CATEGORY_EXTERNAL_SCHEDULE and _requests_schedule_side_effect(
        input_payload,
        parameters,
        operation,
    ):
        operation["dry_run"] = True
        normalized["operation"] = operation
        parameters["calendar_write"] = False
        normalized["parameters"] = parameters
        degradations.append(
            PEIV1Issue(
                code=ERROR_SCHEDULE_SIDE_EFFECT_BLOCKED,
                message="Limited schedule side effects blocked; downgraded to candidate slots and draft invite text.",
                field_path="operation.dry_run",
            )
        )

    if category == CATEGORY_EXTERNAL_MANAGE_TASK:
        errors.append(
            PEIV1Issue(
                code=ERROR_MANAGE_TASK_DISABLED,
                message="external.manage_task is disabled in v1.",
                field_path="category",
            )
        )

    if route_context is not None:
        _validate_route_precheck(
            category=category,
            route_context=route_context,
            errors=errors,
            infos=infos,
        )

    return PEIV1ValidationResult(
        payload=normalized,
        errors=errors,
        degradations=degradations,
        infos=infos,
        pending_confirmation=pending_confirmation,
    )


def preflight_execution_intent_v1(
    intent: ExecutionIntent,
    *,
    decision: Decision | None = None,
    route_context: PEIV1RouteContext | None = None,
) -> PEIV1ValidationResult:
    base_payload = build_execution_intent_v1_payload(intent, decision=decision)
    normalize_result = normalize_execution_intent_v1(base_payload)
    validation = validate_execution_intent_v1(
        normalize_result.payload,
        route_context=route_context,
    )
    if normalize_result.issues:
        validation.degradations = [*normalize_result.issues, *validation.degradations]
    apply_execution_intent_v1_payload(intent, validation.payload)
    return validation


def ensure_minimal_execution_result_output(
    result: ExecutionResult,
    *,
    intent: ExecutionIntent | None = None,
    decision: Decision | None = None,
    category: str = "",
) -> None:
    output = _as_dict(result.output)

    summary = (
        _as_text(output.get("summary"))
        or _as_text(output.get("evidence_summary"))
        or _as_text(output.get("message"))
        or _as_text(output.get("text"))
    )
    if not summary:
        status = _as_text(result.status).lower() or "unknown"
        if _as_text(result.error):
            summary = f"Execution status={status}: {_as_text(result.error)}"
        else:
            summary = f"Execution status={status}."

    items = _normalize_object_list(output.get("items"))
    if not items:
        for key in ("actions", "artifacts", "results", "hits"):
            items = _normalize_object_list(output.get(key))
            if items:
                break
    if not items:
        evidence_items_from_output = _normalize_object_list(output.get("evidence"))
        if evidence_items_from_output:
            items = evidence_items_from_output

    evidence_items = _normalize_object_list(output.get("evidence_items"))
    if not evidence_items:
        evidence_items = _normalize_object_list(output.get("evidence"))
    if not evidence_items and category == CATEGORY_EXTERNAL_EVIDENCE:
        evidence_items = list(items)

    status = _as_text(output.get("status")).lower() or _as_text(result.status).lower() or "unknown"
    source_refs = _collect_source_refs(
        output=output,
        result=result,
        intent=intent,
        decision=decision,
    )
    confidence = _as_float(output.get("confidence"), -1.0)
    if confidence < 0.0:
        confidence = _estimate_default_confidence(status=status, has_items=bool(items or evidence_items))

    output["summary"] = summary
    output["items"] = items
    output["evidence_items"] = evidence_items
    output["status"] = status
    output["source_refs"] = source_refs
    output["confidence"] = round(max(0.0, min(confidence, 1.0)), 2)
    result.output = output


def _validate_route_precheck(
    *,
    category: str,
    route_context: PEIV1RouteContext,
    errors: list[PEIV1Issue],
    infos: list[PEIV1Issue],
) -> None:
    route = route_context.category_route
    if not isinstance(route, dict):
        errors.append(
            PEIV1Issue(
                code=ERROR_ROUTE_MISSING,
                message=f"Route missing for category {category!r}.",
                field_path="route",
            )
        )
        return

    if not bool(route.get("enabled")):
        errors.append(
            PEIV1Issue(
                code=ERROR_ROUTE_DISABLED,
                message=f"Route disabled for category {category!r}.",
                field_path="route.enabled",
            )
        )
        return

    profile_mode = _as_text(route_context.profile_mode).lower()
    expected_capabilities = _route_expected_capabilities(route)
    if profile_mode != "sdep" or not expected_capabilities:
        return

    available_capabilities = {
        _as_text(item)
        for item in route_context.available_capabilities
        if _as_text(item)
    }
    if available_capabilities and available_capabilities.intersection(expected_capabilities):
        return

    fallback_route = route_context.fallback_route
    has_fallback = isinstance(fallback_route, dict) and _as_text(fallback_route.get("operation_name"))
    if not has_fallback:
        errors.append(
            PEIV1Issue(
                code=ERROR_CAPABILITY_MISSING,
                message="SDEP capability missing and no fallback route configured.",
                field_path="route.required_capabilities",
            )
        )
        return

    if not route_context.fallback_available:
        errors.append(
            PEIV1Issue(
                code=ERROR_FALLBACK_CLI_UNAVAILABLE,
                message="Explicit CLI fallback exists but CLI fallback runtime is unavailable.",
                field_path="route.fallback_cli",
            )
        )
        return

    if route_context.fallback_applied:
        infos.append(
            PEIV1Issue(
                code=ERROR_EXECUTION_FALLBACK_APPLIED,
                message="CLI fallback route applied for missing SDEP capability.",
                field_path="route.fallback_cli",
            )
        )
        return

    errors.append(
        PEIV1Issue(
            code=ERROR_CAPABILITY_MISSING,
            message="SDEP capability missing and fallback route was not applied.",
            field_path="route.required_capabilities",
        )
    )


def _coerce_payload(payload: dict[str, Any]) -> ExecutionIntentV1Payload:
    raw = dict(payload) if isinstance(payload, dict) else {}
    return {
        "schema_version": _as_text(raw.get("schema_version")),
        "category": _as_text(raw.get("category")),
        "support_level": _as_text(raw.get("support_level")),
        "goal": _as_text(raw.get("goal")),
        "operation": _as_dict(raw.get("operation")),
        "target": _as_dict(raw.get("target")),
        "input": _as_dict(raw.get("input")),
        "parameters": _as_dict(raw.get("parameters")),
        "constraints": _as_list_of_dict(raw.get("constraints")),
        "success_criteria": _as_list_of_dict(raw.get("success_criteria")),
        "failure_policy": _as_dict(raw.get("failure_policy")),
        "provenance": _as_dict(raw.get("provenance")),
    }


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _as_list_of_dict(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return str(value).strip() if value is not None else ""


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    token = _as_text(value).lower()
    if token in {"1", "true", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "no", "n", "off"}:
        return False
    return bool(value)


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clamp_int(value: Any, *, default: int, min_value: int, max_value: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        return default
    if parsed < min_value:
        return min_value
    if parsed > max_value:
        return max_value
    return parsed


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        token = _as_text(item)
        if not token:
            continue
        normalized.append(token)
    return list(dict.fromkeys(normalized))


def _normalize_object_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            normalized.append(dict(item))
    return normalized


def _derive_search_queries_from_evidence_plan(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    queries: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        fact = _as_text(item.get("fact"))
        if fact:
            queries.append(fact)
    return list(dict.fromkeys(queries))


def _contains_internal_runtime_evidence_marker(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in INTERNAL_RUNTIME_EVIDENCE_MARKERS)


def _is_valid_evidence_plan(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 3:
        return False
    for item in value:
        if not isinstance(item, dict):
            return False
        fact = _as_text(item.get("fact"))
        why = _as_text(item.get("why"))
        if not fact or not why:
            return False
        if _contains_internal_runtime_evidence_marker(fact) or _contains_internal_runtime_evidence_marker(why):
            return False
    return True


def _has_valid_success_criteria(value: list[dict[str, Any]]) -> bool:
    if not value:
        return False
    for item in value:
        if not _as_text(item.get("id")):
            return False
        if not _as_text(item.get("description")):
            return False
    return True


def _contains_evidence_collected_criterion(value: list[dict[str, Any]]) -> bool:
    for item in value:
        if _as_text(item.get("id")) == "evidence.collected":
            return True
    return False


def _has_confirmation_constraint(constraints: list[dict[str, Any]]) -> bool:
    for item in constraints:
        name = _as_text(item.get("name")).lower()
        kind = _as_text(item.get("kind")).lower()
        params = _as_dict(item.get("params"))
        if name == "profile.require_confirmation" and _as_bool(params.get("required")):
            return True
        if kind == "approval_gate" and _as_bool(params.get("required")):
            return True
        if "confirm" in name and _as_bool(params.get("required")):
            return True
    return False


def _looks_destructive_task(value: Any) -> bool:
    task = _as_text(value).lower()
    if not task:
        return False
    return any(marker in task for marker in DESTRUCTIVE_TASK_MARKERS)


def _has_non_empty_scope(value: Any) -> bool:
    if isinstance(value, dict):
        for key in ("id", "kind", "value", "path"):
            if _as_text(value.get(key)):
                return True
        return bool(value)
    return bool(_as_text(value))


def _scope_is_unbounded(value: Any) -> bool:
    if isinstance(value, dict):
        candidate = _as_text(value.get("id")) or _as_text(value.get("value")) or _as_text(value.get("kind"))
    else:
        candidate = _as_text(value)
    if not candidate:
        return False
    return candidate.lower() in UNBOUNDED_SCOPE_MARKERS


def _requests_communicate_side_effect(
    input_payload: dict[str, Any],
    parameters: dict[str, Any],
    operation: dict[str, Any],
) -> bool:
    if not _as_bool(operation.get("dry_run")):
        return True
    delivery_mode = _as_text(parameters.get("delivery_mode")).lower()
    if delivery_mode and delivery_mode != "draft_only":
        return True
    for marker in DIRECT_SEND_MARKERS:
        if _as_bool(input_payload.get(marker)) or _as_bool(parameters.get(marker)):
            return True
    channel = _as_text(input_payload.get("channel")).lower()
    if channel in {"email_live", "sms_live", "api_send", "direct_send"}:
        return True
    return False


def _requests_schedule_side_effect(
    input_payload: dict[str, Any],
    parameters: dict[str, Any],
    operation: dict[str, Any],
) -> bool:
    if not _as_bool(operation.get("dry_run")):
        return True
    if _as_bool(parameters.get("calendar_write")):
        return True
    for marker in CALENDAR_WRITE_MARKERS:
        if _as_bool(input_payload.get(marker)) or _as_bool(parameters.get(marker)):
            return True
    return False


def _route_expected_capabilities(route: dict[str, Any]) -> set[str]:
    required = route.get("required_capabilities")
    if isinstance(required, list):
        normalized = {_as_text(item) for item in required if _as_text(item)}
        if normalized:
            return normalized
    operation_name = _as_text(route.get("operation_name"))
    return {operation_name} if operation_name else set()


def _collect_source_refs(
    *,
    output: dict[str, Any],
    result: ExecutionResult,
    intent: ExecutionIntent | None,
    decision: Decision | None,
) -> list[str]:
    refs: list[str] = []
    source_refs = output.get("source_refs")
    if isinstance(source_refs, list):
        refs.extend(_as_text(item) for item in source_refs if _as_text(item))
    refs.extend(_as_text(item) for item in result.refs if _as_text(item))
    refs.append(_as_text(result.id))
    if intent is not None:
        refs.append(_as_text(intent.id))
        refs.extend(_as_text(item) for item in intent.refs if _as_text(item))
    if decision is not None:
        refs.append(_as_text(decision.id))
        refs.extend(_as_text(item) for item in decision.refs if _as_text(item))
    return list(dict.fromkeys(ref for ref in refs if ref))


def _estimate_default_confidence(*, status: str, has_items: bool) -> float:
    if status == "success":
        return 0.78 if has_items else 0.58
    if status == "failed":
        return 0.22
    return 0.40 if has_items else 0.30
