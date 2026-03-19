from __future__ import annotations

import importlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from typing import Any, Callable, TextIO
from uuid import uuid4

from spice.core import SpiceRuntime, StateStore
from spice.entry.init_domain import run_init_domain_from_spec
from spice.entry.spec import DomainSpec, derive_domain_pack_class_name, derive_package_name
from spice.protocols import (
    Decision,
    ExecutionIntent,
    ExecutionResult,
    Observation,
    Outcome,
    Reflection,
    WorldState,
)
from spice_personal.advisory.personal_advisory import (
    PERSONAL_ADVISORY_ATTRIBUTE_KEYS,
    RESULT_KIND_ACTION_PROPOSAL,
    RESULT_KIND_SUGGESTION,
    build_personal_llm_decision_policy,
)
from spice_personal.config.personal_config import (
    PERSONAL_CONFIG_FILENAME,
    load_personal_connection_config,
)
from spice_personal.config.settings import (
    resolve_executor_config_for_runtime,
    resolve_model_command,
)
from spice_personal.execution.evidence_round import (
    EvidenceRoundResult,
    run_bounded_evidence_round,
    run_mock_evidence_round,
    should_gather_evidence,
)
from spice_personal.execution.execution_intent_v1 import (
    CATEGORY_CANONICAL_OPERATION_MAP,
    CATEGORY_SUPPORT_LEVEL_MAP,
    EXECUTION_INTENT_V1_SCHEMA_VERSION,
    EXECUTION_INTENT_V1_SOURCE_DOMAIN,
    PEIV1RouteContext,
    ensure_minimal_execution_result_output,
    preflight_execution_intent_v1,
)
from spice_personal.executors.factory import PersonalExecutorConfig, build_executor
from spice_personal.profile.contract import (
    CATEGORY_EXTERNAL_SYSTEM,
    EXECUTOR_MODE_CLI,
    EXECUTOR_MODE_MOCK,
    EXECUTOR_MODE_SDEP,
    ensure_minimum_execution_brief,
)
from spice_personal.profile.loader import (
    ensure_workspace_profile,
    load_profile,
    profile_fingerprint,
)
from spice_personal.profile.validate import ProfileValidationResult, validate_profile_contract


PERSONAL_DEFAULT_WORKSPACE = Path(".spice/personal")
PERSONAL_STATE_RELATIVE_PATH = Path("artifacts/personal_state.json")
PERSONAL_PROFILE_VALIDATION_RELATIVE_PATH = Path("artifacts/personal_profile_validation.json")
PERSONAL_CONNECTION_RESOLUTION_RELATIVE_PATH = Path("artifacts/personal_connection_resolution.json")

PERSONAL_DOMAIN_ID = "personal.assistant"
PERSONAL_ENTITY_ID = "personal.assistant.current"

PERSONAL_OBSERVATION_QUESTION_RECEIVED = "personal.assistant.question_received"
PERSONAL_OBSERVATION_CONTEXT_INGEST = "personal.context.ingest"

PERSONAL_ACTION_SUGGEST = "personal.assistant.suggest"
PERSONAL_ACTION_ASK_CLARIFY = "personal.assistant.ask_clarify"
PERSONAL_ACTION_DEFER = "personal.assistant.defer"
PERSONAL_ACTION_GATHER_EVIDENCE = "personal.assistant.gather_evidence"
PERSONAL_OUTCOME_ADVICE_RECORDED = "personal.assistant.advice_recorded"
ADOPTION_STATUS_ADOPTED = "adopted"
ADOPTION_STATUS_DECLINED = "declined"
ADOPTION_STATUS_PENDING = "pending"
EXECUTION_STATUS_NOT_REQUESTED = "not_requested"
EXECUTION_STATUS_SUCCESS = "success"
EXECUTION_STATUS_FAILED = "failed"
EVIDENCE_STATE_NOT_REQUESTED = "not_requested"
EVIDENCE_STATE_AWAITING_MANUAL_INPUT = "awaiting_manual_evidence_input"
EVIDENCE_STATE_REEVALUATED = "evidence_received_and_reevaluated"
EVIDENCE_STATE_EXTERNAL_EXECUTION = "external_execution"
RESULT_KIND_SETUP_REQUIRED = "setup_required"
CONNECTION_STATE_READY = "ready"
CONNECTION_STATE_SETUP_REQUIRED = "setup_required"
MAX_CONTEXT_TEXT_CHARS = 50 * 1024
MAX_CONTEXT_FILE_BYTES = 256 * 1024
MAX_CONTEXT_CONTENT_CHARS = 50 * 1024
MAX_CONTEXT_PREVIEW_CHARS = 50 * 1024
ALLOWED_CONTEXT_FILE_EXTENSIONS = frozenset(
    {
        ".txt",
        ".md",
        ".json",
        ".csv",
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".yml",
        ".yaml",
    }
)
INTERNAL_RUNTIME_TEXT_MARKERS = (
    "signal-",
    "obs-",
    "worldstate-",
    "question_received",
    "selected_action",
    "risk_budget",
    "active session",
    "current state",
    "worldstate",
    "protocol validation",
    "complexity requirement",
)
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


@dataclass(slots=True)
class PersonalAdvice:
    selected_action: str
    suggestion: str
    urgency: str
    confidence: float


@dataclass(slots=True)
class PersonalAskResult:
    advice: PersonalAdvice | None
    auto_initialized: bool
    evidence_notice: str | None = None
    result_kind: str = RESULT_KIND_SUGGESTION
    decision_adoption_status: str = ADOPTION_STATUS_PENDING
    execution_status: str = EXECUTION_STATUS_NOT_REQUESTED
    connection_state: str = CONNECTION_STATE_READY
    onboarding_card: str | None = None
    decision_options: tuple[dict[str, Any], ...] = ()
    recommended_option_id: str = ""


@dataclass(slots=True)
class AdvisoryTurnResult:
    observation: Observation
    decision: Decision
    outcome: Outcome
    reflection: Reflection
    world_state: WorldState
    advice: PersonalAdvice
    evidence_notice: str | None = None
    orchestration_metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class PersonalProfileContext:
    path: Path
    profile: dict[str, Any]
    available_capabilities: tuple[str, ...] = ()


@dataclass(slots=True)
class _IntentRouteResolution:
    resolved_mode: str
    profile_mode: str = ""
    category: str = ""
    category_route: dict[str, Any] | None = None
    fallback_route: dict[str, Any] | None = None
    fallback_applied: bool = False


def load_builtin_personal_spec() -> DomainSpec:
    asset = files("spice_personal.domain.assets").joinpath("personal.domain_spec.json")
    payload = json.loads(asset.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Built-in personal spec payload must be an object.")
    return DomainSpec.from_dict(payload)


def _default_personal_connection_template() -> dict[str, Any]:
    return {
        "schema_version": "spice_personal.connection.v1",
        "model": {
            "api_key_env": "OPENROUTER_API_KEY",
            "model": "anthropic/claude-opus-4-5",
            "provider": "openrouter",
        },
        "agent": {
            "auth_env": "",
            "endpoint": "",
            "mode": "",
            "provider": "",
        },
        "executor": {
            "cli_command": "",
            "mode": "mock",
            "sdep_command": "",
        },
    }


def _ensure_personal_connection_config(
    workspace: Path,
    *,
    preserved_payload: bytes | None = None,
) -> Path:
    config_path = workspace / PERSONAL_CONFIG_FILENAME
    if config_path.exists():
        return config_path
    if preserved_payload is not None:
        config_path.write_bytes(preserved_payload)
        return config_path
    config_path.write_text(
        json.dumps(_default_personal_connection_template(), ensure_ascii=True, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return config_path


def run_personal_init(
    *,
    workspace: str | Path = PERSONAL_DEFAULT_WORKSPACE,
    force: bool = False,
) -> Path:
    workspace_path = Path(workspace)
    existing_config_path = workspace_path / PERSONAL_CONFIG_FILENAME
    preserved_config_payload = (
        existing_config_path.read_bytes() if existing_config_path.exists() else None
    )

    spec = load_builtin_personal_spec()
    report = run_init_domain_from_spec(
        spec=spec,
        output_dir=workspace_path,
        force=force,
        no_run=True,
        interactive=False,
        from_spec_path=None,
    )
    _ensure_personal_connection_config(
        report.output_dir,
        preserved_payload=preserved_config_payload,
    )
    profile_path = ensure_workspace_profile(report.output_dir, force=force)
    profile_payload = load_profile(profile_path)

    workspace_config = load_personal_connection_config(report.output_dir)
    base_executor_config = resolve_executor_config_for_runtime(
        None,
        workspace_config=workspace_config,
    )
    executor_config = _resolve_executor_config_with_profile_mode(
        base_executor_config,
        profile_payload,
    )
    _write_connection_resolution_report(
        report.output_dir,
        model_command=None,
        executor_config=executor_config,
        workspace_config=workspace_config,
    )
    validation = validate_profile_contract(
        profile_payload,
        profile_path=profile_path,
        executor_config=executor_config,
    )
    validation.raise_for_errors()
    _write_profile_validation_report(report.output_dir, validation)
    return report.output_dir


def run_personal_ask(
    *,
    question: str,
    workspace: str | Path = PERSONAL_DEFAULT_WORKSPACE,
    model: str | None = None,
    executor_config: PersonalExecutorConfig | None = None,
    context_text: str | None = None,
    context_file: str | Path | None = None,
) -> PersonalAskResult:
    question_text = _sanitize_user_question_input(question)
    if not question_text:
        raise ValueError("Question cannot be empty.")
    validate_personal_context_inputs(
        context_text=context_text,
        context_file=context_file,
    )
    context_ingests = _build_context_ingest_payloads(
        context_text=context_text,
        context_file=context_file,
    )

    workspace_path = Path(workspace)
    auto_initialized = ensure_personal_workspace(workspace_path)
    workspace_config = load_personal_connection_config(workspace_path)
    resolved_model = resolve_model_command(
        model,
        workspace_config=workspace_config,
    )
    resolved_executor_config = resolve_executor_config_for_runtime(
        executor_config,
        workspace_config=workspace_config,
    )
    if not _as_text(resolved_model):
        _write_connection_resolution_report(
            workspace_path,
            model_command=None,
            executor_config=resolved_executor_config,
            workspace_config=workspace_config,
        )
        return PersonalAskResult(
            advice=None,
            auto_initialized=auto_initialized,
            result_kind=RESULT_KIND_SETUP_REQUIRED,
            decision_adoption_status=ADOPTION_STATUS_PENDING,
            execution_status=EXECUTION_STATUS_NOT_REQUESTED,
            connection_state=CONNECTION_STATE_SETUP_REQUIRED,
            onboarding_card=_build_onboarding_decision_card(
                question=question_text,
                workspace=workspace_path,
            ),
        )
    profile_context = _load_profile_context(
        workspace_path,
        executor_config=resolved_executor_config,
        validate_if_changed=False,
    )
    resolved_executor_config = _resolve_executor_config_with_profile_mode(
        resolved_executor_config,
        profile_context.profile,
    )
    _write_connection_resolution_report(
        workspace_path,
        model_command=resolved_model,
        executor_config=resolved_executor_config,
        workspace_config=workspace_config,
    )
    runtime = _build_personal_runtime(
        workspace_path,
        model=resolved_model,
        executor_config=resolved_executor_config,
        strict_model=True,
    )
    turn = _run_advisory_turn(
        runtime,
        question=question_text,
        source="spice.personal.ask",
        model=resolved_model,
        context_ingests=context_ingests,
        profile=profile_context.profile,
        executor_config=resolved_executor_config,
        available_capabilities=profile_context.available_capabilities,
        decision_brain_output=True,
    )
    return PersonalAskResult(
        advice=turn.advice,
        auto_initialized=auto_initialized,
        evidence_notice=turn.evidence_notice,
        result_kind=_orchestration_value(
            turn.orchestration_metadata,
            "result_kind",
            RESULT_KIND_SUGGESTION,
        ),
        decision_adoption_status=_orchestration_value(
            turn.orchestration_metadata,
            "decision_adoption_status",
            ADOPTION_STATUS_PENDING,
        ),
        execution_status=_orchestration_value(
            turn.orchestration_metadata,
            "execution_status",
            EXECUTION_STATUS_NOT_REQUESTED,
        ),
        connection_state=CONNECTION_STATE_READY,
        decision_options=_extract_decision_options(turn.decision),
        recommended_option_id=_extract_recommended_option_id(turn.decision),
    )


def run_personal_session(
    *,
    workspace: str | Path = PERSONAL_DEFAULT_WORKSPACE,
    model: str | None = None,
    executor_config: PersonalExecutorConfig | None = None,
    input_stream: TextIO,
    output_stream: TextIO,
    verbose: bool = False,
) -> int:
    workspace_path = Path(workspace)
    auto_initialized = ensure_personal_workspace(workspace_path)
    workspace_config = load_personal_connection_config(workspace_path)
    resolved_model = resolve_model_command(
        model,
        workspace_config=workspace_config,
    )
    resolved_executor_config = resolve_executor_config_for_runtime(
        executor_config,
        workspace_config=workspace_config,
    )
    if not _as_text(resolved_model):
        _write_connection_resolution_report(
            workspace_path,
            model_command=None,
            executor_config=resolved_executor_config,
            workspace_config=workspace_config,
        )
        output_stream.write(
            _build_session_setup_required_message(workspace=workspace_path)
        )
        return 1

    profile_context = _load_profile_context(
        workspace_path,
        executor_config=resolved_executor_config,
        validate_if_changed=True,
    )
    resolved_executor_config = _resolve_executor_config_with_profile_mode(
        resolved_executor_config,
        profile_context.profile,
    )
    _write_connection_resolution_report(
        workspace_path,
        model_command=resolved_model,
        executor_config=resolved_executor_config,
        workspace_config=workspace_config,
    )
    runtime = _build_personal_runtime(
        workspace_path,
        model=resolved_model,
        executor_config=resolved_executor_config,
        strict_model=bool(_as_text(resolved_model)),
    )

    output_stream.write("SPICE personal advisor mode. Type 'exit' to quit.\n")
    if auto_initialized:
        output_stream.write(f"Initialized workspace: {workspace_path}\n")
    if model:
        output_stream.write(
            "Model override accepted for personal LLM decision/simulation.\n"
        )

    pending_clarify_original_question = ""
    pending_clarify_questions: tuple[dict[str, str], ...] = ()
    pending_evidence_original_question = ""
    pending_evidence_plan: tuple[dict[str, str], ...] = ()

    while True:
        output_stream.write("you> ")
        output_stream.flush()
        raw = input_stream.readline()
        if raw == "":
            break

        user_input = _sanitize_user_question_input(raw)
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        active_original_question = (
            pending_clarify_original_question
            or pending_evidence_original_question
            or user_input
        )
        effective_question = user_input
        if pending_clarify_original_question:
            effective_question = _compose_clarification_followup_question(
                original_question=pending_clarify_original_question,
                clarifying_questions=pending_clarify_questions,
                user_response=user_input,
            )
        elif pending_evidence_original_question:
            effective_question = _compose_evidence_followup_question(
                original_question=pending_evidence_original_question,
                evidence_plan=pending_evidence_plan,
                user_response=user_input,
            )

        turn = _run_advisory_turn(
            runtime,
            question=effective_question,
            source="spice.personal.session",
            model=resolved_model,
            profile=profile_context.profile,
            executor_config=resolved_executor_config,
            available_capabilities=profile_context.available_capabilities,
            decision_brain_output=True,
            choice_resolver=lambda *, advice, result_kind: _resolve_session_choice(
                advice=advice,
                result_kind=result_kind,
                input_stream=input_stream,
                output_stream=output_stream,
            ),
        )
        advice = turn.advice
        orchestration = dict(turn.orchestration_metadata or {})
        result_kind = _orchestration_value(
            orchestration,
            "result_kind",
            RESULT_KIND_SUGGESTION,
        )
        decision_adoption_status = _orchestration_value(
            orchestration,
            "decision_adoption_status",
            ADOPTION_STATUS_PENDING,
        )
        evidence_state = _orchestration_value(
            orchestration,
            "evidence_state",
            EVIDENCE_STATE_NOT_REQUESTED,
        )
        if turn.evidence_notice:
            output_stream.write(f"advisor> {turn.evidence_notice}\n")
        if evidence_state == EVIDENCE_STATE_AWAITING_MANUAL_INPUT:
            output_stream.write(
                "advisor> Waiting for your manual evidence input. "
                "Please provide externally verifiable facts for the checklist items.\n"
            )
        decision_options = _extract_decision_options(turn.decision)
        recommended_option_id = _extract_recommended_option_id(turn.decision)
        if verbose:
            if result_kind not in {RESULT_KIND_SUGGESTION, RESULT_KIND_ACTION_PROPOSAL}:
                output_stream.write(f"advisor> {advice.suggestion}\n")
            output_stream.write(
                "action={action} urgency={urgency} confidence={confidence:.2f}\n".format(
                    action=advice.selected_action,
                    urgency=advice.urgency,
                    confidence=advice.confidence,
                )
            )
            execution_status = _orchestration_value(
                orchestration,
                "execution_status",
                EXECUTION_STATUS_NOT_REQUESTED,
            )
            output_stream.write(
                "result_kind={result_kind} decision_adoption_status={adoption_status} "
                "execution_status={execution_status} evidence_state={evidence_state}\n".format(
                    result_kind=result_kind,
                    adoption_status=decision_adoption_status,
                    execution_status=execution_status,
                    evidence_state=evidence_state,
                )
            )
            if execution_status == EXECUTION_STATUS_FAILED:
                _emit_execution_failure_debug(
                    output_stream=output_stream,
                    execution_debug=orchestration.get("execution_debug"),
                )
            if decision_options:
                output_stream.write(f"decision_options_count={len(decision_options)}\n")
                if recommended_option_id:
                    output_stream.write(f"recommended_option_id={recommended_option_id}\n")
                for index, option in enumerate(decision_options, start=1):
                    output_stream.write(
                        "option_{index}_id={candidate_id} option_{index}_action={action} "
                        "option_{index}_score={score:.2f} option_{index}_confidence={confidence:.2f} "
                        "option_{index}_recommended={recommended}\n".format(
                            index=index,
                            candidate_id=_as_text(option.get("candidate_id")),
                            action=_as_text(option.get("action")),
                            score=_as_float(option.get("score"), 0.0),
                            confidence=_as_float(option.get("confidence"), 0.0),
                            recommended=(
                                "true"
                                if _as_text(option.get("candidate_id")) == recommended_option_id
                                else "false"
                            ),
                        )
                    )
                    output_stream.write(
                        "option_{index}_entry_contract_passed={passed} option_{index}_entry_contract_reasons={reasons}\n".format(
                            index=index,
                            passed=(
                                "true"
                                if bool(option.get("entry_contract_passed"))
                                else "false"
                            ),
                            reasons=_join_text_values(option.get("entry_contract_reasons")),
                        )
                    )
                    output_stream.write(
                        f"option_{index}_benefits={_join_text_values(option.get('benefits'))}\n"
                    )
                    output_stream.write(
                        f"option_{index}_risks={_join_text_values(option.get('risks'))}\n"
                    )
                    output_stream.write(
                        f"option_{index}_key_assumptions={_join_text_values(option.get('key_assumptions'))}\n"
                    )
                    output_stream.write(
                        f"option_{index}_first_step_24h={_user_visible_text(option.get('first_step_24h'))}\n"
                    )
                    output_stream.write(
                        f"option_{index}_stop_loss_trigger={_user_visible_text(option.get('stop_loss_trigger'))}\n"
                    )
                    output_stream.write(
                        f"option_{index}_change_mind_condition={_user_visible_text(option.get('change_mind_condition'))}\n"
                    )
            output_stream.write(
                "decision_id={decision_id} advisory_outcome_id={outcome_id} "
                "reflection_id={reflection_id}\n".format(
                    decision_id=turn.decision.id,
                    outcome_id=turn.outcome.id,
                    reflection_id=turn.reflection.id,
                )
            )
        elif (
            result_kind == RESULT_KIND_SUGGESTION
            and advice.selected_action == PERSONAL_ACTION_SUGGEST
            and decision_adoption_status == ADOPTION_STATUS_ADOPTED
        ):
            output_stream.write(
                _render_adopted_suggestion_confirmation(
                    decision=turn.decision,
                    decision_options=decision_options,
                    recommended_option_id=recommended_option_id,
                )
                + "\n"
            )
        clarify_questions = _extract_clarifying_questions_from_attributes(
            turn.decision.attributes if isinstance(turn.decision.attributes, dict) else {},
            question=active_original_question,
        )
        evidence_plan = _extract_evidence_plan_from_attributes(
            turn.decision.attributes if isinstance(turn.decision.attributes, dict) else {},
            question=active_original_question,
        )
        executor_mode = _as_text(getattr(resolved_executor_config, "mode", "")).lower()
        if (
            advice.selected_action == PERSONAL_ACTION_ASK_CLARIFY
            and decision_adoption_status == ADOPTION_STATUS_ADOPTED
        ):
            pending_clarify_original_question = active_original_question
            pending_clarify_questions = clarify_questions
            pending_evidence_original_question = ""
            pending_evidence_plan = ()
        elif (
            advice.selected_action == PERSONAL_ACTION_GATHER_EVIDENCE
            and executor_mode == EXECUTOR_MODE_MOCK
            and decision_adoption_status != ADOPTION_STATUS_DECLINED
            and evidence_state == EVIDENCE_STATE_AWAITING_MANUAL_INPUT
        ):
            pending_evidence_original_question = active_original_question
            pending_evidence_plan = evidence_plan
            pending_clarify_original_question = ""
            pending_clarify_questions = ()
        else:
            pending_clarify_original_question = ""
            pending_clarify_questions = ()
            pending_evidence_original_question = ""
            pending_evidence_plan = ()
        _save_personal_state(workspace_path, runtime.state_store.get_state())

    return 0


def ensure_personal_workspace(workspace: Path) -> bool:
    if not workspace.exists():
        run_personal_init(workspace=workspace, force=False)
        return True

    if _is_valid_personal_workspace(workspace):
        return False

    if workspace.is_dir() and not any(workspace.iterdir()):
        run_personal_init(workspace=workspace, force=True)
        return True

    raise RuntimeError(
        f"Existing workspace is not a valid personal advisor scaffold: {workspace}. "
        "Run `spice-personal init --force` to reset it."
    )


def _is_valid_personal_workspace(workspace: Path) -> bool:
    domain_spec_path = workspace / "domain_spec.json"
    if not domain_spec_path.exists():
        return False

    try:
        payload = json.loads(domain_spec_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False

    try:
        spec = DomainSpec.from_dict(payload)
    except Exception:
        return False

    return spec.domain.id == PERSONAL_DOMAIN_ID


def _build_personal_runtime(
    workspace: Path,
    *,
    model: str | None,
    executor_config: PersonalExecutorConfig | None,
    strict_model: bool = False,
) -> SpiceRuntime:
    spec = load_builtin_personal_spec()
    domain_pack = _load_generated_domain_pack(workspace, spec)
    initial_state = _load_personal_state(workspace)
    state_store = StateStore(initial_state=initial_state) if initial_state else StateStore()
    decision_policy = build_personal_llm_decision_policy(
        model=model,
        domain=PERSONAL_DOMAIN_ID,
        allowed_actions=spec.vocabulary.action_types,
        strict_model=strict_model,
    )
    executor = build_executor(executor_config or PersonalExecutorConfig())
    return SpiceRuntime(
        state_store=state_store,
        domain_pack=domain_pack,
        executor=executor,
        decision_policy=decision_policy,
    )


def _load_generated_domain_pack(workspace: Path, spec: DomainSpec) -> Any:
    package_name = derive_package_name(spec.domain.id)
    class_name = derive_domain_pack_class_name(spec.domain.id)

    workspace_str = str(workspace.resolve())
    if workspace_str not in sys.path:
        sys.path.insert(0, workspace_str)

    # Ensure we always load from the selected workspace and not stale module cache.
    for module_name in list(sys.modules):
        if module_name == package_name or module_name.startswith(f"{package_name}."):
            sys.modules.pop(module_name, None)

    module = importlib.import_module(f"{package_name}.domain_pack")
    domain_pack_class = getattr(module, class_name, None)
    if domain_pack_class is None:
        raise RuntimeError(
            f"Generated domain pack class {class_name!r} was not found in {package_name}.domain_pack."
        )
    return domain_pack_class()


def _run_advisory_turn(
    runtime: SpiceRuntime,
    *,
    question: str,
    source: str,
    model: str | None,
    context_ingests: list[dict[str, Any]] | None = None,
    profile: dict[str, Any] | None = None,
    executor_config: PersonalExecutorConfig | None = None,
    available_capabilities: tuple[str, ...] = (),
    decision_brain_output: bool = False,
    choice_resolver: Callable[..., str] | None = None,
) -> AdvisoryTurnResult:
    metadata = {"mode": "personal_advisor"}
    if model:
        metadata["model_override"] = model

    question_attributes = {
        "status": "ready",
        "latest_question": question,
    }
    context_state_preview = _build_context_state_preview(context_ingests)
    if context_state_preview:
        question_attributes["evidence_summary"] = context_state_preview

    observation = runtime.observe(
        observation_type=PERSONAL_OBSERVATION_QUESTION_RECEIVED,
        source=source,
        attributes=question_attributes,
        metadata=metadata,
    )

    state_after_observation = runtime.update_state(observation)
    context_observations: list[Observation] = []
    state_before_decide = state_after_observation
    if isinstance(context_ingests, list):
        for index, payload in enumerate(context_ingests):
            if not isinstance(payload, dict) or not payload:
                continue
            ingest_attributes = _normalize_context_ingest_attributes(payload)
            if not ingest_attributes:
                continue
            context_observation = runtime.observe(
                observation_type=PERSONAL_OBSERVATION_CONTEXT_INGEST,
                source=f"{source}.context.{index + 1}",
                attributes=ingest_attributes,
                metadata={"mode": "personal_advisor", "ingest": True},
            )
            state_before_decide = runtime.update_state(context_observation)
            context_observations.append(context_observation)

    first_decision = runtime.decide(state_before_decide)
    evidence_round: EvidenceRoundResult | None = None
    state_for_final_decision = state_before_decide
    final_decision = first_decision
    evidence_notice: str | None = None
    executor_mode = _as_text(getattr(executor_config, "mode", "")).lower()

    if should_gather_evidence(
        first_decision,
        evidence_action=PERSONAL_ACTION_GATHER_EVIDENCE,
    ):
        if executor_mode == EXECUTOR_MODE_MOCK:
            evidence_round = run_mock_evidence_round(
                decision=first_decision,
                source=f"{source}.evidence.manual",
                notice=(
                    "Evidence action selected, but external evidence agent is not configured "
                    "(executor=mock). Prepared a manual evidence checklist and re-evaluated once."
                ),
            )
            evidence_notice = evidence_round.notice
        else:
            evidence_round = run_bounded_evidence_round(
                runtime,
                decision=first_decision,
                source=f"{source}.evidence",
                prepare_intent=lambda intent: _prepare_evidence_intent(
                    intent,
                    decision=first_decision,
                    question=question,
                    profile=profile,
                    executor_config=executor_config,
                    available_capabilities=available_capabilities,
                ),
            )
            evidence_notice = evidence_round.notice
        if evidence_round is not None and evidence_round.evidence_observation is not None:
            state_for_final_decision = runtime.update_state(
                evidence_round.evidence_observation
            )
        if evidence_round is not None:
            final_decision = runtime.decide(state_for_final_decision)
            if (
                executor_mode == EXECUTOR_MODE_MOCK
                and not _has_manual_evidence_response(question)
                and should_gather_evidence(
                    first_decision,
                    evidence_action=PERSONAL_ACTION_GATHER_EVIDENCE,
                )
                and not should_gather_evidence(
                    final_decision,
                    evidence_action=PERSONAL_ACTION_GATHER_EVIDENCE,
                )
            ):
                # For mock executor, keep gather_evidence active until user provides
                # a concrete manual evidence response in the follow-up turn.
                final_decision = first_decision

    advice = _build_advice_from_decision(
        state=state_for_final_decision,
        question=question,
        decision=final_decision,
        decision_brain_output=decision_brain_output,
    )
    result_kind = _decision_result_kind(final_decision)
    if result_kind == RESULT_KIND_ACTION_PROPOSAL:
        _ensure_action_proposal_execution_brief(final_decision, advice=advice)
    decision_adoption_status = _resolve_decision_adoption_status(
        result_kind=result_kind,
        advice=advice,
        choice_resolver=choice_resolver,
    )
    evidence_state = _resolve_evidence_state(
        evidence_round=evidence_round,
        executor_mode=executor_mode,
        question=question,
    )
    if (
        evidence_state == EVIDENCE_STATE_AWAITING_MANUAL_INPUT
        and decision_adoption_status == ADOPTION_STATUS_DECLINED
    ):
        evidence_state = EVIDENCE_STATE_NOT_REQUESTED
    elif evidence_state == EVIDENCE_STATE_AWAITING_MANUAL_INPUT:
        decision_adoption_status = ADOPTION_STATUS_PENDING
        _set_pending_evidence_decision_attributes(final_decision)

    execution_status = EXECUTION_STATUS_NOT_REQUESTED
    execution_intent: ExecutionIntent | None = None
    execution_result: ExecutionResult | None = None
    execution_outcome: Outcome | None = None
    execution_debug: dict[str, Any] | None = None
    if (
        result_kind == RESULT_KIND_ACTION_PROPOSAL
        and decision_adoption_status == ADOPTION_STATUS_ADOPTED
    ):
        (
            execution_outcome,
            execution_intent,
            execution_result,
            execution_status,
            execution_debug,
        ) = _run_confirmed_action_proposal_execution(
            runtime,
            decision=final_decision,
            profile=profile,
            executor_config=executor_config,
            available_capabilities=available_capabilities,
        )

    orchestration_metadata = {
        "result_kind": result_kind,
        "decision_adoption_status": decision_adoption_status,
        "execution_status": execution_status,
        "evidence_state": evidence_state,
    }
    if isinstance(execution_debug, dict) and execution_debug:
        orchestration_metadata["execution_debug"] = dict(execution_debug)
    if evidence_state == EVIDENCE_STATE_AWAITING_MANUAL_INPUT:
        advisory_mode = "evidence_requested"
    else:
        advisory_mode = "evidence_refined" if evidence_round is not None else "direct"
    advisory_refs: list[str] = []
    if first_decision.id and first_decision.id != final_decision.id:
        advisory_refs.append(first_decision.id)
    for context_observation in context_observations:
        advisory_refs.append(context_observation.id)
    if evidence_round is not None:
        if evidence_round.evidence_observation is not None:
            advisory_refs.append(evidence_round.evidence_observation.id)
        if evidence_round.execution_result is not None:
            advisory_refs.append(evidence_round.execution_result.id)
        if evidence_round.execution_outcome is not None:
            advisory_refs.append(evidence_round.execution_outcome.id)

    if execution_intent is not None:
        advisory_refs.append(execution_intent.id)
    if execution_result is not None:
        advisory_refs.append(execution_result.id)
    if execution_outcome is not None:
        advisory_refs.append(execution_outcome.id)

    final_outcome = _build_advisory_outcome(
        question=question,
        advice=advice,
        observation=observation,
        decision=final_decision,
        advisory_mode=advisory_mode,
        extra_refs=advisory_refs,
        orchestration_metadata=orchestration_metadata,
    )
    _attach_orchestration_metadata(final_outcome, orchestration_metadata)

    state_after_outcome = runtime.update_state(final_outcome)

    reflection_intent = execution_intent
    reflection_result = execution_result
    if reflection_intent is None and evidence_round is not None:
        reflection_intent = evidence_round.execution_intent
    if reflection_result is None and evidence_round is not None:
        reflection_result = evidence_round.execution_result
    reflection = runtime.reflect(
        final_outcome,
        decision=final_decision,
        intent=reflection_intent,
        execution_result=reflection_result,
    )
    _attach_orchestration_metadata(reflection, orchestration_metadata)

    return AdvisoryTurnResult(
        observation=observation,
        decision=final_decision,
        outcome=final_outcome,
        reflection=reflection,
        world_state=state_after_outcome,
        advice=advice,
        evidence_notice=evidence_notice,
        orchestration_metadata=orchestration_metadata,
    )


def _build_advice_from_decision(
    *,
    state: WorldState,
    question: str,
    decision: Decision,
    decision_brain_output: bool = False,
) -> PersonalAdvice:
    # Personal advisory fields are normalized by PersonalLLMDecisionPolicy.
    # Contract keys: suggestion_text, confidence, urgency, score, simulation_rationale, result_kind.
    action = decision.selected_action or PERSONAL_ACTION_SUGGEST
    attributes = decision.attributes if isinstance(decision.attributes, dict) else {}
    entity = _current_personal_entity(state)
    urgency = _as_text(attributes.get("urgency")) or str(entity.get("urgency") or "normal")
    confidence = _as_float(
        attributes.get("confidence"),
        _as_float(entity.get("confidence"), 0.0),
    )
    if action == PERSONAL_ACTION_ASK_CLARIFY:
        clarifying_questions = _extract_clarifying_questions_from_attributes(
            attributes,
            question=question,
        )
        suggestion = _render_clarifying_questions(clarifying_questions)
    elif action == PERSONAL_ACTION_GATHER_EVIDENCE:
        evidence_plan = _extract_evidence_plan_from_attributes(
            attributes,
            question=question,
        )
        suggestion = _render_evidence_plan(evidence_plan)
    elif action == PERSONAL_ACTION_DEFER:
        defer_plan = _extract_defer_plan_from_attributes(attributes)
        suggestion = _render_defer_plan(
            question=question,
            defer_plan=defer_plan,
            urgency=urgency,
            confidence=confidence,
        )
    else:
        if decision_brain_output:
            suggestion = _render_suggest_decision_report(
                question=question,
                decision=decision,
                urgency=urgency,
                confidence=confidence,
            )
        else:
            suggestion = _user_visible_text(attributes.get("suggestion_text"))
            if not suggestion:
                suggestion = _render_suggestion(
                    selected_action=action,
                    question=question,
                    urgency=urgency,
                    confidence=confidence,
                )

    return PersonalAdvice(
        selected_action=action,
        suggestion=suggestion,
        urgency=urgency,
        confidence=confidence,
    )


def _render_adopted_suggestion_confirmation(
    *,
    decision: Decision,
    decision_options: tuple[dict[str, Any], ...],
    recommended_option_id: str,
) -> str:
    recommended = _recommended_option_payload(
        decision_options=decision_options,
        recommended_option_id=recommended_option_id,
    )
    choice_label = _recommended_choice_label(
        decision=decision,
        recommended=recommended,
    )
    first_step = _recommended_first_step_24h(
        decision=decision,
        recommended=recommended,
    )
    secondary_step = _recommended_optional_next_step(
        decision=decision,
        recommended=recommended,
    )
    lines = [
        f"✅ Your choice has been recorded: {choice_label}",
        "",
        "Next steps:",
        f"- {first_step}",
    ]
    if secondary_step:
        lines.append(f"- {secondary_step}")
    lines.extend(
        [
            "",
            (
                "You can come back anytime if new information appears, "
                "and I can help you re-evaluate this decision."
            ),
        ]
    )
    return "\n".join(lines)


def _recommended_option_payload(
    *,
    decision_options: tuple[dict[str, Any], ...],
    recommended_option_id: str,
) -> dict[str, Any]:
    if recommended_option_id:
        for option in decision_options:
            if _as_text(option.get("candidate_id")) == recommended_option_id:
                return dict(option)
    if decision_options:
        first = decision_options[0]
        return dict(first) if isinstance(first, dict) else {}
    return {}


def _recommended_choice_label(
    *,
    decision: Decision,
    recommended: dict[str, Any],
) -> str:
    report_label = _recommended_label_from_decision_report(decision)
    option_label = (
        _as_text(recommended.get("label"))
        or _as_text(recommended.get("option_label"))
        or _as_text(recommended.get("title"))
        or _as_text(recommended.get("name"))
    )
    suggestion_text = _as_text(recommended.get("suggestion_text"))
    for candidate in (report_label, suggestion_text, option_label):
        normalized = _normalize_recommendation_label(candidate)
        if normalized:
            return normalized
    return "this option"


def _recommended_label_from_decision_report(decision: Decision) -> str:
    attributes = decision.attributes if isinstance(decision.attributes, dict) else {}
    payload = attributes.get("decision_brain_report")
    if not isinstance(payload, dict):
        return ""
    return _as_text(payload.get("recommended_option_label"))


def _normalize_recommendation_label(value: str) -> str:
    token = _as_text(value)
    if not token:
        return ""
    for pattern in (
        r"(?i)\boffer\s*([A-C])\b",
        r"(?i)\boption\s*([A-C])\b",
        r"方案\s*([A-C])",
        r"选项\s*([A-C])",
        r"\b([A-C])\b",
    ):
        matched = re.search(pattern, token)
        if matched:
            label = _as_text(matched.group(1)).upper()
            if label in {"A", "B", "C"}:
                return label
    if len(token) <= 24:
        return token
    return ""


def _recommended_first_step_24h(
    *,
    decision: Decision,
    recommended: dict[str, Any],
) -> str:
    step = _user_visible_text(recommended.get("first_step_24h"))
    if step:
        return step
    attributes = decision.attributes if isinstance(decision.attributes, dict) else {}
    step = _user_visible_text(attributes.get("first_step_24h"))
    if step:
        return step
    return "Confirm key conditions and proceed with the next step"


def _recommended_optional_next_step(
    *,
    decision: Decision,
    recommended: dict[str, Any],
) -> str:
    for key in ("next_step_24h", "second_step_24h", "followup_step"):
        token = _user_visible_text(recommended.get(key))
        if token:
            return token
    attributes = decision.attributes if isinstance(decision.attributes, dict) else {}
    for key in ("next_step_24h", "second_step_24h", "followup_step"):
        token = _user_visible_text(attributes.get(key))
        if token:
            return token
    return ""


def _extract_clarifying_questions_from_attributes(
    attributes: dict[str, Any],
    *,
    question: str,
) -> tuple[dict[str, str], ...]:
    degraded = bool(attributes.get("advisory_degraded"))
    payload = attributes.get("clarifying_questions")
    if isinstance(payload, list):
        normalized: list[dict[str, str]] = []
        for item in payload[:3]:
            if isinstance(item, str):
                text = _as_text(item)
                if text:
                    normalized.append(
                        {
                            "question": text,
                            "why": "",
                        }
                    )
                continue
            if not isinstance(item, dict):
                continue
            query = _as_text(item.get("question")) or _as_text(item.get("q"))
            why = _as_text(item.get("why")) or _as_text(item.get("reason"))
            if query:
                normalized.append({"question": query, "why": why})
        if normalized:
            return tuple(normalized)
    if not degraded:
        return ()
    topic = _as_text(question) or "this decision"
    return (
        {
            "question": "What is your one non-negotiable requirement for this decision?",
            "why": f"It narrows choices using must-have constraints for {topic}.",
        },
        {
            "question": "How much short-term downside can you tolerate over the next 12 months?",
            "why": "It determines whether higher-variance options are acceptable.",
        },
        {
            "question": "What measurable outcome defines success in 3 years?",
            "why": "It anchors the final recommendation to a clear objective.",
        },
    )


def _extract_evidence_plan_from_attributes(
    attributes: dict[str, Any],
    *,
    question: str,
) -> tuple[dict[str, str], ...]:
    degraded = bool(attributes.get("advisory_degraded"))
    payload = attributes.get("evidence_plan")
    normalized = _normalize_user_visible_evidence_plan(
        payload,
        question=question,
    )
    if len(normalized) == 3:
        return tuple(normalized)
    if not degraded and normalized:
        # Keep user-visible evidence grounded in externally verifiable facts.
        return _default_external_evidence_plan(question)
    if not degraded and not normalized:
        return _default_external_evidence_plan(question)
    return _default_external_evidence_plan(question)


def _normalize_user_visible_evidence_plan(
    value: Any,
    *,
    question: str,
) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, str]] = []
    for item in value[:3]:
        if isinstance(item, str):
            fact = _as_text(item)
            why = ""
        elif isinstance(item, dict):
            fact = _as_text(item.get("fact")) or _as_text(item.get("item")) or _as_text(item.get("question"))
            why = _as_text(item.get("why")) or _as_text(item.get("reason"))
        else:
            continue
        if not fact or not why:
            continue
        if _contains_internal_runtime_evidence_marker(f"{fact} {why}"):
            continue
        if not _references_question_entities(fact, question=question):
            continue
        normalized.append({"fact": fact, "why": why})
    return normalized


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


def _question_entity_tokens(question: str) -> tuple[str, ...]:
    lowered = _as_text(question).lower()
    if not lowered:
        return ()
    tokens: list[str] = []
    for marker in (
        "offer",
        "option",
        "salary",
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
        "薪资",
        "现金流",
        "经理",
        "导师",
        "团队",
        "离职",
        "流失",
        "晋升",
        "管理",
        "职责",
    ):
        if marker in lowered:
            tokens.append(marker)
    for label in re.findall(r"(?<![A-Za-z0-9])([A-Ca-c])(?![A-Za-z0-9])", question):
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
    for part in re.split(r"[\s,.;:!?()\[\]{}<>/\\|\"'\n\t]+", lowered):
        token = _as_text(part)
        if len(token) < 3:
            continue
        if token in {
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
        }:
            continue
        tokens.append(token)
    return tuple(dict.fromkeys(tokens[:24]))


def _references_question_entities(text: str, *, question: str) -> bool:
    lowered = _as_text(text).lower()
    if not lowered:
        return False
    for entity in _question_entity_tokens(question):
        token = _as_text(entity).lower()
        if token and token in lowered:
            return True
    return False


def _default_external_evidence_plan(question: str) -> tuple[dict[str, str], ...]:
    topic = _as_text(question) or "this decision"
    return (
        {
            "fact": "Verify team stability and attrition patterns for each option.",
            "why": f"It reduces execution risk uncertainty for {topic}.",
        },
        {
            "fact": "Validate manager quality and mentorship track record from direct sources.",
            "why": "It affects long-term growth probability.",
        },
        {
            "fact": "Compare realistic workload and downside scenarios over the next year.",
            "why": "It checks sustainability under stress.",
        },
    )


def _extract_defer_plan_from_attributes(attributes: dict[str, Any]) -> dict[str, str]:
    degraded = bool(attributes.get("advisory_degraded"))
    payload = attributes.get("defer_plan")
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
    if not degraded:
        return {}
    return {
        "revisit_at": "7 days",
        "monitor_signal": "new high-confidence evidence",
        "resume_trigger": "re-evaluate when the signal appears",
    }


def _render_clarifying_questions(questions: tuple[dict[str, str], ...]) -> str:
    lines = ["Clarifying Questions"]
    if not questions:
        lines.append("Contract warning: clarifying_questions is missing or incomplete.")
        return "\n".join(lines)
    for index, item in enumerate(questions[:3], start=1):
        question = _as_text(item.get("question"))
        why = _as_text(item.get("why"))
        if not question:
            continue
        lines.append(f"Q{index}: {question}")
        if why:
            lines.append(f"Why: {why}")
        else:
            lines.append("Why: [missing; should explain how ranking/recommendation would change]")
    if len(questions) != 3:
        lines.append("Contract warning: expected exactly 3 clarifying questions.")
    return "\n".join(lines)


def _render_evidence_plan(plan: tuple[dict[str, str], ...]) -> str:
    lines = [
        "Evidence Collection Plan",
        "External facts only: these items must be verified in the real world.",
    ]
    if not plan:
        lines.append("Contract warning: evidence_plan is missing or incomplete.")
        return "\n".join(lines)
    for index, item in enumerate(plan[:3], start=1):
        fact = _user_visible_text(item.get("fact"))
        why = _user_visible_text(item.get("why"))
        if not fact:
            continue
        lines.append(f"{index}. {fact}")
        if why:
            lines.append(f"   Why: {why}")
        else:
            lines.append("   Why: [missing; should explain recommendation impact]")
    if len(plan) != 3:
        lines.append("Contract warning: expected exactly 3 evidence items.")
    return "\n".join(lines)


def _render_defer_plan(
    *,
    question: str,
    defer_plan: dict[str, str],
    urgency: str,
    confidence: float,
) -> str:
    confidence_text = f"{confidence:.2f}"
    revisit_at = _as_text(defer_plan.get("revisit_at"))
    monitor_signal = _as_text(defer_plan.get("monitor_signal"))
    resume_trigger = _as_text(defer_plan.get("resume_trigger"))
    rendered = (
        "Deferred Decision Plan\n"
        f"Topic: {question}\n"
        f"Revisit at: {revisit_at or '[missing]'}\n"
        f"Monitor signal: {monitor_signal or '[missing]'}\n"
        f"Resume trigger: {resume_trigger or '[missing]'}\n"
        f"Current urgency={urgency}, confidence={confidence_text}"
    )
    if revisit_at and monitor_signal and resume_trigger:
        return rendered
    return rendered + "\nContract warning: defer_plan is incomplete."


def _render_suggest_decision_report(
    *,
    question: str,
    decision: Decision,
    urgency: str,
    confidence: float,
) -> str:
    attributes = decision.attributes if isinstance(decision.attributes, dict) else {}
    summary = _as_text(attributes.get("suggestion_text"))
    decision_brain_report = _extract_decision_brain_report_from_attributes(attributes)
    if decision_brain_report:
        return _render_structured_decision_brain_report(
            question=question,
            urgency=urgency,
            confidence=confidence,
            summary=summary,
            report=decision_brain_report,
        )
    options = _extract_decision_options(decision)
    recommended_option_id = _extract_recommended_option_id(decision)
    lines = [
        "Spice recommendation",
        f"Question: {question}",
        f"Current urgency={urgency}, confidence={confidence:.2f}",
    ]
    if summary:
        lines.append(_user_visible_text(summary))
    if not options:
        return "\n".join(lines)

    recommended_option: dict[str, Any] | None = None
    for index, option in enumerate(options[:3], start=1):
        option_id = _as_text(option.get("candidate_id"))
        marker = " ★" if option_id and option_id == recommended_option_id else ""
        if marker:
            recommended_option = option
        lines.append(f"Option {index}{marker}")
        for segment in _option_content_segments(option):
            lines.append(segment)
        lines.append("")
    if isinstance(recommended_option, dict):
        recommendation_reason = _user_visible_text(recommended_option.get("recommendation_reason"))
        if not recommendation_reason:
            recommendation_reason = _user_visible_text(attributes.get("recommendation_reason"))
        if not recommendation_reason:
            recommendation_reason = _user_visible_text(recommended_option.get("suggestion_text"))
        if recommendation_reason:
            lines.append(f"当前推荐成立的核心原因：{recommendation_reason}")
        change_mind = _user_visible_text(
            recommended_option.get("what_would_change_my_mind")
        )
        if not change_mind:
            change_mind = _user_visible_text(recommended_option.get("change_mind_condition"))
        if not change_mind:
            change_mind = _user_visible_text(attributes.get("what_would_change_my_mind"))
        if not change_mind:
            change_mind = _user_visible_text(attributes.get("change_mind_condition"))
        if change_mind:
            lines.append(f"如果出现以下变化，我会改判：{change_mind}")
    return "\n".join(lines)


def _option_content_segments(option: dict[str, Any]) -> list[str]:
    segments: list[str] = []
    option_positioning = _user_visible_text(option.get("option_positioning"))
    suggestion = _user_visible_text(option.get("suggestion_text"))
    if option_positioning:
        segments.append(option_positioning)
    elif suggestion:
        segments.append(suggestion)

    benefits = _joined_natural_list(option.get("benefits"))
    if benefits:
        segments.append(f"它的主要收益在于：{benefits}")
    risks = _joined_natural_list(option.get("risks"))
    if risks:
        segments.append(f"你需要承担的风险是：{risks}")
    assumptions = _joined_natural_list(option.get("key_assumptions"))
    if assumptions:
        segments.append(f"这个判断依赖的前提是：{assumptions}")

    first_step = _user_visible_text(option.get("first_step_24h"))
    if first_step:
        segments.append(f"接下来24小时建议先：{first_step}")
    stop_loss = _user_visible_text(option.get("stop_loss_trigger"))
    if stop_loss:
        segments.append(f"若出现这个信号应及时止损：{stop_loss}")
    change_mind = _user_visible_text(option.get("change_mind_condition"))
    if change_mind:
        segments.append(f"若条件反转可改判：{change_mind}")

    deduped: list[str] = []
    seen: set[str] = set()
    for segment in segments:
        marker = segment.strip()
        if not marker or marker in seen:
            continue
        deduped.append(marker)
        seen.add(marker)
    return deduped


def _joined_natural_list(value: Any) -> str:
    if isinstance(value, str):
        token = _user_visible_text(value)
        return token if token else ""
    if not isinstance(value, list):
        return ""
    items: list[str] = []
    for item in value:
        token = _user_visible_text(item)
        if token:
            items.append(token)
    if not items:
        return ""
    return "；".join(items)


def _join_text_values(value: Any) -> str:
    if isinstance(value, str):
        token = _user_visible_text(value)
        return token if token else "n/a"
    if not isinstance(value, list):
        return "n/a"
    tokens: list[str] = []
    for item in value:
        token = _user_visible_text(item)
        if token:
            tokens.append(token)
    if not tokens:
        return "n/a"
    return "; ".join(tokens)


def _user_visible_text(value: Any) -> str:
    token = _as_text(value)
    if not token:
        return ""
    if _looks_internal_runtime_text(token):
        return ""
    return token


def _looks_internal_runtime_text(text: str) -> bool:
    lowered = _as_text(text).lower()
    if not lowered:
        return False
    for marker in INTERNAL_RUNTIME_TEXT_MARKERS:
        token = _as_text(marker).lower()
        if token and token in lowered:
            return True
    return False


def _extract_decision_brain_report_from_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    payload = attributes.get("decision_brain_report")
    if not isinstance(payload, dict):
        return {}
    options_raw = payload.get("options")
    if not isinstance(options_raw, list):
        return {}
    options: list[dict[str, Any]] = []
    for index, item in enumerate(options_raw):
        if len(options) >= 3:
            break
        if not isinstance(item, dict):
            continue
        label = _as_text(item.get("label") or item.get("title") or item.get("name"))
        if not label:
            label = f"Option {chr(ord('A') + min(index, 25))}"
        option_payload = {
            "option_id": _as_text(item.get("option_id")) or f"option-{index + 1}",
            "option_rank": _as_int(item.get("option_rank"), index + 1),
            "option_label": _as_text(item.get("option_label")) or label,
            "label": label,
            "option_positioning": _as_text(
                item.get("option_positioning")
                or item.get("positioning")
                or item.get("judgement")
                or item.get("judgment")
            ),
            "suggestion_text": _as_text(item.get("suggestion_text") or item.get("summary")),
            "benefits": _normalize_text_list_local(item.get("benefits")),
            "risks": _normalize_text_list_local(item.get("risks")),
            "key_assumptions": _normalize_text_list_local(item.get("key_assumptions") or item.get("assumptions")),
            "first_step_24h": _as_text(item.get("first_step_24h") or item.get("first_step")),
            "stop_loss_trigger": _as_text(item.get("stop_loss_trigger") or item.get("stop_loss")),
            "change_mind_condition": _as_text(
                item.get("change_mind_condition") or item.get("what_would_change_my_mind")
            ),
        }
        has_content = any(
            [
                option_payload["option_positioning"],
                option_payload["suggestion_text"],
                option_payload["benefits"],
                option_payload["risks"],
                option_payload["key_assumptions"],
                option_payload["first_step_24h"],
                option_payload["stop_loss_trigger"],
            ]
        )
        if not has_content:
            continue
        options.append(option_payload)
    if len(options) < 2:
        return {}
    return {
        "options": options,
        "recommended_option_label": _as_text(payload.get("recommended_option_label") or payload.get("recommended_option")),
        "recommended_option_rank": _as_int(payload.get("recommended_option_rank"), 0),
        "recommendation_reason": _as_text(payload.get("recommendation_reason") or payload.get("recommended_reason")),
        "what_would_change_my_mind": _as_text(
            payload.get("what_would_change_my_mind") or payload.get("change_mind_condition")
        ),
    }


def _render_structured_decision_brain_report(
    *,
    question: str,
    urgency: str,
    confidence: float,
    summary: str,
    report: dict[str, Any],
) -> str:
    options = report.get("options")
    if not isinstance(options, list):
        options = []
    recommended_label = _as_text(report.get("recommended_option_label"))
    recommended_rank = _as_int(report.get("recommended_option_rank"), 0)
    recommendation_reason = _as_text(report.get("recommendation_reason")) or summary
    change_mind = _as_text(report.get("what_would_change_my_mind"))
    if not recommended_label and options:
        first_option = options[0]
        if isinstance(first_option, dict):
            recommended_label = _as_text(first_option.get("label"))

    lines = [
        "Spice recommendation",
        f"Question: {question}",
        f"Current urgency={urgency}, confidence={confidence:.2f}",
    ]
    if summary:
        lines.append(_user_visible_text(summary))
    for option in options[:3]:
        if not isinstance(option, dict):
            continue
        label = _as_text(option.get("label")) or "Option"
        option_rank = _as_int(option.get("option_rank"), 0)
        marker = " ★" if (
            (recommended_label and label == recommended_label)
            or (recommended_rank > 0 and option_rank == recommended_rank)
        ) else ""
        lines.append(f"{label}{marker}")
        for segment in _option_content_segments(option):
            lines.append(segment)
        lines.append("")
    if recommended_label:
        lines.append(f"★ {recommended_label}")
    if recommendation_reason:
        lines.append(f"当前推荐成立的核心原因：{_user_visible_text(recommendation_reason)}")
    if change_mind:
        lines.append(f"如果出现以下变化，我会改判：{_user_visible_text(change_mind)}")
    return "\n".join(lines)


def _normalize_text_list_local(value: Any) -> list[str]:
    if isinstance(value, str):
        token = _as_text(value)
        return [token] if token else []
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        token = _as_text(item)
        if token:
            normalized.append(token)
    return normalized


def _compose_clarification_followup_question(
    *,
    original_question: str,
    clarifying_questions: tuple[dict[str, str], ...],
    user_response: str,
) -> str:
    lines = [
        "Original decision request:",
        _as_text(original_question),
        "",
        "Clarifying questions asked:",
    ]
    for index, item in enumerate(clarifying_questions[:3], start=1):
        query = _as_text(item.get("question"))
        if not query:
            continue
        lines.append(f"Q{index}: {query}")
    lines.extend(
        [
            "",
            "User clarification response:",
            _as_text(user_response),
        ]
    )
    return "\n".join(lines).strip()


def _compose_evidence_followup_question(
    *,
    original_question: str,
    evidence_plan: tuple[dict[str, str], ...],
    user_response: str,
) -> str:
    lines = [
        "Original decision request:",
        _as_text(original_question),
        "",
        "Evidence collection checklist:",
    ]
    for index, item in enumerate(evidence_plan[:3], start=1):
        fact = _user_visible_text(item.get("fact"))
        why = _user_visible_text(item.get("why"))
        if not fact:
            continue
        lines.append(f"E{index}: {fact}")
        if why:
            lines.append(f"Why: {why}")
    lines.extend(
        [
            "",
            "User evidence response:",
            _as_text(user_response),
        ]
    )
    return "\n".join(lines).strip()


def _has_manual_evidence_response(question: str) -> bool:
    token = _as_text(question).lower()
    if not token:
        return False
    return "user evidence response:" in token


def _resolve_evidence_state(
    *,
    evidence_round: EvidenceRoundResult | None,
    executor_mode: str,
    question: str,
) -> str:
    if evidence_round is None:
        return EVIDENCE_STATE_NOT_REQUESTED
    if executor_mode == EXECUTOR_MODE_MOCK:
        if _has_manual_evidence_response(question):
            return EVIDENCE_STATE_REEVALUATED
        return EVIDENCE_STATE_AWAITING_MANUAL_INPUT
    return EVIDENCE_STATE_EXTERNAL_EXECUTION


def _set_pending_evidence_decision_attributes(decision: Decision) -> None:
    if not isinstance(decision.attributes, dict):
        decision.attributes = {}
    decision.attributes["recommended_option_id"] = ""
    decision.attributes["selected_candidate_id"] = ""


def _extract_decision_options(decision: Decision) -> tuple[dict[str, Any], ...]:
    attributes = decision.attributes if isinstance(decision.attributes, dict) else {}
    payload = attributes.get("decision_options")
    if not isinstance(payload, list):
        return ()
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(payload[:3], start=1):
        if not isinstance(item, dict):
            continue
        entry = dict(item)
        if _as_int(entry.get("option_rank"), 0) <= 0:
            entry["option_rank"] = index
        label = _as_text(entry.get("option_label"))
        if not label:
            label = _default_option_label(index)
            entry["option_label"] = label
        option_id = _as_text(entry.get("option_id"))
        if not option_id:
            option_id = _as_text(entry.get("candidate_id"))
            if option_id:
                entry["option_id"] = option_id
        normalized.append(entry)
    return tuple(normalized)


def _extract_recommended_option_id(decision: Decision) -> str:
    attributes = decision.attributes if isinstance(decision.attributes, dict) else {}
    return _as_text(attributes.get("recommended_option_id"))


def _default_option_label(rank: int) -> str:
    normalized = max(1, rank)
    letter = chr(ord("A") + min(normalized - 1, 25))
    return f"Option {letter}"


def _build_advisory_outcome(
    *,
    question: str,
    advice: PersonalAdvice,
    observation: Observation,
    decision: Decision,
    advisory_mode: str = "direct",
    extra_refs: list[str] | None = None,
    orchestration_metadata: dict[str, Any] | None = None,
) -> Outcome:
    refs = [observation.id]
    if decision.id:
        refs.append(decision.id)
    for ref in extra_refs or []:
        if not isinstance(ref, str):
            continue
        token = ref.strip()
        if not token:
            continue
        if token in refs:
            continue
        refs.append(token)

    outcome_attributes: dict[str, Any] = {
        "selected_action": advice.selected_action,
        "suggestion_text": advice.suggestion,
        "advisory_mode": advisory_mode,
    }
    decision_attributes = decision.attributes if isinstance(decision.attributes, dict) else {}
    for key in PERSONAL_ADVISORY_ATTRIBUTE_KEYS:
        if key not in decision_attributes:
            continue
        outcome_attributes[key] = decision_attributes[key]

    entity_changes: dict[str, Any] = {
        "latest_question": question,
        "latest_suggestion": advice.suggestion,
        "confidence": advice.confidence,
        "urgency": advice.urgency,
    }
    choice_state_patch = _choice_state_patch(orchestration_metadata)
    if choice_state_patch:
        entity_changes.update(choice_state_patch)

    return Outcome(
        id=f"out-{uuid4().hex}",
        outcome_type=PERSONAL_OUTCOME_ADVICE_RECORDED,
        status="recorded",
        decision_id=decision.id,
        changes={
            PERSONAL_ENTITY_ID: entity_changes
        },
        refs=refs,
        metadata=dict(orchestration_metadata or {}),
        attributes=outcome_attributes,
    )


def _current_personal_entity(state: WorldState) -> dict[str, Any]:
    entity = state.entities.get(PERSONAL_ENTITY_ID)
    return dict(entity) if isinstance(entity, dict) else {}


def _render_suggestion(
    *,
    selected_action: str,
    question: str,
    urgency: str,
    confidence: float,
) -> str:
    confidence_text = f"{confidence:.2f}"
    if selected_action == PERSONAL_ACTION_ASK_CLARIFY:
        return (
            "Before strong advice, clarify your top constraint for this decision "
            f"(urgency={urgency}, confidence={confidence_text})."
        )
    if selected_action == PERSONAL_ACTION_DEFER:
        return (
            "Best-effort recommendation: defer irreversible moves briefly and re-check "
            f"after one concrete signal (urgency={urgency}, confidence={confidence_text})."
        )
    if selected_action == PERSONAL_ACTION_GATHER_EVIDENCE:
        return (
            "One evidence round is recommended first; until then, use a reversible step and "
            f"treat this as uncertain guidance (urgency={urgency}, confidence={confidence_text})."
        )

    return (
        "Best-effort suggestion: take one small reversible next step on "
        f"'{question}' (urgency={urgency}, confidence={confidence_text})."
    )


def _resolve_decision_adoption_status(
    *,
    result_kind: str,
    advice: PersonalAdvice,
    choice_resolver: Callable[..., str] | None,
) -> str:
    if result_kind == RESULT_KIND_SUGGESTION:
        if choice_resolver is None:
            return ADOPTION_STATUS_PENDING
        candidate = choice_resolver(
            advice=advice,
            result_kind=result_kind,
        )
        normalized = _normalize_adoption_status(candidate)
        if normalized:
            return normalized
        return ADOPTION_STATUS_PENDING
    if result_kind == RESULT_KIND_ACTION_PROPOSAL:
        if choice_resolver is None:
            return ADOPTION_STATUS_PENDING
        candidate = choice_resolver(
            advice=advice,
            result_kind=result_kind,
        )
        normalized = _normalize_adoption_status(candidate)
        if normalized == ADOPTION_STATUS_ADOPTED:
            return ADOPTION_STATUS_ADOPTED
        return ADOPTION_STATUS_DECLINED
    return ADOPTION_STATUS_PENDING


def _resolve_session_choice(
    *,
    advice: PersonalAdvice,
    result_kind: str,
    input_stream: TextIO,
    output_stream: TextIO,
) -> str:
    if result_kind == RESULT_KIND_SUGGESTION:
        output_stream.write(f"advisor> {advice.suggestion}\n")
        if advice.selected_action == PERSONAL_ACTION_ASK_CLARIFY:
            output_stream.write("answer these clarifying questions now? [Y/n] ")
        elif advice.selected_action == PERSONAL_ACTION_GATHER_EVIDENCE:
            output_stream.write("start evidence collection now and provide manual evidence next? [Y/n] ")
        else:
            output_stream.write("adopt this suggestion now? [y/N] ")
    elif result_kind == RESULT_KIND_ACTION_PROPOSAL:
        output_stream.write(f"advisor> {advice.suggestion}\n")
        output_stream.write("confirm execution? [y/N] ")
    else:
        return ADOPTION_STATUS_PENDING

    output_stream.flush()
    raw = input_stream.readline()
    token = raw.strip().lower() if isinstance(raw, str) else ""
    if result_kind == RESULT_KIND_SUGGESTION and advice.selected_action in {
        PERSONAL_ACTION_ASK_CLARIFY,
        PERSONAL_ACTION_GATHER_EVIDENCE,
    }:
        if advice.selected_action == PERSONAL_ACTION_GATHER_EVIDENCE:
            if token in {"", "y", "yes"}:
                return ADOPTION_STATUS_PENDING
            return ADOPTION_STATUS_DECLINED
        if token in {"", "y", "yes"}:
            return ADOPTION_STATUS_ADOPTED
        return ADOPTION_STATUS_DECLINED
    if token in {"y", "yes"}:
        return ADOPTION_STATUS_ADOPTED
    return ADOPTION_STATUS_DECLINED


def _prepare_evidence_intent(
    intent: ExecutionIntent,
    *,
    decision: Decision,
    question: str,
    profile: dict[str, Any] | None,
    executor_config: PersonalExecutorConfig | None,
    available_capabilities: tuple[str, ...],
) -> None:
    attributes = decision.attributes if isinstance(decision.attributes, dict) else {}
    evidence_plan = _extract_evidence_plan_from_attributes(
        attributes,
        question=question,
    )
    plan_payload = [dict(item) for item in evidence_plan]
    search_queries = [
        _as_text(item.get("fact"))
        for item in evidence_plan
        if isinstance(item, dict) and _as_text(item.get("fact"))
    ]

    execution_brief = ensure_minimum_execution_brief(
        {
            "category": "external.evidence",
            "goal": f"Collect externally verifiable evidence for: {_as_text(question)}",
            "inputs": {
                "question": _as_text(question),
                "evidence_plan": plan_payload,
                "search_queries": search_queries,
            },
            "success_criteria": [
                {
                    "id": "evidence.collected",
                    "description": "Collected evidence includes concrete source-backed findings.",
                }
            ],
            "expected_output": {
                "summary": "evidence summary",
                "items": "list of source-backed evidence items",
            },
        },
        selected_action=decision.selected_action or PERSONAL_ACTION_GATHER_EVIDENCE,
        suggestion_text="Collect externally verifiable evidence before final recommendation.",
    )

    decision_for_route = Decision(
        id=decision.id,
        decision_type=decision.decision_type,
        status=decision.status,
        selected_action=decision.selected_action,
        refs=list(decision.refs),
        metadata=dict(decision.metadata),
        attributes=dict(attributes),
    )
    decision_for_route.attributes["execution_brief"] = execution_brief
    route_resolution = _apply_profile_to_intent_with_resolution(
        intent,
        decision=decision_for_route,
        profile=profile,
        available_capabilities=available_capabilities,
    )

    input_payload = dict(intent.input_payload) if isinstance(intent.input_payload, dict) else {}
    input_payload["question"] = _as_text(question)
    input_payload["evidence_plan"] = plan_payload
    input_payload["search_queries"] = search_queries
    input_payload["evidence_request"] = {
        "question": _as_text(question),
        "required_items": len(plan_payload),
        "mode": "external_verification",
    }
    if not isinstance(input_payload.get("execution_brief"), dict):
        input_payload["execution_brief"] = {
            "category": _as_text(execution_brief.get("category")),
            "goal": _as_text(execution_brief.get("goal")),
        }
    intent.input_payload = input_payload

    parameters = dict(intent.parameters) if isinstance(intent.parameters, dict) else {}
    parameters.setdefault("search_depth", "focused")
    parameters.setdefault("max_results", 5)
    parameters.setdefault("require_source_citations", True)
    intent.parameters = parameters

    validation = _preflight_execution_intent(
        intent,
        decision=decision_for_route,
        route_resolution=route_resolution,
        executor_config=executor_config,
        available_capabilities=available_capabilities,
    )
    if not validation.allow_execution:
        raise ValueError(_format_pei_v1_issues(validation))


def _run_confirmed_action_proposal_execution(
    runtime: SpiceRuntime,
    *,
    decision: Decision,
    profile: dict[str, Any] | None,
    executor_config: PersonalExecutorConfig | None,
    available_capabilities: tuple[str, ...],
) -> tuple[
    Outcome | None,
    ExecutionIntent | None,
    ExecutionResult | None,
    str,
    dict[str, Any] | None,
]:
    intent: ExecutionIntent | None = None
    result: ExecutionResult | None = None
    route_resolution = _IntentRouteResolution(resolved_mode="")
    validation: Any | None = None
    execution_debug: dict[str, Any] | None = None
    original_executor = runtime.executor
    try:
        intent = runtime.plan_execution(decision)
        resolved_mode = ""
        if intent is not None:
            route_resolution = _apply_profile_to_intent_with_resolution(
                intent,
                decision=decision,
                profile=profile,
                available_capabilities=available_capabilities,
            )
            resolved_mode = route_resolution.resolved_mode
            validation = _preflight_execution_intent(
                intent,
                decision=decision,
                route_resolution=route_resolution,
                executor_config=executor_config,
                available_capabilities=available_capabilities,
            )
            if not validation.allow_execution:
                blocked_status = (
                    EXECUTION_STATUS_NOT_REQUESTED
                    if validation.pending_confirmation
                    else EXECUTION_STATUS_FAILED
                )
                execution_debug = _build_execution_failure_debug(
                    route_resolution=route_resolution,
                    validation=validation,
                )
                return (
                    None,
                    intent,
                    None,
                    blocked_status,
                    execution_debug,
                )
        if (
            resolved_mode == EXECUTOR_MODE_CLI
            and executor_config is not None
        ):
            runtime.executor = build_executor(
                PersonalExecutorConfig(
                    mode=EXECUTOR_MODE_CLI,
                    timeout_seconds=executor_config.timeout_seconds,
                    cli_profile=executor_config.cli_profile,
                    cli_profile_path=executor_config.cli_profile_path,
                    cli_command=executor_config.cli_command,
                    cli_parser_mode=executor_config.cli_parser_mode,
                    sdep_command=executor_config.sdep_command,
                )
            )
        result = runtime.execute(intent)
        normalized_status = (
            _normalize_execution_status(result.status)
            if result is not None
            else EXECUTION_STATUS_FAILED
        )
        if normalized_status == EXECUTION_STATUS_FAILED:
            execution_debug = _build_execution_failure_debug(
                route_resolution=route_resolution,
                validation=validation,
                execution_result=result,
            )
        if result is not None:
            ensure_minimal_execution_result_output(
                result,
                intent=intent,
                decision=decision,
                category=route_resolution.category,
            )
        outcome = runtime.process_execution_result(
            result,
            decision=decision,
            intent=intent,
        )
    except Exception as exc:
        execution_debug = _build_execution_failure_debug(
            route_resolution=route_resolution,
            validation=validation,
            execution_result=result,
            exception=exc,
        )
        return (
            None,
            intent,
            result,
            EXECUTION_STATUS_FAILED,
            execution_debug,
        )
    finally:
        runtime.executor = original_executor
    final_status = (
        _normalize_execution_status(result.status)
        if result is not None
        else EXECUTION_STATUS_FAILED
    )
    if final_status == EXECUTION_STATUS_FAILED and execution_debug is None:
        execution_debug = _build_execution_failure_debug(
            route_resolution=route_resolution,
            validation=validation,
            execution_result=result,
        )
    return (
        outcome,
        intent,
        result,
        final_status,
        execution_debug,
    )


def _build_execution_failure_debug(
    *,
    route_resolution: _IntentRouteResolution,
    validation: Any | None,
    execution_result: ExecutionResult | None = None,
    exception: Exception | None = None,
) -> dict[str, Any]:
    details: dict[str, Any] = {}

    if validation is not None:
        details["preflight_allow_execution"] = bool(getattr(validation, "allow_execution", False))
        details["preflight_pending_confirmation"] = bool(
            getattr(validation, "pending_confirmation", False)
        )
        errors = _collect_preflight_issue_summaries(getattr(validation, "errors", []))
        if errors:
            details["preflight_errors"] = errors
        degradations = _collect_preflight_issue_summaries(getattr(validation, "degradations", []))
        if degradations:
            details["preflight_degradations"] = degradations

    if route_resolution.category:
        if isinstance(route_resolution.category_route, dict):
            details["route_enabled"] = bool(route_resolution.category_route.get("enabled"))
        else:
            details["route_enabled"] = False
    details["route_fallback_applied"] = bool(route_resolution.fallback_applied)
    resolved_mode = _as_text(route_resolution.resolved_mode)
    if resolved_mode:
        details["route_resolved_mode"] = resolved_mode

    wrapper_error = _extract_wrapper_error_debug(execution_result)
    if wrapper_error:
        details.update(wrapper_error)

    if exception is not None and "wrapper_error_message" not in details:
        exception_message = str(exception).strip()
        if exception_message:
            details["wrapper_error_message"] = exception_message
    return details


def _collect_preflight_issue_summaries(issues: Any) -> list[str]:
    if not isinstance(issues, list):
        return []
    tokens: list[str] = []
    for issue in issues[:10]:
        issue_payload = _coerce_dict(issue)
        code = _as_text(issue_payload.get("code")) or _as_text(getattr(issue, "code", ""))
        message = _as_text(issue_payload.get("message")) or _as_text(getattr(issue, "message", ""))
        if code and message:
            tokens.append(f"{code}: {message}")
        elif code:
            tokens.append(code)
        elif message:
            tokens.append(message)
    return tokens


def _extract_wrapper_error_debug(execution_result: ExecutionResult | None) -> dict[str, Any]:
    if execution_result is None:
        return {}
    details: dict[str, Any] = {}
    attributes = _coerce_dict(execution_result.attributes)
    sdep_payload = _coerce_dict(attributes.get("sdep"))
    response_payload = _coerce_dict(sdep_payload.get("response"))
    error_payload = _coerce_dict(response_payload.get("error"))
    error_details = _coerce_dict(error_payload.get("details"))

    wrapper_error_code = _as_text(error_payload.get("code"))
    if wrapper_error_code:
        details["wrapper_error_code"] = wrapper_error_code

    wrapper_error_message = _as_text(error_payload.get("message")) or _as_text(execution_result.error)
    if wrapper_error_message:
        details["wrapper_error_message"] = wrapper_error_message

    subtype = _as_text(error_details.get("subtype"))
    if subtype:
        details["wrapper_subtype"] = subtype
    stop_reason = _as_text(error_details.get("stop_reason"))
    if stop_reason:
        details["wrapper_stop_reason"] = stop_reason

    permission_denials = error_details.get("permission_denials")
    if isinstance(permission_denials, list) and permission_denials:
        normalized = _normalize_wrapper_permission_denials(permission_denials)
        if normalized:
            details["wrapper_permission_denials"] = normalized
    return details


def _normalize_wrapper_permission_denials(
    value: list[Any],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for entry in value[:5]:
        if not isinstance(entry, dict):
            continue
        item: dict[str, Any] = {}
        tool_name = _as_text(entry.get("tool_name"))
        if tool_name:
            item["tool_name"] = tool_name
        tool_use_id = _as_text(entry.get("tool_use_id"))
        if tool_use_id:
            item["tool_use_id"] = tool_use_id
        reason = _as_text(entry.get("reason"))
        if reason:
            item["reason"] = reason
        tool_input = _coerce_dict(entry.get("tool_input"))
        command = _as_text(tool_input.get("command"))
        path = _as_text(tool_input.get("path"))
        if command:
            item["command"] = command
        if path:
            item["path"] = path
        if item:
            normalized.append(item)
    return normalized


def _emit_execution_failure_debug(
    *,
    output_stream: TextIO,
    execution_debug: Any,
) -> None:
    debug = _coerce_dict(execution_debug)
    if not debug:
        return
    if "preflight_allow_execution" in debug:
        output_stream.write(
            "execution_debug.preflight_allow_execution={value}\n".format(
                value=_debug_bool(debug.get("preflight_allow_execution"))
            )
        )
    if "preflight_pending_confirmation" in debug:
        output_stream.write(
            "execution_debug.preflight_pending_confirmation={value}\n".format(
                value=_debug_bool(debug.get("preflight_pending_confirmation"))
            )
        )
    preflight_errors = debug.get("preflight_errors")
    if isinstance(preflight_errors, list) and preflight_errors:
        output_stream.write(
            "execution_debug.preflight_errors={value}\n".format(
                value=" | ".join(_as_text(item) for item in preflight_errors if _as_text(item))
            )
        )
    if "route_enabled" in debug:
        output_stream.write(
            "execution_debug.route_enabled={value}\n".format(
                value=_debug_bool(debug.get("route_enabled"))
            )
        )
    if "route_fallback_applied" in debug:
        output_stream.write(
            "execution_debug.route_fallback_applied={value}\n".format(
                value=_debug_bool(debug.get("route_fallback_applied"))
            )
        )
    route_resolved_mode = _as_text(debug.get("route_resolved_mode"))
    if route_resolved_mode:
        output_stream.write(f"execution_debug.route_resolved_mode={route_resolved_mode}\n")
    wrapper_error_code = _as_text(debug.get("wrapper_error_code"))
    if wrapper_error_code:
        output_stream.write(f"execution_debug.wrapper_error_code={wrapper_error_code}\n")
    wrapper_error_message = _as_text(debug.get("wrapper_error_message"))
    if wrapper_error_message:
        output_stream.write(f"execution_debug.wrapper_error_message={wrapper_error_message}\n")
    wrapper_subtype = _as_text(debug.get("wrapper_subtype"))
    if wrapper_subtype:
        output_stream.write(f"execution_debug.wrapper_subtype={wrapper_subtype}\n")
    wrapper_stop_reason = _as_text(debug.get("wrapper_stop_reason"))
    if wrapper_stop_reason:
        output_stream.write(f"execution_debug.wrapper_stop_reason={wrapper_stop_reason}\n")
    wrapper_permission_denials = debug.get("wrapper_permission_denials")
    if isinstance(wrapper_permission_denials, list) and wrapper_permission_denials:
        output_stream.write(
            "execution_debug.wrapper_permission_denials={value}\n".format(
                value=json.dumps(wrapper_permission_denials, ensure_ascii=True, separators=(",", ":"))
            )
        )


def _debug_bool(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    token = _as_text(value).lower()
    if token in {"true", "1", "yes"}:
        return "true"
    if token in {"false", "0", "no"}:
        return "false"
    return "unknown"


def _apply_profile_to_intent(
    intent: ExecutionIntent,
    *,
    decision: Decision,
    profile: dict[str, Any] | None,
    available_capabilities: tuple[str, ...],
) -> str:
    resolution = _apply_profile_to_intent_with_resolution(
        intent,
        decision=decision,
        profile=profile,
        available_capabilities=available_capabilities,
    )
    return resolution.resolved_mode


def _apply_profile_to_intent_with_resolution(
    intent: ExecutionIntent,
    *,
    decision: Decision,
    profile: dict[str, Any] | None,
    available_capabilities: tuple[str, ...],
) -> _IntentRouteResolution:
    brief = _decision_execution_brief(decision)
    category = _as_text(brief.get("category"))
    profile_mode = _as_text(_coerce_dict(profile).get("executor_mode")).lower()

    if not isinstance(profile, dict):
        _apply_execution_brief_contract_defaults(
            intent,
            brief=brief,
            category=category,
        )
        _apply_decision_provenance(intent, decision)
        return _IntentRouteResolution(
            resolved_mode="",
            profile_mode=profile_mode,
            category=category,
            category_route=None,
        )

    routes = profile.get("category_routes")
    if not isinstance(routes, dict):
        _apply_execution_brief_contract_defaults(
            intent,
            brief=brief,
            category=category,
        )
        _apply_decision_provenance(intent, decision)
        return _IntentRouteResolution(
            resolved_mode="",
            profile_mode=profile_mode,
            category=category,
            category_route=None,
        )

    if not category:
        _apply_decision_provenance(intent, decision)
        return _IntentRouteResolution(
            resolved_mode="",
            profile_mode=profile_mode,
            category=category,
            category_route=None,
        )

    route = routes.get(category)
    if not isinstance(route, dict):
        _apply_execution_brief_contract_defaults(
            intent,
            brief=brief,
            category=category,
        )
        _apply_decision_provenance(intent, decision)
        return _IntentRouteResolution(
            resolved_mode=profile_mode,
            profile_mode=profile_mode,
            category=category,
            category_route=None,
        )

    resolved_route = route
    resolved_mode = profile_mode
    fallback_applied = False
    fallback_cli = route.get("fallback_cli") if isinstance(route.get("fallback_cli"), dict) else None
    has_fallback = bool(_as_text(_coerce_dict(fallback_cli).get("operation_name")))
    expected_capabilities = _route_expected_capabilities(route)
    if profile_mode == EXECUTOR_MODE_SDEP and expected_capabilities:
        if available_capabilities:
            available = set(available_capabilities)
            if not available.intersection(expected_capabilities) and has_fallback:
                resolved_route = _coerce_dict(fallback_cli)
                resolved_mode = EXECUTOR_MODE_CLI
                fallback_applied = True
        elif has_fallback:
            # SDEP unavailable or capability list not available:
            # fallback to CLI only when explicitly configured.
            resolved_route = _coerce_dict(fallback_cli)
            resolved_mode = EXECUTOR_MODE_CLI
            fallback_applied = True

    operation = dict(intent.operation) if isinstance(intent.operation, dict) else {}
    route_operation = _as_text(resolved_route.get("operation_name"))
    if route_operation:
        operation["name"] = route_operation
    guardrails = resolved_route.get("guardrails")
    if isinstance(guardrails, dict):
        if bool(guardrails.get("force_dry_run")):
            operation["dry_run"] = True
    intent.operation = operation

    route_target = resolved_route.get("target")
    if isinstance(route_target, dict):
        target = dict(intent.target) if isinstance(intent.target, dict) else {}
        target.update(dict(route_target))
        intent.target = target

    input_defaults = resolved_route.get("input_defaults")
    parameter_defaults = resolved_route.get("parameter_defaults")

    input_payload = dict(intent.input_payload) if isinstance(intent.input_payload, dict) else {}
    if isinstance(input_defaults, dict):
        input_payload.update(dict(input_defaults))
    brief_payload = {
        "schema_version": EXECUTION_INTENT_V1_SCHEMA_VERSION,
        "category": _as_text(brief.get("category")),
        "goal": _as_text(brief.get("goal")),
    }
    support_level = CATEGORY_SUPPORT_LEVEL_MAP.get(category)
    if support_level:
        brief_payload["support_level"] = support_level
    brief_inputs = brief.get("inputs")
    if isinstance(brief_inputs, dict):
        brief_payload["inputs"] = dict(brief_inputs)
    expected_output = brief.get("expected_output")
    if isinstance(expected_output, dict):
        brief_payload["expected_output"] = dict(expected_output)
    input_payload["execution_brief"] = brief_payload
    if category == CATEGORY_EXTERNAL_SYSTEM:
        _ensure_system_contract_inputs(input_payload, brief)
    intent.input_payload = input_payload

    parameters = dict(intent.parameters) if isinstance(intent.parameters, dict) else {}
    if isinstance(parameter_defaults, dict):
        parameters.update(dict(parameter_defaults))
    for key in ("risk_level", "dry_run_preferred", "timeout_seconds", "idempotency_hint"):
        if key in brief:
            parameters[key] = brief[key]
    if support_level and not _as_text(parameters.get("support_level")):
        parameters["support_level"] = support_level
    if isinstance(guardrails, dict) and "max_timeout_seconds" in guardrails:
        parameters["max_timeout_seconds"] = guardrails["max_timeout_seconds"]
    intent.parameters = parameters

    constraints = [entry for entry in intent.constraints if isinstance(entry, dict)]
    brief_constraints = brief.get("constraints")
    if isinstance(brief_constraints, list):
        constraints.extend(entry for entry in brief_constraints if isinstance(entry, dict))
    if isinstance(guardrails, dict) and "require_confirmation" in guardrails:
        constraints.append(
            {
                "name": "profile.require_confirmation",
                "kind": "approval_gate",
                "params": {"required": bool(guardrails.get("require_confirmation"))},
            }
        )
    intent.constraints = constraints

    success_criteria = [entry for entry in intent.success_criteria if isinstance(entry, dict)]
    brief_success = brief.get("success_criteria")
    if isinstance(brief_success, list):
        success_criteria.extend(entry for entry in brief_success if isinstance(entry, dict))
    intent.success_criteria = success_criteria

    _apply_decision_provenance(intent, decision)
    return _IntentRouteResolution(
        resolved_mode=resolved_mode,
        profile_mode=profile_mode,
        category=category,
        category_route=dict(route),
        fallback_route=dict(fallback_cli) if isinstance(fallback_cli, dict) else None,
        fallback_applied=fallback_applied,
    )


def _preflight_execution_intent(
    intent: ExecutionIntent,
    *,
    decision: Decision,
    route_resolution: _IntentRouteResolution,
    executor_config: PersonalExecutorConfig | None,
    available_capabilities: tuple[str, ...],
):
    route_context = _build_route_context_for_preflight(
        route_resolution=route_resolution,
        executor_config=executor_config,
        available_capabilities=available_capabilities,
    )
    return preflight_execution_intent_v1(
        intent,
        decision=decision,
        route_context=route_context,
    )


def _build_route_context_for_preflight(
    *,
    route_resolution: _IntentRouteResolution,
    executor_config: PersonalExecutorConfig | None,
    available_capabilities: tuple[str, ...],
) -> PEIV1RouteContext | None:
    if not route_resolution.category:
        return None
    if route_resolution.category_route is None and not route_resolution.profile_mode:
        return None
    fallback_available = True
    if isinstance(route_resolution.fallback_route, dict):
        fallback_available = _can_build_cli_fallback(executor_config)
    return PEIV1RouteContext(
        category_route=route_resolution.category_route,
        fallback_route=route_resolution.fallback_route,
        profile_mode=route_resolution.profile_mode,
        available_capabilities=available_capabilities,
        fallback_applied=route_resolution.fallback_applied,
        fallback_available=fallback_available,
    )


def _can_build_cli_fallback(executor_config: PersonalExecutorConfig | None) -> bool:
    if executor_config is None:
        return False
    try:
        build_executor(
            PersonalExecutorConfig(
                mode=EXECUTOR_MODE_CLI,
                timeout_seconds=executor_config.timeout_seconds,
                cli_profile=executor_config.cli_profile,
                cli_profile_path=executor_config.cli_profile_path,
                cli_command=executor_config.cli_command,
                cli_parser_mode=executor_config.cli_parser_mode,
                sdep_command=executor_config.sdep_command,
            )
        )
    except Exception:
        return False
    return True


def _format_pei_v1_issues(validation: Any) -> str:
    issue_parts: list[str] = []
    for issue in getattr(validation, "errors", []):
        code = _as_text(getattr(issue, "code", ""))
        message = _as_text(getattr(issue, "message", ""))
        if code and message:
            issue_parts.append(f"{code}: {message}")
        elif code:
            issue_parts.append(code)
    for issue in getattr(validation, "degradations", []):
        code = _as_text(getattr(issue, "code", ""))
        message = _as_text(getattr(issue, "message", ""))
        if code and message:
            issue_parts.append(f"{code}: {message}")
        elif code:
            issue_parts.append(code)
    if not issue_parts:
        return "execution_intent_v1.preflight_failed"
    return "; ".join(issue_parts)


def _apply_decision_provenance(intent: ExecutionIntent, decision: Decision) -> None:
    provenance = dict(intent.provenance) if isinstance(intent.provenance, dict) else {}
    if not _as_text(provenance.get("decision_id")):
        provenance["decision_id"] = _as_text(decision.id)
    if not _as_text(provenance.get("selected_action")):
        provenance["selected_action"] = _as_text(decision.selected_action)
    provenance["source_domain"] = EXECUTION_INTENT_V1_SOURCE_DOMAIN
    if not _as_text(provenance.get("source_turn_id")):
        metadata = decision.metadata if isinstance(decision.metadata, dict) else {}
        source_turn_id = _as_text(metadata.get("source_turn_id"))
        provenance["source_turn_id"] = source_turn_id or _as_text(decision.id)
    intent.provenance = provenance


def _apply_execution_brief_contract_defaults(
    intent: ExecutionIntent,
    *,
    brief: dict[str, Any],
    category: str,
) -> None:
    if not category:
        return

    input_payload = dict(intent.input_payload) if isinstance(intent.input_payload, dict) else {}
    existing_brief = (
        dict(input_payload.get("execution_brief"))
        if isinstance(input_payload.get("execution_brief"), dict)
        else {}
    )
    brief_payload = {
        "schema_version": EXECUTION_INTENT_V1_SCHEMA_VERSION,
        "category": _as_text(existing_brief.get("category")) or category,
        "goal": _as_text(existing_brief.get("goal")) or _as_text(brief.get("goal")),
    }
    support_level = CATEGORY_SUPPORT_LEVEL_MAP.get(category)
    if support_level:
        brief_payload["support_level"] = support_level
    brief_inputs = brief.get("inputs")
    if isinstance(brief_inputs, dict):
        brief_payload["inputs"] = dict(brief_inputs)
    if isinstance(existing_brief.get("inputs"), dict):
        merged_inputs = dict(existing_brief.get("inputs"))
        merged_inputs.update(_coerce_dict(brief_payload.get("inputs")))
        brief_payload["inputs"] = merged_inputs
    input_payload["execution_brief"] = brief_payload
    if category == CATEGORY_EXTERNAL_SYSTEM:
        _ensure_system_contract_inputs(input_payload, brief)
    intent.input_payload = input_payload

    parameters = dict(intent.parameters) if isinstance(intent.parameters, dict) else {}
    if support_level and not _as_text(parameters.get("support_level")):
        parameters["support_level"] = support_level
    intent.parameters = parameters

    success_criteria = [entry for entry in intent.success_criteria if isinstance(entry, dict)]
    brief_success = brief.get("success_criteria")
    if isinstance(brief_success, list):
        success_criteria.extend(
            entry
            for entry in brief_success
            if isinstance(entry, dict)
            and _as_text(entry.get("id"))
            and _as_text(entry.get("description"))
        )
    if success_criteria:
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for entry in success_criteria:
            key = f"{_as_text(entry.get('id'))}:{_as_text(entry.get('description'))}"
            if key in seen:
                continue
            seen.add(key)
            deduped.append(dict(entry))
        intent.success_criteria = deduped

    operation = dict(intent.operation) if isinstance(intent.operation, dict) else {}
    canonical_operation = CATEGORY_CANONICAL_OPERATION_MAP.get(category)
    if canonical_operation:
        operation["name"] = canonical_operation
    if not _as_text(operation.get("mode")):
        operation["mode"] = "sync"
    if support_level == "limited":
        operation["dry_run"] = True
    intent.operation = operation

    target = dict(intent.target) if isinstance(intent.target, dict) else {}
    if not _as_text(target.get("kind")):
        target["kind"] = "external.service"
    if not _as_text(target.get("id")):
        target["id"] = _default_target_id_for_category(category)
    intent.target = target


def _ensure_system_contract_inputs(
    input_payload: dict[str, Any],
    brief: dict[str, Any],
) -> None:
    brief_inputs = brief.get("inputs") if isinstance(brief.get("inputs"), dict) else {}
    task = _as_text(input_payload.get("task"))
    if not task:
        task = (
            _as_text(_coerce_dict(brief_inputs).get("task"))
            or _as_text(_coerce_dict(brief_inputs).get("command"))
            or _as_text(input_payload.get("command"))
            or _as_text(brief.get("goal"))
        )
    if task:
        input_payload["task"] = task

    scope = input_payload.get("scope")
    if not _has_non_empty_scope(scope):
        brief_scope = _coerce_dict(brief_inputs).get("scope")
        if _has_non_empty_scope(brief_scope):
            input_payload["scope"] = brief_scope
        else:
            input_payload["scope"] = "workspace"


def _has_non_empty_scope(value: Any) -> bool:
    if isinstance(value, dict):
        for key in ("id", "kind", "value", "path"):
            if _as_text(value.get(key)):
                return True
        return bool(value)
    return bool(_as_text(value))


def _default_target_id_for_category(category: str) -> str:
    if category == "external.evidence":
        return "research"
    if category == "external.system":
        return "system"
    if category == "external.communicate":
        return "communication"
    if category == "external.schedule":
        return "calendar"
    if category == "external.manage_task":
        return "task_manager"
    return "system"


def _route_expected_capabilities(route: dict[str, Any]) -> set[str]:
    required = route.get("required_capabilities")
    if isinstance(required, list):
        normalized = {_as_text(item) for item in required if _as_text(item)}
        if normalized:
            return normalized
    operation_name = _as_text(route.get("operation_name"))
    return {operation_name} if operation_name else set()


def _normalize_execution_status(value: Any) -> str:
    token = _as_text(value).lower()
    if token == EXECUTION_STATUS_SUCCESS:
        return EXECUTION_STATUS_SUCCESS
    if token == EXECUTION_STATUS_FAILED:
        return EXECUTION_STATUS_FAILED
    return EXECUTION_STATUS_FAILED


def _choice_state_patch(orchestration_metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(orchestration_metadata, dict):
        return {}

    result_kind = _orchestration_value(
        orchestration_metadata,
        "result_kind",
        RESULT_KIND_SUGGESTION,
    )
    decision_adoption_status = _orchestration_value(
        orchestration_metadata,
        "decision_adoption_status",
        ADOPTION_STATUS_PENDING,
    )
    evidence_state = _orchestration_value(
        orchestration_metadata,
        "evidence_state",
        EVIDENCE_STATE_NOT_REQUESTED,
    )
    feedback_payload = {
        "latest_result_kind": result_kind,
        "latest_decision_adoption_status": decision_adoption_status,
        "latest_evidence_state": evidence_state,
    }
    return {
        # Existing field in personal state schema; keeps state semantically aligned
        # without introducing new observation types or DomainSpec changes.
        "last_feedback": json.dumps(
            feedback_payload,
            ensure_ascii=True,
            sort_keys=True,
        ),
    }


def _ensure_action_proposal_execution_brief(
    decision: Decision,
    *,
    advice: PersonalAdvice,
) -> None:
    attributes = decision.attributes if isinstance(decision.attributes, dict) else {}
    execution_brief = ensure_minimum_execution_brief(
        attributes.get("execution_brief"),
        selected_action=decision.selected_action or "",
        suggestion_text=advice.suggestion,
    )
    if not isinstance(decision.attributes, dict):
        decision.attributes = {}
    decision.attributes["execution_brief"] = execution_brief


def _decision_execution_brief(decision: Decision) -> dict[str, Any]:
    attributes = decision.attributes if isinstance(decision.attributes, dict) else {}
    payload = attributes.get("execution_brief")
    return dict(payload) if isinstance(payload, dict) else {}


def _decision_result_kind(decision: Decision) -> str:
    attributes = decision.attributes if isinstance(decision.attributes, dict) else {}
    token = _as_text(attributes.get("result_kind")).lower()
    if token == RESULT_KIND_ACTION_PROPOSAL:
        return RESULT_KIND_ACTION_PROPOSAL
    return RESULT_KIND_SUGGESTION


def _normalize_adoption_status(value: Any) -> str:
    token = _as_text(value).lower()
    if token == ADOPTION_STATUS_ADOPTED:
        return ADOPTION_STATUS_ADOPTED
    if token == ADOPTION_STATUS_DECLINED:
        return ADOPTION_STATUS_DECLINED
    if token == ADOPTION_STATUS_PENDING:
        return ADOPTION_STATUS_PENDING
    return ""


def _attach_orchestration_metadata(
    record: Outcome | Reflection,
    metadata: dict[str, Any],
) -> None:
    if not metadata:
        return
    record.metadata.update(dict(metadata))


def _orchestration_value(
    metadata: dict[str, Any] | None,
    key: str,
    default: str,
) -> str:
    if not isinstance(metadata, dict):
        return default
    value = metadata.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _resolve_executor_config_with_profile_mode(
    executor_config: PersonalExecutorConfig,
    profile: dict[str, Any],
) -> PersonalExecutorConfig:
    base_mode = (executor_config.mode or EXECUTOR_MODE_MOCK).strip().lower() or EXECUTOR_MODE_MOCK
    if base_mode not in {EXECUTOR_MODE_MOCK, EXECUTOR_MODE_CLI, EXECUTOR_MODE_SDEP}:
        base_mode = EXECUTOR_MODE_MOCK

    profile_mode = _as_text(profile.get("executor_mode")).lower()
    if profile_mode in {EXECUTOR_MODE_CLI, EXECUTOR_MODE_SDEP}:
        mode = profile_mode
    elif profile_mode == EXECUTOR_MODE_MOCK:
        # `mock` is the packaged default profile mode.
        # Keep user connection defaults when provided via CLI/env/personal.config.
        mode = base_mode
    else:
        mode = base_mode
    return PersonalExecutorConfig(
        mode=mode,
        timeout_seconds=executor_config.timeout_seconds,
        cli_profile=executor_config.cli_profile,
        cli_profile_path=executor_config.cli_profile_path,
        cli_command=executor_config.cli_command,
        cli_parser_mode=executor_config.cli_parser_mode,
        sdep_command=executor_config.sdep_command,
    )


def _load_profile_context(
    workspace: Path,
    *,
    executor_config: PersonalExecutorConfig,
    validate_if_changed: bool,
) -> PersonalProfileContext:
    profile_path = ensure_workspace_profile(workspace, force=False)
    profile = load_profile(profile_path)
    resolved_config = _resolve_executor_config_with_profile_mode(executor_config, profile)

    available_capabilities: tuple[str, ...] = ()
    if validate_if_changed and _profile_requires_validation(
        workspace,
        profile_path,
        executor_mode=resolved_config.mode,
    ):
        validation = validate_profile_contract(
            profile,
            profile_path=profile_path,
            executor_config=resolved_config,
        )
        validation.raise_for_errors()
        _write_profile_validation_report(workspace, validation)
        available_capabilities = tuple(validation.available_capabilities)
    else:
        report = _read_profile_validation_report(workspace)
        if isinstance(report, dict):
            caps = report.get("available_capabilities")
            if isinstance(caps, list):
                available_capabilities = tuple(str(item) for item in caps if str(item).strip())

    return PersonalProfileContext(
        path=profile_path,
        profile=profile,
        available_capabilities=available_capabilities,
    )


def _profile_requires_validation(
    workspace: Path,
    profile_path: Path,
    *,
    executor_mode: str,
) -> bool:
    report = _read_profile_validation_report(workspace)
    if not isinstance(report, dict):
        return True
    previous = _as_text(report.get("fingerprint"))
    if not previous:
        return True
    if previous != profile_fingerprint(profile_path):
        return True
    previous_mode = _as_text(report.get("executor_mode")).lower()
    return previous_mode != _as_text(executor_mode).lower()


def _profile_validation_report_path(workspace: Path) -> Path:
    return workspace / PERSONAL_PROFILE_VALIDATION_RELATIVE_PATH


def _connection_resolution_report_path(workspace: Path) -> Path:
    return workspace / PERSONAL_CONNECTION_RESOLUTION_RELATIVE_PATH


def _write_connection_resolution_report(
    workspace: Path,
    *,
    model_command: str | None,
    executor_config: PersonalExecutorConfig,
    workspace_config: Any,
) -> None:
    report_path = _connection_resolution_report_path(workspace)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "spice_personal.connection_resolution.v1",
        "workspace": str(workspace),
        "model": {
            "command": _as_text(model_command),
            "source": _as_text(getattr(workspace_config, "model_command_source", "")),
            "provider": _as_text(getattr(workspace_config, "model_provider", "")),
            "model": _as_text(getattr(workspace_config, "model_name", "")),
            "api_key_env": _as_text(getattr(workspace_config, "model_api_key_env", "")),
        },
        "executor": {
            "mode": _as_text(executor_config.mode),
            "mode_source": _as_text(getattr(workspace_config, "executor_mode_source", "")),
            "cli_command": _as_text(executor_config.cli_command),
            "cli_command_source": _as_text(getattr(workspace_config, "cli_command_source", "")),
            "sdep_command": _as_text(executor_config.sdep_command),
            "sdep_command_source": _as_text(getattr(workspace_config, "sdep_command_source", "")),
            "provider": _as_text(getattr(workspace_config, "agent_provider", "")),
            "agent_mode": _as_text(getattr(workspace_config, "agent_mode", "")),
            "auth_env": _as_text(getattr(workspace_config, "agent_auth_env", "")),
            "endpoint": _as_text(getattr(workspace_config, "agent_endpoint", "")),
        },
        "config_schema_version": _as_text(getattr(workspace_config, "schema_version", "")),
    }
    report_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_profile_validation_report(
    workspace: Path,
    validation: ProfileValidationResult,
) -> None:
    report_path = _profile_validation_report_path(workspace)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(validation.to_dict(), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_profile_validation_report(workspace: Path) -> dict[str, Any] | None:
    report_path = _profile_validation_report_path(workspace)
    if not report_path.exists():
        return None
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, dict):
        return dict(payload)
    return None


def validate_personal_context_inputs(
    *,
    context_text: str | None,
    context_file: str | Path | None,
) -> None:
    has_text = bool(_as_text(context_text))
    has_file = context_file is not None and _as_text(str(context_file))

    if has_text:
        text = _as_text(context_text)
        if len(text) > MAX_CONTEXT_TEXT_CHARS:
            raise ValueError(
                f"context_text exceeds max length ({MAX_CONTEXT_TEXT_CHARS} chars)."
            )

    if has_file:
        path = Path(str(context_file))
        _validate_context_file_path(path)


def _build_context_ingest_payloads(
    *,
    context_text: str | None,
    context_file: str | Path | None,
) -> list[dict[str, Any]]:
    has_text = bool(_as_text(context_text))
    has_file = context_file is not None and _as_text(str(context_file))
    if not has_text and not has_file:
        return []

    validate_personal_context_inputs(
        context_text=context_text,
        context_file=context_file,
    )

    payloads: list[dict[str, Any]] = []
    if has_text:
        payloads.append(
            _build_context_text_ingest_payload(_as_text(context_text))
        )

    if has_file:
        payloads.append(
            _build_context_file_ingest_payload(Path(str(context_file)))
        )
    return payloads


def _build_onboarding_decision_card(*, question: str, workspace: Path) -> str:
    config_path = workspace / PERSONAL_CONFIG_FILENAME
    quoted_question = question.replace('"', '\\"')
    lines = [
        "Decision Card",
        "",
        "Mode:",
        "Setup required (no model configured)",
        "",
        "Question:",
        question,
        "",
        "Best next step:",
        "-> Connect a model to enable real decision-making.",
        "",
        "Why:",
        "-> SPICE Personal requires a connected model command for real decision recommendations.",
        "",
        "Confidence:",
        "-> N/A",
        "",
        "Control:",
        "-> Nothing has been executed.",
        "",
        "Next:",
        f"-> Edit {config_path}",
        f"-> Then run: spice-personal ask \"{quoted_question}\"",
        "",
        "Optional:",
        "-> Connect an agent later.",
    ]
    return "\n".join(lines)


def _build_session_setup_required_message(*, workspace: Path) -> str:
    config_path = workspace / PERSONAL_CONFIG_FILENAME
    lines = [
        "SPICE personal advisor mode requires a configured model.",
        "",
        "Setup required (no model configured).",
        f"Edit {config_path}",
        "Then run: spice-personal ask \"<question>\"",
    ]
    return "\n".join(lines) + "\n"


def _build_context_text_ingest_payload(content_full: str) -> dict[str, Any]:
    content, truncated = _truncate_text(content_full, max_chars=MAX_CONTEXT_CONTENT_CHARS)
    payload: dict[str, Any] = {
        "source_type": "context_text",
        "content": content,
        "content_length": len(content_full),
    }
    if truncated:
        payload["content_truncated"] = True
    return payload


def _build_context_file_ingest_payload(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    raw = resolved.read_bytes()
    try:
        content_full = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"context_file is not valid UTF-8 text: {resolved}") from exc
    content, truncated = _truncate_text(content_full, max_chars=MAX_CONTEXT_CONTENT_CHARS)
    payload = {
        "source_type": "context_file",
        "source_path": str(resolved),
        "content": content,
        "content_length": len(content_full),
    }
    if truncated:
        payload["content_truncated"] = True
    return payload


def _normalize_context_ingest_attributes(payload: dict[str, Any]) -> dict[str, Any]:
    attributes = dict(payload)
    # Context ingest observations stay per-source and traceable, but state-facing preview
    # is written once via question_received to avoid lossy last-write-wins projection.
    attributes.pop("evidence_summary", None)
    return attributes


def _build_context_state_preview(
    context_ingests: list[dict[str, Any]] | None,
) -> str:
    if not isinstance(context_ingests, list):
        return ""

    blocks: list[str] = []
    for source_type in ("context_text", "context_file"):
        for payload in context_ingests:
            if not isinstance(payload, dict):
                continue
            if _as_text(payload.get("source_type")).lower() != source_type:
                continue
            content = _as_text(payload.get("content"))
            if not content:
                continue
            if source_type == "context_text":
                header = "[context_text]"
            else:
                source_path = _as_text(payload.get("source_path"))
                header = (
                    f"[context_file: {source_path}]"
                    if source_path
                    else "[context_file]"
                )
            blocks.append(f"{header}\n{content}")

    if not blocks:
        return ""

    preview_full = "\n\n".join(blocks)
    preview, _ = _truncate_text(preview_full, max_chars=MAX_CONTEXT_PREVIEW_CHARS)
    return preview


def _validate_context_file_path(path: Path) -> None:
    raw = str(path)
    if any(token in raw for token in ("*", "?", "[", "]")):
        raise ValueError(f"context_file does not support glob patterns: {path}")
    if any(part == ".." for part in path.parts):
        raise ValueError(f"context_file does not allow directory traversal segments: {path}")
    if not path.exists():
        raise ValueError(f"context_file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"context_file must be a file: {path}")
    extension = path.suffix.lower()
    if extension not in ALLOWED_CONTEXT_FILE_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_CONTEXT_FILE_EXTENSIONS))
        raise ValueError(
            f"context_file has unsupported file type {extension!r}; allowed: {allowed}"
        )
    size_bytes = path.stat().st_size
    if size_bytes > MAX_CONTEXT_FILE_BYTES:
        raise ValueError(
            f"context_file is too large ({size_bytes} bytes); max {MAX_CONTEXT_FILE_BYTES} bytes."
        )
    try:
        path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"context_file is not valid UTF-8 text: {path}") from exc


def _truncate_text(text: str, *, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0:
        return "", bool(text)
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return cleaned, False
    if max_chars <= 3:
        return cleaned[:max_chars], True
    return cleaned[: max_chars - 3].rstrip() + "...", True


def _load_personal_state(workspace: Path) -> WorldState | None:
    state_path = workspace / PERSONAL_STATE_RELATIVE_PATH
    if not state_path.exists():
        return None

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid personal state payload: {state_path}")
    return _world_state_from_payload(payload)


def _save_personal_state(workspace: Path, state: WorldState) -> None:
    state_path = workspace / PERSONAL_STATE_RELATIVE_PATH
    state_path.parent.mkdir(parents=True, exist_ok=True)

    payload = asdict(state)
    payload["timestamp"] = state.timestamp.isoformat()
    state_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True, default=_json_default)
        + "\n",
        encoding="utf-8",
    )


def _world_state_from_payload(payload: dict[str, Any]) -> WorldState:
    state = WorldState(
        id=str(payload.get("id") or f"worldstate-{uuid4().hex}"),
        refs=_coerce_str_list(payload.get("refs")),
        metadata=_coerce_dict(payload.get("metadata")),
        schema_version=str(payload.get("schema_version", "0.1")),
        status=str(payload.get("status", "current")),
        entities=_coerce_dict(payload.get("entities")),
        relations=_coerce_list_of_dict(payload.get("relations")),
        goals=_coerce_list_of_dict(payload.get("goals")),
        constraints=_coerce_list_of_dict(payload.get("constraints")),
        resources=_coerce_dict(payload.get("resources")),
        risks=_coerce_list_of_dict(payload.get("risks")),
        signals=_coerce_list_of_dict(payload.get("signals")),
        active_intents=_coerce_list_of_dict(payload.get("active_intents")),
        recent_outcomes=_coerce_list_of_dict(payload.get("recent_outcomes")),
        confidence=_coerce_dict(payload.get("confidence")),
        provenance=_coerce_dict(payload.get("provenance")),
        domain_state=_coerce_dict(payload.get("domain_state")),
    )
    parsed_timestamp = _parse_timestamp(payload.get("timestamp"))
    if parsed_timestamp is not None:
        state.timestamp = parsed_timestamp
    return state


def _coerce_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _coerce_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _coerce_list_of_dict(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _parse_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized)
        except ValueError:
            return None
    return None


def _json_default(value: Any) -> Any:
    iso = getattr(value, "isoformat", None)
    if callable(iso):
        return iso()
    return str(value)


def _sanitize_user_question_input(value: Any) -> str:
    token = _as_text(value)
    if not token:
        return ""

    quote_pairs = (
        ('"', '"'),
        ("'", "'"),
        ("“", "”"),
        ("‘", "’"),
    )
    for _ in range(2):
        removed_pair = False
        for left, right in quote_pairs:
            if token.startswith(left) and token.endswith(right) and len(token) > 1:
                inner = token[1:-1].strip()
                if inner:
                    token = inner
                    removed_pair = True
                    break
        if not removed_pair:
            break

    while token and token[0] in "\"'“”‘’":
        token = token[1:].lstrip()
    while token and token[-1] in "\"'“”‘’":
        token = token[:-1].rstrip()
    return token.strip()


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
