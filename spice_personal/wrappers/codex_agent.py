from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from spice.executors.sdep import build_error_response
from spice.protocols import (
    SDEPActionCapability,
    SDEPAgentDescription,
    SDEPDescribeRequest,
    SDEPDescribeResponse,
    SDEPEndpointIdentity,
    SDEPError,
    SDEPExecutionOutcome,
    SDEPExecuteRequest,
    SDEPExecuteResponse,
    SDEPProtocolSupport,
)
from spice.protocols.sdep import (
    SDEP_AGENT_DESCRIBE_REQUEST,
    SDEP_EXECUTE_REQUEST,
    SDEP_ROLE_EXECUTOR,
    SDEP_VERSION,
)
from spice_personal.wrappers.capability_policy import (
    SUPPORT_LEVEL_LIMITED,
    capability_support_level,
)


CODEX_PROVIDER_ID = "codex"
DEFAULT_AUTH_ENV = "OPENAI_API_KEY"
DEFAULT_CODEX_COMMAND = "codex"
CODEX_COMMAND_ENV = "SPICE_AGENT_CODEX_COMMAND"
WORKSPACE_ENV = "SPICE_AGENT_CODEX_WORKSPACE"
MODEL_ENV = "SPICE_AGENT_CODEX_MODEL"
DEFAULT_TIMEOUT_SECONDS = 180.0
DEFAULT_MAX_OUTPUT_CHARS = 12000

SUPPORTED_ACTIONS = (
    "personal.gather_evidence",
    "personal.system",
    "personal.communicate",
    "personal.schedule",
)
FULL_ACTIONS = frozenset(
    {
        "personal.gather_evidence",
        "personal.system",
    }
)
LIMITED_ACTIONS = frozenset(
    {
        "personal.communicate",
        "personal.schedule",
    }
)


@dataclass(slots=True, frozen=True)
class CodexWrapperConfig:
    auth_env: str
    codex_command: str
    workspace: Path
    endpoint: str
    model: str
    timeout_seconds: float
    sandbox: str
    max_output_chars: int


@dataclass(slots=True, frozen=True)
class CodexInvocationResult:
    status: str
    summary: str
    evidence: list[dict[str, Any]]
    actions: list[dict[str, Any]]
    artifacts: list[dict[str, Any]]
    errors: list[str]
    raw_payload: dict[str, Any]


@dataclass(slots=True)
class CodexWrapperError(Exception):
    code: str
    message: str
    category: str
    details: dict[str, Any]
    retryable: bool = False

    def __str__(self) -> str:
        return self.message


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spice-agent-codex",
        description="Thin Codex-backed SDEP wrapper for SPICE personal execution.",
    )
    parser.add_argument("--auth-env", type=str, default=DEFAULT_AUTH_ENV)
    parser.add_argument("--codex-command", type=str, default="")
    parser.add_argument("--workspace", type=str, default="")
    parser.add_argument("--endpoint", type=str, default="")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--sandbox",
        type=str,
        choices=("read-only", "workspace-write", "danger-full-access"),
        default="workspace-write",
    )
    parser.add_argument("--max-output-chars", type=int, default=DEFAULT_MAX_OUTPUT_CHARS)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        config = _build_config(args)
        responder = SDEPEndpointIdentity(
            id=f"agent.{CODEX_PROVIDER_ID}",
            name="Codex Agent Wrapper",
            version="0.1",
            vendor="SpicePersonal",
            implementation="spice_personal.wrappers.codex_agent",
            role=SDEP_ROLE_EXECUTOR,
        )
        raw = os.sys.stdin.read()
        payload = _route_request(raw=raw, responder=responder, config=config)
        _write(payload)
        return 0
    except Exception as exc:  # hard wrapper startup/crash path
        _write_stderr(f"codex wrapper startup failure: {exc}\n")
        return 1


def _build_config(args: argparse.Namespace) -> CodexWrapperConfig:
    auth_env = _as_text(getattr(args, "auth_env", "")) or DEFAULT_AUTH_ENV
    codex_command = (
        _as_text(getattr(args, "codex_command", ""))
        or _as_text(os.environ.get(CODEX_COMMAND_ENV))
        or DEFAULT_CODEX_COMMAND
    )
    workspace_token = _as_text(getattr(args, "workspace", "")) or _as_text(
        os.environ.get(WORKSPACE_ENV)
    )
    workspace = Path(workspace_token) if workspace_token else Path.cwd()
    workspace = workspace.resolve()
    model = _as_text(getattr(args, "model", "")) or _as_text(os.environ.get(MODEL_ENV))
    timeout_seconds = _clamp_float(
        getattr(args, "timeout_seconds", DEFAULT_TIMEOUT_SECONDS),
        default=DEFAULT_TIMEOUT_SECONDS,
        min_value=5.0,
        max_value=900.0,
    )
    max_output_chars = _clamp_int(
        getattr(args, "max_output_chars", DEFAULT_MAX_OUTPUT_CHARS),
        default=DEFAULT_MAX_OUTPUT_CHARS,
        min_value=256,
        max_value=200_000,
    )
    sandbox = _as_text(getattr(args, "sandbox", "workspace-write")) or "workspace-write"
    return CodexWrapperConfig(
        auth_env=auth_env,
        codex_command=codex_command,
        workspace=workspace,
        endpoint=_as_text(getattr(args, "endpoint", "")),
        model=model,
        timeout_seconds=timeout_seconds,
        sandbox=sandbox,
        max_output_chars=max_output_chars,
    )


def _route_request(
    *,
    raw: str,
    responder: SDEPEndpointIdentity,
    config: CodexWrapperConfig,
) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return _error_response(
            request_id="",
            responder=responder,
            code="request.invalid_json",
            message="SDEP request is not valid JSON.",
            category="response_validity",
        )
    if not isinstance(payload, dict):
        return _error_response(
            request_id="",
            responder=responder,
            code="request.invalid",
            message="SDEP request payload must be a JSON object.",
            category="response_validity",
        )

    message_type = _as_text(payload.get("message_type"))
    if message_type == SDEP_AGENT_DESCRIBE_REQUEST:
        return _handle_describe(payload=payload, responder=responder, config=config)
    if message_type == SDEP_EXECUTE_REQUEST:
        return _handle_execute(payload=payload, responder=responder, config=config)
    return _error_response(
        request_id=payload.get("request_id"),
        responder=responder,
        code="request.invalid_message_type",
        message=(
            f"Unsupported message_type {message_type!r}; expected "
            f"{SDEP_EXECUTE_REQUEST!r} or {SDEP_AGENT_DESCRIBE_REQUEST!r}."
        ),
        category="response_validity",
    )


def _handle_describe(
    *,
    payload: dict[str, Any],
    responder: SDEPEndpointIdentity,
    config: CodexWrapperConfig,
) -> dict[str, Any]:
    request_id = _fallback_request_id(payload.get("request_id"))
    try:
        request = SDEPDescribeRequest.from_dict(payload)
    except Exception as exc:
        return _describe_error_response(
            request_id=request_id,
            responder=responder,
            config=config,
            code="request.invalid",
            message=f"Invalid describe.request payload: {exc}",
        )

    action_filter = {
        _as_text(item)
        for item in request.query.action_types
        if _as_text(item)
    }
    capabilities: list[SDEPActionCapability] = []
    for action_type in SUPPORTED_ACTIONS:
        if action_filter and action_type not in action_filter:
            continue
        capabilities.append(_capability_for_action(action_type))

    response = SDEPDescribeResponse(
        request_id=request.request_id,
        status="success",
        responder=responder,
        description=SDEPAgentDescription(
            protocol_support=SDEPProtocolSupport(protocol="sdep", versions=[SDEP_VERSION]),
            capabilities=capabilities,
            capability_version="codex-wrapper-v1",
            summary=(
                "Codex-backed wrapper via codex exec. "
                "Full: gather_evidence/system. Limited: communicate/schedule."
            ),
            metadata={
                "provider": CODEX_PROVIDER_ID,
                "integration_backend": "codex.exec",
                "codex_command": config.codex_command,
                "endpoint": config.endpoint,
            },
        ),
    )
    return response.to_dict()


def _capability_for_action(action_type: str) -> SDEPActionCapability:
    support_level = capability_support_level(action_type)
    limited = support_level == SUPPORT_LEVEL_LIMITED
    side_effect_class = "read_only" if limited else "state_change"
    mode = "limited_draft" if limited else "live_execution"
    return SDEPActionCapability(
        action_type=action_type,
        target_kinds=["external.service", "workspace"],
        mode_support=["sync"],
        dry_run_supported=True,
        side_effect_class=side_effect_class,
        outcome_type="observation",
        semantic_inputs=["execution_brief", "input", "parameters"],
        input_expectation="json_object",
        parameter_expectation="json_object",
        metadata={
            "provider": CODEX_PROVIDER_ID,
            "support_level": support_level,
            "integration_backend": "codex.exec",
            "mode": mode,
            "phase_scope": "phase1b",
            "honest_limited_support": limited,
        },
    )


def _handle_execute(
    *,
    payload: dict[str, Any],
    responder: SDEPEndpointIdentity,
    config: CodexWrapperConfig,
) -> dict[str, Any]:
    request_id = _fallback_request_id(payload.get("request_id"))
    try:
        request = SDEPExecuteRequest.from_dict(payload)
    except Exception as exc:
        return _error_response(
            request_id=request_id,
            responder=responder,
            code="request.invalid",
            message=f"Invalid execute.request payload: {exc}",
            category="response_validity",
        )

    auth_token = _as_text(os.environ.get(config.auth_env))
    if not auth_token:
        return _error_response(
            request_id=request.request_id,
            responder=responder,
            code="auth.missing",
            message=f"Missing required auth environment variable {config.auth_env!r}.",
            category="auth_config",
            details={"auth_env": config.auth_env},
        )

    action_type = request.execution.action_type
    if action_type not in SUPPORTED_ACTIONS:
        return _error_response(
            request_id=request.request_id,
            responder=responder,
            code="capability.unsupported",
            message=f"Unsupported capability {action_type!r}.",
            category="unsupported_capability",
            details={"supported_capabilities": list(SUPPORTED_ACTIONS)},
        )

    try:
        result = _invoke_codex_exec(
            request=request,
            config=config,
            auth_token=auth_token,
        )
    except CodexWrapperError as exc:
        details = dict(exc.details)
        details.setdefault("error_category", exc.category)
        return _error_response(
            request_id=request.request_id,
            responder=responder,
            code=exc.code,
            message=exc.message,
            category=exc.category,
            retryable=exc.retryable,
            details=details,
        )

    support_level = capability_support_level(action_type)
    limited = support_level == SUPPORT_LEVEL_LIMITED
    output = {
        "provider": CODEX_PROVIDER_ID,
        "action_type": action_type,
        "support_level": support_level,
        "integration_backend": "codex.exec",
        "mode": "limited_draft" if limited else "live_execution",
        "summary": result.summary,
        "evidence": result.evidence,
        "actions": result.actions,
        "artifacts": result.artifacts,
        "errors": result.errors,
        "codex_status": result.status,
    }

    normalized_status = result.status.lower()
    if normalized_status in {"partial"}:
        return _error_response(
            request_id=request.request_id,
            responder=responder,
            code="execution.partial_failure",
            message="Codex reported partial execution.",
            category="partial_execution",
            details={"codex_result": output},
        )
    if normalized_status in {"failed", "error"}:
        return _error_response(
            request_id=request.request_id,
            responder=responder,
            code="execution.failed_outcome",
            message="Codex reported failed execution.",
            category="partial_execution",
            details={"codex_result": output},
        )
    if result.errors:
        return _error_response(
            request_id=request.request_id,
            responder=responder,
            code="execution.partial_failure",
            message="Codex reported errors in the execution payload.",
            category="partial_execution",
            details={"codex_result": output},
        )

    response = SDEPExecuteResponse(
        request_id=request.request_id,
        status="success",
        responder=responder,
        outcome=SDEPExecutionOutcome(
            execution_id=f"exec-{uuid4().hex}",
            status="success",
            outcome_type="observation",
            output=output,
            metadata={
                "provider": CODEX_PROVIDER_ID,
                "support_level": support_level,
            },
        ),
        metadata={
            "provider": CODEX_PROVIDER_ID,
            "support_level": support_level,
        },
    )
    return response.to_dict()


def _invoke_codex_exec(
    *,
    request: SDEPExecuteRequest,
    config: CodexWrapperConfig,
    auth_token: str,
) -> CodexInvocationResult:
    action_type = request.execution.action_type
    limited = action_type in LIMITED_ACTIONS
    prompt = _build_codex_prompt(request=request, limited=limited)

    with tempfile.TemporaryDirectory(prefix="spice_codex_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        schema_path = tmp_path / "schema.json"
        output_path = tmp_path / "last_message.json"
        schema_path.write_text(
            json.dumps(_codex_output_schema(), ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        command = _build_codex_exec_command(
            config=config,
            workspace=config.workspace,
            prompt=prompt,
            schema_path=schema_path,
            output_path=output_path,
            limited=limited,
        )
        env = os.environ.copy()
        env["OPENAI_API_KEY"] = auth_token
        env[config.auth_env] = auth_token

        try:
            completed = subprocess.run(
                command,
                cwd=str(config.workspace),
                env=env,
                text=True,
                capture_output=True,
                timeout=config.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise CodexWrapperError(
                code="transport.timeout",
                message=f"Codex exec timed out after {config.timeout_seconds:.1f}s.",
                category="transport_runtime",
                details={"timeout_seconds": config.timeout_seconds},
                retryable=True,
            ) from exc
        except OSError as exc:
            raise CodexWrapperError(
                code="transport.runtime",
                message=f"Failed to launch Codex command: {exc}",
                category="transport_runtime",
                details={"command": command},
            ) from exc

        if completed.returncode != 0:
            stderr = _truncate_text(completed.stderr or "", max_chars=config.max_output_chars)
            category = "auth_config" if _looks_like_auth_error(stderr) else "transport_runtime"
            code = "auth.failed" if category == "auth_config" else "transport.runtime"
            raise CodexWrapperError(
                code=code,
                message=f"Codex exec failed (exit={completed.returncode}): {stderr or '<no stderr>'}",
                category=category,
                details={
                    "exit_code": completed.returncode,
                    "stderr": stderr,
                    "stdout": _truncate_text(completed.stdout or "", max_chars=config.max_output_chars),
                },
                retryable=(category == "transport_runtime"),
            )

        if not output_path.exists():
            raise CodexWrapperError(
                code="response.invalid",
                message="Codex exec did not produce output-last-message file.",
                category="response_validity",
                details={
                    "output_last_message": str(output_path),
                },
            )
        raw_output = output_path.read_text(encoding="utf-8")
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            raise CodexWrapperError(
                code="response.invalid",
                message="Codex exec output is not valid JSON.",
                category="response_validity",
                details={"raw_output": _truncate_text(raw_output, max_chars=config.max_output_chars)},
            ) from exc
        if not isinstance(parsed, dict):
            raise CodexWrapperError(
                code="response.invalid",
                message="Codex exec output must be a JSON object.",
                category="response_validity",
                details={"raw_output_type": str(type(parsed))},
            )

        return _normalize_codex_result(parsed, max_output_chars=config.max_output_chars)


def _build_codex_exec_command(
    *,
    config: CodexWrapperConfig,
    workspace: Path,
    prompt: str,
    schema_path: Path,
    output_path: Path,
    limited: bool,
) -> list[str]:
    command_prefix = shlex.split(config.codex_command)
    if not command_prefix:
        raise CodexWrapperError(
            code="transport.runtime",
            message="Codex command is empty.",
            category="transport_runtime",
            details={"codex_command": config.codex_command},
        )
    sandbox = "read-only" if limited else config.sandbox
    command: list[str] = [
        *command_prefix,
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "--full-auto",
        "--sandbox",
        sandbox,
        "--cd",
        str(workspace),
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(output_path),
        "--color",
        "never",
    ]
    if config.model:
        command.extend(["--model", config.model])
    command.append(prompt)
    return command


def _build_codex_prompt(*, request: SDEPExecuteRequest, limited: bool) -> str:
    execution_payload = request.execution.to_dict()
    mode_guidance = (
        "LIMITED MODE: do not execute external account side effects; provide draft/plan output only."
        if limited
        else "FULL MODE: perform bounded workspace-oriented execution as needed."
    )
    return (
        "You are spice-agent-codex, a Codex execution wrapper for SPICE Personal.\n"
        "Return ONLY a JSON object matching the provided schema.\n"
        "Do not include markdown.\n"
        f"{mode_guidance}\n"
        "Execution request payload:\n"
        f"{json.dumps(execution_payload, ensure_ascii=True, indent=2, sort_keys=True)}\n"
        "Use concise factual summaries."
    )


def _codex_output_schema() -> dict[str, Any]:
    item_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "id",
            "title",
            "detail",
            "text",
            "source",
            "url",
            "status",
            "confidence",
        ],
        "properties": {
            "id": {"type": ["string", "null"]},
            "title": {"type": ["string", "null"]},
            "detail": {"type": ["string", "null"]},
            "text": {"type": ["string", "null"]},
            "source": {"type": ["string", "null"]},
            "url": {"type": ["string", "null"]},
            "status": {"type": ["string", "null"]},
            "confidence": {"type": ["number", "null"]},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "status",
            "summary",
            "evidence",
            "actions",
            "artifacts",
            "errors",
        ],
        "properties": {
            "status": {
                "type": "string",
                "enum": ["success", "partial", "failed"],
            },
            "summary": {"type": "string"},
            "evidence": {"type": "array", "items": item_schema},
            "actions": {"type": "array", "items": item_schema},
            "artifacts": {"type": "array", "items": item_schema},
            "errors": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
    }


def _normalize_codex_result(
    payload: dict[str, Any],
    *,
    max_output_chars: int,
) -> CodexInvocationResult:
    status = _as_text(payload.get("status")).lower() or "failed"
    if status not in {"success", "partial", "failed"}:
        status = "failed"
    summary = _truncate_text(_as_text(payload.get("summary")), max_chars=max_output_chars)
    evidence = _normalize_object_list(payload.get("evidence"))
    actions = _normalize_object_list(payload.get("actions"))
    artifacts = _normalize_object_list(payload.get("artifacts"))
    errors = _normalize_string_list(payload.get("errors"), max_chars=max_output_chars)
    return CodexInvocationResult(
        status=status,
        summary=summary,
        evidence=evidence,
        actions=actions,
        artifacts=artifacts,
        errors=errors,
        raw_payload=dict(payload),
    )


def _normalize_object_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            normalized.append(dict(item))
    return normalized


def _normalize_string_list(value: Any, *, max_chars: int) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        token = _as_text(item)
        if token:
            normalized.append(_truncate_text(token, max_chars=max_chars))
    return normalized


def _describe_error_response(
    *,
    request_id: str,
    responder: SDEPEndpointIdentity,
    config: CodexWrapperConfig,
    code: str,
    message: str,
) -> dict[str, Any]:
    response = SDEPDescribeResponse(
        request_id=_fallback_request_id(request_id),
        status="error",
        responder=responder,
        description=SDEPAgentDescription(
            protocol_support=SDEPProtocolSupport(protocol="sdep", versions=[SDEP_VERSION]),
            capabilities=[],
            capability_version="codex-wrapper-v1",
            summary="Codex wrapper describe failed.",
            metadata={
                "provider": CODEX_PROVIDER_ID,
                "integration_backend": "codex.exec",
                "codex_command": config.codex_command,
            },
        ),
        error=SDEPError(
            code=code,
            message=message,
            retryable=False,
            details={"error_category": "response_validity"},
        ),
    )
    return response.to_dict()


def _error_response(
    *,
    request_id: Any,
    responder: SDEPEndpointIdentity,
    code: str,
    message: str,
    category: str,
    retryable: bool = False,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = dict(details or {})
    payload.setdefault("error_category", category)
    return build_error_response(
        _fallback_request_id(request_id),
        responder=responder,
        code=code,
        message=message,
        retryable=retryable,
        details=payload,
    )


def _looks_like_auth_error(message: str) -> bool:
    lowered = message.lower()
    patterns = (
        "api key",
        "authentication",
        "unauthorized",
        "forbidden",
        "login",
        "token",
    )
    return any(pattern in lowered for pattern in patterns)


def _fallback_request_id(value: Any) -> str:
    token = _as_text(value)
    if token:
        return token
    return f"sdep-req-{uuid4().hex}"


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return str(value).strip() if value is not None else ""


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


def _clamp_float(value: Any, *, default: float, min_value: float, max_value: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        return default
    if parsed < min_value:
        return min_value
    if parsed > max_value:
        return max_value
    return parsed


def _truncate_text(value: str, *, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    if max_chars <= 3:
        return value[:max_chars]
    return value[: max_chars - 3] + "..."


def _write(payload: dict[str, Any]) -> None:
    os.sys.stdout.write(json.dumps(payload, ensure_ascii=True))
    os.sys.stdout.flush()


def _write_stderr(message: str) -> None:
    os.sys.stderr.write(message)
    os.sys.stderr.flush()


if __name__ == "__main__":
    raise SystemExit(main())
