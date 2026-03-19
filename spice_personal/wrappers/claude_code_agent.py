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
from spice.llm.util import extract_first_json_object, strip_markdown_fences
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


CLAUDE_CODE_PROVIDER_ID = "claude_code"
DEFAULT_AUTH_ENV = "ANTHROPIC_API_KEY"
DEFAULT_CLAUDE_CODE_COMMAND = "claude"
CLAUDE_CODE_COMMAND_ENV = "SPICE_AGENT_CLAUDE_CODE_COMMAND"
WORKSPACE_ENV = "SPICE_AGENT_CLAUDE_CODE_WORKSPACE"
MODEL_ENV = "SPICE_AGENT_CLAUDE_CODE_MODEL"
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
class ClaudeCodeWrapperConfig:
    auth_env: str
    command: str
    workspace: Path
    endpoint: str
    model: str
    timeout_seconds: float
    sandbox: str
    max_output_chars: int


@dataclass(slots=True, frozen=True)
class ClaudeCodeInvocationResult:
    status: str
    summary: str
    evidence: list[dict[str, Any]]
    actions: list[dict[str, Any]]
    artifacts: list[dict[str, Any]]
    errors: list[str]
    raw_payload: dict[str, Any]


@dataclass(slots=True)
class ClaudeCodeWrapperError(Exception):
    code: str
    message: str
    category: str
    details: dict[str, Any]
    retryable: bool = False

    def __str__(self) -> str:
        return self.message


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spice-agent-claude-code",
        description="Thin Claude Code-backed SDEP wrapper for SPICE personal execution.",
    )
    parser.add_argument("--auth-env", type=str, default=DEFAULT_AUTH_ENV)
    parser.add_argument("--claude-command", type=str, default="")
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
            id=f"agent.{CLAUDE_CODE_PROVIDER_ID}",
            name="Claude Code Agent Wrapper",
            version="0.1",
            vendor="SpicePersonal",
            implementation="spice_personal.wrappers.claude_code_agent",
            role=SDEP_ROLE_EXECUTOR,
        )
        raw = os.sys.stdin.read()
        payload = _route_request(raw=raw, responder=responder, config=config)
        _write(payload)
        return 0
    except Exception as exc:  # hard wrapper startup/crash path
        _write_stderr(f"claude code wrapper startup failure: {exc}\n")
        return 1


def _build_config(args: argparse.Namespace) -> ClaudeCodeWrapperConfig:
    auth_env = _as_text(getattr(args, "auth_env", "")) or DEFAULT_AUTH_ENV
    command = (
        _as_text(getattr(args, "claude_command", ""))
        or _as_text(os.environ.get(CLAUDE_CODE_COMMAND_ENV))
        or DEFAULT_CLAUDE_CODE_COMMAND
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
    return ClaudeCodeWrapperConfig(
        auth_env=auth_env,
        command=command,
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
    config: ClaudeCodeWrapperConfig,
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
    config: ClaudeCodeWrapperConfig,
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
            capability_version="claude-code-wrapper-v1",
            summary=(
                "Claude Code-backed wrapper via CLI delegation. "
                "Full: gather_evidence/system. Limited: communicate/schedule."
            ),
            metadata={
                "provider": CLAUDE_CODE_PROVIDER_ID,
                "integration_backend": "claude_code.exec",
                "claude_command": config.command,
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
            "provider": CLAUDE_CODE_PROVIDER_ID,
            "support_level": support_level,
            "integration_backend": "claude_code.exec",
            "mode": mode,
            "phase_scope": "phase1c",
            "honest_limited_support": limited,
        },
    )


def _handle_execute(
    *,
    payload: dict[str, Any],
    responder: SDEPEndpointIdentity,
    config: ClaudeCodeWrapperConfig,
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
        result = _invoke_claude_code_exec(
            request=request,
            config=config,
            auth_token=auth_token,
        )
    except ClaudeCodeWrapperError as exc:
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
        "provider": CLAUDE_CODE_PROVIDER_ID,
        "action_type": action_type,
        "support_level": support_level,
        "integration_backend": "claude_code.exec",
        "mode": "limited_draft" if limited else "live_execution",
        "summary": result.summary,
        "evidence": result.evidence,
        "actions": result.actions,
        "artifacts": result.artifacts,
        "errors": result.errors,
        "claude_code_status": result.status,
    }

    normalized_status = result.status.lower()
    if normalized_status in {"partial"}:
        return _error_response(
            request_id=request.request_id,
            responder=responder,
            code="execution.partial_failure",
            message="Claude Code reported partial execution.",
            category="partial_execution",
            details={"claude_code_result": output},
        )
    if normalized_status in {"failed", "error"}:
        return _error_response(
            request_id=request.request_id,
            responder=responder,
            code="execution.failed_outcome",
            message="Claude Code reported failed execution.",
            category="partial_execution",
            details={"claude_code_result": output},
        )
    if result.errors:
        return _error_response(
            request_id=request.request_id,
            responder=responder,
            code="execution.partial_failure",
            message="Claude Code reported errors in the execution payload.",
            category="partial_execution",
            details={"claude_code_result": output},
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
                "provider": CLAUDE_CODE_PROVIDER_ID,
                "support_level": support_level,
            },
        ),
        metadata={
            "provider": CLAUDE_CODE_PROVIDER_ID,
            "support_level": support_level,
        },
    )
    return response.to_dict()


def _invoke_claude_code_exec(
    *,
    request: SDEPExecuteRequest,
    config: ClaudeCodeWrapperConfig,
    auth_token: str,
) -> ClaudeCodeInvocationResult:
    action_type = request.execution.action_type
    limited = action_type in LIMITED_ACTIONS
    prompt = _build_claude_code_prompt(
        request=request,
        limited=limited,
        workspace_root=config.workspace,
    )

    with tempfile.TemporaryDirectory(prefix="spice_claude_code_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        schema_path = tmp_path / "schema.json"
        output_path = tmp_path / "last_message.json"
        schema_path.write_text(
            json.dumps(_claude_code_output_schema(), ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        command = _build_claude_code_exec_command(
            config=config,
            workspace=config.workspace,
            prompt=prompt,
            schema_path=schema_path,
            output_path=output_path,
            limited=limited,
        )
        env = os.environ.copy()
        env["ANTHROPIC_API_KEY"] = auth_token
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
            raise ClaudeCodeWrapperError(
                code="transport.timeout",
                message=f"Claude Code exec timed out after {config.timeout_seconds:.1f}s.",
                category="transport_runtime",
                details={"timeout_seconds": config.timeout_seconds},
                retryable=True,
            ) from exc
        except OSError as exc:
            raise ClaudeCodeWrapperError(
                code="transport.runtime",
                message=f"Failed to launch Claude Code command: {exc}",
                category="transport_runtime",
                details={"command": command},
            ) from exc

        if completed.returncode != 0:
            stderr = _truncate_text(completed.stderr or "", max_chars=config.max_output_chars)
            if _looks_like_unsupported_option_error(stderr):
                fallback_command = _build_claude_code_print_command(
                    config=config,
                    prompt=prompt,
                    limited=limited,
                )
                try:
                    completed = subprocess.run(
                        fallback_command,
                        cwd=str(config.workspace),
                        env=env,
                        text=True,
                        capture_output=True,
                        timeout=config.timeout_seconds,
                        check=False,
                    )
                except subprocess.TimeoutExpired as exc:
                    raise ClaudeCodeWrapperError(
                        code="transport.timeout",
                        message=f"Claude Code print mode timed out after {config.timeout_seconds:.1f}s.",
                        category="transport_runtime",
                        details={"timeout_seconds": config.timeout_seconds},
                        retryable=True,
                    ) from exc
                except OSError as exc:
                    raise ClaudeCodeWrapperError(
                        code="transport.runtime",
                        message=f"Failed to launch Claude Code command: {exc}",
                        category="transport_runtime",
                        details={"command": fallback_command},
                    ) from exc

                if completed.returncode != 0:
                    fallback_stderr = _truncate_text(completed.stderr or "", max_chars=config.max_output_chars)
                    category = "auth_config" if _looks_like_auth_error(fallback_stderr) else "transport_runtime"
                    code = "auth.failed" if category == "auth_config" else "transport.runtime"
                    envelope = _first_claude_print_envelope(completed.stdout)
                    envelope_details = _summarize_claude_print_envelope(
                        envelope,
                        max_chars=config.max_output_chars,
                    )
                    details = _build_claude_print_base_details(
                        raw_output=completed.stdout,
                        stderr=completed.stderr or "",
                        fallback_command=fallback_command,
                        max_output_chars=config.max_output_chars,
                    )
                    details.update(envelope_details)
                    details["exit_code"] = completed.returncode
                    raise ClaudeCodeWrapperError(
                        code=code,
                        message=(
                            "Claude Code print mode failed "
                            f"(exit={completed.returncode}): {fallback_stderr or '<no stderr>'}"
                        ),
                        category=category,
                        details=details,
                        retryable=(category == "transport_runtime"),
                    )
                try:
                    parsed = _parse_claude_print_json_output(
                        completed.stdout,
                        max_output_chars=config.max_output_chars,
                        stderr=completed.stderr or "",
                        fallback_command=fallback_command,
                    )
                except ClaudeCodeWrapperError as exc:
                    details = dict(exc.details)
                    details.setdefault("fallback_command", fallback_command)
                    details.setdefault("stdout", _truncate_text(completed.stdout or "", max_chars=config.max_output_chars))
                    if completed.stderr:
                        details.setdefault(
                            "stderr",
                            _truncate_text(completed.stderr, max_chars=config.max_output_chars),
                        )
                    raise ClaudeCodeWrapperError(
                        code=exc.code,
                        message=exc.message,
                        category=exc.category,
                        details=details,
                        retryable=exc.retryable,
                    ) from exc
                return _normalize_claude_code_result(
                    parsed,
                    max_output_chars=config.max_output_chars,
                )

            category = "auth_config" if _looks_like_auth_error(stderr) else "transport_runtime"
            code = "auth.failed" if category == "auth_config" else "transport.runtime"
            raise ClaudeCodeWrapperError(
                code=code,
                message=(
                    "Claude Code exec failed "
                    f"(exit={completed.returncode}): {stderr or '<no stderr>'}"
                ),
                category=category,
                details={
                    "exit_code": completed.returncode,
                    "stderr": stderr,
                    "stdout": _truncate_text(completed.stdout or "", max_chars=config.max_output_chars),
                },
                retryable=(category == "transport_runtime"),
            )

        if not output_path.exists():
            raise ClaudeCodeWrapperError(
                code="response.invalid",
                message="Claude Code exec did not produce output-last-message file.",
                category="response_validity",
                details={"output_last_message": str(output_path)},
            )
        raw_output = output_path.read_text(encoding="utf-8")
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            raise ClaudeCodeWrapperError(
                code="response.invalid",
                message="Claude Code exec output is not valid JSON.",
                category="response_validity",
                details={"raw_output": _truncate_text(raw_output, max_chars=config.max_output_chars)},
            ) from exc
        if not isinstance(parsed, dict):
            raise ClaudeCodeWrapperError(
                code="response.invalid",
                message="Claude Code exec output must be a JSON object.",
                category="response_validity",
                details={"raw_output_type": str(type(parsed))},
            )
        return _normalize_claude_code_result(
            parsed,
            max_output_chars=config.max_output_chars,
        )


def _build_claude_code_exec_command(
    *,
    config: ClaudeCodeWrapperConfig,
    workspace: Path,
    prompt: str,
    schema_path: Path,
    output_path: Path,
    limited: bool,
) -> list[str]:
    command_prefix = shlex.split(config.command)
    if not command_prefix:
        raise ClaudeCodeWrapperError(
            code="transport.runtime",
            message="Claude Code command is empty.",
            category="transport_runtime",
            details={"command": config.command},
        )
    sandbox = "read-only" if limited else config.sandbox
    command: list[str] = [
        *command_prefix,
        "exec",
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


def _build_claude_code_print_command(
    *,
    config: ClaudeCodeWrapperConfig,
    prompt: str,
    limited: bool,
) -> list[str]:
    command_prefix = shlex.split(config.command)
    if not command_prefix:
        raise ClaudeCodeWrapperError(
            code="transport.runtime",
            message="Claude Code command is empty.",
            category="transport_runtime",
            details={"command": config.command},
        )
    command: list[str] = [
        *command_prefix,
        "-p",
        prompt,
        "--output-format",
        "json",
        "--json-schema",
        json.dumps(_claude_code_output_schema(), ensure_ascii=True, separators=(",", ":")),
    ]
    if config.model:
        command.extend(["--model", config.model])
    if limited:
        command.extend(["--permission-mode", "plan"])
    else:
        # Non-interactive SDEP path has no human-in-the-loop for tool prompts.
        command.append("--dangerously-skip-permissions")
    command.extend(["--max-turns", "8"])
    return command


def _parse_claude_print_json_output(
    raw_output: str,
    *,
    max_output_chars: int,
    stderr: str = "",
    fallback_command: list[str] | None = None,
) -> dict[str, Any]:
    normalized = strip_markdown_fences(raw_output or "")
    candidates = _collect_json_candidate_objects(normalized)
    if not candidates:
        candidate = extract_first_json_object(normalized)
        if candidate is not None:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                candidates = [parsed]

    if not candidates:
        raise ClaudeCodeWrapperError(
            code="response.invalid",
            message="Claude Code print mode did not return a JSON object.",
            category="response_validity",
            details=_build_claude_print_base_details(
                raw_output=raw_output,
                stderr=stderr,
                fallback_command=fallback_command,
                max_output_chars=max_output_chars,
            ),
        )

    last_error_message = ""
    last_error_envelope: dict[str, Any] | None = None
    for parsed in candidates:
        payload = _extract_claude_result_payload(parsed)
        if payload is not None:
            return payload

        error_message = _extract_claude_print_error_message(parsed)
        if error_message:
            last_error_message = error_message
            last_error_envelope = parsed

    if last_error_message:
        category = "auth_config" if _looks_like_auth_error(last_error_message) else "transport_runtime"
        code = "auth.failed" if category == "auth_config" else "transport.runtime"
        details = _build_claude_print_base_details(
            raw_output=raw_output,
            stderr=stderr,
            fallback_command=fallback_command,
            max_output_chars=max_output_chars,
        )
        details.update(
            _summarize_claude_print_envelope(
                last_error_envelope,
                max_chars=max_output_chars,
            )
        )
        raise ClaudeCodeWrapperError(
            code=code,
            message=f"Claude Code print mode returned error payload: {last_error_message}",
            category=category,
            details=details,
            retryable=(category == "transport_runtime"),
        )

    details = _build_claude_print_base_details(
        raw_output=raw_output,
        stderr=stderr,
        fallback_command=fallback_command,
        max_output_chars=max_output_chars,
    )
    if candidates:
        details.update(
            _summarize_claude_print_envelope(
                candidates[-1],
                max_chars=max_output_chars,
            )
        )
    raise ClaudeCodeWrapperError(
        code="response.invalid",
        message="Claude Code print mode JSON does not match expected result payload.",
        category="response_validity",
        details=details,
    )


def _collect_json_candidate_objects(text: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for value in _scan_json_values(text):
        if isinstance(value, dict):
            candidates.append(dict(value))
            continue
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    candidates.append(dict(item))
    return candidates


def _scan_json_values(text: str) -> list[Any]:
    values: list[Any] = []
    decoder = json.JSONDecoder()
    index = 0
    length = len(text)
    while index < length:
        while index < length and text[index] not in "{[":
            index += 1
        if index >= length:
            break
        try:
            value, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            index += 1
            continue
        values.append(value)
        index = end
    return values


def _extract_claude_result_payload(parsed: dict[str, Any]) -> dict[str, Any] | None:
    if _looks_like_claude_result_payload(parsed):
        return dict(parsed)

    for key in ("structured_output", "result", "output"):
        nested = parsed.get(key)
        payload = _extract_nested_payload(nested)
        if payload is not None:
            return payload
    return None


def _extract_nested_payload(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        if _looks_like_claude_result_payload(value):
            return dict(value)
        return None
    if not isinstance(value, str):
        return None
    nested = extract_first_json_object(strip_markdown_fences(value))
    if nested is None:
        return None
    try:
        nested_payload = json.loads(nested)
    except json.JSONDecodeError:
        return None
    if isinstance(nested_payload, dict) and _looks_like_claude_result_payload(nested_payload):
        return dict(nested_payload)
    return None


def _extract_claude_print_error_message(parsed: dict[str, Any]) -> str:
    if not isinstance(parsed, dict):
        return ""
    result_text = _as_text(parsed.get("result"))
    if bool(parsed.get("is_error")):
        if result_text:
            return result_text
        return _as_text(parsed.get("message"))

    subtype = _as_text(parsed.get("subtype")).lower()
    if subtype.startswith("error_"):
        if result_text:
            return result_text
        return subtype

    permission_denials = parsed.get("permission_denials")
    if isinstance(permission_denials, list) and permission_denials:
        return "permission denied for required tools in non-interactive print mode"
    return ""


def _build_claude_print_base_details(
    *,
    raw_output: str,
    stderr: str,
    fallback_command: list[str] | None,
    max_output_chars: int,
) -> dict[str, Any]:
    details: dict[str, Any] = {
        "stdout": _truncate_text(raw_output or "", max_chars=max_output_chars),
    }
    stderr_text = _truncate_text(stderr or "", max_chars=max_output_chars)
    if stderr_text:
        details["stderr"] = stderr_text
    if isinstance(fallback_command, list):
        details["fallback_command"] = list(fallback_command)
    return details


def _first_claude_print_envelope(raw_output: str) -> dict[str, Any] | None:
    normalized = strip_markdown_fences(raw_output or "")
    candidates = _collect_json_candidate_objects(normalized)
    if candidates:
        return candidates[0]
    return None


def _summarize_claude_print_envelope(
    payload: dict[str, Any] | None,
    *,
    max_chars: int,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    details: dict[str, Any] = {}
    for key in ("subtype", "stop_reason", "session_id", "is_error"):
        if key not in payload:
            continue
        details[key] = payload.get(key)
    result_text = _as_text(payload.get("result"))
    if result_text:
        details["result"] = _truncate_text(result_text, max_chars=max_chars)
    errors = payload.get("errors")
    if isinstance(errors, list):
        details["errors"] = [
            _truncate_text(_as_text(item), max_chars=max_chars)
            for item in errors[:10]
            if _as_text(item)
        ]
    elif _as_text(errors):
        details["errors"] = [_truncate_text(_as_text(errors), max_chars=max_chars)]
    permission_denials = payload.get("permission_denials")
    if isinstance(permission_denials, list):
        details["permission_denials"] = _normalize_permission_denials(permission_denials, max_chars=max_chars)
    return details


def _normalize_permission_denials(
    value: list[Any],
    *,
    max_chars: int,
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
        reason_text = _as_text(entry.get("reason"))
        if reason_text:
            item["reason"] = _truncate_text(reason_text, max_chars=max_chars)
        tool_input = entry.get("tool_input")
        if isinstance(tool_input, dict):
            command_text = _as_text(tool_input.get("command"))
            path_text = _as_text(tool_input.get("path"))
            if command_text:
                item["command"] = _truncate_text(command_text, max_chars=max_chars)
            if path_text:
                item["path"] = _truncate_text(path_text, max_chars=max_chars)
        if item:
            normalized.append(item)
    return normalized


def _looks_like_claude_result_payload(payload: dict[str, Any]) -> bool:
    required = {"status", "summary", "evidence", "actions", "artifacts", "errors"}
    return required.issubset(payload.keys())


def _build_claude_code_prompt(
    *,
    request: SDEPExecuteRequest,
    limited: bool,
    workspace_root: Path,
) -> str:
    execution_payload = request.execution.to_dict()
    mode_guidance = (
        "LIMITED MODE: do not execute external account side effects; provide draft/plan output only."
        if limited
        else "FULL MODE: perform bounded workspace-oriented execution as needed."
    )
    scope = _as_text(_as_dict(request.execution.input).get("scope")) or "workspace"
    scope_guardrails = (
        "Execution boundaries:\n"
        f"- Workspace root boundary: {workspace_root}\n"
        f"- Requested scope boundary: {scope}\n"
        "- You must stay within the workspace root and requested scope.\n"
        "- Never read or modify paths outside the workspace root, including parent/home directories.\n"
        "- Use minimum necessary file/command access; avoid unrelated directory enumeration.\n"
        "- For system health checks, prefer read-only inspection and concise summary output.\n"
        "- Do not enumerate unrelated directories or perform exploratory scans."
    )
    return (
        "You are spice-agent-claude-code, a Claude Code execution wrapper for SPICE Personal.\n"
        "Return ONLY a JSON object matching the provided schema.\n"
        "Do not include markdown.\n"
        f"{mode_guidance}\n"
        f"{scope_guardrails}\n"
        "Execution request payload:\n"
        f"{json.dumps(execution_payload, ensure_ascii=True, indent=2, sort_keys=True)}\n"
        "Use concise factual summaries."
    )


def _claude_code_output_schema() -> dict[str, Any]:
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


def _normalize_claude_code_result(
    payload: dict[str, Any],
    *,
    max_output_chars: int,
) -> ClaudeCodeInvocationResult:
    status = _as_text(payload.get("status")).lower() or "failed"
    if status not in {"success", "partial", "failed"}:
        status = "failed"
    summary = _truncate_text(_as_text(payload.get("summary")), max_chars=max_output_chars)
    evidence = _normalize_object_list(payload.get("evidence"))
    actions = _normalize_object_list(payload.get("actions"))
    artifacts = _normalize_object_list(payload.get("artifacts"))
    errors = _normalize_string_list(payload.get("errors"), max_chars=max_output_chars)
    return ClaudeCodeInvocationResult(
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
    config: ClaudeCodeWrapperConfig,
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
            capability_version="claude-code-wrapper-v1",
            summary="Claude Code wrapper describe failed.",
            metadata={
                "provider": CLAUDE_CODE_PROVIDER_ID,
                "integration_backend": "claude_code.exec",
                "claude_command": config.command,
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


def _looks_like_unsupported_option_error(message: str) -> bool:
    lowered = message.lower()
    patterns = (
        "unknown option",
        "unrecognized option",
        "unknown flag",
        "invalid option",
    )
    return any(pattern in lowered for pattern in patterns)


def _fallback_request_id(value: Any) -> str:
    token = _as_text(value)
    if token:
        return token
    return f"sdep-req-{uuid4().hex}"


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


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
