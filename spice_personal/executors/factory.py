from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spice.executors import (
    CLIActionMapping,
    CLIAdapterExecutor,
    CLIAdapterProfile,
    CLIInvocation,
    Executor,
    MockExecutor,
    SDEPExecutor,
    SubprocessSDEPTransport,
)


PERSONAL_EXECUTOR_MODE_ENV = "SPICE_PERSONAL_EXECUTOR"
PERSONAL_EXECUTOR_TIMEOUT_ENV = "SPICE_PERSONAL_EXECUTOR_TIMEOUT"
PERSONAL_CLI_COMMAND_ENV = "SPICE_PERSONAL_CLI_COMMAND"
PERSONAL_CLI_PROFILE_ENV = "SPICE_PERSONAL_CLI_PROFILE"
PERSONAL_CLI_PROFILE_PATH_ENV = "SPICE_PERSONAL_CLI_PROFILE_PATH"
PERSONAL_CLI_PARSER_MODE_ENV = "SPICE_PERSONAL_CLI_PARSER_MODE"
PERSONAL_SDEP_COMMAND_ENV = "SPICE_PERSONAL_SDEP_COMMAND"


PERSONAL_EVIDENCE_OPERATION = "personal.gather_evidence"
PERSONAL_SYSTEM_OPERATION = "personal.system"
PERSONAL_COMMUNICATE_OPERATION = "personal.communicate"
PERSONAL_SCHEDULE_OPERATION = "personal.schedule"


@dataclass(slots=True)
class PersonalExecutorConfig:
    mode: str = "mock"
    timeout_seconds: float = 20.0
    cli_profile: str | None = None
    cli_profile_path: str | None = None
    cli_command: str | None = None
    cli_parser_mode: str = "json"
    sdep_command: str | None = None


def build_executor(config: PersonalExecutorConfig) -> Executor:
    mode = (config.mode or "mock").strip().lower()
    if mode == "mock":
        return MockExecutor()
    if mode == "cli":
        profile = _build_cli_profile(config)
        return CLIAdapterExecutor(profile)
    if mode == "sdep":
        command = _split_command(config.sdep_command)
        if not command:
            raise ValueError(
                "SDEP executor mode requires --sdep-command or SPICE_PERSONAL_SDEP_COMMAND."
            )
        transport = SubprocessSDEPTransport(
            command,
            timeout_seconds=float(config.timeout_seconds),
        )
        return SDEPExecutor(transport)

    raise ValueError(f"Unsupported personal executor mode: {config.mode!r}")


def _build_cli_profile(config: PersonalExecutorConfig) -> CLIAdapterProfile:
    if config.cli_profile_path:
        return _load_profile_from_json(
            profile_path=Path(config.cli_profile_path),
            fallback_command=config.cli_command,
            timeout_seconds=float(config.timeout_seconds),
        )

    profile_name = (config.cli_profile or "personal.default").strip().lower()
    if profile_name in {"default", "personal.default"}:
        parser_mode = _normalize_parser_mode(config.cli_parser_mode, default="json")
        return _build_builtin_profile(
            profile_id="personal.default",
            display_name="Personal Default CLI",
            command=config.cli_command,
            parser_mode=parser_mode,
            timeout_seconds=float(config.timeout_seconds),
        )
    if profile_name in {"text", "personal.text"}:
        return _build_builtin_profile(
            profile_id="personal.text",
            display_name="Personal Text CLI",
            command=config.cli_command,
            parser_mode="text",
            timeout_seconds=float(config.timeout_seconds),
        )

    raise ValueError(
        "Unsupported --cli-profile. Supported built-ins: default, text. "
        "Use --cli-profile-path for custom profile JSON."
    )


def _build_builtin_profile(
    *,
    profile_id: str,
    display_name: str,
    command: str | None,
    parser_mode: str,
    timeout_seconds: float,
) -> CLIAdapterProfile:
    argv = _split_command(command)
    if not argv:
        raise ValueError(
            "CLI executor mode requires --cli-command or SPICE_PERSONAL_CLI_COMMAND "
            "for the built-in profile."
        )

    return CLIAdapterProfile(
        profile_id=profile_id,
        display_name=display_name,
        default_timeout_seconds=timeout_seconds,
        action_mappings={
            PERSONAL_EVIDENCE_OPERATION: CLIActionMapping(
                action_type=PERSONAL_EVIDENCE_OPERATION,
                parser_mode=parser_mode,
                default_outcome_type="observation",
                render_invocation=_build_render_invocation(
                    argv=argv,
                    timeout_seconds=timeout_seconds,
                ),
            ),
            PERSONAL_SYSTEM_OPERATION: CLIActionMapping(
                action_type=PERSONAL_SYSTEM_OPERATION,
                parser_mode=parser_mode,
                default_outcome_type="observation",
                render_invocation=_build_render_invocation(
                    argv=argv,
                    timeout_seconds=timeout_seconds,
                ),
            ),
            PERSONAL_COMMUNICATE_OPERATION: CLIActionMapping(
                action_type=PERSONAL_COMMUNICATE_OPERATION,
                parser_mode=parser_mode,
                default_outcome_type="observation",
                render_invocation=_build_render_invocation(
                    argv=argv,
                    timeout_seconds=timeout_seconds,
                ),
            ),
            PERSONAL_SCHEDULE_OPERATION: CLIActionMapping(
                action_type=PERSONAL_SCHEDULE_OPERATION,
                parser_mode=parser_mode,
                default_outcome_type="observation",
                render_invocation=_build_render_invocation(
                    argv=argv,
                    timeout_seconds=timeout_seconds,
                ),
            ),
        },
    )


def _load_profile_from_json(
    *,
    profile_path: Path,
    fallback_command: str | None,
    timeout_seconds: float,
) -> CLIAdapterProfile:
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("CLI profile JSON must be an object.")

    profile_id = str(payload.get("profile_id") or "personal.custom")
    display_name = str(payload.get("display_name") or profile_id)
    default_timeout_raw = payload.get("default_timeout_seconds", timeout_seconds)
    default_timeout = float(default_timeout_raw)
    actions_raw = payload.get("actions")
    if not isinstance(actions_raw, dict) or not actions_raw:
        raise ValueError("CLI profile JSON requires non-empty object field `actions`.")

    action_mappings: dict[str, CLIActionMapping] = {}
    for action_type, action_payload in actions_raw.items():
        if not isinstance(action_payload, dict):
            raise ValueError(f"actions[{action_type!r}] must be an object.")

        parser_mode = _normalize_parser_mode(action_payload.get("parser_mode"), default="json")
        default_outcome_type = str(action_payload.get("default_outcome_type") or "observation")
        command = str(action_payload.get("command") or "").strip()
        if not command:
            command = str(fallback_command or "").strip()
        argv = _split_command(command)
        if not argv:
            raise ValueError(
                "Custom CLI profile action requires `command`, or provide --cli-command fallback."
            )

        action_timeout = float(action_payload.get("timeout_seconds", default_timeout))
        action_mappings[str(action_type)] = CLIActionMapping(
            action_type=str(action_type),
            parser_mode=parser_mode,
            default_outcome_type=default_outcome_type,
            render_invocation=_build_render_invocation(
                argv=argv,
                timeout_seconds=action_timeout,
            ),
        )

    return CLIAdapterProfile(
        profile_id=profile_id,
        display_name=display_name,
        default_timeout_seconds=default_timeout,
        action_mappings=action_mappings,
    )


def _build_render_invocation(*, argv: list[str], timeout_seconds: float):
    def render_invocation(ctx: Any) -> CLIInvocation:
        request_payload = {
            "action_type": str(getattr(ctx, "action_type", "")),
            "target": dict(getattr(ctx, "target", {}) or {}),
            "input_payload": dict(getattr(ctx, "input_payload", {}) or {}),
            "parameters": dict(getattr(ctx, "parameters", {}) or {}),
            "constraints": list(getattr(ctx, "constraints", []) or []),
            "mode": str(getattr(ctx, "mode", "sync")),
            "dry_run": bool(getattr(ctx, "dry_run", False)),
        }
        return CLIInvocation(
            argv=list(argv),
            stdin_text=json.dumps(request_payload, ensure_ascii=True),
            timeout_seconds=float(timeout_seconds),
        )

    return render_invocation


def _split_command(command: str | None) -> list[str]:
    if command is None:
        return []
    token = command.strip()
    if not token:
        return []
    return shlex.split(token)


def _normalize_parser_mode(value: Any, *, default: str) -> str:
    token = str(value or default).strip().lower()
    if token in {"json", "text"}:
        return token
    return default
