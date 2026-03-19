from __future__ import annotations

import os
from typing import Any

from spice_personal.config.personal_config import PersonalConnectionConfig
from spice_personal.executors.factory import (
    PERSONAL_CLI_COMMAND_ENV,
    PERSONAL_CLI_PARSER_MODE_ENV,
    PERSONAL_CLI_PROFILE_ENV,
    PERSONAL_CLI_PROFILE_PATH_ENV,
    PERSONAL_EXECUTOR_MODE_ENV,
    PERSONAL_EXECUTOR_TIMEOUT_ENV,
    PERSONAL_SDEP_COMMAND_ENV,
    PersonalExecutorConfig,
)


DEFAULT_EXECUTOR_TIMEOUT_SECONDS = 20.0
PERSONAL_MODEL_ENV = "SPICE_PERSONAL_MODEL"


def build_executor_config_from_env() -> PersonalExecutorConfig:
    return build_executor_config_from_sources(args=None, workspace_config=None)


def build_executor_config_from_args(args: Any) -> PersonalExecutorConfig:
    return build_executor_config_from_sources(args=args, workspace_config=None)


def build_executor_config_from_sources(
    args: Any,
    *,
    workspace_config: PersonalConnectionConfig | None,
) -> PersonalExecutorConfig:
    cli_mode = _empty_to_none(_getattr(args, "executor"))
    env_mode = _empty_to_none(os.environ.get(PERSONAL_EXECUTOR_MODE_ENV))
    workspace_mode = _workspace_value(workspace_config, "executor_mode")

    cli_timeout = _getattr(args, "executor_timeout")
    env_timeout = os.environ.get(PERSONAL_EXECUTOR_TIMEOUT_ENV)

    cli_profile = _empty_to_none(_getattr(args, "cli_profile"))
    env_cli_profile = _empty_to_none(os.environ.get(PERSONAL_CLI_PROFILE_ENV))

    cli_profile_path = _empty_to_none(_getattr(args, "cli_profile_path"))
    env_cli_profile_path = _empty_to_none(os.environ.get(PERSONAL_CLI_PROFILE_PATH_ENV))

    cli_command = _empty_to_none(_getattr(args, "cli_command"))
    env_cli_command = _empty_to_none(os.environ.get(PERSONAL_CLI_COMMAND_ENV))
    workspace_cli_command = _workspace_value(workspace_config, "cli_command")

    cli_parser_mode = _empty_to_none(_getattr(args, "cli_parser_mode"))
    env_cli_parser_mode = _empty_to_none(os.environ.get(PERSONAL_CLI_PARSER_MODE_ENV))

    cli_sdep_command = _empty_to_none(_getattr(args, "sdep_command"))
    env_sdep_command = _empty_to_none(os.environ.get(PERSONAL_SDEP_COMMAND_ENV))
    workspace_sdep_command = _workspace_value(workspace_config, "sdep_command")

    timeout = _read_float(
        cli_timeout if cli_timeout is not None else env_timeout,
        default=DEFAULT_EXECUTOR_TIMEOUT_SECONDS,
    )
    mode = _normalize_mode(
        cli_mode or env_mode or workspace_mode or "mock",
        default="mock",
    )
    parser_mode = _normalize_parser_mode(
        cli_parser_mode or env_cli_parser_mode or "json",
        default="json",
    )

    return PersonalExecutorConfig(
        mode=mode,
        timeout_seconds=timeout,
        cli_profile=cli_profile if cli_profile is not None else env_cli_profile,
        cli_profile_path=(
            cli_profile_path
            if cli_profile_path is not None
            else env_cli_profile_path
        ),
        cli_command=(
            cli_command
            if cli_command is not None
            else env_cli_command
            if env_cli_command is not None
            else workspace_cli_command
        ),
        cli_parser_mode=parser_mode,
        sdep_command=(
            cli_sdep_command
            if cli_sdep_command is not None
            else env_sdep_command
            if env_sdep_command is not None
            else workspace_sdep_command
        ),
    )


def resolve_executor_config_for_runtime(
    executor_config: PersonalExecutorConfig | None,
    *,
    workspace_config: PersonalConnectionConfig | None,
) -> PersonalExecutorConfig:
    explicit_mode = _empty_to_none(getattr(executor_config, "mode", None))
    env_mode = _empty_to_none(os.environ.get(PERSONAL_EXECUTOR_MODE_ENV))
    workspace_mode = _workspace_value(workspace_config, "executor_mode")

    explicit_timeout = getattr(executor_config, "timeout_seconds", None)
    env_timeout = os.environ.get(PERSONAL_EXECUTOR_TIMEOUT_ENV)

    explicit_cli_profile = _empty_to_none(getattr(executor_config, "cli_profile", None))
    env_cli_profile = _empty_to_none(os.environ.get(PERSONAL_CLI_PROFILE_ENV))

    explicit_cli_profile_path = _empty_to_none(getattr(executor_config, "cli_profile_path", None))
    env_cli_profile_path = _empty_to_none(os.environ.get(PERSONAL_CLI_PROFILE_PATH_ENV))

    explicit_cli_command = _empty_to_none(getattr(executor_config, "cli_command", None))
    env_cli_command = _empty_to_none(os.environ.get(PERSONAL_CLI_COMMAND_ENV))
    workspace_cli_command = _workspace_value(workspace_config, "cli_command")

    explicit_parser_mode = _empty_to_none(getattr(executor_config, "cli_parser_mode", None))
    env_parser_mode = _empty_to_none(os.environ.get(PERSONAL_CLI_PARSER_MODE_ENV))

    explicit_sdep_command = _empty_to_none(getattr(executor_config, "sdep_command", None))
    env_sdep_command = _empty_to_none(os.environ.get(PERSONAL_SDEP_COMMAND_ENV))
    workspace_sdep_command = _workspace_value(workspace_config, "sdep_command")

    timeout = _read_float(
        explicit_timeout if explicit_timeout is not None else env_timeout,
        default=DEFAULT_EXECUTOR_TIMEOUT_SECONDS,
    )
    mode = _normalize_mode(
        explicit_mode or env_mode or workspace_mode or "mock",
        default="mock",
    )
    parser_mode = _normalize_parser_mode(
        explicit_parser_mode or env_parser_mode or "json",
        default="json",
    )

    return PersonalExecutorConfig(
        mode=mode,
        timeout_seconds=timeout,
        cli_profile=explicit_cli_profile if explicit_cli_profile is not None else env_cli_profile,
        cli_profile_path=(
            explicit_cli_profile_path
            if explicit_cli_profile_path is not None
            else env_cli_profile_path
        ),
        cli_command=(
            explicit_cli_command
            if explicit_cli_command is not None
            else env_cli_command
            if env_cli_command is not None
            else workspace_cli_command
        ),
        cli_parser_mode=parser_mode,
        sdep_command=(
            explicit_sdep_command
            if explicit_sdep_command is not None
            else env_sdep_command
            if env_sdep_command is not None
            else workspace_sdep_command
        ),
    )


def resolve_model_command(
    model: str | None,
    *,
    workspace_config: PersonalConnectionConfig | None,
) -> str | None:
    explicit_model = _empty_to_none(model)
    if explicit_model is not None:
        return explicit_model

    env_model = _empty_to_none(os.environ.get(PERSONAL_MODEL_ENV))
    if env_model is not None:
        return env_model

    return _workspace_value(workspace_config, "model_command")


def _getattr(args: Any, name: str) -> Any:
    if args is None:
        return None
    return getattr(args, name, None)


def _empty_to_none(value: Any) -> str | None:
    if value is None:
        return None
    token = str(value).strip()
    return token or None


def _read_float(value: Any, *, default: float) -> float:
    try:
        if value is None:
            return default
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed <= 0:
        return default
    return parsed


def _workspace_value(
    workspace_config: PersonalConnectionConfig | None,
    field_name: str,
) -> str | None:
    if workspace_config is None:
        return None
    return _empty_to_none(getattr(workspace_config, field_name, None))


def _normalize_mode(value: Any, *, default: str) -> str:
    token = str(value or default).strip().lower()
    if token in {"mock", "cli", "sdep"}:
        return token
    return default


def _normalize_parser_mode(value: Any, *, default: str) -> str:
    token = str(value or default).strip().lower()
    if token in {"json", "text"}:
        return token
    return default
