from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spice_personal.config.provider_resolver import (
    CONNECTION_SCHEMA_VERSION_V1,
    ProviderConnectionPlan,
    compile_provider_connection_plan,
)


PERSONAL_CONFIG_FILENAME = "personal.config.json"


@dataclass(slots=True)
class PersonalConnectionConfig:
    schema_version: str = ""
    model_provider: str | None = None
    model_name: str | None = None
    model_api_key_env: str | None = None
    model_base_url: str | None = None
    model_command: str | None = None
    model_command_source: str = ""
    agent_provider: str | None = None
    agent_mode: str | None = None
    agent_auth_env: str | None = None
    agent_endpoint: str | None = None
    executor_mode: str | None = None
    executor_mode_source: str = ""
    cli_command: str | None = None
    cli_command_source: str = ""
    sdep_command: str | None = None
    sdep_command_source: str = ""


def workspace_personal_config_path(workspace: Path) -> Path:
    return workspace / PERSONAL_CONFIG_FILENAME


def load_personal_connection_config(workspace: Path) -> PersonalConnectionConfig:
    path = workspace_personal_config_path(workspace)
    if not path.exists():
        return PersonalConnectionConfig()

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return PersonalConnectionConfig()
    if not isinstance(payload, dict):
        return PersonalConnectionConfig()

    provider_plan = compile_provider_connection_plan(payload)

    model_payload = payload.get("model")
    executor_payload = payload.get("executor")

    legacy_model_command = _read_token(model_payload, "command")
    legacy_mode = _read_token(executor_payload, "mode")
    legacy_cli_command = _read_token(executor_payload, "cli_command")
    legacy_sdep_command = _read_token(executor_payload, "sdep_command")

    model_command, model_command_source = _resolve_with_legacy_override(
        legacy_value=legacy_model_command,
        provider_value=provider_plan.model_command,
        provider_source=provider_plan.model_command_source,
        legacy_source="legacy.model.command",
    )
    executor_mode, executor_mode_source = _resolve_with_legacy_override(
        legacy_value=legacy_mode,
        provider_value=provider_plan.executor_mode,
        provider_source=provider_plan.executor_mode_source,
        legacy_source="legacy.executor.mode",
    )
    cli_command, cli_command_source = _resolve_with_legacy_override(
        legacy_value=legacy_cli_command,
        provider_value=provider_plan.cli_command,
        provider_source=provider_plan.cli_command_source,
        legacy_source="legacy.executor.cli_command",
    )
    sdep_command, sdep_command_source = _resolve_with_legacy_override(
        legacy_value=legacy_sdep_command,
        provider_value=provider_plan.sdep_command,
        provider_source=provider_plan.sdep_command_source,
        legacy_source="legacy.executor.sdep_command",
    )

    schema_version = _read_token(payload, "schema_version") or provider_plan.schema_version
    if schema_version == CONNECTION_SCHEMA_VERSION_V1:
        schema_version = CONNECTION_SCHEMA_VERSION_V1
    elif not schema_version:
        schema_version = ""

    return PersonalConnectionConfig(
        schema_version=schema_version,
        model_provider=provider_plan.model_provider,
        model_name=provider_plan.model_name,
        model_api_key_env=provider_plan.model_api_key_env,
        model_base_url=provider_plan.model_base_url,
        model_command=model_command,
        model_command_source=model_command_source,
        agent_provider=provider_plan.agent_provider,
        agent_mode=provider_plan.agent_mode,
        agent_auth_env=provider_plan.agent_auth_env,
        agent_endpoint=provider_plan.agent_endpoint,
        executor_mode=executor_mode,
        executor_mode_source=executor_mode_source,
        cli_command=cli_command,
        cli_command_source=cli_command_source,
        sdep_command=sdep_command,
        sdep_command_source=sdep_command_source,
    )


def _read_token(payload: Any, key: str) -> str | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    if value is None:
        return None
    token = str(value).strip()
    return token or None


def _resolve_with_legacy_override(
    *,
    legacy_value: str | None,
    provider_value: str | None,
    provider_source: str,
    legacy_source: str,
) -> tuple[str | None, str]:
    if legacy_value is not None:
        return legacy_value, legacy_source
    if provider_value is not None:
        return provider_value, provider_source
    return None, ""
