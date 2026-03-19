from __future__ import annotations

import shlex
import sys
from dataclasses import dataclass
from typing import Any


CONNECTION_SCHEMA_VERSION_V1 = "spice_personal.connection.v1"
MODEL_PROVIDER_OPENROUTER = "openrouter"
MODEL_PROVIDER_SUBPROCESS = "subprocess"
AGENT_PROVIDER_OPENCLAW = "openclaw"
AGENT_PROVIDER_CODEX = "codex"
AGENT_PROVIDER_CLAUDE_CODE = "claude_code"
AGENT_PROVIDER_GENERIC_SDEP = "generic_sdep"
AGENT_PROVIDER_GENERIC_CLI = "generic_cli"
SUPPORTED_MODEL_PROVIDERS = {
    MODEL_PROVIDER_OPENROUTER,
    MODEL_PROVIDER_SUBPROCESS,
}
SUPPORTED_AGENT_PROVIDERS = {
    AGENT_PROVIDER_OPENCLAW,
    AGENT_PROVIDER_CODEX,
    AGENT_PROVIDER_CLAUDE_CODE,
    AGENT_PROVIDER_GENERIC_SDEP,
    AGENT_PROVIDER_GENERIC_CLI,
}


@dataclass(slots=True)
class ProviderConnectionPlan:
    schema_version: str = ""
    model_provider: str | None = None
    model_name: str | None = None
    model_api_key_env: str | None = None
    model_command: str | None = None
    model_command_source: str = ""
    agent_provider: str | None = None
    agent_mode: str | None = None
    agent_auth_env: str | None = None
    agent_endpoint: str | None = None
    executor_mode: str | None = None
    cli_command: str | None = None
    sdep_command: str | None = None
    executor_mode_source: str = ""
    cli_command_source: str = ""
    sdep_command_source: str = ""


def compile_provider_connection_plan(payload: dict[str, Any]) -> ProviderConnectionPlan:
    if not isinstance(payload, dict):
        return ProviderConnectionPlan()

    schema_version = _as_token(payload.get("schema_version")) or ""
    model_payload = payload.get("model")
    agent_payload = payload.get("agent")

    model_provider = _as_token(_read_field(model_payload, "provider"))
    model_name = _as_token(_read_field(model_payload, "model"))
    model_api_key_env = _as_token(_read_field(model_payload, "api_key_env"))

    model_command: str | None = None
    if model_provider == MODEL_PROVIDER_OPENROUTER:
        if model_name:
            env_name = model_api_key_env or "OPENROUTER_API_KEY"
            model_api_key_env = env_name
            model_command = _join_command(
                [
                    sys.executable,
                    "-m",
                    "spice_personal.wrappers.openrouter_model",
                    "--model",
                    model_name,
                    "--api-key-env",
                    env_name,
                ]
            )
    elif model_provider == MODEL_PROVIDER_SUBPROCESS:
        model_command = _as_token(_read_field(model_payload, "provider_command"))

    model_source = "provider" if model_command else ""

    agent_provider = _as_token(_read_field(agent_payload, "provider"))
    agent_mode = _normalize_mode(_read_field(agent_payload, "mode"))
    agent_auth_env = _as_token(_read_field(agent_payload, "auth_env"))
    agent_endpoint = _as_token(_read_field(agent_payload, "endpoint"))
    provider_command = _as_token(_read_field(agent_payload, "provider_command"))

    executor_mode: str | None = None
    cli_command: str | None = None
    sdep_command: str | None = None
    executor_mode_source = ""
    cli_command_source = ""
    sdep_command_source = ""

    if agent_provider == AGENT_PROVIDER_CLAUDE_CODE:
        executor_mode = "sdep"
        resolved_auth_env = agent_auth_env or "ANTHROPIC_API_KEY"
        agent_auth_env = resolved_auth_env
        sdep_command = provider_command or _build_claude_code_agent_command(
            auth_env=resolved_auth_env,
            endpoint=agent_endpoint,
        )
    elif agent_provider == AGENT_PROVIDER_CODEX:
        executor_mode = "sdep"
        resolved_auth_env = agent_auth_env or "OPENAI_API_KEY"
        agent_auth_env = resolved_auth_env
        sdep_command = provider_command or _build_codex_agent_command(
            auth_env=resolved_auth_env,
            endpoint=agent_endpoint,
        )
    elif agent_provider == AGENT_PROVIDER_OPENCLAW:
        executor_mode = "sdep"
        sdep_command = provider_command or _build_sdep_provider_bridge_command(
            provider=AGENT_PROVIDER_OPENCLAW,
            auth_env=agent_auth_env,
            endpoint=agent_endpoint,
        )
    elif agent_provider == AGENT_PROVIDER_GENERIC_SDEP:
        executor_mode = "sdep"
        sdep_command = provider_command or _build_sdep_provider_bridge_command(
            provider=AGENT_PROVIDER_GENERIC_SDEP,
            auth_env=agent_auth_env,
            endpoint=agent_endpoint,
        )
    elif agent_provider == AGENT_PROVIDER_GENERIC_CLI:
        executor_mode = "cli"
        cli_command = provider_command or _build_cli_provider_bridge_command(
            provider=AGENT_PROVIDER_GENERIC_CLI,
            auth_env=agent_auth_env,
            endpoint=agent_endpoint,
        )
    elif agent_mode in {"cli", "sdep"}:
        executor_mode = agent_mode
        if agent_mode == "cli":
            cli_command = provider_command
        if agent_mode == "sdep":
            sdep_command = provider_command

    if executor_mode:
        executor_mode_source = "provider"
    if cli_command:
        cli_command_source = "provider"
    if sdep_command:
        sdep_command_source = "provider"

    return ProviderConnectionPlan(
        schema_version=schema_version,
        model_provider=model_provider,
        model_name=model_name,
        model_api_key_env=model_api_key_env,
        model_command=model_command,
        model_command_source=model_source,
        agent_provider=agent_provider,
        agent_mode=agent_mode,
        agent_auth_env=agent_auth_env,
        agent_endpoint=agent_endpoint,
        executor_mode=executor_mode,
        cli_command=cli_command,
        sdep_command=sdep_command,
        executor_mode_source=executor_mode_source,
        cli_command_source=cli_command_source,
        sdep_command_source=sdep_command_source,
    )


def _build_sdep_provider_bridge_command(
    *,
    provider: str,
    auth_env: str | None,
    endpoint: str | None,
) -> str:
    parts = [
        sys.executable,
        "-m",
        "spice_personal.provider_bridges.sdep_agent_provider_bridge",
        "--provider",
        provider,
    ]
    if auth_env:
        parts.extend(["--auth-env", auth_env])
    if endpoint:
        parts.extend(["--endpoint", endpoint])
    return _join_command(parts)


def _build_codex_agent_command(
    *,
    auth_env: str,
    endpoint: str | None,
) -> str:
    parts = [
        sys.executable,
        "-m",
        "spice_personal.wrappers.codex_agent",
        "--auth-env",
        auth_env,
    ]
    if endpoint:
        parts.extend(["--endpoint", endpoint])
    return _join_command(parts)


def _build_claude_code_agent_command(
    *,
    auth_env: str,
    endpoint: str | None,
) -> str:
    parts = [
        sys.executable,
        "-m",
        "spice_personal.wrappers.claude_code_agent",
        "--auth-env",
        auth_env,
    ]
    if endpoint:
        parts.extend(["--endpoint", endpoint])
    return _join_command(parts)


def _build_cli_provider_bridge_command(
    *,
    provider: str,
    auth_env: str | None,
    endpoint: str | None,
) -> str:
    parts = [
        sys.executable,
        "-m",
        "spice_personal.provider_bridges.cli_agent_provider_bridge",
        "--provider",
        provider,
    ]
    if auth_env:
        parts.extend(["--auth-env", auth_env])
    if endpoint:
        parts.extend(["--endpoint", endpoint])
    return _join_command(parts)


def _join_command(parts: list[str]) -> str:
    tokens = [str(item).strip() for item in parts if str(item).strip()]
    if not tokens:
        return ""
    return shlex.join(tokens)


def _read_field(payload: Any, key: str) -> Any:
    if not isinstance(payload, dict):
        return None
    return payload.get(key)


def _as_token(value: Any) -> str | None:
    if value is None:
        return None
    token = str(value).strip()
    return token or None


def _normalize_mode(value: Any) -> str | None:
    token = _as_token(value)
    if token is None:
        return None
    normalized = token.lower()
    if normalized in {"mock", "cli", "sdep"}:
        return normalized
    return None
