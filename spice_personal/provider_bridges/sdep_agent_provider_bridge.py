from __future__ import annotations

import argparse
import json
import os
import sys
from uuid import uuid4

from spice.executors.sdep import build_error_response
from spice.protocols import (
    SDEPActionCapability,
    SDEPAgentDescription,
    SDEPDescribeRequest,
    SDEPDescribeResponse,
    SDEPEndpointIdentity,
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
from spice_personal.wrappers.capability_policy import capability_support_level


DEFAULT_CAPABILITIES = (
    "personal.gather_evidence",
    "personal.system",
    "personal.communicate",
    "personal.schedule",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spice-personal-sdep-agent-provider-bridge",
        description="Product-layer SDEP bridge for provider-style agent config.",
    )
    parser.add_argument("--provider", type=str, required=True)
    parser.add_argument("--auth-env", type=str, default="")
    parser.add_argument("--endpoint", type=str, default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    provider = str(args.provider).strip().lower()
    auth_env = str(args.auth_env).strip()
    endpoint = str(args.endpoint).strip()
    responder = SDEPEndpointIdentity(
        id=f"agent.{provider}",
        name=f"{provider} provider bridge",
        version="0.1",
        vendor="SpicePersonal",
        implementation="spice_personal.provider_bridges.sdep_agent_provider_bridge",
        role=SDEP_ROLE_EXECUTOR,
    )

    raw = sys.stdin.read()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        _write(
            build_error_response(
                "",
                responder=responder,
                code="request.invalid",
                message="Invalid JSON payload for SDEP provider bridge.",
            )
        )
        return 0
    if not isinstance(payload, dict):
        _write(
            build_error_response(
                "",
                responder=responder,
                code="request.invalid",
                message="SDEP provider bridge payload must be an object.",
            )
        )
        return 0

    message_type = str(payload.get("message_type", "")).strip()
    if message_type == SDEP_AGENT_DESCRIBE_REQUEST:
        _write(_handle_describe(payload, responder=responder, provider=provider, endpoint=endpoint))
        return 0
    if message_type == SDEP_EXECUTE_REQUEST:
        _write(
            _handle_execute(
                payload,
                responder=responder,
                provider=provider,
                auth_env=auth_env,
                endpoint=endpoint,
            )
        )
        return 0

    _write(
        build_error_response(
            str(payload.get("request_id", "")).strip(),
            responder=responder,
            code="request.invalid",
            message=(
                f"Unsupported message_type {message_type!r}; expected "
                f"{SDEP_EXECUTE_REQUEST!r} or {SDEP_AGENT_DESCRIBE_REQUEST!r}."
            ),
        )
    )
    return 0


def _handle_describe(
    payload: dict[str, object],
    *,
    responder: SDEPEndpointIdentity,
    provider: str,
    endpoint: str,
) -> dict[str, object]:
    request_id = str(payload.get("request_id", "")).strip()
    try:
        request = SDEPDescribeRequest.from_dict(payload)
        request_id = request.request_id
        action_filter = {
            str(item).strip()
            for item in request.query.action_types
            if str(item).strip()
        }
    except Exception:
        action_filter = set()

    capabilities = []
    for action_type in DEFAULT_CAPABILITIES:
        if action_filter and action_type not in action_filter:
            continue
        capabilities.append(
            SDEPActionCapability(
                action_type=action_type,
                target_kinds=["external.service"],
                mode_support=["sync"],
                dry_run_supported=True,
                side_effect_class="",
                outcome_type="observation",
                semantic_inputs=[],
                input_expectation="json_object",
                parameter_expectation="json_object",
                metadata={
                    "provider": provider,
                    "support_level": capability_support_level(action_type),
                },
            )
        )

    response = SDEPDescribeResponse(
        request_id=request_id,
        status="ok",
        responder=responder,
        description=SDEPAgentDescription(
            protocol_support=SDEPProtocolSupport(
                protocol="sdep",
                versions=[SDEP_VERSION],
            ),
            capabilities=capabilities,
            capability_version="provider-bridge-v1",
            summary=f"{provider} provider bridge capabilities",
            metadata={
                "provider": provider,
                "endpoint": endpoint,
            },
        ),
    )
    return response.to_dict()


def _handle_execute(
    payload: dict[str, object],
    *,
    responder: SDEPEndpointIdentity,
    provider: str,
    auth_env: str,
    endpoint: str,
) -> dict[str, object]:
    request_id = str(payload.get("request_id", "")).strip()
    try:
        request = SDEPExecuteRequest.from_dict(payload)
    except Exception as exc:
        return build_error_response(
            request_id,
            responder=responder,
            code="request.invalid",
            message=f"Invalid execute.request payload: {exc}",
        )

    auth_present = bool(auth_env and os.environ.get(auth_env))
    outcome = SDEPExecutionOutcome(
        execution_id=f"exec-{uuid4().hex}",
        status="success",
        outcome_type="observation",
        output={
            "provider": provider,
            "action_type": request.execution.action_type,
            "auth_env": auth_env,
            "auth_present": auth_present,
            "endpoint": endpoint,
            "note": "provider bridge stub execution",
        },
        metadata={"provider_bridge": True},
    )
    response = SDEPExecuteResponse(
        request_id=request.request_id,
        status="success",
        responder=responder,
        outcome=outcome,
        metadata={"provider": provider},
    )
    return response.to_dict()


def _write(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=True))
    sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
