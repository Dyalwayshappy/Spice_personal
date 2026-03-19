from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from spice.llm.core.provider import (
    LLMAuthError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTransportError,
)


class WrapperErrorCategory(str, Enum):
    AUTH_CONFIG = "auth_config"
    TRANSPORT_RUNTIME = "transport_runtime"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    RESPONSE_VALIDITY = "response_validity"


@dataclass(slots=True, frozen=True)
class WrapperErrorInfo:
    category: WrapperErrorCategory
    code: str
    message: str
    source: str
    stage: str = ""


class WrapperIntegrationError(RuntimeError):
    def __init__(
        self,
        info: WrapperErrorInfo,
        *,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(info.message)
        self.info = info
        self.cause = cause


def wrap_model_exception(exc: Exception, *, stage: str) -> WrapperIntegrationError:
    message = _normalize_message(exc)
    lowered = message.lower()
    if isinstance(exc, LLMAuthError) or _contains_any(
        lowered,
        ("unauthorized", "forbidden", "api key", "authentication", "auth"),
    ):
        return WrapperIntegrationError(
            WrapperErrorInfo(
                category=WrapperErrorCategory.AUTH_CONFIG,
                code="model.auth_config",
                message=message,
                source="model",
                stage=stage,
            ),
            cause=exc,
        )
    if isinstance(exc, LLMRateLimitError):
        return WrapperIntegrationError(
            WrapperErrorInfo(
                category=WrapperErrorCategory.TRANSPORT_RUNTIME,
                code="model.rate_limit",
                message=message,
                source="model",
                stage=stage,
            ),
            cause=exc,
        )
    if isinstance(exc, LLMTransportError) or _contains_any(
        lowered,
        ("timeout", "timed out", "connection", "network", "transport", "dns"),
    ):
        return WrapperIntegrationError(
            WrapperErrorInfo(
                category=WrapperErrorCategory.TRANSPORT_RUNTIME,
                code="model.transport_runtime",
                message=message,
                source="model",
                stage=stage,
            ),
            cause=exc,
        )
    if isinstance(exc, LLMResponseError) or _contains_any(
        lowered,
        ("json", "parse", "empty stdout", "invalid response", "malformed"),
    ):
        return WrapperIntegrationError(
            WrapperErrorInfo(
                category=WrapperErrorCategory.RESPONSE_VALIDITY,
                code="model.response_validity",
                message=message,
                source="model",
                stage=stage,
            ),
            cause=exc,
        )
    return WrapperIntegrationError(
        WrapperErrorInfo(
            category=WrapperErrorCategory.TRANSPORT_RUNTIME,
            code="model.runtime_error",
            message=message,
            source="model",
            stage=stage,
        ),
        cause=exc,
    )


def wrap_agent_exception(exc: Exception, *, stage: str) -> WrapperIntegrationError:
    message = _normalize_message(exc)
    lowered = message.lower()
    if _contains_any(lowered, ("unauthorized", "forbidden", "api key", "authentication", "auth")):
        category = WrapperErrorCategory.AUTH_CONFIG
        code = "agent.auth_config"
    elif _contains_any(lowered, ("unsupported", "not supported", "capability")):
        category = WrapperErrorCategory.UNSUPPORTED_CAPABILITY
        code = "agent.unsupported_capability"
    elif _contains_any(lowered, ("json", "parse", "invalid", "malformed")):
        category = WrapperErrorCategory.RESPONSE_VALIDITY
        code = "agent.response_validity"
    else:
        category = WrapperErrorCategory.TRANSPORT_RUNTIME
        code = "agent.transport_runtime"

    return WrapperIntegrationError(
        WrapperErrorInfo(
            category=category,
            code=code,
            message=message,
            source="agent",
            stage=stage,
        ),
        cause=exc,
    )


def model_unsupported_capability_error(
    message: str,
    *,
    stage: str,
) -> WrapperIntegrationError:
    return WrapperIntegrationError(
        WrapperErrorInfo(
            category=WrapperErrorCategory.UNSUPPORTED_CAPABILITY,
            code="model.unsupported_capability",
            message=message.strip() or "Model proposed unsupported capability.",
            source="model",
            stage=stage,
        )
    )


def model_response_validity_error(
    message: str,
    *,
    stage: str,
) -> WrapperIntegrationError:
    return WrapperIntegrationError(
        WrapperErrorInfo(
            category=WrapperErrorCategory.RESPONSE_VALIDITY,
            code="model.response_validity",
            message=message.strip() or "Model response was not valid for this stage.",
            source="model",
            stage=stage,
        )
    )


def format_wrapper_error(error: WrapperIntegrationError) -> str:
    info = error.info
    stage = f" stage={info.stage}" if info.stage else ""
    return (
        f"{info.source} integration error "
        f"(category={info.category.value} code={info.code}{stage}): {info.message}"
    )


def _normalize_message(exc: Exception) -> str:
    message = str(exc).strip()
    if message:
        return message
    return exc.__class__.__name__


def _contains_any(value: str, patterns: tuple[str, ...]) -> bool:
    for pattern in patterns:
        if pattern in value:
            return True
    return False
