from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spice.executors import SDEPExecutor
from spice_personal.executors.factory import (
    PersonalExecutorConfig,
    _build_cli_profile,
    _split_command,
    build_executor,
)
from spice_personal.profile.contract import (
    EXECUTOR_MODE_CLI,
    EXECUTOR_MODE_MOCK,
    EXECUTOR_MODE_SDEP,
    P0_CATEGORIES,
    P1_CATEGORIES,
    PROFILE_SCHEMA_VERSION,
    VALID_EXECUTOR_MODES,
)
from spice_personal.profile.loader import profile_fingerprint


@dataclass(slots=True)
class ProfileValidationIssue:
    severity: str
    code: str
    message: str
    category: str = ""
    field_path: str = ""
    expected_operation: str = ""
    actual_available_capabilities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "category": self.category,
            "field_path": self.field_path,
            "expected_operation": self.expected_operation,
            "actual_available_capabilities": list(self.actual_available_capabilities),
        }


@dataclass(slots=True)
class ProfileValidationResult:
    profile_path: Path
    fingerprint: str
    executor_mode: str
    validated_at: str
    errors: list[ProfileValidationIssue] = field(default_factory=list)
    warnings: list[ProfileValidationIssue] = field(default_factory=list)
    available_capabilities: list[str] = field(default_factory=list)

    def has_errors(self) -> bool:
        return bool(self.errors)

    def add_error(
        self,
        *,
        code: str,
        message: str,
        category: str = "",
        field_path: str = "",
        expected_operation: str = "",
        actual_available_capabilities: list[str] | None = None,
    ) -> None:
        self.errors.append(
            ProfileValidationIssue(
                severity="error",
                code=code,
                message=message,
                category=category,
                field_path=field_path,
                expected_operation=expected_operation,
                actual_available_capabilities=list(actual_available_capabilities or []),
            )
        )

    def add_warning(
        self,
        *,
        code: str,
        message: str,
        category: str = "",
        field_path: str = "",
        expected_operation: str = "",
        actual_available_capabilities: list[str] | None = None,
    ) -> None:
        self.warnings.append(
            ProfileValidationIssue(
                severity="warning",
                code=code,
                message=message,
                category=category,
                field_path=field_path,
                expected_operation=expected_operation,
                actual_available_capabilities=list(actual_available_capabilities or []),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_path": str(self.profile_path),
            "fingerprint": self.fingerprint,
            "executor_mode": self.executor_mode,
            "validated_at": self.validated_at,
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
            "available_capabilities": list(self.available_capabilities),
        }

    def raise_for_errors(self) -> None:
        if not self.errors:
            return
        lines = ["Profile validation failed:"]
        for issue in self.errors:
            lines.append(
                "- [{code}] {message} (category={category}, expected_operation={expected}, "
                "field={field}, profile={profile})".format(
                    code=issue.code,
                    message=issue.message,
                    category=issue.category or "-",
                    expected=issue.expected_operation or "-",
                    field=issue.field_path or "-",
                    profile=self.profile_path,
                )
            )
            if issue.actual_available_capabilities:
                lines.append(
                    "  available_capabilities={caps}".format(
                        caps=",".join(issue.actual_available_capabilities),
                    )
                )
        raise RuntimeError("\n".join(lines))


def validate_profile_contract(
    profile: dict[str, Any],
    *,
    profile_path: Path,
    executor_config: PersonalExecutorConfig,
) -> ProfileValidationResult:
    executor_mode = _as_text(profile.get("executor_mode")).lower() or EXECUTOR_MODE_MOCK
    result = ProfileValidationResult(
        profile_path=profile_path,
        fingerprint=profile_fingerprint(profile_path),
        executor_mode=executor_mode,
        validated_at=datetime.now(timezone.utc).isoformat(),
    )
    _validate_schema(profile, result)
    _validate_category_coverage(profile, result)
    if result.has_errors():
        return result
    _validate_executor_compatibility(profile, executor_config=executor_config, result=result)
    return result


def _validate_schema(profile: dict[str, Any], result: ProfileValidationResult) -> None:
    if _as_text(profile.get("schema_version")) != PROFILE_SCHEMA_VERSION:
        result.add_error(
            code="profile.schema_version.invalid",
            message=f"schema_version must be {PROFILE_SCHEMA_VERSION!r}",
            field_path="schema_version",
        )
    if not _as_text(profile.get("profile_id")):
        result.add_error(
            code="profile.profile_id.required",
            message="profile_id is required.",
            field_path="profile_id",
        )

    executor_mode = _as_text(profile.get("executor_mode")).lower()
    if executor_mode not in VALID_EXECUTOR_MODES:
        result.add_error(
            code="profile.executor_mode.invalid",
            message=f"executor_mode must be one of {list(VALID_EXECUTOR_MODES)!r}.",
            field_path="executor_mode",
        )

    routes = profile.get("category_routes")
    if not isinstance(routes, dict):
        result.add_error(
            code="profile.category_routes.invalid",
            message="category_routes must be an object.",
            field_path="category_routes",
        )
        return

    for category, route in routes.items():
        route_path = f"category_routes.{category}"
        is_p1 = category in P1_CATEGORIES
        if not isinstance(route, dict):
            _add_schema_issue(
                result,
                is_p1=is_p1,
                code="profile.route.invalid",
                message="category route must be an object.",
                category=category,
                field_path=route_path,
            )
            continue

        if not isinstance(route.get("enabled"), bool):
            _add_schema_issue(
                result,
                is_p1=is_p1,
                code="profile.route.enabled.invalid",
                message="enabled must be a boolean.",
                category=category,
                field_path=f"{route_path}.enabled",
            )
        if not _as_text(route.get("operation_name")):
            _add_schema_issue(
                result,
                is_p1=is_p1,
                code="profile.route.operation_name.required",
                message="operation_name is required.",
                category=category,
                field_path=f"{route_path}.operation_name",
            )
        _validate_target_field(
            route.get("target"),
            route_path=f"{route_path}.target",
            category=category,
            is_p1=is_p1,
            result=result,
        )

        fallback_cli = route.get("fallback_cli")
        if fallback_cli is None:
            continue
        if not isinstance(fallback_cli, dict):
            _add_schema_issue(
                result,
                is_p1=is_p1,
                code="profile.route.fallback_cli.invalid",
                message="fallback_cli must be an object when provided.",
                category=category,
                field_path=f"{route_path}.fallback_cli",
            )
            continue
        if not _as_text(fallback_cli.get("operation_name")):
            _add_schema_issue(
                result,
                is_p1=is_p1,
                code="profile.route.fallback_cli.operation_name.required",
                message="fallback_cli.operation_name is required.",
                category=category,
                field_path=f"{route_path}.fallback_cli.operation_name",
            )
        _validate_target_field(
            fallback_cli.get("target"),
            route_path=f"{route_path}.fallback_cli.target",
            category=category,
            is_p1=is_p1,
            result=result,
        )


def _validate_target_field(
    target: Any,
    *,
    route_path: str,
    category: str,
    is_p1: bool,
    result: ProfileValidationResult,
) -> None:
    if not isinstance(target, dict):
        _add_schema_issue(
            result,
            is_p1=is_p1,
            code="profile.route.target.invalid",
            message="target must be an object.",
            category=category,
            field_path=route_path,
        )
        return
    if not _as_text(target.get("kind")):
        _add_schema_issue(
            result,
            is_p1=is_p1,
            code="profile.route.target.kind.required",
            message="target.kind is required.",
            category=category,
            field_path=f"{route_path}.kind",
        )
    if not _as_text(target.get("id")):
        _add_schema_issue(
            result,
            is_p1=is_p1,
            code="profile.route.target.id.required",
            message="target.id is required.",
            category=category,
            field_path=f"{route_path}.id",
        )


def _add_schema_issue(
    result: ProfileValidationResult,
    *,
    is_p1: bool,
    code: str,
    message: str,
    category: str,
    field_path: str,
) -> None:
    if is_p1:
        result.add_warning(
            code=code,
            message=message,
            category=category,
            field_path=field_path,
        )
        return
    result.add_error(
        code=code,
        message=message,
        category=category,
        field_path=field_path,
    )


def _validate_category_coverage(profile: dict[str, Any], result: ProfileValidationResult) -> None:
    routes = profile.get("category_routes")
    if not isinstance(routes, dict):
        return

    for category in P0_CATEGORIES:
        route = routes.get(category)
        if not isinstance(route, dict):
            result.add_error(
                code="profile.category.missing_p0",
                message=f"P0 category {category!r} is missing.",
                category=category,
                field_path=f"category_routes.{category}",
            )
            continue
        if not bool(route.get("enabled")):
            result.add_error(
                code="profile.category.disabled_p0",
                message=f"P0 category {category!r} must be enabled.",
                category=category,
                field_path=f"category_routes.{category}.enabled",
            )

    for category in P1_CATEGORIES:
        route = routes.get(category)
        if route is None:
            result.add_warning(
                code="profile.category.missing_p1",
                message=f"P1 category {category!r} is missing.",
                category=category,
                field_path=f"category_routes.{category}",
            )
            continue
        if isinstance(route, dict) and not bool(route.get("enabled")):
            result.add_warning(
                code="profile.category.disabled_p1",
                message=f"P1 category {category!r} is disabled.",
                category=category,
                field_path=f"category_routes.{category}.enabled",
            )


def _validate_executor_compatibility(
    profile: dict[str, Any],
    *,
    executor_config: PersonalExecutorConfig,
    result: ProfileValidationResult,
) -> None:
    routes = profile.get("category_routes")
    if not isinstance(routes, dict):
        return

    mode = result.executor_mode
    if mode == EXECUTOR_MODE_MOCK:
        return
    if mode == EXECUTOR_MODE_CLI:
        _validate_cli_mode(routes=routes, executor_config=executor_config, result=result)
        return
    if mode == EXECUTOR_MODE_SDEP:
        _validate_sdep_mode(routes=routes, executor_config=executor_config, result=result)
        return


def _validate_cli_mode(
    *,
    routes: dict[str, Any],
    executor_config: PersonalExecutorConfig,
    result: ProfileValidationResult,
) -> None:
    cli_config = PersonalExecutorConfig(
        mode=EXECUTOR_MODE_CLI,
        timeout_seconds=executor_config.timeout_seconds,
        cli_profile=executor_config.cli_profile,
        cli_profile_path=executor_config.cli_profile_path,
        cli_command=executor_config.cli_command,
        cli_parser_mode=executor_config.cli_parser_mode,
        sdep_command=executor_config.sdep_command,
    )
    try:
        cli_profile = _build_cli_profile(cli_config)
    except Exception as exc:
        result.add_error(
            code="profile.cli.profile.invalid",
            message=f"CLI profile build failed: {exc}",
            field_path="executor_mode=cli",
        )
        return

    command_error = _validate_cli_command_exists(cli_config)
    if command_error:
        result.add_error(
            code="profile.cli.command.missing",
            message=command_error,
            field_path="cli_command",
        )

    supported_actions = set(cli_profile.action_mappings.keys())
    for category in P0_CATEGORIES:
        route = routes.get(category)
        if not isinstance(route, dict) or not bool(route.get("enabled")):
            continue
        operation = _as_text(route.get("operation_name"))
        if operation in supported_actions:
            continue
        result.add_error(
            code="profile.cli.mapping.missing_p0",
            message=f"CLI mapping missing for operation {operation!r}.",
            category=category,
            field_path=f"category_routes.{category}.operation_name",
            expected_operation=operation,
        )

    for category in P1_CATEGORIES:
        route = routes.get(category)
        if not isinstance(route, dict) or not bool(route.get("enabled")):
            continue
        operation = _as_text(route.get("operation_name"))
        if operation in supported_actions:
            continue
        result.add_warning(
            code="profile.cli.mapping.missing_p1",
            message=f"CLI mapping missing for P1 operation {operation!r}.",
            category=category,
            field_path=f"category_routes.{category}.operation_name",
            expected_operation=operation,
        )


def _validate_sdep_mode(
    *,
    routes: dict[str, Any],
    executor_config: PersonalExecutorConfig,
    result: ProfileValidationResult,
) -> None:
    sdep_config = PersonalExecutorConfig(
        mode=EXECUTOR_MODE_SDEP,
        timeout_seconds=executor_config.timeout_seconds,
        cli_profile=executor_config.cli_profile,
        cli_profile_path=executor_config.cli_profile_path,
        cli_command=executor_config.cli_command,
        cli_parser_mode=executor_config.cli_parser_mode,
        sdep_command=executor_config.sdep_command,
    )
    try:
        executor = build_executor(sdep_config)
    except Exception as exc:
        result.add_error(
            code="profile.sdep.executor.invalid",
            message=f"SDEP executor build failed: {exc}",
            field_path="executor_mode=sdep",
        )
        return

    if not isinstance(executor, SDEPExecutor):
        result.add_error(
            code="profile.sdep.executor.type",
            message="Configured sdep executor is not an SDEPExecutor instance.",
            field_path="executor_mode=sdep",
        )
        return

    expected_action_types = _collect_expected_action_types(routes)
    available_capabilities: list[str] = []
    describe_error: str | None = None
    try:
        payload = executor.describe(action_types=sorted(expected_action_types))
        available_capabilities = _extract_available_capabilities(payload)
    except Exception as exc:
        describe_error = str(exc)

    result.available_capabilities = list(available_capabilities)
    available_set = set(available_capabilities)
    if describe_error:
        for category in P0_CATEGORIES:
            route = routes.get(category)
            if not isinstance(route, dict) or not bool(route.get("enabled")):
                continue
            if _route_has_cli_fallback(route):
                fallback_operation = _as_text(route.get("fallback_cli", {}).get("operation_name"))
                if _validate_cli_fallback_operation(
                    fallback_operation,
                    executor_config=executor_config,
                ):
                    result.add_warning(
                        code="profile.sdep.describe.failed_cli_fallback",
                        message=(
                            f"SDEP describe failed ({describe_error}); CLI fallback route will be used."
                        ),
                        category=category,
                        field_path=f"category_routes.{category}.fallback_cli.operation_name",
                        expected_operation=fallback_operation,
                    )
                    continue
            result.add_error(
                code="profile.sdep.describe.failed",
                message=f"SDEP describe failed and no usable CLI fallback: {describe_error}",
                category=category,
                field_path=f"category_routes.{category}",
                expected_operation=_as_text(route.get("operation_name")),
            )
        return

    for category in P0_CATEGORIES:
        route = routes.get(category)
        if not isinstance(route, dict) or not bool(route.get("enabled")):
            continue
        expected = _route_expected_capabilities(route)
        if available_set.intersection(expected):
            continue

        if _route_has_cli_fallback(route):
            fallback_operation = _as_text(route.get("fallback_cli", {}).get("operation_name"))
            if _validate_cli_fallback_operation(
                fallback_operation,
                executor_config=executor_config,
            ):
                result.add_warning(
                    code="profile.sdep.capability.missing_cli_fallback",
                    message="SDEP capability missing; explicit CLI fallback will be used.",
                    category=category,
                    field_path=f"category_routes.{category}.fallback_cli.operation_name",
                    expected_operation=fallback_operation,
                    actual_available_capabilities=available_capabilities,
                )
                continue

        result.add_error(
            code="profile.sdep.capability.missing_p0",
            message="Required SDEP capability is missing.",
            category=category,
            field_path=f"category_routes.{category}.required_capabilities",
            expected_operation="|".join(sorted(expected)),
            actual_available_capabilities=available_capabilities,
        )

    for category in P1_CATEGORIES:
        route = routes.get(category)
        if not isinstance(route, dict) or not bool(route.get("enabled")):
            continue
        expected = _route_expected_capabilities(route)
        if available_set.intersection(expected):
            continue
        result.add_warning(
            code="profile.sdep.capability.missing_p1",
            message="P1 SDEP capability is missing.",
            category=category,
            field_path=f"category_routes.{category}.required_capabilities",
            expected_operation="|".join(sorted(expected)),
            actual_available_capabilities=available_capabilities,
        )


def _collect_expected_action_types(routes: dict[str, Any]) -> set[str]:
    expected: set[str] = set()
    for category in (*P0_CATEGORIES, *P1_CATEGORIES):
        route = routes.get(category)
        if not isinstance(route, dict) or not bool(route.get("enabled")):
            continue
        expected.update(_route_expected_capabilities(route))
    return expected


def _route_expected_capabilities(route: dict[str, Any]) -> set[str]:
    required = route.get("required_capabilities")
    if isinstance(required, list):
        normalized = {_as_text(item) for item in required if _as_text(item)}
        if normalized:
            return normalized
    operation = _as_text(route.get("operation_name"))
    return {operation} if operation else set()


def _extract_available_capabilities(payload: dict[str, Any]) -> list[str]:
    description = payload.get("description")
    if not isinstance(description, dict):
        return []
    capabilities = description.get("capabilities")
    if not isinstance(capabilities, list):
        return []
    available: list[str] = []
    for entry in capabilities:
        if not isinstance(entry, dict):
            continue
        action_type = _as_text(entry.get("action_type"))
        if action_type:
            available.append(action_type)
    return sorted(set(available))


def _route_has_cli_fallback(route: dict[str, Any]) -> bool:
    fallback = route.get("fallback_cli")
    if not isinstance(fallback, dict):
        return False
    return bool(_as_text(fallback.get("operation_name")))


def _validate_cli_fallback_operation(
    operation_name: str,
    *,
    executor_config: PersonalExecutorConfig,
) -> bool:
    if not operation_name:
        return False
    cli_config = PersonalExecutorConfig(
        mode=EXECUTOR_MODE_CLI,
        timeout_seconds=executor_config.timeout_seconds,
        cli_profile=executor_config.cli_profile,
        cli_profile_path=executor_config.cli_profile_path,
        cli_command=executor_config.cli_command,
        cli_parser_mode=executor_config.cli_parser_mode,
        sdep_command=executor_config.sdep_command,
    )
    try:
        cli_profile = _build_cli_profile(cli_config)
    except Exception:
        return False
    if operation_name not in cli_profile.action_mappings:
        return False
    return _validate_cli_command_exists(cli_config) is None


def _validate_cli_command_exists(config: PersonalExecutorConfig) -> str | None:
    if config.cli_profile_path:
        try:
            payload = json.loads(Path(config.cli_profile_path).read_text(encoding="utf-8"))
        except Exception as exc:
            return f"Could not read cli_profile_path: {exc}"
        if not isinstance(payload, dict):
            return "cli_profile_path payload must be an object."
        actions = payload.get("actions")
        if not isinstance(actions, dict):
            return "cli_profile_path payload requires object field `actions`."
        fallback = _split_command(config.cli_command)
        for action_type, action_payload in actions.items():
            if not isinstance(action_payload, dict):
                continue
            command_raw = _as_text(action_payload.get("command"))
            argv = _split_command(command_raw) if command_raw else list(fallback)
            if not argv:
                return f"Action {action_type!r} has no command and no fallback cli_command."
            if not _command_exists(argv[0]):
                return f"Command not found for action {action_type!r}: {argv[0]!r}"
        return None

    argv = _split_command(config.cli_command)
    if not argv:
        return "cli_command is required for built-in CLI profile."
    if not _command_exists(argv[0]):
        return f"Command not found: {argv[0]!r}"
    return None


def _command_exists(command: str) -> bool:
    token = command.strip()
    if not token:
        return False
    if "/" in token:
        return Path(token).exists()
    return shutil.which(token) is not None


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""
