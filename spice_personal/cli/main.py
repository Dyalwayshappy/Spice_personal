from __future__ import annotations

import argparse
import sys
from pathlib import Path

from spice_personal.app.personal import (
    MAX_CONTEXT_FILE_BYTES,
    MAX_CONTEXT_TEXT_CHARS,
    PERSONAL_DEFAULT_WORKSPACE,
    CONNECTION_STATE_SETUP_REQUIRED,
    validate_personal_context_inputs,
    run_personal_ask,
    run_personal_init,
    run_personal_session,
)
from spice_personal.config.personal_config import load_personal_connection_config
from spice_personal.config.settings import build_executor_config_from_sources
from spice_personal.wrappers.errors import WrapperIntegrationError, format_wrapper_error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spice-personal",
        description="Personal advisor product layer built on SPICE core.",
    )
    subparsers = parser.add_subparsers(dest="command")

    init_parser = subparsers.add_parser(
        "init",
        help="Initialize personal advisor workspace scaffold.",
    )
    init_parser.add_argument(
        "--workspace",
        type=Path,
        default=PERSONAL_DEFAULT_WORKSPACE,
        help="Personal workspace directory (default: .spice/personal).",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing workspace directory.",
    )
    init_parser.set_defaults(handler=_handle_init)

    ask_parser = subparsers.add_parser(
        "ask",
        help="Run one personal advisor question (outcome/reflection in-memory, no state persisted to disk).",
    )
    ask_parser.add_argument("question", help="Question for the advisor.")
    _add_runtime_args(ask_parser)
    ask_parser.add_argument(
        "--context-text",
        type=str,
        default=None,
        help=f"Optional context text ingested before decision (max {MAX_CONTEXT_TEXT_CHARS} chars).",
    )
    ask_parser.add_argument(
        "--context-file",
        type=Path,
        default=None,
        help=f"Optional context file ingested before decision (single file, max {MAX_CONTEXT_FILE_BYTES} bytes).",
    )
    ask_parser.set_defaults(handler=_handle_ask)

    session_parser = subparsers.add_parser(
        "session",
        help="Start interactive personal advisor session.",
    )
    _add_runtime_args(session_parser)
    session_parser.set_defaults(handler=_handle_session)

    return parser


def main(argv: list[str] | None = None) -> int:
    args_list = list(argv) if argv is not None else list(sys.argv[1:])
    if not args_list or args_list[0].startswith("-"):
        args_list = ["session", *args_list]

    parser = build_parser()
    args = parser.parse_args(args_list)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 2
    return int(handler(args))


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workspace",
        type=Path,
        default=PERSONAL_DEFAULT_WORKSPACE,
        help="Personal workspace directory (default: .spice/personal).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Advisor model override for personal LLM decision/simulation. "
            "If omitted, SPICE personal resolves model command from env/config."
        ),
    )
    parser.add_argument(
        "--executor",
        type=str,
        choices=("mock", "cli", "sdep"),
        default=None,
        help="Executor mode for evidence gathering (default from env, fallback: mock).",
    )
    parser.add_argument(
        "--executor-timeout",
        type=float,
        default=None,
        help="Executor timeout seconds for CLI/SDEP modes.",
    )
    parser.add_argument(
        "--cli-profile",
        type=str,
        default=None,
        help="Built-in CLI profile name (default|text).",
    )
    parser.add_argument(
        "--cli-profile-path",
        type=Path,
        default=None,
        help="Path to custom CLI profile JSON.",
    )
    parser.add_argument(
        "--cli-command",
        type=str,
        default=None,
        help=(
            "Command for CLI executor (example: \"python3 agent.py\"). "
            "Used by built-in profile or as fallback for profile-path actions missing command."
        ),
    )
    parser.add_argument(
        "--cli-parser-mode",
        type=str,
        choices=("json", "text"),
        default=None,
        help="Parser mode for built-in CLI profile.",
    )
    parser.add_argument(
        "--sdep-command",
        type=str,
        default=None,
        help="Subprocess command for SDEP executor (example: \"python3 examples/sdep_agent_demo/echo_agent.py\").",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show debug fields (actions, ids, option metrics) instead of only user-facing report output.",
    )


def _handle_init(args: argparse.Namespace) -> int:
    workspace: Path = args.workspace
    force = bool(args.force)
    try:
        output_dir = run_personal_init(
            workspace=workspace,
            force=force,
        )
        print("Personal init complete.")
        print(f"Workspace: {output_dir}")
        print()
        print("Created:")
        print("- personal.config.json (configure model / agent / executor here)")
        print()
        print("Next step:")
        print('1) export OPENROUTER_API_KEY="your_key"')
        print('2) spice-personal ask "What should I do next?"')
        print()
        print("Optional:")
        print("- configure agent later in personal.config.json")
        return 0
    except Exception as exc:
        print(f"personal init failed: {exc}", file=sys.stderr)
        return 1


def _handle_ask(args: argparse.Namespace) -> int:
    workspace: Path = args.workspace
    model = args.model
    verbose = bool(getattr(args, "verbose", False))
    question = str(args.question)
    context_text = args.context_text
    context_file = args.context_file
    workspace_config = load_personal_connection_config(workspace)
    executor_config = build_executor_config_from_sources(
        args,
        workspace_config=workspace_config,
    )

    try:
        validate_personal_context_inputs(
            context_text=context_text,
            context_file=context_file,
        )
        result = run_personal_ask(
            question=question,
            workspace=workspace,
            model=model,
            executor_config=executor_config,
            context_text=context_text,
            context_file=context_file,
        )
        if result.auto_initialized:
            print(f"Initialized workspace: {workspace}")
        if result.connection_state == CONNECTION_STATE_SETUP_REQUIRED:
            if result.onboarding_card:
                print(result.onboarding_card)
            return 0
        if verbose and model:
            print(
                "Model override accepted for personal LLM decision/simulation: "
                f"{model}"
            )
        if verbose:
            print(f"executor_mode={executor_config.mode}")
            if result.evidence_notice:
                print(f"notice={result.evidence_notice}")
            print(f"action={result.advice.selected_action}")
            print(f"result_kind={result.result_kind}")
            print(f"decision_adoption_status={result.decision_adoption_status}")
            print(f"execution_status={result.execution_status}")
            print(f"suggestion={result.advice.suggestion}")
            print(f"urgency={result.advice.urgency}")
            print(f"confidence={result.advice.confidence:.2f}")
            if result.decision_options:
                print(f"decision_options_count={len(result.decision_options)}")
                if result.recommended_option_id:
                    print(f"recommended_option_id={result.recommended_option_id}")
                for index, option in enumerate(result.decision_options, start=1):
                    print(f"option_{index}_id={_option_text(option, 'candidate_id')}")
                    print(f"option_{index}_action={_option_text(option, 'action')}")
                    print(f"option_{index}_score={_option_float(option, 'score'):.2f}")
                    print(f"option_{index}_confidence={_option_float(option, 'confidence'):.2f}")
                    print(f"option_{index}_urgency={_option_text(option, 'urgency')}")
                    print(f"option_{index}_risk={_option_float(option, 'risk'):.2f}")
                    print(f"option_{index}_suggestion={_option_text(option, 'suggestion_text')}")
                    print(f"option_{index}_rationale={_option_text(option, 'simulation_rationale')}")
                    print(f"option_{index}_benefits={_option_joined_list(option, 'benefits')}")
                    print(f"option_{index}_risks={_option_joined_list(option, 'risks')}")
                    print(
                        f"option_{index}_key_assumptions={_option_joined_list(option, 'key_assumptions')}"
                    )
                    print(f"option_{index}_first_step_24h={_option_text(option, 'first_step_24h')}")
                    print(f"option_{index}_stop_loss_trigger={_option_text(option, 'stop_loss_trigger')}")
                    print(
                        f"option_{index}_change_mind_condition={_option_text(option, 'change_mind_condition')}"
                    )
        else:
            if result.evidence_notice:
                print(result.evidence_notice)
            suggestion = str(result.advice.suggestion).strip()
            if suggestion:
                print(suggestion)
        return 0
    except WrapperIntegrationError as exc:
        print(f"personal ask failed: {format_wrapper_error(exc)}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"personal ask failed: {exc}", file=sys.stderr)
        return 1


def _option_text(option: dict[str, object], key: str) -> str:
    value = option.get(key)
    if value is None:
        return ""
    return str(value).strip()


def _option_float(option: dict[str, object], key: str) -> float:
    value = option.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _option_joined_list(option: dict[str, object], key: str) -> str:
    value = option.get(key)
    if not isinstance(value, list):
        return ""
    items: list[str] = []
    for entry in value:
        token = str(entry).strip()
        if token:
            items.append(token)
    return " | ".join(items)


def _handle_session(args: argparse.Namespace) -> int:
    workspace: Path = args.workspace
    model = args.model
    verbose = bool(getattr(args, "verbose", False))
    workspace_config = load_personal_connection_config(workspace)
    executor_config = build_executor_config_from_sources(
        args,
        workspace_config=workspace_config,
    )
    if verbose:
        print(f"executor_mode={executor_config.mode}")
    try:
        return int(
            run_personal_session(
                workspace=workspace,
                model=model,
                executor_config=executor_config,
                input_stream=sys.stdin,
                output_stream=sys.stdout,
                verbose=verbose,
            )
        )
    except WrapperIntegrationError as exc:
        print(f"personal session failed: {format_wrapper_error(exc)}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"personal session failed: {exc}", file=sys.stderr)
        return 1
