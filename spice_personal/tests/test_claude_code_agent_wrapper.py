from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from spice.executors.sdep_mapping import (
    build_sdep_describe_request,
    build_sdep_execute_request,
)
from spice.protocols import ExecutionIntent, SDEPDescribeResponse, SDEPExecuteResponse
from spice_personal.wrappers import claude_code_agent as claude_wrapper


REPO_ROOT = Path(__file__).resolve().parents[2]


class ClaudeCodeAgentWrapperTests(unittest.TestCase):
    def test_sdep_describe_happy_path_reports_honest_support_levels(self) -> None:
        request = build_sdep_describe_request(
            action_types=[
                "personal.gather_evidence",
                "personal.system",
                "personal.communicate",
                "personal.schedule",
            ]
        )
        completed = self._run_wrapper(
            payload=request.to_dict(),
            env_overrides={"ANTHROPIC_API_KEY": "test-token"},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        response = SDEPDescribeResponse.from_dict(json.loads(completed.stdout))
        self.assertEqual(response.status, "success")

        capability_by_action = {
            capability.action_type: capability
            for capability in response.description.capabilities
        }
        self.assertEqual(
            capability_by_action["personal.gather_evidence"].metadata.get("support_level"),
            "full",
        )
        self.assertEqual(
            capability_by_action["personal.system"].metadata.get("support_level"),
            "full",
        )
        self.assertEqual(
            capability_by_action["personal.communicate"].metadata.get("support_level"),
            "limited",
        )
        self.assertEqual(
            capability_by_action["personal.schedule"].metadata.get("support_level"),
            "limited",
        )
        self.assertEqual(
            capability_by_action["personal.system"].metadata.get("integration_backend"),
            "claude_code.exec",
        )

    def test_execute_happy_path_delegates_to_claude_code_exec(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            root = Path(tmp_dir)
            workspace = root / "workspace"
            workspace.mkdir(parents=True, exist_ok=True)
            fake_cli = root / "fake_claude_code.py"
            fake_cli.write_text(self._fake_claude_code_script(), encoding="utf-8")
            log_path = root / "claude_code_log.json"

            request = self._build_execute_request(
                action_type="personal.system",
                target_id="system",
                input_payload={"task": "inspect workspace state"},
            )
            completed = self._run_wrapper(
                payload=request,
                cwd=workspace,
                env_overrides={
                    "ANTHROPIC_API_KEY": "test-token",
                    "SPICE_AGENT_CLAUDE_CODE_COMMAND": f"{sys.executable} {fake_cli}",
                    "FAKE_CLAUDE_CODE_MODE": "success",
                    "FAKE_CLAUDE_CODE_LOG": str(log_path),
                },
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            response = SDEPExecuteResponse.from_dict(json.loads(completed.stdout))
            self.assertEqual(response.status, "success")
            output = response.outcome.output
            self.assertEqual(output.get("integration_backend"), "claude_code.exec")
            self.assertEqual(output.get("support_level"), "full")
            self.assertEqual(output.get("mode"), "live_execution")

            invocation = json.loads(log_path.read_text(encoding="utf-8"))
            argv = invocation.get("argv", [])
            self.assertIn("exec", argv)
            self.assertIn("--output-schema", argv)
            self.assertIn("--output-last-message", argv)
            self.assertIn("--sandbox", argv)
            sandbox_index = argv.index("--sandbox")
            self.assertEqual(argv[sandbox_index + 1], "workspace-write")
            prompt = invocation.get("prompt", "")
            self.assertIn("FULL MODE", prompt)
            self.assertIn("Workspace root boundary:", prompt)
            self.assertIn("Requested scope boundary: workspace", prompt)
            self.assertIn("Never read or modify paths outside the workspace root", prompt)
            self.assertIn("avoid unrelated directory enumeration", prompt)

    def test_limited_capability_uses_read_only_mode(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            root = Path(tmp_dir)
            fake_cli = root / "fake_claude_code.py"
            fake_cli.write_text(self._fake_claude_code_script(), encoding="utf-8")
            log_path = root / "claude_code_log.json"

            request = self._build_execute_request(
                action_type="personal.schedule",
                target_id="calendar",
                input_payload={"task": "draft schedule update"},
            )
            completed = self._run_wrapper(
                payload=request,
                cwd=root,
                env_overrides={
                    "ANTHROPIC_API_KEY": "test-token",
                    "SPICE_AGENT_CLAUDE_CODE_COMMAND": f"{sys.executable} {fake_cli}",
                    "FAKE_CLAUDE_CODE_MODE": "success",
                    "FAKE_CLAUDE_CODE_LOG": str(log_path),
                },
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            response = SDEPExecuteResponse.from_dict(json.loads(completed.stdout))
            self.assertEqual(response.status, "success")
            output = response.outcome.output
            self.assertEqual(output.get("support_level"), "limited")
            self.assertEqual(output.get("mode"), "limited_draft")

            invocation = json.loads(log_path.read_text(encoding="utf-8"))
            argv = invocation.get("argv", [])
            self.assertIn("--sandbox", argv)
            sandbox_index = argv.index("--sandbox")
            self.assertEqual(argv[sandbox_index + 1], "read-only")
            self.assertIn("LIMITED MODE", invocation.get("prompt", ""))

    def test_execute_rejects_unsupported_capability(self) -> None:
        request = self._build_execute_request(
            action_type="personal.manage_task",
            target_id="task_manager",
            input_payload={},
        )
        completed = self._run_wrapper(
            payload=request,
            env_overrides={"ANTHROPIC_API_KEY": "test-token"},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        response = SDEPExecuteResponse.from_dict(json.loads(completed.stdout))
        self.assertEqual(response.status, "error")
        self.assertIsNotNone(response.error)
        self.assertEqual(response.error.code, "capability.unsupported")

    def test_execute_returns_auth_error_when_token_missing(self) -> None:
        request = self._build_execute_request(
            action_type="personal.system",
            target_id="system",
            input_payload={},
        )
        completed = self._run_wrapper(
            payload=request,
            env_overrides={"ANTHROPIC_API_KEY": ""},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        response = SDEPExecuteResponse.from_dict(json.loads(completed.stdout))
        self.assertEqual(response.status, "error")
        self.assertIsNotNone(response.error)
        self.assertEqual(response.error.code, "auth.missing")

    def test_malformed_request_returns_valid_error_response(self) -> None:
        for raw in ("{ invalid", "[]"):
            with self.subTest(raw=raw):
                completed = self._run_wrapper(
                    payload=raw,
                    env_overrides={"ANTHROPIC_API_KEY": "test-token"},
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                response = SDEPExecuteResponse.from_dict(json.loads(completed.stdout))
                self.assertEqual(response.status, "error")
                self.assertIsNotNone(response.error)
                self.assertIn("request.", response.error.code)

    def test_transport_runtime_failure_is_reported(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            root = Path(tmp_dir)
            fake_cli = root / "fake_claude_code.py"
            fake_cli.write_text(self._fake_claude_code_script(), encoding="utf-8")
            request = self._build_execute_request(
                action_type="personal.system",
                target_id="system",
                input_payload={},
            )
            completed = self._run_wrapper(
                payload=request,
                cwd=root,
                env_overrides={
                    "ANTHROPIC_API_KEY": "test-token",
                    "SPICE_AGENT_CLAUDE_CODE_COMMAND": f"{sys.executable} {fake_cli}",
                    "FAKE_CLAUDE_CODE_MODE": "fail-exit",
                },
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            response = SDEPExecuteResponse.from_dict(json.loads(completed.stdout))
            self.assertEqual(response.status, "error")
            self.assertIsNotNone(response.error)
            self.assertEqual(response.error.code, "transport.runtime")

    def test_print_fallback_error_envelope_preserves_details(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            root = Path(tmp_dir)
            fake_cli = root / "fake_claude_code.py"
            fake_cli.write_text(self._fake_claude_code_script(), encoding="utf-8")
            request = self._build_execute_request(
                action_type="personal.system",
                target_id="system",
                input_payload={},
            )
            completed = self._run_wrapper(
                payload=request,
                cwd=root,
                env_overrides={
                    "ANTHROPIC_API_KEY": "test-token",
                    "SPICE_AGENT_CLAUDE_CODE_COMMAND": f"{sys.executable} {fake_cli}",
                    "FAKE_CLAUDE_CODE_MODE": "unsupported-option-print-error",
                },
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            response = SDEPExecuteResponse.from_dict(json.loads(completed.stdout))
            self.assertEqual(response.status, "error")
            self.assertIsNotNone(response.error)
            self.assertEqual(response.error.code, "transport.runtime")
            details = response.error.details if isinstance(response.error.details, dict) else {}
            self.assertEqual(details.get("subtype"), "error_max_turns")
            self.assertEqual(details.get("stop_reason"), "max_turns")
            self.assertEqual(details.get("session_id"), "sess-fallback")
            self.assertTrue(bool(details.get("permission_denials")))
            self.assertIn("fallback_command", details)
            self.assertIn("stdout", details)
            self.assertIn("stderr", details)

    def test_invalid_response_is_reported(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            root = Path(tmp_dir)
            fake_cli = root / "fake_claude_code.py"
            fake_cli.write_text(self._fake_claude_code_script(), encoding="utf-8")
            request = self._build_execute_request(
                action_type="personal.system",
                target_id="system",
                input_payload={},
            )
            completed = self._run_wrapper(
                payload=request,
                cwd=root,
                env_overrides={
                    "ANTHROPIC_API_KEY": "test-token",
                    "SPICE_AGENT_CLAUDE_CODE_COMMAND": f"{sys.executable} {fake_cli}",
                    "FAKE_CLAUDE_CODE_MODE": "invalid-json",
                },
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            response = SDEPExecuteResponse.from_dict(json.loads(completed.stdout))
            self.assertEqual(response.status, "error")
            self.assertIsNotNone(response.error)
            self.assertEqual(response.error.code, "response.invalid")

    def test_parse_print_output_accepts_structured_output_envelope(self) -> None:
        raw_output = json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "structured_output": {
                    "status": "success",
                    "summary": "one-line summary",
                    "evidence": [],
                    "actions": [],
                    "artifacts": [],
                    "errors": [],
                },
            },
            ensure_ascii=True,
        )

        parsed = claude_wrapper._parse_claude_print_json_output(
            raw_output,
            max_output_chars=4096,
        )

        self.assertEqual(parsed.get("status"), "success")
        self.assertEqual(parsed.get("summary"), "one-line summary")
        self.assertIsInstance(parsed.get("evidence"), list)

    def test_parse_print_output_surfaces_error_envelope(self) -> None:
        raw_output = json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "is_error": True,
                "result": "selected model may not exist",
            },
            ensure_ascii=True,
        )

        with self.assertRaises(claude_wrapper.ClaudeCodeWrapperError) as ctx:
            claude_wrapper._parse_claude_print_json_output(
                raw_output,
                max_output_chars=4096,
            )

        self.assertEqual(ctx.exception.code, "transport.runtime")
        self.assertIn("selected model may not exist", ctx.exception.message)

    def test_parse_print_output_surfaces_error_subtype_even_when_is_error_false(self) -> None:
        raw_output = json.dumps(
            {
                "type": "result",
                "subtype": "error_max_turns",
                "stop_reason": "max_turns",
                "session_id": "sess-print-1",
                "is_error": False,
                "result": "max turns reached before tool approval",
                "errors": ["max turns reached"],
                "permission_denials": [
                    {
                        "tool_name": "Bash",
                        "tool_use_id": "tooluse-1",
                        "reason": "approval required",
                        "tool_input": {"command": "ls /", "path": "/"},
                    }
                ],
            },
            ensure_ascii=True,
        )

        with self.assertRaises(claude_wrapper.ClaudeCodeWrapperError) as ctx:
            claude_wrapper._parse_claude_print_json_output(
                raw_output,
                max_output_chars=4096,
                stderr="simulated print stderr",
                fallback_command=["claude", "-p", "prompt"],
            )

        self.assertEqual(ctx.exception.code, "transport.runtime")
        self.assertIn("max turns reached", ctx.exception.message)
        self.assertEqual(ctx.exception.details.get("subtype"), "error_max_turns")
        self.assertEqual(ctx.exception.details.get("stop_reason"), "max_turns")
        self.assertEqual(ctx.exception.details.get("session_id"), "sess-print-1")
        self.assertTrue(bool(ctx.exception.details.get("permission_denials")))
        self.assertIn("stdout", ctx.exception.details)
        self.assertIn("stderr", ctx.exception.details)
        self.assertIn("fallback_command", ctx.exception.details)

    def test_build_print_command_full_mode_keeps_skip_permissions_and_max_turns(self) -> None:
        config = claude_wrapper.ClaudeCodeWrapperConfig(
            auth_env="ANTHROPIC_API_KEY",
            command="claude",
            workspace=REPO_ROOT,
            endpoint="",
            model="",
            timeout_seconds=120.0,
            sandbox="workspace-write",
            max_output_chars=4096,
        )
        command = claude_wrapper._build_claude_code_print_command(
            config=config,
            prompt="hello",
            limited=False,
        )
        self.assertIn("--dangerously-skip-permissions", command)
        self.assertIn("--max-turns", command)
        self.assertEqual(command[command.index("--max-turns") + 1], "8")

    def test_partial_execution_is_reported_explicitly(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            root = Path(tmp_dir)
            fake_cli = root / "fake_claude_code.py"
            fake_cli.write_text(self._fake_claude_code_script(), encoding="utf-8")
            request = self._build_execute_request(
                action_type="personal.system",
                target_id="system",
                input_payload={},
            )
            completed = self._run_wrapper(
                payload=request,
                cwd=root,
                env_overrides={
                    "ANTHROPIC_API_KEY": "test-token",
                    "SPICE_AGENT_CLAUDE_CODE_COMMAND": f"{sys.executable} {fake_cli}",
                    "FAKE_CLAUDE_CODE_MODE": "partial",
                },
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            response = SDEPExecuteResponse.from_dict(json.loads(completed.stdout))
            self.assertEqual(response.status, "error")
            self.assertIsNotNone(response.error)
            self.assertEqual(response.error.code, "execution.partial_failure")

    def test_personal_ask_e2e_uses_claude_code_provider_path(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            root = Path(tmp_dir)
            workspace = root / "personal_workspace"
            fake_cli = root / "fake_claude_code.py"
            fake_cli.write_text(self._fake_claude_code_script(), encoding="utf-8")
            model_script = root / "llm_evidence_model.py"
            model_script.write_text(self._evidence_personal_model_script(), encoding="utf-8")
            model_cmd = f"{sys.executable} {model_script}"

            init_completed = self._run_personal(
                "init",
                "--workspace",
                str(workspace),
            )
            self.assertEqual(init_completed.returncode, 0, init_completed.stderr)

            config_payload = {
                "schema_version": "spice_personal.connection.v1",
                "model": {
                    "command": model_cmd,
                },
                "agent": {
                    "provider": "claude_code",
                    "mode": "sdep",
                    "auth_env": "ANTHROPIC_API_KEY",
                },
            }
            (workspace / "personal.config.json").write_text(
                json.dumps(config_payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            completed = self._run_personal(
                "ask",
                "Should I switch teams this quarter?",
                "--workspace",
                str(workspace),
                "--verbose",
                env_overrides={
                    "ANTHROPIC_API_KEY": "test-token",
                    "SPICE_AGENT_CLAUDE_CODE_COMMAND": f"{sys.executable} {fake_cli}",
                    "FAKE_CLAUDE_CODE_MODE": "success",
                },
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("executor_mode=sdep", completed.stdout)
            self.assertIn(
                "notice=Gathering one bounded evidence snapshot before final advice.",
                completed.stdout,
            )
            self.assertIn("action=personal.assistant.suggest", completed.stdout)

    @staticmethod
    def _build_execute_request(
        *,
        action_type: str,
        target_id: str,
        input_payload: dict[str, object],
    ) -> dict[str, object]:
        intent = ExecutionIntent(
            id=f"intent-{uuid4().hex}",
            intent_type="personal.assistant.execution",
            status="planned",
            objective={"id": "obj-1", "description": "test execution"},
            executor_type="external-agent",
            target={"kind": "external.service", "id": target_id},
            operation={"name": action_type, "mode": "sync", "dry_run": False},
            input_payload=dict(input_payload),
            parameters={},
            constraints=[],
            success_criteria=[],
            failure_policy={"strategy": "fail_fast", "max_retries": 0},
            refs=[],
            provenance={"source": "tests"},
        )
        return build_sdep_execute_request(intent).to_dict()

    @staticmethod
    def _run_wrapper(
        payload: dict[str, object] | str,
        *,
        cwd: Path | None = None,
        env_overrides: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.pop("ANTHROPIC_API_KEY", None)
        pythonpath = str(REPO_ROOT)
        existing_pythonpath = env.get("PYTHONPATH", "").strip()
        env["PYTHONPATH"] = (
            pythonpath + os.pathsep + existing_pythonpath
            if existing_pythonpath
            else pythonpath
        )
        if isinstance(env_overrides, dict):
            env.update(env_overrides)
        stdin_text = (
            json.dumps(payload, ensure_ascii=True)
            if isinstance(payload, dict)
            else str(payload)
        )
        return subprocess.run(
            [sys.executable, "-m", "spice_personal.wrappers.claude_code_agent"],
            cwd=str(cwd or REPO_ROOT),
            env=env,
            text=True,
            input=stdin_text,
            capture_output=True,
            check=False,
        )

    @staticmethod
    def _run_personal(
        *args: str,
        env_overrides: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        for key in (
            "SPICE_PERSONAL_MODEL",
            "SPICE_PERSONAL_EXECUTOR",
            "SPICE_PERSONAL_EXECUTOR_TIMEOUT",
            "SPICE_PERSONAL_CLI_COMMAND",
            "SPICE_PERSONAL_CLI_PROFILE",
            "SPICE_PERSONAL_CLI_PROFILE_PATH",
            "SPICE_PERSONAL_CLI_PARSER_MODE",
            "SPICE_PERSONAL_SDEP_COMMAND",
        ):
            env.pop(key, None)
        if isinstance(env_overrides, dict):
            env.update(env_overrides)
        return subprocess.run(
            [sys.executable, "-m", "spice_personal.cli", *args],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    @staticmethod
    def _fake_claude_code_script() -> str:
        return (
            "from __future__ import annotations\n"
            "import json\n"
            "import os\n"
            "import pathlib\n"
            "import sys\n"
            "\n"
            "def _value(flag: str, args: list[str]) -> str:\n"
            "    for idx, token in enumerate(args):\n"
            "        if token == flag and idx + 1 < len(args):\n"
            "            return args[idx + 1]\n"
            "    return ''\n"
            "\n"
            "def main() -> int:\n"
            "    args = sys.argv[1:]\n"
            "    mode = os.environ.get('FAKE_CLAUDE_CODE_MODE', 'success').strip().lower()\n"
            "    if args and args[0] == 'exec':\n"
            "        output_path = _value('--output-last-message', args)\n"
            "        prompt = args[-1] if args else ''\n"
            "        log_path = os.environ.get('FAKE_CLAUDE_CODE_LOG', '').strip()\n"
            "        if log_path:\n"
            "            pathlib.Path(log_path).write_text(\n"
            "                json.dumps({'argv': args, 'prompt': prompt}, ensure_ascii=True),\n"
            "                encoding='utf-8',\n"
            "            )\n"
            "        if mode == 'unsupported-option-print-error':\n"
            "            print('unknown option --output-schema', file=sys.stderr)\n"
            "            return 2\n"
            "        if mode == 'fail-exit':\n"
            "            print('simulated claude code failure', file=sys.stderr)\n"
            "            return 7\n"
            "        if mode == 'invalid-json':\n"
            "            pathlib.Path(output_path).write_text('not-json', encoding='utf-8')\n"
            "            return 0\n"
            "        status = 'partial' if mode == 'partial' else 'success'\n"
            "        payload = {\n"
            "            'status': status,\n"
            "            'summary': 'fake claude code execution summary',\n"
            "            'evidence': [{'type': 'observation', 'detail': 'checked workspace'}],\n"
            "            'actions': [{'kind': 'draft' if 'LIMITED MODE' in prompt else 'command', 'result': 'ok'}],\n"
            "            'artifacts': [],\n"
            "            'errors': ['simulated partial issue'] if mode == 'partial' else [],\n"
            "        }\n"
            "        pathlib.Path(output_path).write_text(json.dumps(payload, ensure_ascii=True), encoding='utf-8')\n"
            "        return 0\n"
            "    if args and args[0] == '-p':\n"
            "        if mode == 'unsupported-option-print-error':\n"
            "            payload = {\n"
            "                'type': 'result',\n"
            "                'subtype': 'error_max_turns',\n"
            "                'stop_reason': 'max_turns',\n"
            "                'session_id': 'sess-fallback',\n"
            "                'is_error': False,\n"
            "                'result': 'max turns reached in print mode',\n"
            "                'errors': ['max turns reached'],\n"
            "                'permission_denials': [\n"
            "                    {\n"
            "                        'tool_name': 'Bash',\n"
            "                        'tool_use_id': 'tooluse-fallback',\n"
            "                        'reason': 'approval required',\n"
            "                        'tool_input': {'command': 'ls /', 'path': '/'},\n"
            "                    }\n"
            "                ],\n"
            "            }\n"
            "            print(json.dumps(payload, ensure_ascii=True))\n"
            "            print('simulated print mode failure', file=sys.stderr)\n"
            "            return 9\n"
            "        payload = {\n"
            "            'type': 'result',\n"
            "            'subtype': 'success',\n"
            "            'is_error': False,\n"
            "            'structured_output': {\n"
            "                'status': 'success',\n"
            "                'summary': 'fallback success',\n"
            "                'evidence': [],\n"
            "                'actions': [],\n"
            "                'artifacts': [],\n"
            "                'errors': [],\n"
            "            },\n"
            "        }\n"
            "        print(json.dumps(payload, ensure_ascii=True))\n"
            "        return 0\n"
            "    print('expected exec or -p invocation', file=sys.stderr)\n"
            "    return 2\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    raise SystemExit(main())\n"
        )

    @staticmethod
    def _evidence_personal_model_script() -> str:
        return (
            "import json\n"
            "import sys\n"
            "prompt = sys.stdin.read()\n"
            "has_evidence = 'personal.assistant.evidence_received' in prompt\n"
            "if 'Decision proposals' in prompt:\n"
            "    if has_evidence:\n"
            "        payload = [\n"
            "            {\n"
            "                'decision_type': 'personal.assistant.llm',\n"
            "                'status': 'proposed',\n"
            "                'selected_action': 'personal.assistant.suggest',\n"
            "                'attributes': {'confidence': 0.90, 'urgency': 'normal'},\n"
            "            }\n"
            "        ]\n"
            "    else:\n"
            "        payload = [\n"
            "            {\n"
            "                'decision_type': 'personal.assistant.llm',\n"
            "                'status': 'proposed',\n"
            "                'selected_action': 'personal.assistant.gather_evidence',\n"
            "                'attributes': {'confidence': 0.42, 'urgency': 'normal'},\n"
            "            }\n"
            "        ]\n"
            "elif 'simulation advice' in prompt:\n"
            "    if has_evidence:\n"
            "        payload = {\n"
            "            'score': 0.96,\n"
            "            'confidence': 0.91,\n"
            "            'urgency': 'normal',\n"
            "            'suggestion_text': 'Refined advice after evidence: Option B is lower-regret because manager quality is stronger and team attrition is lower.',\n"
            "            'simulation_rationale': 'evidence_applied',\n"
            "            'benefits': ['Verified mentor quality supports management progression.'],\n"
            "            'risks': ['Compensation upside is lower than the high-risk alternative.'],\n"
            "            'key_assumptions': ['Mentor quality remains consistent over the next year.'],\n"
            "            'first_step_24h': 'Within 24h, confirm first-quarter ownership scope with the manager.',\n"
            "            'stop_loss_trigger': 'If ownership scope is not confirmed by week 4, reopen alternatives.',\n"
            "            'change_mind_condition': 'Switch if verified attrition and mentorship data reverse.',\n"
            "        }\n"
            "    else:\n"
            "        payload = {\n"
            "            'score': 0.92,\n"
            "            'confidence': 0.42,\n"
            "            'urgency': 'normal',\n"
            "            'suggestion_text': 'Collect one evidence snapshot before a strong recommendation.',\n"
            "            'simulation_rationale': 'needs_evidence',\n"
            "            'evidence_plan': [\n"
            "                {'fact': 'Verify attrition trend for each team in the last 12 months.', 'why': 'Attrition materially changes execution risk ranking.'},\n"
            "                {'fact': 'Validate manager coaching outcomes from direct references.', 'why': 'Coaching quality can change management-path probability.'},\n"
            "                {'fact': 'Confirm first-6-month scope ownership expectations.', 'why': 'Ownership scope affects promotion readiness.'},\n"
            "            ],\n"
            "        }\n"
            "else:\n"
            "    payload = {'score': 0.0}\n"
            "print(json.dumps(payload))\n"
        )


if __name__ == "__main__":
    unittest.main()
