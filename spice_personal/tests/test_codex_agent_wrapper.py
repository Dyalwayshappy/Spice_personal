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


REPO_ROOT = Path(__file__).resolve().parents[2]


class CodexAgentWrapperTests(unittest.TestCase):
    def test_sdep_describe_happy_path_reports_honest_support_levels(self) -> None:
        request = build_sdep_describe_request(
            action_types=[
                "personal.gather_evidence",
                "personal.system",
                "personal.communicate",
                "personal.schedule",
            ]
        )
        completed = self._run_codex_wrapper(
            payload=request.to_dict(),
            env_overrides={"OPENAI_API_KEY": "test-token"},
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
            "codex.exec",
        )

    def test_execute_happy_path_delegates_to_codex_exec(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            root = Path(tmp_dir)
            workspace = root / "workspace"
            workspace.mkdir(parents=True, exist_ok=True)
            fake_codex = root / "fake_codex.py"
            fake_codex.write_text(self._fake_codex_script(), encoding="utf-8")
            log_path = root / "codex_log.json"

            request = self._build_execute_request(
                action_type="personal.system",
                target_id="system",
                input_payload={"task": "inspect workspace state"},
            )
            completed = self._run_codex_wrapper(
                payload=request,
                cwd=workspace,
                env_overrides={
                    "OPENAI_API_KEY": "test-token",
                    "SPICE_AGENT_CODEX_COMMAND": f"{sys.executable} {fake_codex}",
                    "FAKE_CODEX_MODE": "success",
                    "FAKE_CODEX_LOG": str(log_path),
                },
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            response = SDEPExecuteResponse.from_dict(json.loads(completed.stdout))
            self.assertEqual(response.status, "success")
            output = response.outcome.output
            self.assertEqual(output.get("integration_backend"), "codex.exec")
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
            self.assertIn("FULL MODE", invocation.get("prompt", ""))

    def test_limited_capability_uses_read_only_mode(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            root = Path(tmp_dir)
            fake_codex = root / "fake_codex.py"
            fake_codex.write_text(self._fake_codex_script(), encoding="utf-8")
            log_path = root / "codex_log.json"

            request = self._build_execute_request(
                action_type="personal.schedule",
                target_id="calendar",
                input_payload={"task": "draft schedule update"},
            )
            completed = self._run_codex_wrapper(
                payload=request,
                cwd=root,
                env_overrides={
                    "OPENAI_API_KEY": "test-token",
                    "SPICE_AGENT_CODEX_COMMAND": f"{sys.executable} {fake_codex}",
                    "FAKE_CODEX_MODE": "success",
                    "FAKE_CODEX_LOG": str(log_path),
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
        completed = self._run_codex_wrapper(
            payload=request,
            env_overrides={"OPENAI_API_KEY": "test-token"},
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
        completed = self._run_codex_wrapper(
            payload=request,
            env_overrides={"OPENAI_API_KEY": ""},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        response = SDEPExecuteResponse.from_dict(json.loads(completed.stdout))
        self.assertEqual(response.status, "error")
        self.assertIsNotNone(response.error)
        self.assertEqual(response.error.code, "auth.missing")

    def test_malformed_request_returns_valid_error_response(self) -> None:
        for raw in ("{ invalid", "[]"):
            with self.subTest(raw=raw):
                completed = self._run_codex_wrapper(
                    payload=raw,
                    env_overrides={"OPENAI_API_KEY": "test-token"},
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                response = SDEPExecuteResponse.from_dict(json.loads(completed.stdout))
                self.assertEqual(response.status, "error")
                self.assertIsNotNone(response.error)
                self.assertIn("request.", response.error.code)

    def test_transport_runtime_failure_is_reported(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            root = Path(tmp_dir)
            fake_codex = root / "fake_codex.py"
            fake_codex.write_text(self._fake_codex_script(), encoding="utf-8")

            request = self._build_execute_request(
                action_type="personal.system",
                target_id="system",
                input_payload={},
            )
            completed = self._run_codex_wrapper(
                payload=request,
                cwd=root,
                env_overrides={
                    "OPENAI_API_KEY": "test-token",
                    "SPICE_AGENT_CODEX_COMMAND": f"{sys.executable} {fake_codex}",
                    "FAKE_CODEX_MODE": "fail-exit",
                },
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            response = SDEPExecuteResponse.from_dict(json.loads(completed.stdout))
            self.assertEqual(response.status, "error")
            self.assertIsNotNone(response.error)
            self.assertEqual(response.error.code, "transport.runtime")

    def test_invalid_codex_response_is_reported(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            root = Path(tmp_dir)
            fake_codex = root / "fake_codex.py"
            fake_codex.write_text(self._fake_codex_script(), encoding="utf-8")

            request = self._build_execute_request(
                action_type="personal.system",
                target_id="system",
                input_payload={},
            )
            completed = self._run_codex_wrapper(
                payload=request,
                cwd=root,
                env_overrides={
                    "OPENAI_API_KEY": "test-token",
                    "SPICE_AGENT_CODEX_COMMAND": f"{sys.executable} {fake_codex}",
                    "FAKE_CODEX_MODE": "invalid-json",
                },
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            response = SDEPExecuteResponse.from_dict(json.loads(completed.stdout))
            self.assertEqual(response.status, "error")
            self.assertIsNotNone(response.error)
            self.assertEqual(response.error.code, "response.invalid")

    def test_partial_execution_is_reported_explicitly(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            root = Path(tmp_dir)
            fake_codex = root / "fake_codex.py"
            fake_codex.write_text(self._fake_codex_script(), encoding="utf-8")

            request = self._build_execute_request(
                action_type="personal.system",
                target_id="system",
                input_payload={},
            )
            completed = self._run_codex_wrapper(
                payload=request,
                cwd=root,
                env_overrides={
                    "OPENAI_API_KEY": "test-token",
                    "SPICE_AGENT_CODEX_COMMAND": f"{sys.executable} {fake_codex}",
                    "FAKE_CODEX_MODE": "partial",
                },
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            response = SDEPExecuteResponse.from_dict(json.loads(completed.stdout))
            self.assertEqual(response.status, "error")
            self.assertIsNotNone(response.error)
            self.assertEqual(response.error.code, "execution.partial_failure")

    def test_personal_ask_e2e_uses_codex_provider_path(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            root = Path(tmp_dir)
            workspace = root / "personal_workspace"
            fake_codex = root / "fake_codex.py"
            fake_codex.write_text(self._fake_codex_script(), encoding="utf-8")
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
                    "provider": "codex",
                    "mode": "sdep",
                    "auth_env": "OPENAI_API_KEY",
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
                    "OPENAI_API_KEY": "test-token",
                    "SPICE_AGENT_CODEX_COMMAND": f"{sys.executable} {fake_codex}",
                    "FAKE_CODEX_MODE": "success",
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
    def _run_codex_wrapper(
        payload: dict[str, object] | str,
        *,
        cwd: Path | None = None,
        env_overrides: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.pop("OPENAI_API_KEY", None)
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
            [sys.executable, "-m", "spice_personal.wrappers.codex_agent"],
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
    def _fake_codex_script() -> str:
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
            "    if not args or args[0] != 'exec':\n"
            "        print('expected exec subcommand', file=sys.stderr)\n"
            "        return 2\n"
            "    output_path = _value('--output-last-message', args)\n"
            "    mode = os.environ.get('FAKE_CODEX_MODE', 'success').strip().lower()\n"
            "    prompt = args[-1] if args else ''\n"
            "    log_path = os.environ.get('FAKE_CODEX_LOG', '').strip()\n"
            "    if log_path:\n"
            "        pathlib.Path(log_path).write_text(\n"
            "            json.dumps({'argv': args, 'prompt': prompt}, ensure_ascii=True),\n"
            "            encoding='utf-8',\n"
            "        )\n"
            "    if mode == 'fail-exit':\n"
            "        print('simulated codex failure', file=sys.stderr)\n"
            "        return 7\n"
            "    if mode == 'missing-output':\n"
            "        return 0\n"
            "    if mode == 'invalid-json':\n"
            "        pathlib.Path(output_path).write_text('not-json', encoding='utf-8')\n"
            "        return 0\n"
            "    status = 'partial' if mode == 'partial' else 'success'\n"
            "    payload = {\n"
            "        'status': status,\n"
            "        'summary': 'fake codex execution summary',\n"
            "        'evidence': [{'type': 'observation', 'detail': 'checked workspace'}],\n"
            "        'actions': [{'kind': 'draft' if 'LIMITED MODE' in prompt else 'command', 'result': 'ok'}],\n"
            "        'artifacts': [],\n"
            "        'errors': ['simulated partial issue'] if mode == 'partial' else [],\n"
            "    }\n"
            "    pathlib.Path(output_path).write_text(json.dumps(payload, ensure_ascii=True), encoding='utf-8')\n"
            "    return 0\n"
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
