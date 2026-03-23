from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from spice_personal.config.personal_config import load_personal_connection_config


REPO_ROOT = Path(__file__).resolve().parents[2]
ECHO_AGENT = REPO_ROOT / "examples" / "sdep_agent_demo" / "echo_agent.py"


class PersonalConnectionConfigTests(unittest.TestCase):
    def test_provider_style_model_config_compiles_to_internal_model_command(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            workspace = Path(tmp_dir) / "personal_workspace"
            workspace.mkdir(parents=True, exist_ok=True)
            self._write_personal_config(
                workspace,
                payload={
                    "schema_version": "spice_personal.connection.v1",
                    "model": {
                        "provider": "openrouter",
                        "model": "anthropic/claude-opus-4-5",
                        "api_key_env": "OPENROUTER_API_KEY",
                        "base_url": "http://127.0.0.1:8317/v1",
                    },
                },
            )
            config = load_personal_connection_config(workspace)

            self.assertEqual(config.schema_version, "spice_personal.connection.v1")
            self.assertEqual(config.model_provider, "openrouter")
            self.assertEqual(config.model_name, "anthropic/claude-opus-4-5")
            self.assertEqual(config.model_api_key_env, "OPENROUTER_API_KEY")
            self.assertEqual(config.model_base_url, "http://127.0.0.1:8317/v1")
            self.assertIsNotNone(config.model_command)
            self.assertIn("spice_personal.wrappers.openrouter_model", config.model_command or "")
            self.assertIn("--model", config.model_command or "")
            self.assertIn("anthropic/claude-opus-4-5", config.model_command or "")
            self.assertIn("--api-key-env", config.model_command or "")
            self.assertIn("OPENROUTER_API_KEY", config.model_command or "")
            self.assertIn("--base-url", config.model_command or "")
            self.assertIn("http://127.0.0.1:8317/v1", config.model_command or "")
            self.assertEqual(config.model_command_source, "provider")

    def test_provider_style_agent_config_compiles_to_internal_sdep_plan(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            workspace = Path(tmp_dir) / "personal_workspace"
            workspace.mkdir(parents=True, exist_ok=True)
            self._write_personal_config(
                workspace,
                payload={
                    "schema_version": "spice_personal.connection.v1",
                    "agent": {
                        "provider": "openclaw",
                        "mode": "sdep",
                        "auth_env": "OPENCLAW_API_KEY",
                        "endpoint": "https://api.openclaw.example/sdep",
                    },
                },
            )
            config = load_personal_connection_config(workspace)

            self.assertEqual(config.agent_provider, "openclaw")
            self.assertEqual(config.executor_mode, "sdep")
            self.assertEqual(config.executor_mode_source, "provider")
            self.assertIsNotNone(config.sdep_command)
            self.assertIn(
                "spice_personal.provider_bridges.sdep_agent_provider_bridge",
                config.sdep_command or "",
            )

    def test_provider_style_codex_agent_config_compiles_to_internal_sdep_plan(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            workspace = Path(tmp_dir) / "personal_workspace"
            workspace.mkdir(parents=True, exist_ok=True)
            self._write_personal_config(
                workspace,
                payload={
                    "schema_version": "spice_personal.connection.v1",
                    "agent": {
                        "provider": "codex",
                        "mode": "sdep",
                        "auth_env": "OPENAI_API_KEY",
                    },
                },
            )
            config = load_personal_connection_config(workspace)

            self.assertEqual(config.agent_provider, "codex")
            self.assertEqual(config.agent_auth_env, "OPENAI_API_KEY")
            self.assertEqual(config.executor_mode, "sdep")
            self.assertEqual(config.executor_mode_source, "provider")
            self.assertEqual(config.sdep_command_source, "provider")
            self.assertIn(
                "spice_personal.wrappers.codex_agent",
                config.sdep_command or "",
            )
            self.assertIn("--auth-env", config.sdep_command or "")
            self.assertIn("OPENAI_API_KEY", config.sdep_command or "")

    def test_provider_style_claude_code_agent_config_compiles_to_internal_sdep_plan(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            workspace = Path(tmp_dir) / "personal_workspace"
            workspace.mkdir(parents=True, exist_ok=True)
            self._write_personal_config(
                workspace,
                payload={
                    "schema_version": "spice_personal.connection.v1",
                    "agent": {
                        "provider": "claude_code",
                        "mode": "sdep",
                        "auth_env": "ANTHROPIC_API_KEY",
                    },
                },
            )
            config = load_personal_connection_config(workspace)

            self.assertEqual(config.agent_provider, "claude_code")
            self.assertEqual(config.agent_auth_env, "ANTHROPIC_API_KEY")
            self.assertEqual(config.executor_mode, "sdep")
            self.assertEqual(config.executor_mode_source, "provider")
            self.assertEqual(config.sdep_command_source, "provider")
            self.assertIn(
                "spice_personal.wrappers.claude_code_agent",
                config.sdep_command or "",
            )
            self.assertIn("--auth-env", config.sdep_command or "")
            self.assertIn("ANTHROPIC_API_KEY", config.sdep_command or "")

    def test_legacy_command_fields_override_provider_fields(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            workspace = Path(tmp_dir) / "personal_workspace"
            workspace.mkdir(parents=True, exist_ok=True)
            self._write_personal_config(
                workspace,
                payload={
                    "schema_version": "spice_personal.connection.v1",
                    "model": {
                        "provider": "openrouter",
                        "model": "anthropic/claude-opus-4-5",
                        "api_key_env": "OPENROUTER_API_KEY",
                        "command": "python3 legacy_model.py",
                    },
                    "agent": {
                        "provider": "openclaw",
                        "mode": "sdep",
                        "auth_env": "OPENCLAW_API_KEY",
                    },
                    "executor": {
                        "mode": "cli",
                        "cli_command": "python3 legacy_agent.py",
                        "sdep_command": "python3 legacy_sdep.py",
                    },
                },
            )
            config = load_personal_connection_config(workspace)

            self.assertEqual(config.model_command, "python3 legacy_model.py")
            self.assertEqual(config.model_command_source, "legacy.model.command")
            self.assertEqual(config.executor_mode, "cli")
            self.assertEqual(config.executor_mode_source, "legacy.executor.mode")
            self.assertEqual(config.cli_command, "python3 legacy_agent.py")
            self.assertEqual(config.cli_command_source, "legacy.executor.cli_command")
            self.assertEqual(config.sdep_command, "python3 legacy_sdep.py")
            self.assertEqual(config.sdep_command_source, "legacy.executor.sdep_command")

    def test_personal_config_model_command_used_when_cli_and_env_are_absent(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            root = Path(tmp_dir)
            workspace = root / "personal_workspace"
            model_script = root / "llm_personal_model.py"
            model_script.write_text(self._custom_personal_model_script(), encoding="utf-8")
            model_cmd = f"{sys.executable} {model_script}"

            init_completed = self._run_personal(
                "init",
                "--workspace",
                str(workspace),
            )
            self.assertEqual(init_completed.returncode, 0, init_completed.stderr)

            self._write_personal_config(
                workspace,
                payload={
                    "model": {
                        "command": model_cmd,
                    }
                },
            )

            completed = self._run_personal(
                "ask",
                "Should I switch teams?",
                "--workspace",
                str(workspace),
                "--verbose",
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn(
                "suggestion=Clarifying Questions",
                completed.stdout,
            )

    def test_env_model_command_overrides_personal_config_model_command(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            root = Path(tmp_dir)
            workspace = root / "personal_workspace"
            model_script = root / "llm_personal_model.py"
            model_script.write_text(self._custom_personal_model_script(), encoding="utf-8")
            model_cmd = f"{sys.executable} {model_script}"

            init_completed = self._run_personal(
                "init",
                "--workspace",
                str(workspace),
            )
            self.assertEqual(init_completed.returncode, 0, init_completed.stderr)

            self._write_personal_config(
                workspace,
                payload={
                    "model": {
                        "command": model_cmd,
                    }
                },
            )

            completed = self._run_personal(
                "ask",
                "Should I switch teams?",
                "--workspace",
                str(workspace),
                "--verbose",
                env_overrides={"SPICE_PERSONAL_MODEL": "deterministic"},
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn(
                "suggestion=Spice recommendation",
                completed.stdout,
            )
            self.assertNotIn("Config model: clarify your top tradeoff first.", completed.stdout)

    def test_personal_config_executor_mode_and_sdep_command_used_by_default(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            root = Path(tmp_dir)
            workspace = root / "personal_workspace"
            model_script = root / "llm_evidence_model.py"
            model_script.write_text(self._evidence_personal_model_script(), encoding="utf-8")
            model_cmd = f"{sys.executable} {model_script}"
            sdep_cmd = f"{sys.executable} {ECHO_AGENT}"

            init_completed = self._run_personal(
                "init",
                "--workspace",
                str(workspace),
            )
            self.assertEqual(init_completed.returncode, 0, init_completed.stderr)

            self._write_personal_config(
                workspace,
                payload={
                    "executor": {
                        "mode": "sdep",
                        "sdep_command": sdep_cmd,
                    }
                },
            )

            completed = self._run_personal(
                "ask",
                "Should I take one next step?",
                "--workspace",
                str(workspace),
                "--model",
                model_cmd,
                "--verbose",
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("executor_mode=sdep", completed.stdout)
            self.assertIn(
                "notice=Gathering one bounded evidence snapshot before final advice.",
                completed.stdout,
            )

    def test_cli_overrides_provider_style_executor_mode(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            root = Path(tmp_dir)
            workspace = root / "personal_workspace"
            init_completed = self._run_personal(
                "init",
                "--workspace",
                str(workspace),
            )
            self.assertEqual(init_completed.returncode, 0, init_completed.stderr)

            self._write_personal_config(
                workspace,
                payload={
                    "schema_version": "spice_personal.connection.v1",
                    "agent": {
                        "provider": "openclaw",
                        "mode": "sdep",
                        "auth_env": "OPENCLAW_API_KEY",
                    },
                },
            )
            completed = self._run_personal(
                "ask",
                "Should I switch teams?",
                "--workspace",
                str(workspace),
                "--executor",
                "mock",
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("Setup required (no model configured)", completed.stdout)
            resolution_path = workspace / "artifacts" / "personal_connection_resolution.json"
            self.assertTrue(resolution_path.exists())
            resolution_payload = json.loads(resolution_path.read_text(encoding="utf-8"))
            executor_payload = resolution_payload.get("executor", {})
            self.assertEqual(executor_payload.get("mode"), "mock")

    def test_invalid_personal_config_payload_requires_setup(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            root = Path(tmp_dir)
            workspace = root / "personal_workspace"
            init_completed = self._run_personal(
                "init",
                "--workspace",
                str(workspace),
            )
            self.assertEqual(init_completed.returncode, 0, init_completed.stderr)

            (workspace / "personal.config.json").write_text("{ invalid json", encoding="utf-8")
            completed = self._run_personal(
                "ask",
                "Should I switch teams?",
                "--workspace",
                str(workspace),
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("Setup required (no model configured)", completed.stdout)

    def test_connection_resolution_artifact_does_not_leak_secret_values(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            root = Path(tmp_dir)
            workspace = root / "personal_workspace"
            model_script = root / "llm_personal_model.py"
            model_script.write_text(self._custom_personal_model_script(), encoding="utf-8")
            model_cmd = f"{sys.executable} {model_script}"
            init_completed = self._run_personal(
                "init",
                "--workspace",
                str(workspace),
            )
            self.assertEqual(init_completed.returncode, 0, init_completed.stderr)

            self._write_personal_config(
                workspace,
                payload={
                    "schema_version": "spice_personal.connection.v1",
                    "model": {
                        "provider": "openrouter",
                        "model": "anthropic/claude-opus-4-5",
                        "api_key_env": "OPENROUTER_API_KEY",
                        "command": model_cmd,
                    },
                    "agent": {
                        "provider": "openclaw",
                        "mode": "sdep",
                        "auth_env": "OPENCLAW_API_KEY",
                    },
                },
            )
            completed = self._run_personal(
                "ask",
                "Should I switch teams?",
                "--workspace",
                str(workspace),
                env_overrides={
                    "OPENROUTER_API_KEY": "super-secret-openrouter",
                    "OPENCLAW_API_KEY": "super-secret-openclaw",
                },
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

            resolution_path = workspace / "artifacts" / "personal_connection_resolution.json"
            self.assertTrue(resolution_path.exists())
            resolution_raw = resolution_path.read_text(encoding="utf-8")
            self.assertIn("OPENROUTER_API_KEY", resolution_raw)
            self.assertIn("OPENCLAW_API_KEY", resolution_raw)
            self.assertNotIn("super-secret-openrouter", resolution_raw)
            self.assertNotIn("super-secret-openclaw", resolution_raw)

    @staticmethod
    def _write_personal_config(workspace: Path, *, payload: dict[str, object]) -> None:
        path = workspace / "personal.config.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
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
    def _custom_personal_model_script() -> str:
        return (
            "import json\n"
            "import sys\n"
            "prompt = sys.stdin.read()\n"
            "if 'Decision proposals' in prompt:\n"
            "    payload = [\n"
            "        {\n"
            "            'decision_type': 'personal.assistant.llm',\n"
            "            'status': 'proposed',\n"
            "            'selected_action': 'personal.assistant.ask_clarify',\n"
            "            'attributes': {'confidence': 0.82, 'urgency': 'high'},\n"
            "        }\n"
            "    ]\n"
            "elif 'simulation advice' in prompt:\n"
            "    payload = {\n"
            "        'score': 0.91,\n"
            "        'confidence': 0.82,\n"
            "        'urgency': 'high',\n"
            "        'suggestion_text': 'Config model: clarify your top tradeoff first.',\n"
            "        'clarifying_questions': [\n"
            "            {'question': 'What is your top non-negotiable?', 'why': 'It can change option ranking.'},\n"
            "            {'question': 'How much downside can you tolerate this year?', 'why': 'It can reorder risk profiles.'},\n"
            "            {'question': 'What 3-year outcome defines success?', 'why': 'It can shift the recommendation target.'},\n"
            "        ],\n"
            "    }\n"
            "else:\n"
            "    payload = {'score': 0.0}\n"
            "print(json.dumps(payload))\n"
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
            "            'score': 0.93,\n"
            "            'confidence': 0.90,\n"
            "            'urgency': 'normal',\n"
            "            'suggestion_text': 'Evidence-informed recommendation: Option B better supports management growth with lower verified attrition risk.',\n"
            "            'benefits': ['Verified stability and mentorship support long-term management progression.'],\n"
            "            'risks': ['Compensation upside may be lower than Option A.'],\n"
            "            'key_assumptions': ['Mentorship quality remains stable through year one.'],\n"
            "            'first_step_24h': 'Within 24h, confirm first-quarter ownership scope with Option B manager.',\n"
            "            'stop_loss_trigger': 'If scope ownership is unclear by week 4, reopen Option A.',\n"
            "            'change_mind_condition': 'Switch if verified stability and growth signals reverse in favor of Option A.',\n"
            "        }\n"
            "    else:\n"
            "        payload = {\n"
            "            'score': 0.41,\n"
            "            'confidence': 0.42,\n"
            "            'urgency': 'normal',\n"
            "            'suggestion_text': 'Collect one evidence snapshot first.',\n"
            "            'evidence_plan': [\n"
            "                {'fact': 'Verify attrition trend for each option over 12 months.', 'why': 'Attrition changes execution risk ranking.'},\n"
            "                {'fact': 'Validate manager coaching outcomes from direct references.', 'why': 'Coaching quality changes management-path probability.'},\n"
            "                {'fact': 'Confirm scope ownership expectations in first 6 months.', 'why': 'Ownership scope affects promotion readiness.'},\n"
            "            ],\n"
            "        }\n"
            "else:\n"
            "    payload = {'score': 0.0}\n"
            "print(json.dumps(payload))\n"
        )


if __name__ == "__main__":
    unittest.main()
