from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class PersonalEntryTests(unittest.TestCase):
    def test_personal_init_creates_workspace_scaffold(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            workspace = Path(tmp_dir) / "personal_workspace"
            completed = self._run_personal(
                "init",
                "--workspace",
                str(workspace),
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((workspace / "domain_spec.json").exists())
            self.assertTrue((workspace / "run_demo.py").exists())
            self.assertTrue((workspace / "config" / "personal.profile.json").exists())
            self.assertTrue((workspace / "artifacts" / "personal_profile_validation.json").exists())
            config_path = workspace / "personal.config.json"
            self.assertTrue(config_path.exists())
            config_payload = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertIsInstance(config_payload, dict)
            self.assertIn("model", config_payload)
            self.assertIn("agent", config_payload)
            self.assertIn("executor", config_payload)

    def test_personal_init_force_does_not_overwrite_existing_personal_config(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            workspace = Path(tmp_dir) / "personal_workspace"
            init_completed = self._run_personal(
                "init",
                "--workspace",
                str(workspace),
            )
            self.assertEqual(init_completed.returncode, 0, init_completed.stderr)

            config_path = workspace / "personal.config.json"
            custom_payload = {
                "schema_version": "spice_personal.connection.v1",
                "model": {"command": "python3 custom_model.py"},
                "agent": {"provider": "generic_sdep", "mode": "sdep"},
                "executor": {"mode": "sdep", "sdep_command": "python3 custom_agent.py"},
            }
            config_path.write_text(
                json.dumps(custom_payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            force_completed = self._run_personal(
                "init",
                "--workspace",
                str(workspace),
                "--force",
            )
            self.assertEqual(force_completed.returncode, 0, force_completed.stderr)

            reloaded_payload = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(reloaded_payload, custom_payload)

    def test_personal_ask_shows_setup_decision_card_when_openrouter_key_missing(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            workspace = Path(tmp_dir) / "personal_workspace"
            init_completed = self._run_personal(
                "init",
                "--workspace",
                str(workspace),
            )
            self.assertEqual(init_completed.returncode, 0, init_completed.stderr)

            env = os.environ.copy()
            env.pop("OPENROUTER_API_KEY", None)
            env.pop("SPICE_PERSONAL_MODEL", None)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "spice_personal.cli",
                    "ask",
                    "What should I do next?",
                    "--workspace",
                    str(workspace),
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
                env=env,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("Decision Card", completed.stdout)
            self.assertIn("Setup required (no model configured)", completed.stdout)
            self.assertIn(f"Edit {workspace / 'personal.config.json'}", completed.stdout)
            self.assertIn(
                'Then run: spice-personal ask "What should I do next?"',
                completed.stdout,
            )

    def test_personal_ask_auto_initializes_workspace_without_state_write(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            workspace = Path(tmp_dir) / "personal_workspace"
            completed = self._run_personal(
                "ask",
                "Should I switch jobs this year?",
                "--workspace",
                str(workspace),
                "--model",
                "deterministic",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("Spice recommendation", completed.stdout)
            self.assertNotIn("action=", completed.stdout)
            self.assertNotIn("option_1_id=", completed.stdout)
            self.assertTrue((workspace / "domain_spec.json").exists())
            self.assertFalse((workspace / "artifacts" / "personal_state.json").exists())

    def test_personal_interactive_session_persists_state_when_model_configured(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            workspace = Path(tmp_dir) / "personal_workspace"
            completed = self._run_personal(
                "--workspace",
                str(workspace),
                "--model",
                "deterministic",
                input_text="Should I move to a new city?\nexit\n",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("adopt this suggestion now? [y/N]", completed.stdout)
            self.assertNotIn("decision_id=", completed.stdout)
            self.assertNotIn("advisory_outcome_id=", completed.stdout)
            self.assertNotIn("reflection_id=", completed.stdout)
            state_path = workspace / "artifacts" / "personal_state.json"
            self.assertTrue(state_path.exists())

            payload = json.loads(state_path.read_text(encoding="utf-8"))
            entities = payload.get("entities", {})
            self.assertIsInstance(entities, dict)
            current = entities.get("personal.assistant.current", {})
            self.assertIsInstance(current, dict)
            self.assertEqual(
                current.get("latest_question"),
                "Should I move to a new city?",
            )
            self.assertTrue(str(current.get("latest_suggestion", "")).strip())

            recent_outcomes = payload.get("recent_outcomes", [])
            self.assertIsInstance(recent_outcomes, list)
            self.assertGreaterEqual(len(recent_outcomes), 1)

    def test_personal_session_requires_model_when_not_configured(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            workspace = Path(tmp_dir) / "personal_workspace"
            completed = self._run_personal(
                "--workspace",
                str(workspace),
                "--model",
                "missing_command_that_should_fail",
                input_text="Should I move to a new city?\n",
            )
            self.assertEqual(completed.returncode, 1)
            self.assertIn("category=transport_runtime", completed.stderr)

    def test_personal_interactive_adopt_suggest_shows_user_friendly_confirmation(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            workspace = Path(tmp_dir) / "personal_workspace"
            completed = self._run_personal(
                "--workspace",
                str(workspace),
                "--model",
                "deterministic",
                input_text="Should I move to a new city?\ny\nexit\n",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("✅ Your choice has been recorded: B", completed.stdout)
            self.assertIn("Next steps:", completed.stdout)
            self.assertNotIn("decision_id=", completed.stdout)
            self.assertNotIn("action=", completed.stdout)

    def test_personal_interactive_session_verbose_keeps_debug_logs(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            workspace = Path(tmp_dir) / "personal_workspace"
            completed = self._run_personal(
                "--workspace",
                str(workspace),
                "--model",
                "deterministic",
                "--verbose",
                input_text="Should I move to a new city?\nexit\n",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("executor_mode=mock", completed.stdout)
            self.assertIn("action=", completed.stdout)
            self.assertIn("decision_id=", completed.stdout)
            self.assertIn("advisory_outcome_id=", completed.stdout)

    def test_personal_session_fails_fast_when_profile_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            workspace = Path(tmp_dir) / "personal_workspace"
            init_completed = self._run_personal(
                "init",
                "--workspace",
                str(workspace),
            )
            self.assertEqual(init_completed.returncode, 0, init_completed.stderr)

            profile_path = workspace / "config" / "personal.profile.json"
            payload = json.loads(profile_path.read_text(encoding="utf-8"))
            routes = dict(payload.get("category_routes", {}))
            routes.pop("external.schedule", None)
            payload["category_routes"] = routes
            profile_path.write_text(
                json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            completed = self._run_personal(
                "--workspace",
                str(workspace),
                "--model",
                "deterministic",
                input_text="exit\n",
            )
            self.assertEqual(completed.returncode, 1)
            self.assertIn("Profile validation failed", completed.stderr)

    def test_personal_ask_with_llm_model_uses_llm_suggestion(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            root = Path(tmp_dir)
            workspace = root / "personal_workspace"
            model_script = root / "llm_personal_model.py"
            model_script.write_text(self._valid_personal_model_script(), encoding="utf-8")
            model_cmd = f"{sys.executable} {model_script}"

            completed = self._run_personal(
                "ask",
                "Should I move for work?",
                "--workspace",
                str(workspace),
                "--model",
                model_cmd,
                "--verbose",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("action=personal.assistant.ask_clarify", completed.stdout)
            self.assertIn("decision_adoption_status=pending", completed.stdout)
            self.assertIn("execution_status=not_requested", completed.stdout)
            self.assertIn("suggestion=Clarifying Questions", completed.stdout)
            self.assertIn("Q1:", completed.stdout)
            self.assertFalse((workspace / "artifacts" / "personal_state.json").exists())

    def test_personal_ask_prints_three_options_and_recommended_option(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            root = Path(tmp_dir)
            workspace = root / "personal_workspace"
            model_script = root / "llm_three_options_model.py"
            model_script.write_text(self._three_options_model_script(), encoding="utf-8")
            model_cmd = f"{sys.executable} {model_script}"

            completed = self._run_personal(
                "ask",
                "I have two job offers, help me decide.",
                "--workspace",
                str(workspace),
                "--model",
                model_cmd,
                "--verbose",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("decision_options_count=3", completed.stdout)
            self.assertIn("recommended_option_id=", completed.stdout)
            self.assertIn("option_1_action=", completed.stdout)
            self.assertIn("option_2_action=", completed.stdout)
            self.assertIn("option_3_action=", completed.stdout)
            self.assertIn("option_1_first_step_24h=", completed.stdout)
            self.assertIn("option_1_stop_loss_trigger=", completed.stdout)
            self.assertIn("option_1_change_mind_condition=", completed.stdout)

    def test_personal_ask_renders_structured_decision_brain_report_when_available(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            root = Path(tmp_dir)
            workspace = root / "personal_workspace"
            model_script = root / "llm_structured_report_model.py"
            model_script.write_text(self._structured_report_model_script(), encoding="utf-8")
            model_cmd = f"{sys.executable} {model_script}"

            completed = self._run_personal(
                "ask",
                "I have offer A and offer B, help me choose.",
                "--workspace",
                str(workspace),
                "--model",
                model_cmd,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("Spice recommendation", completed.stdout)
            self.assertIn("方案 A（偏收益）", completed.stdout)
            self.assertIn("方案 B（偏长期成长）", completed.stdout)
            self.assertIn("方案 C（折中策略）", completed.stdout)
            self.assertIn("★ 方案 B（偏长期成长）", completed.stdout)
            self.assertIn("若 A 能提供可验证稳定性和明确管理路径", completed.stdout)
            self.assertNotIn("option_1_id=", completed.stdout)

    def test_personal_ask_evidence_round_refines_advice(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            root = Path(tmp_dir)
            workspace = root / "personal_workspace"
            model_script = root / "llm_evidence_model.py"
            model_script.write_text(self._evidence_personal_model_script(), encoding="utf-8")
            model_cmd = f"{sys.executable} {model_script}"

            completed = self._run_personal(
                "ask",
                "Should I switch teams this quarter?",
                "--workspace",
                str(workspace),
                "--model",
                model_cmd,
                "--verbose",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn(
                "notice=Evidence action selected, but external evidence agent is not configured (executor=mock).",
                completed.stdout,
            )
            self.assertIn("action=personal.assistant.gather_evidence", completed.stdout)
            self.assertIn("decision_adoption_status=pending", completed.stdout)
            self.assertIn("execution_status=not_requested", completed.stdout)
            self.assertIn(
                "suggestion=Evidence Collection Plan",
                completed.stdout,
            )
            self.assertFalse((workspace / "artifacts" / "personal_state.json").exists())

    def test_personal_interactive_with_llm_persists_llm_suggestion(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            root = Path(tmp_dir)
            workspace = root / "personal_workspace"
            model_script = root / "llm_personal_model.py"
            model_script.write_text(self._valid_personal_model_script(), encoding="utf-8")
            model_cmd = f"{sys.executable} {model_script}"

            completed = self._run_personal(
                "--workspace",
                str(workspace),
                "--model",
                model_cmd,
                input_text="Should I move to a new city?\nexit\n",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            state_path = workspace / "artifacts" / "personal_state.json"
            self.assertTrue(state_path.exists())
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            entities = payload.get("entities", {})
            self.assertIsInstance(entities, dict)
            current = entities.get("personal.assistant.current", {})
            self.assertIsInstance(current.get("latest_suggestion"), str)
            self.assertIn("Clarifying Questions", current.get("latest_suggestion", ""))
            self.assertIn("Q1:", current.get("latest_suggestion", ""))
            recent_outcomes = payload.get("recent_outcomes", [])
            self.assertIsInstance(recent_outcomes, list)
            self.assertGreaterEqual(len(recent_outcomes), 1)

    def test_personal_interactive_evidence_round_persists_refined_suggestion(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            root = Path(tmp_dir)
            workspace = root / "personal_workspace"
            model_script = root / "llm_evidence_model.py"
            model_script.write_text(self._evidence_personal_model_script(), encoding="utf-8")
            model_cmd = f"{sys.executable} {model_script}"

            completed = self._run_personal(
                "--workspace",
                str(workspace),
                "--model",
                model_cmd,
                input_text="Should I relocate for a new role?\nn\nexit\n",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn(
                "advisor> Evidence Collection Plan",
                completed.stdout,
            )
            self.assertIn(
                "start evidence collection now and provide manual evidence next? [Y/n] ",
                completed.stdout,
            )
            self.assertNotIn("decision_adoption_status=", completed.stdout)
            self.assertNotIn("evidence_state=", completed.stdout)
            state_path = workspace / "artifacts" / "personal_state.json"
            self.assertTrue(state_path.exists())
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            entities = payload.get("entities", {})
            self.assertIsInstance(entities, dict)
            current = entities.get("personal.assistant.current", {})
            self.assertIn("Evidence Collection Plan", current.get("latest_suggestion", ""))
            self.assertEqual(current.get("latest_question"), "Should I relocate for a new role?")

    def test_personal_interactive_mock_evidence_checklist_followup_reaches_suggest(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            root = Path(tmp_dir)
            workspace = root / "personal_workspace"
            model_script = root / "llm_evidence_followup_model.py"
            model_script.write_text(self._evidence_followup_personal_model_script(), encoding="utf-8")
            model_cmd = f"{sys.executable} {model_script}"

            completed = self._run_personal(
                "--workspace",
                str(workspace),
                "--model",
                model_cmd,
                input_text=(
                    "I have offer A and offer B, which should I pick?\n"
                    "y\n"
                    "A attrition is high; B attrition is low and manager mentorship is strong.\n"
                    "n\n"
                    "exit\n"
                ),
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("Waiting for your manual evidence input", completed.stdout)
            self.assertNotIn("action=personal.assistant.suggest", completed.stdout)
            self.assertIn(
                "Given the verified evidence, prioritize Option B and set a 90-day management-skill checkpoint.",
                completed.stdout,
            )

    def test_personal_ask_fails_when_llm_output_is_malformed(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            root = Path(tmp_dir)
            workspace = root / "personal_workspace"
            model_script = root / "llm_bad_json.py"
            model_script.write_text("print('not-json')\n", encoding="utf-8")
            model_cmd = f"{sys.executable} {model_script}"

            completed = self._run_personal(
                "ask",
                "Should I switch teams?",
                "--workspace",
                str(workspace),
                "--model",
                model_cmd,
            )

            self.assertEqual(completed.returncode, 1)
            self.assertIn("category=response_validity", completed.stderr)

    def test_personal_ask_accepts_action_type_fields_with_personal_alias_mapping(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            root = Path(tmp_dir)
            workspace = root / "personal_workspace"
            model_script = root / "llm_compat_alias.py"
            model_script.write_text(self._compat_alias_model_script(), encoding="utf-8")
            model_cmd = f"{sys.executable} {model_script}"

            completed = self._run_personal(
                "ask",
                "Should I move to a new city?",
                "--workspace",
                str(workspace),
                "--model",
                model_cmd,
                "--verbose",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("action=personal.assistant.suggest", completed.stdout)
            self.assertIn("suggestion=Spice recommendation", completed.stdout)
            self.assertIn("Alias-mapped advisory from compatibility path", completed.stdout)

    def test_personal_ask_fails_when_llm_action_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            root = Path(tmp_dir)
            workspace = root / "personal_workspace"
            model_script = root / "llm_invalid_action.py"
            model_script.write_text(self._invalid_action_model_script(), encoding="utf-8")
            model_cmd = f"{sys.executable} {model_script}"

            completed = self._run_personal(
                "ask",
                "Should I change my role?",
                "--workspace",
                str(workspace),
                "--model",
                model_cmd,
            )

            self.assertEqual(completed.returncode, 1)
            self.assertIn("category=unsupported_capability", completed.stderr)

    def test_personal_ask_fails_when_action_field_is_unknown(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            root = Path(tmp_dir)
            workspace = root / "personal_workspace"
            model_script = root / "llm_unknown_action_field.py"
            model_script.write_text(self._unknown_action_field_model_script(), encoding="utf-8")
            model_cmd = f"{sys.executable} {model_script}"

            completed = self._run_personal(
                "ask",
                "Should I change my role?",
                "--workspace",
                str(workspace),
                "--model",
                model_cmd,
            )

            self.assertEqual(completed.returncode, 1)
            self.assertIn("category=unsupported_capability", completed.stderr)

    def test_personal_ask_fails_when_llm_command_fails(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            workspace = Path(tmp_dir) / "personal_workspace"
            completed = self._run_personal(
                "ask",
                "Should I switch projects?",
                "--workspace",
                str(workspace),
                "--model",
                "missing_command_that_should_fail",
            )

            self.assertEqual(completed.returncode, 1)
            self.assertIn("category=transport_runtime", completed.stderr)

    @staticmethod
    def _run_personal(
        *args: str,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "spice_personal.cli", *args],
            cwd=REPO_ROOT,
            text=True,
            input=input_text,
            capture_output=True,
            check=False,
        )

    @staticmethod
    def _valid_personal_model_script() -> str:
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
            "        'score': 0.95,\n"
            "        'confidence': 0.84,\n"
            "        'urgency': 'high',\n"
            "        'suggestion_text': 'LLM advisory: clarify your top tradeoff before deciding.',\n"
            "        'simulation_rationale': 'high_uncertainty_requires_clarification',\n"
            "        'clarifying_questions': [\n"
            "            {'question': 'What is your top non-negotiable in the next role?', 'why': 'It can reorder options by filtering hard constraints first.'},\n"
            "            {'question': 'How much short-term volatility can you tolerate over 12 months?', 'why': 'It can change risk-adjusted ranking between stable and high-upside options.'},\n"
            "            {'question': 'What 3-year management milestone matters most?', 'why': 'It can shift recommendation toward the option with stronger management-path evidence.'},\n"
            "        ],\n"
            "    }\n"
            "else:\n"
            "    payload = {'score': 0.0}\n"
            "print(json.dumps(payload))\n"
        )

    @staticmethod
    def _three_options_model_script() -> str:
        return (
            "import json\n"
            "import sys\n"
            "prompt = sys.stdin.read()\n"
            "if 'Decision proposals' in prompt:\n"
            "    payload = [\n"
            "        {\n"
            "            'decision_type': 'personal.assistant.llm',\n"
            "            'status': 'proposed',\n"
            "            'selected_action': 'personal.assistant.suggest',\n"
            "            'attributes': {'confidence': 0.60, 'urgency': 'normal'},\n"
            "        },\n"
            "        {\n"
            "            'decision_type': 'personal.assistant.llm',\n"
            "            'status': 'proposed',\n"
            "            'selected_action': 'personal.assistant.ask_clarify',\n"
            "            'attributes': {'confidence': 0.80, 'urgency': 'high'},\n"
            "        },\n"
            "        {\n"
            "            'decision_type': 'personal.assistant.llm',\n"
            "            'status': 'proposed',\n"
            "            'selected_action': 'personal.assistant.defer',\n"
            "            'attributes': {'confidence': 0.55, 'urgency': 'normal'},\n"
            "        },\n"
            "    ]\n"
            "elif 'simulation advice' in prompt:\n"
            "    if 'personal.assistant.ask_clarify' in prompt:\n"
            "        payload = {\n"
            "            'score': 0.95,\n"
            "            'confidence': 0.82,\n"
            "            'urgency': 'high',\n"
            "            'suggestion_text': 'Ask one clarifying question before committing.',\n"
            "            'simulation_rationale': 'best_tradeoff',\n"
            "            'clarifying_questions': [\n"
            "                {'question': 'What is your non-negotiable in the next 12 months?', 'why': 'It can eliminate options that violate hard constraints.'},\n"
            "                {'question': 'How much downside can you tolerate this year?', 'why': 'It can reorder stable and high-variance options.'},\n"
            "                {'question': 'What 3-year outcome defines success?', 'why': 'It can change which option best supports long-term goals.'},\n"
            "            ],\n"
            "        }\n"
            "    elif 'personal.assistant.defer' in prompt:\n"
            "        payload = {\n"
            "            'score': 0.70,\n"
            "            'confidence': 0.65,\n"
            "            'urgency': 'normal',\n"
            "            'suggestion_text': 'Defer one week pending one extra signal.',\n"
            "            'simulation_rationale': 'uncertainty',\n"
            "            'defer_plan': {\n"
            "                'revisit_at': '7 days',\n"
            "                'monitor_signal': 'verified manager quality and team-stability update',\n"
            "                'resume_trigger': 'resume when both signals are available',\n"
            "            },\n"
            "        }\n"
            "    else:\n"
            "        payload = {\n"
            "            'score': 0.62,\n"
            "            'confidence': 0.58,\n"
            "            'urgency': 'normal',\n"
            "            'suggestion_text': 'Take a small reversible next step now.',\n"
            "            'simulation_rationale': 'baseline',\n"
            "            'benefits': ['Preserves optionality while keeping progress.'],\n"
            "            'risks': ['May delay upside if decision window closes quickly.'],\n"
            "            'key_assumptions': ['Both offers remain open long enough for one verification step.'],\n"
            "            'first_step_24h': 'Within 24h, confirm role scope and manager expectations for each option.',\n"
            "            'stop_loss_trigger': 'If scope clarity is missing by day 7, pause and re-evaluate.',\n"
            "            'change_mind_condition': 'Switch if verified stability and mentorship signals materially reverse.',\n"
            "        }\n"
            "else:\n"
            "    payload = {'score': 0.0}\n"
            "print(json.dumps(payload))\n"
        )

    @staticmethod
    def _structured_report_model_script() -> str:
        return (
            "import json\n"
            "import sys\n"
            "prompt = sys.stdin.read()\n"
            "if 'Decision proposals' in prompt:\n"
            "    payload = [\n"
            "        {\n"
            "            'decision_type': 'personal.assistant.llm',\n"
            "            'status': 'proposed',\n"
            "            'selected_action': 'personal.assistant.suggest',\n"
            "            'attributes': {'confidence': 0.86, 'urgency': 'high'},\n"
            "        }\n"
            "    ]\n"
            "elif 'simulation advice' in prompt:\n"
            "    payload = {\n"
            "        'score': 0.91,\n"
            "        'confidence': 0.86,\n"
            "        'urgency': 'high',\n"
            "        'suggestion_text': 'Given your management goal and medium risk tolerance, Option B is recommended.',\n"
            "        'benefits': ['Stable environment for leadership development.'],\n"
            "        'risks': ['Short-term compensation upside is lower.'],\n"
            "        'key_assumptions': ['Mentor support converts into real ownership.'],\n"
            "        'first_step_24h': 'Confirm manager expectations and ownership scope.',\n"
            "        'stop_loss_trigger': 'Reevaluate if ownership is unclear by week 4.',\n"
            "        'change_mind_condition': 'Switch if verified data strongly favors Option A trajectory.',\n"
            "        'decision_brain_report': {\n"
            "            'options': [\n"
            "                {\n"
            "                    'label': '方案 A（偏收益）',\n"
            "                    'benefits': ['短期现金流提升和议价空间更大。'],\n"
            "                    'risks': ['团队不稳定可能削弱管理成长连续性。'],\n"
            "                    'key_assumptions': ['高薪会带来清晰的管理 exposure。'],\n"
            "                    'first_step_24h': '向 A 方确认 6-12 个月组织稳定性与晋升路径。',\n"
            "                    'stop_loss_trigger': '90 天内组织持续变动且管理路径不清晰则重评。'\n"
            "                },\n"
            "                {\n"
            "                    'label': '方案 B（偏长期成长）',\n"
            "                    'benefits': ['稳定环境和导师支持更贴合 3 年管理目标。'],\n"
            "                    'risks': ['短期薪资机会成本。'],\n"
            "                    'key_assumptions': ['导师可提供真实带人机会与关键项目 ownership。'],\n"
            "                    'first_step_24h': '确认管理培养计划、带人机会时间点、晋升标准。',\n"
            "                    'stop_loss_trigger': '6 个月内无带人或 owner 机会则重评。'\n"
            "                },\n"
            "                {\n"
            "                    'label': '方案 C（折中策略）',\n"
            "                    'benefits': ['保留成长主线同时争取更好补偿条款。'],\n"
            "                    'risks': ['谈判失败可能影响 offer 关系。'],\n"
            "                    'key_assumptions': ['B 方对候选人重视且有谈判空间。'],\n"
            "                    'first_step_24h': '基于 A offer 与 B 进行有边界的补偿+成长承诺谈判。',\n"
            "                    'stop_loss_trigger': '若 B 不改关键条款且 A 无稳定性证据则转 defer。'\n"
            "                }\n"
            "            ],\n"
            "            'recommended_option_label': '方案 B（偏长期成长）',\n"
            "            'recommendation_reason': '它更稳定地提升你在 3 年内达成管理岗目标的概率。',\n"
            "            'what_would_change_my_mind': '若 A 能提供可验证稳定性和明确管理路径，且成长机会显著高于 B。'\n"
            "        }\n"
            "    }\n"
            "else:\n"
            "    payload = {'score': 0.0}\n"
            "print(json.dumps(payload, ensure_ascii=False))\n"
        )

    @staticmethod
    def _invalid_action_model_script() -> str:
        return (
            "import json\n"
            "import sys\n"
            "prompt = sys.stdin.read()\n"
            "if 'Decision proposals' in prompt:\n"
            "    payload = [\n"
            "        {\n"
            "            'decision_type': 'personal.assistant.llm',\n"
            "            'status': 'proposed',\n"
            "            'selected_action': 'personal.assistant.invalid_action',\n"
            "        }\n"
            "    ]\n"
            "else:\n"
            "    payload = {'score': 0.1, 'suggestion_text': 'this should not be used'}\n"
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
            "                'attributes': {'confidence': 0.91, 'urgency': 'normal'},\n"
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

    @staticmethod
    def _evidence_followup_personal_model_script() -> str:
        return (
            "import json\n"
            "import sys\n"
            "from pathlib import Path\n"
            "prompt = sys.stdin.read()\n"
            "counter_path = Path(__file__).with_suffix('.counter')\n"
            "try:\n"
            "    invocation = int(counter_path.read_text(encoding='utf-8').strip() or '0')\n"
            "except Exception:\n"
            "    invocation = 0\n"
            "invocation += 1\n"
            "counter_path.write_text(str(invocation), encoding='utf-8')\n"
            "has_manual_evidence = ('User evidence response:' in prompt) or invocation >= 3\n"
            "if 'Decision proposals' in prompt:\n"
            "    if has_manual_evidence:\n"
            "        payload = [\n"
            "            {\n"
            "                'decision_type': 'personal.assistant.llm',\n"
            "                'status': 'proposed',\n"
            "                'selected_action': 'personal.assistant.suggest',\n"
            "                'attributes': {'confidence': 0.86, 'urgency': 'normal'},\n"
            "            }\n"
            "        ]\n"
            "    else:\n"
            "        payload = [\n"
            "            {\n"
            "                'decision_type': 'personal.assistant.llm',\n"
            "                'status': 'proposed',\n"
            "                'selected_action': 'personal.assistant.gather_evidence',\n"
            "                'attributes': {'confidence': 0.50, 'urgency': 'normal'},\n"
            "            }\n"
            "        ]\n"
            "elif 'simulation advice' in prompt:\n"
            "    if '\"selected_action\": \"personal.assistant.gather_evidence\"' in prompt:\n"
            "        payload = {\n"
            "            'score': 0.82,\n"
            "            'confidence': 0.50,\n"
            "            'urgency': 'normal',\n"
            "            'suggestion_text': 'Collect externally verifiable evidence before finalizing the recommendation.',\n"
            "            'simulation_rationale': 'evidence_first',\n"
            "            'evidence_plan': [\n"
            "                {'fact': 'Compare manager coaching track records for A and B.', 'why': 'Manager leverage affects promotion trajectory.'},\n"
            "                {'fact': 'Verify attrition trend over the last 12 months.', 'why': 'High churn increases execution risk.'},\n"
            "                {'fact': 'Check scope growth opportunities in first 6 months.', 'why': 'Scope growth is a leading signal for management readiness.'},\n"
            "            ],\n"
            "        }\n"
            "    else:\n"
            "        payload = {\n"
            "            'score': 0.90,\n"
            "            'confidence': 0.86,\n"
            "            'urgency': 'normal',\n"
            "            'suggestion_text': 'Given the verified evidence, prioritize Option B and set a 90-day management-skill checkpoint.',\n"
            "            'simulation_rationale': 'manual_evidence_applied',\n"
            "            'benefits': ['Higher manager quality improves promotion readiness.'],\n"
            "            'risks': ['Short-term compensation upside is lower than Option A.'],\n"
            "            'key_assumptions': ['Mentorship quality stays consistent for at least 12 months.'],\n"
            "            'first_step_24h': 'Book a call with the hiring manager to align on first-quarter scope ownership.',\n"
            "            'stop_loss_trigger': 'If scope ownership is not confirmed by week 4, reopen Option A evaluation.',\n"
            "            'change_mind_condition': 'Switch if verified attrition and mentorship signals converge in favor of Option A.',\n"
            "        }\n"
            "else:\n"
            "    payload = {'score': 0.0}\n"
            "print(json.dumps(payload))\n"
        )

    @staticmethod
    def _compat_alias_model_script() -> str:
        return (
            "import json\n"
            "import sys\n"
            "prompt = sys.stdin.read()\n"
            "if 'Decision proposals' in prompt:\n"
            "    payload = {\n"
            "        'candidates': [\n"
            "            {\n"
            "                'id': 'decision-direct-respond-ac94ee36',\n"
            "                'action': 'personal.assistant.answer',\n"
            "                'type': 'action_response',\n"
            "                'description': 'Formulate and output a direct answer based on the received question and available context.',\n"
            "            },\n"
            "            {\n"
            "                'id': 'decision-clarify-req-0494438d',\n"
            "                'action': 'personal.assistant.question_refinement',\n"
            "                'type': 'action_follow_up',\n"
            "                'description': 'Request additional context or clarification to ensure an accurate response.',\n"
            "            },\n"
            "            {\n"
            "                'id': 'decision-defer-0494438db89f',\n"
            "                'action': 'personal.assistant.retrieval_task',\n"
            "                'type': 'action_search',\n"
            "                'description': 'Delegate the query to a search tool or knowledge base before responding.',\n"
            "            },\n"
            "        ]\n"
            "    }\n"
            "elif 'simulation advice' in prompt:\n"
            "    payload = {\n"
            "        'score': 0.93,\n"
            "        'confidence': 0.77,\n"
            "        'urgency': 'normal',\n"
            "        'suggestion_text': 'Alias-mapped advisory from compatibility path: Option B better fits your management goal with lower team risk.',\n"
            "        'simulation_rationale': 'compatibility_alias_test',\n"
            "        'benefits': ['Maintains progress while reducing avoidable downside.'],\n"
            "        'risks': ['Could delay upside if the opportunity window is short.'],\n"
            "        'key_assumptions': ['Current constraints and opportunity remain stable this month.'],\n"
            "        'first_step_24h': 'Within 24h, confirm manager expectations and scope across options.',\n"
            "        'stop_loss_trigger': 'If role scope remains ambiguous by week 1, pause and reassess.',\n"
            "        'change_mind_condition': 'Switch if verified stability and growth signals materially reverse.',\n"
            "    }\n"
            "else:\n"
            "    payload = {'score': 0.0}\n"
            "print(json.dumps(payload))\n"
        )

    @staticmethod
    def _unknown_action_field_model_script() -> str:
        return (
            "import json\n"
            "import sys\n"
            "prompt = sys.stdin.read()\n"
            "if 'Decision proposals' in prompt:\n"
            "    payload = {\n"
            "        'candidates': [\n"
            "            {\n"
            "                'id': 'decision-unknown-action-1',\n"
            "                'action': 'personal.assistant.not_supported_action',\n"
            "                'type': 'action_response',\n"
            "            }\n"
            "        ]\n"
            "    }\n"
            "else:\n"
            "    payload = {'score': 0.1, 'suggestion_text': 'this should not be used'}\n"
            "print(json.dumps(payload))\n"
        )


if __name__ == "__main__":
    unittest.main()
