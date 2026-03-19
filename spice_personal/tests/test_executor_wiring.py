from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ECHO_AGENT = REPO_ROOT / "examples" / "sdep_agent_demo" / "echo_agent.py"


class PersonalExecutorWiringTests(unittest.TestCase):
    def test_personal_ask_cli_executor_mode_supports_evidence_round(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            root = Path(tmp_dir)
            workspace = root / "personal_workspace"
            model_script = root / "llm_evidence_model.py"
            cli_agent = root / "cli_agent.py"
            model_script.write_text(self._evidence_personal_model_script(), encoding="utf-8")
            cli_agent.write_text(self._cli_agent_script(), encoding="utf-8")

            model_cmd = f"{sys.executable} {model_script}"
            cli_cmd = f"{sys.executable} {cli_agent}"
            completed = self._run_personal(
                "ask",
                "Should I switch teams now?",
                "--workspace",
                str(workspace),
                "--model",
                model_cmd,
                "--executor",
                "cli",
                "--cli-command",
                cli_cmd,
                "--verbose",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("executor_mode=cli", completed.stdout)
            self.assertIn("notice=Gathering one bounded evidence snapshot before final advice.", completed.stdout)
            self.assertIn("action=personal.assistant.suggest", completed.stdout)

    def test_personal_ask_sdep_executor_mode_supports_evidence_round(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            root = Path(tmp_dir)
            workspace = root / "personal_workspace"
            model_script = root / "llm_evidence_model.py"
            model_script.write_text(self._evidence_personal_model_script(), encoding="utf-8")
            model_cmd = f"{sys.executable} {model_script}"
            sdep_cmd = f"{sys.executable} {ECHO_AGENT}"

            completed = self._run_personal(
                "ask",
                "Should I take the offer?",
                "--workspace",
                str(workspace),
                "--model",
                model_cmd,
                "--executor",
                "sdep",
                "--sdep-command",
                sdep_cmd,
                "--verbose",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("executor_mode=sdep", completed.stdout)
            self.assertIn("notice=Gathering one bounded evidence snapshot before final advice.", completed.stdout)
            self.assertIn("action=personal.assistant.suggest", completed.stdout)

    @staticmethod
    def _run_personal(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "spice_personal.cli", *args],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    @staticmethod
    def _cli_agent_script() -> str:
        return (
            "import json\n"
            "import sys\n"
            "_ = json.loads(sys.stdin.read() or '{}')\n"
            "print(json.dumps({'outcome_type':'observation','summary':'cli evidence ok','items':[{'id':'item-1','text':'source-a'}]}))\n"
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


if __name__ == "__main__":
    unittest.main()
