from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from spice_personal.advisory import personal_advisory as advisory_module
from spice_personal.executors.factory import PersonalExecutorConfig
from spice_personal.profile.contract import (
    CATEGORY_EXTERNAL_EVIDENCE,
    PROFILE_SCHEMA_VERSION,
)
from spice_personal.profile.loader import ensure_workspace_profile, load_profile
from spice_personal.profile.validate import validate_profile_contract


REPO_ROOT = Path(__file__).resolve().parents[2]


class ProfileContractV1Tests(unittest.TestCase):
    def test_default_profile_schema_and_p0_coverage_validate(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            workspace = Path(tmp_dir) / "personal_workspace"
            ensure_workspace_profile(workspace, force=False)
            profile_path = workspace / "config" / "personal.profile.json"
            profile = load_profile(profile_path)

            validation = validate_profile_contract(
                profile,
                profile_path=profile_path,
                executor_config=PersonalExecutorConfig(mode="mock"),
            )

            self.assertFalse(validation.errors)
            self.assertEqual(profile.get("schema_version"), PROFILE_SCHEMA_VERSION)
            routes = profile.get("category_routes", {})
            self.assertTrue(routes.get("external.evidence", {}).get("enabled"))
            self.assertTrue(routes.get("external.system", {}).get("enabled"))
            self.assertTrue(routes.get("external.communicate", {}).get("enabled"))
            self.assertTrue(routes.get("external.schedule", {}).get("enabled"))

    def test_missing_p0_category_fails_validation(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            workspace = Path(tmp_dir) / "personal_workspace"
            ensure_workspace_profile(workspace, force=False)
            profile_path = workspace / "config" / "personal.profile.json"
            profile = load_profile(profile_path)
            routes = dict(profile.get("category_routes", {}))
            routes.pop("external.schedule", None)
            profile["category_routes"] = routes
            profile_path.write_text(
                json.dumps(profile, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            validation = validate_profile_contract(
                profile,
                profile_path=profile_path,
                executor_config=PersonalExecutorConfig(mode="mock"),
            )

            self.assertTrue(validation.errors)
            self.assertTrue(
                any(issue.code == "profile.category.missing_p0" for issue in validation.errors)
            )

    def test_action_proposal_advisory_attributes_include_minimum_execution_brief(self) -> None:
        attributes = advisory_module._normalize_advisory_attributes(
            selected_action="personal.assistant.gather_evidence",
            suggestion_text="collect one evidence snapshot",
            result_kind=advisory_module.RESULT_KIND_ACTION_PROPOSAL,
            execution_brief={},
        )

        execution_brief = attributes.get("execution_brief", {})
        self.assertIsInstance(execution_brief, dict)
        self.assertEqual(execution_brief.get("category"), CATEGORY_EXTERNAL_EVIDENCE)
        self.assertTrue(str(execution_brief.get("goal", "")).strip())
        success_criteria = execution_brief.get("success_criteria")
        self.assertIsInstance(success_criteria, list)
        self.assertGreaterEqual(len(success_criteria), 1)

    def test_cli_mode_profile_validation_passes_with_command_and_p0_mappings(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            workspace = Path(tmp_dir) / "personal_workspace"
            ensure_workspace_profile(workspace, force=False)
            profile_path = workspace / "config" / "personal.profile.json"
            profile = load_profile(profile_path)
            profile["executor_mode"] = "cli"
            profile_path.write_text(
                json.dumps(profile, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            validation = validate_profile_contract(
                profile,
                profile_path=profile_path,
                executor_config=PersonalExecutorConfig(mode="cli", cli_command="python3 -c 'print(1)'"),
            )

            self.assertFalse(validation.errors)

    def test_invalid_p1_route_emits_warning_not_error(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp_dir:
            workspace = Path(tmp_dir) / "personal_workspace"
            ensure_workspace_profile(workspace, force=False)
            profile_path = workspace / "config" / "personal.profile.json"
            profile = load_profile(profile_path)
            route = dict(profile.get("category_routes", {}).get("external.manage_task", {}))
            route["enabled"] = True
            route["target"] = {"kind": "external.service"}
            profile["category_routes"]["external.manage_task"] = route
            profile_path.write_text(
                json.dumps(profile, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            validation = validate_profile_contract(
                profile,
                profile_path=profile_path,
                executor_config=PersonalExecutorConfig(mode="mock"),
            )

            self.assertFalse(validation.errors)
            self.assertTrue(validation.warnings)


if __name__ == "__main__":
    unittest.main()
