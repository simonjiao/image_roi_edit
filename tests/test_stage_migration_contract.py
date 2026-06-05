from __future__ import annotations

import ast
from pathlib import Path
import unittest

from roi_image_edit.stage_migration import stage_migration_contract_report
from roi_image_edit.stage_policy import STAGE_ORDER


REPO_ROOT = Path(__file__).resolve().parents[1]


class StageMigrationContractTest(unittest.TestCase):
    def test_every_stage_migration_declares_detector_patcher_and_evidence(self) -> None:
        report = stage_migration_contract_report()

        self.assertEqual(tuple(report["stage_order"]), STAGE_ORDER)
        self.assertEqual(set(report["stages"]), set(STAGE_ORDER))
        for stage_id in STAGE_ORDER:
            stage = report["stages"][stage_id]
            self.assertEqual(stage["detector"], f"detect_{stage_id}")
            self.assertEqual(stage["detector_module"], "roi_image_edit.stages")
            self.assertIsInstance(stage["allowed_patch_keys"], list)
            self.assertIsInstance(stage["blocked_patch_keys"], list)
            if stage_id == "hard_boundary":
                self.assertIsNone(stage["patcher"])
            else:
                self.assertEqual(stage["patcher"]["stage_id"], stage_id)
                self.assertEqual(stage["patcher"]["primary_stage"], stage_id)
                self.assertEqual(stage["patcher"]["allowed_patch_keys"], stage["allowed_patch_keys"])

            failure_case = stage["failure_case"]
            self.assertEqual(failure_case["blocking_stage"], stage_id)
            failure_evidence = failure_case["stage_evidence"]
            self.assertEqual(failure_evidence["blocking_stage"], stage_id)
            self.assertFalse(failure_evidence["stage_status"][stage_id]["pass"])
            self.assertEqual(failure_evidence["allowed_patch_keys"], stage["allowed_patch_keys"])
            self.assertEqual(failure_evidence["blocked_patch_keys"], stage["blocked_patch_keys"])

            pass_case = stage["pass_case"]
            self.assertIsNone(pass_case["blocking_stage"])
            self.assertIsNone(pass_case["stage_evidence"]["blocking_stage"])
            self.assertTrue(all(item["pass"] for item in pass_case["stage_evidence"]["stage_status"].values()))

    def test_stage_policy_does_not_call_local_issue_detectors(self) -> None:
        policy_path = REPO_ROOT / "src" / "roi_image_edit" / "stage_policy.py"
        tree = ast.parse(policy_path.read_text(encoding="utf-8"), filename=str(policy_path))
        imported_modules: list[str] = []
        local_issue_calls: list[str] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported_modules.append(str(node.module or ""))
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            else:
                continue
            if name.startswith("local_") and name.endswith("_issues"):
                local_issue_calls.append(f"{node.lineno}:{name}")

        self.assertNotIn("roi_image_edit.local_validation", imported_modules)
        self.assertEqual(local_issue_calls, [])

    def test_local_validation_stage_gate_delegates_to_stages_module(self) -> None:
        validation_path = REPO_ROOT / "src" / "roi_image_edit" / "local_validation.py"
        tree = ast.parse(validation_path.read_text(encoding="utf-8"), filename=str(validation_path))
        stage_gate_function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_stage_gate_for_report"
        )
        calls: list[str] = []
        for node in ast.walk(stage_gate_function):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)

        self.assertEqual(calls, ["canonical_stage_gate_for_report", "str", "get"])


if __name__ == "__main__":
    unittest.main()
