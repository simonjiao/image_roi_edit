from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path
import unittest

from roi_image_edit.stage_patchers import filter_patches_for_stage, patch_allowed_for_stage
from roi_image_edit.stage_policy import STAGE_LABELS, STAGE_ORDER
from roi_image_edit.stage_profiles import stage_profile, stage_profile_choices
from roi_image_edit.stages import (
    STAGE_SPECS,
    StageResult,
    StageSpec,
    prompt_stage_context,
    stage_gate_for_report,
    stage_specs,
)


EXPECTED_STAGE_ORDER = (
    "hard_boundary",
    "text_shape",
    "ink_gray_balance",
    "photo_texture",
    "background_cleanup",
)


class StageContractsTest(unittest.TestCase):
    def test_stage_order_is_the_five_stage_contract(self) -> None:
        self.assertEqual(STAGE_ORDER, EXPECTED_STAGE_ORDER)
        self.assertEqual(tuple(STAGE_SPECS), EXPECTED_STAGE_ORDER)
        self.assertEqual(tuple(spec.id for spec in stage_specs("photo_scan")), EXPECTED_STAGE_ORDER)
        self.assertEqual(set(STAGE_LABELS), set(EXPECTED_STAGE_ORDER))

    def test_stage_spec_field_contract_and_reports(self) -> None:
        self.assertEqual(
            tuple(field.name for field in fields(StageSpec)),
            (
                "id",
                "display_name",
                "blocks_next",
                "detect",
                "optimization_steps",
                "allowed_patch_keys",
                "blocked_patch_keys",
            ),
        )
        for spec in stage_specs("photo_scan"):
            report = spec.as_report()
            self.assertNotIn("detect", report)
            self.assertEqual(report["id"], spec.id)
            self.assertIsInstance(report["optimization_steps"], tuple)
            self.assertIsInstance(report["allowed_patch_keys"], list)
            self.assertIsInstance(report["blocked_patch_keys"], list)

    def test_stage_result_field_contract_and_stage_context(self) -> None:
        self.assertEqual(
            tuple(field.name for field in fields(StageResult)),
            (
                "stage_id",
                "display_name",
                "passed",
                "blocks_next",
                "severity",
                "issues",
                "reason",
                "allowed_patch_keys",
                "blocked_patch_keys",
            ),
        )
        gate = stage_gate_for_report({"pass": False, "issues": [{"type": "roi_outside"}]})
        self.assertFalse(gate["pass"])
        self.assertEqual(gate["blocking_stage"], "hard_boundary")
        self.assertTrue(gate["blocking_stage_blocks_next"])
        self.assertEqual(tuple(gate["stage_status"]), EXPECTED_STAGE_ORDER)
        for stage_id in EXPECTED_STAGE_ORDER:
            stage = gate["stage_status"][stage_id]
            self.assertIn("pass", stage)
            self.assertIn("blocks_next", stage)
            self.assertIn("reason", stage)
            self.assertIn("allowed_patch_keys", stage)
            self.assertIn("blocked_patch_keys", stage)
            self.assertIsInstance(stage["pass"], bool)
            self.assertIsInstance(stage["blocks_next"], bool)
            self.assertIsInstance(stage["allowed_patch_keys"], list)
            self.assertIsInstance(stage["blocked_patch_keys"], list)
        hard_boundary = gate["stage_status"]["hard_boundary"]
        self.assertFalse(hard_boundary["pass"])
        self.assertTrue(hard_boundary["blocks_next"])
        self.assertEqual(hard_boundary["reason"], "roi_outside")
        self.assertEqual(hard_boundary["allowed_patch_keys"], [])
        self.assertIn("font_size_delta", hard_boundary["blocked_patch_keys"])

        context = prompt_stage_context({"pass": False, "issues": [{"type": "roi_outside"}]})
        self.assertEqual(context["stage_order"], list(EXPECTED_STAGE_ORDER))
        self.assertEqual(context["blocking_stage"], "hard_boundary")
        self.assertTrue(context["blocking_stage_blocks_next"])
        self.assertEqual(context["allowed_patch_keys"], [])
        self.assertIn("font_size_delta", context["blocked_patch_keys"])

    def test_stage_contract_entrypoints_are_only_defined_in_stages_module(self) -> None:
        src_root = Path(__file__).resolve().parents[1] / "src" / "roi_image_edit"
        definitions = {
            "StageSpec": [],
            "StageResult": [],
            "STAGE_SPECS": [],
            "stage_gate_for_report": [],
            "prompt_stage_context": [],
        }

        for path in src_root.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    if node.name in definitions:
                        definitions[node.name].append(path.name)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in definitions:
                            definitions[target.id].append(path.name)

        self.assertEqual(
            definitions,
            {
                "StageSpec": ["stages.py"],
                "StageResult": ["stages.py"],
                "STAGE_SPECS": ["stages.py"],
                "stage_gate_for_report": ["stages.py"],
                "prompt_stage_context": ["stages.py"],
            },
        )

    def test_profile_contracts_cover_current_profiles(self) -> None:
        self.assertEqual(
            stage_profile_choices(),
            ("clean_digital", "low_res_thumbnail", "manual_roi_quick", "photo_scan"),
        )
        photo_scan = stage_profile("photo_scan")
        self.assertEqual(photo_scan.stage_order, EXPECTED_STAGE_ORDER)
        self.assertEqual(photo_scan.enabled_stage_ids, frozenset(EXPECTED_STAGE_ORDER))
        clean_digital = stage_profile("clean_digital")
        self.assertFalse(clean_digital.enable_photo_texture)
        self.assertNotIn("photo_texture", clean_digital.enabled_stage_ids)
        manual_quick = stage_profile("manual_roi_quick")
        self.assertTrue(manual_quick.manual_roi)

    def test_text_shape_stage_rejects_photo_or_background_primary_patches(self) -> None:
        accepted, rejected = filter_patches_for_stage(
            [
                {"font_size_delta": 1},
                {"text_dx_delta": -1},
                {"blur_delta": 0.1},
                {"photo_noise_delta": 0.03},
                {"jpeg_quality_delta": -4},
                {"mask_threshold_delta": 3},
            ],
            "text_shape",
            limit=10,
        )
        self.assertEqual(accepted, [{"font_size_delta": 1}, {"text_dx_delta": -1}])
        self.assertEqual(len(rejected), 4)
        self.assertTrue(all(not item["allowed"] for item in rejected))

        blur_audit = patch_allowed_for_stage({"blur_delta": 0.1}, "text_shape")
        self.assertFalse(blur_audit["allowed"])
        self.assertEqual(blur_audit["optimization_steps"], ["photo_texture"])

        background_audit = patch_allowed_for_stage({"mask_threshold_delta": 3}, "text_shape")
        self.assertFalse(background_audit["allowed"])
        self.assertEqual(background_audit["optimization_steps"], ["background_cleanup"])


if __name__ == "__main__":
    unittest.main()
