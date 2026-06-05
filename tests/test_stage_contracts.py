from __future__ import annotations

from dataclasses import fields
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
        hard_boundary = gate["stage_status"]["hard_boundary"]
        self.assertFalse(hard_boundary["pass"])
        self.assertEqual(hard_boundary["reason"], "roi_outside")
        self.assertEqual(hard_boundary["allowed_patch_keys"], [])
        self.assertIn("font_size_delta", hard_boundary["blocked_patch_keys"])

        context = prompt_stage_context({"pass": False, "issues": [{"type": "roi_outside"}]})
        self.assertEqual(context["stage_order"], list(EXPECTED_STAGE_ORDER))
        self.assertEqual(context["blocking_stage"], "hard_boundary")
        self.assertEqual(context["allowed_patch_keys"], [])
        self.assertIn("font_size_delta", context["blocked_patch_keys"])

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
