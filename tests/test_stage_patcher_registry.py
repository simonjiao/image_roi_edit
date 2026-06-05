from __future__ import annotations

import unittest

from roi_image_edit.iterative_pipeline import CandidateParams
from roi_image_edit.stage_patchers import (
    patch_key_audit_for_stage_patcher,
    revision_patches_for_round,
    stage_patcher_registry_report,
    stage_patcher_specs,
)


class StagePatcherRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.params = CandidateParams(
            candidate_id="base",
            font_name="test-font",
            font_path="/tmp/test-font.ttf",
            font_size=20,
            opacity=0.82,
            blur=0.2,
        )

    def test_stage_patchers_declare_primary_allowed_and_blocked_keys(self) -> None:
        specs = stage_patcher_specs()
        self.assertEqual(
            tuple(spec.stage_id for spec in specs),
            ("text_shape", "ink_gray_balance", "photo_texture", "background_cleanup"),
        )
        for spec in specs:
            self.assertEqual(spec.primary_stage, spec.stage_id)
            self.assertTrue(callable(spec.patcher))
            self.assertFalse(spec.allowed_patch_keys & spec.blocked_patch_keys)
            self.assertFalse(spec.secondary_patch_keys & spec.blocked_patch_keys)
            self.assertTrue(spec.declared_patch_keys)
            report = spec.as_report()
            self.assertEqual(report["stage_id"], spec.stage_id)
            self.assertEqual(report["primary_stage"], spec.primary_stage)
            self.assertIsInstance(report["allowed_patch_keys"], list)
            self.assertIsInstance(report["blocked_patch_keys"], list)

    def test_registry_report_is_stable_and_explicit(self) -> None:
        report = stage_patcher_registry_report()
        self.assertEqual(set(report), {"text_shape", "ink_gray_balance", "photo_texture", "background_cleanup"})
        self.assertEqual(report["text_shape"]["primary_stage"], "text_shape")
        self.assertIn("font_size_delta", report["text_shape"]["allowed_patch_keys"])
        self.assertIn("mask_threshold_delta", report["text_shape"]["blocked_patch_keys"])

    def test_stage_patcher_outputs_do_not_contain_undeclared_keys(self) -> None:
        acceptances = {
            "text_shape": {
                "visual_findings": {"stroke_weight": "too_thin"},
                "parameter_suggestions": [{"name": "font_size", "delta": 1}],
            },
            "ink_gray_balance": {
                "visual_findings": {"darkness": "too_light"},
                "parameter_suggestions": [{"name": "opacity", "delta": 0.03}],
            },
            "photo_texture": {
                "visual_findings": {"sharpness": "too_sharp"},
                "parameter_suggestions": [{"name": "blur", "delta": 0.05}],
            },
            "background_cleanup": {
                "visual_findings": {"background": "patch_visible"},
                "parameter_suggestions": [{"name": "mask_threshold", "delta": -4}],
            },
        }
        for spec in stage_patcher_specs():
            patches = spec.patcher(
                self.params,
                acceptances[spec.stage_id],
                {"pass": True, "pipeline_profile": "photo_scan"},
                rank_patch={"blur_delta": 0.1},
            )
            self.assertTrue(patches)
            for patch in patches:
                audit = patch_key_audit_for_stage_patcher(spec.stage_id, patch)
                self.assertTrue(audit["declared"], audit)

    def test_photo_texture_dispatch_rejects_ink_gray_suggestions(self) -> None:
        patches = revision_patches_for_round(
            self.params,
            {
                "visual_findings": {"darkness": "too_dark", "sharpness": "too_sharp"},
                "parameter_suggestions": [
                    {"name": "opacity", "delta": -0.05},
                    {"name": "blur", "delta": 0.05},
                ],
            },
            {
                "pass": True,
                "pipeline_profile": "photo_scan",
                "local_photo_texture_issues": [{"type": "photo_texture_too_sharp"}],
            },
            rank_patch={"opacity_delta": -0.10},
        )
        self.assertTrue(patches)
        for patch in patches:
            self.assertNotIn("opacity_delta", patch)
            audit = patch_key_audit_for_stage_patcher("photo_texture", patch)
            self.assertTrue(audit["declared"], audit)


if __name__ == "__main__":
    unittest.main()
