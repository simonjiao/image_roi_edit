from __future__ import annotations

import inspect
import unittest

from roi_image_edit.iterative_pipeline import CandidateParams
from roi_image_edit.stage_patchers import (
    dispatch_revision_patches,
    patch_key_audit_for_stage_patcher,
    revision_patches_for_round,
    select_stage_patcher,
    stage_patch_filter_report,
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
        dispatch = dispatch_revision_patches(
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
        self.assertEqual(dispatch["patcher_stage"], "photo_texture")
        self.assertEqual(dispatch["selection_reason"], "blocking_stage")
        self.assertIn("stage_filter_report", dispatch)
        patches = dispatch["patches"]
        self.assertTrue(patches)
        for patch in patches:
            self.assertNotIn("opacity_delta", patch)
            audit = patch_key_audit_for_stage_patcher("photo_texture", patch)
            self.assertTrue(audit["declared"], audit)

    def test_legacy_revision_entry_delegates_to_dispatcher(self) -> None:
        source = inspect.getsource(revision_patches_for_round)
        self.assertIn("dispatch_revision_patches", source)
        self.assertNotIn("final_acceptance_patches", source)
        self.assertNotIn("STAGE_PATCHER_SPECS", source)
        self.assertNotIn("stage_patch_filter_report", source)

        acceptance = {
            "visual_findings": {"background": "patch_visible"},
            "parameter_suggestions": [{"name": "mask_threshold", "delta": -4}],
        }
        report = {
            "pass": True,
            "pipeline_profile": "photo_scan",
            "local_background_texture_issues": [{"type": "background_patch_visible"}],
        }
        dispatch = dispatch_revision_patches(self.params, acceptance, report)
        legacy = revision_patches_for_round(self.params, acceptance, report)
        self.assertEqual(legacy, dispatch["patches"])

    def test_select_stage_patcher_reports_selection_reason(self) -> None:
        report = {
            "pass": True,
            "pipeline_profile": "photo_scan",
            "local_ink_balance_issues": [{"type": "ink_too_light"}],
        }
        selection = select_stage_patcher(report, None)
        self.assertEqual(selection["blocking_stage"], "ink_gray_balance")
        self.assertEqual(selection["patcher_stage"], "ink_gray_balance")
        self.assertEqual(selection["selection_reason"], "blocking_stage")
        self.assertEqual(selection["patcher"]["primary_stage"], "ink_gray_balance")

    def test_filter_report_declares_secondary_impacts_and_rejects_cross_stage_primary(self) -> None:
        report = stage_patch_filter_report(
            [
                {"font_size_delta": 1, "blur_delta": 0.04},
                {"mask_threshold_delta": -3},
            ],
            "text_shape",
            limit=10,
        )
        self.assertEqual(report["stage_id"], "text_shape")
        self.assertEqual(report["primary_stage"], "text_shape")
        self.assertEqual(report["accepted_count"], 1)
        self.assertEqual(report["rejected_count"], 1)

        accepted = report["decisions"][0]
        self.assertEqual(accepted["decision"], "accepted")
        self.assertEqual(accepted["primary_stage"], "text_shape")
        self.assertEqual(accepted["primary_optimization_steps"], ["text_shape"])
        self.assertEqual(accepted["secondary_optimization_steps"], ["photo_texture"])
        self.assertEqual(
            accepted["decision_basis"],
            "secondary effects are declared and current stage remains primary",
        )

        rejected = report["decisions"][1]
        self.assertEqual(rejected["decision"], "rejected")
        self.assertEqual(rejected["primary_stage"], "text_shape")
        self.assertEqual(rejected["optimization_steps"], ["background_cleanup"])
        self.assertIn("forbidden optimization steps", rejected["decision_basis"])


if __name__ == "__main__":
    unittest.main()
