from __future__ import annotations

import unittest

from roi_image_edit.iterative_pipeline import CandidateParams, mutate_params
from roi_image_edit.stage_policy import optimization_policy_audit
from roi_image_edit.vision_targets import (
    non_regression_guard_report,
    vision_target_alignment,
    vision_target_alignment_complete,
    vision_target_from_acceptance,
    vision_target_recipe_patches,
    vision_target_recipe_report,
)


def params(**overrides: object) -> CandidateParams:
    base = CandidateParams(
        candidate_id="base",
        font_name="font",
        font_path="/tmp/font.ttf",
        font_size=22,
        opacity=0.82,
        blur=0.30,
        core_ink_gain=0.10,
        core_darken_strength=0.08,
        edge_breakup=0.010,
        photo_noise=0.020,
        mask_threshold=165,
        mask_dilate_iterations=2,
        inpaint_radius=3,
        jpeg_quality=90,
    )
    return mutate_params(base, **overrides)


class VisionTargetsTest(unittest.TestCase):
    def test_local_pass_vision_too_sharp_maps_to_photo_texture_target(self) -> None:
        target = vision_target_from_acceptance(
            {"pass": True, "stage_gate": {"blocking_stage": None}},
            {
                "pass": False,
                "acceptance_level": "marginal",
                "final_decision": "revise",
                "visual_findings": {"sharpness": "too_sharp"},
            },
            round_index=1,
            basis_candidate_id="c1",
        )

        self.assertTrue(target["active"])
        self.assertEqual(target["stage"], "photo_texture")
        self.assertEqual(target["stage_source"], "vision_acceptance")
        self.assertEqual(target["axis_keys"], ["too_sharp"])
        self.assertTrue(any(bound["parameter"] == "blur" for bound in target["bounds"]))

    def test_local_pass_vision_patch_visible_maps_to_background_target(self) -> None:
        target = vision_target_from_acceptance(
            {"pass": True, "stage_gate": {"blocking_stage": None}},
            {
                "pass": False,
                "acceptance_level": "marginal",
                "final_decision": "revise",
                "visual_findings": {"background": "patch_visible"},
            },
        )

        self.assertTrue(target["active"])
        self.assertEqual(target["stage"], "background_cleanup")
        self.assertEqual(target["axis_keys"], ["patch_visible"])

    def test_local_blocking_stage_prevents_vision_disagreement_target(self) -> None:
        target = vision_target_from_acceptance(
            {
                "pass": True,
                "strict_gate": {"issues": [{"type": "font_style_score_too_high"}]},
            },
            {
                "pass": False,
                "acceptance_level": "marginal",
                "final_decision": "revise",
                "visual_findings": {"sharpness": "too_sharp"},
            },
        )

        self.assertFalse(target["active"])
        self.assertEqual(target["axes"], [])

    def test_repeated_visual_axis_escalates_target(self) -> None:
        first = vision_target_from_acceptance(
            {"pass": True, "stage_gate": {"blocking_stage": None}},
            {
                "pass": False,
                "acceptance_level": "marginal",
                "final_decision": "revise",
                "visual_findings": {"sharpness": "too_sharp"},
            },
        )
        second = vision_target_from_acceptance(
            {"pass": True, "stage_gate": {"blocking_stage": None}},
            {
                "pass": False,
                "acceptance_level": "marginal",
                "final_decision": "revise",
                "visual_findings": {"sharpness": "too_sharp"},
            },
            prior_targets=[first],
        )

        self.assertTrue(second["repeated"])
        self.assertEqual(second["axes"][0]["repeat_count"], 2)
        self.assertTrue(second["axes"][0]["escalated"])

    def test_alignment_rewards_target_direction_and_penalizes_opposite(self) -> None:
        target = vision_target_from_acceptance(
            {"pass": True, "stage_gate": {"blocking_stage": None}},
            {
                "pass": False,
                "acceptance_level": "marginal",
                "final_decision": "revise",
                "visual_findings": {"sharpness": "too_sharp"},
            },
        )
        basis = params()
        aligned = vision_target_alignment(target, mutate_params(basis, blur=0.40, edge_breakup=0.016), basis)
        opposite = vision_target_alignment(target, mutate_params(basis, blur=0.24, edge_breakup=0.004), basis)

        self.assertEqual(aligned["direction"], "aligned")
        self.assertLess(aligned["score_adjustment"], 0)
        self.assertEqual(opposite["direction"], "opposite")
        self.assertGreater(opposite["score_adjustment"], 0)

    def test_too_dark_target_rejects_darker_candidate_direction(self) -> None:
        target = vision_target_from_acceptance(
            {"pass": True, "stage_gate": {"blocking_stage": None}},
            {
                "pass": False,
                "acceptance_level": "marginal",
                "final_decision": "revise",
                "visual_findings": {"darkness": "slightly_dark"},
            },
        )
        basis = params()
        darker = vision_target_alignment(
            target,
            mutate_params(basis, opacity=0.86, core_ink_gain=0.14, core_darken_strength=0.12),
            basis,
        )
        guard = non_regression_guard_report(target, darker)

        self.assertEqual(darker["direction"], "opposite")
        self.assertFalse(guard["pass"])
        self.assertIn("slightly_dark", guard["axes"])

    def test_combo_recipe_is_budgeted_and_declared_for_background_cleanup(self) -> None:
        target = vision_target_from_acceptance(
            {"pass": True, "stage_gate": {"blocking_stage": None}},
            {
                "pass": False,
                "acceptance_level": "marginal",
                "final_decision": "revise",
                "visual_findings": {"sharpness": "too_sharp", "background": "patch_visible"},
            },
        )
        patches = vision_target_recipe_patches(target)
        report = vision_target_recipe_report(target)

        self.assertEqual(target["combo_recipe"], "too_sharp_plus_patch_visible")
        self.assertEqual(target["stage"], "background_cleanup")
        self.assertLessEqual(len(patches), 3)
        self.assertFalse(report["cross_stage_cartesian_search"])
        self.assertTrue(
            all(audit["allowed"] for audit in report["patch_audits"]),
            report["patch_audits"],
        )
        for patch in patches:
            audit = optimization_policy_audit("background_cleanup", patch)
            self.assertTrue(audit["allowed"], audit)
            self.assertIn("background_cleanup", audit["effective_optimization_steps"])

    def test_combo_target_keeps_background_stage_when_ink_axis_is_also_present(self) -> None:
        target = vision_target_from_acceptance(
            {"pass": True, "stage_gate": {"blocking_stage": None}},
            {
                "pass": False,
                "acceptance_level": "marginal",
                "final_decision": "revise",
                "visual_findings": {
                    "background": "patch_visible",
                    "sharpness": "too_sharp",
                    "darkness": "slightly_dark",
                },
            },
        )

        self.assertEqual(target["stage"], "background_cleanup")
        self.assertEqual(target["axis_keys"], ["patch_visible", "too_sharp", "slightly_dark"])

    def test_background_combo_requires_background_and_photo_axes_to_be_complete(self) -> None:
        target = vision_target_from_acceptance(
            {"pass": True, "stage_gate": {"blocking_stage": None}},
            {
                "pass": False,
                "acceptance_level": "marginal",
                "final_decision": "revise",
                "visual_findings": {
                    "background": "patch_visible",
                    "sharpness": "too_sharp",
                    "darkness": "too_dark",
                },
            },
            prior_targets=[
                {
                    "active": True,
                    "axes": [
                        {"axis": "patch_visible"},
                        {"axis": "too_sharp"},
                        {"axis": "too_dark"},
                    ],
                }
            ],
        )
        basis = params()
        partial = vision_target_alignment(
            target,
            mutate_params(basis, photo_noise=0.032),
            basis,
        )
        no_background_change = vision_target_alignment(
            target,
            mutate_params(basis, blur=0.40),
            basis,
        )
        complete = vision_target_alignment(
            target,
            mutate_params(basis, mask_threshold=170, blur=0.40, photo_noise=0.032),
            basis,
        )

        partial_completion = vision_target_alignment_complete(target, partial)
        no_background_completion = vision_target_alignment_complete(target, no_background_change)
        complete_completion = vision_target_alignment_complete(target, complete)

        self.assertFalse(partial_completion["complete"])
        self.assertIn("too_sharp", partial_completion["missing_axes"])
        self.assertFalse(no_background_completion["complete"])
        self.assertIn("patch_visible", no_background_completion["missing_axes"])
        self.assertTrue(complete_completion["complete"])
        self.assertEqual(complete_completion["required_axes"], ["patch_visible", "too_sharp"])


if __name__ == "__main__":
    unittest.main()
