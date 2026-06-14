from __future__ import annotations

import unittest

from roi_image_edit.iterative_pipeline import CandidateParams
from roi_image_edit.model_suggestions import (
    combined_model_suggestion_patch,
    filter_model_patch_records,
    model_stage_response_contract,
    model_suggestion_filter_report,
)
from roi_image_edit.prompt_assets import load_prompt
from roi_image_edit.run_artifacts import model_stage_context
from roi_image_edit.stage_patchers import model_patch_records


class ModelSuggestionsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.params = CandidateParams(
            candidate_id="base",
            font_name="test-font",
            font_path="/tmp/test-font.ttf",
            font_size=20,
            opacity=0.82,
            blur=0.2,
        )

    def test_prompt_assets_require_stage_context_allowed_and_blocked_keys(self) -> None:
        prompts = {
            "candidate_rank_prompt.txt": load_prompt("candidate_rank_prompt.txt"),
            "final_acceptance_prompt.txt": load_prompt("final_acceptance_prompt.txt"),
            "tuning_prompt.txt": load_prompt("tuning_prompt.txt"),
        }
        for name, prompt in prompts.items():
            with self.subTest(prompt=name):
                self.assertIn("stage_context", prompt)
                self.assertIn("profile_constraints", prompt)
                self.assertIn("blocking_stage", prompt)
                self.assertIn("stage_status", prompt)
                self.assertIn("pass_with_deferred", prompt)
                self.assertIn("deferred_issues", prompt)
                self.assertIn("deferred_to_stage", prompt)
                self.assertIn("blocked_patch_keys", prompt)
                self.assertIn("stage_assessment", prompt)
                self.assertIn("blocking_stage_exists", prompt)
                self.assertIn("suggestion_target_stage", prompt)
                self.assertIn("basis", prompt)

    def test_forbidden_model_suggestion_is_rejected_and_audited(self) -> None:
        report = {
            "pass": True,
            "pipeline_profile": "photo_scan",
            "font_style_gate": {"issues": [{"type": "font_style_mismatch"}]},
        }
        stage_context = model_stage_context(report, "photo_scan")
        self.assertEqual(stage_context["blocking_stage"], "text_shape")
        self.assertIn("font_size_delta", stage_context["allowed_patch_keys"])
        self.assertIn("mask_threshold_delta", stage_context["blocked_patch_keys"])

        model_json = {
            "blocking_stage": "text_shape",
            "direction": "fix_text_shape",
            "suggested_patch": {"mask_threshold_delta": -4},
            "parameter_suggestions": [
                {"name": "font_size", "delta": 1},
                {"name": "opacity", "delta": -0.05},
            ],
        }
        records = model_patch_records(self.params, model_json, source="final_acceptance")
        filtered = filter_model_patch_records(records, stage_context["blocking_stage"])
        report = model_suggestion_filter_report(filtered)

        self.assertEqual(report["stage_id"], "text_shape")
        self.assertEqual(report["record_count"], 3)
        self.assertEqual(report["accepted_count"], 1)
        self.assertEqual(report["rejected_count"], 2)
        self.assertEqual(filtered["allowed_patches"], [{"font_size_delta": 1}])

        attempts = report["attempt_records"]
        accepted_attempts = [attempt for attempt in attempts if attempt["accepted_for_candidate_generation"]]
        rejected_attempts = [attempt for attempt in attempts if not attempt["accepted_for_candidate_generation"]]
        self.assertEqual(len(accepted_attempts), 1)
        self.assertEqual(accepted_attempts[0]["patch"], {"font_size_delta": 1})
        self.assertEqual(len(rejected_attempts), 2)
        self.assertEqual(
            {tuple(sorted(attempt["patch"].items())) for attempt in rejected_attempts},
            {
                (("mask_threshold_delta", -4),),
                (("opacity_delta", -0.05),),
            },
        )
        for attempt in rejected_attempts:
            self.assertEqual(attempt["local_blocking_stage"], "text_shape")
            self.assertIn("optimization steps", attempt["rejection_reason"])

    def test_unconvertible_model_suggestion_is_recorded_with_reason(self) -> None:
        model_json = {
            "blocking_stage": "ink_gray_balance",
            "direction": "recover_core_black",
            "parameter_suggestions": [
                {"name": "unknown_knob", "delta": 0.1},
                {"name": "opacity"},
                "bad item",
            ],
        }
        records = model_patch_records(self.params, model_json, source="final_acceptance")
        self.assertEqual(len(records), 3)
        self.assertEqual({record["conversion_status"] for record in records}, {"unconvertible"})
        self.assertIn("unsupported parameter", records[0]["conversion_reason"])
        self.assertIn("missing delta", records[1]["conversion_reason"])
        self.assertIn("not an object", records[2]["conversion_reason"])

        filtered = filter_model_patch_records(records, "ink_gray_balance")
        report = model_suggestion_filter_report(filtered)
        self.assertEqual(report["accepted_count"], 0)
        self.assertEqual(report["rejected_count"], 3)
        self.assertEqual(len(report["attempt_records"]), 3)
        self.assertEqual(
            [attempt["rejection_reason"] for attempt in report["attempt_records"]],
            [record["conversion_reason"] for record in records],
        )

    def test_background_cleanup_parameter_suggestions_are_convertible_and_allowed(self) -> None:
        model_json = {
            "blocking_stage": "background_cleanup",
            "direction": "recover_background_texture",
            "parameter_suggestions": [
                {"name": "mask_threshold", "to": 170},
                {"name": "mask_dilate_iterations", "to": 3},
                {"name": "inpaint_radius", "delta": -1},
            ],
        }
        records = model_patch_records(self.params, model_json, source="final_acceptance")
        self.assertEqual({record["conversion_status"] for record in records}, {"converted"})
        self.assertEqual(
            [record["patch"] for record in records],
            [
                {"mask_threshold_delta": 5},
                {"mask_dilate_iterations_delta": 1},
                {"inpaint_radius_delta": -1},
            ],
        )

        filtered = filter_model_patch_records(records, "background_cleanup")
        self.assertEqual(
            filtered["allowed_patches"],
            [
                {"mask_threshold_delta": 5},
                {"mask_dilate_iterations_delta": 1},
                {"inpaint_radius_delta": -1},
            ],
        )
        self.assertEqual(filtered["rejected_records"], [])

    def test_same_response_allowed_suggestions_are_combined_as_candidate_patch(self) -> None:
        params = CandidateParams(
            candidate_id="base",
            font_name="test-font",
            font_path="/tmp/test-font.ttf",
            font_size=20,
            opacity=0.82,
            blur=0.42,
            photo_noise=0.07,
            mask_threshold=165,
            inpaint_radius=3,
        )
        source = "final_acceptance_basis_round_5"
        model_json = {
            "blocking_stage": "background_cleanup",
            "direction": "recover_background_texture",
            "parameter_suggestions": [
                {"name": "mask_threshold", "to": 170},
                {"name": "photo_noise", "to": 0.04},
                {"name": "blur", "to": 0.52},
            ],
        }
        records = model_patch_records(params, model_json, source=source)
        filtered = filter_model_patch_records(records, "background_cleanup")
        combo = combined_model_suggestion_patch(filtered, source=source)

        self.assertTrue(combo["enabled"], combo)
        self.assertEqual(
            combo["patch"],
            {
                "mask_threshold_delta": 5,
                "photo_noise_delta": -0.03,
                "blur_delta": 0.1,
            },
        )
        self.assertEqual(combo["record_count"], 3)
        self.assertTrue(combo["optimization_policy"]["allowed"])

    def test_model_stage_response_contract_records_current_stage_and_basis(self) -> None:
        response = {
            "blocking_stage": "text_shape",
            "stage_assessment": {
                "blocking_stage_exists": True,
                "current_blocking_stage": "text_shape",
                "suggestion_target_stage": "text_shape",
                "basis": "stage_context reports text_shape as the blocking stage.",
            },
        }
        contract = model_stage_response_contract(response, "text_shape")
        self.assertTrue(contract["stage_assessment_present"])
        self.assertTrue(contract["blocking_stage_exists_matches_local"])
        self.assertTrue(contract["current_blocking_stage_matches_local"])
        self.assertTrue(contract["suggestion_targets_current_stage"])
        self.assertTrue(contract["basis_present"])
        self.assertTrue(contract["schema_complete"])

    def test_model_stage_response_contract_flags_cross_stage_suggestion(self) -> None:
        response = {
            "blocking_stage": "photo_texture",
            "stage_assessment": {
                "blocking_stage_exists": True,
                "current_blocking_stage": "text_shape",
                "suggestion_target_stage": "photo_texture",
                "basis": "model thinks the text looks sharp.",
            },
        }
        contract = model_stage_response_contract(response, "text_shape")
        self.assertTrue(contract["blocking_stage_exists_matches_local"])
        self.assertTrue(contract["current_blocking_stage_matches_local"])
        self.assertFalse(contract["suggestion_targets_current_stage"])
        self.assertEqual(contract["suggestion_target_stage"], "photo_texture")


if __name__ == "__main__":
    unittest.main()
