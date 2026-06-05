from __future__ import annotations

import unittest

from roi_image_edit.iterative_pipeline import CandidateParams
from roi_image_edit.model_suggestions import (
    filter_model_patch_records,
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
                self.assertIn("blocking_stage", prompt)
                self.assertIn("blocked_patch_keys", prompt)

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


if __name__ == "__main__":
    unittest.main()
