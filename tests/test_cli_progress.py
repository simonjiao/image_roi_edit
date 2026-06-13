from __future__ import annotations

import unittest

from roi_image_edit.cli import format_process_progress_line


class CliProgressTest(unittest.TestCase):
    def test_revision_round_candidates_line_exposes_stage_progress_fields(self) -> None:
        line = format_process_progress_line(
            "revision_round_candidates",
            {
                "round": 3,
                "pipeline_profile": "photo_scan",
                "class_key": "photo_document.form_field_value_replace.cjk",
                "roi_policy": "manual_anchor",
                "internal_profile": "photo_scan",
                "profile_source": "classification",
                "basis_blocking_stage": "text_shape",
                "blocking_stage_reason": "font_style_mismatch",
                "allowed_patch_keys": ["font_size_delta", "text_dx_delta"],
                "blocked_patch_keys": ["mask_threshold_delta", "blur_delta"],
                "stage_optimization_policy": {"optimization_step": "text_shape"},
                "patch_count": 4,
                "shape_reset_count": 2,
                "ink_gray_count": 0,
                "ink_guard_count": 8,
                "photo_texture_count": 0,
                "basis_stage_severity": 8.5,
            },
        )
        self.assertIsNotNone(line)
        assert line is not None
        self.assertIn("round=3", line)
        self.assertIn("class_key=photo_document.form_field_value_replace.cjk", line)
        self.assertIn("roi_policy=manual_anchor", line)
        self.assertIn("internal_profile=photo_scan", line)
        self.assertIn("profile_source=classification", line)
        self.assertIn("blocking_stage=text_shape", line)
        self.assertIn("reason=font_style_mismatch", line)
        self.assertIn("allowed_params=font_size_delta,text_dx_delta", line)
        self.assertIn("blocked_params=mask_threshold_delta,blur_delta", line)
        self.assertIn("selected_optimization_step=text_shape", line)
        self.assertIn("patches=4", line)
        self.assertIn("shape_resets=2", line)
        self.assertIn("ink_gray=0", line)
        self.assertIn("ink_guard=8", line)
        self.assertIn("photo_texture=0", line)

    def test_revision_round_finished_line_uses_selected_attempt_step(self) -> None:
        line = format_process_progress_line(
            "revision_round_finished",
            {
                "round": 1,
                "pipeline_profile": "clean_digital",
                "class_key": "clean_digital.numeric_or_date_replace",
                "roi_policy": "auto",
                "internal_profile": "clean_digital",
                "profile_source": "classification",
                "blocking_stage": "ink_gray_balance",
                "blocking_stage_reason": "ink_too_light",
                "allowed_patch_keys": ["opacity_delta"],
                "blocked_patch_keys": ["font_size_delta"],
                "selected_optimization_step": "ink_gray_balance",
                "accepted": False,
                "final_decision": "revise",
                "score": 123.4,
            },
        )
        self.assertIsNotNone(line)
        assert line is not None
        self.assertIn("round=1", line)
        self.assertIn("class_key=clean_digital.numeric_or_date_replace", line)
        self.assertIn("roi_policy=auto", line)
        self.assertIn("internal_profile=clean_digital", line)
        self.assertIn("blocking_stage=ink_gray_balance", line)
        self.assertIn("reason=ink_too_light", line)
        self.assertIn("allowed_params=opacity_delta", line)
        self.assertIn("blocked_params=font_size_delta", line)
        self.assertIn("selected_optimization_step=ink_gray_balance", line)
        self.assertIn("accepted=False", line)
        self.assertIn("decision=revise", line)


if __name__ == "__main__":
    unittest.main()
