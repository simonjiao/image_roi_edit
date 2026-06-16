from __future__ import annotations

import inspect
import unittest

from roi_image_edit.iterative_pipeline import CandidateParams
import roi_image_edit.processing_service as processing_service
from roi_image_edit.processing_service import prior_stage_regression_report
from roi_image_edit.revision_solver import (
    PHOTO_TEXTURE_GRID_ALLOWED_DELTA_KEYS,
    PHOTO_TEXTURE_GRID_BLOCKED_DELTA_KEYS,
    PHOTO_TEXTURE_PRUNE_REASON_CATEGORIES,
    photo_texture_candidate_grid,
)


def params() -> CandidateParams:
    return CandidateParams(
        candidate_id="ink-top-01",
        font_name="BaseFont",
        font_path="/tmp/base.ttf",
        font_size=18,
        opacity=0.84,
        blur=0.18,
        stroke_opacity=0.06,
        ink_gain=0.08,
        alpha_contrast=0.12,
        core_ink_gain=0.16,
        core_darken_strength=0.14,
        core_darken_threshold=132,
        core_darken_target_gray=28,
        text_dx=1,
        text_dy=-1,
        char_offsets=((0, 0), (1, 0)),
        mask_threshold=177,
        mask_dilate_iterations=2,
        inpaint_radius=3,
        photo_warp=0.02,
        edge_breakup=0.004,
        photo_noise=0.010,
        jpeg_quality=94,
    )


def photo_texture_report() -> dict:
    return {
        "pass": True,
        "pipeline_profile": "photo_scan",
        "local_photo_texture_issues": [
            {"type": "photo_texture_too_clean", "residual_ratio": 0.2},
            {"type": "photo_texture_edge_breakup_missing", "edge_breakup": 0.0},
        ],
    }


class PhotoTextureCandidateGridTest(unittest.TestCase):
    def test_photo_texture_grid_reports_budget_and_allowed_delta_keys(self) -> None:
        base = params()
        grid = photo_texture_candidate_grid(base, photo_texture_report(), limit=6)

        self.assertTrue(grid.report["enabled"])
        self.assertEqual(grid.report["stage_id"], "photo_texture")
        self.assertEqual(grid.report["optimization_step"], "photo_texture")
        self.assertEqual(grid.report["parent_candidate_id"], "ink-top-01")
        self.assertEqual(grid.report["candidate_count"], 6)
        self.assertEqual(len(grid.candidates), 6)
        self.assertTrue(grid.report["budget"]["within_budget"])
        self.assertGreaterEqual(grid.report["budget"]["raw_candidate_budget"], 30)
        self.assertLessEqual(grid.report["budget"]["raw_candidate_budget"], 200)
        self.assertGreaterEqual(grid.report["budget"]["retained_count"], 3)
        self.assertLessEqual(grid.report["budget"]["retained_count"], 8)
        self.assertGreater(grid.report["budget"]["pruned_count"], 0)
        self.assertEqual(set(grid.report["allowed_delta_keys"]), PHOTO_TEXTURE_GRID_ALLOWED_DELTA_KEYS)
        self.assertEqual(set(grid.report["blocked_delta_keys"]), PHOTO_TEXTURE_GRID_BLOCKED_DELTA_KEYS)
        self.assertEqual(grid.report["axes"]["alpha_adjustment_scope"], "forbidden_in_photo_texture; use_ink_gray_balance")
        self.assertEqual(
            grid.report["axes"]["residual_retexture_keys"],
            ["edge_breakup", "photo_noise", "jpeg_quality"],
        )
        self.assertEqual(
            tuple(grid.report["prune_reason_contract"]["required_categories"]),
            PHOTO_TEXTURE_PRUNE_REASON_CATEGORIES,
        )
        self.assertEqual(
            grid.report["prune_reason_contract"]["category_sources"]["white_or_shadow_ghost"],
            ["background_white_ghost_residual", "background_shadow_ghost_residual"],
        )
        self.assertEqual(grid.report["violations"], [])

        for candidate, audit in zip(grid.candidates, grid.report["candidate_delta_audit"]):
            self.assertTrue(audit["allowed_delta_keys_only"], audit)
            self.assertFalse(audit["blocked_delta_keys"], audit)
            self.assertFalse(audit["undeclared_delta_keys"], audit)
            self.assertTrue(set(audit["reason_categories"]) <= set(PHOTO_TEXTURE_PRUNE_REASON_CATEGORIES))
            self.assertEqual(audit["parent_candidate_id"], base.candidate_id)
            self.assertLessEqual(abs(candidate.blur - base.blur), 0.08)
            self.assertEqual(candidate.alpha_contrast, base.alpha_contrast)
            self.assertLessEqual(abs(candidate.photo_warp - base.photo_warp), 0.02)
            self.assertLessEqual(abs(candidate.edge_breakup - base.edge_breakup), 0.018)
            self.assertLessEqual(abs(candidate.photo_noise - base.photo_noise), 0.030)
            self.assertLessEqual(abs(candidate.jpeg_quality - base.jpeg_quality), 6)
            self.assertEqual(candidate.font_name, base.font_name)
            self.assertEqual(candidate.font_path, base.font_path)
            self.assertEqual(candidate.font_size, base.font_size)
            self.assertEqual(candidate.opacity, base.opacity)
            self.assertEqual(candidate.stroke_opacity, base.stroke_opacity)
            self.assertEqual(candidate.ink_gain, base.ink_gain)
            self.assertEqual(candidate.core_ink_gain, base.core_ink_gain)
            self.assertEqual(candidate.core_darken_strength, base.core_darken_strength)
            self.assertEqual(candidate.text_dx, base.text_dx)
            self.assertEqual(candidate.text_dy, base.text_dy)
            self.assertEqual(candidate.char_offsets, base.char_offsets)
            self.assertEqual(candidate.mask_threshold, base.mask_threshold)
            self.assertEqual(candidate.mask_dilate_iterations, base.mask_dilate_iterations)
            self.assertEqual(candidate.inpaint_radius, base.inpaint_radius)

    def test_grid_disabled_when_photo_texture_is_not_blocking(self) -> None:
        grid = photo_texture_candidate_grid(
            params(),
            {"pass": True, "pipeline_profile": "photo_scan"},
        )
        self.assertFalse(grid.report["enabled"])
        self.assertEqual(grid.report["reason"], "photo_texture_not_blocking")
        self.assertEqual(grid.candidates, [])

    def test_processing_service_preserves_photo_texture_grid_report_in_revision_rounds(self) -> None:
        source = inspect.getsource(processing_service.run_region_vision_checks)
        self.assertIn("photo_texture_candidate_grid", source)
        self.assertIn('"photo_texture_candidate_grid": photo_candidate_grid.report', source)
        self.assertIn('"photo_texture_count": len(photo_texture_params)', source)
        self.assertIn('"prior_stage_regression": prior_regression', source)

    def test_photo_texture_candidate_records_prior_shape_or_ink_regression(self) -> None:
        before = {
            "pass": True,
            "pipeline_profile": "photo_scan",
            "local_photo_texture_issues": [{"type": "photo_texture_too_clean"}],
        }
        after = {
            "pass": True,
            "pipeline_profile": "photo_scan",
            "font_style_gate": {"issues": [{"type": "font_style_mismatch"}]},
            "local_photo_texture_issues": [{"type": "photo_texture_too_clean"}],
        }

        audit = prior_stage_regression_report(before, after, "photo_texture")

        self.assertFalse(audit["pass"])
        self.assertEqual(audit["prior_stage_ids"], ["hard_boundary", "text_shape", "ink_gray_balance"])
        self.assertEqual(audit["regressions"][0]["stage_id"], "text_shape")
        self.assertIn("font_style_mismatch", audit["regressions"][0]["after_issue_types"])
        self.assertFalse(audit["stage_severity"]["text_shape"]["after_pass"])


if __name__ == "__main__":
    unittest.main()
