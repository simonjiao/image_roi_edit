from __future__ import annotations

import inspect
import unittest
from dataclasses import replace

from roi_image_edit.iterative_pipeline import CandidateParams
import roi_image_edit.processing_service as processing_service
from roi_image_edit.revision_solver import (
    INK_GRAY_GRID_ALLOWED_DELTA_KEYS,
    INK_GRAY_GRID_BLOCKED_DELTA_KEYS,
    ink_gray_candidate_grid,
    layered_candidate_search_report,
)


def params() -> CandidateParams:
    return CandidateParams(
        candidate_id="shape-top-01",
        font_name="BaseFont",
        font_path="/tmp/base.ttf",
        font_size=18,
        opacity=0.86,
        blur=0.24,
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
        photo_warp=0.05,
        edge_breakup=0.009,
        photo_noise=0.018,
        jpeg_quality=92,
    )


def ink_gray_report() -> dict:
    return {
        "pass": True,
        "pipeline_profile": "photo_scan",
        "local_ink_balance_issues": [
            {
                "type": "changed_char_core_too_black",
                "lt55_delta": 120.0,
                "limit": 40.0,
            }
        ],
    }


class InkGrayCandidateGridTest(unittest.TestCase):
    def test_ink_gray_grid_reports_budget_parent_and_allowed_delta_keys(self) -> None:
        base = params()
        grid = ink_gray_candidate_grid(base, ink_gray_report(), limit=16)

        self.assertTrue(grid.report["enabled"])
        self.assertEqual(grid.report["stage_id"], "ink_gray_balance")
        self.assertEqual(grid.report["optimization_step"], "ink_gray_balance")
        self.assertEqual(grid.report["parent_shape_candidate_id"], "shape-top-01")
        self.assertEqual(
            grid.report["parent_shape_contract"]["required_parent_state"],
            "text_shape_passed_before_ink_gray",
        )
        self.assertEqual(
            grid.report["parent_shape_contract"]["parent_shape_source"],
            "current_candidate_after_text_shape_pass",
        )
        self.assertTrue(grid.report["parent_shape_contract"]["parent_shape_stage_passed"])
        self.assertTrue(grid.report["parent_shape_contract"]["candidate_parent_trace_complete"])
        self.assertEqual(
            grid.report["parent_shape_contract"]["candidate_parent_shape_ids"],
            ["shape-top-01"],
        )
        self.assertEqual(grid.report["candidate_count"], 16)
        self.assertEqual(len(grid.candidates), 16)
        self.assertTrue(grid.report["budget"]["within_budget"])
        self.assertGreaterEqual(grid.report["budget"]["raw_candidate_budget"], 100)
        self.assertLessEqual(grid.report["budget"]["raw_candidate_budget"], 800)
        self.assertGreaterEqual(grid.report["budget"]["retained_count"], 8)
        self.assertLessEqual(grid.report["budget"]["retained_count"], 20)
        self.assertGreater(grid.report["budget"]["pruned_count"], 0)
        self.assertEqual(set(grid.report["allowed_delta_keys"]), INK_GRAY_GRID_ALLOWED_DELTA_KEYS)
        self.assertEqual(set(grid.report["blocked_delta_keys"]), INK_GRAY_GRID_BLOCKED_DELTA_KEYS)
        self.assertEqual(
            set(grid.report["preserved_shape_keys"]),
            {"font_name", "font_path", "font_size", "text_dx", "text_dy", "char_offsets"},
        )
        self.assertEqual(grid.report["shape_key_changes_require_stage"], "text_shape")
        self.assertEqual(grid.report["violations"], [])

        for candidate, audit in zip(grid.candidates, grid.report["candidate_delta_audit"]):
            self.assertTrue(audit["allowed_delta_keys_only"], audit)
            self.assertFalse(audit["blocked_delta_keys"], audit)
            self.assertFalse(audit["undeclared_delta_keys"], audit)
            self.assertEqual(audit["parent_candidate_id"], base.candidate_id)
            self.assertEqual(audit["parent_shape_candidate_id"], base.candidate_id)
            self.assertEqual(candidate.font_name, base.font_name)
            self.assertEqual(candidate.font_path, base.font_path)
            self.assertEqual(candidate.font_size, base.font_size)
            self.assertEqual(candidate.blur, base.blur)
            self.assertEqual(candidate.text_dx, base.text_dx)
            self.assertEqual(candidate.text_dy, base.text_dy)
            self.assertEqual(candidate.char_offsets, base.char_offsets)
            self.assertEqual(candidate.mask_threshold, base.mask_threshold)
            self.assertEqual(candidate.mask_dilate_iterations, base.mask_dilate_iterations)
            self.assertEqual(candidate.inpaint_radius, base.inpaint_radius)
            self.assertEqual(candidate.photo_warp, base.photo_warp)
            self.assertEqual(candidate.edge_breakup, base.edge_breakup)
            self.assertEqual(candidate.photo_noise, base.photo_noise)
            self.assertEqual(candidate.jpeg_quality, base.jpeg_quality)

    def test_grid_disabled_when_ink_gray_is_not_blocking(self) -> None:
        grid = ink_gray_candidate_grid(
            params(),
            {"pass": True, "pipeline_profile": "photo_scan"},
        )
        self.assertFalse(grid.report["enabled"])
        self.assertEqual(grid.report["reason"], "ink_gray_balance_not_blocking")
        self.assertEqual(grid.candidates, [])

    def test_parent_shape_candidate_survives_multiple_ink_gray_rounds(self) -> None:
        ink_round_parent = replace(params(), candidate_id="ink-round-01")
        grid = ink_gray_candidate_grid(
            ink_round_parent,
            ink_gray_report(),
            limit=8,
            parent_shape_candidate_id="shape-top-01",
        )

        self.assertEqual(grid.report["parent_candidate_id"], "ink-round-01")
        self.assertEqual(grid.report["parent_shape_candidate_id"], "shape-top-01")
        self.assertEqual(
            grid.report["parent_shape_contract"]["parent_candidate_id"],
            "ink-round-01",
        )
        self.assertEqual(
            grid.report["parent_shape_contract"]["parent_shape_candidate_id"],
            "shape-top-01",
        )
        for audit in grid.report["candidate_delta_audit"]:
            self.assertEqual(audit["parent_candidate_id"], "ink-round-01")
            self.assertEqual(audit["parent_shape_candidate_id"], "shape-top-01")

    def test_processing_service_preserves_ink_gray_grid_report_in_revision_rounds(self) -> None:
        source = inspect.getsource(processing_service.run_region_vision_checks)
        self.assertIn("ink_gray_candidate_grid", source)
        self.assertIn('"ink_gray_candidate_grid": ink_candidate_grid.report', source)
        self.assertIn('"ink_gray_count": len(ink_gray_params)', source)
        self.assertIn("current_shape_parent_candidate_id = current_params.candidate_id", source)
        self.assertIn("parent_shape_candidate_id=current_shape_parent_candidate_id", source)

    def test_layered_search_report_preserves_ink_parent_shape_trace(self) -> None:
        grid = ink_gray_candidate_grid(params(), ink_gray_report(), limit=16)
        report = layered_candidate_search_report(grid.report)

        self.assertEqual(
            report["parent_shape_trace"]["ink_gray_balance"]["parent_shape_candidate_id"],
            "shape-top-01",
        )
        self.assertTrue(
            report["parent_shape_trace"]["ink_gray_balance"]["parent_shape_stage_passed"]
        )
        self.assertTrue(
            report["parent_shape_trace"]["ink_gray_balance"]["candidate_parent_trace_complete"]
        )
        self.assertEqual(
            report["stages"][0]["parent_shape_candidate_id"],
            "shape-top-01",
        )


if __name__ == "__main__":
    unittest.main()
