from __future__ import annotations

import unittest
import inspect

from roi_image_edit.iterative_pipeline import CandidateParams, RenderPlan, TextRun, generate_candidates
import roi_image_edit.processing_service as processing_service
from roi_image_edit.revision_solver import (
    TEXT_SHAPE_GRID_ALLOWED_DELTA_KEYS,
    TEXT_SHAPE_GRID_BLOCKED_DELTA_KEYS,
    TEXT_SHAPE_PRUNE_REASON_CATEGORIES,
    text_shape_reset_candidate_grid,
    text_shape_reset_candidates,
)


def params() -> CandidateParams:
    return CandidateParams(
        candidate_id="base",
        font_name="BaseFont",
        font_path="/tmp/base.ttf",
        font_size=18,
        opacity=0.83,
        blur=0.27,
        stroke_opacity=0.01,
        ink_gain=0.02,
        alpha_contrast=0.03,
        core_ink_gain=0.04,
        core_darken_strength=0.05,
        text_dx=0,
        text_dy=0,
        char_offsets=((0, 0), (0, 0)),
        mask_threshold=177,
        mask_dilate_iterations=2,
        inpaint_radius=3,
        photo_warp=0.07,
        edge_breakup=0.011,
        photo_noise=0.021,
        jpeg_quality=91,
    )


def plan() -> RenderPlan:
    return RenderPlan(
        target_text="丙乙",
        source_text="甲乙",
        search_roi=(0, 0, 80, 44),
        target_roi=(8, 8, 50, 38),
        slot_boxes=(TextRun(10, 10, 22, 30, 120), TextRun(28, 10, 40, 30, 120)),
        protected_boxes=((54, 8, 72, 32),),
        source_reference_box=(10, 10, 40, 30),
        style_reference_box=None,
        style_reference_text=None,
        draw_mode="auto",
        placement_strategy="center_primary",
        placement_strategy_reason="test",
        slot_quality_report={"pass": True},
    )


def font_style_reference() -> dict:
    return {
        "ranked_fonts": [
            {"font_name": "Songti", "font_path": "/tmp/songti.ttf", "font_size": 18},
            {"font_name": "GBSN", "font_path": "/tmp/gbsn.ttf", "font_size": 18},
            {"font_name": "SimSun", "font_path": "/tmp/simsun.ttf", "font_size": 18},
            {"font_name": "FangSong", "font_path": "/tmp/fangsong.ttf", "font_size": 18},
        ],
    }


def text_shape_report() -> dict:
    return {
        "pass": True,
        "pipeline_profile": "photo_scan",
        "strict_gate": {
            "issues": [{"type": "font_style_score_too_high"}],
        },
        "local_stroke_body_issues": [{"type": "stroke_body_too_thin"}],
    }


class ShapeCandidateGridTest(unittest.TestCase):
    def test_initial_candidate_grid_includes_row_baseline_y_offsets(self) -> None:
        candidates = generate_candidates(
            params(),
            font_candidates=[
                ("Songti", "/tmp/songti.ttf"),
                ("GBSN", "/tmp/gbsn.ttf"),
            ],
            font_style_reference=font_style_reference(),
            font_pool_size=2,
            iteration=0,
            limit=80,
        )

        text_dy_values = {candidate.text_dy for candidate in candidates}
        self.assertIn(-2, text_dy_values)
        self.assertIn(-1, text_dy_values)
        self.assertIn(1, text_dy_values)

    def test_text_shape_grid_reports_budget_and_allowed_delta_keys(self) -> None:
        base = params()
        grid = text_shape_reset_candidate_grid(
            base,
            font_style_reference(),
            plan(),
            text_shape_report(),
            limit=48,
        )

        self.assertTrue(grid.report["enabled"])
        self.assertEqual(grid.report["stage_id"], "text_shape")
        self.assertEqual(grid.report["optimization_step"], "shape_reset")
        self.assertEqual(grid.report["candidate_count"], 48)
        self.assertEqual(len(grid.candidates), 48)
        self.assertTrue(grid.report["budget"]["within_budget"])
        self.assertGreaterEqual(grid.report["budget"]["raw_candidate_budget"], 300)
        self.assertLessEqual(grid.report["budget"]["raw_candidate_budget"], 1500)
        self.assertEqual(grid.report["budget"]["retained_count"], 48)
        self.assertGreater(grid.report["budget"]["pruned_count"], 0)
        self.assertEqual(set(grid.report["allowed_delta_keys"]), TEXT_SHAPE_GRID_ALLOWED_DELTA_KEYS)
        self.assertEqual(set(grid.report["blocked_delta_keys"]), TEXT_SHAPE_GRID_BLOCKED_DELTA_KEYS)
        self.assertEqual(grid.report["violations"], [])
        self.assertEqual(
            grid.report["axes"]["pose_shear_source"],
            "renderer_reference_slot_shear_from_source_slots_and_neighbors",
        )
        self.assertEqual(grid.report["axes"]["placement_strategy"], "center_primary")
        self.assertEqual(
            tuple(grid.report["prune_reason_contract"]["required_categories"]),
            TEXT_SHAPE_PRUNE_REASON_CATEGORIES,
        )
        self.assertEqual(
            grid.report["prune_reason_contract"]["category_sources"]["protected_distance"],
            ["protected_boxes", "target_roi", "right_boundary"],
        )

        for candidate, audit in zip(grid.candidates, grid.report["candidate_delta_audit"]):
            self.assertTrue(audit["allowed_delta_keys_only"], audit)
            self.assertFalse(audit["blocked_delta_keys"], audit)
            self.assertFalse(audit["undeclared_delta_keys"], audit)
            self.assertTrue(set(audit["reason_categories"]) <= set(TEXT_SHAPE_PRUNE_REASON_CATEGORIES))
            self.assertEqual(candidate.opacity, base.opacity)
            self.assertEqual(candidate.blur, base.blur)
            self.assertEqual(candidate.mask_threshold, base.mask_threshold)
            self.assertEqual(candidate.mask_dilate_iterations, base.mask_dilate_iterations)
            self.assertEqual(candidate.inpaint_radius, base.inpaint_radius)
            self.assertEqual(candidate.photo_warp, base.photo_warp)
            self.assertEqual(candidate.edge_breakup, base.edge_breakup)
            self.assertEqual(candidate.photo_noise, base.photo_noise)
            self.assertEqual(candidate.jpeg_quality, base.jpeg_quality)

    def test_legacy_shape_candidate_entrypoint_returns_grid_candidates(self) -> None:
        direct = text_shape_reset_candidate_grid(
            params(),
            font_style_reference(),
            plan(),
            text_shape_report(),
            limit=24,
        )
        legacy = text_shape_reset_candidates(
            params(),
            font_style_reference(),
            plan(),
            text_shape_report(),
            limit=24,
        )
        self.assertEqual(legacy, direct.candidates)

    def test_grid_disabled_when_text_shape_is_not_blocking(self) -> None:
        grid = text_shape_reset_candidate_grid(
            params(),
            font_style_reference(),
            plan(),
            {"pass": True, "pipeline_profile": "photo_scan"},
        )
        self.assertFalse(grid.report["enabled"])
        self.assertEqual(grid.report["reason"], "text_shape_not_blocking")
        self.assertEqual(grid.candidates, [])

    def test_processing_service_preserves_shape_grid_report_in_revision_rounds(self) -> None:
        source = inspect.getsource(processing_service.run_region_vision_checks)
        self.assertIn("text_shape_reset_candidate_grid", source)
        self.assertIn('"shape_candidate_grid": shape_candidate_grid.report', source)


if __name__ == "__main__":
    unittest.main()
