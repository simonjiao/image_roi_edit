from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

from PIL import Image

from roi_image_edit.iterative_pipeline import CandidateParams, RenderPlan, TextRun
from roi_image_edit.local_validation import candidate_report, processing_candidate_score
from roi_image_edit.shape_scoring import shape_score_breakdown


def plan() -> RenderPlan:
    return RenderPlan(
        target_text="丙乙",
        source_text="甲乙",
        search_roi=(0, 0, 80, 44),
        target_roi=(8, 8, 48, 38),
        slot_boxes=(TextRun(10, 10, 20, 30, 200), TextRun(28, 10, 38, 30, 200)),
        protected_boxes=((37, 8, 52, 32),),
        source_reference_box=(10, 10, 38, 30),
        style_reference_box=None,
        style_reference_text=None,
        draw_mode="auto",
        placement_strategy="center_primary",
        placement_strategy_reason="test",
        slot_quality_report={"pass": True},
    )


def params() -> CandidateParams:
    return CandidateParams(
        candidate_id="c1",
        font_name="test",
        font_path="/tmp/test-font.ttf",
        font_size=20,
        opacity=0.8,
        blur=0.2,
    )


def report() -> dict:
    return {
        "pass": True,
        "params": {"font_path": "/tmp/test-font.ttf"},
        "stage_gate": {"blocking_stage": None, "stages": []},
        "strict_visual_metrics": {"thresholds": {}, "bands": {}},
        "strict_gate": {"pass": True, "text_complexity_ratio": 1.35, "issues": []},
        "char_gray_band_metrics": {"enabled": False},
        "local_ink_balance_issues": [],
        "local_stroke_body_issues": [],
        "local_neighbor_style_issues": [],
        "shape_change_report": {
            "enabled": True,
            "per_char": [
                {
                    "source_char": "甲",
                    "target_char": "丙",
                    "slot_box": [10, 10, 20, 30],
                    "candidate_box": [11, 9, 23, 30],
                    "bbox_width_delta_ratio": 0.2,
                    "bbox_height_delta_ratio": 0.05,
                    "centroid_dx": 2.0,
                    "centroid_dy": -0.5,
                    "ink_area_ratio": 1.18,
                    "source_image_metrics": {"center_y": 20.0},
                    "target_image_metrics": {"center_y": 19.5},
                },
                {
                    "source_char": "乙",
                    "target_char": "乙",
                    "slot_box": [28, 10, 38, 30],
                    "candidate_box": [26, 12, 35, 29],
                    "bbox_width_delta_ratio": -0.1,
                    "bbox_height_delta_ratio": -0.15,
                    "centroid_dx": -1.5,
                    "centroid_dy": 0.5,
                    "ink_area_ratio": 0.9,
                    "source_image_metrics": {"center_y": 20.0},
                    "target_image_metrics": {"center_y": 20.5},
                },
            ],
            "issues": [],
        },
        "char_pose_metrics": {
            "enabled": True,
            "per_char": [
                {"reference_shear": 0.08, "applied_shear": 0.03},
                {"reference_shear": 0.05, "applied_shear": 0.01},
            ],
        },
        "font_style_gate": {
            "enabled": True,
            "pass": False,
            "issues": [{"type": "font_render_style_score_ratio"}],
            "font_best": {"score_ratio_to_best": 1.08},
            "candidate": {"score_ratio_to_best": 1.22},
        },
    }


class ShapeScoringTest(unittest.TestCase):
    def test_candidate_report_records_shape_score_breakdown(self) -> None:
        image = Image.new("RGB", (80, 44), (200, 200, 200))
        boxes = ((11, 9, 23, 30), (26, 12, 35, 29))
        with (
            patch("roi_image_edit.local_validation.replacement_char_bboxes", return_value=boxes),
            patch("roi_image_edit.iterative_pipeline.replacement_char_bboxes", return_value=boxes),
            patch(
                "roi_image_edit.local_validation.char_pose_metrics",
                return_value={
                    "enabled": True,
                    "per_char": [{"reference_shear": 0.08, "applied_shear": 0.03}],
                },
            ),
            patch("roi_image_edit.shape_scoring.font_missing_text_chars", return_value=[]),
        ):
            hard_report = candidate_report(image, image, plan(), params(), {"enabled": False})

        self.assertTrue(hard_report["shape_score_breakdown"]["enabled"])
        self.assertIn("center_error", hard_report["shape_score_breakdown"]["components"])

    def test_breakdown_records_all_shape_ranking_components(self) -> None:
        with patch("roi_image_edit.shape_scoring.font_missing_text_chars", return_value=["丙"]):
            breakdown = shape_score_breakdown(report(), plan())

        self.assertTrue(breakdown["enabled"])
        self.assertGreater(breakdown["score"], 0.0)
        components = breakdown["components"]
        self.assertEqual(
            set(components),
            {
                "height_width_spacing_baseline",
                "center_error",
                "boundary_error",
                "ink_area_complexity",
                "pose_inheritance",
                "protected_distance",
                "font_style",
            },
        )
        self.assertIn("avg_abs_width_delta_ratio", components["height_width_spacing_baseline"])
        self.assertIn("avg_abs_height_delta_ratio", components["height_width_spacing_baseline"])
        self.assertIn("avg_abs_gap_delta_px", components["height_width_spacing_baseline"])
        self.assertIn("candidate_baseline_center_y_range", components["height_width_spacing_baseline"])
        self.assertEqual(components["center_error"]["max_abs_centroid_dx"], 2.0)
        self.assertIn("max_abs_left_delta_px", components["boundary_error"])
        self.assertIn("max_abs_right_delta_px", components["boundary_error"])
        self.assertEqual(components["ink_area_complexity"]["text_complexity_ratio"], 1.35)
        self.assertEqual(components["pose_inheritance"]["max_abs_shear_error"], 0.05)
        self.assertEqual(components["protected_distance"]["min_horizontal_gap_px"], 2.0)
        self.assertEqual(components["font_style"]["font_style_score_ratio"], 1.22)
        self.assertEqual(components["font_style"]["font_family_score_ratio"], 1.08)
        self.assertFalse(components["font_style"]["renderable_text_check"]["pass"])
        self.assertEqual(components["font_style"]["renderable_text_check"]["missing_chars"], ["丙"])

    def test_processing_candidate_score_includes_shape_breakdown_score(self) -> None:
        base = report()
        with_shape = copy.deepcopy(base)
        with_shape["shape_score_breakdown"] = {"enabled": True, "score": 17.25}
        without_shape = copy.deepcopy(base)
        without_shape.pop("shape_score_breakdown", None)

        delta = processing_candidate_score(with_shape) - processing_candidate_score(without_shape)

        self.assertAlmostEqual(delta, 17.25)


if __name__ == "__main__":
    unittest.main()
