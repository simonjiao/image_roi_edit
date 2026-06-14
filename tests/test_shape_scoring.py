from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

from roi_image_edit.iterative_pipeline import (
    CandidateParams,
    RenderPlan,
    TextRun,
    char_alignment_metrics,
    char_gray_band_metrics,
    char_pose_metrics,
    target_char_slots_for_plan,
)
from roi_image_edit.local_validation import (
    candidate_report,
    processing_candidate_score,
    row_baseline_metrics,
    strict_gate_stage_issues,
)
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


def alpha_layer(size: tuple[int, int], box: tuple[int, int, int, int]) -> Image.Image:
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    alpha = Image.new("L", size, 0)
    draw = ImageDraw.Draw(alpha)
    draw.rectangle((box[0], box[1], box[2] - 1, box[3] - 1), fill=255)
    layer.putalpha(alpha)
    return layer


def center_mode_plan() -> RenderPlan:
    return RenderPlan(
        target_text="丙丁戊",
        source_text="甲乙",
        search_roi=(0, 0, 96, 44),
        target_roi=(20, 8, 76, 36),
        slot_boxes=(TextRun(24, 10, 36, 28, 140), TextRun(38, 10, 50, 28, 140)),
        protected_boxes=((5, 10, 18, 28), (78, 10, 90, 28)),
        source_reference_box=(24, 10, 50, 28),
        style_reference_box=None,
        style_reference_text=None,
        draw_mode="center",
        placement_strategy="left_anchor_span",
        placement_strategy_reason="test",
        slot_quality_report={"pass": True},
    )


def longer_append_plan() -> RenderPlan:
    return RenderPlan(
        target_text="丙丁戊",
        source_text="甲乙",
        search_roi=(0, 0, 96, 44),
        target_roi=(10, 8, 72, 34),
        slot_boxes=(TextRun(12, 10, 28, 28, 140), TextRun(31, 10, 47, 28, 140)),
        protected_boxes=((76, 10, 90, 28),),
        source_reference_box=(12, 10, 47, 28),
        style_reference_box=None,
        style_reference_text=None,
        draw_mode="line_chars",
        placement_strategy="left_anchor_span",
        placement_strategy_reason="target_text_longer_than_source",
        slot_quality_report={
            "pass": True,
            "length_change_report": {
                "length_change": "longer",
                "right_boundary": {"pass": True},
            },
        },
    )


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

    def test_row_baseline_metrics_flags_centered_text_that_sits_too_low(self) -> None:
        image = Image.new("RGB", (96, 44), (200, 200, 200))
        with patch(
            "roi_image_edit.local_validation.draw_replacement_layer",
            return_value=alpha_layer(image.size, (24, 14, 70, 32)),
        ):
            metrics = row_baseline_metrics(image, center_mode_plan(), params())

        self.assertTrue(metrics["enabled"])
        self.assertEqual(metrics["basis"], "source_slots_primary_same_row_context")
        self.assertGreater(metrics["center_delta_y"], metrics["limits"]["center_delta_y"])
        self.assertIn(
            "row_baseline_center_y_too_low",
            {issue["type"] for issue in metrics["issues"]},
        )

    def test_row_baseline_metrics_accepts_centered_text_on_old_row(self) -> None:
        image = Image.new("RGB", (96, 44), (200, 200, 200))
        with patch(
            "roi_image_edit.local_validation.draw_replacement_layer",
            return_value=alpha_layer(image.size, (24, 10, 70, 28)),
        ):
            metrics = row_baseline_metrics(image, center_mode_plan(), params())

        self.assertTrue(metrics["enabled"])
        self.assertEqual(metrics["issues"], [])

    def test_row_baseline_issues_are_text_shape_stage_issues(self) -> None:
        stages = strict_gate_stage_issues(
            {"strict_gate": {"issues": [{"type": "row_baseline_center_y_too_low"}]}}
        )

        self.assertEqual(stages["text_shape"][0]["type"], "row_baseline_center_y_too_low")

    def test_longer_replacement_appends_target_slots_without_compressing_old_slots(self) -> None:
        slots = target_char_slots_for_plan(longer_append_plan())

        self.assertEqual(len(slots), 3)
        self.assertEqual((slots[0].x1, slots[0].y1, slots[0].x2, slots[0].y2), (12, 10, 28, 28))
        self.assertEqual((slots[1].x1, slots[1].y1, slots[1].x2, slots[1].y2), (31, 10, 47, 28))
        self.assertGreaterEqual(slots[2].x1, slots[1].x2 + 1)
        self.assertEqual(slots[2].x2 - slots[2].x1, slots[1].x2 - slots[1].x1)
        self.assertEqual((slots[2].y1, slots[2].y2), (10, 28))

    def test_longer_replacement_alignment_uses_appended_slots(self) -> None:
        plan_value = longer_append_plan()
        with patch(
            "roi_image_edit.iterative_pipeline.replacement_char_bboxes",
            return_value=[(13, 10, 27, 28), (32, 10, 46, 28), (51, 10, 65, 28)],
        ):
            metrics = char_alignment_metrics((96, 44), plan_value, params())

        self.assertTrue(metrics["enabled"])
        self.assertEqual(len(metrics["per_char"]), 3)
        self.assertEqual(metrics["per_char"][2]["slot_box"], [50, 10, 66, 28])
        self.assertAlmostEqual(metrics["center_distance_delta"], 0.0)

    def test_longer_replacement_gray_metrics_reference_old_ink_not_empty_append_area(self) -> None:
        plan_value = longer_append_plan()
        original = Image.new("RGB", (96, 44), (200, 200, 200))
        candidate = Image.new("RGB", (96, 44), (200, 200, 200))
        old_draw = ImageDraw.Draw(original)
        new_draw = ImageDraw.Draw(candidate)
        for box in ((12, 10, 28, 28), (31, 10, 47, 28)):
            old_draw.rectangle((box[0], box[1], box[2] - 1, box[3] - 1), fill=(80, 80, 80))
        for slot in target_char_slots_for_plan(plan_value):
            new_draw.rectangle((slot.x1, slot.y1, slot.x2 - 1, slot.y2 - 1), fill=(80, 80, 80))

        metrics = char_gray_band_metrics(original, candidate, plan_value)

        self.assertTrue(metrics["enabled"])
        self.assertEqual(len(metrics["per_char"]), 3)
        third = metrics["per_char"][2]
        self.assertEqual(third["slot_box"], [50, 10, 66, 28])
        self.assertEqual(third["reference_slot_box"], [31, 10, 47, 28])
        self.assertEqual(third["delta"]["lt165"], 0)

    def test_longer_replacement_pose_uses_previous_source_slot_for_extra_char(self) -> None:
        plan_value = longer_append_plan()
        image = Image.new("RGB", (96, 44), (200, 200, 200))

        def fake_shear(_gray, slot, *, threshold):
            if slot.x1 == 12:
                return 0.08
            if slot.x1 == 31:
                return 0.05
            return None

        with patch("roi_image_edit.iterative_pipeline.estimate_slot_edge_shear", side_effect=fake_shear):
            metrics = char_pose_metrics(image, plan_value, params())

        self.assertTrue(metrics["enabled"])
        self.assertEqual(len(metrics["per_char"]), 3)
        third = metrics["per_char"][2]
        self.assertIsNone(third["source_char"])
        self.assertEqual(third["target_char"], "戊")
        self.assertTrue(third["changed"])
        self.assertEqual(third["slot_box"], [50, 10, 66, 28])
        self.assertEqual(third["source_slot_index"], 1)
        self.assertEqual(third["source_slot_box"], [31, 10, 47, 28])
        self.assertEqual(third["reference_shear"], 0.0575)
        self.assertEqual(third["applied_shear"], 0.0483)


if __name__ == "__main__":
    unittest.main()
