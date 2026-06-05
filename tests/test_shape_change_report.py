from __future__ import annotations

import unittest
from unittest.mock import patch

from roi_image_edit.iterative_pipeline import CandidateParams, RenderPlan, TextRun
from roi_image_edit.local_validation import shape_change_report


def plan() -> RenderPlan:
    return RenderPlan(
        target_text="丙乙",
        source_text="甲乙",
        search_roi=(0, 0, 60, 44),
        target_roi=(8, 8, 42, 38),
        slot_boxes=(TextRun(10, 10, 20, 30, 200), TextRun(28, 10, 38, 30, 200)),
        protected_boxes=(),
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
        font_path="/tmp/test.ttf",
        font_size=20,
        opacity=0.8,
        blur=0.2,
    )


class ShapeChangeReportTest(unittest.TestCase):
    def test_changed_char_schema_records_bbox_projection_margin_and_threshold_sources(self) -> None:
        with patch(
            "roi_image_edit.local_validation.replacement_char_bboxes",
            return_value=((12, 16, 28, 36), (28, 10, 38, 30)),
        ):
            report = shape_change_report((64, 48), plan(), params())

        self.assertTrue(report["enabled"])
        self.assertTrue(report["shape_change_large"])
        self.assertEqual(len(report["per_char"]), 2)
        self.assertEqual(len(report["changed_chars"]), 1)
        changed = report["changed_chars"][0]
        self.assertEqual(changed["source_char"], "甲")
        self.assertEqual(changed["target_char"], "丙")
        for field in (
            "bbox_width_delta_ratio",
            "bbox_height_delta_ratio",
            "centroid_dx",
            "centroid_dy",
            "ink_area_ratio",
            "row_projection_distance",
            "col_projection_distance",
            "margin_distribution_delta",
        ):
            self.assertIn(field, changed)
        self.assertIn("source_image_metrics", changed)
        self.assertIn("target_image_metrics", changed)
        self.assertEqual(changed["source_image_metrics"]["box"], [10, 10, 20, 30])
        self.assertEqual(changed["target_image_metrics"]["box"], [12, 16, 28, 36])
        self.assertEqual(changed["thresholds"]["bbox_width_delta_ratio"]["threshold_source"], "default")
        self.assertEqual(changed["thresholds"]["row_projection_distance"]["threshold_source"], "default")
        issue_types = {issue["type"] for issue in report["issues"]}
        self.assertIn("bbox_width_delta_ratio_large", issue_types)
        self.assertIn("row_projection_distance_large", issue_types)
        self.assertIn("col_projection_distance_large", issue_types)
        self.assertIn("margin_distribution_delta_large", issue_types)
        for issue in report["issues"]:
            self.assertIn("threshold_source", issue)

    def test_unchanged_chars_keep_metrics_but_do_not_enter_changed_chars(self) -> None:
        with patch(
            "roi_image_edit.local_validation.replacement_char_bboxes",
            return_value=((10, 10, 20, 30), (28, 10, 38, 30)),
        ):
            report = shape_change_report((64, 48), plan(), params())
        self.assertEqual(len(report["per_char"]), 2)
        self.assertEqual(len(report["changed_chars"]), 1)
        unchanged = report["per_char"][1]
        self.assertEqual(unchanged["source_char"], "乙")
        self.assertEqual(unchanged["target_char"], "乙")
        self.assertIn("row_projection_distance", unchanged)
        self.assertEqual(unchanged["issues"], [])


if __name__ == "__main__":
    unittest.main()
