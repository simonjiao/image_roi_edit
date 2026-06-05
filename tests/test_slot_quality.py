from __future__ import annotations

import unittest

from PIL import Image, ImageDraw

from roi_image_edit.iterative_pipeline import TextRun
from roi_image_edit.slot_quality import slot_quality_report


def slot(x1: int, y1: int = 10, x2: int | None = None, y2: int = 24) -> TextRun:
    return TextRun(x1=x1, y1=y1, x2=x2 if x2 is not None else x1 + 10, y2=y2, area=100)


def image_with_slots(slots: tuple[TextRun, ...], *, extra_marks: tuple[tuple[int, int, int, int], ...] = ()) -> Image.Image:
    img = Image.new("RGB", (90, 42), (214, 214, 214))
    draw = ImageDraw.Draw(img)
    for run in slots:
        draw.rectangle((run.x1 + 2, run.y1 + 3, run.x2 - 3, run.y2 - 4), fill=(44, 44, 44))
        draw.rectangle((run.x1 + 1, run.y1 + 2, run.x2 - 2, run.y2 - 3), outline=(132, 132, 132))
    for mark in extra_marks:
        draw.rectangle(mark, fill=(52, 52, 52))
    return img


class SlotQualityTest(unittest.TestCase):
    def test_slot_quality_schema_records_counts_coverage_overlap_and_masks(self) -> None:
        slots = (slot(10), slot(26))
        report = slot_quality_report(
            image_with_slots(slots),
            (0, 0, 80, 36),
            slots,
            source_text="赵芳",
            target_text="陈慧",
            protected_boxes=((48, 9, 56, 25),),
        )
        self.assertTrue(report["pass"])
        self.assertEqual(report["source_count"], 2)
        self.assertEqual(report["target_count"], 2)
        self.assertEqual(report["length_change"], "same")
        self.assertEqual(report["overlap_report"]["protected_overlap_pixels"], 0)
        self.assertEqual(
            report["slot_coverage_schema"]["coverage_fields"],
            ["core_coverage", "gray_edge_coverage", "bottom_coverage", "tilt_coverage"],
        )
        self.assertTrue(report["old_text_coverage_report"]["pass"])
        self.assertEqual(
            report["old_text_coverage_report"]["components"],
            ["core", "gray_edge", "bottom_overflow", "tilt_overflow", "cleanup_mask"],
        )
        self.assertTrue(report["old_text_coverage_report"]["source_slot_or_cleanup_mask_required"])
        self.assertEqual(len(report["per_slot"]), 2)
        for item in report["per_slot"]:
            self.assertIn("core_pixels", item)
            self.assertIn("gray_edge_pixels", item)
            self.assertGreater(item["coverage"]["core_coverage"], 0.9)
            self.assertGreater(item["coverage"]["gray_edge_coverage"], 0.8)
            self.assertEqual(item["coverage"]["tilt_overflow_pixels"], 0)
            self.assertEqual(item["coverage"]["diagonal_dark_pixels"], 0)
            self.assertEqual(item["overlap"]["protected_overlap_pixels"], 0)
        for item in report["old_text_coverage_report"]["per_slot"]:
            self.assertTrue(item["pass"])
            self.assertGreater(item["core_coverage"], 0.9)
            self.assertGreater(item["gray_edge_coverage"], 0.8)
            self.assertEqual(item["tilt_overflow_pixels"], 0)
        cleanup = report["length_change_report"]["cleanup_mask_report"]
        self.assertFalse(cleanup["enabled"])
        self.assertEqual(cleanup["pixel_count"], 0)

    def test_bottom_overflow_fails_slot_quality_before_candidate_generation(self) -> None:
        slots = (slot(10, y2=20),)
        report = slot_quality_report(
            image_with_slots(slots, extra_marks=((12, 20, 17, 23),)),
            (0, 0, 60, 34),
            slots,
            source_text="男",
            target_text="女",
            protected_boxes=(),
        )
        issue_types = {issue["type"] for issue in report["issues"]}
        self.assertFalse(report["pass"])
        self.assertFalse(report["old_text_coverage_report"]["pass"])
        self.assertIn("slot_bottom_overflow", issue_types)
        self.assertLess(report["per_slot"][0]["coverage"]["bottom_coverage"], 1.0)
        self.assertGreater(report["per_slot"][0]["coverage"]["bottom_dark_pixels"], 0)

    def test_tilt_overflow_fails_slot_quality_before_candidate_generation(self) -> None:
        slots = (slot(10),)
        report = slot_quality_report(
            image_with_slots(slots, extra_marks=((20, 13, 21, 18),)),
            (0, 0, 60, 34),
            slots,
            source_text="男",
            target_text="女",
            protected_boxes=(),
        )
        issue_types = {issue["type"] for issue in report["issues"]}
        self.assertFalse(report["pass"])
        self.assertFalse(report["old_text_coverage_report"]["pass"])
        self.assertIn("slot_tilt_overflow", issue_types)
        self.assertLess(report["per_slot"][0]["coverage"]["tilt_coverage"], 1.0)
        self.assertGreater(report["per_slot"][0]["coverage"]["right_dark_pixels"], 0)
        self.assertGreater(report["old_text_coverage_report"]["per_slot"][0]["tilt_overflow_pixels"], 0)

    def test_shorter_replacement_reports_extra_source_cleanup_mask(self) -> None:
        slots = (slot(10), slot(26), slot(42))
        report = slot_quality_report(
            image_with_slots(slots),
            (0, 0, 80, 36),
            slots,
            source_text="赵真真",
            target_text="陈芸",
            protected_boxes=(),
        )
        length = report["length_change_report"]
        cleanup = length["cleanup_mask_report"]
        self.assertTrue(report["pass"])
        self.assertEqual(length["length_change"], "shorter")
        self.assertEqual(length["source_count"], 3)
        self.assertEqual(length["target_count"], 2)
        self.assertEqual(length["extra_source_slots_for_cleanup"], [[42, 10, 52, 24]])
        self.assertTrue(cleanup["enabled"])
        self.assertEqual(cleanup["boxes"], [[42, 10, 52, 24]])
        self.assertEqual(cleanup["span"], [42, 10, 52, 24])
        self.assertGreater(cleanup["pixel_count"], 0)

    def test_shorter_replacement_cleanup_slot_covers_right_bottom_overflow(self) -> None:
        slots = (slot(10), slot(26), slot(42))
        report = slot_quality_report(
            image_with_slots(slots, extra_marks=((52, 24, 53, 26),)),
            (0, 0, 80, 36),
            slots,
            source_text="赵真真",
            target_text="陈芸",
            protected_boxes=((50, 20, 55, 27),),
        )
        issue_types = {issue["type"] for issue in report["issues"]}
        cleanup = report["length_change_report"]["cleanup_mask_report"]
        old_text_coverage = report["old_text_coverage_report"]
        cleanup_slot_report = old_text_coverage["per_slot"][2]

        self.assertTrue(report["pass"])
        self.assertNotIn("slot_overlaps_protected_text", issue_types)
        self.assertNotIn("slot_tilt_overflow", issue_types)
        self.assertEqual(report["overlap_report"]["protected_overlap_pixels"], 0)
        self.assertGreater(report["overlap_report"]["old_source_cleanup_overlap_pixels"], 0)
        self.assertTrue(cleanup["enabled"])
        self.assertIn([42, 10, 52, 24], cleanup["boxes"])
        self.assertTrue(any(box[0] >= 52 and box[1] >= 24 for box in cleanup["boxes"]))
        self.assertTrue(old_text_coverage["pass"])
        self.assertTrue(cleanup_slot_report["cleanup_slot"])
        self.assertTrue(cleanup_slot_report["overflow_covered_by_cleanup_mask"])
        self.assertGreater(cleanup_slot_report["diagonal_dark_pixels"], 0)
        self.assertTrue(any(box[0] >= 52 and box[1] >= 24 for box in cleanup_slot_report["cleanup_mask_boxes"]))

    def test_slot_count_mismatch_fails_with_source_and_target_counts(self) -> None:
        slots = (slot(10),)
        report = slot_quality_report(
            image_with_slots(slots),
            (0, 0, 70, 36),
            slots,
            source_text="赵芳",
            target_text="陈慧",
            protected_boxes=(),
        )
        self.assertFalse(report["pass"])
        self.assertEqual(report["source_count"], 2)
        self.assertEqual(report["target_count"], 2)
        self.assertEqual(report["expected_count"], 2)
        self.assertEqual(report["actual_count"], 1)
        self.assertIn("slot_count_too_low", {issue["type"] for issue in report["issues"]})

    def test_longer_replacement_right_boundary_blocks_protected_text_collision(self) -> None:
        slots = (slot(10), slot(26))
        report = slot_quality_report(
            image_with_slots(slots),
            (0, 0, 72, 36),
            slots,
            source_text="赵芳",
            target_text="陈小慧",
            protected_boxes=((38, 9, 54, 25),),
        )
        length = report["length_change_report"]
        boundary = length["right_boundary"]
        issue_types = {issue["type"] for issue in report["issues"]}
        self.assertFalse(report["pass"])
        self.assertEqual(length["length_change"], "longer")
        self.assertTrue(boundary["enabled"])
        self.assertTrue(boundary["limited_by_protected_text"])
        self.assertFalse(boundary["pass"])
        self.assertEqual(boundary["roi_right_edge"], 72)
        self.assertEqual(boundary["minimum_safe_gap_px"], 3)
        self.assertEqual(boundary["protected_gap_px"], 2)
        self.assertEqual(boundary["protected_distances_px"], [2])
        self.assertEqual(boundary["protected_right_boxes"], [[38, 9, 54, 25]])
        self.assertLess(boundary["available_right_px"], boundary["estimated_extra_width"])
        boundary_issues = [issue for issue in report["issues"] if issue["type"] == "right_boundary_too_close_to_protected_text"]
        self.assertEqual(len(boundary_issues), 1)
        self.assertEqual(boundary_issues[0]["minimum_safe_gap_px"], 3)
        self.assertIn("right_boundary_too_close_to_protected_text", issue_types)

    def test_slot_overlapping_protected_text_is_a_gate_failure(self) -> None:
        slots = (slot(10),)
        report = slot_quality_report(
            image_with_slots(slots),
            (0, 0, 60, 36),
            slots,
            source_text="男",
            target_text="女",
            protected_boxes=((14, 10, 24, 24),),
        )
        self.assertFalse(report["pass"])
        self.assertGreater(report["overlap_report"]["protected_overlap_pixels"], 0)
        self.assertIn("slot_overlaps_protected_text", {issue["type"] for issue in report["issues"]})

    def test_slot_overlapping_left_label_is_reported_separately(self) -> None:
        slots = (slot(10), slot(26))
        report = slot_quality_report(
            image_with_slots(slots),
            (0, 0, 70, 36),
            slots,
            source_text="赵芳",
            target_text="陈慧",
            protected_boxes=((4, 10, 14, 24),),
        )
        self.assertFalse(report["pass"])
        self.assertGreater(report["overlap_report"]["protected_overlap_pixels"], 0)
        self.assertGreater(report["overlap_report"]["label_overlap_pixels"], 0)
        self.assertEqual(report["overlap_report"]["right_protected_overlap_pixels"], 0)
        self.assertGreater(report["per_slot"][0]["overlap"]["label_overlap_pixels"], 0)


if __name__ == "__main__":
    unittest.main()
