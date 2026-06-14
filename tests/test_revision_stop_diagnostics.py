from __future__ import annotations

import unittest

from PIL import Image, ImageDraw

from roi_image_edit.region_processing import (
    build_candidate_rejection_table,
    revision_visual_delta_report,
)


class RevisionStopDiagnosticsTest(unittest.TestCase):

    def test_no_selectable_revision_candidate_writes_candidate_rejection_table(self) -> None:
        attempts = [
            {
                "params": {"candidate_id": "c001"},
                "origin": "shape_reset",
                "stage_id": "text_shape",
                "optimization_step": "shape_reset",
                "strict_pass": True,
                "stage_pass": False,
                "blocking_stage": "text_shape",
                "current_stage_severity_before": 15.0,
                "current_stage_severity_after": 14.0,
                "prior_stage_regression": {"pass": True},
                "progresses_past_text_shape": False,
                "progresses_past_current_stage": False,
                "improves_current_stage": False,
                "ink_guard": {"selectable": False},
                "current_blocking_stage": "text_shape",
            }
        ]
        table = build_candidate_rejection_table(attempts, "text_shape")
        self.assertEqual(len(table), 1)
        entry = table[0]
        self.assertEqual(entry["candidate_id"], "c001")
        self.assertEqual(entry["origin"], "shape_reset")
        self.assertEqual(entry["primary_stage"], "text_shape")
        self.assertEqual(entry["optimization_step"], "shape_reset")
        self.assertTrue(entry["strict_pass"])
        self.assertFalse(entry["stage_pass"])
        self.assertEqual(entry["blocking_stage"], "text_shape")
        self.assertFalse(entry["selectable"])
        self.assertEqual(entry["rejection_reason"], "stage_gate_failed")

    def test_rejection_table_records_mixed_origins(self) -> None:
        attempts = [
            {
                "params": {"candidate_id": "c001"},
                "origin": "ink_gray_grid",
                "stage_id": "ink_gray_balance",
                "optimization_step": "ink_gray_balance",
                "strict_pass": True,
                "stage_pass": True,
                "blocking_stage": "photo_texture",
                "current_stage_severity_before": 8.0,
                "current_stage_severity_after": 7.0,
                "prior_stage_regression": {"pass": True},
                "progresses_past_text_shape": False,
                "progresses_past_current_stage": False,
                "improves_current_stage": False,
                "ink_guard": {"selectable": False},
                "current_blocking_stage": "ink_gray_balance",
            },
            {
                "params": {"candidate_id": "c002"},
                "origin": "patch",
                "stage_id": "ink_gray_balance",
                "optimization_step": "core_black_search",
                "strict_pass": False,
                "stage_pass": False,
                "blocking_stage": "ink_gray_balance",
                "current_stage_severity_before": 8.0,
                "current_stage_severity_after": 12.0,
                "prior_stage_regression": {"pass": False},
                "progresses_past_text_shape": False,
                "progresses_past_current_stage": False,
                "improves_current_stage": False,
                "ink_guard": {"selectable": False},
                "current_blocking_stage": "ink_gray_balance",
            },
        ]
        table = build_candidate_rejection_table(attempts, "ink_gray_balance")
        self.assertEqual(len(table), 2)
        self.assertFalse(table[0]["selectable"])
        self.assertEqual(table[0]["rejection_reason"], "no_selectable_progress")
        self.assertFalse(table[1]["selectable"])
        self.assertEqual(table[1]["rejection_reason"], "strict_gate_failed")

    def test_rejection_table_missing_fields_default(self) -> None:
        table = build_candidate_rejection_table([{"params": {}}], None)
        self.assertEqual(len(table), 1)
        entry = table[0]
        self.assertEqual(entry["candidate_id"], "")
        self.assertEqual(entry["origin"], "unknown")
        self.assertIn(entry["rejection_reason"], ["strict_gate_failed", "selectable"])

    def test_revision_visual_delta_marks_near_identical_roi_as_stalled(self) -> None:
        before = Image.new("RGB", (80, 40), "white")
        after = before.copy()
        draw = ImageDraw.Draw(after)
        draw.point((20, 15), fill=(253, 253, 253))

        report = revision_visual_delta_report(before, after, [10, 10, 50, 30])

        self.assertTrue(report["stalled"])
        self.assertLess(report["mae"], 0.08)
        self.assertLess(report["changed_ratio_gt2"], 0.01)

    def test_revision_visual_delta_keeps_visible_roi_change_running(self) -> None:
        before = Image.new("RGB", (80, 40), "white")
        after = before.copy()
        draw = ImageDraw.Draw(after)
        draw.rectangle((20, 15, 35, 25), fill=(80, 80, 80))

        report = revision_visual_delta_report(before, after, [10, 10, 50, 30])

        self.assertFalse(report["stalled"])
        self.assertGreater(report["changed_ratio_gt2"], 0.05)


if __name__ == "__main__":
    unittest.main()
