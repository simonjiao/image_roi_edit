from __future__ import annotations

import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

import roi_image_edit.roi_locator as roi_locator
from roi_image_edit.iterative_pipeline import TextRun
from roi_image_edit.pre_candidate_gates import classify_pre_candidate_slot_failure


def run(x1: int, y1: int = 20, x2: int | None = None, y2: int = 38) -> TextRun:
    return TextRun(x1=x1, y1=y1, x2=x2 if x2 is not None else x1 + 12, y2=y2, area=120)


class RoiLocatorSlotTest(unittest.TestCase):
    def test_labeled_cjk_value_strategy_precedes_unlabeled_fallback(self) -> None:
        image = Image.new("RGB", (120, 64), (235, 235, 235))
        wrong_slots = (run(10), run(28))
        value_slots = (run(54), run(72))

        with patch.object(roi_locator, "non_cjk_value_slot_after_label", return_value=()):
            with patch.object(roi_locator, "cjk_value_slots_after_colon", return_value=()):
                with patch.object(roi_locator, "source_slots_after_label_components", return_value=value_slots):
                    with patch.object(roi_locator, "cjk_value_slots_without_label", return_value=wrong_slots):
                        slots = roi_locator.component_slots_for_region(
                            image,
                            (0, 0, 110, 52),
                            source_text="旧值",
                            target_text="新新新",
                        )

        self.assertEqual(slots, value_slots)

    def test_target_roi_protected_guard_is_direction_independent(self) -> None:
        image = Image.new("RGB", (80, 70), (235, 235, 235))
        draw = ImageDraw.Draw(image)
        draw.rectangle([23, 23, 32, 35], fill=(30, 30, 30))
        slot = run(20, 20, 35, 38)
        protected_by_side = {
            "left": (16, 24, 19, 32),
            "right": (36, 24, 40, 32),
            "top": (24, 12, 31, 15),
            "bottom": (24, 43, 31, 47),
        }

        for side, protected_box in protected_by_side.items():
            with self.subTest(side=side):
                with patch.object(roi_locator, "slots_for_region", return_value=(slot,)):
                    with patch.object(roi_locator, "protected_boxes_for_region", return_value=(protected_box,)):
                        plan = roi_locator.build_region_plan(
                            image,
                            (0, 0, 80, 70),
                            source_text="男",
                            target_text="女",
                        )

                report = plan.slot_quality_report
                issue_types = {issue["type"] for issue in report["issues"]}
                self.assertFalse(report["pass"])
                self.assertIn("target_roi_overlaps_protected_text", issue_types)
                self.assertEqual(report["overlap_report"]["protected_text_guard_scope"], "all_directions")
                self.assertEqual(classify_pre_candidate_slot_failure(report), "protected_text_guard")


if __name__ == "__main__":
    unittest.main()
