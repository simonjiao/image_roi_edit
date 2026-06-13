from __future__ import annotations

import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

import roi_image_edit.roi_locator as roi_locator
from roi_image_edit.image_classification import classify_image_workflow, classify_region_roi_policy
from roi_image_edit.iterative_pipeline import TextRun
from roi_image_edit.pre_candidate_gates import classify_pre_candidate_slot_failure, pre_candidate_gate_report
from roi_image_edit.roi_locator import parse_instruction_details


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

    def test_longer_replacement_expands_edit_roi_beyond_manual_search_roi(self) -> None:
        image = Image.new("RGB", (100, 60), (235, 235, 235))
        draw = ImageDraw.Draw(image)
        slots = (run(10, 20, 22, 38), run(26, 20, 38, 38))
        for slot in slots:
            draw.rectangle([slot.x1 + 2, slot.y1 + 3, slot.x2 - 2, slot.y2 - 3], fill=(35, 35, 35))

        with patch.object(roi_locator, "slots_for_region", return_value=slots):
            with patch.object(roi_locator, "protected_boxes_for_region", return_value=()):
                plan = roi_locator.build_region_plan(
                    image,
                    (0, 0, 40, 50),
                    source_text="甲乙",
                    target_text="丙丁戊",
                )

        length_report = plan.slot_quality_report["length_change_report"]
        self.assertEqual(length_report["length_change"], "longer")
        self.assertIsNotNone(length_report["expanded_edit_roi"])
        self.assertGreater(length_report["expanded_edit_roi"][2], 40)
        self.assertTrue(length_report["expansion_report"]["expanded_beyond_search_roi"])
        self.assertTrue(plan.slot_quality_report["pass"])

    def test_manual_large_roi_is_anchor_search_and_relocated_edit_roi(self) -> None:
        image = Image.new("RGB", (150, 70), (235, 235, 235))
        draw = ImageDraw.Draw(image)
        slots = (run(72, 24, 84, 42), run(90, 24, 102, 42))
        for slot in slots:
            draw.rectangle([slot.x1 + 2, slot.y1 + 3, slot.x2 - 2, slot.y2 - 3], fill=(35, 35, 35))
        protected_label = (10, 24, 58, 42)

        classification = classify_image_workflow(
            image,
            instruction_details=parse_instruction_details("姓名甲乙修改为丙丁"),
            regions=[{"id": "manual_big", "rect": {"x": 0, "y": 0, "w": 130, "h": 58}}],
        )
        with patch.object(roi_locator, "slots_for_region", return_value=slots):
            with patch.object(roi_locator, "protected_boxes_for_region", return_value=(protected_label,)):
                plan = roi_locator.build_region_plan(
                    image,
                    (0, 0, 130, 58),
                    source_text="甲乙",
                    target_text="丙丁",
                    field_key="name",
                    field_label_text="姓名",
                    protected_texts=("姓名",),
                )

        roi_policy = classify_region_roi_policy(
            image_classification=classification,
            search_roi=plan.search_roi,
            edit_roi=plan.target_roi,
            source_text="甲乙",
        )
        pre_gate = pre_candidate_gate_report(
            candidate_count=3,
            regions=[{"id": "manual_big", "roi": list(plan.search_roi)}],
            slot_quality_report=plan.slot_quality_report,
        )

        self.assertEqual(classification["roi_policy"], "manual_anchor")
        self.assertEqual(roi_policy, "manual_anchor")
        self.assertEqual(plan.search_roi, (0, 0, 130, 58))
        self.assertGreater(plan.target_roi[0], protected_label[2])
        self.assertLess(plan.target_roi[2] - plan.target_roi[0], plan.search_roi[2] - plan.search_roi[0])
        self.assertTrue(plan.slot_quality_report["pass"])
        self.assertTrue(pre_gate["pass"])
        self.assertIsNone(pre_gate["failed_gate"])


if __name__ == "__main__":
    unittest.main()
