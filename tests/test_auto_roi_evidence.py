from __future__ import annotations

import unittest
from unittest.mock import patch

from PIL import Image

from roi_image_edit.roi_locator import (
    auto_orient_for_instruction,
    auto_region_evidence,
    old_value_location_quality_summary,
    target_field_quality_summary,
)


def auto_region(
    *,
    orientation_tag: str,
    score: float,
    slot_pass: bool = True,
) -> dict:
    return {
        "id": f"auto_name_{orientation_tag}",
        "rect": {"x": 10, "y": 8, "w": 26, "h": 14},
        "auto": True,
        "sourceText": "赵真真",
        "targetText": "陈芸",
        "_autoScore": score,
        "_autoFieldKey": "name",
        "_autoSearchRoi": [2, 4, 52, 26],
        "_autoEditRoi": [10, 8, 36, 22],
        "_autoTargetRoi": [10, 8, 46, 22],
        "_autoSlotBoxes": [[10, 8, 18, 22], [20, 8, 28, 22], [30, 8, 38, 22]],
        "_autoProtectedBoxes": [[0, 7, 8, 23]],
        "_autoSlotQualityReport": {
            "pass": slot_pass,
            "source_count": 3,
            "target_count": 2,
            "actual_count": 3,
            "length_change": "shorter",
            "issues": [] if slot_pass else [{"type": "slot_bottom_overflow"}],
        },
    }


class AutoRoiEvidenceTest(unittest.TestCase):
    def test_auto_region_evidence_records_search_edit_field_and_old_value_quality(self) -> None:
        evidence = auto_region_evidence(auto_region(orientation_tag="none", score=12.5))
        self.assertEqual(evidence["field_key"], "name")
        self.assertEqual(evidence["search_roi"], [2, 4, 52, 26])
        self.assertEqual(evidence["edit_roi"], [10, 8, 36, 22])
        self.assertEqual(evidence["target_roi"], [10, 8, 46, 22])
        self.assertEqual(evidence["source_text"], "赵真真")
        self.assertEqual(evidence["target_text"], "陈芸")
        self.assertTrue(evidence["slot_quality_pass"])
        self.assertEqual(evidence["source_count"], 3)
        self.assertEqual(evidence["target_count"], 2)
        self.assertEqual(evidence["actual_slot_count"], 3)

        field_quality = target_field_quality_summary([evidence])
        self.assertTrue(field_quality["available"])
        self.assertEqual(field_quality["field_keys"], ["name"])
        self.assertTrue(field_quality["has_search_roi"])
        self.assertTrue(field_quality["has_edit_roi"])

        old_value_quality = old_value_location_quality_summary([evidence])
        self.assertTrue(old_value_quality["available"])
        self.assertTrue(old_value_quality["all_slot_quality_pass"])
        self.assertEqual(old_value_quality["source_texts"], ["赵真真"])
        self.assertEqual(old_value_quality["slot_counts"], [3])

    def test_auto_orientation_records_field_and_old_value_quality_for_each_attempt(self) -> None:
        img = Image.new("RGB", (40, 24), (220, 220, 220))
        variants = (
            ("none", img),
            ("rotate_90_cw", img.transpose(Image.Transpose.ROTATE_270)),
        )
        qualities = [
            {"score": 1.0, "line_count": 1},
            {"score": 8.0, "line_count": 4},
        ]
        regions = [
            [auto_region(orientation_tag="none", score=9.0)],
            [auto_region(orientation_tag="rotated", score=12.0)],
        ]
        with patch("roi_image_edit.roi_locator.rotated_auto_orientation_variants", return_value=variants):
            with patch("roi_image_edit.roi_locator.document_orientation_quality", side_effect=qualities):
                with patch("roi_image_edit.roi_locator.auto_select_regions_for_instruction", side_effect=regions):
                    oriented, selected_regions, summary = auto_orient_for_instruction(
                        img,
                        instruction="姓名赵真真修改为陈芸",
                        source_text="赵真真",
                        target_text="陈芸",
                    )

        self.assertEqual(oriented.size, variants[1][1].size)
        self.assertEqual(selected_regions[0]["id"], "auto_name_rotated")
        self.assertTrue(summary["applied"])
        self.assertEqual(summary["orientation"], "rotate_90_cw")
        self.assertIn("target field quality", summary["final_direction_reason"])
        self.assertEqual(len(summary["attempts"]), 2)
        selected = summary["selected_attempt"]
        self.assertEqual(selected["orientation"], "rotate_90_cw")
        self.assertTrue(selected["selection_basis"]["uses_page_direction_quality"])
        self.assertTrue(selected["selection_basis"]["uses_target_field_quality"])
        self.assertTrue(selected["selection_basis"]["uses_old_value_location_quality"])
        self.assertEqual(selected["target_field_quality"]["field_keys"], ["name"])
        self.assertTrue(selected["old_value_location_quality"]["all_slot_quality_pass"])
        self.assertEqual(selected["regions"][0]["search_roi"], [2, 4, 52, 26])
        self.assertEqual(selected["regions"][0]["edit_roi"], [10, 8, 36, 22])
        self.assertEqual(selected["regions"][0]["evidence"]["source_count"], 3)


if __name__ == "__main__":
    unittest.main()
