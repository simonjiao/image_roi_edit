from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from roi_image_edit.auto_roi_artifacts import (
    auto_roi_evidence_payload,
    draw_auto_roi_overlay,
    save_auto_roi_overlay,
)


def region() -> dict:
    return {
        "id": "auto_name",
        "auto": True,
        "rect": {"x": 10, "y": 8, "w": 22, "h": 14},
        "sourceText": "赵芳",
        "targetText": "陈慧",
        "_autoScore": 8.2,
        "_autoFieldKey": "name",
        "_autoSearchRoi": [2, 4, 46, 26],
        "_autoEditRoi": [10, 8, 32, 22],
        "_autoTargetRoi": [9, 7, 33, 23],
        "_autoProtectedBoxes": [[0, 7, 8, 23]],
        "_autoSlotQualityReport": {
            "pass": True,
            "source_count": 2,
            "target_count": 2,
            "actual_count": 2,
            "length_change": "same",
            "issues": [],
        },
    }


class AutoRoiArtifactsTest(unittest.TestCase):
    def test_auto_roi_evidence_payload_records_search_and_edit_roi_contract(self) -> None:
        payload = auto_roi_evidence_payload([region()])
        self.assertEqual(payload["region_count"], 1)
        self.assertTrue(payload["all_have_search_roi"])
        self.assertTrue(payload["all_have_edit_roi"])
        self.assertTrue(payload["all_edit_roi_within_search_roi"])
        self.assertTrue(payload["all_search_area_gte_edit_area"])
        self.assertTrue(payload["all_edit_roi_avoid_protected_text"])
        item = payload["regions"][0]
        self.assertEqual(item["field_key"], "name")
        self.assertEqual(item["search_roi"], [2, 4, 46, 26])
        self.assertEqual(item["edit_roi"], [10, 8, 32, 22])
        self.assertEqual(item["target_roi"], [9, 7, 33, 23])
        self.assertGreater(item["roi_geometry"]["search_protected_overlap_pixels"], 0)
        self.assertEqual(item["roi_geometry"]["edit_protected_overlap_pixels"], 0)
        self.assertTrue(item["slot_quality_pass"])

    def test_auto_roi_overlay_draws_and_saves_annotated_regions(self) -> None:
        image = Image.new("RGB", (54, 34), (230, 230, 230))
        overlay = draw_auto_roi_overlay(image, [region()])
        pixels = overlay.load()
        self.assertEqual(pixels[2, 4], (0, 130, 0))
        self.assertEqual(pixels[9, 7], (0, 86, 210))
        self.assertEqual(pixels[10, 8], (210, 28, 28))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "auto_roi_overlay.png"
            saved = save_auto_roi_overlay(image, [region()], path)
            self.assertEqual(saved, str(path))
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
