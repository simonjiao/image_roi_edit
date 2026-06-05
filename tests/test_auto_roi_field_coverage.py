from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

import roi_image_edit.processing_service as processing_service
from roi_image_edit.processing_service import image_to_data_url, process_payload
from roi_image_edit.roi_locator import auto_select_regions_for_instruction, parse_instruction_details


def component_field_image(
    *,
    label_parts: int,
    value_parts: int,
    y: int,
    width: int = 220,
    height: int = 70,
) -> Image.Image:
    image = Image.new("RGB", (width, height), (235, 235, 235))
    draw = ImageDraw.Draw(image)
    x = 12
    for _idx in range(label_parts):
        draw.rectangle([x, y, x + 10, y + 16], fill=(20, 20, 20))
        x += 13
    separator_x = x + 3
    draw.rectangle([separator_x, y, separator_x + 3, y + 12], fill=(20, 20, 20))
    x = separator_x + 14
    for _idx in range(value_parts):
        draw.rectangle([x, y, x + 10, y + 16], fill=(20, 20, 20))
        x += 15
    return image


class AutoRoiFieldCoverageTest(unittest.TestCase):
    def assert_auto_field_region(
        self,
        *,
        field: str,
        instruction: str,
        image: Image.Image,
    ) -> dict:
        details = parse_instruction_details(instruction)
        self.assertEqual(details["field"], field)
        regions = auto_select_regions_for_instruction(
            image,
            instruction=instruction,
            source_text=details["source_text"],
            target_text=details["target_text"],
        )
        self.assertEqual(len(regions), 1)
        region = regions[0]
        self.assertEqual(region["_autoFieldKey"], field)
        self.assertTrue(region["auto"])
        self.assertGreater(region["_autoScore"], -1000)
        self.assertGreater(region["rect"]["w"], 0)
        self.assertGreater(region["rect"]["h"], 0)
        self.assertLessEqual(region["_autoSearchRoi"][0], region["_autoEditRoi"][0])
        self.assertLessEqual(region["_autoSearchRoi"][1], region["_autoEditRoi"][1])
        self.assertGreaterEqual(region["_autoSearchRoi"][2], region["_autoEditRoi"][2])
        self.assertGreaterEqual(region["_autoSearchRoi"][3], region["_autoEditRoi"][3])
        self.assertIn("pass", region["_autoSlotQualityReport"])
        return region

    def test_auto_roi_common_path_covers_name_date_age_and_number_fields(self) -> None:
        name = self.assert_auto_field_region(
            field="name",
            instruction="姓名A修改为B",
            image=component_field_image(label_parts=2, value_parts=1, y=4),
        )
        self.assertEqual(name["sourceText"], "A")
        self.assertEqual(name["targetText"], "B")
        self.assertTrue(name["_autoSlotQualityReport"]["pass"])

        date = self.assert_auto_field_region(
            field="date",
            instruction="日期修改为2025-02-03",
            image=component_field_image(label_parts=1, value_parts=6, y=10),
        )
        self.assertEqual(date["sourceText"], "0000-00-00")
        self.assertEqual(date["targetText"], "2025-02-03")

        age = self.assert_auto_field_region(
            field="age",
            instruction="年龄修改为19",
            image=component_field_image(label_parts=1, value_parts=2, y=10),
        )
        self.assertEqual(age["sourceText"], "00")
        self.assertEqual(age["targetText"], "19")

        number = self.assert_auto_field_region(
            field="number",
            instruction="编号修改为67890",
            image=component_field_image(label_parts=1, value_parts=5, y=10),
        )
        self.assertEqual(number["sourceText"], "00000")
        self.assertEqual(number["targetText"], "67890")

    def test_manual_roi_fallback_skips_auto_roi_selection(self) -> None:
        image = component_field_image(label_parts=2, value_parts=1, y=4)

        def fake_process_region(region_image, roi, *, run_dir, region_id, **_kwargs):
            region_dir = Path(run_dir) / "regions" / region_id
            region_dir.mkdir(parents=True, exist_ok=True)
            selected_candidate = region_dir / "selected_candidate.png"
            selected_compare = region_dir / "selected_candidate_compare.png"
            slot_report = region_dir / "slot_quality_report.json"
            pre_gate_report = region_dir / "pre_candidate_gate_report.json"
            image.save(selected_candidate)
            image.save(selected_compare)
            slot_report.write_text('{"pass": true}', encoding="utf-8")
            pre_gate_report.write_text('{"pass": true}', encoding="utf-8")
            summary = {
                "plan": {"slot_quality_report": {"pass": True}},
                "hard_check": {"stage_gate": {"blocking_stage": None}},
                "vision": {"artifacts": {}, "revision_rounds": []},
                "trace": {
                    "accepted": True,
                    "final_is_rejected_candidate": False,
                    "final_blocking_stage": None,
                },
                "accepted": True,
                "applied": True,
                "artifacts": {
                    "selected_candidate": str(selected_candidate),
                    "selected_compare": str(selected_compare),
                    "slot_quality_report": str(slot_report),
                    "pre_candidate_gate_report": str(pre_gate_report),
                    "display_image_is_candidate": False,
                },
            }
            return region_image.copy(), region_image.copy(), [], summary, True

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "output"
            with patch.object(processing_service, "OUTPUT_DIR", output_dir):
                with patch.object(processing_service, "load_processing_prompts", return_value=("master", "rank", "final")):
                    with patch.object(processing_service, "VisionClient", return_value=object()):
                        with patch.object(processing_service, "auto_orient_for_instruction") as auto_roi:
                            with patch.object(
                                processing_service,
                                "process_region",
                                side_effect=fake_process_region,
                            ) as region_processor:
                                response = process_payload(
                                    {
                                        "profile": "photo_scan",
                                        "images": [
                                            {
                                                "id": "img1",
                                                "filename": "manual.png",
                                                "instruction": "B",
                                                "dataUrl": image_to_data_url(image),
                                                "regions": [
                                                    {
                                                        "id": "manual_1",
                                                        "rect": {"x": 4, "y": 0, "w": 88, "h": 26},
                                                        "sourceText": "A",
                                                        "targetText": "B",
                                                    }
                                                ],
                                            }
                                        ],
                                    }
                                )

        auto_roi.assert_not_called()
        region_processor.assert_called_once()
        self.assertTrue(response["images"][0]["accepted"])
        self.assertFalse(response["images"][0]["regions"][0]["auto"])
        self.assertEqual(response["images"][0]["regions"][0]["roi"], [4, 0, 92, 26])


if __name__ == "__main__":
    unittest.main()
