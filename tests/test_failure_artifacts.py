from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import roi_image_edit.processing_service as processing_service
from roi_image_edit.iterative_pipeline import read_json
from roi_image_edit.processing_service import image_to_data_url, process_payload


class FailureArtifactsTest(unittest.TestCase):
    def test_auto_roi_failure_preserves_rejected_artifacts_and_result_evidence(self) -> None:
        image = Image.new("RGB", (32, 20), (230, 230, 230))
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "output"
            with patch.object(processing_service, "OUTPUT_DIR", output_dir):
                with patch.object(processing_service, "load_processing_prompts", return_value=("master", "rank", "final")):
                    with patch.object(processing_service, "VisionClient", return_value=object()):
                        with patch.object(
                            processing_service,
                            "auto_orient_for_instruction",
                            side_effect=ValueError("无法自动定位name中的旧文字：张三"),
                        ):
                            response = process_payload(
                                {
                                    "profile": "photo_scan",
                                    "images": [
                                        {
                                            "id": "img1",
                                            "filename": "missing_name.png",
                                            "instruction": "姓名张三修改为李四",
                                            "dataUrl": image_to_data_url(image),
                                            "regions": [],
                                        }
                                    ],
                                }
                            )
            self.assertTrue(response["ok"])
            image_result = response["images"][0]
            self.assertFalse(image_result["ok"])
            self.assertFalse(image_result["accepted"])
            self.assertFalse(image_result["applied"])
            self.assertEqual(image_result["candidates"], [])
            self.assertEqual(image_result["regions"], [])
            self.assertIn("无法自动定位name中的旧文字", image_result["error"])
            self.assertEqual(image_result["instructionDetails"]["field"], "name")
            failure = image_result["stage_evidence"]["failure"]
            self.assertEqual(failure["failure_stage"], "pre_candidate_generation")
            self.assertEqual(failure["candidate_count"], 0)
            rejected_input = Path(image_result["artifacts"]["rejected_input"])
            failure_report = Path(image_result["artifacts"]["failure_report"])
            self.assertTrue(rejected_input.exists())
            self.assertTrue(failure_report.exists())
            saved_report = read_json(failure_report)
            self.assertEqual(saved_report["accepted"], False)
            self.assertEqual(saved_report["applied"], False)
            result_json = read_json(Path(response["runDir"]) / "result.json")
            result_image = result_json["images"][0]
            self.assertFalse(result_image["applied"])
            self.assertEqual(result_image["stage_evidence"]["failure"]["candidate_count"], 0)
            progress_lines = (Path(response["runDir"]) / "progress.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertTrue(any('"event": "image_failed"' in line for line in progress_lines))


if __name__ == "__main__":
    unittest.main()
