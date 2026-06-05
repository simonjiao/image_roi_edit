from __future__ import annotations

import json
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
            self.assertEqual(failure["pre_candidate_gate_report"]["failed_gate"], "field_roi_selection")
            self.assertEqual(failure["pre_candidate_gate_report"]["candidate_count"], 0)
            rejected_input = Path(image_result["artifacts"]["rejected_input"])
            failure_report = Path(image_result["artifacts"]["failure_report"])
            orientation_report = Path(image_result["artifacts"]["auto_orientation_report"])
            self.assertTrue(rejected_input.exists())
            self.assertTrue(failure_report.exists())
            self.assertTrue(orientation_report.exists())
            self.assertTrue(image_result["artifacts"]["final_is_rejected_candidate"])
            saved_report = read_json(failure_report)
            self.assertEqual(saved_report["accepted"], False)
            self.assertEqual(saved_report["applied"], False)
            self.assertEqual(read_json(orientation_report)["orientation"], "none")
            result_path = Path(response["runDir"]) / "result.json"
            progress_path = Path(response["runDir"]) / "progress.jsonl"
            artifact_manifest_path = Path(response["artifactManifest"])
            self.assertTrue(result_path.exists())
            self.assertTrue(progress_path.exists())
            self.assertTrue(artifact_manifest_path.exists())
            result_json = read_json(result_path)
            self.assertEqual(result_json["artifactManifest"], str(artifact_manifest_path))
            result_image = result_json["images"][0]
            self.assertFalse(result_image["applied"])
            self.assertEqual(result_image["stage_evidence"]["failure"]["candidate_count"], 0)
            self.assertEqual(result_image["artifacts"]["final_is_rejected_candidate"], True)
            artifact_manifest = read_json(artifact_manifest_path)
            self.assertTrue(artifact_manifest["all_explainable"])
            self.assertEqual(artifact_manifest["missing_required"], [])
            manifest_image = artifact_manifest["images"][0]
            self.assertEqual(manifest_image["status"], "failed")
            self.assertTrue(manifest_image["explainable"])
            self.assertEqual(manifest_image["blocking_stage"], "pre_candidate_generation")
            self.assertIn(
                "failure_report",
                [item["key"] for item in manifest_image["reports"]],
            )
            self.assertIn(
                "auto_orientation_report",
                [item["key"] for item in manifest_image["reports"]],
            )
            self.assertIn(
                "rejected_input",
                [item["key"] for item in manifest_image["candidate_images"]],
            )
            self.assertIn(
                "pre_candidate_gate_report",
                [item["key"] for item in manifest_image["embedded_reports"]],
            )
            progress_lines = progress_path.read_text(encoding="utf-8").splitlines()
            self.assertTrue(any('"event": "image_failed"' in line for line in progress_lines))
            self.assertTrue(any('"event": "run_finished"' in line for line in progress_lines))
            progress_records = [json.loads(line) for line in progress_lines]
            run_finished = [record for record in progress_records if record.get("event") == "run_finished"][0]
            self.assertEqual(run_finished["artifact_manifest"], str(artifact_manifest_path))
            self.assertTrue(run_finished["all_explainable"])
            pre_candidate_events = [
                record for record in progress_records if record.get("event") == "pre_candidate_gate_failed"
            ]
            self.assertEqual(len(pre_candidate_events), 1)
            self.assertEqual(pre_candidate_events[0]["failed_gate"], "field_roi_selection")
            self.assertEqual(pre_candidate_events[0]["candidate_count"], 0)
            self.assertEqual(
                pre_candidate_events[0]["pre_candidate_gate_report"]["gate_order"],
                ["orientation_check", "field_roi_selection", "slot_quality_gate", "protected_text_guard"],
            )

    def test_rejected_auto_roi_task_preserves_required_artifact_family(self) -> None:
        image = Image.new("RGB", (64, 32), (230, 230, 230))

        def fake_auto_orient_for_instruction(*args, **kwargs):
            return (
                image.copy(),
                [
                    {
                        "id": "auto_name",
                        "auto": True,
                        "rect": {"x": 12, "y": 8, "w": 24, "h": 14},
                        "sourceText": "甲",
                        "targetText": "乙",
                        "_autoFieldKey": "name",
                        "_autoSearchRoi": [4, 5, 46, 25],
                        "_autoEditRoi": [12, 8, 36, 22],
                        "_autoTargetRoi": [12, 8, 36, 22],
                        "_autoProtectedBoxes": [[4, 8, 10, 22]],
                        "_autoSlotQualityReport": {"pass": True, "issues": []},
                    }
                ],
                {
                    "applied": False,
                    "orientation": "none",
                    "attempts": [
                        {
                            "orientation": "none",
                            "direction_quality": {"pass": True},
                            "target_field_quality": {"field": "name", "score": 1.0},
                            "old_value_location_quality": {"all_slot_quality_pass": True},
                            "selection_basis": "field_and_old_value_quality",
                        }
                    ],
                    "selected_attempt": {"orientation": "none", "direction_score": 1.0},
                    "selected_score": 1.0,
                    "final_direction_reason": "field_and_old_value_quality",
                },
            )

        def fake_process_region(region_image, roi, *, run_dir, region_id, **kwargs):
            region_dir = Path(run_dir) / "regions" / region_id
            evidence_dir = region_dir / "stage_evidence"
            evidence_dir.mkdir(parents=True, exist_ok=True)
            paths = {
                "selected_candidate": region_dir / "selected_candidate.png",
                "selected_compare": region_dir / "selected_candidate_compare.png",
                "slot_quality_report": region_dir / "slot_quality_report.json",
                "pre_candidate_gate_report": region_dir / "pre_candidate_gate_report.json",
                "candidate_sheet": region_dir / "vision_candidate_sheet.png",
                "final_compare": region_dir / "vision_final_compare.png",
                "text_shape_compare": evidence_dir / "text_shape_top_compare.png",
                "text_shape_report": evidence_dir / "text_shape_top_report.json",
                "ink_compare": evidence_dir / "ink_gray_balance_top_compare.png",
                "ink_report": evidence_dir / "ink_gray_balance_top_report.json",
                "photo_compare": evidence_dir / "photo_texture_top_compare.png",
                "photo_report": evidence_dir / "photo_texture_top_report.json",
                "stage_summary": evidence_dir / "summary.json",
            }
            for key, path in paths.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.suffix == ".png":
                    image.save(path)
                else:
                    path.write_text(json.dumps({"key": key}), encoding="utf-8")
            stage_evidence = {
                "summary": str(paths["stage_summary"]),
                "stages": {
                    "text_shape": {
                        "available": True,
                        "compare_path": str(paths["text_shape_compare"]),
                        "report_path": str(paths["text_shape_report"]),
                    },
                    "ink_gray_balance": {
                        "available": True,
                        "compare_path": str(paths["ink_compare"]),
                        "report_path": str(paths["ink_report"]),
                    },
                    "photo_texture": {
                        "available": True,
                        "compare_path": str(paths["photo_compare"]),
                        "report_path": str(paths["photo_report"]),
                    },
                },
            }
            summary = {
                "plan": {
                    "slot_quality_report": {"pass": True, "issues": []},
                },
                "hard_check": {
                    "stage_gate": {
                        "blocking_stage": "text_shape",
                    },
                },
                "vision": {
                    "artifacts": {
                        "candidate_sheet": str(paths["candidate_sheet"]),
                        "final_compare": str(paths["final_compare"]),
                        "revision_previews": [],
                    },
                    "revision_rounds": [],
                },
                "trace": {
                    "accepted": False,
                    "final_is_rejected_candidate": True,
                    "final_blocking_stage": "text_shape",
                    "last_round_stop_reason": "text_shape_not_passed",
                },
                "accepted": False,
                "applied": False,
                "artifacts": {
                    "selected_candidate": str(paths["selected_candidate"]),
                    "selected_compare": str(paths["selected_compare"]),
                    "slot_quality_report": str(paths["slot_quality_report"]),
                    "pre_candidate_gate_report": str(paths["pre_candidate_gate_report"]),
                    "stage_evidence": stage_evidence,
                    "display_image_is_candidate": True,
                },
            }
            return region_image.copy(), region_image.copy(), [], summary, False

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "output"
            with patch.object(processing_service, "OUTPUT_DIR", output_dir):
                with patch.object(processing_service, "load_processing_prompts", return_value=("master", "rank", "final")):
                    with patch.object(processing_service, "VisionClient", return_value=object()):
                        with patch.object(
                            processing_service,
                            "auto_orient_for_instruction",
                            side_effect=fake_auto_orient_for_instruction,
                        ):
                            with patch.object(
                                processing_service,
                                "process_region",
                                side_effect=fake_process_region,
                            ):
                                response = process_payload(
                                    {
                                        "profile": "photo_scan",
                                        "images": [
                                            {
                                                "id": "img1",
                                                "filename": "rejected_name.png",
                                                "instruction": "姓名甲修改为乙",
                                                "dataUrl": image_to_data_url(image),
                                                "regions": [],
                                            }
                                        ],
                                    }
                                )

            self.assertTrue(response["ok"])
            image_result = response["images"][0]
            self.assertFalse(image_result["accepted"])
            artifact_manifest = read_json(Path(response["artifactManifest"]))
            self.assertTrue(artifact_manifest["all_explainable"])
            manifest_image = artifact_manifest["images"][0]
            report_keys = [item["key"] for item in manifest_image["reports"]]
            image_keys = [item["key"] for item in manifest_image["candidate_images"]]
            for report_key in (
                "auto_orientation_report",
                "auto_roi_evidence_report",
                "slot_quality_report",
                "pre_candidate_gate_report",
                "text_shape_top_report",
                "ink_gray_balance_top_report",
                "photo_texture_top_report",
            ):
                self.assertIn(report_key, report_keys)
            for image_key in (
                "auto_roi_overlay",
                "selected_candidate",
                "selected_compare",
                "vision_candidate_sheet",
                "vision_final_compare",
                "text_shape_top_compare",
                "ink_gray_balance_top_compare",
                "photo_texture_top_compare",
                "final_image",
            ):
                self.assertIn(image_key, image_keys)
            for entry in manifest_image["reports"] + manifest_image["candidate_images"]:
                if entry["key"] in set(report_keys + image_keys):
                    self.assertTrue(Path(entry["path"]).exists(), entry["key"])


if __name__ == "__main__":
    unittest.main()
