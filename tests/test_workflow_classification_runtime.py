from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

import roi_image_edit.processing_service as processing_service
from roi_image_edit.processing_service import image_to_data_url, process_payload


def text_image(size: tuple[int, int] = (260, 180), *, color: tuple[int, int, int] = (238, 236, 232)) -> Image.Image:
    image = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(image)
    draw.rectangle([24, 34, 60, 46], fill=(32, 32, 32))
    draw.rectangle([78, 34, 112, 46], fill=(32, 32, 32))
    return image


def clean_number_image() -> Image.Image:
    image = Image.new("RGB", (320, 220), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle([24, 34, 60, 46], fill=(30, 30, 30))
    return image


class WorkflowClassificationRuntimeTest(unittest.TestCase):
    def fake_region_processor(self, region_image, roi, *, run_dir, region_id, **kwargs):
        region_dir = Path(run_dir) / "regions" / region_id
        region_dir.mkdir(parents=True, exist_ok=True)
        selected_candidate = region_dir / "selected_candidate.png"
        selected_compare = region_dir / "selected_candidate_compare.png"
        slot_report = region_dir / "slot_quality_report.json"
        pre_gate_report = region_dir / "pre_candidate_gate_report.json"
        roi_plan_report = region_dir / "roi_plan_report.json"
        region_image.save(selected_candidate)
        region_image.save(selected_compare)
        classification = dict(kwargs.get("classification") or {})
        roi_plan = {
            "search_roi": list(roi),
            "edit_roi": [roi[0] + 8, roi[1] + 4, min(roi[2], roi[0] + 60), min(roi[3], roi[1] + 24)],
            "expanded_edit_roi": None,
            "roi_policy": classification.get("roi_policy"),
        }
        slot_report.write_text(json.dumps({"pass": True, "classification": classification}), encoding="utf-8")
        pre_gate_report.write_text(json.dumps({"pass": True, "candidate_count": 3}), encoding="utf-8")
        roi_plan_report.write_text(json.dumps(roi_plan), encoding="utf-8")
        summary = {
            "plan": {
                "classification": classification,
                "class_key": classification.get("class_key"),
                "roi_policy": classification.get("roi_policy"),
                "internal_profile": classification.get("internal_profile"),
                "profile_source": classification.get("profile_source"),
                "roi_plan": roi_plan,
                "slot_quality_report": {"pass": True, "classification": classification},
            },
            "hard_check": {
                "stage_gate": {
                    "blocking_stage": None,
                    "order": ["hard_boundary", "text_shape", "ink_gray_balance"],
                },
            },
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
                "roi_plan_report": str(roi_plan_report),
                "display_image_is_candidate": False,
            },
        }
        return region_image.copy(), region_image.copy(), [], summary, True

    def fake_text_removal_processor(self, region_image, roi, *, run_dir, region_id, **kwargs):
        region_dir = Path(run_dir) / "regions" / region_id
        region_dir.mkdir(parents=True, exist_ok=True)
        selected_candidate = region_dir / "selected_text_removal.png"
        selected_compare = region_dir / "selected_text_removal_compare.png"
        report_path = region_dir / "text_removal_report.json"
        roi_plan_report = region_dir / "roi_plan_report.json"
        region_image.save(selected_candidate)
        region_image.save(selected_compare)
        classification = dict(kwargs.get("classification") or {})
        roi_plan = {
            "search_roi": list(roi),
            "edit_roi": list(roi),
            "expanded_edit_roi": None,
            "roi_policy": classification.get("roi_policy"),
        }
        report_path.write_text(json.dumps({"pass": True, "operation": "remove_text"}), encoding="utf-8")
        roi_plan_report.write_text(json.dumps(roi_plan), encoding="utf-8")
        summary = {
            "plan": {
                "classification": classification,
                "class_key": classification.get("class_key"),
                "roi_policy": classification.get("roi_policy"),
                "internal_profile": classification.get("internal_profile"),
                "profile_source": classification.get("profile_source"),
                "roi_plan": roi_plan,
            },
            "hard_check": {"pass": True},
            "vision": {"artifacts": {}, "revision_rounds": [], "enabled": False},
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
                "text_removal_report": str(report_path),
                "roi_plan_report": str(roi_plan_report),
                "display_image_is_candidate": False,
            },
        }
        return region_image.copy(), region_image.copy(), [], summary, True

    def fake_text_redaction_processor(self, region_image, roi, *, run_dir, region_id, **kwargs):
        region_dir = Path(run_dir) / "regions" / region_id
        region_dir.mkdir(parents=True, exist_ok=True)
        selected_candidate = region_dir / "selected_text_redaction.png"
        selected_compare = region_dir / "selected_text_redaction_compare.png"
        report_path = region_dir / "text_redaction_report.json"
        roi_plan_report = region_dir / "roi_plan_report.json"
        region_image.save(selected_candidate)
        region_image.save(selected_compare)
        classification = dict(kwargs.get("classification") or {})
        roi_plan = {
            "search_roi": list(roi),
            "edit_roi": list(roi),
            "expanded_edit_roi": None,
            "roi_policy": classification.get("roi_policy"),
        }
        report_path.write_text(json.dumps({"pass": True, "operation": "redact_text"}), encoding="utf-8")
        roi_plan_report.write_text(json.dumps(roi_plan), encoding="utf-8")
        summary = {
            "plan": {
                "classification": classification,
                "class_key": classification.get("class_key"),
                "roi_policy": classification.get("roi_policy"),
                "internal_profile": classification.get("internal_profile"),
                "profile_source": classification.get("profile_source"),
                "roi_plan": roi_plan,
            },
            "hard_check": {"pass": True},
            "vision": {"artifacts": {}, "revision_rounds": [], "enabled": False},
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
                "text_redaction_report": str(report_path),
                "roi_plan_report": str(roi_plan_report),
                "display_image_is_candidate": False,
            },
        }
        return region_image.copy(), region_image.copy(), [], summary, True

    def fake_amount_replacement_processor(self, region_image, roi, *, run_dir, region_id, **kwargs):
        region_dir = Path(run_dir) / "regions" / region_id
        region_dir.mkdir(parents=True, exist_ok=True)
        selected_candidate = region_dir / "selected_amount_replacement.png"
        selected_compare = region_dir / "selected_amount_replacement_compare.png"
        report_path = region_dir / "amount_replacement_report.json"
        roi_plan_report = region_dir / "roi_plan_report.json"
        region_image.save(selected_candidate)
        region_image.save(selected_compare)
        classification = dict(kwargs.get("classification") or {})
        roi_plan = {
            "search_roi": list(roi),
            "edit_roi": list(roi),
            "expanded_edit_roi": None,
            "roi_policy": classification.get("roi_policy"),
            "alignment": "right_anchor_preserve_suffix",
        }
        report_path.write_text(json.dumps({"pass": True, "operation": "replace_amount_value"}), encoding="utf-8")
        roi_plan_report.write_text(json.dumps(roi_plan), encoding="utf-8")
        summary = {
            "plan": {
                "classification": classification,
                "class_key": classification.get("class_key"),
                "roi_policy": classification.get("roi_policy"),
                "internal_profile": classification.get("internal_profile"),
                "profile_source": classification.get("profile_source"),
                "roi_plan": roi_plan,
            },
            "hard_check": {"pass": True},
            "vision": {"artifacts": {}, "revision_rounds": [], "enabled": False},
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
                "amount_replacement_report": str(report_path),
                "roi_plan_report": str(roi_plan_report),
                "display_image_is_candidate": False,
            },
        }
        return region_image.copy(), region_image.copy(), [], summary, True

    def fake_amount_glyph_clone_processor(self, region_image, roi, *, run_dir, region_id, glyph_sources, **kwargs):
        region_dir = Path(run_dir) / "regions" / region_id
        region_dir.mkdir(parents=True, exist_ok=True)
        selected_candidate = region_dir / "selected_amount_glyph_clone.png"
        selected_compare = region_dir / "selected_amount_glyph_clone_preview.png"
        report_path = region_dir / "amount_glyph_clone_report.json"
        region_image.save(selected_candidate)
        region_image.save(selected_compare)
        classification = dict(kwargs.get("classification") or {})
        report_path.write_text(
            json.dumps(
                {
                    "pass": True,
                    "operation": "amount_glyph_clone",
                    "glyph_source_count": len(glyph_sources),
                }
            ),
            encoding="utf-8",
        )
        summary = {
            "plan": {
                "classification": classification,
                "class_key": classification.get("class_key"),
                "roi_policy": classification.get("roi_policy"),
                "internal_profile": classification.get("internal_profile"),
                "profile_source": classification.get("profile_source"),
            },
            "hard_check": {"pass": True},
            "vision": {"artifacts": {}, "revision_rounds": [], "enabled": False},
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
                "amount_glyph_clone_report": str(report_path),
                "display_image_is_candidate": False,
            },
        }
        return region_image.copy(), region_image.copy(), [], summary, True

    def test_manual_roi_is_classified_before_region_processing_and_progress_records_manual_anchor(self) -> None:
        image = text_image()
        progress_records: list[dict] = []

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "output"
            with patch.object(processing_service, "OUTPUT_DIR", output_dir):
                with patch.object(processing_service, "load_processing_prompts", return_value=("master", "rank", "final")):
                    with patch.object(processing_service, "VisionClient", return_value=object()):
                        with patch.object(processing_service, "process_region", side_effect=self.fake_region_processor) as region_processor:
                            response = process_payload(
                                {
                                    "images": [
                                        {
                                            "id": "img1",
                                            "filename": "manual_anchor.png",
                                            "instruction": "姓名甲乙修改为丙丁",
                                            "dataUrl": image_to_data_url(image),
                                            "regions": [
                                                {
                                                    "id": "manual_big",
                                                    "rect": {"x": 0, "y": 20, "w": 180, "h": 52},
                                                    "sourceText": "甲乙",
                                                    "targetText": "丙丁",
                                                }
                                            ],
                                        }
                                    ],
                                },
                                progress=lambda _event, record: progress_records.append(record),
                            )

        region_processor.assert_called_once()
        classification = region_processor.call_args.kwargs["classification"]
        self.assertEqual(classification["roi_input"], "manual")
        self.assertEqual(classification["roi_policy"], "manual_anchor")
        self.assertEqual(classification["profile_source"], "classification")
        region_started = [record for record in progress_records if record.get("event") == "region_started"][0]
        self.assertEqual(region_started["roi_policy"], "manual_anchor")
        self.assertEqual(region_started["class_key"], "photo_document.form_field_value_replace.cjk")
        self.assertTrue(response["images"][0]["accepted"])

    def test_mixed_batch_classifies_each_image_and_profile_independently(self) -> None:
        photo = text_image()
        clean = clean_number_image()
        progress_records: list[dict] = []

        def fake_auto_orient(image, **_kwargs):
            return (
                image,
                [
                    {
                        "id": "auto_number",
                        "rect": {"x": 10, "y": 20, "w": 90, "h": 40},
                        "auto": True,
                        "sourceText": "12",
                        "targetText": "34",
                    }
                ],
                {
                    "applied": False,
                    "orientation": "none",
                    "attempts": [],
                    "selected_attempt": {"orientation": "none", "direction_score": 1.0},
                    "selected_score": 1.0,
                },
            )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "output"
            with patch.object(processing_service, "OUTPUT_DIR", output_dir):
                with patch.object(processing_service, "load_processing_prompts", return_value=("master", "rank", "final")):
                    with patch.object(processing_service, "VisionClient", return_value=object()):
                        with patch.object(processing_service, "auto_orient_for_instruction", side_effect=fake_auto_orient):
                            with patch.object(processing_service, "process_region", side_effect=self.fake_region_processor) as region_processor:
                                response = process_payload(
                                    {
                                        "images": [
                                            {
                                                "id": "photo",
                                                "filename": "photo.png",
                                                "instruction": "姓名甲乙修改为丙丁",
                                                "dataUrl": image_to_data_url(photo),
                                                "regions": [
                                                    {
                                                        "id": "manual_photo",
                                                        "rect": {"x": 0, "y": 20, "w": 180, "h": 52},
                                                        "sourceText": "甲乙",
                                                        "targetText": "丙丁",
                                                    }
                                                ],
                                            },
                                            {
                                                "id": "clean",
                                                "filename": "clean.png",
                                                "instruction": "编号12修改为34",
                                                "dataUrl": image_to_data_url(clean),
                                                "regions": [],
                                            },
                                        ],
                                    },
                                    progress=lambda _event, record: progress_records.append(record),
                                )

        self.assertEqual(region_processor.call_count, 2)
        first_classification = region_processor.call_args_list[0].kwargs["classification"]
        second_classification = region_processor.call_args_list[1].kwargs["classification"]
        self.assertEqual(first_classification["class_key"], "photo_document.form_field_value_replace.cjk")
        self.assertEqual(first_classification["internal_profile"], "photo_scan")
        self.assertEqual(first_classification["roi_policy"], "manual_anchor")
        self.assertEqual(second_classification["class_key"], "clean_digital.numeric_or_date_replace")
        self.assertEqual(second_classification["internal_profile"], "clean_digital")
        self.assertEqual(second_classification["roi_policy"], "auto")
        self.assertNotEqual(first_classification["class_key"], second_classification["class_key"])
        self.assertNotEqual(first_classification["internal_profile"], second_classification["internal_profile"])
        image_started = [record for record in progress_records if record.get("event") == "image_started"]
        self.assertEqual([record["image_id"] for record in image_started], ["photo", "clean"])
        self.assertEqual(image_started[0]["internal_profile"], "photo_scan")
        self.assertEqual(image_started[1]["internal_profile"], "clean_digital")
        self.assertEqual(response["images"][0]["classification"]["class_key"], first_classification["class_key"])
        self.assertEqual(response["images"][1]["classification"]["class_key"], second_classification["class_key"])

    def test_text_removal_class_uses_isolated_processor(self) -> None:
        image = text_image()
        progress_records: list[dict] = []

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "output"
            with patch.object(processing_service, "OUTPUT_DIR", output_dir):
                with patch.object(processing_service, "load_processing_prompts", return_value=("master", "rank", "final")):
                    with patch.object(processing_service, "VisionClient", return_value=object()):
                        with patch.object(processing_service, "process_region", side_effect=AssertionError("replacement path used")):
                            with patch.object(
                                processing_service,
                                "process_text_removal_region",
                                side_effect=self.fake_text_removal_processor,
                            ) as text_removal_processor:
                                response = process_payload(
                                    {
                                        "images": [
                                            {
                                                "id": "remove",
                                                "filename": "remove.png",
                                                "instruction": "将图片中的提示区下面的甲乙丙丁四个字抹除",
                                                "dataUrl": image_to_data_url(image),
                                                "regions": [
                                                    {
                                                        "id": "manual_remove",
                                                        "rect": {"x": 20, "y": 30, "w": 100, "h": 30},
                                                        "sourceText": "甲乙丙丁",
                                                        "targetText": "",
                                                    }
                                                ],
                                            }
                                        ],
                                    },
                                    progress=lambda _event, record: progress_records.append(record),
                                )

        text_removal_processor.assert_called_once()
        classification = text_removal_processor.call_args.kwargs["classification"]
        self.assertEqual(classification["scenario"], "anchored_text_removal")
        self.assertEqual(classification["class_key"], "photo_document.anchored_text_removal.cjk")
        self.assertEqual(classification["operation"], "remove_text")
        self.assertTrue(response["images"][0]["accepted"])
        region_started = [record for record in progress_records if record.get("event") == "region_started"][0]
        self.assertEqual(region_started["class_key"], "photo_document.anchored_text_removal.cjk")

    def test_text_redaction_class_uses_isolated_processor(self) -> None:
        image = text_image()
        progress_records: list[dict] = []

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "output"
            with patch.object(processing_service, "OUTPUT_DIR", output_dir):
                with patch.object(processing_service, "load_processing_prompts", return_value=("master", "rank", "final")):
                    with patch.object(processing_service, "VisionClient", return_value=object()):
                        with patch.object(processing_service, "process_region", side_effect=AssertionError("replacement path used")):
                            with patch.object(
                                processing_service,
                                "process_text_removal_region",
                                side_effect=AssertionError("removal path used"),
                            ):
                                with patch.object(
                                    processing_service,
                                    "process_text_redaction_region",
                                    side_effect=self.fake_text_redaction_processor,
                                ) as text_redaction_processor:
                                    response = process_payload(
                                        {
                                            "images": [
                                                {
                                                    "id": "redact",
                                                    "filename": "redact.png",
                                                    "instruction": "将 Tim 打码",
                                                    "dataUrl": image_to_data_url(image),
                                                    "regions": [
                                                        {
                                                            "id": "manual_redact",
                                                            "rect": {"x": 20, "y": 30, "w": 100, "h": 30},
                                                            "sourceText": "Tim",
                                                            "targetText": "",
                                                        }
                                                    ],
                                                }
                                            ],
                                        },
                                        progress=lambda _event, record: progress_records.append(record),
                                    )

        text_redaction_processor.assert_called_once()
        classification = text_redaction_processor.call_args.kwargs["classification"]
        self.assertEqual(classification["scenario"], "text_redaction")
        self.assertEqual(classification["class_key"], "photo_document.text_redaction.latin")
        self.assertEqual(classification["operation"], "redact_text")
        self.assertTrue(response["images"][0]["accepted"])
        region_started = [record for record in progress_records if record.get("event") == "region_started"][0]
        self.assertEqual(region_started["class_key"], "photo_document.text_redaction.latin")

    def test_amount_replacement_class_uses_isolated_processor(self) -> None:
        image = text_image(size=(591, 1280), color=(255, 255, 255))
        progress_records: list[dict] = []

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "output"
            with patch.object(processing_service, "OUTPUT_DIR", output_dir):
                with patch.object(processing_service, "load_processing_prompts", return_value=("master", "rank", "final")):
                    with patch.object(processing_service, "VisionClient", return_value=object()):
                        with patch.object(processing_service, "process_region", side_effect=AssertionError("replacement path used")):
                            with patch.object(
                                processing_service,
                                "process_text_removal_region",
                                side_effect=AssertionError("removal path used"),
                            ):
                                with patch.object(
                                    processing_service,
                                    "process_text_redaction_region",
                                    side_effect=AssertionError("redaction path used"),
                                ):
                                    with patch.object(
                                        processing_service,
                                        "process_amount_replacement_region",
                                        side_effect=self.fake_amount_replacement_processor,
                                    ) as amount_processor:
                                        response = process_payload(
                                            {
                                                "images": [
                                                    {
                                                        "id": "amount",
                                                        "filename": "amount.png",
                                                        "instruction": "金额+9764修改为+12749",
                                                        "dataUrl": image_to_data_url(image),
                                                        "regions": [
                                                            {
                                                                "id": "manual_amount",
                                                                "rect": {"x": 428, "y": 306, "w": 84, "h": 26},
                                                                "sourceText": "+9764",
                                                                "targetText": "+12749",
                                                            }
                                                        ],
                                                    }
                                                ],
                                            },
                                            progress=lambda _event, record: progress_records.append(record),
                                        )

        amount_processor.assert_called_once()
        classification = amount_processor.call_args.kwargs["classification"]
        self.assertEqual(classification["scenario"], "amount_value_replace")
        self.assertEqual(classification["class_key"], "photo_document.amount_value_replace.numeric_or_date")
        self.assertEqual(classification["internal_profile"], "clean_digital")
        self.assertTrue(response["images"][0]["accepted"])
        region_started = [record for record in progress_records if record.get("event") == "region_started"][0]
        self.assertEqual(region_started["class_key"], "photo_document.amount_value_replace.numeric_or_date")

    def test_amount_glyph_clone_class_uses_isolated_processor(self) -> None:
        image = text_image(size=(591, 1280), color=(255, 255, 255))
        progress_records: list[dict] = []

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "output"
            with patch.object(processing_service, "OUTPUT_DIR", output_dir):
                with patch.object(processing_service, "load_processing_prompts", return_value=("master", "rank", "final")):
                    with patch.object(processing_service, "VisionClient", return_value=object()):
                        with patch.object(processing_service, "process_region", side_effect=AssertionError("replacement path used")):
                            with patch.object(
                                processing_service,
                                "process_amount_replacement_region",
                                side_effect=AssertionError("amount replacement path used"),
                            ):
                                with patch.object(
                                    processing_service,
                                    "process_amount_glyph_clone_region",
                                    side_effect=self.fake_amount_glyph_clone_processor,
                                ) as clone_processor:
                                    response = process_payload(
                                        {
                                            "images": [
                                                {
                                                    "id": "amount",
                                                    "filename": "amount.png",
                                                    "instruction": "金额+5739复用为+22882",
                                                    "dataUrl": image_to_data_url(image),
                                                    "glyphSources": [
                                                        {
                                                            "text": "+5626",
                                                            "dataUrl": image_to_data_url(image),
                                                            "rect": {"x": 428, "y": 302, "w": 84, "h": 32},
                                                        }
                                                    ],
                                                    "regions": [
                                                        {
                                                            "id": "manual_amount",
                                                            "rect": {"x": 428, "y": 1154, "w": 84, "h": 28},
                                                            "sourceText": "+5739",
                                                            "targetText": "+22882",
                                                        }
                                                    ],
                                                }
                                            ],
                                        },
                                        progress=lambda _event, record: progress_records.append(record),
                                    )

        clone_processor.assert_called_once()
        classification = clone_processor.call_args.kwargs["classification"]
        self.assertEqual(classification["scenario"], "amount_glyph_clone")
        self.assertEqual(classification["class_key"], "photo_document.amount_glyph_clone.numeric_or_date")
        self.assertEqual(classification["internal_profile"], "clean_digital")
        self.assertEqual(len(clone_processor.call_args.kwargs["glyph_sources"]), 1)
        self.assertTrue(response["images"][0]["accepted"])
        region_started = [record for record in progress_records if record.get("event") == "region_started"][0]
        self.assertEqual(region_started["class_key"], "photo_document.amount_glyph_clone.numeric_or_date")


if __name__ == "__main__":
    unittest.main()
