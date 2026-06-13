from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

from roi_image_edit.iterative_pipeline import CandidateParams, RenderPlan, TextRun
from roi_image_edit.local_validation import apply_local_acceptance_gate
from roi_image_edit.image_classification import classify_image_workflow
from roi_image_edit.placement_strategy import choose_placement_strategy, placement_strategy_report
import roi_image_edit.roi_locator as roi_locator
from roi_image_edit.roi_locator import parse_instruction_details
from roi_image_edit.slot_quality import slot_quality_report
from roi_image_edit.stage_patchers import dispatch_revision_patches, stage_patch_filter_report
from roi_image_edit.stage_profiles import stage_profile
from roi_image_edit.stages import prompt_stage_context, stage_gate_for_report


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "regression_cases"


def _load_case(case_id: str) -> dict[str, Any]:
    path = FIXTURE_DIR / f"{case_id}.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _slot(box: list[int]) -> TextRun:
    x1, y1, x2, y2 = [int(value) for value in box]
    return TextRun(x1=x1, y1=y1, x2=x2, y2=y2, area=max(1, (x2 - x1) * (y2 - y1)))


def _image_with_slots(slots: tuple[TextRun, ...]) -> Image.Image:
    img = Image.new("RGB", (90, 42), (214, 214, 214))
    draw = ImageDraw.Draw(img)
    for run in slots:
        draw.rectangle((run.x1 + 2, run.y1 + 3, run.x2 - 3, run.y2 - 4), fill=(44, 44, 44))
        draw.rectangle((run.x1 + 1, run.y1 + 2, run.x2 - 2, run.y2 - 3), outline=(132, 132, 132))
    return img


def _classification_for_instruction(instruction: str, *, size: tuple[int, int] = (260, 220)) -> dict[str, Any]:
    return classify_image_workflow(
        Image.new("RGB", size, (214, 214, 214)),
        instruction_details=parse_instruction_details(instruction),
    )


def _value_at(payload: dict[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        if isinstance(value, list):
            value = value[int(part)]
        else:
            value = value[part]
    return value


def _assert_expected_fields(test: unittest.TestCase, payload: dict[str, Any], expected: list[dict[str, Any]]) -> None:
    for item in expected:
        with test.subTest(path=item["path"]):
            value = _value_at(payload, str(item["path"]))
            if "equals" in item:
                test.assertEqual(value, item["equals"])
            if "greater_equal" in item:
                test.assertGreaterEqual(value, item["greater_equal"])
            if "less_equal" in item:
                test.assertLessEqual(value, item["less_equal"])
            if "contains" in item:
                expected_value = item["contains"]
                if isinstance(expected_value, dict) and isinstance(value, list):
                    matches = [
                        entry
                        for entry in value
                        if isinstance(entry, dict)
                        and all(entry.get(key) == expected_item for key, expected_item in expected_value.items())
                    ]
                    test.assertTrue(matches, f"{expected_value!r} missing from {value!r}")
                else:
                    test.assertIn(expected_value, value)
            if "not_contains" in item:
                test.assertNotIn(item["not_contains"], value)
            if "contains_path" in item:
                expected_path = item["contains_path"]["path"]
                expected_value = item["contains_path"]["equals"]
                matches = [
                    entry
                    for entry in value
                    if isinstance(entry, dict) and _value_at(entry, expected_path) == expected_value
                ]
                test.assertTrue(matches, f"{expected_path}={expected_value!r} missing from {value!r}")
            if "no_patch_keys" in item:
                blocked = set(item["no_patch_keys"])
                offenders = [
                    patch
                    for patch in value
                    if isinstance(patch, dict) and blocked.intersection(patch)
                ]
                test.assertEqual(offenders, [])


class RegressionCaseContractsTest(unittest.TestCase):
    def test_case_a_field_edit_blocks_shape_before_cleanup_and_visual_deliver(self) -> None:
        case = _load_case("case_a_field_text_shape")
        report = dict(case["local_report"])
        stage_gate = stage_gate_for_report(report, case["profile"])
        visual_gate = apply_local_acceptance_gate(case["visual_acceptance"], report)
        patch_filter = stage_patch_filter_report(
            case["candidate_patches"],
            stage_gate["blocking_stage"],
            limit=10,
        )

        payload = {
            "classification": _classification_for_instruction(case["instruction"]),
            "instruction_details": parse_instruction_details(case["instruction"]),
            "local_report": report,
            "stage_gate": stage_gate,
            "visual_gate": visual_gate,
            "patch_filter": patch_filter,
        }
        _assert_expected_fields(self, payload, case["expected_report_fields"])

    def test_case_b_shorter_replacement_cleans_extra_slots_and_keeps_residue_out_of_photo_texture(self) -> None:
        case = _load_case("case_b_shorter_cleanup")
        slots = tuple(_slot(box) for box in case["slots"])
        slot_report = slot_quality_report(
            _image_with_slots(slots),
            tuple(case["roi"]),
            slots,
            source_text=case["source_text"],
            target_text=case["target_text"],
            protected_boxes=tuple(tuple(box) for box in case["protected_boxes"]),
        )
        strategy, reason = choose_placement_strategy(
            source_text=case["source_text"],
            target_text=case["target_text"],
            slots=slots,
            slot_report=slot_report,
            draw_mode="line_chars",
        )
        plan = RenderPlan(
            target_text=case["target_text"],
            source_text=case["source_text"],
            search_roi=tuple(case["roi"]),
            target_roi=(10, 10, 52, 24),
            slot_boxes=slots,
            protected_boxes=tuple(tuple(box) for box in case["protected_boxes"]),
            source_reference_box=(10, 10, 52, 24),
            style_reference_box=None,
            style_reference_text=None,
            draw_mode="line_chars",
            placement_strategy=strategy,
            placement_strategy_reason=reason,
            slot_quality_report=slot_report,
        )
        placement = placement_strategy_report(
            plan,
            case["alignment_metrics"],
            [],
            max_char_center_dx=2.0,
            max_char_center_dy=2.5,
            max_char_center_distance_delta=2.0,
            max_replacement_center_y_range=2.0,
        )
        residual_stage_gate = stage_gate_for_report(case["residual_report"], case["profile"])

        payload = {
            "classification": _classification_for_instruction(case["instruction"]),
            "slot_quality": slot_report,
            "placement": placement,
            "hard_report": case["hard_report"],
            "residual_stage_gate": residual_stage_gate,
        }
        _assert_expected_fields(self, payload, case["expected_report_fields"])

    def test_case_c_longer_replacement_expands_roi_and_shape_before_color(self) -> None:
        case = _load_case("case_c_longer_protected")
        slots = tuple(_slot(box) for box in case["slots"])
        image = _image_with_slots(slots)
        with patch.object(roi_locator, "slots_for_region", return_value=slots):
            with patch.object(
                roi_locator,
                "protected_boxes_for_region",
                return_value=tuple(tuple(box) for box in case["protected_boxes"]),
            ):
                plan = roi_locator.build_region_plan(
                    image,
                    tuple(case["roi"]),
                    source_text=case["source_text"],
                    target_text=case["target_text"],
                    field_key="name",
                )
        classification = classify_image_workflow(
            image.resize((220, 220)),
            instruction_details=parse_instruction_details(case["instruction"]),
        )
        stage_gate = stage_gate_for_report(case["shape_report"], case["profile"])
        patch_filter = stage_patch_filter_report(
            case["candidate_patches"],
            stage_gate["blocking_stage"],
            limit=10,
        )

        payload = {
            "classification": classification,
            "plan": {
                "roi_plan": {
                    "expanded_edit_roi": plan.slot_quality_report["length_change_report"]["expanded_edit_roi"],
                    "expansion_report": plan.slot_quality_report["length_change_report"]["expansion_report"],
                },
                "slot_quality_report": plan.slot_quality_report,
            },
            "slot_quality": plan.slot_quality_report,
            "stage_gate": stage_gate,
            "patch_filter": patch_filter,
            "candidate_count": 3,
        }
        _assert_expected_fields(self, payload, case["expected_report_fields"])

    def test_case_d_clean_digital_keeps_photo_texture_and_noise_disabled(self) -> None:
        case = _load_case("case_d_clean_digital_number")
        profile = stage_profile(case["profile"]).as_report()
        report = dict(case["local_report"])
        params = CandidateParams(
            candidate_id="case_d",
            font_name="test-font",
            font_path="/tmp/test-font.ttf",
            font_size=18,
            opacity=0.86,
            blur=0.0,
        )
        dispatch = dispatch_revision_patches(params, case["visual_acceptance"], report)
        payload = {
            "instruction_details": parse_instruction_details(case["instruction"]),
            "classification": case["classification"],
            "profile": profile,
            "stage_gate": stage_gate_for_report(report, case["profile"]),
            "prompt_context": prompt_stage_context(report, case["profile"]),
            "dispatch": dispatch,
        }
        _assert_expected_fields(self, payload, case["expected_report_fields"])

    def test_all_regression_fixtures_store_commands_and_expected_report_fields(self) -> None:
        for path in sorted(FIXTURE_DIR.glob("case_*.json")):
            with self.subTest(path=path.name):
                case = json.loads(path.read_text(encoding="utf-8"))
                self.assertTrue(case.get("case_id"))
                self.assertIn("scripts/roi_image_edit_cli.py process", case.get("command", ""))
                self.assertIn("--instruction", case.get("command", ""))
                self.assertNotIn("--profile", case.get("command", ""))
                expected = case.get("expected_report_fields")
                self.assertIsInstance(expected, list)
                self.assertGreaterEqual(len(expected), 5)
                for item in expected:
                    self.assertIn("path", item)
                    self.assertFalse(str(item["path"]).endswith("image_exists"))
                expected_paths = {str(item.get("path") or "") for item in expected}
                self.assertTrue(
                    any(path.startswith("classification") for path in expected_paths),
                    f"{path.name} must assert classification-driven workflow fields",
                )


if __name__ == "__main__":
    unittest.main()
