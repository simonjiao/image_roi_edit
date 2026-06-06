from __future__ import annotations

import json
import unittest

from roi_image_edit.iterative_pipeline import CandidateParams, RenderPlan, TextRun, vision_task_context
from roi_image_edit.prompt_assets import load_prompt
from roi_image_edit.prompt_contracts import (
    ACTIVE_VISION_PROMPTS,
    PROMPT_INPUT_CONTRACTS,
    PROMPT_OUTPUT_FIELD_HANDLING,
    RESERVED_DIAGNOSTIC_PROMPTS,
    prompt_io_contract_report,
)
from roi_image_edit.region_processing import processing_prompt_context
from roi_image_edit.run_artifacts import model_stage_context


def contract_plan() -> RenderPlan:
    return RenderPlan(
        target_text="25岁",
        source_text="24岁",
        search_roi=(10, 20, 90, 42),
        target_roi=(42, 22, 78, 40),
        slot_boxes=(TextRun(42, 22, 70, 40, 120),),
        protected_boxes=((12, 22, 36, 40), (37, 24, 40, 38)),
        source_reference_box=(42, 22, 78, 40),
        style_reference_box=None,
        style_reference_text=None,
        draw_mode="auto",
        text_angle_degrees=1.25,
        field_key="age",
        field_label_text="年龄",
        field_separator_text="：",
        protected_texts=("年龄", "："),
    )


class PromptIoContractsTest(unittest.TestCase):
    def test_every_prompt_output_json_field_has_a_declared_handling_status(self) -> None:
        report = prompt_io_contract_report()

        for prompt_name, prompt_report in report["prompts"].items():
            with self.subTest(prompt=prompt_name):
                self.assertEqual(prompt_report["unregistered_output_fields"], [])
                self.assertEqual(prompt_report["stale_registered_fields"], [])

        self.assertEqual(
            PROMPT_OUTPUT_FIELD_HANDLING["darkness_blur_prompt.txt"]["best_opacity"],
            "reserved_not_called",
        )
        self.assertEqual(
            PROMPT_OUTPUT_FIELD_HANDLING["darkness_blur_prompt.txt"]["best_blur"],
            "reserved_not_called",
        )

    def test_active_and_reserved_prompt_output_contracts_are_not_mixed(self) -> None:
        for prompt_name in ACTIVE_VISION_PROMPTS:
            with self.subTest(active_prompt=prompt_name):
                handling = PROMPT_OUTPUT_FIELD_HANDLING[prompt_name]
                self.assertTrue(handling)
                self.assertFalse(any(value.startswith("reserved_not_called") for value in handling.values()))

        for prompt_name in RESERVED_DIAGNOSTIC_PROMPTS:
            with self.subTest(reserved_prompt=prompt_name):
                handling = PROMPT_OUTPUT_FIELD_HANDLING[prompt_name]
                self.assertTrue(handling)
                self.assertTrue(all(value.startswith("reserved_not_called") for value in handling.values()))

    def test_web_candidate_and_final_prompts_contain_declared_formatted_inputs(self) -> None:
        plan = contract_plan()
        stage_context = {
            "stage_context_by_candidate": {
                "c1": {
                    "blocking_stage": "text_shape",
                    "allowed_patch_keys": ["font_size_delta"],
                    "blocked_patch_keys": ["opacity_delta"],
                }
            },
            "stage_filter_contract": {"policy": "local_stage_filter"},
        }
        hard_report = {
            "pipeline_profile": "photo_scan",
            "stage_context_by_candidate": stage_context["stage_context_by_candidate"],
            "stage_filter_contract": stage_context["stage_filter_contract"],
        }
        candidate_prompt = load_prompt("candidate_rank_prompt.txt").replace(
            "{hard_check_report}",
            json.dumps(hard_report, ensure_ascii=False),
        )
        candidate_prompt += processing_prompt_context(plan, stage_context)

        final_stage_context = model_stage_context(
            {
                "pass": True,
                "pipeline_profile": "photo_scan",
                "font_style_gate": {"issues": [{"type": "font_style_mismatch"}]},
            },
            "photo_scan",
        )
        final_payload = {"task": {"stage_context": final_stage_context}, "hard_check": {"pass": True}}
        final_prompt = load_prompt("final_acceptance_prompt.txt").replace(
            "{final_params}",
            json.dumps({"font_size": 20}, ensure_ascii=False),
        ).replace(
            "{hard_check_report}",
            json.dumps(final_payload, ensure_ascii=False),
        )
        final_prompt += processing_prompt_context(plan, final_stage_context)

        for prompt_name, prompt_text in (
            ("candidate_rank_prompt.txt", candidate_prompt),
            ("final_acceptance_prompt.txt", final_prompt),
        ):
            contract = PROMPT_INPUT_CONTRACTS[prompt_name]
            for group_name, fields in contract.items():
                if group_name == "formatted_payloads":
                    continue
                with self.subTest(prompt=prompt_name, group=group_name):
                    for field in fields:
                        self.assertIn(field, prompt_text)
            self.assertNotIn("{hard_check_report}", prompt_text)
            self.assertNotIn("{final_params}", prompt_text)

    def test_cli_tuning_prompt_contains_declared_formatted_inputs(self) -> None:
        plan = contract_plan()
        params = CandidateParams(
            candidate_id="c1",
            font_name="Test",
            font_path="/tmp/test.ttf",
            font_size=20,
            opacity=0.82,
            blur=0.2,
        )
        report = {
            "pass": True,
            "pipeline_profile": "photo_scan",
            "font_style_gate": {"issues": [{"type": "font_style_mismatch"}]},
        }
        report["stage_context"] = model_stage_context(report, "photo_scan")
        prompt = load_prompt("tuning_prompt.txt").replace(
            "{current_params}",
            json.dumps(params.__dict__, ensure_ascii=False),
        ).replace(
            "{hard_check_report}",
            json.dumps(report, ensure_ascii=False),
        )
        prompt += vision_task_context(plan)

        contract = PROMPT_INPUT_CONTRACTS["tuning_prompt.txt"]
        for group_name, fields in contract.items():
            if group_name == "formatted_payloads":
                continue
            with self.subTest(group=group_name):
                for field in fields:
                    self.assertIn(field, prompt)
        self.assertNotIn("{current_params}", prompt)
        self.assertNotIn("{hard_check_report}", prompt)


if __name__ == "__main__":
    unittest.main()
