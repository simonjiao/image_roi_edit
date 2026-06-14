from __future__ import annotations

import unittest

from roi_image_edit.historical_acceptance import (
    DEFAULT_FALSE_PASS_CASES,
    apply_historical_false_pass_gate,
    historical_false_pass_target,
    historical_target_completion,
    load_false_pass_cases,
    validate_false_pass_case,
)
from roi_image_edit.iterative_pipeline import RenderPlan, TextRun


def cjk_longer_plan() -> RenderPlan:
    return RenderPlan(
        target_text="甲乙丙",
        source_text="甲乙",
        search_roi=(10, 10, 120, 42),
        target_roi=(42, 12, 112, 40),
        slot_boxes=(
            TextRun(42, 12, 64, 40, 100),
            TextRun(66, 12, 88, 40, 100),
        ),
        protected_boxes=(),
        source_reference_box=None,
        style_reference_box=None,
        style_reference_text=None,
        draw_mode="line_chars",
        field_key="name",
        field_label_text="",
        field_separator_text="",
        protected_texts=(),
    )


class HistoricalAcceptanceTest(unittest.TestCase):
    def test_false_pass_case_library_is_versioned_schema_and_not_runtime_database(self) -> None:
        cases = load_false_pass_cases()

        self.assertGreaterEqual(len(cases), 1)
        self.assertTrue(DEFAULT_FALSE_PASS_CASES.exists())
        self.assertEqual(DEFAULT_FALSE_PASS_CASES.suffix, ".jsonl")
        for case in cases:
            with self.subTest(case=case["case_id"]):
                self.assertEqual(validate_false_pass_case(case), [])
                self.assertEqual(case["generalization_constraints"]["forbid_specific_text"], True)
                self.assertRegex(case["artifact_hashes"]["input_sha256"], r"^[0-9a-f]{64}$")
                self.assertRegex(case["artifact_hashes"]["output_sha256"], r"^[0-9a-f]{64}$")

        raw = DEFAULT_FALSE_PASS_CASES.read_text(encoding="utf-8")
        forbidden_fragments = ("赵真真", "陈芸", "5681779868443", "5621779868438")
        for fragment in forbidden_fragments:
            self.assertNotIn(fragment, raw)

    def test_historical_false_pass_case_matches_generic_cjk_two_to_three_workflow(self) -> None:
        report = {
            "classification": {"class_key": "photo_document.form_field_value_replace.cjk"},
            "roi_plan": {"source_slot_count": 2, "target_slot_count": 3},
            "stage_gate": {"blocking_stage": None},
        }

        target = historical_false_pass_target(report, cjk_longer_plan())

        self.assertTrue(target["active"])
        self.assertEqual(target["source"], "historical_false_pass")
        self.assertEqual(target["stage"], "ink_gray_balance")
        self.assertEqual(target["max_steps"], 3)
        self.assertEqual(target["axis_keys"], ["too_gray", "too_blurry"])

    def test_historical_false_pass_gate_blocks_deliver_until_axes_are_closed(self) -> None:
        target = {
            "active": True,
            "source": "historical_false_pass",
            "stage": "ink_gray_balance",
            "case_ids": ["case"],
            "axes": [
                {"axis": "too_gray", "stage": "ink_gray_balance"},
                {"axis": "too_blurry", "stage": "photo_texture"},
            ],
            "axis_keys": ["too_gray", "too_blurry"],
        }
        acceptance = {
            "pass": True,
            "acceptance_level": "pass",
            "final_decision": "deliver",
            "visual_findings": {"darkness": "ok", "sharpness": "ok", "background": "ok"},
        }

        gated = apply_historical_false_pass_gate(acceptance, target)

        self.assertFalse(gated["pass"])
        self.assertEqual(gated["final_decision"], "revise")
        self.assertEqual(gated["blocking_stage"], "ink_gray_balance")
        self.assertEqual(gated["visual_findings"]["darkness"], "too_light")
        self.assertEqual(gated["visual_findings"]["sharpness"], "too_blurry")
        self.assertEqual(gated["historical_target_completion"]["missing_axes"], ["too_gray", "too_blurry"])

    def test_historical_false_pass_gate_allows_deliver_when_all_axes_are_closed(self) -> None:
        target = {
            "active": True,
            "source": "historical_false_pass",
            "stage": "ink_gray_balance",
            "case_ids": ["case"],
            "axes": [
                {"axis": "too_gray", "stage": "ink_gray_balance"},
                {"axis": "too_blurry", "stage": "photo_texture"},
            ],
        }
        acceptance = {
            "pass": True,
            "acceptance_level": "pass",
            "final_decision": "deliver",
            "historical_target_closure": {
                "axes": [
                    {"axis": "too_gray", "closed": True, "basis": "gray now matches row reference"},
                    {"axis": "too_blurry", "closed": True, "basis": "edge softness now matches scan"},
                ]
            },
        }

        completion = historical_target_completion(target, acceptance)
        gated = apply_historical_false_pass_gate(acceptance, target)

        self.assertTrue(completion["complete"])
        self.assertTrue(gated["pass"])
        self.assertEqual(gated["final_decision"], "deliver")


if __name__ == "__main__":
    unittest.main()
