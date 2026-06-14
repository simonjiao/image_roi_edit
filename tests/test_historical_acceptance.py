from __future__ import annotations

import unittest

from roi_image_edit.historical_acceptance import (
    DEFAULT_FALSE_PASS_CASES,
    apply_historical_false_pass_gate,
    historical_false_pass_target,
    historical_target_completion,
    load_false_pass_cases,
    normalize_history_axis,
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
        self.assertEqual(target["stage"], "text_shape")
        self.assertEqual(target["max_steps"], 3)
        self.assertEqual(
            target["axis_keys"],
            ["too_gray", "too_blurry", "too_dark", "stroke_too_bold", "font_mismatch", "shape_unnatural"],
        )

    def test_history_axis_aliases_keep_shape_problems_in_text_shape(self) -> None:
        self.assertEqual(normalize_history_axis("too_bold"), "stroke_too_bold")
        self.assertEqual(normalize_history_axis("font_style_mismatch"), "font_mismatch")
        self.assertEqual(normalize_history_axis("glyph_unnatural"), "shape_unnatural")

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

    def test_historical_false_pass_gate_rejects_vision_closure_without_local_gray_and_blur_evidence(self) -> None:
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
                    {"axis": "too_gray", "closed": True, "basis": "vision says gray is ok"},
                    {"axis": "too_blurry", "closed": True, "basis": "vision says sharpness is ok"},
                ]
            },
        }
        report = {
            "roi_plan": {"source_slot_count": 2, "target_slot_count": 3},
            "strict_visual_metrics": {
                "bands": {
                    "old_lt55_pixels": 233,
                    "lt55_delta": -22,
                    "old_lt90_pixels": 428,
                    "lt90_delta": 324,
                    "old_lt55_share_of_lt165": 0.1215,
                    "new_lt55_share_of_lt165": 0.0873,
                }
            },
            "char_gray_band_metrics": {
                "enabled": True,
                "per_char": [
                    {
                        "index": 0,
                        "source_char": "陈",
                        "target_char": "赵",
                        "old": {"lt55": 157},
                        "delta": {"lt55": -65, "band_55_70": 55, "band_70_90": 25},
                    }
                ],
            },
            "photo_texture_metrics": {
                "edge_laplacian_ratio": 0.6583,
                "old_edge_laplacian_mean": 78.305,
                "params": {"blur": 0.61},
            },
        }

        gated = apply_historical_false_pass_gate(acceptance, target, report=report)

        self.assertFalse(gated["pass"])
        self.assertEqual(gated["final_decision"], "revise")
        self.assertEqual(gated["historical_target_completion"]["missing_axes"], ["too_gray", "too_blurry"])
        self.assertFalse(gated["historical_target_completion"]["local_closure"]["too_gray"]["closed"])
        self.assertFalse(gated["historical_target_completion"]["local_closure"]["too_blurry"]["closed"])

    def test_historical_false_pass_gate_rejects_vision_dark_closure_without_local_deep_gray_evidence(self) -> None:
        target = {
            "active": True,
            "source": "historical_false_pass",
            "stage": "ink_gray_balance",
            "case_ids": ["case"],
            "axes": [{"axis": "too_dark", "stage": "ink_gray_balance"}],
        }
        acceptance = {
            "pass": True,
            "acceptance_level": "pass",
            "final_decision": "deliver",
            "historical_target_closure": {
                "axes": [{"axis": "too_dark", "closed": True, "basis": "vision says darkness is ok"}]
            },
        }
        report = {
            "roi_plan": {"source_slot_count": 2, "target_slot_count": 3},
            "strict_visual_metrics": {
                "bands": {
                    "old_lt55_pixels": 233,
                    "lt55_delta": -2,
                    "old_lt55_share_of_lt165": 0.1215,
                    "new_lt55_share_of_lt165": 0.1191,
                }
            },
            "char_gray_band_metrics": {
                "enabled": True,
                "per_char": [
                    {
                        "index": 1,
                        "source_char": "乙",
                        "target_char": "丁",
                        "old": {"lt55": 76, "lt165": 532},
                        "delta": {"lt55": 20, "band_55_70": 83},
                    }
                ],
            },
        }

        gated = apply_historical_false_pass_gate(acceptance, target, report=report)

        self.assertFalse(gated["pass"])
        self.assertEqual(gated["final_decision"], "revise")
        self.assertEqual(gated["visual_findings"]["darkness"], "too_dark")
        self.assertEqual(gated["historical_target_completion"]["missing_axes"], ["too_dark"])
        self.assertFalse(gated["historical_target_completion"]["local_closure"]["too_dark"]["closed"])

    def test_historical_false_pass_gate_rejects_shape_closure_without_local_text_shape_evidence(self) -> None:
        target = {
            "active": True,
            "source": "historical_false_pass",
            "stage": "text_shape",
            "case_ids": ["case"],
            "axes": [
                {"axis": "stroke_too_bold", "stage": "text_shape"},
                {"axis": "font_mismatch", "stage": "text_shape"},
                {"axis": "shape_unnatural", "stage": "text_shape"},
            ],
        }
        acceptance = {
            "pass": True,
            "acceptance_level": "pass",
            "final_decision": "deliver",
            "historical_target_closure": {
                "axes": [
                    {"axis": "stroke_too_bold", "closed": True, "basis": "vision says stroke is ok"},
                    {"axis": "font_mismatch", "closed": True, "basis": "vision says font is ok"},
                    {"axis": "shape_unnatural", "closed": True, "basis": "vision says shape is natural"},
                ]
            },
        }
        report = {
            "strict_gate": {
                "pass": False,
                "issues": [
                    {"type": "changed_char_alpha_stroke_body_too_bold", "body_area_ratio": 0.81},
                    {"type": "font_render_style_score_ratio", "actual": 1.34, "limit": 1.25},
                ],
            },
            "stroke_body_shape_metrics": {
                "enabled": True,
                "per_char": [
                    {"changed": True, "body_area_ratio": 0.81},
                    {"changed": True, "body_area_ratio": 0.84},
                ],
            },
            "stage_gate": {
                "stage_status": {
                    "text_shape": {
                        "id": "text_shape",
                        "pass": False,
                        "issues": [{"type": "changed_char_alpha_stroke_body_too_bold"}],
                    }
                }
            },
        }

        gated = apply_historical_false_pass_gate(acceptance, target, report=report)

        self.assertFalse(gated["pass"])
        self.assertEqual(gated["final_decision"], "revise")
        self.assertEqual(gated["blocking_stage"], "text_shape")
        self.assertEqual(
            gated["historical_target_completion"]["missing_axes"],
            ["stroke_too_bold", "font_mismatch", "shape_unnatural"],
        )
        self.assertEqual(gated["visual_findings"]["stroke_weight"], "too_bold")
        self.assertEqual(gated["visual_findings"]["font_similarity"], "mismatch")
        self.assertEqual(gated["visual_findings"]["shape"], "unnatural")


if __name__ == "__main__":
    unittest.main()
