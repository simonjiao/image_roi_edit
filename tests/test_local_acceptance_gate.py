from __future__ import annotations

import unittest
from unittest.mock import patch

from roi_image_edit.local_validation import apply_local_acceptance_gate


class LocalAcceptanceGateTest(unittest.TestCase):
    def test_local_blocking_stage_overrides_visual_deliver(self) -> None:
        acceptance = {
            "pass": True,
            "acceptance_level": "pass",
            "final_decision": "deliver",
            "visual_findings": {"font_similarity": "ok", "baseline": "ok"},
            "reason": "model accepted candidate",
        }
        report = {
            "pass": True,
            "pipeline_profile": "photo_scan",
            "strict_gate": {
                "issues": [
                    {
                        "type": "font_style_score_too_high",
                        "actual": 1.4,
                        "limit": 1.25,
                    }
                ]
            },
        }

        gated = apply_local_acceptance_gate(acceptance, report)

        self.assertFalse(gated["pass"])
        self.assertEqual(gated["acceptance_level"], "marginal")
        self.assertEqual(gated["final_decision"], "revise")
        self.assertEqual(gated["stage_gate"]["blocking_stage"], "text_shape")
        self.assertEqual(gated["visual_findings"]["font_similarity"], "ok")
        self.assertIn("local_shape_stage_issues", gated)
        self.assertIn("本地形态阶段", gated["reason"])

    def test_acceptance_is_preserved_when_no_local_stage_blocks(self) -> None:
        acceptance = {
            "pass": True,
            "acceptance_level": "pass",
            "final_decision": "deliver",
            "visual_findings": {"font_similarity": "ok"},
        }
        report = {"pass": True, "pipeline_profile": "photo_scan"}

        self.assertEqual(apply_local_acceptance_gate(acceptance, report), acceptance)

    def test_visual_shape_arbitration_advances_text_shape_to_ink(self) -> None:
        acceptance = {
            "pass": False,
            "acceptance_level": "marginal",
            "final_decision": "revise",
            "blocking_stage": "text_shape",
            "stage_assessment": {
                "blocking_stage_exists": True,
                "current_blocking_stage": "text_shape",
                "suggestion_target_stage": "text_shape",
                "basis": "model saw shape as usable",
            },
            "visual_findings": {
                "char_positions": "ok",
                "spacing": "ok",
                "baseline": "ok",
                "font_similarity": "ok",
                "size": "ok",
                "stroke_weight": "ok",
                "darkness": "too_light",
                "sharpness": "ok",
                "background": "ok",
            },
            "reason": "shape can proceed but ink is light",
        }
        report = {
            "pass": True,
            "pipeline_profile": "photo_scan",
            "strict_gate": {
                "pass": False,
                "issues": [
                    {"type": "font_family_style_score_ratio", "actual": 1.31, "limit": 1.25},
                    {"type": "changed_char_stroke_body_too_small", "lt165_delta": -8.0},
                ],
            },
        }

        with patch(
            "roi_image_edit.local_validation.local_ink_balance_issues",
            return_value=[{"type": "changed_char_core_too_gray"}],
        ):
            gated = apply_local_acceptance_gate(acceptance, report)

        self.assertFalse(gated["pass"])
        self.assertEqual(gated["final_decision"], "revise")
        self.assertEqual(gated["blocking_stage"], "ink_gray_balance")
        self.assertEqual(gated["stage_gate"]["blocking_stage"], "ink_gray_balance")
        self.assertTrue(gated["local_shape_arbitration"]["active"])
        self.assertTrue(gated["stage_gate"]["stage_status"]["text_shape"]["pass_with_deferred"])
        self.assertEqual(gated["stage_assessment"]["suggestion_target_stage"], "ink_gray_balance")
        self.assertIn("Vision 形态仲裁", gated["reason"])

    def test_visual_shape_arbitration_requires_shape_findings_ok(self) -> None:
        acceptance = {
            "pass": False,
            "acceptance_level": "marginal",
            "final_decision": "revise",
            "visual_findings": {
                "char_positions": "ok",
                "spacing": "ok",
                "baseline": "ok",
                "font_similarity": "ok",
                "size": "ok",
                "stroke_weight": "too_bold",
                "darkness": "too_light",
            },
        }
        report = {
            "pass": True,
            "pipeline_profile": "photo_scan",
            "strict_gate": {
                "pass": False,
                "issues": [{"type": "font_family_style_score_ratio", "actual": 1.31, "limit": 1.25}],
            },
        }

        with patch(
            "roi_image_edit.local_validation.local_ink_balance_issues",
            return_value=[{"type": "changed_char_core_too_gray"}],
        ):
            gated = apply_local_acceptance_gate(acceptance, report)

        self.assertEqual(gated["stage_gate"]["blocking_stage"], "text_shape")
        self.assertNotIn("local_shape_arbitration", gated)

    def test_visual_shape_arbitration_does_not_defer_hard_stroke_weight_gap(self) -> None:
        acceptance = {
            "pass": False,
            "acceptance_level": "marginal",
            "final_decision": "revise",
            "visual_findings": {
                "char_positions": "ok",
                "spacing": "ok",
                "baseline": "ok",
                "font_similarity": "ok",
                "size": "ok",
                "stroke_weight": "ok",
                "darkness": "too_light",
            },
        }
        report = {
            "pass": True,
            "pipeline_profile": "photo_scan",
            "strict_gate": {
                "pass": False,
                "issues": [
                    {
                        "type": "changed_char_alpha_stroke_body_too_thin",
                        "body_area_ratio": 0.494,
                        "limit": 0.58,
                    }
                ],
            },
        }

        with patch(
            "roi_image_edit.local_validation.local_ink_balance_issues",
            return_value=[{"type": "changed_char_core_too_gray"}],
        ):
            gated = apply_local_acceptance_gate(acceptance, report)

        self.assertEqual(gated["stage_gate"]["blocking_stage"], "text_shape")
        self.assertNotIn("local_shape_arbitration", gated)


if __name__ == "__main__":
    unittest.main()
