from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
