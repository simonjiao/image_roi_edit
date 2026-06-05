from __future__ import annotations

import unittest

from roi_image_edit.stage_concerns import (
    DIAGNOSTIC_CONCERN_MAPPINGS,
    diagnostic_concern_mapping_report,
    mapping_for_concern,
)
from roi_image_edit.stage_policy import STAGE_ORDER
from roi_image_edit.stages import stage_gate_for_report


class StageConcernMappingTest(unittest.TestCase):
    def test_old_diagnostic_concerns_are_reported_as_mappings_not_public_stages(self) -> None:
        report = diagnostic_concern_mapping_report()
        self.assertEqual(
            tuple(item["concern_id"] for item in report),
            (
                "slot_alignment",
                "font_structure",
                "pose_geometry",
                "stroke_body",
                "tone_gray",
                "edge_quality",
                "photo_texture",
            ),
        )
        public_stages = set(STAGE_ORDER)
        for item in report:
            self.assertTrue(item["current_stages_valid"], item)
            self.assertTrue(set(item["current_stages"]) <= public_stages, item)
            self.assertTrue(item["optimization_steps"], item)
            self.assertTrue(item["report_fields"], item)
            self.assertEqual(item["optimization_step_scope"], "within_stage_not_public_stage")
            self.assertEqual(tuple(item["public_stage_ids"]), STAGE_ORDER)

    def test_stage_gate_exposes_diagnostic_concern_mapping_as_stage_evidence(self) -> None:
        gate = stage_gate_for_report(
            {"pass": False, "pipeline_profile": "photo_scan", "issues": [{"type": "roi_outside"}]},
            "photo_scan",
        )
        mapping = gate["diagnostic_concern_mapping"]
        self.assertEqual(len(mapping), len(DIAGNOSTIC_CONCERN_MAPPINGS))
        slot_alignment = next(item for item in mapping if item["concern_id"] == "slot_alignment")
        self.assertEqual(slot_alignment["current_stages"], ("hard_boundary", "text_shape"))
        self.assertIn("slot_quality_gate", slot_alignment["optimization_steps"])
        self.assertIn("slot_quality_report", slot_alignment["report_fields"])

    def test_photo_texture_concern_maps_to_named_within_stage_steps(self) -> None:
        photo = mapping_for_concern("photo_texture")
        self.assertEqual(photo["current_stages"], ("photo_texture",))
        self.assertEqual(
            photo["optimization_steps"],
            (
                "blur_match",
                "edge_breakup_match",
                "noise_texture_match",
                "jpeg_texture_match",
                "residual_retexture",
            ),
        )
        self.assertIn("photo_texture_metrics", photo["report_fields"])
        self.assertEqual(photo["optimization_step_scope"], "within_stage_not_public_stage")

    def test_font_stroke_tone_and_edge_concerns_map_to_named_within_stage_steps(self) -> None:
        font = mapping_for_concern("font_structure")
        self.assertEqual(font["current_stages"], ("text_shape",))
        self.assertEqual(font["optimization_steps"], ("font_style_search", "font_size_search"))

        stroke = mapping_for_concern("stroke_body")
        self.assertEqual(stroke["current_stages"], ("text_shape",))
        self.assertEqual(stroke["optimization_steps"], ("stroke_body_search",))

        tone = mapping_for_concern("tone_gray")
        self.assertEqual(tone["current_stages"], ("ink_gray_balance",))
        self.assertEqual(
            tone["optimization_steps"],
            ("core_black_search", "mid_gray_body_search", "opacity_search"),
        )

        edge = mapping_for_concern("edge_quality")
        self.assertEqual(edge["current_stages"], ("ink_gray_balance", "photo_texture"))
        self.assertEqual(edge["optimization_steps"], ("outer_gray_control", "edge_breakup_match"))


if __name__ == "__main__":
    unittest.main()
