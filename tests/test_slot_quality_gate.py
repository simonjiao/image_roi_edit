from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from roi_image_edit.iterative_pipeline import RenderPlan, TextRun
from roi_image_edit.processing_service import process_region


class SlotQualityGateTest(unittest.TestCase):
    def test_failed_slot_quality_stops_before_candidate_generation(self) -> None:
        original = Image.new("RGB", (48, 28), (210, 210, 210))
        plan = RenderPlan(
            target_text="女",
            source_text="男",
            search_roi=(4, 4, 34, 24),
            target_roi=(8, 8, 20, 20),
            slot_boxes=(TextRun(8, 8, 20, 18, 80),),
            protected_boxes=(),
            source_reference_box=(8, 8, 20, 20),
            style_reference_box=None,
            style_reference_text=None,
            draw_mode="auto",
            slot_quality_report={
                "pass": False,
                "issues": [{"type": "slot_bottom_overflow", "index": 0}],
                "per_slot": [
                    {
                        "index": 0,
                        "coverage": {
                            "bottom_coverage": 0.78,
                            "bottom_dark_pixels": 8,
                        },
                    }
                ],
            },
        )
        events: list[tuple[str, dict]] = []
        with tempfile.TemporaryDirectory() as tmp:
            with patch("roi_image_edit.processing_service.build_region_plan", return_value=plan):
                result, display, candidates, summary, accepted = process_region(
                    original,
                    (4, 4, 34, 24),
                    source_text="男",
                    target_text="女",
                    run_dir=Path(tmp),
                    region_id="r1",
                    vision_client=object(),
                    prompts=("master", "rank", "final"),
                    max_candidates=20,
                    vision_candidate_limit=3,
                    max_revision_rounds=2,
                    pipeline_profile="photo_scan",
                    progress=lambda event, payload: events.append((event, payload)),
                )
            self.assertEqual(result.size, original.size)
            self.assertEqual(display.size, original.size)
            self.assertEqual(candidates, [])
            self.assertFalse(accepted)
            self.assertFalse(summary["accepted"])
            self.assertFalse(summary["applied"])
            self.assertEqual(summary["vision"]["reason"], "slot_quality_failed_before_candidate_generation")
            self.assertEqual(summary["trace"]["revision_round_count"], 0)
            self.assertEqual(summary["trace"]["final_blocking_stage"], "hard_boundary")
            self.assertEqual(summary["hard_check"]["stage_gate"]["blocking_stage"], "hard_boundary")
            self.assertEqual(summary["hard_check"]["stage_gate"]["stage_status"]["hard_boundary"]["reason"], "slot_quality_failed")
            self.assertEqual(events[0][0], "pre_candidate_gate_failed")
            self.assertEqual(events[0][1]["candidate_count"], 0)
            self.assertEqual(events[0][1]["failed_gate"], "slot_quality_gate")
            gate_report = events[0][1]["pre_candidate_gate_report"]
            self.assertEqual(
                gate_report["gate_order"],
                ["orientation_check", "field_roi_selection", "slot_quality_gate", "protected_text_guard"],
            )
            self.assertFalse(gate_report["pass"])
            self.assertEqual(gate_report["statuses"]["slot_quality_gate"]["pass"], False)
            self.assertEqual(events[1][0], "slot_quality_failed")
            self.assertEqual(events[1][1]["slot_quality_report"]["issues"][0]["type"], "slot_bottom_overflow")
            self.assertEqual(summary["trace"]["pre_candidate_gate_report"]["failed_gate"], "slot_quality_gate")
            region_dir = Path(tmp) / "regions" / "r1"
            self.assertTrue((region_dir / "slot_quality_rejected_compare.png").exists())
            self.assertTrue((region_dir / "slot_quality_report.json").exists())
            self.assertTrue((region_dir / "pre_candidate_gate_report.json").exists())
            self.assertEqual(summary["artifacts"]["slot_quality_report"], str(region_dir / "slot_quality_report.json"))
            self.assertEqual(
                summary["artifacts"]["pre_candidate_gate_report"],
                str(region_dir / "pre_candidate_gate_report.json"),
            )

    def test_protected_text_guard_failure_stops_before_candidate_generation(self) -> None:
        original = Image.new("RGB", (48, 28), (210, 210, 210))
        plan = RenderPlan(
            target_text="女",
            source_text="男",
            search_roi=(4, 4, 34, 24),
            target_roi=(8, 8, 20, 20),
            slot_boxes=(TextRun(8, 8, 20, 18, 80),),
            protected_boxes=((14, 8, 28, 20),),
            source_reference_box=(8, 8, 20, 20),
            style_reference_box=None,
            style_reference_text=None,
            draw_mode="auto",
            slot_quality_report={
                "pass": False,
                "issues": [{"type": "slot_overlaps_protected_text", "index": 0}],
                "per_slot": [
                    {
                        "index": 0,
                        "overlap": {
                            "protected_overlap_pixels": 40,
                        },
                    }
                ],
            },
        )
        events: list[tuple[str, dict]] = []
        with tempfile.TemporaryDirectory() as tmp:
            with patch("roi_image_edit.processing_service.build_region_plan", return_value=plan):
                _result, _display, candidates, summary, accepted = process_region(
                    original,
                    (4, 4, 34, 24),
                    source_text="男",
                    target_text="女",
                    run_dir=Path(tmp),
                    region_id="r1",
                    vision_client=object(),
                    prompts=("master", "rank", "final"),
                    max_candidates=20,
                    vision_candidate_limit=3,
                    max_revision_rounds=2,
                    pipeline_profile="photo_scan",
                    progress=lambda event, payload: events.append((event, payload)),
                )
        self.assertEqual(candidates, [])
        self.assertFalse(accepted)
        self.assertFalse(summary["accepted"])
        self.assertEqual(events[0][0], "pre_candidate_gate_failed")
        self.assertEqual(events[0][1]["candidate_count"], 0)
        self.assertEqual(events[0][1]["failed_gate"], "protected_text_guard")
        gate_report = events[0][1]["pre_candidate_gate_report"]
        self.assertEqual(gate_report["statuses"]["slot_quality_gate"]["pass"], True)
        self.assertEqual(gate_report["statuses"]["protected_text_guard"]["pass"], False)
        self.assertEqual(summary["trace"]["pre_candidate_gate_report"]["failed_gate"], "protected_text_guard")


if __name__ == "__main__":
    unittest.main()
