from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from roi_image_edit.iterative_pipeline import CandidateParams, RenderPlan, TextRun
from roi_image_edit.region_processing import save_stage_candidate_evidence


def params() -> CandidateParams:
    return CandidateParams(
        candidate_id="c001",
        font_name="TestFont",
        font_path="/tmp/test.ttf",
        font_size=18,
        opacity=0.8,
        blur=0.2,
    )


def plan() -> RenderPlan:
    return RenderPlan(
        target_text="丙丁戊",
        source_text="甲乙",
        search_roi=(0, 0, 80, 40),
        target_roi=(20, 8, 70, 34),
        slot_boxes=(TextRun(22, 10, 34, 28, 120), TextRun(36, 10, 48, 28, 120)),
        protected_boxes=((4, 10, 18, 28),),
        source_reference_box=(22, 10, 48, 28),
        style_reference_box=None,
        style_reference_text=None,
        draw_mode="center",
        placement_strategy="left_anchor_span",
        placement_strategy_reason="test",
        slot_quality_report={"pass": True},
    )


class StageEvidenceTest(unittest.TestCase):
    def test_stage_evidence_preserves_row_baseline_and_placement_reports(self) -> None:
        row_baseline = {
            "enabled": True,
            "basis": "source_slots_primary_same_row_context",
            "candidate_box": [22, 14, 66, 32],
            "source_slot_boxes": [[22, 10, 34, 28], [36, 10, 48, 28]],
            "same_row_protected_boxes": [[4, 10, 18, 28]],
            "reference_center_y": 19.0,
            "center_delta_y": 4.0,
            "issues": [{"type": "row_baseline_center_y_too_low"}],
        }
        placement = {
            "enabled": True,
            "placement_strategy": "left_anchor_span",
            "actual_errors": {"baseline_dy": 4.0},
            "pass": False,
        }
        report = {
            "pass": True,
            "placement_strategy": "left_anchor_span",
            "placement_strategy_reason": "test",
            "stage_gate": {
                "pass": False,
                "blocking_stage": "text_shape",
                "blocking_stage_reason": "row_baseline_center_y_too_low",
                "stages": [
                    {
                        "id": "text_shape",
                        "pass": False,
                        "issues": [{"type": "row_baseline_center_y_too_low"}],
                    }
                ],
            },
            "strict_gate": {
                "pass": False,
                "issues": [{"type": "row_baseline_center_y_too_low"}],
            },
            "row_baseline_metrics": row_baseline,
            "placement_strategy_report": placement,
            "shape_change_report": {"enabled": False},
        }
        image = Image.new("RGB", (80, 40), (200, 200, 200))

        with tempfile.TemporaryDirectory() as tmp:
            evidence = save_stage_candidate_evidence(
                image,
                [(params(), image, report, 12.0)],
                plan(),
                Path(tmp),
                pipeline_profile="photo_scan",
            )
            report_path = Path(evidence["stages"]["text_shape"]["report_path"])
            saved = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(saved["row_baseline_metrics"], row_baseline)
        self.assertEqual(saved["placement_strategy_report"], placement)


if __name__ == "__main__":
    unittest.main()
