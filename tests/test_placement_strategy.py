from __future__ import annotations

import unittest

from roi_image_edit.iterative_pipeline import RenderPlan, TextRun
from roi_image_edit.local_validation import placement_strategy_report
from roi_image_edit.run_artifacts import result_audit_payload
from roi_image_edit.roi_locator import choose_placement_strategy


def run(x1: int, y1: int = 2, x2: int | None = None, y2: int = 12) -> TextRun:
    return TextRun(x1=x1, y1=y1, x2=x2 if x2 is not None else x1 + 8, y2=y2, area=80)


class PlacementStrategyTest(unittest.TestCase):
    def test_strategy_reason_is_declared_for_core_scenarios(self) -> None:
        scenarios = [
            {
                "name": "same_length_cjk_changed",
                "source": "赵芳",
                "target": "陈慧",
                "slots": (run(2), run(14)),
                "slot_report": {"pass": True},
                "draw_mode": "auto",
                "strategy": "center_primary",
                "reason": "same_length_cjk_changed_chars_use_slot_center",
            },
            {
                "name": "longer_name",
                "source": "赵芳",
                "target": "陈小慧",
                "slots": (run(2), run(14)),
                "slot_report": {"pass": True},
                "draw_mode": "center",
                "strategy": "left_anchor_span",
                "reason": "target_text_longer_than_source",
            },
            {
                "name": "shorter_name",
                "source": "赵真真",
                "target": "陈芸",
                "slots": (run(2), run(14), run(26)),
                "slot_report": {"pass": True},
                "draw_mode": "line_chars",
                "strategy": "left_anchor_span",
                "reason": "target_text_shorter_than_source",
            },
            {
                "name": "date_or_age",
                "source": "2024-01-01",
                "target": "2025-02-03",
                "slots": tuple(run(2 + idx * 8, x2=2 + idx * 8 + 5) for idx in range(10)),
                "slot_report": {"pass": True},
                "draw_mode": "auto",
                "strategy": "baseline_numeric",
                "reason": "non_cjk_value_uses_baseline_priority",
            },
            {
                "name": "manual_roi_no_source",
                "source": "",
                "target": "陈芸",
                "slots": (),
                "slot_report": {"pass": True},
                "draw_mode": "auto",
                "strategy": "manual_fallback",
                "reason": "source_text_missing",
            },
        ]
        for item in scenarios:
            with self.subTest(item["name"]):
                strategy, reason = choose_placement_strategy(
                    source_text=item["source"],
                    target_text=item["target"],
                    slots=item["slots"],
                    slot_report=item["slot_report"],
                    draw_mode=item["draw_mode"],
                )
                self.assertEqual(strategy, item["strategy"])
                self.assertEqual(reason, item["reason"])

    def test_placement_report_schema_records_conditions_constraints_errors_and_pass(self) -> None:
        plan = RenderPlan(
            target_text="陈慧",
            source_text="赵芳",
            search_roi=(0, 0, 40, 20),
            target_roi=(2, 2, 26, 14),
            slot_boxes=(run(2), run(14)),
            protected_boxes=(),
            source_reference_box=(2, 2, 26, 14),
            style_reference_box=None,
            style_reference_text=None,
            draw_mode="auto",
            placement_strategy="center_primary",
            placement_strategy_reason="same_length_cjk_changed_chars_use_slot_center",
            slot_quality_report={"pass": True},
        )
        report = placement_strategy_report(
            plan,
            {
                "enabled": True,
                "center_distance_delta": 0.5,
                "candidate_center_y_range": 1.0,
                "per_char": [
                    {"center_dx": -0.5, "center_dy": 0.25},
                    {"center_dx": 0.75, "center_dy": -0.5},
                ],
            },
            [],
            max_char_center_dx=2.0,
            max_char_center_dy=2.5,
            max_char_center_distance_delta=2.0,
            max_replacement_center_y_range=2.0,
        )
        self.assertEqual(report["strategy"], "center_primary")
        self.assertEqual(report["reason"], "same_length_cjk_changed_chars_use_slot_center")
        self.assertTrue(report["pass"])
        self.assertEqual(report["conditions"]["length_change"], "same")
        self.assertEqual(report["conditions"]["slot_count"], 2)
        self.assertEqual(report["constraints"]["max_char_center_dx"], 2.0)
        self.assertEqual(report["actual_errors"]["max_abs_center_dx"], 0.75)
        self.assertEqual(report["actual_errors"]["center_distance_delta"], 0.5)

    def test_result_audit_preserves_placement_strategy_report(self) -> None:
        placement_report = {
            "strategy": "center_primary",
            "reason": "same_length_cjk_changed_chars_use_slot_center",
            "pass": True,
            "conditions": {"length_change": "same", "slot_count": 2},
            "constraints": {"max_char_center_dx": 2.0},
            "actual_errors": {"max_abs_center_dx": 0.75},
            "issues": [],
        }
        audit = result_audit_payload(
            {
                "ok": True,
                "runDir": "output/web/run1",
                "profile": "photo_scan",
                "profileResolution": {"id": "photo_scan"},
                "images": [
                    {
                        "id": "img1",
                        "ok": True,
                        "regions": [
                            {
                                "id": "region_1",
                                "summary": {
                                    "hard_check": {
                                        "placement_strategy_report": placement_report,
                                    }
                                },
                            }
                        ],
                        "candidates": [],
                    }
                ],
            }
        )
        saved = audit["images"][0]["regions"][0]["summary"]["hard_check"]["placement_strategy_report"]
        self.assertEqual(saved["strategy"], "center_primary")
        self.assertEqual(saved["reason"], "same_length_cjk_changed_chars_use_slot_center")
        self.assertTrue(saved["pass"])
        self.assertEqual(saved["conditions"]["length_change"], "same")
        self.assertEqual(saved["constraints"]["max_char_center_dx"], 2.0)
        self.assertEqual(saved["actual_errors"]["max_abs_center_dx"], 0.75)


if __name__ == "__main__":
    unittest.main()
