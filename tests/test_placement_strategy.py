from __future__ import annotations

import unittest

from roi_image_edit.iterative_pipeline import char_alignment_issues, RenderPlan, TextRun
from roi_image_edit.placement_strategy import choose_placement_strategy, placement_strategy_report
from roi_image_edit.run_artifacts import result_audit_payload


def run(x1: int, y1: int = 2, x2: int | None = None, y2: int = 12) -> TextRun:
    return TextRun(x1=x1, y1=y1, x2=x2 if x2 is not None else x1 + 8, y2=y2, area=80)


def plan_for(
    *,
    source: str,
    target: str,
    slots: tuple[TextRun, ...],
    strategy: str,
    reason: str,
    slot_report: dict,
    draw_mode: str = "auto",
) -> RenderPlan:
    return RenderPlan(
        target_text=target,
        source_text=source,
        search_roi=(0, 0, 80, 24),
        target_roi=(2, 2, 54, 16),
        slot_boxes=slots,
        protected_boxes=(),
        source_reference_box=(2, 2, 54, 16),
        style_reference_box=None,
        style_reference_text=None,
        draw_mode=draw_mode,
        placement_strategy=strategy,
        placement_strategy_reason=reason,
        slot_quality_report=slot_report,
    )


def strategy_report(plan: RenderPlan, metrics: dict) -> dict:
    return placement_strategy_report(
        plan,
        metrics,
        [],
        max_char_center_dx=2.0,
        max_char_center_dy=2.5,
        max_char_center_distance_delta=2.0,
        max_replacement_center_y_range=2.0,
    )


class PlacementStrategyTest(unittest.TestCase):
    def test_char_height_allows_subpixel_bbox_quantization_slack(self) -> None:
        near_threshold = char_alignment_issues(
            {
                "enabled": True,
                "per_char": [
                    {
                        "index": 0,
                        "char": "慧",
                        "slot_box": [199, 358, 219, 380],
                        "candidate_box": [200, 360, 219, 379],
                    }
                ],
            },
            max_char_center_dx=2.0,
            max_char_center_distance_delta=2.0,
        )
        clearly_small = char_alignment_issues(
            {
                "enabled": True,
                "per_char": [
                    {
                        "index": 0,
                        "char": "慧",
                        "slot_box": [199, 358, 219, 380],
                        "candidate_box": [200, 361, 219, 378],
                    }
                ],
            },
            max_char_center_dx=2.0,
            max_char_center_distance_delta=2.0,
        )

        self.assertEqual(near_threshold, [])
        self.assertEqual(clearly_small[0]["type"], "char_height_too_small")
        self.assertEqual(clearly_small[0]["tolerance_px"], 0.66)

    def test_strategy_reason_is_declared_for_core_scenarios(self) -> None:
        scenarios = [
            {
                "name": "same_length_cjk_small_shape_change",
                "source": "日月",
                "target": "曰朋",
                "slots": (run(2), run(14)),
                "slot_report": {"pass": True, "shape_change_report": {"shape_change_large": False}},
                "draw_mode": "auto",
                "strategy": "top_left_anchor",
                "reason": "same_length_cjk_small_shape_change_uses_top_left_anchor",
            },
            {
                "name": "same_length_cjk_large_shape_change",
                "source": "赵芳",
                "target": "陈慧",
                "slots": (run(2), run(14)),
                "slot_report": {"pass": True, "shape_change_report": {"shape_change_large": True}},
                "draw_mode": "auto",
                "strategy": "center_primary",
                "reason": "same_length_cjk_large_shape_change_uses_slot_center",
            },
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
                "draw_mode": "line_chars",
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
        self.assertEqual(report["strategy_contract"]["anchor_priority"], "slot_center")

    def test_classification_driven_strategy_report_matrix_records_workflow_schema(self) -> None:
        scenarios = [
            {
                "name": "same_cjk",
                "source": "日月",
                "target": "曰朋",
                "slots": (run(2), run(14)),
                "draw_mode": "auto",
                "classification": {
                    "class_key": "photo_document.form_field_value_replace.cjk",
                    "roi_policy": "auto",
                    "internal_profile": "photo_scan",
                    "profile_source": "classification",
                },
                "expected_reason": "same_length_cjk_changed_chars_use_slot_center",
            },
            {
                "name": "longer_cjk",
                "source": "赵芳",
                "target": "陈小慧",
                "slots": (run(2), run(14)),
                "draw_mode": "line_chars",
                "classification": {
                    "class_key": "photo_document.form_field_value_replace.cjk",
                    "roi_policy": "auto",
                    "internal_profile": "photo_scan",
                    "profile_source": "classification",
                },
                "expected_reason": "target_text_longer_than_source",
                "expanded_edit_roi": [2, 2, 70, 16],
            },
            {
                "name": "shorter_cjk",
                "source": "赵真真",
                "target": "陈芸",
                "slots": (run(2), run(14), run(26)),
                "draw_mode": "line_chars",
                "classification": {
                    "class_key": "photo_document.form_field_value_replace.cjk",
                    "roi_policy": "auto",
                    "internal_profile": "photo_scan",
                    "profile_source": "classification",
                },
                "expected_reason": "target_text_shorter_than_source",
            },
            {
                "name": "date_number",
                "source": "2024-01-01",
                "target": "2025-02-03",
                "slots": tuple(run(2 + idx * 8, x2=2 + idx * 8 + 5) for idx in range(10)),
                "draw_mode": "auto",
                "classification": {
                    "class_key": "clean_digital.numeric_or_date_replace",
                    "roi_policy": "auto",
                    "internal_profile": "clean_digital",
                    "profile_source": "classification",
                },
                "expected_reason": "non_cjk_value_uses_baseline_priority",
            },
            {
                "name": "manual_exact",
                "source": "",
                "target": "陈芸",
                "slots": (),
                "draw_mode": "auto",
                "classification": {
                    "class_key": "photo_document.inline_text_replace.cjk",
                    "roi_policy": "manual_exact",
                    "internal_profile": "manual_roi_quick",
                    "profile_source": "classification",
                },
                "expected_reason": "source_text_missing",
            },
            {
                "name": "manual_anchor",
                "source": "甲乙",
                "target": "丙丁",
                "slots": (run(20), run(32)),
                "draw_mode": "auto",
                "classification": {
                    "class_key": "photo_document.form_field_value_replace.cjk",
                    "roi_policy": "manual_anchor",
                    "internal_profile": "photo_scan",
                    "profile_source": "classification",
                },
                "expected_reason": "same_length_cjk_changed_chars_use_slot_center",
            },
        ]
        for item in scenarios:
            with self.subTest(item["name"]):
                slot_report = {
                    "pass": True,
                    "classification": item["classification"],
                    "class_key": item["classification"]["class_key"],
                    "roi_policy": item["classification"]["roi_policy"],
                    "internal_profile": item["classification"]["internal_profile"],
                    "profile_source": "classification",
                    "roi_plan": {
                        "search_roi": [0, 0, 80, 24],
                        "edit_roi": [2, 2, 54, 16],
                        "expanded_edit_roi": item.get("expanded_edit_roi"),
                    },
                    "length_change_report": {
                        "expanded_edit_roi": item.get("expanded_edit_roi"),
                        "right_boundary": {
                            "pass": True,
                            "diagnostic_only": True,
                            "space_sufficient": item.get("expanded_edit_roi") is None,
                        },
                    },
                }
                strategy, reason = choose_placement_strategy(
                    source_text=item["source"],
                    target_text=item["target"],
                    slots=item["slots"],
                    slot_report=slot_report,
                    draw_mode=item["draw_mode"],
                )
                plan = plan_for(
                    source=item["source"],
                    target=item["target"],
                    slots=item["slots"],
                    strategy=strategy,
                    reason=reason,
                    slot_report=slot_report,
                    draw_mode=item["draw_mode"],
                )
                report = strategy_report(plan, {"enabled": True})
                self.assertEqual(report["reason"], item["expected_reason"])
                self.assertEqual(report["classification"]["class_key"], item["classification"]["class_key"])
                self.assertEqual(report["class_key"], item["classification"]["class_key"])
                self.assertEqual(report["roi_policy"], item["classification"]["roi_policy"])
                self.assertEqual(report["internal_profile"], item["classification"]["internal_profile"])
                self.assertEqual(report["profile_source"], "classification")
                self.assertEqual(report["expanded_edit_roi"], item.get("expanded_edit_roi"))
                self.assertTrue(report["pass"])

    def test_same_length_small_cjk_uses_top_left_with_center_spacing_and_baseline_constraints(self) -> None:
        strategy, reason = choose_placement_strategy(
            source_text="日月",
            target_text="曰朋",
            slots=(run(2), run(14)),
            slot_report={"pass": True, "shape_change_report": {"shape_change_large": False}},
            draw_mode="auto",
        )
        plan = plan_for(
            source="日月",
            target="曰朋",
            slots=(run(2), run(14)),
            strategy=strategy,
            reason=reason,
            slot_report={"pass": True, "shape_change_report": {"shape_change_large": False}},
        )
        report = strategy_report(
            plan,
            {
                "enabled": True,
                "per_char": [{"center_dx": 0.5, "center_dy": 0.25}, {"center_dx": -0.25, "center_dy": 0.5}],
                "char_spacing_delta": 0.5,
                "baseline_dy": 0.25,
            },
        )
        self.assertEqual(report["strategy"], "top_left_anchor")
        self.assertEqual(report["conditions"]["shape_change_large"], False)
        self.assertEqual(report["strategy_contract"]["anchor_priority"], "top_left")
        self.assertTrue(report["strategy_contract"]["baseline_checked"])
        self.assertTrue(report["strategy_contract"]["char_spacing_checked"])
        self.assertEqual(report["constraints"]["max_char_spacing_delta"], 2.0)
        self.assertEqual(report["actual_errors"]["char_spacing_delta"], 0.5)
        self.assertEqual(report["actual_errors"]["baseline_dy"], 0.25)
        self.assertTrue(report["pass"])

    def test_same_length_large_cjk_uses_center_with_left_baseline_and_spacing_constraints(self) -> None:
        strategy, reason = choose_placement_strategy(
            source_text="赵芳",
            target_text="陈慧",
            slots=(run(2), run(14)),
            slot_report={"pass": True, "shape_change_report": {"shape_change_large": True}},
            draw_mode="auto",
        )
        plan = plan_for(
            source="赵芳",
            target="陈慧",
            slots=(run(2), run(14)),
            strategy=strategy,
            reason=reason,
            slot_report={"pass": True, "shape_change_report": {"shape_change_large": True}},
        )
        report = strategy_report(
            plan,
            {
                "enabled": True,
                "per_char": [{"center_dx": 0.75, "center_dy": 0.5}, {"center_dx": -0.5, "center_dy": 0.25}],
                "left_boundary_dx": 0.5,
                "char_spacing_delta": 0.75,
                "baseline_dy": 0.5,
            },
        )
        self.assertEqual(report["strategy"], "center_primary")
        self.assertEqual(report["conditions"]["shape_change_large"], True)
        self.assertEqual(report["strategy_contract"]["anchor_priority"], "slot_center")
        self.assertEqual(report["constraints"]["max_left_boundary_dx"], 2.0)
        self.assertEqual(report["actual_errors"]["left_boundary_dx"], 0.5)
        self.assertEqual(report["actual_errors"]["baseline_dy"], 0.5)
        self.assertEqual(report["actual_errors"]["char_spacing_delta"], 0.75)
        self.assertTrue(report["pass"])

    def test_shorter_replacement_uses_old_span_and_requires_extra_slot_cleanup(self) -> None:
        strategy, reason = choose_placement_strategy(
            source_text="赵真真",
            target_text="陈芸",
            slots=(run(2), run(14), run(26)),
            slot_report={
                "pass": True,
                "length_change_report": {"cleanup_mask_report": {"enabled": True}},
            },
            draw_mode="line_chars",
        )
        plan = plan_for(
            source="赵真真",
            target="陈芸",
            slots=(run(2), run(14), run(26)),
            strategy=strategy,
            reason=reason,
            slot_report={
                "pass": True,
                "length_change_report": {"cleanup_mask_report": {"enabled": True}},
            },
            draw_mode="line_chars",
        )
        report = strategy_report(
            plan,
            {
                "enabled": True,
                "left_boundary_dx": 0.25,
                "span_width_delta": 1.0,
                "char_spacing_delta": 0.5,
                "baseline_dy": 0.25,
            },
        )
        self.assertEqual(report["conditions"]["length_change"], "shorter")
        self.assertEqual(report["strategy"], "left_anchor_span")
        self.assertEqual(report["strategy_contract"]["anchor_priority"], "left_boundary")
        self.assertTrue(report["strategy_contract"]["cleanup_required"])
        self.assertTrue(report["actual_errors"]["extra_source_cleanup_enabled"])
        self.assertEqual(report["actual_errors"]["span_width_delta"], 1.0)
        self.assertTrue(report["pass"])

    def test_longer_replacement_left_anchors_extends_right_and_guards_protected_text(self) -> None:
        slot_report = {
            "pass": True,
            "length_change_report": {
                "right_boundary": {
                    "pass": True,
                    "space_sufficient": True,
                    "protected_gap_px": 12,
                    "minimum_safe_gap_px": 3,
                }
            },
        }
        strategy, reason = choose_placement_strategy(
            source_text="赵芳",
            target_text="陈小慧",
            slots=(run(2), run(14)),
            slot_report=slot_report,
            draw_mode="line_chars",
        )
        plan = plan_for(
            source="赵芳",
            target="陈小慧",
            slots=(run(2), run(14)),
            strategy=strategy,
            reason=reason,
            slot_report=slot_report,
            draw_mode="line_chars",
        )
        report = strategy_report(
            plan,
            {
                "enabled": True,
                "left_boundary_dx": 0.25,
                "span_width_delta": 1.5,
                "char_spacing_delta": 0.5,
                "baseline_dy": 0.25,
                "protected_text_overlap_pixels": 0,
            },
        )
        self.assertEqual(report["conditions"]["length_change"], "longer")
        self.assertEqual(report["conditions"]["source_slot_count"], 2)
        self.assertEqual(report["conditions"]["target_slot_count"], 3)
        self.assertEqual(report["strategy"], "left_anchor_span")
        self.assertTrue(report["strategy_contract"]["protected_text_guard_checked"])
        self.assertTrue(report["strategy_contract"]["longer_text_appends_slots"])
        self.assertEqual(report["constraints"]["protected_text_overlap_pixels"], 0)
        self.assertTrue(report["constraints"]["right_boundary_diagnostic_required"])
        self.assertEqual(report["actual_errors"]["protected_text_overlap_pixels"], 0)
        self.assertTrue(report["actual_errors"]["right_boundary_pass"])
        self.assertTrue(report["actual_errors"]["right_boundary_space_sufficient"])
        self.assertTrue(report["pass"])

    def test_numeric_date_and_age_use_left_baseline_rhythm_and_field_width_constraints(self) -> None:
        for source, target, slot_count in (("2024-01-01", "2025-02-03", 10), ("18", "19", 2)):
            with self.subTest(source=source):
                strategy, reason = choose_placement_strategy(
                    source_text=source,
                    target_text=target,
                    slots=tuple(run(2 + idx * 8, x2=2 + idx * 8 + 5) for idx in range(slot_count)),
                    slot_report={"pass": True},
                    draw_mode="auto",
                )
                plan = plan_for(
                    source=source,
                    target=target,
                    slots=tuple(run(2 + idx * 8, x2=2 + idx * 8 + 5) for idx in range(slot_count)),
                    strategy=strategy,
                    reason=reason,
                    slot_report={"pass": True},
                )
                report = strategy_report(
                    plan,
                    {
                        "enabled": True,
                        "left_boundary_dx": 0.25,
                        "baseline_dy": 0.25,
                        "rhythm_delta": 0.5,
                        "field_width_delta": 1.0,
                    },
                )
                self.assertEqual(report["strategy"], "baseline_numeric")
                self.assertEqual(report["strategy_contract"]["anchor_priority"], "left_baseline")
                self.assertEqual(report["constraints"]["max_rhythm_delta"], 1.5)
                self.assertEqual(report["actual_errors"]["field_width_delta"], 1.0)
                self.assertTrue(report["pass"])

    def test_manual_roi_without_source_uses_conservative_fallback_and_reduced_acceptance_confidence(self) -> None:
        strategy, reason = choose_placement_strategy(
            source_text="",
            target_text="陈芸",
            slots=(),
            slot_report={"pass": True},
            draw_mode="auto",
        )
        plan = plan_for(
            source="",
            target="陈芸",
            slots=(),
            strategy=strategy,
            reason=reason,
            slot_report={"pass": True},
        )
        report = strategy_report(plan, {"enabled": False, "reason": "manual_roi_no_source"})
        self.assertEqual(report["strategy"], "manual_fallback")
        self.assertEqual(report["conditions"]["length_change"], "unknown")
        self.assertTrue(report["conditions"]["manual_source_missing"])
        self.assertEqual(report["strategy_contract"]["anchor_priority"], "manual_roi_conservative")
        self.assertEqual(report["strategy_contract"]["auto_acceptance_confidence"], "reduced")
        self.assertEqual(report["strategy_contract"]["auto_acceptance_confidence_cap"], 0.45)
        self.assertTrue(report["pass"])

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
