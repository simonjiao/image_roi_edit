from __future__ import annotations

import unittest

from roi_image_edit.revision_solver import (
    controlled_escape_candidate_grid,
    _current_stage_near_threshold,
)
from roi_image_edit.run_artifacts import revision_round_continuation_contract


class ControlledEscapeCandidatesTest(unittest.TestCase):

    def test_no_blocking_stage_disables_escape(self) -> None:
        report = controlled_escape_candidate_grid(
            None, {}, None, hard_boundary_passed=True, prior_stage_pass=True
        )
        self.assertFalse(report["enabled"])
        self.assertEqual(report["reason"], "blocking_stage_not_near_threshold")

    def test_hard_boundary_failure_disables_escape(self) -> None:
        from tests.test_ink_gray_candidate_grid import params
        report = {
            "pass": True,
            "local_ink_balance_issues": [
                {"type": "roi_core_too_black", "actual": 51.0, "limit": 48.0},
            ],
        }
        grid = controlled_escape_candidate_grid(
            params(), report, "ink_gray_balance", hard_boundary_passed=False, prior_stage_pass=True
        )
        self.assertFalse(grid["enabled"])
        self.assertIn("hard_boundary", grid["reason"])

    def test_prior_stage_regression_disables_escape(self) -> None:
        from tests.test_ink_gray_candidate_grid import params
        report = {
            "pass": True,
            "local_ink_balance_issues": [
                {"type": "roi_core_too_black", "actual": 51.0, "limit": 48.0},
            ],
        }
        grid = controlled_escape_candidate_grid(
            params(), report, "ink_gray_balance", hard_boundary_passed=True, prior_stage_pass=False
        )
        self.assertFalse(grid["enabled"])
        self.assertIn("prior_stage_regression", grid["reason"])

    def test_escape_marks_controlled_escape_true(self) -> None:
        from tests.test_ink_gray_candidate_grid import params

        base = params()
        report = {
            "pass": True,
            "local_ink_balance_issues": [
                {"type": "roi_core_too_black", "actual": 51.0, "limit": 48.0},
            ],
        }
        grid = controlled_escape_candidate_grid(
            base, report, "ink_gray_balance", hard_boundary_passed=True, prior_stage_pass=True
        )
        self.assertTrue(grid["controlled_escape"])
        self.assertIn("primary_stage", grid)
        self.assertIn("secondary_stage", grid)
        self.assertTrue(grid.get("cross_stage_cartesian_disabled"))

    def test_escape_without_near_threshold_is_disabled(self) -> None:
        grid = controlled_escape_candidate_grid(
            None, {"pass": True}, "ink_gray_balance", hard_boundary_passed=True, prior_stage_pass=True
        )
        self.assertFalse(grid["enabled"])
        self.assertEqual(grid["reason"], "blocking_stage_not_near_threshold")

    def test_escape_candidates_have_upper_budget_limit(self) -> None:
        from roi_image_edit.revision_solver import CONTROLLED_ESCAPE_LIMIT
        self.assertLessEqual(CONTROLLED_ESCAPE_LIMIT, 6, "escape limit must be small to prevent cartesian cross-stage search")

    def test_escape_declares_allowed_secondary_delta_bounds(self) -> None:
        grid = controlled_escape_candidate_grid(
            None, {}, "photo_texture", hard_boundary_passed=True, prior_stage_pass=True
        )
        if grid.get("enabled"):
            self.assertIn("allowed_secondary_delta_bounds", grid)
            self.assertIsInstance(grid["allowed_secondary_delta_bounds"], dict)

    def test_longer_mid_gray_body_escape_can_use_bounded_font_size_step(self) -> None:
        from tests.test_ink_gray_candidate_grid import params

        base = params()
        report = {
            "pass": True,
            "pipeline_profile": "photo_scan",
            "local_ink_balance_issues": [
                {
                    "type": "longer_mid_gray_body_too_black",
                    "actual": 197.0,
                    "limit": 111.28,
                    "lt90_delta": 411.0,
                    "expected_lt90_delta": 214.0,
                    "length_change": "longer",
                }
            ],
        }

        self.assertTrue(_current_stage_near_threshold(report, "ink_gray_balance"))
        grid = controlled_escape_candidate_grid(
            base,
            report,
            "ink_gray_balance",
            hard_boundary_passed=True,
            prior_stage_pass=True,
        )

        self.assertTrue(grid["enabled"], grid)
        self.assertEqual(grid["secondary_stage"], "text_shape")
        self.assertEqual(grid["escape_strategy"], "longer_mid_gray_font_size_micro_escape")
        self.assertEqual(grid["candidate_count"], 4)
        self.assertTrue(grid["cross_stage_cartesian_disabled"])
        bounds = grid["allowed_secondary_delta_bounds"]
        self.assertEqual(bounds["font_size"], (-1, -1))
        for candidate in grid["candidates"]:
            self.assertEqual(candidate.font_size, base.font_size - 1)
            self.assertLess(candidate.opacity, base.opacity)
            self.assertGreater(candidate.blur, base.blur)
            self.assertGreater(candidate.alpha_contrast, base.alpha_contrast)
        for audit in grid["candidate_delta_audit"]:
            self.assertIn("font_size", audit["delta_keys"])
            self.assertIn("opacity", audit["delta_keys"])
            self.assertIn("blur", audit["delta_keys"])
            self.assertTrue(audit["controlled_escape"])

    def test_cjk_longer_overblack_escape_uses_large_shape_and_texture_step(self) -> None:
        from tests.test_ink_gray_candidate_grid import params

        base = params()
        report = {
            "pass": True,
            "classification": {
                "image_type": "photo_document",
                "scenario": "form_field_value_replace",
                "script": "cjk",
                "length_change": "longer",
                "class_key": "photo_document.form_field_value_replace.cjk",
            },
            "class_key": "photo_document.form_field_value_replace.cjk",
            "roi_plan": {
                "source_slot_count": 2,
                "target_slot_count": 3,
            },
            "stage_gate": {
                "blocking_stage": "ink_gray_balance",
                "stage_status": {
                    "text_shape": {
                        "pass": True,
                        "deferred_issues": [
                            {"type": "changed_char_stroke_body_too_narrow"}
                        ],
                    },
                    "ink_gray_balance": {
                        "pass": False,
                        "issues": [
                            {"type": "roi_core_too_black", "actual": 119.0, "limit": 83.88}
                        ],
                    },
                },
                "stages": [
                    {
                        "id": "text_shape",
                        "pass": True,
                        "issues": [],
                    },
                    {
                        "id": "ink_gray_balance",
                        "pass": False,
                        "issues": [
                            {"type": "roi_core_too_black", "actual": 119.0, "limit": 83.88}
                        ],
                    },
                ],
            },
            "local_ink_balance_issues": [
                {"type": "roi_core_too_black", "actual": 119.0, "limit": 83.88}
            ],
        }

        self.assertTrue(_current_stage_near_threshold(report, "ink_gray_balance"))
        grid = controlled_escape_candidate_grid(
            base,
            report,
            "ink_gray_balance",
            hard_boundary_passed=True,
            prior_stage_pass=True,
        )

        self.assertTrue(grid["enabled"], grid)
        self.assertEqual(grid["escape_strategy"], "overblack_body_shape_escape")
        self.assertEqual(grid["secondary_stage"], "text_shape")
        self.assertEqual(grid["candidate_count"], 4)
        self.assertEqual(grid["trigger"]["reason"], "cjk_longer_overblack_needs_shape_ink_escape")
        bounds = grid["allowed_secondary_delta_bounds"]
        self.assertEqual(bounds["font_size"], (-1, 0))
        self.assertEqual(bounds["opacity"], (-0.12, -0.05))
        self.assertEqual(bounds["blur"], (-0.28, -0.12))
        self.assertLess(grid["candidates"][0].opacity, base.opacity - 0.05)
        self.assertLess(grid["candidates"][0].blur, base.blur - 0.12)
        self.assertEqual(grid["candidates"][0].font_size, base.font_size - 1)
        for audit in grid["candidate_delta_audit"]:
            self.assertIn("opacity", audit["delta_keys"])
            self.assertIn("blur", audit["delta_keys"])
            self.assertTrue(audit["controlled_escape"])

    def test_continuation_contract_allows_controlled_escape_only_round(self) -> None:
        contract = revision_round_continuation_contract(
            {
                "round": 3,
                "basis_blocking_stage": "ink_gray_balance",
                "controlled_escape_grid": {
                    "enabled": True,
                    "primary_stage": "ink_gray_balance",
                    "secondary_stage": "text_shape",
                    "escape_strategy": "longer_mid_gray_font_size_micro_escape",
                    "candidate_count": 4,
                },
            },
            max_revision_rounds=8,
        )

        self.assertTrue(contract["continuation_allowed"])
        self.assertEqual(
            contract["candidate_direction_sources"][0]["source"],
            "controlled_escape_grid",
        )
        self.assertEqual(
            contract["candidate_direction_sources"][0]["escape_strategy"],
            "longer_mid_gray_font_size_micro_escape",
        )

    def test_cross_stage_cartesian_disabled_stays_false_in_layered_search(self) -> None:
        from roi_image_edit.revision_solver import layered_candidate_search_report
        layered = layered_candidate_search_report()
        self.assertFalse(layered.get("cross_stage_cartesian_search", True))


if __name__ == "__main__":
    unittest.main()
