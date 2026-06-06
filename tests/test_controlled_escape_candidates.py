from __future__ import annotations

import unittest

from roi_image_edit.revision_solver import (
    controlled_escape_candidate_grid,
    _current_stage_near_threshold,
)


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

    def test_cross_stage_cartesian_disabled_stays_false_in_layered_search(self) -> None:
        from roi_image_edit.revision_solver import layered_candidate_search_report
        layered = layered_candidate_search_report()
        self.assertFalse(layered.get("cross_stage_cartesian_search", True))


if __name__ == "__main__":
    unittest.main()
