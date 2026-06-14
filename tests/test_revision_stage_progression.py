from __future__ import annotations

import inspect
import unittest

from roi_image_edit.region_processing import (
    progresses_past_blocking_stage,
    report_stage_status_pass,
    run_region_vision_checks,
    text_shape_ink_guard_selectable,
)


def staged_report(*, ink_pass: bool, blocking_stage: str | None) -> dict:
    return {
        "stage_gate": {
            "pass": blocking_stage is None,
            "blocking_stage": blocking_stage,
            "stage_status": {
                "hard_boundary": {"pass": True},
                "text_shape": {"pass": True},
                "ink_gray_balance": {"pass": ink_pass},
                "background_cleanup": {"pass": blocking_stage is None},
            },
        }
    }


def text_shape_ink_guard_report(*, text_issue_count: int, ink_lt55_delta: float) -> dict:
    return {
        "pass": False,
        "local_ink_balance_issues": [
            {
                "type": "roi_core_too_black",
                "lt55_delta": ink_lt55_delta,
                "limit": 80.0,
            }
        ],
        "stage_gate": {
            "pass": False,
            "blocking_stage": "text_shape",
            "stages": [
                {"id": "hard_boundary", "pass": True, "issues": []},
                {
                    "id": "text_shape",
                    "pass": False,
                    "issues": [
                        {"type": f"shape_issue_{index}"}
                        for index in range(text_issue_count)
                    ],
                },
                {
                    "id": "ink_gray_balance",
                    "pass": False,
                    "issues": [
                        {
                            "type": "roi_core_too_black",
                            "lt55_delta": ink_lt55_delta,
                            "limit": 80.0,
                        }
                    ],
                },
            ],
        },
    }


class RevisionStageProgressionTest(unittest.TestCase):
    def test_stage_status_pass_reads_named_stage(self) -> None:
        report = staged_report(ink_pass=True, blocking_stage="background_cleanup")

        self.assertTrue(report_stage_status_pass(report, "ink_gray_balance"))
        self.assertFalse(report_stage_status_pass(report, "background_cleanup"))

    def test_candidate_that_passes_current_stage_and_blocks_later_progresses(self) -> None:
        report = staged_report(ink_pass=True, blocking_stage="background_cleanup")

        self.assertTrue(
            progresses_past_blocking_stage(
                report,
                current_blocking_stage="ink_gray_balance",
                next_blocking_stage="background_cleanup",
            )
        )

    def test_candidate_that_moves_to_prior_stage_does_not_progress(self) -> None:
        report = staged_report(ink_pass=True, blocking_stage="text_shape")

        self.assertFalse(
            progresses_past_blocking_stage(
                report,
                current_blocking_stage="ink_gray_balance",
                next_blocking_stage="text_shape",
            )
        )

    def test_candidate_still_blocked_at_current_stage_does_not_progress(self) -> None:
        report = staged_report(ink_pass=False, blocking_stage="ink_gray_balance")

        self.assertFalse(
            progresses_past_blocking_stage(
                report,
                current_blocking_stage="ink_gray_balance",
                next_blocking_stage="ink_gray_balance",
            )
        )

    def test_revision_loop_allows_current_stage_progression_as_selectable(self) -> None:
        source = inspect.getsource(run_region_vision_checks)

        self.assertIn("progresses_past_current_stage = progresses_past_blocking_stage", source)
        self.assertIn("or progresses_past_current_stage", source)

    def test_text_shape_ink_guard_is_selectable_only_when_shape_does_not_regress(self) -> None:
        before = text_shape_ink_guard_report(text_issue_count=2, ink_lt55_delta=300.0)
        after = text_shape_ink_guard_report(text_issue_count=2, ink_lt55_delta=120.0)

        guard = text_shape_ink_guard_selectable(before, after)

        self.assertTrue(guard["enabled"])
        self.assertTrue(guard["text_shape_not_regressed"])
        self.assertTrue(guard["ink_gray_improved"])
        self.assertTrue(guard["selectable"])

    def test_text_shape_ink_guard_rejects_shape_regression_even_if_ink_improves(self) -> None:
        before = text_shape_ink_guard_report(text_issue_count=1, ink_lt55_delta=300.0)
        after = text_shape_ink_guard_report(text_issue_count=2, ink_lt55_delta=120.0)

        guard = text_shape_ink_guard_selectable(before, after)

        self.assertFalse(guard["text_shape_not_regressed"])
        self.assertTrue(guard["ink_gray_improved"])
        self.assertFalse(guard["selectable"])

    def test_revision_loop_records_ink_guard_selection_condition(self) -> None:
        source = inspect.getsource(run_region_vision_checks)

        self.assertIn("text_shape_ink_guard_selectable", source)
        self.assertIn("text_shape_ink_guard_reduces_excess_black_core", source)
        self.assertIn('or ink_guard_selection.get("selectable")', source)


if __name__ == "__main__":
    unittest.main()
