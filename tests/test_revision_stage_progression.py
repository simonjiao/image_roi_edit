from __future__ import annotations

import inspect
import unittest

from roi_image_edit.region_processing import (
    progresses_past_blocking_stage,
    report_stage_status_pass,
    run_region_vision_checks,
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


if __name__ == "__main__":
    unittest.main()
