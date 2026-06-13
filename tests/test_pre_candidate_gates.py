from __future__ import annotations

import unittest

from roi_image_edit.pre_candidate_gates import (
    PRE_CANDIDATE_GATE_ORDER,
    classify_pre_candidate_slot_failure,
    pre_candidate_gate_report,
)


class PreCandidateGatesTest(unittest.TestCase):
    def test_gate_order_is_stable(self) -> None:
        self.assertEqual(
            PRE_CANDIDATE_GATE_ORDER,
            ("orientation_check", "field_roi_selection", "slot_quality_gate", "protected_text_guard"),
        )

    def test_slot_and_protected_failures_are_classified_separately(self) -> None:
        self.assertEqual(
            classify_pre_candidate_slot_failure(
                {"pass": False, "issues": [{"type": "slot_bottom_overflow"}]}
            ),
            "slot_quality_gate",
        )
        self.assertEqual(
            classify_pre_candidate_slot_failure(
                {"pass": False, "issues": [{"type": "slot_overlaps_protected_text"}]}
            ),
            "protected_text_guard",
        )

    def test_pre_candidate_report_records_candidate_count_and_statuses(self) -> None:
        report = pre_candidate_gate_report(
            candidate_count=0,
            regions=[{"id": "r1"}],
            slot_quality_report={"pass": False, "issues": [{"type": "target_roi_overlaps_protected_text"}]},
        )
        self.assertFalse(report["pass"])
        self.assertEqual(report["failed_gate"], "protected_text_guard")
        self.assertEqual(report["failure_stage"], "pre_candidate_generation")
        self.assertEqual(report["candidate_count"], 0)
        self.assertTrue(report["statuses"]["field_roi_selection"]["pass"])
        self.assertTrue(report["statuses"]["slot_quality_gate"]["pass"])
        self.assertFalse(report["statuses"]["protected_text_guard"]["pass"])

    def test_right_boundary_diagnostic_does_not_fail_pre_candidate_gate(self) -> None:
        report = pre_candidate_gate_report(
            candidate_count=3,
            regions=[{"id": "r1"}],
            slot_quality_report={
                "pass": True,
                "issues": [],
                "length_change_report": {
                    "right_boundary": {
                        "diagnostic_only": True,
                        "space_sufficient": False,
                        "diagnostic_issue": {"type": "right_boundary_too_close_to_protected_text"},
                    }
                },
            },
        )
        self.assertTrue(report["pass"])
        self.assertIsNone(report["failed_gate"])


if __name__ == "__main__":
    unittest.main()
