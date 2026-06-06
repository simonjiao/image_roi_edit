from __future__ import annotations

import unittest

from roi_image_edit.revision_solver import (
    _core_light_micro_variants,
    _core_overblack_micro_variants,
    _core_overblack_near_threshold,
    _build_micro_tuning_report,
    ink_gray_candidate_grid,
    ink_gray_micro_tuning_candidates,
    ink_gray_near_threshold_micro_tuning,
)

from tests.test_ink_gray_candidate_grid import params


def _overblack_report(gap: float = 32.0) -> dict:
    limit = 48.0
    actual = limit + gap
    return {
        "pass": True,
        "local_ink_balance_issues": [
            {"type": "roi_core_too_black", "actual": actual, "limit": limit},
        ],
    }


def _overblack_report_changed_char(gap: float = 32.0) -> dict:
    limit = 48.0
    actual = limit + gap
    return {
        "pass": True,
        "local_ink_balance_issues": [
            {"type": "changed_char_core_too_black", "actual": actual, "limit": limit},
        ],
    }


def _mixed_overblack_report() -> dict:
    return {
        "pass": True,
        "local_ink_balance_issues": [
            {"type": "roi_core_too_black", "actual": 52.0, "limit": 48.0},
            {"type": "core_mean_gray_too_light", "actual": 5.2, "limit": 4.0},
        ],
    }


class InkGrayMicroTuningTest(unittest.TestCase):

    def test_near_threshold_overblack_generates_core_reduction_micro_candidates(self) -> None:
        report = _overblack_report(gap=4.0)
        micro = ink_gray_near_threshold_micro_tuning(report)
        self.assertTrue(micro["enabled"], f"expected enabled, got {micro}")
        self.assertEqual(micro["candidate_family"], "core_only_micro_reduction")

        base = params()
        candidates = ink_gray_micro_tuning_candidates(base, report)
        self.assertGreater(len(candidates), 0)
        for candidate in candidates:
            self.assertEqual(candidate.font_name, base.font_name)
            self.assertEqual(candidate.font_size, base.font_size)
            self.assertEqual(candidate.text_dx, base.text_dx)
            self.assertEqual(candidate.text_dy, base.text_dy)
            self.assertEqual(candidate.blur, base.blur)
            self.assertEqual(candidate.mask_threshold, base.mask_threshold)
            has_ink_change = (
                candidate.opacity != base.opacity
                or candidate.core_ink_gain != base.core_ink_gain
                or candidate.core_darken_strength != base.core_darken_strength
                or candidate.alpha_contrast != base.alpha_contrast
            )
            self.assertTrue(has_ink_change, "overblack micro candidate must change ink parameters")

    def test_changed_char_core_too_black_near_threshold(self) -> None:
        report = _overblack_report_changed_char(gap=3.0)
        micro = ink_gray_near_threshold_micro_tuning(report)
        self.assertTrue(micro["enabled"], f"expected enabled for changed_char_core_too_black, got {micro}")
        self.assertEqual(micro["candidate_family"], "core_only_micro_reduction")

    def test_overblack_not_near_threshold_disabled(self) -> None:
        report = _overblack_report(gap=80.0)
        micro = ink_gray_near_threshold_micro_tuning(report)
        self.assertFalse(micro["enabled"])
        self.assertEqual(micro["reason"], "overblack_gap_not_near_threshold")

    def test_overblack_micro_tuning_report_records_metric_gap_and_candidates(self) -> None:
        report = _overblack_report(gap=4.0)
        micro = ink_gray_near_threshold_micro_tuning(report)
        self.assertTrue(micro["enabled"])
        self.assertIn("family", micro or {})
        self.assertIn("candidate_family", micro)
        self.assertIn("metric", micro)
        self.assertIn("actual", micro)
        self.assertIn("limit_value", micro)
        self.assertIn("gap", micro)
        self.assertIn("gap_ratio", micro)

        base = params()
        candidates = ink_gray_micro_tuning_candidates(base, report)
        candidate_ids = [c.candidate_id for c in candidates]
        report_out = _build_micro_tuning_report(micro, candidates, candidate_ids)
        self.assertTrue(report_out["enabled"])
        self.assertEqual(report_out["family"], "core_only_micro_reduction")
        self.assertEqual(report_out["stage_id"], "ink_gray_balance")
        self.assertIn("metric", report_out)
        self.assertIn("actual", report_out)
        self.assertIn("limit", report_out)
        self.assertIn("gap", report_out)
        self.assertIn("gap_ratio", report_out)
        self.assertEqual(report_out["candidate_count"], len(candidates))
        self.assertEqual(report_out["candidate_ids"], candidate_ids)

    def test_overblack_micro_tuning_report_disabled_reason(self) -> None:
        report = _overblack_report(gap=80.0)
        micro = ink_gray_near_threshold_micro_tuning(report)
        report_out = _build_micro_tuning_report(micro, [], [])
        self.assertFalse(report_out["enabled"])
        self.assertEqual(report_out["family"], "")
        self.assertEqual(report_out["candidate_count"], 0)
        self.assertEqual(report_out["candidate_ids"], [])
        self.assertIn("disabled_reason", report_out)

    def test_overblack_mixed_with_other_issues_disabled(self) -> None:
        report = _mixed_overblack_report()
        micro = ink_gray_near_threshold_micro_tuning(report)
        self.assertFalse(micro["enabled"])
        self.assertIn("other_ink_gray_issues_present", micro.get("reason", ""))

    def test_overblack_no_issues_disabled(self) -> None:
        report = {"pass": True}
        micro = ink_gray_near_threshold_micro_tuning(report)
        self.assertFalse(micro["enabled"])
        self.assertEqual(micro["reason"], "no_ink_gray_issues")

    def test_ink_gray_grid_exposes_overblack_micro_report(self) -> None:
        report = _overblack_report(gap=4.0)
        base = params()
        grid = ink_gray_candidate_grid(base, report, limit=8)
        report_out = grid.report
        self.assertTrue(report_out.get("enabled"))
        overblack_report = report_out.get("overblack_micro_tuning_report")
        self.assertIsInstance(overblack_report, dict)
        self.assertEqual(overblack_report.get("stage_id"), "ink_gray_balance")
        self.assertIn("candidate_count", overblack_report)
        self.assertIn("candidate_ids", overblack_report)

    def test_overblack_micro_variants_only_change_ink_params(self) -> None:
        base = params()
        variants = _core_overblack_micro_variants(base)
        self.assertGreater(len(variants), 0)
        for candidate in variants:
            self.assertEqual(candidate.font_name, base.font_name)
            self.assertEqual(candidate.font_path, base.font_path)
            self.assertEqual(candidate.font_size, base.font_size)
            self.assertEqual(candidate.text_dx, base.text_dx)
            self.assertEqual(candidate.text_dy, base.text_dy)
            self.assertEqual(candidate.char_offsets, base.char_offsets)
            self.assertEqual(candidate.blur, base.blur)
            self.assertEqual(candidate.mask_threshold, base.mask_threshold)
            self.assertEqual(candidate.photo_warp, base.photo_warp)
            self.assertEqual(candidate.edge_breakup, base.edge_breakup)
            self.assertEqual(candidate.photo_noise, base.photo_noise)
            self.assertEqual(candidate.jpeg_quality, base.jpeg_quality)

    def test_core_light_micro_variants_only_change_ink_params(self) -> None:
        base = params()
        variants = _core_light_micro_variants(base)
        self.assertGreater(len(variants), 0)
        for candidate in variants:
            self.assertEqual(candidate.font_name, base.font_name)
            self.assertEqual(candidate.font_path, base.font_path)
            self.assertEqual(candidate.font_size, base.font_size)
            self.assertEqual(candidate.text_dx, base.text_dx)
            self.assertEqual(candidate.text_dy, base.text_dy)
            self.assertEqual(candidate.char_offsets, base.char_offsets)
            self.assertEqual(candidate.blur, base.blur)
            self.assertEqual(candidate.mask_threshold, base.mask_threshold)
            self.assertEqual(candidate.photo_warp, base.photo_warp)
            self.assertEqual(candidate.edge_breakup, base.edge_breakup)
            self.assertEqual(candidate.photo_noise, base.photo_noise)
            self.assertEqual(candidate.jpeg_quality, base.jpeg_quality)


if __name__ == "__main__":
    unittest.main()
