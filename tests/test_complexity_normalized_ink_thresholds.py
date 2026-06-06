from __future__ import annotations

import unittest

from roi_image_edit.complexity_normalization import (
    complexity_normalized_ink_limits,
    text_complexity,
)


class ComplexityNormalizedInkThresholdsTest(unittest.TestCase):

    def test_same_length_same_complexity_no_normalization(self) -> None:
        result = complexity_normalized_ink_limits("姓名", "姓名")
        self.assertFalse(result["enabled"])
        self.assertEqual(result["text_count_ratio"], 1.0)
        self.assertEqual(result["normalized_core_black_limit"], 48.0)
        self.assertEqual(result["normalization_reason"], "no_normalization_needed")

    def test_longer_replacement_records_normalized_fields(self) -> None:
        result = complexity_normalized_ink_limits("张三", "张三四")
        self.assertTrue(result["enabled"])
        self.assertEqual(result["source_text_count"], 2)
        self.assertEqual(result["target_text_count"], 3)
        self.assertGreater(result["text_count_ratio"], 1.0)
        self.assertIn("source_complexity", result)
        self.assertIn("target_complexity", result)
        self.assertIn("complexity_ratio", result)
        self.assertIn("normalized_core_black_limit", result)
        self.assertIn("normalization_reason", result)

    def test_complexity_ratio_recorded_when_different(self) -> None:
        simple = "一二三"
        medium = "繁體字"
        result = complexity_normalized_ink_limits(simple, medium)
        self.assertGreater(result["complexity_ratio"], 1.0)
        self.assertIn("source_complexity", result)
        self.assertIn("target_complexity", result)

    def test_complexity_normalization_never_relaxes_hard_boundary(self) -> None:
        result = complexity_normalized_ink_limits("张", "张三李四王五")
        self.assertTrue(result["enabled"])
        self.assertTrue(result["hard_boundary_not_relaxed"])
        self.assertTrue(result["protected_text_not_relaxed"])
        self.assertTrue(result["slot_quality_not_relaxed"])
        self.assertTrue(result["outside_roi_not_relaxed"])
        self.assertTrue(result["old_text_residual_not_relaxed"])
        self.assertTrue(result["affects_only_ink_quality_thresholds"])

    def test_normalized_limit_has_upper_cap(self) -> None:
        result = complexity_normalized_ink_limits("一", "繁繁繁繁繁繁繁")
        self.assertLessEqual(result["normalized_core_black_limit"], 48.0 * 2.5)

    def test_text_complexity_cjk_higher_than_ascii(self) -> None:
        cjk = text_complexity("繁")
        ascii_score = text_complexity("a")
        self.assertGreaterEqual(cjk, ascii_score)

    def test_empty_text_returns_zero_complexity(self) -> None:
        self.assertEqual(text_complexity(""), 0.0)
        self.assertEqual(text_complexity("  "), 0.0)


if __name__ == "__main__":
    unittest.main()
