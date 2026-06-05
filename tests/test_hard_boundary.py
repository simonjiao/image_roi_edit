from __future__ import annotations

import unittest

from PIL import Image

from roi_image_edit.iterative_pipeline import hard_check


def base_image(size: tuple[int, int] = (8, 8)) -> Image.Image:
    return Image.new("RGB", size, (220, 220, 220))


def changed_pixel(image: Image.Image, xy: tuple[int, int], color: tuple[int, int, int]) -> Image.Image:
    out = image.copy()
    out.putpixel(xy, color)
    return out


class HardBoundaryTest(unittest.TestCase):
    def test_size_mismatch_fails_and_records_sizes(self) -> None:
        original = base_image((8, 8))
        candidate = base_image((9, 8))
        report = hard_check(original, candidate, (2, 2, 6, 6))
        self.assertFalse(report["pass"])
        self.assertEqual(report["original_size"], [8, 8])
        self.assertEqual(report["candidate_size"], [9, 8])
        self.assertFalse(report["size_match"])
        self.assertIsNone(report["outside_roi_changed_pixels"])
        self.assertIsNone(report["border_changed_pixels"])

    def test_inside_roi_change_passes_with_zero_outside_and_border_diff(self) -> None:
        original = base_image()
        candidate = changed_pixel(original, (3, 3), (20, 20, 20))
        report = hard_check(original, candidate, (2, 2, 6, 6))
        self.assertTrue(report["pass"])
        self.assertTrue(report["size_match"])
        self.assertEqual(report["outside_roi_changed_pixels"], 0)
        self.assertEqual(report["border_changed_pixels"], 0)
        self.assertEqual(report["protected_changed_pixels"], 0)

    def test_outside_roi_change_fails_and_records_pixel_count(self) -> None:
        original = base_image()
        candidate = changed_pixel(original, (1, 3), (20, 20, 20))
        report = hard_check(original, candidate, (2, 2, 6, 6))
        self.assertFalse(report["pass"])
        self.assertEqual(report["outside_roi_changed_pixels"], 1)
        self.assertEqual(report["border_changed_pixels"], 0)

    def test_border_change_fails_and_records_pixel_count(self) -> None:
        original = base_image()
        candidate = changed_pixel(original, (0, 4), (20, 20, 20))
        report = hard_check(original, candidate, (2, 2, 6, 6))
        self.assertFalse(report["pass"])
        self.assertEqual(report["outside_roi_changed_pixels"], 1)
        self.assertEqual(report["border_changed_pixels"], 1)

    def test_protected_text_change_fails_and_records_diff_for_manual_and_auto_regions(self) -> None:
        original = base_image()
        candidate = changed_pixel(original, (3, 3), (20, 20, 20))
        for label, protected_boxes in {
            "manual_roi": ((3, 3, 5, 5),),
            "auto_roi": ((2, 2, 4, 4),),
        }.items():
            with self.subTest(label=label):
                report = hard_check(original, candidate, (2, 2, 6, 6), protected_boxes)
                self.assertFalse(report["pass"])
                self.assertEqual(report["protected_changed_pixels"], 1)
                self.assertEqual(report["outside_roi_changed_pixels"], 0)
                self.assertEqual(report["border_changed_pixels"], 0)


if __name__ == "__main__":
    unittest.main()
