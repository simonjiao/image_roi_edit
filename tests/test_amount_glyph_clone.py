from __future__ import annotations

import unittest

import numpy as np
from PIL import Image, ImageDraw

from roi_image_edit.amount_glyph_clone import AmountGlyphSource, replace_amount_with_glyph_sources


def draw_amount(image: Image.Image, *, x: int, y: int, text: str) -> tuple[int, int, int, int]:
    draw = ImageDraw.Draw(image)
    cursor = x
    boxes: list[tuple[int, int, int, int]] = []
    for char in text:
        width = 8 if char != "1" else 5
        height = 16 if char != "+" else 11
        top = y if char != "+" else y + 3
        box = (cursor, top, cursor + width, top + height)
        draw.rectangle(box, fill=(55, 55, 55))
        if char in {"2", "8"}:
            draw.line((cursor + 1, top + height // 2, cursor + width - 1, top + height // 2), fill=(230, 230, 230))
        boxes.append(box)
        cursor += width + 6
    return (min(box[0] for box in boxes), min(box[1] for box in boxes), max(box[2] for box in boxes), max(box[3] for box in boxes))


class AmountGlyphCloneTest(unittest.TestCase):
    def test_clones_missing_amount_digits_from_reference_rows_and_keeps_suffix_unchanged(self) -> None:
        image = Image.new("RGB", (220, 120), (255, 255, 255))
        edit_roi = (35, 10, 180, 42)
        reference_roi = (35, 55, 180, 88)
        draw_amount(image, x=88, y=18, text="+9764")
        draw_amount(image, x=88, y=62, text="+1222")
        draw = ImageDraw.Draw(image)
        draw.rectangle((184, 18, 212, 34), fill=(55, 55, 55))

        edited, report = replace_amount_with_glyph_sources(
            image,
            edit_roi,
            source_text="+9764",
            target_text="+12749",
            glyph_sources=[AmountGlyphSource(image, reference_roi, "+1222", label="reference_row")],
        )

        self.assertTrue(report["pass"], report)
        self.assertEqual(report["operation"], "amount_glyph_clone")
        self.assertEqual(report["target_box"][2], report["source_box"][2])
        self.assertEqual(report["hard_check"]["outside_roi_changed_pixels"], 0)
        self.assertEqual(report["unexpected_changed_pixels"], 0)
        self.assertIn(
            "reference_row",
            {placement["glyph"]["label"] for placement in report["placement"] if placement["char"] == "2"},
        )

        original_arr = np.array(image)
        edited_arr = np.array(edited)
        self.assertFalse(np.array_equal(original_arr, edited_arr))
        self.assertTrue(np.array_equal(original_arr[:, 180:], edited_arr[:, 180:]))

    def test_reports_missing_glyphs_without_modifying_image(self) -> None:
        image = Image.new("RGB", (180, 80), (255, 255, 255))
        edit_roi = (30, 10, 160, 42)
        draw_amount(image, x=80, y=18, text="+9764")

        edited, report = replace_amount_with_glyph_sources(
            image,
            edit_roi,
            source_text="+9764",
            target_text="+12849",
            glyph_sources=[],
        )

        self.assertFalse(report["pass"])
        self.assertEqual(report["reason"], "missing_target_glyphs")
        self.assertEqual(report["missing"][0]["char"], "1")
        self.assertTrue(np.array_equal(np.array(image), np.array(edited)))


if __name__ == "__main__":
    unittest.main()
