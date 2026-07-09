from __future__ import annotations

import unittest

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from roi_image_edit.amount_replacement import _choose_amount_font, replace_amount_in_roi


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


class AmountReplacementTest(unittest.TestCase):
    def test_amount_font_selection_avoids_bold_near_tie_for_longer_amount(self) -> None:
        _font, report = _choose_amount_font("+5739", "+22882", (444, 1157, 509, 1176))

        self.assertNotIn("Arial Bold", report["font_path"])
        self.assertIn("geometric_score", report)
        self.assertIn("preference_penalty", report)

    def test_amount_replacement_right_anchors_target_and_preserves_suffix_outside_roi(self) -> None:
        image = Image.new("RGB", (180, 70), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        font = _font(22)
        draw.text((56, 24), "+9764", font=font, fill=(50, 50, 50))
        draw.text((124, 24), "USDT", font=font, fill=(50, 50, 50))
        roi = (20, 18, 124, 50)

        edited, report = replace_amount_in_roi(
            image,
            roi,
            source_text="+9764",
            target_text="+12749",
        )

        self.assertTrue(report["pass"])
        self.assertEqual(report["alignment"], "right_anchor_preserve_suffix")
        self.assertLessEqual(report["target_box"][2], report["source_box"][2])
        self.assertGreaterEqual(report["target_box"][0], roi[0])
        self.assertEqual(report["hard_check"]["outside_roi_changed_pixels"], 0)

        original_arr = np.array(image)
        edited_arr = np.array(edited)
        suffix_box = (124, 18, 178, 50)
        sx1, sy1, sx2, sy2 = suffix_box
        self.assertTrue(np.array_equal(original_arr[sy1:sy2, sx1:sx2], edited_arr[sy1:sy2, sx1:sx2]))


if __name__ == "__main__":
    unittest.main()
