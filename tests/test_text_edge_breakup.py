from __future__ import annotations

import unittest

import numpy as np
from PIL import Image, ImageDraw

from roi_image_edit.iterative_pipeline import apply_scan_edge_breakup


class TextEdgeBreakupTest(unittest.TestCase):
    def test_edge_breakup_thins_without_punching_white_holes(self) -> None:
        alpha = Image.new("L", (32, 24), 0)
        draw = ImageDraw.Draw(alpha)
        draw.rectangle((4, 4, 27, 19), fill=80)

        broken = apply_scan_edge_breakup(
            alpha,
            (4, 4, 28, 20),
            strength=0.60,
            quant_step=1,
            scanline_strength=0.0,
        )

        before = np.array(alpha)
        after = np.array(broken)
        original_edge = before > 0
        self.assertTrue(np.any(after[original_edge] < before[original_edge]))
        self.assertGreaterEqual(int(after[original_edge].min()), 1)
        self.assertEqual(int(after[~original_edge].max()), 0)


if __name__ == "__main__":
    unittest.main()
