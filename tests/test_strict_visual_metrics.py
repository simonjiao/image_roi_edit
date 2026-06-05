from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from roi_image_edit.iterative_pipeline import strict_visual_metrics


class StrictVisualMetricsTest(unittest.TestCase):
    def test_ink_gray_report_records_core_dark_body_and_outer_gray_bands(self) -> None:
        original = Image.fromarray(np.array([[30, 60, 100, 140, 200]], dtype=np.uint8))
        candidate = Image.fromarray(np.array([[30, 40, 100, 150, 200]], dtype=np.uint8))

        bands = strict_visual_metrics(original, candidate, (0, 0, 5, 1))["bands"]

        self.assertEqual(bands["old_lt55_pixels"], 1)
        self.assertEqual(bands["new_lt55_pixels"], 2)
        self.assertEqual(bands["lt55_delta"], 1)
        self.assertEqual(bands["old_lt70_pixels"], 2)
        self.assertEqual(bands["new_lt70_pixels"], 2)
        self.assertEqual(bands["lt70_delta"], 0)
        self.assertEqual(bands["old_70_120_pixels"], 1)
        self.assertEqual(bands["new_70_120_pixels"], 1)
        self.assertEqual(bands["band_70_120_delta"], 0)
        self.assertEqual(bands["old_120_165_pixels"], 1)
        self.assertEqual(bands["new_120_165_pixels"], 1)
        self.assertEqual(bands["band_120_165_delta"], 0)


if __name__ == "__main__":
    unittest.main()
