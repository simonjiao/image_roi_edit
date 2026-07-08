from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from roi_image_edit.iterative_pipeline import RenderPlan, TextRun
from roi_image_edit.region_processing import process_region
from roi_image_edit.source_glyph_reuse import source_glyph_reuse_candidate


def number_image() -> Image.Image:
    image = Image.new("RGB", (120, 48), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    # source text slots for "504.1". The third glyph is the reusable "4".
    draw.rectangle([20, 10, 30, 30], fill=(90, 90, 90))
    draw.rectangle([36, 10, 49, 30], outline=(90, 90, 90), width=3)
    draw.rectangle([56, 10, 69, 30], outline=(90, 90, 90), width=3)
    draw.rectangle([62, 10, 65, 30], fill=(90, 90, 90))
    draw.rectangle([75, 27, 78, 30], fill=(90, 90, 90))
    draw.rectangle([84, 10, 88, 30], fill=(90, 90, 90))
    return image


def failed_numeric_plan() -> RenderPlan:
    return RenderPlan(
        target_text="404.1",
        source_text="504.1",
        search_roi=(12, 4, 96, 38),
        target_roi=(18, 8, 92, 34),
        slot_boxes=(TextRun(20, 10, 88, 30, 200),),
        protected_boxes=(),
        source_reference_box=(18, 8, 92, 34),
        style_reference_box=None,
        style_reference_text=None,
        draw_mode="auto",
        slot_quality_report={
            "pass": False,
            "issues": [{"type": "slot_count_too_low", "expected": 5, "actual": 1}],
        },
    )


class SourceGlyphReuseTest(unittest.TestCase):
    def test_reuses_existing_target_glyph_without_touching_unchanged_tail(self) -> None:
        original = number_image()
        candidate, report = source_glyph_reuse_candidate(original, failed_numeric_plan())

        self.assertIsNotNone(candidate, report)
        self.assertTrue(report["pass"])
        self.assertEqual(report["strategy"], "source_glyph_reuse")
        self.assertEqual(report["changed_indices"], [0])
        self.assertEqual(report["replacements"][0]["reference_index"], 2)
        self.assertEqual(report["unexpected_changed_pixels"], 0)

        diff = np.any(np.array(original) != np.array(candidate), axis=2)
        self.assertGreater(int(diff.sum()), 0)
        self.assertEqual(int(diff[:, 34:].sum()), 0)

    def test_process_region_accepts_source_glyph_reuse_before_candidate_grid(self) -> None:
        original = number_image()
        plan = failed_numeric_plan()
        events: list[tuple[str, dict]] = []
        with tempfile.TemporaryDirectory() as tmp:
            with patch("roi_image_edit.region_processing.build_region_plan", return_value=plan):
                result, display, candidates, summary, accepted = process_region(
                    original,
                    plan.search_roi,
                    source_text="504.1",
                    target_text="404.1",
                    run_dir=Path(tmp),
                    region_id="r1",
                    vision_client=object(),
                    prompts=("master", "rank", "final"),
                    max_candidates=20,
                    vision_candidate_limit=3,
                    max_revision_rounds=2,
                    pipeline_profile="photo_scan",
                    progress=lambda event, payload: events.append((event, payload)),
                )

            self.assertTrue(accepted)
            self.assertTrue(summary["accepted"])
            self.assertTrue(summary["applied"])
            self.assertEqual(summary["trace"]["final_candidate_id"], "source_glyph_reuse")
            self.assertEqual(summary["vision"]["reason"], "source_glyph_reuse_local_strategy")
            self.assertEqual(candidates[0]["kind"], "source_glyph_reuse")
            self.assertEqual(events[0][0], "source_glyph_reuse_applied")
            self.assertFalse(np.array_equal(np.array(original), np.array(result)))
            self.assertEqual(result.size, original.size)
            self.assertEqual(display.size, original.size)

            region_dir = Path(tmp) / "regions" / "r1"
            self.assertTrue((region_dir / "source_glyph_reuse_report.json").exists())
            self.assertTrue((region_dir / "source_glyph_reuse_candidate.png").exists())


if __name__ == "__main__":
    unittest.main()
