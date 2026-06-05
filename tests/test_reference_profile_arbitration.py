from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from roi_image_edit.iterative_pipeline import CandidateParams, RenderPlan, TextRun
from roi_image_edit.local_validation import build_reference_profile


def params() -> CandidateParams:
    return CandidateParams(
        candidate_id="base",
        font_name="BaseFont",
        font_path="/tmp/base.ttf",
        font_size=18,
        opacity=0.82,
        blur=0.2,
        stroke_opacity=0.04,
        ink_gain=0.08,
        alpha_contrast=0.1,
        core_ink_gain=0.12,
        core_darken_strength=0.12,
        text_dx=0,
        text_dy=0,
        char_offsets=((0, 0),),
        mask_threshold=177,
        mask_dilate_iterations=2,
        inpaint_radius=3,
        photo_warp=0.04,
        edge_breakup=0.008,
        photo_noise=0.018,
        jpeg_quality=92,
    )


def plan(*, protected_boxes: tuple[tuple[int, int, int, int], ...]) -> RenderPlan:
    return RenderPlan(
        target_text="乙",
        source_text="甲",
        search_roi=(0, 0, 80, 32),
        target_roi=(10, 5, 30, 25),
        slot_boxes=(TextRun(10, 5, 30, 25, 80),),
        protected_boxes=protected_boxes,
        source_reference_box=(10, 5, 30, 25),
        style_reference_box=None,
        style_reference_text=None,
        draw_mode="auto",
        placement_strategy="top_left_anchor",
        placement_strategy_reason="test",
        slot_quality_report={"pass": True},
    )


def arbitration_image() -> Image.Image:
    arr = np.full((32, 80), 230, dtype=np.uint8)
    arr[8:18, 12:14] = 40
    arr[8:20, 42:47] = 40
    return Image.fromarray(arr, mode="L").convert("RGB")


class ReferenceProfileArbitrationTest(unittest.TestCase):
    def test_same_row_neighbor_wins_when_source_and_neighbor_ink_conflict(self) -> None:
        profile = build_reference_profile(
            arbitration_image(),
            plan(protected_boxes=((40, 5, 60, 25),)),
            params(),
        )
        dynamic_ink = profile["dynamic_ink"]
        arbitration = dynamic_ink["arbitration"]

        self.assertEqual(dynamic_ink["basis"], "source_text_region_and_same_row_neighbors")
        self.assertEqual(len(profile["same_row_neighbors"]), 1)
        self.assertTrue(arbitration["conflict_detected"])
        self.assertEqual(arbitration["selected_core_reference"], "same_row_neighbor")
        self.assertEqual(arbitration["rule"], "use_darker_of_source_and_same_row_neighbor")
        self.assertGreater(
            dynamic_ink["neighbor_core_density"],
            dynamic_ink["source_core_density"],
        )
        self.assertGreaterEqual(
            dynamic_ink["opacity_floor_for_excess_core"],
            0.68,
        )

    def test_source_only_arbitration_when_no_same_row_neighbor_exists(self) -> None:
        profile = build_reference_profile(
            arbitration_image(),
            plan(protected_boxes=()),
            params(),
        )
        dynamic_ink = profile["dynamic_ink"]
        arbitration = dynamic_ink["arbitration"]

        self.assertEqual(profile["same_row_neighbors"], [])
        self.assertIsNone(dynamic_ink["neighbor_core_density"])
        self.assertFalse(arbitration["conflict_detected"])
        self.assertEqual(arbitration["selected_core_reference"], "source")
        self.assertEqual(arbitration["rule"], "source_only_no_same_row_neighbor")


if __name__ == "__main__":
    unittest.main()
