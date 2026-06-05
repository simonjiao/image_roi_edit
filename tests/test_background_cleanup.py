from __future__ import annotations

import unittest

from PIL import Image, ImageDraw

from roi_image_edit.background_cleanup import (
    background_cleanup_stage_report,
    extra_source_cleanup_coverage_report,
    post_blend_report,
    source_slot_precleanup_report,
)
from roi_image_edit.iterative_pipeline import CandidateParams, RenderPlan, TextRun
from roi_image_edit.stages import stage_gate_for_report


def plan(
    *,
    source: str = "甲",
    target: str = "乙",
    slots: tuple[TextRun, ...] = (TextRun(10, 6, 22, 18, 120),),
    target_roi: tuple[int, int, int, int] = (8, 4, 26, 20),
    protected_boxes: tuple[tuple[int, int, int, int], ...] = (),
    slot_quality_report: dict | None = None,
) -> RenderPlan:
    return RenderPlan(
        target_text=target,
        source_text=source,
        search_roi=(0, 0, 64, 28),
        target_roi=target_roi,
        slot_boxes=slots,
        protected_boxes=protected_boxes,
        source_reference_box=target_roi,
        style_reference_box=None,
        style_reference_text=None,
        draw_mode="auto",
        placement_strategy="top_left_anchor",
        placement_strategy_reason="test",
        slot_quality_report=slot_quality_report or {"pass": True},
    )


def original_with_old_text() -> Image.Image:
    image = Image.new("RGB", (64, 28), (220, 220, 220))
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 8, 19, 16), fill=(60, 60, 60))
    draw.rectangle((11, 7, 20, 17), outline=(140, 140, 140))
    return image


class BackgroundCleanupTest(unittest.TestCase):
    def test_source_slot_precleanup_removes_old_core_and_gray_edge(self) -> None:
        original = original_with_old_text()
        cleaned = Image.new("RGB", original.size, (220, 220, 220))
        report = source_slot_precleanup_report(original, cleaned, plan())
        self.assertTrue(report["enabled"])
        self.assertEqual(report["mask_scope"], "source_slots_excluding_new_text_alpha")
        self.assertTrue(report["pass"])
        self.assertEqual(report["issues"], [])
        item = report["per_slot"][0]
        self.assertGreater(item["old_core_pixels"], 0)
        self.assertGreater(item["old_gray_edge_pixels"], 0)
        self.assertEqual(item["candidate_core_residual_pixels"], 0)
        self.assertEqual(item["candidate_gray_residual_pixels"], 0)

    def test_source_slot_precleanup_residual_blocks_background_cleanup_stage(self) -> None:
        original = original_with_old_text()
        report = source_slot_precleanup_report(original, original.copy(), plan())
        issue_types = {issue["type"] for issue in report["issues"]}
        self.assertFalse(report["pass"])
        self.assertIn("source_slot_core_residue", issue_types)
        self.assertIn("source_slot_gray_edge_residue", issue_types)
        gate = stage_gate_for_report(
            {
                "pass": True,
                "strict_gate": {"issues": report["issues"]},
                "local_background_texture_issues": [],
            },
            "photo_scan",
        )
        self.assertEqual(gate["blocking_stage"], "background_cleanup")

    def test_source_slot_precleanup_continues_when_replacement_font_is_unavailable(self) -> None:
        original = original_with_old_text()
        cleaned = Image.new("RGB", original.size, (220, 220, 220))
        report = source_slot_precleanup_report(
            original,
            cleaned,
            plan(),
            CandidateParams(
                candidate_id="bad_font",
                font_name="missing",
                font_path="/tmp/does-not-exist.ttf",
                font_size=16,
                opacity=0.8,
                blur=0.0,
            ),
        )
        self.assertTrue(report["enabled"])
        self.assertFalse(report["replacement_alpha"]["available"])
        self.assertEqual(report["replacement_alpha"]["reason"], "replacement_layer_font_unavailable")
        self.assertTrue(report["pass"])

    def test_shorter_replacement_requires_extra_source_cleanup_coverage(self) -> None:
        slots = (
            TextRun(8, 6, 18, 18, 100),
            TextRun(22, 6, 32, 18, 100),
            TextRun(36, 6, 46, 18, 100),
        )
        report = extra_source_cleanup_coverage_report(
            plan(
                source="甲乙丙",
                target="丁戊",
                slots=slots,
                target_roi=(6, 4, 56, 22),
                slot_quality_report={
                    "pass": True,
                    "length_change_report": {
                        "cleanup_mask_report": {
                            "enabled": True,
                            "boxes": [[36, 6, 56, 22]],
                        }
                    },
                },
            )
        )
        self.assertTrue(report["enabled"])
        self.assertEqual(report["expected_extra_slots"], 1)
        self.assertTrue(report["covered"])
        self.assertTrue(report["pass"])
        self.assertTrue(report["extra_source_cleanup_boxes"])
        self.assertEqual(report["slot_quality_cleanup_mask_boxes"], [[36, 6, 56, 22]])

    def test_post_blend_scope_stays_inside_target_roi_and_avoids_protected_text(self) -> None:
        report = post_blend_report(
            plan(protected_boxes=((40, 4, 52, 20),)),
            {
                "enabled": True,
                "target_roi": [8, 4, 26, 20],
                "new_reference_mean_delta": 1.0,
                "std_ratio": 0.9,
                "residual_ratio": 0.95,
                "white_ghost_probe": {},
                "trailing_cleanup_patch": {},
            },
        )
        self.assertTrue(report["pass"])
        self.assertEqual(report["scope"]["scope_box"], [8, 4, 26, 20])
        self.assertEqual(report["scope"]["outside_target_roi_pixels"], 0)
        self.assertEqual(report["scope"]["protected_overlap_pixels"], 0)

    def test_post_blend_reports_patch_white_dark_smooth_texture_and_roi_edge_axes(self) -> None:
        report = post_blend_report(
            plan(),
            {
                "enabled": True,
                "target_roi": [8, 4, 26, 20],
                "new_reference_mean_delta": 16.0,
                "std_ratio": 0.20,
                "residual_ratio": 0.20,
                "white_ghost_probe": {
                    "bright_over_background_p95_ratio": 0.20,
                    "dark_under_background_p10_ratio": 0.35,
                },
                "trailing_cleanup_patch": {
                    "residual_ratio": 0.20,
                },
            },
        )
        axes = report["artifact_axes"]
        self.assertEqual(
            set(axes),
            {"patch_visible", "white_ghost", "dark_shadow", "smooth_smear", "texture_break", "roi_edge_seam"},
        )
        issue_types = {issue["type"] for issue in report["issues"]}
        self.assertIn("post_blend_patch_visible", issue_types)
        self.assertIn("post_blend_white_ghost", issue_types)
        self.assertIn("post_blend_dark_shadow", issue_types)
        self.assertIn("post_blend_smooth_smear", issue_types)
        self.assertIn("post_blend_texture_break", issue_types)
        self.assertIn("post_blend_roi_edge_seam", issue_types)

    def test_pre_cleanup_failure_takes_priority_over_post_blend_naturalness(self) -> None:
        stage_report = background_cleanup_stage_report(
            {
                "pass": False,
                "issues": [{"type": "source_slot_core_residue"}],
            },
            {
                "pass": True,
                "issues": [],
            },
        )
        self.assertFalse(stage_report["pass"])
        self.assertEqual(stage_report["priority_order"], ["pre_cleanup", "post_blend"])
        self.assertEqual(stage_report["blocking_step"], "pre_cleanup")
        self.assertEqual(stage_report["blocking_reason"], "pre_cleanup_failed")
        self.assertFalse(stage_report["post_blend_can_deliver"])


if __name__ == "__main__":
    unittest.main()
