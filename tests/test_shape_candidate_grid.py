from __future__ import annotations

import unittest
import inspect

from PIL import Image

from roi_image_edit.iterative_pipeline import (
    CandidateParams,
    RenderPlan,
    TextRun,
    generate_candidates,
    source_slot_for_target_index,
)
from roi_image_edit.local_validation import stroke_weight_fit_score
import roi_image_edit.region_processing as region_processing
from roi_image_edit.region_processing import (
    cjk_longer_form_first_batch_candidates,
    cjk_longer_form_first_batch_enabled,
    longer_replacement_soft_scan_candidates,
    select_vision_rendered_candidates,
)
import roi_image_edit.processing_service as processing_service
from roi_image_edit.revision_solver import (
    TEXT_SHAPE_GRID_ALLOWED_DELTA_KEYS,
    TEXT_SHAPE_GRID_BLOCKED_DELTA_KEYS,
    TEXT_SHAPE_PRUNE_REASON_CATEGORIES,
    text_shape_reset_candidate_grid,
    text_shape_reset_candidates,
)


def params() -> CandidateParams:
    return CandidateParams(
        candidate_id="base",
        font_name="BaseFont",
        font_path="/tmp/base.ttf",
        font_size=18,
        opacity=0.83,
        blur=0.27,
        stroke_opacity=0.01,
        ink_gain=0.02,
        alpha_contrast=0.03,
        core_ink_gain=0.04,
        core_darken_strength=0.05,
        text_dx=0,
        text_dy=0,
        char_offsets=((0, 0), (0, 0)),
        mask_threshold=177,
        mask_dilate_iterations=2,
        inpaint_radius=3,
        photo_warp=0.07,
        edge_breakup=0.011,
        photo_noise=0.021,
        jpeg_quality=91,
    )


def plan() -> RenderPlan:
    return RenderPlan(
        target_text="丙乙",
        source_text="甲乙",
        search_roi=(0, 0, 80, 44),
        target_roi=(8, 8, 50, 38),
        slot_boxes=(TextRun(10, 10, 22, 30, 120), TextRun(28, 10, 40, 30, 120)),
        protected_boxes=((54, 8, 72, 32),),
        source_reference_box=(10, 10, 40, 30),
        style_reference_box=None,
        style_reference_text=None,
        draw_mode="auto",
        placement_strategy="center_primary",
        placement_strategy_reason="test",
        slot_quality_report={"pass": True},
    )


def font_style_reference() -> dict:
    return {
        "ranked_fonts": [
            {"font_name": "Songti", "font_path": "/tmp/songti.ttf", "font_size": 18},
            {"font_name": "GBSN", "font_path": "/tmp/gbsn.ttf", "font_size": 18},
            {"font_name": "SimSun", "font_path": "/tmp/simsun.ttf", "font_size": 18},
            {"font_name": "FangSong", "font_path": "/tmp/fangsong.ttf", "font_size": 18},
        ],
    }


def text_shape_report() -> dict:
    return {
        "pass": True,
        "pipeline_profile": "photo_scan",
        "strict_gate": {
            "issues": [{"type": "font_style_score_too_high"}],
        },
        "local_stroke_body_issues": [{"type": "stroke_body_too_thin"}],
    }


def longer_report(
    *,
    ratio: float,
    direction: str,
    stage_pass: bool,
    blocking_stage: str | None,
) -> dict:
    return {
        "pass": True,
        "strict_gate": {"pass": True},
        "roi_plan": {"source_slot_count": 2, "target_slot_count": 3},
        "stage_gate": {
            "pass": stage_pass,
            "blocking_stage": blocking_stage,
            "stages": [
                {
                    "id": "text_shape",
                    "pass": stage_pass or blocking_stage != "text_shape",
                    "issues": [] if stage_pass else [{"type": "changed_char_alpha_stroke_body_too_thin"}],
                }
            ],
        },
        "stroke_body_shape_metrics": {
            "enabled": True,
            "per_char": [
                {
                    "index": 0,
                    "source_char": "甲",
                    "target_char": "丙",
                    "changed": True,
                    "body_area_ratio": ratio,
                    "stroke_weight_direction": direction,
                }
            ],
            "issues": [],
        },
    }


class ShapeCandidateGridTest(unittest.TestCase):
    def test_shorter_replacement_grid_includes_light_no_core_candidates(self) -> None:
        source = inspect.getsource(region_processing.process_region)

        self.assertIn("(0.50, 0.58, 0.00, 0.00, 0.00, 0.00, 0.00", source)
        self.assertIn("(0.50, 0.62, 0.00, 0.00, 0.00, 0.00, 0.00", source)
        self.assertIn("(0.50, 0.78, 0.00, 0.00, 0.00, 0.00, 0.00", source)
        self.assertIn("(0.56, 0.72, 0.00, 0.00, 0.00, 0.00, 0.00", source)
        self.assertIn("(0.58, 0.65, 0.00, 0.00, 0.00, 0.04, 0.00", source)

    def test_longer_replacement_soft_scan_grid_preserves_geometry_with_low_core_ink(self) -> None:
        base = params()
        candidates = longer_replacement_soft_scan_candidates(
            base,
            font_candidates=[("Songti", "/tmp/songti.ttf")],
            font_style_reference=font_style_reference(),
            max_font_size=24,
        )

        self.assertTrue(
            any(
                candidate.font_name == "Songti"
                and candidate.font_size == 17
                and candidate.opacity == 0.64
                and candidate.blur == 0.32
                and candidate.alpha_contrast == 0.40
                and candidate.ink_gain == 0.0
                and candidate.core_ink_gain == 0.0
                and candidate.core_darken_strength == 0.0
                and candidate.char_offsets == base.char_offsets
                for candidate in candidates
            )
        )
        self.assertTrue(
            any(
                candidate.font_name == "Songti"
                and candidate.font_size == 17
                and candidate.opacity == 0.66
                and candidate.blur == 0.44
                and candidate.alpha_contrast == 0.25
                and candidate.ink_gain == 0.0
                and candidate.core_ink_gain == 0.0
                and candidate.core_darken_strength == 0.0
                and candidate.char_offsets == base.char_offsets
                for candidate in candidates
            )
        )
        self.assertTrue(
            any(
                candidate.font_name == "Songti"
                and candidate.font_size == 17
                and candidate.opacity == 0.60
                and candidate.blur == 0.55
                and candidate.alpha_contrast == 0.0
                and candidate.ink_gain == 0.0
                and candidate.core_ink_gain == 0.0
                and candidate.core_darken_strength == 0.0
                and candidate.char_offsets == base.char_offsets
                for candidate in candidates
            )
        )
        self.assertTrue(
            any(
                candidate.font_name == "Songti"
                and candidate.font_size == 17
                and candidate.opacity == 0.78
                and candidate.blur == 0.75
                and candidate.alpha_contrast == 0.0
                and candidate.ink_gain == 0.0
                and candidate.core_ink_gain == 0.0
                and candidate.core_darken_strength == 0.0
                and candidate.char_offsets == base.char_offsets
                for candidate in candidates
            )
        )

    def test_cjk_longer_form_first_batch_uses_bounded_low_core_seed_rules(self) -> None:
        base = params()

        self.assertTrue(
            cjk_longer_form_first_batch_enabled(
                {
                    "image_type": "photo_document",
                    "scenario": "form_field_value_replace",
                    "script": "cjk",
                    "length_change": "longer",
                    "class_key": "photo_document.form_field_value_replace.cjk",
                },
                {"source_slot_count": 2, "target_slot_count": 3},
            )
        )
        candidates = cjk_longer_form_first_batch_candidates(
            base,
            font_candidates=[("Songti", "/tmp/songti.ttf")],
            font_style_reference=font_style_reference(),
            max_font_size=24,
        )

        self.assertTrue(candidates)
        self.assertTrue(all(0.62 <= candidate.opacity <= 0.72 for candidate in candidates))
        self.assertTrue(all(0.28 <= candidate.blur <= 0.42 for candidate in candidates))
        self.assertTrue(all(0.06 <= candidate.alpha_contrast <= 0.22 for candidate in candidates))
        self.assertTrue(all(candidate.core_ink_gain == 0.0 for candidate in candidates))
        self.assertTrue(all(candidate.core_darken_strength == 0.0 for candidate in candidates))
        self.assertTrue(all(candidate.char_offsets == base.char_offsets for candidate in candidates))
        self.assertTrue(
            any(
                candidate.font_name == "Songti"
                and candidate.font_size == 17
                and candidate.opacity == 0.66
                and candidate.blur == 0.36
                and candidate.alpha_contrast == 0.18
                for candidate in candidates
            )
        )
        self.assertTrue(
            any(
                candidate.font_name == "Songti"
                and candidate.font_size == 20
                and candidate.opacity == 0.72
                and candidate.blur == 0.42
                and candidate.alpha_contrast == 0.06
                for candidate in candidates
            )
        )
        self.assertTrue(
            any(
                candidate.font_name == "Songti"
                and candidate.font_size == 17
                and candidate.opacity == 0.69
                and candidate.blur == 0.30
                and candidate.alpha_contrast == 0.08
                for candidate in candidates
            )
        )

    def test_cjk_longer_form_first_batch_includes_thin_font_internal_darken_pool(self) -> None:
        base = params()

        candidates = cjk_longer_form_first_batch_candidates(
            base,
            font_candidates=[("Songti", "/tmp/songti.ttf"), ("GBSN", "/tmp/gbsn.ttf")],
            font_style_reference=font_style_reference(),
            max_font_size=24,
        )

        internal_darken = [
            candidate
            for candidate in candidates
            if candidate.font_name == "GBSN"
            and candidate.stroke_opacity == 0.0
            and candidate.ink_gain == 0.0
            and candidate.core_ink_gain > 0.0
            and candidate.core_darken_strength > 0.0
        ]
        self.assertTrue(internal_darken)
        self.assertTrue(all(17 <= candidate.font_size <= 20 for candidate in internal_darken))
        self.assertTrue(all(candidate.blur <= 0.32 for candidate in internal_darken))
        self.assertTrue(all(candidate.core_ink_gain <= 0.12 for candidate in internal_darken))
        self.assertTrue(all(candidate.core_darken_strength <= 0.08 for candidate in internal_darken))

    def test_source_slot_for_added_cjk_char_reuses_source_slots_only(self) -> None:
        render_plan = RenderPlan(
            target_text="赵真真",
            source_text="陈芸",
            search_roi=(0, 0, 120, 40),
            target_roi=(10, 8, 104, 36),
            slot_boxes=(
                TextRun(10, 10, 30, 32, 120),
                TextRun(34, 10, 54, 32, 110),
                TextRun(58, 10, 78, 32, 0),
            ),
            protected_boxes=(),
            source_reference_box=(10, 10, 54, 32),
            style_reference_box=None,
            style_reference_text=None,
            draw_mode="line_chars",
            placement_strategy="left_anchor_span",
        )

        source_slot = source_slot_for_target_index(render_plan, 2)

        self.assertEqual(source_slot, TextRun(34, 10, 54, 32, 110))

    def test_cjk_longer_form_first_batch_excludes_general_soft_scan_pool(self) -> None:
        source = inspect.getsource(region_processing.process_region)

        self.assertIn("params_list = dedupe_params(first_batch_seeds, max_candidates)", source)
        self.assertIn('"general_candidate_pool": "excluded_from_first_batch"', source)
        self.assertNotIn("dedupe_params([*first_batch_seeds, *params_list]", source)

    def test_stroke_weight_fit_prefers_natural_weight_shape_before_mid_bold(self) -> None:
        natural_weight = stroke_weight_fit_score(
            longer_report(ratio=0.62, direction="ok", stage_pass=False, blocking_stage="text_shape")
        )
        mid_bold = stroke_weight_fit_score(
            longer_report(ratio=0.72, direction="slightly_bold", stage_pass=True, blocking_stage=None)
        )
        too_thin = stroke_weight_fit_score(
            longer_report(ratio=0.57, direction="too_thin", stage_pass=False, blocking_stage="text_shape")
        )

        self.assertEqual(natural_weight["selection_bucket"], 0)
        self.assertEqual(too_thin["selection_bucket"], 1)
        self.assertEqual(mid_bold["selection_bucket"], 3)
        self.assertLess(natural_weight["score"], mid_bold["score"])

    def test_longer_vision_selection_uses_stroke_fit_before_stage_passed_bold(self) -> None:
        image = Image.new("RGB", (16, 16), (255, 255, 255))
        slight_params = params().__class__(**{**params().__dict__, "candidate_id": "slight"})
        bold_params = params().__class__(**{**params().__dict__, "candidate_id": "bold"})
        rendered = [
            (
                bold_params,
                image,
                longer_report(ratio=1.24, direction="too_bold", stage_pass=True, blocking_stage=None),
                1.0,
            ),
            (
                slight_params,
                image,
                longer_report(ratio=0.62, direction="ok", stage_pass=False, blocking_stage="text_shape"),
                10.0,
            ),
        ]

        selected = select_vision_rendered_candidates(rendered, 2)

        self.assertEqual([item[0].candidate_id for item in selected], ["slight", "bold"])

    def test_longer_vision_selection_includes_visual_arbitrable_text_shape_candidate(self) -> None:
        image = Image.new("RGB", (16, 16), (255, 255, 255))
        candidate = params().__class__(**{**params().__dict__, "candidate_id": "arbitrable"})
        report = longer_report(ratio=0.62, direction="ok", stage_pass=False, blocking_stage="text_shape")
        report["strict_gate"] = {
            "pass": False,
            "issues": [{"type": "font_family_style_score_ratio", "actual": 1.31, "limit": 1.25}],
        }
        report["stage_gate"]["stage_status"] = {
            "text_shape": {
                "id": "text_shape",
                "pass": False,
                "issues": [{"type": "font_family_style_score_ratio", "actual": 1.31, "limit": 1.25}],
            }
        }
        report["stage_gate"]["stage_status"]["text_shape"]["issues"] = [
            {"type": "font_family_style_score_ratio", "actual": 1.31, "limit": 1.25}
        ]
        rendered = [(candidate, image, report, 10.0)]

        selected = select_vision_rendered_candidates(rendered, 1)

        self.assertEqual([item[0].candidate_id for item in selected], ["arbitrable"])

    def test_cjk_longer_form_first_batch_disabled_for_non_matching_class(self) -> None:
        self.assertFalse(
            cjk_longer_form_first_batch_enabled(
                {
                    "image_type": "clean_digital",
                    "scenario": "inline_text_replace",
                    "script": "latin",
                    "length_change": "same",
                    "class_key": "clean_digital.inline_text_replace.latin",
                },
                {"source_slot_count": 2, "target_slot_count": 2},
            )
        )

    def test_initial_candidate_grid_includes_row_baseline_y_offsets(self) -> None:
        candidates = generate_candidates(
            params(),
            font_candidates=[
                ("Songti", "/tmp/songti.ttf"),
                ("GBSN", "/tmp/gbsn.ttf"),
            ],
            font_style_reference=font_style_reference(),
            font_pool_size=2,
            iteration=0,
            limit=80,
        )

        text_dy_values = {candidate.text_dy for candidate in candidates}
        self.assertIn(-2, text_dy_values)
        self.assertIn(-1, text_dy_values)
        self.assertIn(1, text_dy_values)

    def test_text_shape_grid_reports_budget_and_allowed_delta_keys(self) -> None:
        base = params()
        grid = text_shape_reset_candidate_grid(
            base,
            font_style_reference(),
            plan(),
            text_shape_report(),
            limit=48,
        )

        self.assertTrue(grid.report["enabled"])
        self.assertEqual(grid.report["stage_id"], "text_shape")
        self.assertEqual(grid.report["optimization_step"], "shape_reset")
        self.assertEqual(grid.report["candidate_count"], 48)
        self.assertEqual(len(grid.candidates), 48)
        self.assertTrue(grid.report["budget"]["within_budget"])
        self.assertGreaterEqual(grid.report["budget"]["raw_candidate_budget"], 300)
        self.assertLessEqual(grid.report["budget"]["raw_candidate_budget"], 1500)
        self.assertEqual(grid.report["budget"]["retained_count"], 48)
        self.assertGreater(grid.report["budget"]["pruned_count"], 0)
        self.assertEqual(set(grid.report["allowed_delta_keys"]), TEXT_SHAPE_GRID_ALLOWED_DELTA_KEYS)
        self.assertEqual(set(grid.report["blocked_delta_keys"]), TEXT_SHAPE_GRID_BLOCKED_DELTA_KEYS)
        self.assertEqual(grid.report["violations"], [])
        self.assertEqual(
            grid.report["axes"]["pose_shear_source"],
            "renderer_reference_slot_shear_from_source_slots_and_neighbors",
        )
        self.assertEqual(grid.report["axes"]["placement_strategy"], "center_primary")
        self.assertEqual(
            tuple(grid.report["prune_reason_contract"]["required_categories"]),
            TEXT_SHAPE_PRUNE_REASON_CATEGORIES,
        )
        self.assertEqual(
            grid.report["prune_reason_contract"]["category_sources"]["protected_distance"],
            ["protected_boxes", "target_roi", "right_boundary"],
        )

        for candidate, audit in zip(grid.candidates, grid.report["candidate_delta_audit"]):
            self.assertTrue(audit["allowed_delta_keys_only"], audit)
            self.assertFalse(audit["blocked_delta_keys"], audit)
            self.assertFalse(audit["undeclared_delta_keys"], audit)
            self.assertTrue(set(audit["reason_categories"]) <= set(TEXT_SHAPE_PRUNE_REASON_CATEGORIES))
            self.assertEqual(candidate.opacity, base.opacity)
            self.assertEqual(candidate.blur, base.blur)
            self.assertEqual(candidate.mask_threshold, base.mask_threshold)
            self.assertEqual(candidate.mask_dilate_iterations, base.mask_dilate_iterations)
            self.assertEqual(candidate.inpaint_radius, base.inpaint_radius)
            self.assertEqual(candidate.photo_warp, base.photo_warp)
            self.assertEqual(candidate.edge_breakup, base.edge_breakup)
            self.assertEqual(candidate.photo_noise, base.photo_noise)
            self.assertEqual(candidate.jpeg_quality, base.jpeg_quality)

    def test_legacy_shape_candidate_entrypoint_returns_grid_candidates(self) -> None:
        direct = text_shape_reset_candidate_grid(
            params(),
            font_style_reference(),
            plan(),
            text_shape_report(),
            limit=24,
        )
        legacy = text_shape_reset_candidates(
            params(),
            font_style_reference(),
            plan(),
            text_shape_report(),
            limit=24,
        )
        self.assertEqual(legacy, direct.candidates)

    def test_grid_disabled_when_text_shape_is_not_blocking(self) -> None:
        grid = text_shape_reset_candidate_grid(
            params(),
            font_style_reference(),
            plan(),
            {"pass": True, "pipeline_profile": "photo_scan"},
        )
        self.assertFalse(grid.report["enabled"])
        self.assertEqual(grid.report["reason"], "text_shape_not_blocking")
        self.assertEqual(grid.candidates, [])

    def test_processing_service_preserves_shape_grid_report_in_revision_rounds(self) -> None:
        source = inspect.getsource(processing_service.run_region_vision_checks)
        self.assertIn("text_shape_reset_candidate_grid", source)
        self.assertIn('"shape_candidate_grid": shape_candidate_grid.report', source)


if __name__ == "__main__":
    unittest.main()
