from __future__ import annotations

import ast
import inspect
from pathlib import Path
import unittest

from roi_image_edit.iterative_pipeline import CandidateParams, RenderPlan, TextRun
import roi_image_edit.revision_selector as revision_selector
import roi_image_edit.revision_solver as revision_solver
from roi_image_edit.revision_solver import (
    TEXT_SHAPE_GRID_BLOCKED_DELTA_KEYS,
    final_font_revision_candidates,
    params_delta_keys,
)


class RevisionSelectorBoundaryTest(unittest.TestCase):
    def test_selector_functions_are_not_defined_in_revision_solver(self) -> None:
        src_root = Path(__file__).resolve().parents[1] / "src" / "roi_image_edit"
        solver_tree = ast.parse((src_root / "revision_solver.py").read_text(encoding="utf-8"))
        selector_tree = ast.parse((src_root / "revision_selector.py").read_text(encoding="utf-8"))
        moved_functions = {
            "final_acceptance_delivers",
            "revision_selection_score",
            "constrained_revision_params",
        }

        solver_defs = {node.name for node in ast.walk(solver_tree) if isinstance(node, ast.FunctionDef)}
        selector_defs = {node.name for node in ast.walk(selector_tree) if isinstance(node, ast.FunctionDef)}
        self.assertFalse(moved_functions & solver_defs)
        self.assertTrue(moved_functions <= selector_defs)

    def test_region_processing_imports_selection_from_revision_selector(self) -> None:
        region_path = Path(__file__).resolve().parents[1] / "src" / "roi_image_edit" / "region_processing.py"
        processing_path = Path(__file__).resolve().parents[1] / "src" / "roi_image_edit" / "processing_service.py"
        tree = ast.parse(region_path.read_text(encoding="utf-8"))
        processing_tree = ast.parse(processing_path.read_text(encoding="utf-8"))
        selector_imports: set[str] = set()
        solver_imports: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            names = {alias.name for alias in node.names}
            if node.module == "roi_image_edit.revision_selector":
                selector_imports.update(names)
            if node.module == "roi_image_edit.revision_solver":
                solver_imports.update(names)
        processing_selector_imports = [
            node.lineno
            for node in ast.walk(processing_tree)
            if isinstance(node, ast.ImportFrom) and node.module == "roi_image_edit.revision_selector"
        ]

        expected_selector = {
            "final_acceptance_delivers",
            "revision_selection_score",
            "constrained_revision_params",
        }
        self.assertTrue(expected_selector <= selector_imports)
        self.assertFalse(expected_selector & solver_imports)
        self.assertEqual(processing_selector_imports, [])

    def test_final_font_revision_delegates_to_text_shape_grid_without_mixed_tuning(self) -> None:
        source = inspect.getsource(revision_solver.final_font_revision_candidates)
        self.assertIn("text_shape_reset_candidates", source)
        self.assertNotIn("opacity=", source)
        self.assertNotIn("blur=", source)
        self.assertNotIn("photo_noise", source)
        self.assertNotIn("jpeg_quality", source)

        base = CandidateParams(
            candidate_id="base",
            font_name="base-font",
            font_path="/tmp/base-font.ttf",
            font_size=18,
            opacity=0.84,
            blur=0.12,
            stroke_opacity=0.02,
        )
        plan = RenderPlan(
            target_text="丙丁",
            source_text="甲乙",
            search_roi=(0, 0, 80, 24),
            target_roi=(10, 4, 42, 20),
            slot_boxes=(
                TextRun(x1=10, y1=4, x2=22, y2=20, area=160),
                TextRun(x1=26, y1=4, x2=38, y2=20, area=160),
            ),
            protected_boxes=(),
            source_reference_box=(10, 4, 42, 20),
            style_reference_box=None,
            style_reference_text=None,
            draw_mode="auto",
            placement_strategy="center_primary",
            placement_strategy_reason="same_length_cjk_changed_chars_use_slot_center",
            slot_quality_report={"pass": True},
        )
        candidates = final_font_revision_candidates(
            base,
            {
                "ranked_fonts": [
                    {
                        "font_name": "candidate-font",
                        "font_path": "/tmp/candidate-font.ttf",
                        "font_size": 18,
                    }
                ]
            },
            plan,
            {
                "pass": True,
                "pipeline_profile": "photo_scan",
                "strict_gate": {"issues": [{"type": "font_style_score_too_high"}]},
            },
        )

        self.assertTrue(candidates)
        for candidate in candidates:
            delta_keys = params_delta_keys(base, candidate)
            self.assertFalse(delta_keys & TEXT_SHAPE_GRID_BLOCKED_DELTA_KEYS)

    def test_revision_selector_module_exports_selection_contract(self) -> None:
        self.assertTrue(revision_selector.final_acceptance_delivers(
            {"pass": True, "acceptance_level": "pass", "final_decision": "deliver"}
        ))
        self.assertFalse(revision_selector.final_acceptance_delivers(
            {"pass": True, "acceptance_level": "pass", "final_decision": "revise"}
        ))

    def test_background_patch_constraints_keep_mask_and_inpaint_from_drifting_too_low(self) -> None:
        params = CandidateParams(
            candidate_id="patched",
            font_name="font",
            font_path="/tmp/font.ttf",
            font_size=30,
            opacity=0.66,
            blur=0.36,
            alpha_contrast=0.30,
            mask_threshold=101,
            mask_dilate_iterations=1,
            inpaint_radius=1,
            photo_noise=0.12,
            edge_breakup=0.06,
            jpeg_quality=80,
        )
        basis = CandidateParams(
            candidate_id="basis",
            font_name="font",
            font_path="/tmp/font.ttf",
            font_size=30,
            opacity=0.66,
            blur=0.36,
            alpha_contrast=0.30,
            mask_threshold=165,
            mask_dilate_iterations=2,
            inpaint_radius=3,
            photo_noise=0.018,
            edge_breakup=0.01,
            jpeg_quality=94,
        )

        constrained = revision_selector.constrained_revision_params(
            params,
            basis,
            {"visual_findings": {"background": "patch_visible"}},
            {"pass": True, "stage_gate": {"blocking_stage": None}},
            round_idx=8,
        )

        self.assertEqual(constrained.mask_threshold, 165)
        self.assertEqual(constrained.mask_dilate_iterations, 2)
        self.assertEqual(constrained.inpaint_radius, 3)
        self.assertLessEqual(constrained.photo_noise, 0.090)
        self.assertLessEqual(constrained.edge_breakup, 0.030)
        self.assertGreaterEqual(constrained.jpeg_quality, 88)

    def test_revision_selection_score_prefers_vision_target_alignment(self) -> None:
        basis = CandidateParams(
            candidate_id="basis",
            font_name="font",
            font_path="/tmp/font.ttf",
            font_size=30,
            opacity=0.66,
            blur=0.36,
            alpha_contrast=0.30,
            edge_breakup=0.010,
            photo_noise=0.020,
        )
        aligned = CandidateParams(
            candidate_id="aligned",
            font_name="font",
            font_path="/tmp/font.ttf",
            font_size=30,
            opacity=0.66,
            blur=0.46,
            alpha_contrast=0.30,
            edge_breakup=0.018,
            photo_noise=0.026,
        )
        opposite = CandidateParams(
            candidate_id="opposite",
            font_name="font",
            font_path="/tmp/font.ttf",
            font_size=30,
            opacity=0.66,
            blur=0.30,
            alpha_contrast=0.30,
            edge_breakup=0.004,
            photo_noise=0.060,
        )
        vision_target = {
            "active": True,
            "stage": "photo_texture",
            "stage_source": "vision_acceptance",
            "axes": [{"axis": "too_sharp", "stage": "photo_texture", "repeat_count": 2}],
            "axis_keys": ["too_sharp"],
            "repeated": True,
        }

        aligned_score = revision_selector.revision_selection_score(
            100.0,
            aligned,
            basis,
            {"visual_findings": {"sharpness": "too_sharp"}},
            {"pass": True, "stage_gate": {"blocking_stage": None}},
            {"pass": True, "stage_gate": {"blocking_stage": None}},
            vision_target=vision_target,
        )
        opposite_score = revision_selector.revision_selection_score(
            100.0,
            opposite,
            basis,
            {"visual_findings": {"sharpness": "too_sharp"}},
            {"pass": True, "stage_gate": {"blocking_stage": None}},
            {"pass": True, "stage_gate": {"blocking_stage": None}},
            vision_target=vision_target,
        )

        self.assertLess(aligned_score, opposite_score)

    def test_repeated_combo_target_prefers_complete_alignment_over_partial(self) -> None:
        basis = CandidateParams(
            candidate_id="basis",
            font_name="font",
            font_path="/tmp/font.ttf",
            font_size=30,
            opacity=0.66,
            blur=0.36,
            alpha_contrast=0.30,
            mask_threshold=165,
            inpaint_radius=3,
            edge_breakup=0.030,
            photo_noise=0.054,
        )
        partial = CandidateParams(
            candidate_id="partial",
            font_name="font",
            font_path="/tmp/font.ttf",
            font_size=30,
            opacity=0.66,
            blur=0.36,
            alpha_contrast=0.30,
            mask_threshold=165,
            inpaint_radius=3,
            edge_breakup=0.030,
            photo_noise=0.084,
        )
        complete = CandidateParams(
            candidate_id="complete",
            font_name="font",
            font_path="/tmp/font.ttf",
            font_size=30,
            opacity=0.66,
            blur=0.48,
            alpha_contrast=0.30,
            mask_threshold=170,
            inpaint_radius=3,
            edge_breakup=0.030,
            photo_noise=0.084,
        )
        vision_target = {
            "active": True,
            "stage": "background_cleanup",
            "stage_source": "vision_acceptance",
            "axes": [
                {"axis": "patch_visible", "stage": "background_cleanup", "repeat_count": 4},
                {"axis": "too_sharp", "stage": "photo_texture", "repeat_count": 4},
            ],
            "axis_keys": ["patch_visible", "too_sharp"],
            "repeated": True,
        }
        acceptance = {"visual_findings": {"background": "patch_visible", "sharpness": "too_sharp"}}
        report = {"pass": True, "stage_gate": {"blocking_stage": None}}

        partial_score = revision_selector.revision_selection_score(
            100.0,
            partial,
            basis,
            acceptance,
            report,
            report,
            vision_target=vision_target,
        )
        complete_score = revision_selector.revision_selection_score(
            1000.0,
            complete,
            basis,
            acceptance,
            report,
            report,
            vision_target=vision_target,
        )

        self.assertLess(complete_score, partial_score)

    def test_constrained_revision_allows_visual_photo_texture_when_local_stage_passes(self) -> None:
        basis = CandidateParams(
            candidate_id="basis",
            font_name="font",
            font_path="/tmp/font.ttf",
            font_size=30,
            opacity=0.66,
            blur=0.48,
            edge_breakup=0.03,
            photo_noise=0.054,
            jpeg_quality=90,
        )
        suggested = CandidateParams(
            candidate_id="suggested",
            font_name="font",
            font_path="/tmp/font.ttf",
            font_size=30,
            opacity=0.66,
            blur=0.62,
            edge_breakup=0.05,
            photo_noise=0.084,
            jpeg_quality=88,
        )

        constrained = revision_selector.constrained_revision_params(
            suggested,
            basis,
            {"visual_findings": {"sharpness": "too_sharp"}},
            {"pass": True, "stage_gate": {"blocking_stage": None}},
            round_idx=8,
        )

        self.assertEqual(constrained.blur, 0.62)
        self.assertEqual(constrained.edge_breakup, 0.05)
        self.assertEqual(constrained.photo_noise, 0.084)


if __name__ == "__main__":
    unittest.main()
