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


if __name__ == "__main__":
    unittest.main()
