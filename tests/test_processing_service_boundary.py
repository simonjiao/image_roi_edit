from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ProcessingServiceBoundaryTest(unittest.TestCase):
    def test_processing_service_keeps_only_payload_orchestration_defs(self) -> None:
        path = ROOT / "src" / "roi_image_edit" / "processing_service.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        function_defs = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        self.assertEqual(function_defs, {"process_payload", "emit", "region_progress"})

    def test_processing_service_imports_region_core_instead_of_defining_it(self) -> None:
        path = ROOT / "src" / "roi_image_edit" / "processing_service.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        region_imports: set[str] = set()
        forbidden_modules = {
            "roi_image_edit.local_validation",
            "roi_image_edit.model_suggestions",
            "roi_image_edit.revision_selector",
            "roi_image_edit.revision_solver",
            "roi_image_edit.stage_patchers",
            "roi_image_edit.stages",
        }
        forbidden_imports: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module == "roi_image_edit.region_processing":
                region_imports.update(alias.name for alias in node.names)
            if node.module in forbidden_modules:
                forbidden_imports.append(f"{node.lineno}: {node.module}")

        self.assertIn("process_region", region_imports)
        self.assertIn("run_region_vision_checks", region_imports)
        self.assertIn("prior_stage_regression_report", region_imports)
        self.assertEqual(forbidden_imports, [])

    def test_processing_service_does_not_call_image_candidate_or_revision_core(self) -> None:
        path = ROOT / "src" / "roi_image_edit" / "processing_service.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        forbidden_calls = {
            "apply_local_acceptance_gate",
            "build_region_plan",
            "candidate_report",
            "dispatch_revision_patches",
            "final_font_revision_candidates",
            "find_font_candidates",
            "generate_candidates",
            "ink_gray_candidate_grid",
            "photo_texture_candidate_grid",
            "region_candidate_score",
            "render_candidate",
            "revision_selection_score",
            "stage_gate_for_report",
            "text_shape_reset_candidate_grid",
        }
        violations: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                call_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                call_name = node.func.attr
            else:
                continue
            if call_name in forbidden_calls:
                violations.append(f"{node.lineno}: {call_name}")

        self.assertEqual(violations, [])

    def test_region_processing_is_the_region_core_owner(self) -> None:
        path = ROOT / "src" / "roi_image_edit" / "region_processing.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        function_defs = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        self.assertIn("process_region", function_defs)
        self.assertIn("run_region_vision_checks", function_defs)
        self.assertIn("save_stage_candidate_evidence", function_defs)
        self.assertIn("prior_stage_regression_report", function_defs)


if __name__ == "__main__":
    unittest.main()
