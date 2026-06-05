from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class WebBoundaryTest(unittest.TestCase):
    def test_web_app_imports_only_the_processing_service_from_core(self) -> None:
        path = ROOT / "src" / "roi_image_edit" / "web_app.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        violations: list[str] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in {"PIL", "cv2", "numpy", "openai"}:
                        violations.append(f"{node.lineno}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "roi_image_edit.processing_service":
                    imported = {alias.name for alias in node.names}
                    if imported != {"process_payload"}:
                        violations.append(f"{node.lineno}: from {module} import {sorted(imported)}")
                elif module.startswith("roi_image_edit."):
                    violations.append(f"{node.lineno}: from {module} import ...")
                elif module.split(".")[0] in {"PIL", "cv2", "numpy", "openai"}:
                    violations.append(f"{node.lineno}: from {module} import ...")

        self.assertEqual(violations, [])

    def test_web_app_does_not_call_stage_or_image_processing_entrypoints(self) -> None:
        path = ROOT / "src" / "roi_image_edit" / "web_app.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        forbidden_calls = {
            "apply_local_acceptance_gate",
            "build_region_plan",
            "candidate_report",
            "dispatch_revision_patches",
            "generate_candidates",
            "model_stage_context",
            "render_candidate",
            "run_region_vision_checks",
            "stage_gate_for_report",
            "vision_candidate_request_payload",
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


if __name__ == "__main__":
    unittest.main()
