from __future__ import annotations

import ast
import inspect
import unittest

import roi_image_edit.iterative_pipeline as iterative_pipeline


class IterativePipelineStageContextTest(unittest.TestCase):
    def test_stage_gate_is_attached_only_through_stage_context_helper(self) -> None:
        source = inspect.getsource(iterative_pipeline)
        tree = ast.parse(source)
        calls: list[tuple[str, int]] = []

        parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id == "stage_gate_for_report":
                parent = parents.get(node)
                while parent is not None and not isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    parent = parents.get(parent)
                calls.append((parent.name if isinstance(parent, ast.FunctionDef) else "<module>", node.lineno))

        self.assertEqual([name for name, _line in calls], ["attach_report_stage_context"])

    def test_vision_prompt_payloads_use_stage_context_structures(self) -> None:
        source = inspect.getsource(iterative_pipeline.run_pipeline)

        self.assertIn("vision_candidate_request_payload", source)
        self.assertIn("attach_stage_context_to_rank_report", source)
        self.assertIn('"stage_context_by_candidate": vision_hard_payload.get("stage_context_by_candidate")', source)
        self.assertIn("attach_report_stage_context(current_report, pipeline_profile)", source)
        self.assertIn('"stage_context": model_stage_context(final_crop_report, pipeline_profile)', source)
        self.assertIn("json.dumps(current_report, ensure_ascii=False, indent=2)", source)
        self.assertIn("json.dumps(hard_payload, ensure_ascii=False, indent=2)", source)


if __name__ == "__main__":
    unittest.main()
