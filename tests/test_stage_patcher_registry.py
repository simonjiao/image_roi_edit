from __future__ import annotations

import ast
import inspect
from pathlib import Path
import unittest

from roi_image_edit.iterative_pipeline import CandidateParams
import roi_image_edit.stage_patchers as stage_patchers_module
from roi_image_edit.stage_patchers import (
    dispatch_revision_patches,
    patch_key_audit_for_stage_patcher,
    revision_patches_for_round,
    select_stage_patcher,
    stage_patch_filter_report,
    stage_patcher_registry_report,
    stage_patcher_specs,
)


class StagePatcherRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.params = CandidateParams(
            candidate_id="base",
            font_name="test-font",
            font_path="/tmp/test-font.ttf",
            font_size=20,
            opacity=0.82,
            blur=0.2,
        )

    def test_stage_patchers_declare_primary_allowed_and_blocked_keys(self) -> None:
        specs = stage_patcher_specs()
        self.assertEqual(
            tuple(spec.stage_id for spec in specs),
            ("text_shape", "ink_gray_balance", "photo_texture", "background_cleanup"),
        )
        for spec in specs:
            self.assertEqual(spec.primary_stage, spec.stage_id)
            self.assertTrue(callable(spec.patcher))
            self.assertFalse(spec.allowed_patch_keys & spec.blocked_patch_keys)
            self.assertFalse(spec.secondary_patch_keys & spec.blocked_patch_keys)
            self.assertTrue(spec.declared_patch_keys)
            report = spec.as_report()
            self.assertEqual(report["stage_id"], spec.stage_id)
            self.assertEqual(report["primary_stage"], spec.primary_stage)
            self.assertIsInstance(report["allowed_patch_keys"], list)
            self.assertIsInstance(report["blocked_patch_keys"], list)

    def test_revision_patch_entrypoints_are_registered_or_explicit_fallback(self) -> None:
        registered = {spec.patcher.__name__ for spec in stage_patcher_specs()}
        allowed_fallbacks = {"final_acceptance_patches"}
        allowed_dispatchers = {"dispatch_revision_patches"}
        entrypoints: set[str] = set()

        for name, value in vars(stage_patchers_module).items():
            if not callable(value) or not name.endswith("_patches"):
                continue
            signature = inspect.signature(value)
            params = tuple(signature.parameters)
            if params[:3] == ("params", "acceptance", "report"):
                entrypoints.add(name)

        self.assertEqual(entrypoints, registered | allowed_fallbacks | allowed_dispatchers)

    def test_runtime_code_uses_dispatcher_not_concrete_stage_patchers(self) -> None:
        concrete_patchers = {spec.patcher.__name__ for spec in stage_patcher_specs()}
        forbidden_calls: list[str] = []
        src_root = Path(__file__).resolve().parents[1] / "src" / "roi_image_edit"

        for path in src_root.glob("*.py"):
            if path.name == "stage_patchers.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if isinstance(node.func, ast.Name):
                    call_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    call_name = node.func.attr
                else:
                    continue
                if call_name in concrete_patchers:
                    forbidden_calls.append(f"{path.name}:{node.lineno}:{call_name}")

        self.assertEqual(forbidden_calls, [])

    def test_registry_report_is_stable_and_explicit(self) -> None:
        report = stage_patcher_registry_report()
        self.assertEqual(set(report), {"text_shape", "ink_gray_balance", "photo_texture", "background_cleanup"})
        self.assertEqual(report["text_shape"]["primary_stage"], "text_shape")
        self.assertIn("font_size_delta", report["text_shape"]["allowed_patch_keys"])
        self.assertIn("mask_threshold_delta", report["text_shape"]["blocked_patch_keys"])

    def test_stage_patcher_outputs_do_not_contain_undeclared_keys(self) -> None:
        acceptances = {
            "text_shape": {
                "visual_findings": {"stroke_weight": "too_thin"},
                "parameter_suggestions": [{"name": "font_size", "delta": 1}],
            },
            "ink_gray_balance": {
                "visual_findings": {"darkness": "too_light"},
                "parameter_suggestions": [{"name": "opacity", "delta": 0.03}],
            },
            "photo_texture": {
                "visual_findings": {"sharpness": "too_sharp"},
                "parameter_suggestions": [{"name": "blur", "delta": 0.05}],
            },
            "background_cleanup": {
                "visual_findings": {"background": "patch_visible"},
                "parameter_suggestions": [{"name": "mask_threshold", "delta": -4}],
            },
        }
        for spec in stage_patcher_specs():
            patches = spec.patcher(
                self.params,
                acceptances[spec.stage_id],
                {"pass": True, "pipeline_profile": "photo_scan"},
                rank_patch={"blur_delta": 0.1},
            )
            self.assertTrue(patches)
            for patch in patches:
                audit = patch_key_audit_for_stage_patcher(spec.stage_id, patch)
                self.assertTrue(audit["declared"], audit)

    def test_photo_texture_dispatch_rejects_ink_gray_suggestions(self) -> None:
        dispatch = dispatch_revision_patches(
            self.params,
            {
                "visual_findings": {"darkness": "too_dark", "sharpness": "too_sharp"},
                "parameter_suggestions": [
                    {"name": "opacity", "delta": -0.05},
                    {"name": "blur", "delta": 0.05},
                ],
            },
            {
                "pass": True,
                "pipeline_profile": "photo_scan",
                "local_photo_texture_issues": [{"type": "photo_texture_too_sharp"}],
            },
            rank_patch={"opacity_delta": -0.10},
        )
        self.assertEqual(dispatch["patcher_stage"], "photo_texture")
        self.assertEqual(dispatch["selection_reason"], "blocking_stage")
        self.assertIn("stage_filter_report", dispatch)
        patches = dispatch["patches"]
        self.assertTrue(patches)
        for patch in patches:
            self.assertNotIn("opacity_delta", patch)
            audit = patch_key_audit_for_stage_patcher("photo_texture", patch)
            self.assertTrue(audit["declared"], audit)

    def test_photo_texture_patcher_declares_only_small_texture_parameter_family(self) -> None:
        report = stage_patcher_registry_report()["photo_texture"]
        self.assertEqual(
            set(report["allowed_patch_keys"]),
            {
                "blur_delta",
                "alpha_contrast_delta",
                "photo_warp_delta",
                "edge_breakup_delta",
                "photo_noise_delta",
                "jpeg_quality_delta",
            },
        )
        self.assertFalse(set(report["allowed_patch_keys"]) & {"opacity_delta", "stroke_opacity_delta", "font_size_delta"})
        self.assertNotIn("alpha_contrast_delta", report["blocked_patch_keys"])
        audit = patch_key_audit_for_stage_patcher(
            "photo_texture",
            {"blur_delta": 0.04, "alpha_contrast_delta": -0.02, "photo_noise_delta": 0.012},
        )
        self.assertTrue(audit["declared"], audit)

    def test_ink_gray_dispatch_generates_opposite_directions_for_black_core_and_light_core(self) -> None:
        too_black = dispatch_revision_patches(
            self.params,
            {"visual_findings": {"darkness": "too_dark"}},
            {
                "pass": True,
                "pipeline_profile": "photo_scan",
                "local_ink_balance_issues": [{"type": "changed_char_core_too_black"}],
            },
        )
        self.assertEqual(too_black["patcher_stage"], "ink_gray_balance")
        black_patches = too_black["patches"]
        self.assertTrue(any(float(patch.get("opacity_delta") or 0.0) < 0 for patch in black_patches))
        self.assertTrue(
            any(
                float(patch.get("core_ink_gain_delta") or 0.0) < 0
                or float(patch.get("core_darken_strength_delta") or 0.0) < 0
                for patch in black_patches
            )
        )

        too_light = dispatch_revision_patches(
            self.params,
            {"visual_findings": {"darkness": "too_light"}},
            {
                "pass": True,
                "pipeline_profile": "photo_scan",
                "local_ink_balance_issues": [{"type": "core_mean_gray_too_light"}],
            },
        )
        self.assertEqual(too_light["patcher_stage"], "ink_gray_balance")
        light_patches = too_light["patches"]
        self.assertTrue(any(float(patch.get("opacity_delta") or 0.0) > 0 for patch in light_patches))
        self.assertTrue(
            any(
                float(patch.get("alpha_contrast_delta") or 0.0) > 0
                or float(patch.get("core_darken_strength_delta") or 0.0) > 0
                for patch in light_patches
            )
        )
        self.assertFalse(
            any(
                float(patch.get("opacity_delta") or 0.0) < 0
                or float(patch.get("core_ink_gain_delta") or 0.0) < 0
                or float(patch.get("core_darken_strength_delta") or 0.0) < 0
                for patch in light_patches
            )
        )

    def test_ink_gray_core_light_with_outer_halo_rejects_blur_and_recovers_core(self) -> None:
        dispatch = dispatch_revision_patches(
            self.params,
            {
                "visual_findings": {"darkness": "too_light"},
                "parameter_suggestions": [{"name": "blur", "delta": 0.08}],
            },
            {
                "pass": True,
                "pipeline_profile": "photo_scan",
                "local_ink_balance_issues": [
                    {"type": "core_mean_gray_too_light", "actual": 92, "limit": 84},
                    {
                        "type": "changed_char_neighbor_outer_gray_halo_too_high",
                        "outer_share_gap": 0.12,
                    },
                ],
            },
            rank_patch={"blur_delta": 0.08, "photo_noise_delta": 0.02},
        )

        self.assertEqual(dispatch["patcher_stage"], "ink_gray_balance")
        patches = dispatch["patches"]
        self.assertTrue(patches)
        self.assertTrue(
            any(
                float(patch.get("core_ink_gain_delta") or 0.0) > 0
                and float(patch.get("core_darken_strength_delta") or 0.0) > 0
                for patch in patches
            )
        )
        self.assertTrue(
            any(
                float(patch.get("alpha_contrast_delta") or 0.0) > 0
                or float(patch.get("stroke_opacity_delta") or 0.0) < 0
                for patch in patches
            )
        )
        for patch in patches:
            self.assertLessEqual(float(patch.get("blur_delta") or 0.0), 0.0)
            self.assertLessEqual(float(patch.get("photo_noise_delta") or 0.0), 0.0)
            self.assertLessEqual(float(patch.get("edge_breakup_delta") or 0.0), 0.0)
            self.assertGreaterEqual(float(patch.get("core_ink_gain_delta") or 0.0), 0.0)
            self.assertGreaterEqual(float(patch.get("core_darken_strength_delta") or 0.0), 0.0)
        rejected_steps = {
            tuple(item.get("optimization_steps") or [])
            for item in dispatch["stage_filter_report"]["rejected_patches"]
            if isinstance(item, dict)
        }
        self.assertIn(("photo_texture",), rejected_steps)

    def test_legacy_revision_entry_delegates_to_dispatcher(self) -> None:
        source = inspect.getsource(revision_patches_for_round)
        self.assertIn("dispatch_revision_patches", source)
        self.assertNotIn("final_acceptance_patches", source)
        self.assertNotIn("STAGE_PATCHER_SPECS", source)
        self.assertNotIn("stage_patch_filter_report", source)

        acceptance = {
            "visual_findings": {"background": "patch_visible"},
            "parameter_suggestions": [{"name": "mask_threshold", "delta": -4}],
        }
        report = {
            "pass": True,
            "pipeline_profile": "photo_scan",
            "local_background_texture_issues": [{"type": "background_patch_visible"}],
        }
        dispatch = dispatch_revision_patches(self.params, acceptance, report)
        legacy = revision_patches_for_round(self.params, acceptance, report)
        self.assertEqual(legacy, dispatch["patches"])

    def test_background_cleanup_visual_patches_are_callable(self) -> None:
        acceptance = {
            "visual_findings": {"background": "patch_visible"},
            "visual_findings_text": "背景有补丁感和涂抹残影。",
        }
        report = {
            "pass": True,
            "pipeline_profile": "photo_scan",
            "local_background_texture_issues": [{"type": "background_patch_visible"}],
        }

        dispatch = dispatch_revision_patches(self.params, acceptance, report)

        self.assertEqual(dispatch["patcher_stage"], "background_cleanup")
        self.assertTrue(dispatch["patches"])
        self.assertTrue(
            any(
                "photo_noise_delta" in patch
                or "edge_breakup_delta" in patch
                or "inpaint_radius_delta" in patch
                for patch in dispatch["patches"]
            )
        )

    def test_no_runtime_switch_points_to_old_mixed_revision_path(self) -> None:
        src_root = Path(__file__).resolve().parents[1] / "src" / "roi_image_edit"
        violations: list[str] = []
        for path in src_root.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if isinstance(node.func, ast.Name):
                    call_name = node.func.id
                    owner_name = ""
                elif isinstance(node.func, ast.Attribute):
                    call_name = node.func.attr
                    if isinstance(node.func.value, ast.Name):
                        owner_name = node.func.value.id
                    elif isinstance(node.func.value, ast.Attribute):
                        owner_name = node.func.value.attr
                    else:
                        owner_name = ""
                else:
                    continue

                if path.name != "stage_patchers.py" and call_name == "final_acceptance_patches":
                    violations.append(f"{path.name}:{node.lineno}: direct final_acceptance_patches call")

                string_args = [
                    arg.value
                    for arg in node.args
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                ]
                for value in string_args:
                    normalized = value.lower().replace("-", "_")
                    names_old_path = "legacy" in normalized or "mixed" in normalized
                    names_revision = "revision" in normalized or "patch" in normalized
                    if call_name == "add_argument" and names_old_path and names_revision:
                        violations.append(f"{path.name}:{node.lineno}: runtime revision fallback arg {value}")
                    if (
                        call_name in {"get", "getenv"}
                        and owner_name in {"environ", "os"}
                        and names_old_path
                        and names_revision
                    ):
                        violations.append(f"{path.name}:{node.lineno}: runtime revision fallback env {value}")

        self.assertEqual(violations, [])

    def test_select_stage_patcher_reports_selection_reason(self) -> None:
        report = {
            "pass": True,
            "pipeline_profile": "photo_scan",
            "local_ink_balance_issues": [{"type": "ink_too_light"}],
        }
        selection = select_stage_patcher(report, None)
        self.assertEqual(selection["blocking_stage"], "ink_gray_balance")
        self.assertEqual(selection["patcher_stage"], "ink_gray_balance")
        self.assertEqual(selection["selection_reason"], "blocking_stage")
        self.assertEqual(selection["patcher"]["primary_stage"], "ink_gray_balance")

    def test_filter_report_declares_secondary_impacts_and_rejects_cross_stage_primary(self) -> None:
        report = stage_patch_filter_report(
            [
                {"font_size_delta": 1, "blur_delta": 0.04},
                {"mask_threshold_delta": -3},
            ],
            "text_shape",
            limit=10,
        )
        self.assertEqual(report["stage_id"], "text_shape")
        self.assertEqual(report["primary_stage"], "text_shape")
        self.assertEqual(report["accepted_count"], 1)
        self.assertEqual(report["rejected_count"], 1)

        accepted = report["decisions"][0]
        self.assertEqual(accepted["decision"], "accepted")
        self.assertEqual(accepted["primary_stage"], "text_shape")
        self.assertEqual(accepted["primary_optimization_steps"], ["text_shape"])
        self.assertEqual(accepted["optimization_step"], "text_shape")
        self.assertEqual(accepted["secondary_optimization_steps"], ["photo_texture"])
        self.assertEqual(
            accepted["decision_basis"],
            "secondary effects are declared and current stage remains primary",
        )

        rejected = report["decisions"][1]
        self.assertEqual(rejected["decision"], "rejected")
        self.assertEqual(rejected["primary_stage"], "text_shape")
        self.assertEqual(rejected["optimization_steps"], ["background_cleanup"])
        self.assertIn("forbidden optimization steps", rejected["decision_basis"])

    def test_dispatch_does_not_mix_all_patch_families_when_shape_blocks(self) -> None:
        dispatch = dispatch_revision_patches(
            self.params,
            {
                "visual_findings": {
                    "font_similarity": "wrong",
                    "darkness": "too_light",
                    "sharpness": "too_sharp",
                    "background": "patch_visible",
                }
            },
            {
                "pass": True,
                "pipeline_profile": "photo_scan",
                "strict_gate": {
                    "issues": [{"type": "font_style_score_too_high"}],
                },
            },
            extra_patches=[
                {"opacity_delta": 0.06},
                {"blur_delta": 0.08},
                {"mask_threshold_delta": -4},
                {"font_size_delta": 1},
            ],
        )

        self.assertEqual(dispatch["patcher_stage"], "text_shape")
        self.assertEqual(dispatch["selection_reason"], "blocking_stage")
        self.assertTrue(dispatch["patches"])
        for patch in dispatch["patches"]:
            audit = stage_patch_filter_report([patch], "text_shape", limit=1)["decisions"][0]
            self.assertEqual(audit["decision"], "accepted")
            self.assertNotIn("ink_gray_balance", audit["primary_optimization_steps"])
            self.assertNotIn("photo_texture", audit["primary_optimization_steps"])
            self.assertNotIn("background_cleanup", audit["primary_optimization_steps"])

        rejected_steps = {
            tuple(decision.get("optimization_steps") or [])
            for decision in dispatch["stage_filter_report"]["decisions"]
            if decision.get("decision") == "rejected"
        }
        self.assertIn(("ink_gray_balance",), rejected_steps)
        self.assertIn(("photo_texture",), rejected_steps)
        self.assertIn(("background_cleanup",), rejected_steps)


if __name__ == "__main__":
    unittest.main()
