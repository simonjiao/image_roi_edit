from __future__ import annotations

import ast
from dataclasses import fields
import importlib
from pathlib import Path
import unittest

from roi_image_edit.stage_patchers import filter_patches_for_stage, patch_allowed_for_stage
from roi_image_edit.stage_policy import STAGE_LABELS, STAGE_ORDER, optimization_policy_audit, selected_optimization_step
from roi_image_edit.stage_profiles import stage_profile, stage_profile_choices
from roi_image_edit.stages import (
    STAGE_SPECS,
    StageResult,
    StageSpec,
    prompt_stage_context,
    stage_gate_for_report,
    stage_specs,
)


EXPECTED_STAGE_ORDER = (
    "hard_boundary",
    "text_shape",
    "ink_gray_balance",
    "photo_texture",
    "background_cleanup",
)


class StageContractsTest(unittest.TestCase):
    def test_stage_order_is_the_five_stage_contract(self) -> None:
        self.assertEqual(STAGE_ORDER, EXPECTED_STAGE_ORDER)
        self.assertEqual(tuple(STAGE_SPECS), EXPECTED_STAGE_ORDER)
        self.assertEqual(tuple(spec.id for spec in stage_specs("photo_scan")), EXPECTED_STAGE_ORDER)
        self.assertEqual(set(STAGE_LABELS), set(EXPECTED_STAGE_ORDER))

    def test_photo_texture_blocks_only_after_shape_and_ink_pass(self) -> None:
        shape_and_photo = stage_gate_for_report(
            {
                "pass": True,
                "pipeline_profile": "photo_scan",
                "font_style_gate": {"issues": [{"type": "font_style_mismatch"}]},
                "local_ink_balance_issues": [{"type": "ink_too_light"}],
                "local_photo_texture_issues": [{"type": "photo_texture_too_sharp"}],
            },
            "photo_scan",
        )
        self.assertEqual(shape_and_photo["blocking_stage"], "text_shape")
        self.assertFalse(shape_and_photo["stage_status"]["photo_texture"]["pass"])

        ink_and_photo = stage_gate_for_report(
            {
                "pass": True,
                "pipeline_profile": "photo_scan",
                "local_ink_balance_issues": [{"type": "ink_too_light"}],
                "local_photo_texture_issues": [{"type": "photo_texture_too_sharp"}],
            },
            "photo_scan",
        )
        self.assertEqual(ink_and_photo["blocking_stage"], "ink_gray_balance")
        self.assertFalse(ink_and_photo["stage_status"]["photo_texture"]["pass"])

        photo_only = stage_gate_for_report(
            {
                "pass": True,
                "pipeline_profile": "photo_scan",
                "local_photo_texture_issues": [{"type": "photo_texture_too_sharp"}],
            },
            "photo_scan",
        )
        self.assertEqual(photo_only["blocking_stage"], "photo_texture")

    def test_stage_spec_field_contract_and_reports(self) -> None:
        self.assertEqual(
            tuple(field.name for field in fields(StageSpec)),
            (
                "id",
                "display_name",
                "blocks_next",
                "detect",
                "optimization_steps",
                "allowed_patch_keys",
                "blocked_patch_keys",
            ),
        )
        for spec in stage_specs("photo_scan"):
            report = spec.as_report()
            self.assertNotIn("detect", report)
            self.assertEqual(report["id"], spec.id)
            self.assertIsInstance(report["optimization_steps"], tuple)
            self.assertIsInstance(report["allowed_patch_keys"], list)
            self.assertIsInstance(report["blocked_patch_keys"], list)

    def test_stage_result_field_contract_and_stage_context(self) -> None:
        self.assertEqual(
            tuple(field.name for field in fields(StageResult)),
            (
                "stage_id",
                "display_name",
                "passed",
                "blocks_next",
                "severity",
                "issues",
                "reason",
                "allowed_patch_keys",
                "blocked_patch_keys",
            ),
        )
        gate = stage_gate_for_report({"pass": False, "issues": [{"type": "roi_outside"}]})
        self.assertFalse(gate["pass"])
        self.assertEqual(gate["blocking_stage"], "hard_boundary")
        self.assertTrue(gate["blocking_stage_blocks_next"])
        self.assertEqual(tuple(gate["stage_status"]), EXPECTED_STAGE_ORDER)
        for stage_id in EXPECTED_STAGE_ORDER:
            stage = gate["stage_status"][stage_id]
            self.assertIn("pass", stage)
            self.assertIn("blocks_next", stage)
            self.assertIn("reason", stage)
            self.assertIn("allowed_patch_keys", stage)
            self.assertIn("blocked_patch_keys", stage)
            self.assertIsInstance(stage["pass"], bool)
            self.assertIsInstance(stage["blocks_next"], bool)
            self.assertIsInstance(stage["allowed_patch_keys"], list)
            self.assertIsInstance(stage["blocked_patch_keys"], list)
        hard_boundary = gate["stage_status"]["hard_boundary"]
        self.assertFalse(hard_boundary["pass"])
        self.assertTrue(hard_boundary["blocks_next"])
        self.assertEqual(hard_boundary["reason"], "roi_outside")
        self.assertEqual(hard_boundary["allowed_patch_keys"], [])
        self.assertIn("font_size_delta", hard_boundary["blocked_patch_keys"])

        context = prompt_stage_context({"pass": False, "issues": [{"type": "roi_outside"}]})
        self.assertEqual(context["stage_order"], list(EXPECTED_STAGE_ORDER))
        self.assertEqual(context["blocking_stage"], "hard_boundary")
        self.assertTrue(context["blocking_stage_blocks_next"])
        self.assertEqual(context["allowed_patch_keys"], [])
        self.assertIn("font_size_delta", context["blocked_patch_keys"])

    def test_stage_contract_entrypoints_are_only_defined_in_stages_module(self) -> None:
        src_root = Path(__file__).resolve().parents[1] / "src" / "roi_image_edit"
        definitions = {
            "StageSpec": [],
            "StageResult": [],
            "STAGE_SPECS": [],
            "stage_gate_for_report": [],
            "prompt_stage_context": [],
        }

        for path in src_root.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    if node.name in definitions:
                        definitions[node.name].append(path.name)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in definitions:
                            definitions[target.id].append(path.name)

        self.assertEqual(
            definitions,
            {
                "StageSpec": ["stages.py"],
                "StageResult": ["stages.py"],
                "STAGE_SPECS": ["stages.py"],
                "stage_gate_for_report": ["stages.py"],
                "prompt_stage_context": ["stages.py"],
            },
        )

    def test_stage_gate_is_not_imported_from_local_validation(self) -> None:
        src_root = Path(__file__).resolve().parents[1] / "src" / "roi_image_edit"
        bad_imports: list[str] = []
        for path in src_root.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                if node.module != "roi_image_edit.local_validation":
                    continue
                for alias in node.names:
                    if alias.name == "stage_gate_for_report":
                        bad_imports.append(f"{path.name}:{node.lineno}")

        self.assertEqual(bad_imports, [])

    def test_runtime_stage_modules_import_cleanly(self) -> None:
        for module_name in (
            "roi_image_edit.processing_service",
            "roi_image_edit.revision_solver",
            "roi_image_edit.stage_patchers",
        ):
            with self.subTest(module_name=module_name):
                importlib.import_module(module_name)

    def test_profile_contracts_cover_current_profiles(self) -> None:
        self.assertEqual(
            stage_profile_choices(),
            ("clean_digital", "low_res_thumbnail", "manual_roi_quick", "photo_scan"),
        )
        photo_scan = stage_profile("photo_scan")
        self.assertEqual(photo_scan.stage_order, EXPECTED_STAGE_ORDER)
        self.assertEqual(photo_scan.enabled_stage_ids, frozenset(EXPECTED_STAGE_ORDER))
        self.assertTrue(photo_scan.enable_pose)
        self.assertTrue(photo_scan.enable_photo_texture)
        clean_digital = stage_profile("clean_digital")
        self.assertFalse(clean_digital.enable_photo_texture)
        self.assertFalse(clean_digital.enable_photo_warp)
        self.assertNotIn("photo_texture", clean_digital.enabled_stage_ids)
        low_res = stage_profile("low_res_thumbnail")
        self.assertEqual(low_res.vision_context_scale, "magnified")
        self.assertIn("stroke_body_weight", low_res.shape_priority)
        manual_quick = stage_profile("manual_roi_quick")
        self.assertTrue(manual_quick.manual_roi)
        self.assertFalse(manual_quick.enable_photo_texture)
        self.assertEqual(manual_quick.revision_complexity, "minimal")
        self.assertEqual(
            manual_quick.enabled_stage_ids,
            frozenset(("hard_boundary", "text_shape", "ink_gray_balance")),
        )

    def test_text_shape_stage_rejects_photo_or_background_primary_patches(self) -> None:
        accepted, rejected = filter_patches_for_stage(
            [
                {"font_size_delta": 1},
                {"text_dx_delta": -1},
                {"blur_delta": 0.1},
                {"photo_noise_delta": 0.03},
                {"jpeg_quality_delta": -4},
                {"mask_threshold_delta": 3},
            ],
            "text_shape",
            limit=10,
        )
        self.assertEqual(accepted, [{"font_size_delta": 1}, {"text_dx_delta": -1}])
        self.assertEqual(len(rejected), 4)
        self.assertTrue(all(not item["allowed"] for item in rejected))

        blur_audit = patch_allowed_for_stage({"blur_delta": 0.1}, "text_shape")
        self.assertFalse(blur_audit["allowed"])
        self.assertEqual(blur_audit["optimization_steps"], ["photo_texture"])

        background_audit = patch_allowed_for_stage({"mask_threshold_delta": 3}, "text_shape")
        self.assertFalse(background_audit["allowed"])
        self.assertEqual(background_audit["optimization_steps"], ["background_cleanup"])

    def test_font_structure_failure_blocks_ink_gray_primary_patches(self) -> None:
        gate = stage_gate_for_report(
            {
                "pass": True,
                "pipeline_profile": "photo_scan",
                "font_style_gate": {"issues": [{"type": "font_style_mismatch"}]},
                "local_ink_balance_issues": [{"type": "ink_too_light"}],
            },
            "photo_scan",
        )
        self.assertEqual(gate["blocking_stage"], "text_shape")
        self.assertFalse(gate["stage_status"]["ink_gray_balance"]["pass"])

        accepted, rejected = filter_patches_for_stage(
            [
                {"opacity_delta": 0.03},
                {"font_size_delta": 1},
            ],
            "text_shape",
            limit=10,
        )
        self.assertEqual(accepted, [{"font_size_delta": 1}])
        self.assertEqual(rejected[0]["optimization_steps"], ["ink_gray_balance"])
        self.assertFalse(rejected[0]["allowed"])

    def test_stroke_body_failure_blocks_gray_cleanup_and_photo_noise(self) -> None:
        gate = stage_gate_for_report(
            {
                "pass": True,
                "pipeline_profile": "photo_scan",
                "strict_gate": {
                    "issues": [
                        {
                            "type": "ink_area_ratio_too_low",
                            "char_index": 0,
                        }
                    ]
                },
                "local_photo_texture_issues": [{"type": "photo_texture_too_clean"}],
            },
            "photo_scan",
        )
        self.assertEqual(gate["blocking_stage"], "text_shape")
        self.assertFalse(gate["stage_status"]["photo_texture"]["pass"])

        accepted, rejected = filter_patches_for_stage(
            [
                {"stroke_opacity_delta": 0.04},
                {"mask_threshold_delta": -3},
                {"photo_noise_delta": 0.02},
            ],
            "text_shape",
            limit=10,
        )
        self.assertEqual(accepted, [{"stroke_opacity_delta": 0.04}])
        rejected_steps = [item["optimization_steps"] for item in rejected]
        self.assertIn(["background_cleanup"], rejected_steps)
        self.assertIn(["photo_texture"], rejected_steps)

    def test_ink_coupled_text_shape_issues_defer_to_ink_when_black_core_is_excessive(self) -> None:
        report = {
            "pass": True,
            "pipeline_profile": "photo_scan",
            "strict_gate": {
                "issues": [
                    {
                        "type": "ink_area_ratio_too_low",
                        "char_index": 0,
                    },
                    {
                        "type": "char_center_dx",
                        "actual": 2.5,
                        "limit": 2.0,
                    },
                ]
            },
            "local_ink_balance_issues": [
                {
                    "type": "roi_core_too_black",
                    "lt55_delta": 742.0,
                    "limit": 113.687,
                }
            ],
        }
        gate = stage_gate_for_report(report, "photo_scan")

        self.assertEqual(gate["blocking_stage"], "ink_gray_balance")
        text_shape = gate["stage_status"]["text_shape"]
        self.assertTrue(text_shape["pass"])
        self.assertTrue(text_shape["pass_with_deferred"])
        self.assertEqual(text_shape["deferred_to_stage"], "ink_gray_balance")
        deferred_types = [issue["type"] for issue in text_shape["deferred_issues"]]
        self.assertIn("ink_area_ratio_too_low", deferred_types)
        self.assertIn("char_center_dx", deferred_types)

        prompt_context = prompt_stage_context(report, "photo_scan")
        self.assertEqual(prompt_context["blocking_stage"], "ink_gray_balance")
        prompt_text_shape = prompt_context["stage_status"]["text_shape"]
        self.assertTrue(prompt_text_shape["pass_with_deferred"])
        self.assertEqual(prompt_text_shape["deferred_to_stage"], "ink_gray_balance")

    def test_hard_text_shape_issue_still_blocks_even_when_black_core_is_excessive(self) -> None:
        gate = stage_gate_for_report(
            {
                "pass": True,
                "pipeline_profile": "photo_scan",
                "strict_gate": {
                    "issues": [
                        {
                            "type": "char_center_dx",
                            "actual": 4.5,
                            "limit": 2.0,
                        }
                    ]
                },
                "local_ink_balance_issues": [
                    {
                        "type": "roi_core_too_black",
                        "lt55_delta": 742.0,
                        "limit": 113.687,
                    }
                ],
            },
            "photo_scan",
        )

        self.assertEqual(gate["blocking_stage"], "text_shape")
        text_shape = gate["stage_status"]["text_shape"]
        self.assertFalse(text_shape["pass"])
        self.assertFalse(text_shape["pass_with_deferred"])
        self.assertEqual(text_shape["issues"][0]["type"], "char_center_dx")

    def test_stroke_body_too_bold_blocks_as_text_shape_before_ink_balance(self) -> None:
        gate = stage_gate_for_report(
            {
                "pass": True,
                "pipeline_profile": "photo_scan",
                "params": {"alpha_contrast": 0.30, "blur": 0.36},
                "roi_plan": {"source_slot_count": 2, "target_slot_count": 3},
                "strict_gate": {"text_complexity_ratio": 1.4455},
                "strict_visual_metrics": {"bands": {}},
                "char_gray_band_metrics": {
                    "enabled": True,
                    "per_char": [
                        {
                            "index": 1,
                            "source_char": "乙",
                            "target_char": "丁",
                            "old": {"lt55": 76, "lt165": 532},
                            "delta": {
                                "lt55": 20,
                                "lt165": -172,
                                "band_55_70": 83,
                                "band_70_90": -22,
                                "band_90_120": -88,
                                "band_120_165": -165,
                            },
                        }
                    ],
                },
                "local_ink_balance_issues": [
                    {
                        "type": "changed_char_deep_gray_too_dark",
                        "index": 1,
                        "actual": 83.0,
                        "limit": 48.0,
                    }
                ],
            },
            "photo_scan",
        )

        self.assertEqual(gate["blocking_stage"], "text_shape")
        text_shape = gate["stage_status"]["text_shape"]
        self.assertFalse(text_shape["pass"])
        self.assertEqual(text_shape["issues"][0]["type"], "changed_char_stroke_body_too_bold")

    def test_high_opacity_blur_body_expansion_blocks_as_text_shape_before_ink_balance(self) -> None:
        gate = stage_gate_for_report(
            {
                "pass": True,
                "pipeline_profile": "photo_scan",
                "params": {"opacity": 0.70, "blur": 0.55, "alpha_contrast": 0.0},
                "roi_plan": {"source_slot_count": 2, "target_slot_count": 3},
                "strict_gate": {"text_complexity_ratio": 1.4455},
                "strict_visual_metrics": {"bands": {}},
                "char_gray_band_metrics": {
                    "enabled": True,
                    "per_char": [
                        {
                            "index": 0,
                            "source_char": "甲",
                            "target_char": "丙",
                            "old": {"lt55": 157, "lt165": 581},
                            "delta": {
                                "lt55": -73,
                                "lt165": 82,
                                "band_55_70": 87,
                                "band_70_90": 33,
                                "band_90_120": -20,
                                "band_120_165": 55,
                            },
                        }
                    ],
                },
                "local_ink_balance_issues": [
                    {
                        "type": "changed_char_core_too_gray",
                        "index": 0,
                        "actual": 120.0,
                        "limit": 37.68,
                    }
                ],
            },
            "photo_scan",
        )

        self.assertEqual(gate["blocking_stage"], "text_shape")
        self.assertEqual(
            gate["stage_status"]["text_shape"]["issues"][0]["basis"],
            "high_opacity_high_blur_cjk_longer_body_expansion",
        )

    def test_ink_gray_stage_rejects_photo_noise_as_primary_fix(self) -> None:
        gate = stage_gate_for_report(
            {
                "pass": True,
                "pipeline_profile": "photo_scan",
                "local_ink_balance_issues": [{"type": "changed_char_neighbor_outer_gray_halo_too_high"}],
                "local_photo_texture_issues": [{"type": "photo_texture_too_clean"}],
            },
            "photo_scan",
        )
        self.assertEqual(gate["blocking_stage"], "ink_gray_balance")
        self.assertFalse(gate["stage_status"]["photo_texture"]["pass"])

        accepted, rejected = filter_patches_for_stage(
            [
                {"opacity_delta": -0.02},
                {"photo_noise_delta": 0.02},
            ],
            "ink_gray_balance",
            limit=10,
        )
        self.assertEqual(accepted, [{"opacity_delta": -0.02}])
        self.assertEqual(rejected[0]["optimization_steps"], ["photo_texture"])
        self.assertFalse(rejected[0]["allowed"])

    def test_optimization_step_is_not_a_stage_id(self) -> None:
        audit = optimization_policy_audit(
            "text_shape",
            {"stroke_opacity_delta": 0.02, "blur_delta": 0.04},
        )
        self.assertEqual(audit["stage_id"], "text_shape")
        self.assertEqual(audit["optimization_steps"], ["stroke_body_shape", "ink_gray_balance", "photo_texture"])
        self.assertEqual(audit["primary_optimization_steps"], ["stroke_body_shape"])
        self.assertEqual(audit["optimization_step"], "stroke_body_shape")
        self.assertEqual(selected_optimization_step(audit), "stroke_body_shape")
        self.assertNotEqual(audit["stage_id"], audit["optimization_step"])


if __name__ == "__main__":
    unittest.main()
