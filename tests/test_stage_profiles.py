from __future__ import annotations

import ast
from pathlib import Path
import unittest

from roi_image_edit.cli import build_parser
from roi_image_edit.local_validation import apply_local_acceptance_gate
from roi_image_edit.run_artifacts import result_audit_payload
from roi_image_edit.stage_profiles import (
    resolve_internal_stage_profile,
    resolve_stage_profile,
    stage_profile,
    stage_profile_choices,
    stage_profile_summary,
)
from roi_image_edit.stages import stage_gate_for_report, stage_specs
from roi_image_edit.stages import prompt_stage_context


class StageProfilesTest(unittest.TestCase):
    def test_profile_entrypoints_are_only_defined_in_stage_profiles_module(self) -> None:
        src_root = Path(__file__).resolve().parents[1] / "src" / "roi_image_edit"
        definitions = {
            "StageProfile": [],
            "STAGE_PROFILES": [],
            "DEFAULT_STAGE_PROFILE_ID": [],
            "stage_profile": [],
            "stage_profile_choices": [],
            "default_stage_profile": [],
            "resolve_stage_profile": [],
            "stage_profile_summary": [],
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
                "StageProfile": ["stage_profiles.py"],
                "STAGE_PROFILES": ["stage_profiles.py"],
                "DEFAULT_STAGE_PROFILE_ID": ["stage_profiles.py"],
                "stage_profile": ["stage_profiles.py"],
                "stage_profile_choices": ["stage_profiles.py"],
                "default_stage_profile": ["stage_profiles.py"],
                "resolve_stage_profile": ["stage_profiles.py"],
                "stage_profile_summary": ["stage_profiles.py"],
            },
        )

    def test_profile_matrix_drives_stage_specs_and_gate_reports(self) -> None:
        for profile_id in stage_profile_choices():
            with self.subTest(profile_id=profile_id):
                profile = stage_profile(profile_id)
                self.assertEqual(stage_profile_summary(profile_id)["id"], profile_id)
                specs = stage_specs(profile_id)
                expected_order = tuple(stage for stage in profile.stage_order if stage in profile.enabled_stage_ids)
                self.assertEqual(tuple(spec.id for spec in specs), expected_order)
                gate = stage_gate_for_report({"pass": True, "pipeline_profile": profile_id}, profile_id)
                self.assertTrue(gate["pass"])
                self.assertEqual(gate["profile"], profile_id)
                self.assertEqual(gate["order"], [spec.id for spec in specs])
                if profile_id == "clean_digital":
                    self.assertNotIn("photo_texture", gate["order"])
                if profile_id == "manual_roi_quick":
                    self.assertEqual(
                        gate["order"],
                        ["hard_boundary", "text_shape", "ink_gray_balance"],
                    )

    def test_profile_shape_priorities_do_not_reintroduce_old_stage_ids(self) -> None:
        old_public_ids = {
            "slot_alignment",
            "font_structure",
            "pose_geometry",
            "stroke_body",
            "tone_gray",
            "edge_quality",
        }
        for profile_id in stage_profile_choices():
            with self.subTest(profile_id=profile_id):
                self.assertFalse(set(stage_profile(profile_id).shape_priority) & old_public_ids)

    def test_photo_scan_profile_enables_pose_texture_and_blocks_visual_deliver_on_shape(self) -> None:
        profile = stage_profile("photo_scan")
        self.assertTrue(profile.enable_pose)
        self.assertTrue(profile.enable_photo_texture)
        self.assertTrue(profile.enable_photo_warp)
        self.assertIn("local_pose_match", profile.shape_priority)
        self.assertIn("photo_texture", profile.enabled_stage_ids)

        report = {
            "pass": True,
            "pipeline_profile": "photo_scan",
            "strict_gate": {
                "issues": [{"type": "ink_area_ratio_too_low"}],
            },
            "local_photo_texture_issues": [{"type": "photo_texture_too_clean"}],
        }
        gate = stage_gate_for_report(report, "photo_scan")
        self.assertEqual(gate["blocking_stage"], "text_shape")
        gated = apply_local_acceptance_gate(
            {
                "pass": True,
                "acceptance_level": "pass",
                "final_decision": "deliver",
            },
            report,
        )
        self.assertFalse(gated["pass"])
        self.assertEqual(gated["final_decision"], "revise")
        self.assertEqual(gated["stage_gate"]["blocking_stage"], "text_shape")

    def test_clean_digital_profile_disables_photo_texture_and_photo_warp(self) -> None:
        profile = stage_profile("clean_digital")
        self.assertFalse(profile.enable_photo_texture)
        self.assertFalse(profile.enable_photo_warp)
        self.assertEqual(profile.edge_policy, "clean_edges_no_photo_warp")
        gate = stage_gate_for_report(
            {
                "pass": True,
                "pipeline_profile": "clean_digital",
                "local_photo_texture_issues": [{"type": "photo_texture_too_clean"}],
            },
            "clean_digital",
        )
        self.assertTrue(gate["pass"])
        self.assertNotIn("photo_texture", gate["order"])
        context = prompt_stage_context({"pass": True}, "clean_digital")
        self.assertFalse(context["profile_constraints"]["enable_photo_texture"])
        self.assertFalse(context["profile_constraints"]["enable_photo_warp"])
        self.assertEqual(
            context["profile_constraints"]["edge_policy"],
            "clean_edges_no_photo_warp",
        )

    def test_low_res_thumbnail_profile_prioritizes_shape_and_uses_magnified_context(self) -> None:
        profile = stage_profile("low_res_thumbnail")
        self.assertEqual(profile.vision_context_scale, "magnified")
        self.assertEqual(
            profile.shape_priority,
            ("font_family_similarity", "stroke_body_weight", "slot_geometry", "ink_gray_density"),
        )
        context = prompt_stage_context({"pass": True}, "low_res_thumbnail")
        self.assertEqual(context["profile_constraints"]["vision_context_scale"], "magnified")
        self.assertEqual(
            context["profile_constraints"]["shape_priority"],
            ["font_family_similarity", "stroke_body_weight", "slot_geometry", "ink_gray_density"],
        )

    def test_manual_roi_quick_profile_uses_minimal_stages_and_rejected_artifacts(self) -> None:
        profile = stage_profile("manual_roi_quick")
        self.assertTrue(profile.manual_roi)
        self.assertFalse(profile.enable_photo_texture)
        self.assertFalse(profile.enable_photo_warp)
        self.assertEqual(profile.revision_complexity, "minimal")
        self.assertTrue(profile.preserve_rejected_candidate)
        self.assertEqual(
            tuple(spec.id for spec in stage_specs("manual_roi_quick")),
            ("hard_boundary", "text_shape", "ink_gray_balance"),
        )
        gate = stage_gate_for_report(
            {
                "pass": True,
                "pipeline_profile": "manual_roi_quick",
                "local_photo_texture_issues": [{"type": "photo_texture_too_clean"}],
                "local_background_texture_issues": [{"type": "background_patch_visible"}],
            },
            "manual_roi_quick",
        )
        self.assertTrue(gate["pass"])
        self.assertNotIn("photo_texture", gate["order"])
        self.assertNotIn("background_cleanup", gate["order"])
        context = prompt_stage_context({"pass": True}, "manual_roi_quick")
        self.assertEqual(context["profile_constraints"]["revision_complexity"], "minimal")
        self.assertTrue(context["profile_constraints"]["preserve_rejected_candidate"])

    def test_explicit_profile_overrides_auto_suggestion(self) -> None:
        resolution = resolve_stage_profile("clean_digital", "photo_scan")
        self.assertEqual(resolution["id"], "clean_digital")
        self.assertEqual(resolution["source"], "explicit_request")
        self.assertEqual(resolution["requested_profile"], "clean_digital")
        self.assertEqual(resolution["suggested_profile"], "photo_scan")

    def test_internal_profile_resolution_is_preserved_in_result_audit(self) -> None:
        resolution = resolve_internal_stage_profile(
            {
                "class_key": "clean_digital.numeric_or_date_replace",
                "internal_profile": "clean_digital",
                "profile_source": "classification",
            }
        )
        audit = result_audit_payload(
            {
                "ok": True,
                "runDir": "output/web/run1",
                "profileResolution": resolution,
                "images": [],
            }
        )
        self.assertNotIn("profile", audit)
        self.assertEqual(audit["profileResolution"]["id"], "clean_digital")
        self.assertEqual(audit["profileResolution"]["source"], "classification")
        self.assertEqual(audit["profileResolution"]["profile_source"], "classification")

    def test_same_image_profile_matrix_records_stage_order_difference(self) -> None:
        def audit_for(profile_id: str) -> dict:
            gate = stage_gate_for_report({"pass": True, "pipeline_profile": profile_id}, profile_id)
            return result_audit_payload(
                {
                    "ok": True,
                    "runDir": f"output/web/{profile_id}",
                    "profileResolution": resolve_internal_stage_profile(
                        {"internal_profile": profile_id, "profile_source": "classification"}
                    ),
                    "images": [
                        {
                            "id": "same-image",
                            "ok": True,
                            "regions": [
                                {
                                    "id": "region_1",
                                    "summary": {
                                        "stage_evidence": {
                                            "stage_order": gate["order"],
                                        }
                                    },
                                }
                            ],
                            "candidates": [
                                {
                                    "id": "c1",
                                    "stage_context": {
                                        "stage_order": gate["order"],
                                        "blocking_stage": gate["blocking_stage"],
                                    },
                                }
                            ],
                        }
                    ],
                }
            )

        photo = audit_for("photo_scan")
        clean = audit_for("clean_digital")
        self.assertEqual(photo["images"][0]["id"], clean["images"][0]["id"])
        photo_order = photo["images"][0]["regions"][0]["summary"]["stage_evidence"]["stage_order"]
        clean_order = clean["images"][0]["regions"][0]["summary"]["stage_evidence"]["stage_order"]
        self.assertIn("photo_texture", photo_order)
        self.assertNotIn("photo_texture", clean_order)
        self.assertEqual(clean["profileResolution"]["id"], "clean_digital")
        self.assertEqual(clean["profileResolution"]["source"], "classification")
        self.assertEqual(
            clean["images"][0]["candidates"][0]["stage_context"]["stage_order"],
            clean_order,
        )

    def test_auto_suggestion_used_only_without_explicit_profile(self) -> None:
        resolution = resolve_stage_profile(None, "low_res_thumbnail")
        self.assertEqual(resolution["id"], "low_res_thumbnail")
        self.assertEqual(resolution["source"], "auto_suggestion")

    def test_cli_run_profile_argument_is_available_on_legacy_pipeline(self) -> None:
        args = build_parser().parse_args(
            [
                "run",
                "--metadata",
                "tests/fixtures/example.json",
                "--profile",
                "clean_digital",
            ]
        )
        self.assertEqual(args.command, "run")
        self.assertEqual(args.profile, "clean_digital")


if __name__ == "__main__":
    unittest.main()
