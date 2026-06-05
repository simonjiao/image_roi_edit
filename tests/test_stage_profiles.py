from __future__ import annotations

import ast
from pathlib import Path
import unittest

from roi_image_edit.stage_profiles import (
    resolve_stage_profile,
    stage_profile,
    stage_profile_choices,
    stage_profile_summary,
)
from roi_image_edit.stages import stage_gate_for_report, stage_specs


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

    def test_explicit_profile_overrides_auto_suggestion(self) -> None:
        resolution = resolve_stage_profile("clean_digital", "photo_scan")
        self.assertEqual(resolution["id"], "clean_digital")
        self.assertEqual(resolution["source"], "explicit_request")
        self.assertEqual(resolution["requested_profile"], "clean_digital")
        self.assertEqual(resolution["suggested_profile"], "photo_scan")

    def test_auto_suggestion_used_only_without_explicit_profile(self) -> None:
        resolution = resolve_stage_profile(None, "low_res_thumbnail")
        self.assertEqual(resolution["id"], "low_res_thumbnail")
        self.assertEqual(resolution["source"], "auto_suggestion")


if __name__ == "__main__":
    unittest.main()
