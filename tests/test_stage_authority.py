from __future__ import annotations

from pathlib import Path
import re
import unittest

from roi_image_edit.prompt_assets import PROMPT_NAMES, load_prompt
from roi_image_edit.run_artifacts import (
    attach_stage_context_to_rank_report,
    model_stage_context,
    result_audit_payload,
    stage_progress_fields,
)
from roi_image_edit.stage_policy import STAGE_ORDER


EXPECTED_STAGE_ORDER = (
    "hard_boundary",
    "text_shape",
    "ink_gray_balance",
    "photo_texture",
    "background_cleanup",
)
OLD_PUBLIC_STAGE_IDS = (
    "slot_alignment",
    "font_structure",
    "pose_geometry",
    "stroke_body",
    "tone_gray",
    "edge_quality",
)


def collect_public_stage_values(payload: object) -> list[str]:
    values: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in {"stage_id", "blocking_stage", "stage_order", "order"}:
                if isinstance(value, str):
                    values.append(value)
                elif isinstance(value, list):
                    values.extend(str(item) for item in value)
            values.extend(collect_public_stage_values(value))
    elif isinstance(payload, list):
        for item in payload:
            values.extend(collect_public_stage_values(item))
    return values


class StageAuthorityTest(unittest.TestCase):
    def test_stage_order_constant_is_the_five_stage_authority(self) -> None:
        self.assertEqual(STAGE_ORDER, EXPECTED_STAGE_ORDER)

    def test_runtime_prompts_declare_only_the_five_public_stages(self) -> None:
        stage_pattern = re.compile(r"`([a-z_]+)`")
        for prompt_name in PROMPT_NAMES:
            text = load_prompt(prompt_name)
            with self.subTest(prompt_name=prompt_name):
                for stage_id in EXPECTED_STAGE_ORDER:
                    self.assertIn(f"`{stage_id}`", text)
                for old_stage_id in OLD_PUBLIC_STAGE_IDS:
                    self.assertNotIn(f"`{old_stage_id}`", text)
                public_stage_mentions = {
                    match.group(1)
                    for match in stage_pattern.finditer(text)
                    if match.group(1) in set(EXPECTED_STAGE_ORDER) | set(OLD_PUBLIC_STAGE_IDS)
                }
                self.assertTrue(public_stage_mentions <= set(EXPECTED_STAGE_ORDER))

    def test_prompt_payload_and_reports_expose_only_five_stage_ids(self) -> None:
        report = {
            "pass": False,
            "pipeline_profile": "photo_scan",
            "issues": [{"type": "roi_outside"}],
        }
        progress = stage_progress_fields(report)
        context = model_stage_context(report, "photo_scan")
        rank_payload = attach_stage_context_to_rank_report(
            {"candidates": {"c1": {"hard_check": report}}},
            pipeline_profile="photo_scan",
        )
        result_payload = result_audit_payload(
            {
                "ok": True,
                "runDir": "output/web/run1",
                "profile": "photo_scan",
                "profileResolution": {"id": "photo_scan"},
                "images": [
                    {
                        "id": "img1",
                        "ok": True,
                        "candidates": [{"id": "c1", "stage_context": context}],
                        "regions": [{"summary": {"stage_evidence": {"stage_order": list(STAGE_ORDER)}}}],
                    }
                ],
            }
        )

        for payload in (progress, context, rank_payload, result_payload):
            with self.subTest(payload=type(payload).__name__):
                stage_values = collect_public_stage_values(payload)
                self.assertTrue(stage_values)
                self.assertTrue(set(stage_values) <= set(EXPECTED_STAGE_ORDER), stage_values)

    def test_runtime_prompts_cli_and_web_copy_do_not_expose_old_concerns_as_gates(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        checked_paths = [
            *(repo_root / "src" / "roi_image_edit" / "prompts").glob("*.txt"),
            repo_root / "src" / "roi_image_edit" / "cli.py",
            repo_root / "src" / "roi_image_edit" / "web_app.py",
            repo_root / "src" / "roi_image_edit" / "run_artifacts.py",
            *(repo_root / "web").glob("*"),
        ]
        checked_paths = [path for path in checked_paths if path.is_file()]
        self.assertTrue(checked_paths)
        for path in checked_paths:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=str(path.relative_to(repo_root))):
                for old_stage_id in OLD_PUBLIC_STAGE_IDS:
                    self.assertNotIn(old_stage_id, text)


if __name__ == "__main__":
    unittest.main()
