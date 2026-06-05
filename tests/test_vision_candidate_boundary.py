from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from PIL import Image

from roi_image_edit.iterative_pipeline import CandidateParams, RenderPlan
from roi_image_edit.processing_service import run_region_vision_checks


class FakeVisionClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def call_json(self, *, system_prompt: str, user_prompt: str, image_paths: list[Path]) -> dict[str, object]:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "image_paths": image_paths,
            }
        )
        if len(self.calls) == 1:
            return {"pass": True, "best_candidate": "c1", "reason": "ranked top local candidate"}
        return {
            "pass": True,
            "acceptance_level": "pass",
            "final_decision": "deliver",
            "visual_findings": {},
            "reason": "accepted",
        }


class VisionCandidateBoundaryTest(unittest.TestCase):
    def test_region_vision_request_uses_top_candidates_with_stage_context(self) -> None:
        original = Image.new("RGB", (16, 16), (220, 220, 220))
        plan = RenderPlan(
            target_text="乙",
            source_text="甲",
            search_roi=(0, 0, 16, 16),
            target_roi=(2, 2, 10, 10),
            slot_boxes=(),
            protected_boxes=(),
            source_reference_box=None,
            style_reference_box=None,
            style_reference_text=None,
            draw_mode="replace",
        )
        rendered = []
        for idx, score in enumerate((0.1, 0.2, 0.3), start=1):
            params = CandidateParams(
                candidate_id=f"c{idx}",
                font_name="test",
                font_path="/tmp/test.ttf",
                font_size=12,
                opacity=0.8,
                blur=0.1,
            )
            report = {"pass": True, "pipeline_profile": "photo_scan"}
            rendered.append((params, original.copy(), report, score))

        fake_client = FakeVisionClient()
        with tempfile.TemporaryDirectory() as tmp:
            _chosen, summary = run_region_vision_checks(
                original=original,
                rendered=rendered,
                plan=plan,
                region_dir=Path(tmp),
                vision_client=fake_client,  # type: ignore[arg-type]
                prompts=("master", "candidate {hard_check_report}", "final {final_params} {hard_check_report}"),
                candidate_limit=2,
                font_style_reference={},
                max_revision_rounds=0,
                pipeline_profile="photo_scan",
            )
            request_path = Path(tmp) / "vision_candidate_request.json"
            self.assertTrue(request_path.exists())

        self.assertEqual(len(fake_client.calls), 2)
        rank_prompt = str(fake_client.calls[0]["user_prompt"])
        self.assertIn('"candidate_count": 3', rank_prompt)
        self.assertIn('"vision_candidate_limit": 3', rank_prompt)
        self.assertIn('"total_candidate_count": 3', rank_prompt)
        self.assertIn('"stage_context_by_candidate"', rank_prompt)
        self.assertIn('"c1"', rank_prompt)
        self.assertIn('"c2"', rank_prompt)
        self.assertIn('"c3"', rank_prompt)
        self.assertTrue(summary["candidate_rank"]["local_stage_context"]["stage_context_by_candidate"])


if __name__ == "__main__":
    unittest.main()
