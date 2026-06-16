from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from roi_image_edit.iterative_pipeline import CandidateParams, RenderPlan
from roi_image_edit.region_processing import (
    initial_candidate_stage_frontier,
    select_vision_rendered_candidates,
)
from roi_image_edit.processing_service import run_region_vision_checks


class FakeVisionClient:
    def __init__(self, best_candidate: str = "c1") -> None:
        self.calls: list[dict[str, object]] = []
        self.best_candidate = best_candidate

    def call_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[Path],
        prompt_name: str | None = None,
        audit_path: Path | None = None,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "image_paths": image_paths,
                "prompt_name": prompt_name,
                "audit_path": audit_path,
            }
        )
        if len(self.calls) == 1:
            return {"pass": True, "best_candidate": self.best_candidate, "reason": "ranked top local candidate"}
        return {
            "pass": True,
            "acceptance_level": "pass",
            "final_decision": "deliver",
            "visual_findings": {},
            "reason": "accepted",
        }


class VisionCandidateBoundaryTest(unittest.TestCase):
    def staged_report(self, blocking_stage: str | None) -> dict:
        stage_status = {
            "hard_boundary": {"id": "hard_boundary", "pass": True, "issues": []},
            "text_shape": {"id": "text_shape", "pass": True, "issues": []},
            "ink_gray_balance": {"id": "ink_gray_balance", "pass": True, "issues": []},
            "photo_texture": {"id": "photo_texture", "pass": True, "issues": []},
            "background_cleanup": {"id": "background_cleanup", "pass": True, "issues": []},
        }
        if blocking_stage:
            stage_status[blocking_stage] = {
                "id": blocking_stage,
                "pass": False,
                "issues": [{"type": f"{blocking_stage}_issue"}],
            }
        return {
            "pass": True,
            "pipeline_profile": "photo_scan",
            "strict_gate": {"pass": True, "issues": []},
            "stage_gate": {
                "pass": blocking_stage is None,
                "blocking_stage": blocking_stage,
                "stage_status": stage_status,
                "stages": list(stage_status.values()),
            },
        }

    def rendered_item(
        self,
        candidate_id: str,
        score: float,
        blocking_stage: str | None,
        image: Image.Image,
    ) -> tuple[CandidateParams, Image.Image, dict, float]:
        return (
            CandidateParams(
                candidate_id=candidate_id,
                font_name="test",
                font_path="/tmp/test.ttf",
                font_size=12,
                opacity=0.8,
                blur=0.1,
            ),
            image.copy(),
            self.staged_report(blocking_stage),
            score,
        )

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
        self.assertEqual([Path(path).name for path in fake_client.calls[0]["image_paths"]], ["vision_candidate_sheet.png"])
        self.assertEqual([Path(path).name for path in fake_client.calls[1]["image_paths"]], ["final_acceptance_vision_compact.png"])
        rank_prompt = str(fake_client.calls[0]["user_prompt"])
        final_prompt = str(fake_client.calls[1]["user_prompt"])
        self.assertIn('"candidate_count": 3', rank_prompt)
        self.assertIn('"vision_candidate_limit": 3', rank_prompt)
        self.assertIn('"total_candidate_count": 3', rank_prompt)
        self.assertIn('"stage_context_by_candidate"', rank_prompt)
        self.assertIn('"c1"', rank_prompt)
        self.assertIn('"c2"', rank_prompt)
        self.assertIn('"c3"', rank_prompt)
        self.assertIn('"vision_brief_schema": 1', rank_prompt)
        self.assertIn('"vision_brief_schema": 1', final_prompt)
        self.assertIn('"full_local_report_artifact"', final_prompt)
        self.assertNotIn('"slot_quality_report"', final_prompt)
        self.assertNotIn('"strict_visual_metrics"', final_prompt)
        self.assertTrue(summary["candidate_rank"]["local_stage_context"]["stage_context_by_candidate"])

    def test_initial_candidate_frontier_orders_later_stage_before_lower_score_ink_candidate(self) -> None:
        image = Image.new("RGB", (16, 16), (220, 220, 220))
        ink = self.rendered_item("ink", 0.1, "ink_gray_balance", image)
        background = self.rendered_item("background", 9.0, "background_cleanup", image)

        selected = select_vision_rendered_candidates([ink, background], 2)

        self.assertEqual([item[0].candidate_id for item in selected], ["background", "ink"])
        self.assertGreater(
            initial_candidate_stage_frontier(background[2]),
            initial_candidate_stage_frontier(ink[2]),
        )

    def test_model_choice_at_later_stage_is_not_overridden_by_full_stage_pass_requirement(self) -> None:
        original = Image.new("RGB", (40, 40), (220, 220, 220))
        plan = RenderPlan(
            target_text="乙",
            source_text="甲",
            search_roi=(0, 0, 40, 40),
            target_roi=(4, 4, 24, 24),
            slot_boxes=(),
            protected_boxes=(),
            source_reference_box=None,
            style_reference_box=None,
            style_reference_text=None,
            draw_mode="replace",
        )
        rendered = [
            self.rendered_item("c1", 0.1, "ink_gray_balance", original),
            self.rendered_item("c2", 9.0, "background_cleanup", original),
        ]
        fake_client = FakeVisionClient(best_candidate="c2")
        with tempfile.TemporaryDirectory() as tmp:
            chosen, summary = run_region_vision_checks(
                original=original,
                rendered=rendered,
                plan=plan,
                region_dir=Path(tmp),
                vision_client=fake_client,  # type: ignore[arg-type]
                prompts=("master", "candidate {hard_check_report}", "final {final_params} {hard_check_report}"),
                candidate_limit=8,
                font_style_reference={},
                max_revision_rounds=0,
                pipeline_profile="photo_scan",
            )

        self.assertIsNotNone(chosen)
        self.assertEqual(chosen.candidate_id, "c2")
        self.assertNotIn("model_choice_overridden", summary["candidate_rank"])
        self.assertEqual(
            summary["candidate_rank"]["local_initial_selection"]["selection_rule"],
            "stroke_weight_fit_then_stage_frontier_then_shape_score_then_local_score",
        )

    def test_model_choice_at_earlier_stage_is_overridden_by_frontier_fallback(self) -> None:
        original = Image.new("RGB", (40, 40), (220, 220, 220))
        plan = RenderPlan(
            target_text="乙",
            source_text="甲",
            search_roi=(0, 0, 40, 40),
            target_roi=(4, 4, 24, 24),
            slot_boxes=(),
            protected_boxes=(),
            source_reference_box=None,
            style_reference_box=None,
            style_reference_text=None,
            draw_mode="replace",
        )
        rendered = [
            self.rendered_item("c1", 0.1, "ink_gray_balance", original),
            self.rendered_item("c2", 9.0, "background_cleanup", original),
        ]
        fake_client = FakeVisionClient(best_candidate="c1")
        with tempfile.TemporaryDirectory() as tmp:
            chosen, summary = run_region_vision_checks(
                original=original,
                rendered=rendered,
                plan=plan,
                region_dir=Path(tmp),
                vision_client=fake_client,  # type: ignore[arg-type]
                prompts=("master", "candidate {hard_check_report}", "final {final_params} {hard_check_report}"),
                candidate_limit=8,
                font_style_reference={},
                max_revision_rounds=0,
                pipeline_profile="photo_scan",
            )

        self.assertIsNotNone(chosen)
        self.assertEqual(chosen.candidate_id, "c2")
        self.assertEqual(
            summary["candidate_rank"]["model_choice_overridden"]["reason"],
            "model selected an earlier-stage candidate than the best local stage frontier",
        )

    def test_final_acceptance_uses_compact_single_image_and_short_prompt_payload(self) -> None:
        original = Image.new("RGB", (120, 80), (220, 220, 220))
        plan = RenderPlan(
            target_text="乙",
            source_text="甲",
            search_roi=(20, 20, 86, 46),
            target_roi=(42, 22, 66, 42),
            slot_boxes=(),
            protected_boxes=(),
            source_reference_box=None,
            style_reference_box=None,
            style_reference_text=None,
            draw_mode="replace",
        )
        params = CandidateParams(
            candidate_id="c1",
            font_name="test",
            font_path="/tmp/test.ttf",
            font_size=12,
            opacity=0.8,
            blur=0.1,
        )
        noisy_full_report = {
            "pass": True,
            "pipeline_profile": "photo_scan",
            "stage_gate": {"pass": True, "blocking_stage": None},
            "strict_gate": {"pass": True, "issues": []},
            "slot_quality_report": {"large": "x" * 5000},
            "strict_visual_metrics": {"large": "y" * 5000},
        }
        fake_client = FakeVisionClient()
        with tempfile.TemporaryDirectory() as tmp:
            run_region_vision_checks(
                original=original,
                rendered=[(params, original.copy(), noisy_full_report, 0.1)],
                plan=plan,
                region_dir=Path(tmp),
                vision_client=fake_client,  # type: ignore[arg-type]
                prompts=("master", "candidate {hard_check_report}", "final {final_params} {hard_check_report}"),
                candidate_limit=8,
                font_style_reference={},
                max_revision_rounds=0,
                pipeline_profile="photo_scan",
            )
            compact_path = Path(tmp) / "final_acceptance_vision_compact.png"
            brief_path = Path(tmp) / "final_acceptance_vision_brief.json"
            local_report_path = Path(tmp) / "final_acceptance_local_report.json"
            self.assertTrue(compact_path.exists())
            self.assertTrue(brief_path.exists())
            self.assertTrue(local_report_path.exists())
            with Image.open(compact_path) as compact:
                self.assertLessEqual(compact.width, 1800)

        final_call = fake_client.calls[1]
        final_prompt = str(final_call["user_prompt"])
        self.assertEqual(len(final_call["image_paths"]), 1)
        self.assertEqual(Path(final_call["image_paths"][0]).name, "final_acceptance_vision_compact.png")
        self.assertIn('"vision_brief_schema": 1', final_prompt)
        self.assertIn('"local_deterministic_checks"', final_prompt)
        self.assertNotIn('"large": "' + ("x" * 64), final_prompt)
        self.assertNotIn('"large": "' + ("y" * 64), final_prompt)

    def test_region_vision_request_excludes_stage_blocked_candidates_when_passed_candidates_exist(self) -> None:
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
            report = {
                "pass": True,
                "pipeline_profile": "photo_scan",
                "strict_gate": {"pass": True, "issues": []},
                "stage_gate": {"pass": True, "blocking_stage": None},
            }
            if idx == 1:
                report["stage_gate"] = {
                    "pass": False,
                    "blocking_stage": "ink_gray_balance",
                    "stages": [
                        {
                            "id": "ink_gray_balance",
                            "pass": False,
                            "issues": [{"type": "roi_core_too_black"}],
                        }
                    ],
                }
            rendered.append((params, original.copy(), report, score))

        fake_client = FakeVisionClient(best_candidate="c2")
        with tempfile.TemporaryDirectory() as tmp:
            _chosen, _summary = run_region_vision_checks(
                original=original,
                rendered=rendered,
                plan=plan,
                region_dir=Path(tmp),
                vision_client=fake_client,  # type: ignore[arg-type]
                prompts=("master", "candidate {hard_check_report}", "final {final_params} {hard_check_report}"),
                candidate_limit=8,
                font_style_reference={},
                max_revision_rounds=0,
                pipeline_profile="photo_scan",
            )
            request = json.loads((Path(tmp) / "vision_candidate_request.json").read_text(encoding="utf-8"))

        self.assertEqual(request["candidate_ids"], ["c2", "c3"])
        self.assertEqual(request["candidate_count"], 2)
        self.assertNotIn("c1", request["candidates"])

    def test_longer_replacement_vision_selection_keeps_mid_blur_alpha_alternatives(self) -> None:
        original = Image.new("RGB", (16, 16), (220, 220, 220))
        rendered = []
        specs = [
            ("sharp1", 0.1, 0.36, 0.30),
            ("sharp2", 0.2, 0.32, 0.40),
            ("sharp3", 0.3, 0.28, 0.40),
            ("plain1", 0.4, 0.55, 0.00),
            ("plain2", 0.5, 0.60, 0.00),
            ("bridge1", 9.0, 0.44, 0.25),
            ("bridge2", 10.0, 0.48, 0.22),
        ]
        for candidate_id, score, blur, alpha_contrast in specs:
            params = CandidateParams(
                candidate_id=candidate_id,
                font_name="test",
                font_path="/tmp/test.ttf",
                font_size=12,
                opacity=0.66,
                blur=blur,
                alpha_contrast=alpha_contrast,
            )
            report = {
                "pass": True,
                "pipeline_profile": "photo_scan",
                "strict_gate": {"pass": True, "issues": []},
                "stage_gate": {"pass": True, "blocking_stage": None},
                "roi_plan": {"source_slot_count": 2, "target_slot_count": 3},
            }
            rendered.append((params, original.copy(), report, score))

        selected = select_vision_rendered_candidates(rendered, 6)
        candidate_ids = [item[0].candidate_id for item in selected]

        self.assertEqual(candidate_ids[:3], ["sharp1", "sharp2", "sharp3"])
        self.assertIn("bridge1", candidate_ids)
        self.assertIn("bridge2", candidate_ids)


if __name__ == "__main__":
    unittest.main()
