from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from roi_image_edit.iterative_pipeline import (
    CandidateParams,
    RenderPlan,
    TextRun,
    build_font_style_reference,
    build_style_reference_specs,
    font_style_gate,
    rank_fonts_by_style_reference,
    style_profile_distance,
    style_profile_from_mask,
)


def plan() -> RenderPlan:
    return RenderPlan(
        target_text="赵真真",
        source_text="陈芸",
        search_roi=(0, 0, 140, 48),
        target_roi=(10, 8, 92, 36),
        slot_boxes=(TextRun(12, 10, 30, 32, 120), TextRun(34, 10, 52, 32, 110)),
        protected_boxes=((98, 10, 122, 32),),
        source_reference_box=(12, 10, 52, 32),
        style_reference_box=(60, 10, 92, 32),
        style_reference_text="同栏",
        draw_mode="line_chars",
        placement_strategy="left_anchor_span",
        protected_texts=("保护",),
    )


class FontStyleReferenceTest(unittest.TestCase):
    def test_style_reference_specs_include_source_neighbor_and_protected_contexts(self) -> None:
        refs = build_style_reference_specs(plan())

        self.assertEqual([ref["kind"] for ref in refs], [
            "source_text_roi",
            "neighbor_text_context",
            "protected_text_context",
        ])
        self.assertEqual(refs[0]["role"], "primary")
        self.assertTrue(all(ref["weight"] > 0 for ref in refs))

    def test_style_profile_distance_separates_different_stroke_rhythm(self) -> None:
        source_like = np.zeros((24, 36), dtype=bool)
        source_like[3:21, 4:7] = True
        source_like[3:21, 18:21] = True
        source_like[5:8, 4:21] = True
        source_like[15:18, 4:21] = True
        different = np.zeros((24, 36), dtype=bool)
        different[4:20, 4:28] = True

        source_profile = style_profile_from_mask(source_like, reference_kind="source_text_roi")
        similar_profile = style_profile_from_mask(source_like.copy(), category="song_ming")
        different_profile = style_profile_from_mask(different, category="modern_sans")

        self.assertLess(
            style_profile_distance(source_profile, similar_profile),
            style_profile_distance(source_profile, different_profile),
        )

    def test_build_font_style_reference_records_quality_profiles_and_numeric_rhythm(self) -> None:
        render_plan = RenderPlan(
            target_text="2026-06-16",
            source_text="2024-01-01",
            search_roi=(0, 0, 180, 48),
            target_roi=(10, 8, 130, 36),
            slot_boxes=(),
            protected_boxes=(),
            source_reference_box=(10, 8, 110, 34),
            style_reference_box=None,
            style_reference_text=None,
            draw_mode="auto",
            placement_strategy="baseline_numeric",
        )
        image = Image.new("RGB", (180, 48), (230, 230, 230))

        with patch("roi_image_edit.iterative_pipeline.score_font_against_style_references") as scorer:
            def fake_score(_image, _refs, *, font_name: str, font_path: str, font_size: int, **_kwargs):
                base = 10.0 if font_name == "Songti" else 40.0
                return {
                    "font_name": font_name,
                    "font_path": font_path,
                    "font_size": font_size,
                    "category": "song_ming" if font_name == "Songti" else "modern_sans",
                    "category_penalty": 0.0,
                    "raw_score": base,
                    "score": base,
                    "style_profile_distance": 0.05 if font_name == "Songti" else 0.45,
                    "reference_scores": [
                        {
                            "reference_kind": "source_text_roi",
                            "weight": 0.58,
                            "effective_weight": 0.50,
                            "reference_quality": {"score": 0.86, "reason": "strong_reference"},
                            "original_style_profile": {
                                "enabled": True,
                                "ink_density": 0.24,
                                "aspect_ratio": 3.0,
                                "row_projection_std": 0.2,
                                "col_projection_std": 0.4,
                                "horizontal_vertical_run_ratio": 1.0,
                                "edge_density": 0.5,
                            },
                        }
                    ],
                }
            scorer.side_effect = fake_score

            reference = build_font_style_reference(
                image,
                render_plan,
                [("Sans", "/tmp/sans.ttf"), ("Songti", "/tmp/songti.ttf")],
                min_size=18,
                max_size=18,
            )

        self.assertTrue(reference["enabled"])
        self.assertTrue(reference["style_profile_contract"]["enabled"])
        self.assertEqual(reference["style_profile_contract"]["background_refiner_provider"], "not_configured")
        self.assertTrue(reference["numeric_rhythm_profile"]["enabled"])
        self.assertEqual(reference["numeric_rhythm_profile"]["numeric_char_count"], 8)
        self.assertEqual(reference["style_profile_references"][0]["quality"]["reason"], "strong_reference")
        self.assertEqual(
            rank_fonts_by_style_reference(
                [("Sans", "/tmp/sans.ttf"), ("Songti", "/tmp/songti.ttf")],
                reference,
            )[0][0],
            "Songti",
        )

    def test_font_style_gate_reports_profile_distance_issue_without_background_refiner(self) -> None:
        image = Image.new("RGB", (96, 40), (230, 230, 230))
        render_plan = plan()
        candidate_params = CandidateParams(
            candidate_id="bad",
            font_name="Sans",
            font_path="/tmp/sans.ttf",
            font_size=20,
            opacity=0.8,
            blur=0.2,
        )
        reference = {
            "enabled": True,
            "references": [{"kind": "source_text_roi", "reference_text": "陈芸", "reference_box": [12, 10, 52, 32], "weight": 1.0}],
            "best_available": {"font_name": "Songti", "font_path": "/tmp/songti.ttf", "score": 10.0},
            "preferred_best_available": None,
            "preferred_categories": ["song_ming", "cjk_serif"],
            "prefer_serif_categories": False,
            "style_profile_contract": {"enabled": True, "background_refiner_provider": "not_configured"},
            "by_path": {
                "/tmp/sans.ttf": {
                    "font_name": "Sans",
                    "font_path": "/tmp/sans.ttf",
                    "score": 10.5,
                    "style_profile_distance": 0.05,
                }
            },
        }

        with patch("roi_image_edit.iterative_pipeline.score_font_against_style_references") as scorer:
            scorer.return_value = {
                "font_name": "Sans",
                "font_path": "/tmp/sans.ttf",
                "font_size": 20,
                "category": "modern_sans",
                "category_penalty": 0.0,
                "raw_score": 10.8,
                "score": 10.8,
                "style_profile_distance": 0.24,
                "reference_scores": [],
            }

            report = font_style_gate(
                image,
                render_plan,
                candidate_params,
                reference,
                max_score_ratio=1.25,
            )

        self.assertFalse(report["pass"])
        self.assertIn("font_style_profile_distance_ratio", [issue["type"] for issue in report["issues"]])
        self.assertEqual(report["style_profile_contract"]["background_refiner_provider"], "not_configured")

    def test_background_generation_providers_are_not_introduced(self) -> None:
        import roi_image_edit.background_cleanup as background_cleanup

        source = "\n".join(
            [
                background_cleanup.__doc__ or "",
                " ".join(name for name in dir(background_cleanup) if not name.startswith("__")),
            ]
        )

        self.assertNotIn("LaMa", source)
        self.assertNotIn("BrushNet", source)
        self.assertNotIn("PowerPaint", source)


if __name__ == "__main__":
    unittest.main()
