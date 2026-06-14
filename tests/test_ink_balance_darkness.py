from __future__ import annotations

import copy
import inspect
import unittest

from roi_image_edit.local_validation import (
    candidate_report,
    local_ink_balance_issues,
    local_stroke_body_issues,
    processing_candidate_score,
)


class InkBalanceDarknessTest(unittest.TestCase):
    def test_shorter_replacement_allows_cleanup_edge_lightening(self) -> None:
        source = inspect.getsource(candidate_report)

        self.assertIn("max_edge_lighten_delta = 8.0 if shorter_replacement else 4.0", source)

    def test_longer_replacement_caps_dynamic_roi_core_allowance(self) -> None:
        report = {
            "roi_plan": {"source_slot_count": 2, "target_slot_count": 3},
            "reference_profile": {"dynamic_ink": {"roi_lt55_delta_limit": 114.718}},
            "strict_gate": {"text_complexity_ratio": 1.4594},
            "char_gray_band_metrics": {"enabled": False},
            "strict_visual_metrics": {
                "bands": {
                    "old_lt55_pixels": 233,
                    "lt55_delta": 92,
                    "old_lt55_share_of_lt165": 0.1215,
                    "new_lt55_share_of_lt165": 0.1191,
                }
            },
        }

        issues = local_ink_balance_issues(report)

        self.assertEqual(issues[0]["type"], "roi_core_too_black")
        self.assertEqual(issues[0]["length_change"], "longer")
        self.assertAlmostEqual(issues[0]["actual"], 92.0)
        self.assertLess(issues[0]["limit"], 92.0)

    def test_longer_replacement_flags_excess_mid_gray_body_after_count_normalization(self) -> None:
        report = {
            "roi_plan": {"source_slot_count": 2, "target_slot_count": 3},
            "reference_profile": {"dynamic_ink": {}},
            "strict_gate": {"text_complexity_ratio": 1.4594},
            "char_gray_band_metrics": {"enabled": False},
            "strict_visual_metrics": {
                "bands": {
                    "old_lt55_pixels": 233,
                    "lt55_delta": -184,
                    "old_lt90_pixels": 428,
                    "lt90_delta": 415,
                    "old_lt55_share_of_lt165": 0.1215,
                    "new_lt55_share_of_lt165": 0.0189,
                }
            },
        }

        issues = local_ink_balance_issues(report)

        self.assertEqual(issues[0]["type"], "longer_mid_gray_body_too_black")
        self.assertEqual(issues[0]["length_change"], "longer")
        self.assertGreater(issues[0]["actual"], issues[0]["limit"])

    def test_longer_replacement_allows_count_normalized_lighter_mid_gray_body(self) -> None:
        report = {
            "roi_plan": {"source_slot_count": 2, "target_slot_count": 3},
            "reference_profile": {"dynamic_ink": {}},
            "strict_gate": {"text_complexity_ratio": 1.4594},
            "char_gray_band_metrics": {"enabled": False},
            "strict_visual_metrics": {
                "bands": {
                    "old_lt55_pixels": 233,
                    "lt55_delta": -184,
                    "old_lt90_pixels": 428,
                    "lt90_delta": 239,
                    "old_lt55_share_of_lt165": 0.1215,
                    "new_lt55_share_of_lt165": 0.0,
                }
            },
        }

        self.assertEqual(local_ink_balance_issues(report), [])

    def test_longer_replacement_allows_alpha_contrast_core_with_count_normalized_mid_gray(self) -> None:
        report = {
            "params": {"alpha_contrast": 0.40, "blur": 0.32},
            "roi_plan": {"source_slot_count": 2, "target_slot_count": 3},
            "reference_profile": {"dynamic_ink": {}},
            "strict_gate": {"text_complexity_ratio": 1.4594},
            "char_gray_band_metrics": {"enabled": False},
            "strict_visual_metrics": {
                "bands": {
                    "old_lt55_pixels": 233,
                    "lt55_delta": 5,
                    "old_lt90_pixels": 428,
                    "lt90_delta": 300,
                    "old_lt55_share_of_lt165": 0.1215,
                    "new_lt55_share_of_lt165": 0.1100,
                }
            },
        }

        self.assertEqual(local_ink_balance_issues(report), [])

    def test_longer_replacement_flags_deep_core_overblack_from_user_rejected_final(self) -> None:
        report = {
            "params": {"alpha_contrast": 0.30, "blur": 0.48},
            "roi_plan": {"source_slot_count": 2, "target_slot_count": 3},
            "reference_profile": {"dynamic_ink": {}},
            "strict_gate": {"text_complexity_ratio": 1.4455},
            "char_gray_band_metrics": {"enabled": False},
            "strict_visual_metrics": {
                "bands": {
                    "old_lt55_pixels": 233,
                    "lt55_delta": -33,
                    "old_lt90_pixels": 428,
                    "lt90_delta": 352,
                    "old_lt55_share_of_lt165": 0.1215,
                    "new_lt55_share_of_lt165": 0.0951,
                }
            },
        }

        issues = local_ink_balance_issues(report)

        self.assertEqual(issues[0]["type"], "longer_mid_gray_body_too_black")
        self.assertEqual(issues[0]["lt90_delta"], 352)
        self.assertEqual(issues[0]["expected_lt90_delta"], 214)
        self.assertLess(issues[0]["limit"], issues[0]["actual"])

    def test_longer_replacement_still_flags_sharp_alpha_candidate_well_over_core_cap(self) -> None:
        report = {
            "params": {"alpha_contrast": 0.30, "blur": 0.36},
            "roi_plan": {"source_slot_count": 2, "target_slot_count": 3},
            "reference_profile": {"dynamic_ink": {}},
            "strict_gate": {"text_complexity_ratio": 1.4594},
            "char_gray_band_metrics": {"enabled": False},
            "strict_visual_metrics": {
                "bands": {
                    "old_lt55_pixels": 233,
                    "lt55_delta": 96,
                    "old_lt90_pixels": 428,
                    "lt90_delta": 379,
                    "old_lt55_share_of_lt165": 0.1215,
                    "new_lt55_share_of_lt165": 0.1636,
                }
            },
        }

        issues = local_ink_balance_issues(report)

        self.assertEqual(issues[0]["type"], "roi_core_too_black")
        self.assertGreater(issues[0]["actual"], issues[0]["limit"])

    def test_longer_replacement_allows_sharp_alpha_body_compression_when_roi_body_is_preserved(self) -> None:
        report = {
            "params": {"alpha_contrast": 0.30, "blur": 0.44, "stroke_opacity": 0.0},
            "roi_plan": {"source_slot_count": 2, "target_slot_count": 3},
            "strict_gate": {"text_complexity_ratio": 1.4455},
            "strict_visual_metrics": {
                "bands": {
                    "old_lt165_pixels": 1918,
                    "new_lt165_pixels": 1984,
                }
            },
            "char_gray_band_metrics": {
                "enabled": True,
                "per_char": [
                    {
                        "index": 0,
                        "source_char": "陈",
                        "target_char": "赵",
                        "old": {"lt165": 500, "band_120_165": 280},
                        "delta": {
                            "lt55": -37,
                            "lt165": -154,
                            "band_55_70": 40,
                            "band_70_90": -15,
                            "band_90_120": -15,
                            "band_120_165": -144,
                        },
                    },
                    {
                        "index": 1,
                        "source_char": "芸",
                        "target_char": "真",
                        "old": {"lt165": 520, "band_120_165": 280},
                        "delta": {
                            "lt55": 20,
                            "lt165": -196,
                            "band_55_70": 30,
                            "band_70_90": -60,
                            "band_90_120": -61,
                            "band_120_165": -189,
                        },
                    },
                    {
                        "index": 2,
                        "source_char": None,
                        "target_char": "真",
                        "old": {"lt165": 1, "band_120_165": 1},
                        "delta": {
                            "lt55": 26,
                            "lt165": 82,
                            "band_55_70": 71,
                            "band_70_90": -65,
                            "band_90_120": -65,
                            "band_120_165": 89,
                        },
                    },
                ],
            },
        }

        self.assertEqual(local_stroke_body_issues(report), [])

    def test_shorter_replacement_flags_mid_gray_as_too_black_when_core_is_light(self) -> None:
        report = {
            "roi_plan": {"source_slot_count": 3, "target_slot_count": 2},
            "reference_profile": {"dynamic_ink": {}},
            "strict_gate": {"text_complexity_ratio": 0.8045},
            "strict_visual_metrics": {"bands": {}},
            "char_gray_band_metrics": {
                "enabled": True,
                "per_char": [
                    {
                        "index": 0,
                        "source_char": "赵",
                        "target_char": "陈",
                        "old": {"lt55": 0, "lt120": 114, "lt165": 350},
                        "delta": {
                            "lt55": 0,
                            "lt120": 54,
                            "lt165": 29,
                            "band_70_90": 50,
                            "band_90_120": 7,
                        },
                    },
                    {
                        "index": 1,
                        "source_char": "真",
                        "target_char": "慧",
                        "old": {"lt55": 0, "lt120": 109, "lt165": 341},
                        "delta": {
                            "lt55": 0,
                            "lt120": 41,
                            "lt165": 68,
                            "band_70_90": 24,
                            "band_90_120": 17,
                        },
                    },
                ],
            },
        }

        issues = local_ink_balance_issues(report)

        self.assertEqual(
            [issue["type"] for issue in issues],
            ["changed_char_mid_gray_too_black", "changed_char_mid_gray_too_black"],
        )
        self.assertTrue(all(issue["length_change"] == "shorter" for issue in issues))
        self.assertTrue(all(issue["actual"] > issue["limit"] for issue in issues))

    def test_shorter_replacement_counts_deep_gray_body_as_too_black(self) -> None:
        report = {
            "roi_plan": {"source_slot_count": 3, "target_slot_count": 2},
            "reference_profile": {"dynamic_ink": {}},
            "strict_gate": {"text_complexity_ratio": 0.8045},
            "strict_visual_metrics": {"bands": {}},
            "char_gray_band_metrics": {
                "enabled": True,
                "per_char": [
                    {
                        "index": 0,
                        "source_char": "赵",
                        "target_char": "陈",
                        "old": {"lt55": 0, "lt120": 114, "lt165": 350},
                        "delta": {
                            "lt55": 11,
                            "lt120": 78,
                            "lt165": 30,
                            "band_55_70": 66,
                            "band_70_90": 19,
                            "band_90_120": -18,
                        },
                    }
                ],
            },
        }

        issues = local_ink_balance_issues(report)

        self.assertEqual(issues[0]["type"], "changed_char_mid_gray_too_black")
        self.assertEqual(issues[0]["body_dark_delta"], 67)
        self.assertEqual(issues[0]["band_55_70_delta"], 66)

    def test_shorter_mid_gray_rule_does_not_apply_to_same_length_replacement(self) -> None:
        report = {
            "roi_plan": {"source_slot_count": 2, "target_slot_count": 2},
            "reference_profile": {"dynamic_ink": {}},
            "strict_gate": {"text_complexity_ratio": 0.8},
            "strict_visual_metrics": {"bands": {}},
            "char_gray_band_metrics": {
                "enabled": True,
                "per_char": [
                    {
                        "index": 0,
                        "source_char": "甲",
                        "target_char": "乙",
                        "old": {"lt55": 0, "lt120": 114, "lt165": 350},
                        "delta": {
                            "lt55": 0,
                            "lt120": 54,
                            "lt165": 29,
                            "band_70_90": 50,
                            "band_90_120": 7,
                        },
                    }
                ],
            },
        }

        self.assertEqual(local_ink_balance_issues(report), [])

    def test_longer_replacement_score_prefers_lighter_pass_candidate(self) -> None:
        base_report = {
            "pass": True,
            "stage_gate": {"blocking_stage": None, "stages": []},
            "strict_gate": {"pass": True, "max_dark_pixel_ratio": 1.875, "issues": []},
            "strict_visual_metrics": {"thresholds": {}, "bands": {}},
            "local_ink_balance_issues": [],
            "local_stroke_body_issues": [],
            "font_style_gate": {"enabled": False},
        }
        lighter = copy.deepcopy(base_report)
        lighter["params"] = {"opacity": 0.70, "blur": 0.55}
        darker = copy.deepcopy(base_report)
        darker["params"] = {"opacity": 0.78, "blur": 0.75}

        self.assertLess(processing_candidate_score(lighter), processing_candidate_score(darker))


if __name__ == "__main__":
    unittest.main()
