from __future__ import annotations

import unittest

from roi_image_edit.run_artifacts import (
    attach_stage_context_to_rank_report,
    model_stage_context,
    request_audit_payload,
    result_audit_payload,
    stage_progress_fields,
)


EXPECTED_STAGE_ORDER = [
    "hard_boundary",
    "text_shape",
    "ink_gray_balance",
    "photo_texture",
    "background_cleanup",
]


class RunArtifactsTest(unittest.TestCase):
    def test_request_audit_payload_keeps_runtime_limits_and_strips_image_data(self) -> None:
        payload = {
            "profile": "photo_scan",
            "profileSuggestion": "clean_digital",
            "maxCandidates": 130,
            "visionCandidateLimit": 8,
            "maxRevisionRounds": 12,
            "images": [
                {
                    "id": "img1",
                    "filename": "sample.png",
                    "instruction": "姓名甲修改为乙",
                    "dataUrl": "data:image/png;base64,abc",
                    "regions": [{"id": "r1", "rect": {"x": 1, "y": 2, "w": 3, "h": 4}}],
                }
            ],
        }
        audit = request_audit_payload(payload)
        self.assertEqual(audit["profile"], "photo_scan")
        self.assertEqual(audit["profileSuggestion"], "clean_digital")
        self.assertEqual(audit["maxCandidates"], 130)
        self.assertEqual(audit["visionCandidateLimit"], 8)
        self.assertEqual(audit["maxRevisionRounds"], 12)
        self.assertNotIn("dataUrl", audit["images"][0])
        self.assertEqual(audit["images"][0]["regions"][0]["rect"], {"x": 1, "y": 2, "w": 3, "h": 4})

    def test_result_audit_payload_strips_data_urls_but_keeps_stage_artifact_fields(self) -> None:
        response = {
            "ok": True,
            "runDir": "output/web/run1",
            "profile": "clean_digital",
            "profileResolution": {
                "id": "clean_digital",
                "source": "explicit_request",
                "suggested_profile": "photo_scan",
            },
            "images": [
                {
                    "id": "img1",
                    "ok": True,
                    "accepted": False,
                    "sourceDataUrl": "data:image/png;base64,source",
                    "resultDataUrl": "data:image/png;base64,result",
                    "candidates": [
                        {
                            "id": "c1",
                            "dataUrl": "data:image/png;base64,candidate",
                            "stage_context": {"blocking_stage": "text_shape"},
                        }
                    ],
                    "regions": [
                        {
                            "id": "r1",
                            "summary": {
                                "trace": {"final_blocking_stage": "text_shape"},
                                "vision": {
                                    "revision_rounds": [
                                        {
                                            "round": 1,
                                            "stage_id": "text_shape",
                                            "selected_optimization_step": "stroke_body_shape",
                                        }
                                    ],
                                    "revision_attempts": [
                                        {
                                            "stage_id": "text_shape",
                                            "optimization_step": "stroke_body_shape",
                                        }
                                    ],
                                },
                            },
                        }
                    ],
                }
            ],
        }
        audit = result_audit_payload(response)
        self.assertEqual(audit["profile"], "clean_digital")
        self.assertEqual(audit["profileResolution"]["source"], "explicit_request")
        self.assertEqual(audit["profileResolution"]["suggested_profile"], "photo_scan")
        image = audit["images"][0]
        self.assertNotIn("sourceDataUrl", image)
        self.assertNotIn("resultDataUrl", image)
        self.assertNotIn("dataUrl", image["candidates"][0])
        self.assertEqual(image["candidates"][0]["stage_context"]["blocking_stage"], "text_shape")
        self.assertEqual(image["regions"][0]["summary"]["trace"]["final_blocking_stage"], "text_shape")
        self.assertEqual(
            image["regions"][0]["summary"]["vision"]["revision_rounds"][0]["stage_id"],
            "text_shape",
        )
        self.assertEqual(
            image["regions"][0]["summary"]["vision"]["revision_rounds"][0]["selected_optimization_step"],
            "stroke_body_shape",
        )
        self.assertEqual(
            image["regions"][0]["summary"]["vision"]["revision_attempts"][0]["optimization_step"],
            "stroke_body_shape",
        )

    def test_stage_progress_fields_expose_stable_stage_keys(self) -> None:
        report = {"pass": False, "pipeline_profile": "photo_scan", "issues": [{"type": "roi_outside"}]}
        progress = stage_progress_fields(report)
        self.assertEqual(progress["pipeline_profile"], "photo_scan")
        self.assertEqual(progress["stage_order"], EXPECTED_STAGE_ORDER)
        self.assertEqual(progress["blocking_stage"], "hard_boundary")
        self.assertTrue(progress["blocking_stage_blocks_next"])
        self.assertEqual(progress["blocking_stage_reason"], "roi_outside")
        self.assertEqual(progress["allowed_patch_keys"], [])
        self.assertIn("font_size_delta", progress["blocked_patch_keys"])

    def test_model_stage_context_includes_stage_filter_and_optimization_policy(self) -> None:
        context = model_stage_context(
            {"pass": False, "pipeline_profile": "photo_scan", "issues": [{"type": "roi_outside"}]},
            "photo_scan",
        )
        self.assertEqual(context["stage_order"], EXPECTED_STAGE_ORDER)
        self.assertEqual(context["blocking_stage"], "hard_boundary")
        self.assertTrue(context["blocking_stage_blocks_next"])
        self.assertEqual(context["allowed_patch_keys"], [])
        self.assertIn("font_size_delta", context["blocked_patch_keys"])
        self.assertEqual(context["optimization_policy"]["stage_id"], "hard_boundary")
        self.assertFalse(context["optimization_policy"]["allowed_steps"])

    def test_rank_report_attaches_stage_context_candidate_count_and_contract(self) -> None:
        hard_reports = {
            "candidates": {
                "c1": {
                    "hard_check": {
                        "pass": False,
                        "pipeline_profile": "photo_scan",
                        "issues": [{"type": "roi_outside"}],
                    }
                },
                "c2": {"hard_check": {"pass": True, "pipeline_profile": "photo_scan"}},
            }
        }
        enriched = attach_stage_context_to_rank_report(hard_reports, pipeline_profile="photo_scan")
        self.assertEqual(enriched["pipeline_profile"], "photo_scan")
        self.assertEqual(enriched["candidate_count"], 2)
        self.assertEqual(enriched["candidate_ids"], ["c1", "c2"])
        self.assertEqual(
            enriched["stage_context_by_candidate"]["c1"]["blocking_stage"],
            "hard_boundary",
        )
        self.assertIsNone(enriched["stage_context_by_candidate"]["c2"]["blocking_stage"])
        self.assertEqual(enriched["stage_filter_contract"]["authoritative"], "local_stage_filter")


if __name__ == "__main__":
    unittest.main()
