from __future__ import annotations

import inspect
from pathlib import Path
import tempfile
import unittest

import roi_image_edit.processing_service as processing_service
from roi_image_edit.run_artifacts import (
    EXTERNAL_ARTIFACT_SCHEMA_VERSION,
    attach_stage_context_to_rank_report,
    delivery_artifact_manifest,
    external_artifact_schema_report,
    model_stage_context,
    progress_record,
    request_audit_payload,
    result_audit_payload,
    revision_round_continuation_contract,
    stage_progress_fields,
    vision_candidate_request_payload,
)


EXPECTED_STAGE_ORDER = [
    "hard_boundary",
    "text_shape",
    "ink_gray_balance",
    "photo_texture",
    "background_cleanup",
]


class RunArtifactsTest(unittest.TestCase):
    def _touch(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return path

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
                    "autoRoiEvidence": {
                        "region_count": 1,
                        "all_have_search_roi": True,
                        "all_have_edit_roi": True,
                        "regions": [
                            {
                                "field_key": "name",
                                "search_roi": [2, 4, 46, 26],
                                "edit_roi": [10, 8, 32, 22],
                            }
                        ],
                    },
                    "stage_evidence": {
                        "auto_roi": {
                            "region_count": 1,
                            "overlay_path": "output/web/run1/sample_auto_roi_overlay.png",
                            "regions": [
                                {
                                    "search_roi": [2, 4, 46, 26],
                                    "edit_roi": [10, 8, 32, 22],
                                }
                            ],
                        }
                    },
                    "artifacts": {
                        "auto_roi_overlay": "output/web/run1/sample_auto_roi_overlay.png",
                    },
                    "candidates": [
                        {
                            "id": "c1",
                            "dataUrl": "data:image/png;base64,candidate",
                            "stage_context": {"blocking_stage": "text_shape"},
                            "blocking_stage": "text_shape",
                            "patch": {"font_size_delta": 1},
                            "model_suggestions": [{"parameter": "font_size", "delta": 1}],
                            "rejection_reason": "text_shape_not_yet_passed",
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
        self.assertEqual(audit["artifactSchemaVersion"], EXTERNAL_ARTIFACT_SCHEMA_VERSION)
        self.assertEqual(audit["profile"], "clean_digital")
        self.assertEqual(audit["profileResolution"]["source"], "explicit_request")
        self.assertEqual(audit["profileResolution"]["suggested_profile"], "photo_scan")
        image = audit["images"][0]
        self.assertNotIn("sourceDataUrl", image)
        self.assertNotIn("resultDataUrl", image)
        self.assertNotIn("dataUrl", image["candidates"][0])
        self.assertEqual(image["candidates"][0]["stage_context"]["blocking_stage"], "text_shape")
        self.assertEqual(image["candidates"][0]["blocking_stage"], "text_shape")
        self.assertEqual(image["candidates"][0]["patch"], {"font_size_delta": 1})
        self.assertEqual(image["candidates"][0]["model_suggestions"][0]["parameter"], "font_size")
        self.assertEqual(image["candidates"][0]["rejection_reason"], "text_shape_not_yet_passed")
        self.assertEqual(image["autoRoiEvidence"]["regions"][0]["search_roi"], [2, 4, 46, 26])
        self.assertEqual(image["autoRoiEvidence"]["regions"][0]["edit_roi"], [10, 8, 32, 22])
        self.assertEqual(image["stage_evidence"]["auto_roi"]["regions"][0]["search_roi"], [2, 4, 46, 26])
        self.assertEqual(image["stage_evidence"]["auto_roi"]["regions"][0]["edit_roi"], [10, 8, 32, 22])
        self.assertEqual(
            image["stage_evidence"]["auto_roi"]["overlay_path"],
            "output/web/run1/sample_auto_roi_overlay.png",
        )
        self.assertEqual(image["artifacts"]["auto_roi_overlay"], "output/web/run1/sample_auto_roi_overlay.png")
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

    def test_external_artifact_schema_covers_result_progress_and_explanations(self) -> None:
        schema = external_artifact_schema_report()

        self.assertEqual(schema["artifact_schema_version"], EXTERNAL_ARTIFACT_SCHEMA_VERSION)
        prompt_contract = schema["prompt_io_contract"]
        self.assertIn("candidate_rank_prompt.txt", prompt_contract["prompts"])
        self.assertEqual(
            prompt_contract["prompts"]["darkness_blur_prompt.txt"]["output_field_handling"]["best_opacity"],
            "reserved_not_called",
        )
        result_schema = schema["result_json"]
        self.assertIn("artifactManifest", result_schema["root_required"])
        self.assertIn("profileResolution", result_schema["root_required"])
        self.assertIn("stage_evidence", result_schema["image_required"])
        self.assertIn("stage_context", result_schema["candidate_required"])
        self.assertIn("patch", result_schema["candidate_required"])
        self.assertIn("model_suggestions", result_schema["candidate_required"])
        self.assertIn("rejection_reason", result_schema["candidate_required"])
        self.assertIn("final_acceptance", result_schema["vision_required"])
        self.assertIn("revision_attempts", result_schema["vision_required"])
        self.assertIn("final_is_rejected_candidate", result_schema["rejection_required"])

        progress_schema = schema["progress_jsonl"]
        self.assertIn("artifactSchemaVersion", progress_schema["record_required"])
        self.assertIn("blocking_stage", progress_schema["stage_fields"])
        self.assertIn("stage_evidence", progress_schema["candidate_fields"])
        self.assertIn("stage_filter_report", progress_schema["patch_fields"])
        self.assertIn("model_suggestion_filter", progress_schema["vision_suggestion_fields"])
        self.assertIn("rejection_reason", progress_schema["rejection_reason_fields"])

        manifest_schema = schema["artifact_manifest"]
        self.assertIn("global_artifacts", manifest_schema["root_required"])
        self.assertIn("all_explainable", manifest_schema["root_required"])
        self.assertIn("candidate_images", manifest_schema["image_required"])
        self.assertIn("stage_evidence", manifest_schema["image_required"])

    def test_delivery_artifact_manifest_explains_rejected_region_with_stage_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_path = self._touch(run_dir / "result.json")
            progress_path = self._touch(run_dir / "progress.jsonl")
            final_path = self._touch(run_dir / "final.png")
            orientation_report = self._touch(run_dir / "auto_orientation_report.json")
            auto_roi_evidence_report = self._touch(run_dir / "auto_roi_evidence.json")
            auto_overlay = self._touch(run_dir / "auto_roi_overlay.png")
            selected_candidate = self._touch(run_dir / "r1" / "selected_candidate.png")
            selected_compare = self._touch(run_dir / "r1" / "selected_compare.png")
            slot_quality_report = self._touch(run_dir / "r1" / "slot_quality_report.json")
            pre_candidate_gate_report = self._touch(run_dir / "r1" / "pre_candidate_gate_report.json")
            candidate_sheet = self._touch(run_dir / "r1" / "vision_candidate_sheet.png")
            final_compare = self._touch(run_dir / "r1" / "vision_final_compare.png")
            iter_compare = self._touch(run_dir / "r1" / "vision_final_compare_iter01.png")
            shape_compare = self._touch(run_dir / "r1" / "stage_evidence" / "text_shape_top_compare.png")
            shape_report = self._touch(run_dir / "r1" / "stage_evidence" / "text_shape_top_report.json")
            summary_path = self._touch(run_dir / "r1" / "stage_evidence" / "summary.json")
            response = {
                "ok": True,
                "runDir": str(run_dir),
                "artifactManifest": str(run_dir / "artifact_manifest.json"),
                "images": [
                    {
                        "id": "img1",
                        "ok": True,
                        "accepted": False,
                        "applied": False,
                        "artifacts": {
                            "final": str(final_path),
                            "auto_orientation_report": str(orientation_report),
                            "auto_roi_evidence_report": str(auto_roi_evidence_report),
                            "auto_roi_overlay": str(auto_overlay),
                            "final_is_rejected_candidate": True,
                        },
                        "stage_evidence": {
                            "auto_roi": {
                                "overlay_path": str(auto_overlay),
                                "orientation_report_path": str(orientation_report),
                                "report_path": str(auto_roi_evidence_report),
                            },
                        },
                        "regions": [
                            {
                                "id": "r1",
                                "accepted": False,
                                "summary": {
                                    "trace": {"final_blocking_stage": "text_shape"},
                                    "plan": {
                                        "slot_quality_report": {
                                            "pass": True,
                                            "source_count": 2,
                                            "target_count": 2,
                                        }
                                    },
                                    "vision": {
                                        "artifacts": {
                                            "candidate_sheet": str(candidate_sheet),
                                            "final_compare": str(final_compare),
                                            "revision_previews": [
                                                {"round": 1, "path": str(iter_compare)}
                                            ],
                                        }
                                    },
                                    "artifacts": {
                                        "selected_candidate": str(selected_candidate),
                                        "selected_compare": str(selected_compare),
                                        "slot_quality_report": str(slot_quality_report),
                                        "pre_candidate_gate_report": str(pre_candidate_gate_report),
                                        "stage_evidence": {
                                            "summary": str(summary_path),
                                            "stages": {
                                                "text_shape": {
                                                    "available": True,
                                                    "compare_path": str(shape_compare),
                                                    "report_path": str(shape_report),
                                                },
                                                "ink_gray_balance": {
                                                    "available": False,
                                                    "reason": "no_candidate_blocked_at_stage",
                                                },
                                            },
                                        },
                                    },
                                },
                            }
                        ],
                    }
                ],
            }

            manifest = delivery_artifact_manifest(
                response,
                run_dir=run_dir,
                result_path=result_path,
                progress_path=progress_path,
            )

        self.assertTrue(manifest["all_explainable"])
        self.assertEqual(manifest["missing_required"], [])
        image = manifest["images"][0]
        self.assertEqual(image["status"], "rejected")
        self.assertFalse(image["accepted"])
        self.assertEqual(image["blocking_stage"], "text_shape")
        self.assertTrue(image["explainable"])
        self.assertIn("result_json", [item["key"] for item in manifest["global_artifacts"]])
        self.assertIn("progress_jsonl", [item["key"] for item in manifest["global_artifacts"]])
        report_keys = [item["key"] for item in image["reports"]]
        image_keys = [item["key"] for item in image["candidate_images"]]
        embedded_keys = [item["key"] for item in image["embedded_reports"]]
        self.assertIn("auto_orientation_report", report_keys)
        self.assertIn("auto_roi_evidence_report", report_keys)
        self.assertIn("slot_quality_report", report_keys)
        self.assertIn("pre_candidate_gate_report", report_keys)
        self.assertIn("stage_evidence_summary", report_keys)
        self.assertIn("text_shape_top_report", report_keys)
        self.assertIn("selected_candidate", image_keys)
        self.assertIn("selected_compare", image_keys)
        self.assertIn("vision_candidate_sheet", image_keys)
        self.assertIn("vision_final_compare", image_keys)
        self.assertIn("revision_preview_round_1", image_keys)
        self.assertIn("text_shape_top_compare", image_keys)
        self.assertIn("slot_quality_report", embedded_keys)

    def test_progress_record_adds_stable_schema_version(self) -> None:
        record = progress_record(
            "revision_round_finished",
            {
                "blocking_stage": "ink_gray_balance",
                "stage_filter_report": {"stage_id": "ink_gray_balance"},
                "model_suggestion_filter": {"attempt_records": [{"rejection_reason": "forbidden"}]},
                "stop_reason": "no_ink_gray_balance_severity_improvement",
            },
            timestamp="2026-06-05T12:00:00+0800",
        )

        self.assertEqual(record["artifactSchemaVersion"], EXTERNAL_ARTIFACT_SCHEMA_VERSION)
        self.assertEqual(record["time"], "2026-06-05T12:00:00+0800")
        self.assertEqual(record["event"], "revision_round_finished")
        self.assertEqual(record["blocking_stage"], "ink_gray_balance")
        self.assertEqual(record["stage_filter_report"]["stage_id"], "ink_gray_balance")
        self.assertEqual(
            record["model_suggestion_filter"]["attempt_records"][0]["rejection_reason"],
            "forbidden",
        )
        self.assertEqual(record["stop_reason"], "no_ink_gray_balance_severity_improvement")

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

    def test_model_stage_context_default_includes_profile_constraints(self) -> None:
        context = model_stage_context(None, "low_res_thumbnail")

        self.assertEqual(context["pipeline_profile"], "low_res_thumbnail")
        self.assertEqual(context["profile_constraints"]["vision_context_scale"], "magnified")
        self.assertIn("stroke_body_weight", context["profile_constraints"]["shape_priority"])
        self.assertIsNone(context["blocking_stage"])
        self.assertIsNone(context["optimization_policy"]["stage_id"])

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

    def test_vision_candidate_request_records_limit_and_stage_context(self) -> None:
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
        request = vision_candidate_request_payload(
            hard_reports,
            pipeline_profile="photo_scan",
            requested_vision_candidate_limit=2,
            total_candidate_count=8,
        )

        self.assertEqual(request["requested_vision_candidate_limit"], 2)
        self.assertEqual(request["vision_candidate_limit"], 3)
        self.assertEqual(request["total_candidate_count"], 8)
        self.assertEqual(request["candidate_count"], 2)
        self.assertTrue(request["candidate_count_within_limit"])
        self.assertTrue(request["stage_context_complete"])
        self.assertEqual(set(request["stage_context_by_candidate"]), {"c1", "c2"})
        self.assertEqual(
            request["stage_context_by_candidate"]["c1"]["blocking_stage"],
            "hard_boundary",
        )

        high_request = vision_candidate_request_payload(
            hard_reports,
            pipeline_profile="photo_scan",
            requested_vision_candidate_limit=12,
            total_candidate_count=20,
        )
        self.assertEqual(high_request["requested_vision_candidate_limit"], 12)
        self.assertEqual(high_request["vision_candidate_limit"], 8)

    def test_revision_round_continuation_contract_requires_stage_direction(self) -> None:
        contract = revision_round_continuation_contract(
            {
                "round": 1,
                "basis_blocking_stage": "text_shape",
                "basis_stage_source": "local_report",
                "shape_candidate_grid": {
                    "enabled": True,
                    "stage_id": "text_shape",
                    "optimization_step": "shape_reset",
                    "candidate_count": 48,
                    "budget": {
                        "raw_candidate_budget": 640,
                        "retained_count": 48,
                    },
                },
            },
            max_revision_rounds=12,
        )

        self.assertFalse(contract["max_rounds_is_strategy"])
        self.assertTrue(contract["requires_stage_specific_candidate_direction"])
        self.assertTrue(contract["has_stage_specific_candidate_direction"])
        self.assertTrue(contract["continuation_allowed"])
        self.assertEqual(contract["max_revision_rounds"], 12)
        self.assertEqual(contract["candidate_direction_sources"][0]["source"], "shape_candidate_grid")
        self.assertEqual(contract["candidate_direction_sources"][0]["stage_id"], "text_shape")
        self.assertEqual(contract["candidate_direction_sources"][0]["candidate_count"], 48)
        self.assertIsNone(contract["missing_direction_reason"])

        patch_contract = revision_round_continuation_contract(
            {
                "round": 2,
                "basis_blocking_stage": "ink_gray_balance",
                "stage_filter_report": {
                    "stage_id": "ink_gray_balance",
                    "accepted_count": 2,
                    "patcher": {
                        "optimization_steps": ["core_black_search", "opacity_search"],
                    },
                },
                "selected_optimization_step": "core_black_search",
            },
            max_revision_rounds=8,
        )
        self.assertTrue(patch_contract["continuation_allowed"])
        self.assertEqual(
            patch_contract["candidate_direction_sources"][0]["source"],
            "stage_patcher_dispatch",
        )
        self.assertEqual(
            patch_contract["candidate_direction_sources"][0]["optimization_step"],
            "core_black_search",
        )

    def test_revision_round_continuation_contract_rejects_max_rounds_only(self) -> None:
        contract = revision_round_continuation_contract(
            {
                "round": 1,
                "basis_blocking_stage": "ink_gray_balance",
                "basis_stage_source": "local_report",
                "shape_candidate_grid": {
                    "enabled": True,
                    "stage_id": "text_shape",
                    "optimization_step": "shape_reset",
                    "candidate_count": 48,
                },
                "stage_filter_report": {
                    "stage_id": "text_shape",
                    "accepted_count": 3,
                },
            },
            max_revision_rounds=12,
        )

        self.assertFalse(contract["max_rounds_is_strategy"])
        self.assertTrue(contract["requires_stage_specific_candidate_direction"])
        self.assertFalse(contract["has_stage_specific_candidate_direction"])
        self.assertFalse(contract["continuation_allowed"])
        self.assertEqual(contract["candidate_direction_sources"], [])
        self.assertEqual(
            contract["missing_direction_reason"],
            "no stage-specific candidate grid or accepted stage patch for current blocking stage",
        )

    def test_processing_service_records_revision_continuation_contract(self) -> None:
        source = inspect.getsource(processing_service.run_region_vision_checks)
        self.assertIn("revision_round_continuation_contract", source)
        self.assertIn('"revision_continuation_contract": continuation_contract', source)
        self.assertIn('"no_stage_specific_candidate_direction"', source)
        self.assertIn('"revision_continuation_contract": selected_continuation_contract', source)


if __name__ == "__main__":
    unittest.main()
