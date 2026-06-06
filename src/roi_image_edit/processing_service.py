from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from roi_image_edit.auto_roi_artifacts import auto_roi_evidence_payload, save_auto_roi_overlay
from roi_image_edit.failure_artifacts import failed_image_result
from roi_image_edit.iterative_pipeline import VisionClient, clamp_box, write_json
from roi_image_edit.pre_candidate_gates import pre_candidate_gate_report
from roi_image_edit.result_previews import rejected_region_preview_candidate
from roi_image_edit.roi_locator import auto_orient_for_instruction, parse_instruction_details
from roi_image_edit.run_artifacts import (
    delivery_artifact_manifest,
    progress_record,
    request_audit_payload,
    result_audit_payload,
)
from roi_image_edit.stage_profiles import resolve_stage_profile
from roi_image_edit.region_processing import (
    ENV_PATH,
    OUTPUT_DIR,
    candidate_trace_summary,
    compare_region_preview,
    image_from_data_url,
    image_to_data_url,
    load_processing_prompts,
    prior_stage_regression_report,
    process_region,
    run_region_vision_checks,
    save_stage_candidate_evidence,
)

ProgressCallback = Callable[[str, dict[str, Any]], None]


def process_payload(payload: dict[str, Any], progress: ProgressCallback | None = None) -> dict[str, Any]:
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.jsonl"
    profile_resolution = resolve_stage_profile(
        str(payload.get("profile") or ""),
        str(payload.get("profileSuggestion") or ""),
    )
    pipeline_profile = str(profile_resolution["id"])

    def emit(event: str, fields: dict[str, Any] | None = None) -> None:
        record = progress_record(
            event,
            fields,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        )
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if progress:
            progress(event, record)

    write_json(run_dir / "request.json", request_audit_payload(payload))
    emit(
        "run_started",
        {
            "run_dir": str(run_dir),
            "pipeline_profile": pipeline_profile,
            "profile_resolution": profile_resolution,
        },
    )
    prompts = load_processing_prompts()
    vision_client = VisionClient(ENV_PATH)
    results: list[dict[str, Any]] = []
    for image_item in payload.get("images", []):
        image_id = str(image_item.get("id") or "")
        filename = str(image_item.get("filename") or "image.png")
        safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(filename).stem)[:80] or image_id or "image"
        instruction_details: dict[str, Any] | None = None
        failure_image: Image.Image | None = None
        pre_candidate_report: dict[str, Any] | None = None
        orientation_summary: dict[str, Any] | None = None
        try:
            instruction_details = parse_instruction_details(str(image_item.get("instruction") or ""))
            source_text = instruction_details["source_text"]
            target_text = instruction_details["target_text"]
            if not target_text:
                raise ValueError("missing replacement instruction")
            emit(
                "image_started",
                {
                    "image_id": image_id,
                    "filename": filename,
                    "instruction_details": instruction_details,
                    "source_text": source_text,
                    "target_text": target_text,
                    "pipeline_profile": pipeline_profile,
                },
            )
            image = image_from_data_url(str(image_item.get("dataUrl") or ""))
            failure_image = image.copy()
            original_image = image.copy()
            orientation_summary = {
                "applied": False,
                "orientation": "none",
                "attempts": [],
            }
            candidates: list[dict[str, Any]] = []
            region_results: list[dict[str, Any]] = []
            image_accepted = True
            display_image: Image.Image | None = None
            regions = list(image_item.get("regions", []))
            if not regions:
                try:
                    image, regions, orientation_summary = auto_orient_for_instruction(
                        image,
                        instruction=str(image_item.get("instruction") or ""),
                        source_text=source_text,
                        target_text=target_text,
                    )
                except Exception as exc:
                    pre_candidate_report = pre_candidate_gate_report(
                        candidate_count=0,
                        orientation_summary=orientation_summary,
                        regions=[],
                        failure_step="field_roi_selection",
                        error=str(exc),
                    )
                    emit(
                        "pre_candidate_gate_failed",
                        {
                            "image_id": image_id,
                            "candidate_count": 0,
                            "failed_gate": pre_candidate_report["failed_gate"],
                            "pre_candidate_gate_report": pre_candidate_report,
                        },
                    )
                    raise
                emit(
                    "auto_roi_finished",
                    {
                        "image_id": image_id,
                        "orientation": orientation_summary.get("orientation"),
                        "direction_score": (orientation_summary.get("selected_attempt") or {}).get("direction_score"),
                        "selected_score": orientation_summary.get("selected_score"),
                        "attempt_count": len(orientation_summary.get("attempts") or []),
                        "region_count": len(regions),
                    },
                )
                original_image = image.copy()
            display_image = image.copy()
            for region in regions:
                rect = region.get("rect") or {}
                x = int(round(float(rect.get("x", 0))))
                y = int(round(float(rect.get("y", 0))))
                w = int(round(float(rect.get("w", 0))))
                h = int(round(float(rect.get("h", 0))))
                if w < 2 or h < 2:
                    continue
                roi = clamp_box((x, y, x + w, y + h), image.size)
                region_id = str(region.get("id") or f"region_{len(region_results) + 1}")
                region_source_text = str(region.get("sourceText") or source_text)
                region_target_text = str(region.get("targetText") or target_text)
                protected_texts = [
                    str(item)
                    for item in (
                        region.get("_autoProtectedTexts")
                        or (instruction_details or {}).get("protected_texts")
                        or []
                    )
                    if str(item)
                ]
                for key in ("_autoFieldLabelText", "_autoFieldSeparatorText"):
                    value = str(region.get(key) or "")
                    if value and value not in protected_texts:
                        protected_texts.append(value)
                field_context = {
                    "field_key": region.get("_autoFieldKey") or (instruction_details or {}).get("field_key"),
                    "field": region.get("_autoFieldKey") or (instruction_details or {}).get("field"),
                    "field_label_text": region.get("_autoFieldLabelText") or (instruction_details or {}).get("field_label_text"),
                    "field_separator_text": region.get("_autoFieldSeparatorText")
                    or (instruction_details or {}).get("field_separator_text"),
                    "protected_texts": protected_texts,
                }
                emit(
                    "region_started",
                    {
                        "image_id": image_id,
                        "region_id": region_id,
                        "roi": list(roi),
                        "source_text": region_source_text,
                        "target_text": region_target_text,
                        "pipeline_profile": pipeline_profile,
                    },
                )

                def region_progress(event: str, fields: dict[str, Any]) -> None:
                    emit(event, {"image_id": image_id, **(fields or {})})

                image, region_display_image, region_candidates, summary, accepted = process_region(
                    image,
                    roi,
                    source_text=region_source_text,
                    target_text=region_target_text,
                    run_dir=run_dir,
                    region_id=region_id,
                    vision_client=vision_client,
                    prompts=prompts,
                    max_candidates=int(payload.get("maxCandidates") or 120),
                    vision_candidate_limit=int(payload.get("visionCandidateLimit") or 8),
                    max_revision_rounds=int(payload.get("maxRevisionRounds") or 8),
                    pipeline_profile=pipeline_profile,
                    progress=region_progress,
                    field_context=field_context,
                )
                display_image = image.copy() if accepted else region_display_image.copy()
                emit(
                    "region_finished",
                    {
                        "image_id": image_id,
                        "region_id": region_id,
                        "accepted": accepted,
                        "pipeline_profile": pipeline_profile,
                        "revision_rounds": len((summary.get("vision") or {}).get("revision_rounds", [])),
                        "blocking_stage": (summary.get("trace") or {}).get("final_blocking_stage"),
                        "stage_severity": (summary.get("trace") or {}).get("final_stage_severity"),
                        "stop_reason": (summary.get("trace") or {}).get("last_round_stop_reason"),
                    },
                )
                image_accepted = image_accepted and accepted
                visible_region_candidates = list(region_candidates)
                if not visible_region_candidates and not accepted:
                    rejected_preview = rejected_region_preview_candidate(
                        region_id=region_id,
                        summary=summary,
                        pipeline_profile=pipeline_profile,
                    )
                    if rejected_preview:
                        visible_region_candidates.append(rejected_preview)
                for candidate in visible_region_candidates:
                    candidate["regionId"] = region_id
                candidates.extend(visible_region_candidates)
                region_results.append(
                    {
                        "id": region_id,
                        "roi": list(roi),
                        "sourceText": region_source_text,
                        "targetText": region_target_text,
                        "auto": bool(region.get("auto")),
                        "accepted": accepted,
                        "summary": summary,
                    }
                )
            if not region_results:
                raise ValueError("no valid rectangles")

            auto_roi_evidence = auto_roi_evidence_payload(regions)
            auto_roi_overlay_path: Path | None = None
            if auto_roi_evidence["region_count"]:
                auto_roi_overlay_path = run_dir / f"{safe_stem}_auto_roi_overlay.png"
                save_auto_roi_overlay(original_image, regions, auto_roi_overlay_path)
            auto_orientation_report_path = run_dir / f"{safe_stem}_auto_orientation_report.json"
            auto_roi_evidence_report_path = run_dir / f"{safe_stem}_auto_roi_evidence.json"
            auto_roi_stage_evidence = {
                **auto_roi_evidence,
                "overlay_path": str(auto_roi_overlay_path) if auto_roi_overlay_path else None,
                "orientation_report_path": str(auto_orientation_report_path),
                "report_path": str(auto_roi_evidence_report_path),
            }
            write_json(auto_orientation_report_path, orientation_summary)
            write_json(auto_roi_evidence_report_path, auto_roi_stage_evidence)
            original_path = run_dir / f"{safe_stem}_original.png"
            final_path = run_dir / f"{safe_stem}_final.png"
            applied_path = run_dir / f"{safe_stem}_applied.png"
            original_image.save(original_path)
            image.save(applied_path)
            result_image = display_image or image
            result_image.save(final_path)
            results.append(
                {
                    "id": image_id,
                    "ok": True,
                    "accepted": image_accepted,
                    "applied": image_accepted,
                    "filename": filename,
                    "instructionDetails": instruction_details,
                    "sourceDataUrl": image_to_data_url(original_image),
                    "resultDataUrl": image_to_data_url(result_image),
                    "candidates": candidates[:5],
                    "orientation": orientation_summary,
                    "autoRoiEvidence": auto_roi_evidence,
                    "stage_evidence": {
                        "auto_roi": auto_roi_stage_evidence,
                    },
                    "regions": region_results,
                    "artifacts": {
                        "original": str(original_path),
                        "final": str(final_path),
                        "applied": str(applied_path),
                        "auto_orientation_report": str(auto_orientation_report_path),
                        "auto_roi_evidence_report": str(auto_roi_evidence_report_path),
                        "auto_roi_overlay": str(auto_roi_overlay_path) if auto_roi_overlay_path else None,
                        "final_is_rejected_candidate": not image_accepted,
                    },
                }
            )
            emit("image_finished", {"image_id": image_id, "accepted": image_accepted})
        except Exception as exc:
            failure_progress: dict[str, Any] = {"image_id": image_id, "error": str(exc)}
            if pre_candidate_report:
                failure_progress.update(
                    {
                        "failure_stage": "pre_candidate_generation",
                        "candidate_count": pre_candidate_report["candidate_count"],
                        "failed_gate": pre_candidate_report["failed_gate"],
                        "pre_candidate_gate_report": pre_candidate_report,
                    }
                )
            emit("image_failed", failure_progress)
            results.append(
                failed_image_result(
                    run_dir=run_dir,
                    filename=filename,
                    image_id=image_id,
                    error=str(exc),
                    image=failure_image,
                    instruction_details=instruction_details,
                    pre_candidate_gate_report=pre_candidate_report,
                    orientation_summary=orientation_summary,
                )
            )
    response = {
        "ok": True,
        "runDir": str(run_dir),
        "artifactManifest": str(run_dir / "artifact_manifest.json"),
        "profile": pipeline_profile,
        "profileResolution": profile_resolution,
        "images": results,
    }
    result_path = run_dir / "result.json"
    artifact_manifest_path = run_dir / "artifact_manifest.json"
    write_json(result_path, result_audit_payload(response))
    manifest = delivery_artifact_manifest(
        response,
        run_dir=run_dir,
        result_path=result_path,
        progress_path=progress_path,
    )
    write_json(artifact_manifest_path, manifest)
    emit(
        "run_finished",
        {
            "ok": True,
            "artifact_manifest": str(artifact_manifest_path),
            "all_explainable": manifest.get("all_explainable"),
            "missing_required": manifest.get("missing_required"),
        },
    )
    return response
