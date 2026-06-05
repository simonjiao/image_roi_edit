from __future__ import annotations

from typing import Any

from roi_image_edit.stage_policy import STAGE_ORDER, stage_optimization_summary
from roi_image_edit.stages import prompt_stage_context, stage_gate_for_report


def request_audit_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "profile": payload.get("profile"),
        "profileSuggestion": payload.get("profileSuggestion"),
        "maxCandidates": payload.get("maxCandidates"),
        "visionCandidateLimit": payload.get("visionCandidateLimit"),
        "maxRevisionRounds": payload.get("maxRevisionRounds"),
        "images": [
            {
                "id": str(item.get("id") or ""),
                "filename": str(item.get("filename") or ""),
                "instruction": str(item.get("instruction") or ""),
                "regions": [
                    {
                        "id": str(region.get("id") or ""),
                        "rect": region.get("rect") or {},
                    }
                    for region in item.get("regions", [])
                ],
            }
            for item in payload.get("images", [])
        ],
    }


def result_audit_payload(response: dict[str, Any]) -> dict[str, Any]:
    images: list[dict[str, Any]] = []
    for item in response.get("images", []):
        image_record = {
            key: value
            for key, value in item.items()
            if key not in {"sourceDataUrl", "resultDataUrl", "candidates"}
        }
        image_record["candidates"] = [
            {key: value for key, value in candidate.items() if key != "dataUrl"}
            for candidate in item.get("candidates", [])
        ]
        images.append(image_record)
    return {
        "ok": response.get("ok"),
        "runDir": response.get("runDir"),
        "profile": response.get("profile"),
        "profileResolution": response.get("profileResolution"),
        "images": images,
    }


def stage_progress_fields(report: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    stage_gate = report.get("stage_gate")
    if not isinstance(stage_gate, dict):
        stage_gate = stage_gate_for_report(report)
    blocking_stage = stage_gate.get("blocking_stage") if isinstance(stage_gate, dict) else None
    blocking_status = (
        (stage_gate.get("stage_status") or {}).get(blocking_stage)
        if isinstance(stage_gate.get("stage_status"), dict) and blocking_stage
        else None
    )
    return {
        "pipeline_profile": report.get("pipeline_profile") or stage_gate.get("profile"),
        "stage_order": stage_gate.get("order"),
        "stage_status": stage_gate.get("stage_status"),
        "blocking_stage": blocking_stage,
        "blocking_stage_blocks_next": stage_gate.get("blocking_stage_blocks_next", False),
        "blocking_stage_reason": (
            blocking_status.get("reason")
            if isinstance(blocking_status, dict)
            else None
        ),
        "allowed_patch_keys": (
            blocking_status.get("allowed_patch_keys")
            if isinstance(blocking_status, dict)
            else []
        ),
        "blocked_patch_keys": (
            blocking_status.get("blocked_patch_keys")
            if isinstance(blocking_status, dict)
            else []
        ),
    }


def model_stage_context(report: dict[str, Any] | None, pipeline_profile: str) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {
            "pipeline_profile": pipeline_profile,
            "stage_order": list(STAGE_ORDER),
            "blocking_stage": None,
            "blocking_stage_blocks_next": False,
            "stage_status": {},
            "allowed_patch_keys": [],
            "blocked_patch_keys": [],
            "optimization_policy": stage_optimization_summary(None),
        }
    context = prompt_stage_context(report, pipeline_profile)
    blocking_stage = context.get("blocking_stage")
    context["optimization_policy"] = stage_optimization_summary(
        str(blocking_stage) if blocking_stage else None
    )
    return context


def attach_stage_context_to_rank_report(
    hard_reports: dict[str, Any],
    *,
    pipeline_profile: str,
) -> dict[str, Any]:
    enriched = dict(hard_reports)
    candidates = hard_reports.get("candidates")
    stage_context_by_candidate: dict[str, Any] = {}
    if isinstance(candidates, dict):
        for candidate_id, candidate in candidates.items():
            if not isinstance(candidate, dict):
                continue
            report = candidate.get("hard_check")
            stage_context_by_candidate[str(candidate_id)] = model_stage_context(
                report if isinstance(report, dict) else None,
                pipeline_profile,
            )
    candidate_ids = sorted(stage_context_by_candidate)
    enriched["pipeline_profile"] = pipeline_profile
    enriched["candidate_count"] = len(candidate_ids)
    enriched["candidate_ids"] = candidate_ids
    enriched["stage_context_by_candidate"] = stage_context_by_candidate
    enriched["stage_filter_contract"] = {
        "authoritative": "local_stage_filter",
        "rule": "Vision suggestions may diagnose any issue, but executable patches must stay within the current blocking stage allowed_patch_keys.",
    }
    return enriched
