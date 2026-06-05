from __future__ import annotations

from typing import Any

from roi_image_edit.stage_policy import STAGE_ORDER, stage_optimization_summary
from roi_image_edit.stages import prompt_stage_context, stage_gate_for_report


VISION_CANDIDATE_MIN_LIMIT = 3
VISION_CANDIDATE_MAX_LIMIT = 8


def normalize_vision_candidate_limit(requested_limit: int | None, total_candidate_count: int) -> int:
    raw_limit = int(requested_limit or VISION_CANDIDATE_MAX_LIMIT)
    normalized = max(VISION_CANDIDATE_MIN_LIMIT, min(VISION_CANDIDATE_MAX_LIMIT, raw_limit))
    return normalized


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


def _positive_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _grid_direction_source(
    round_record: dict[str, Any],
    field_name: str,
    *,
    basis_blocking_stage: str | None,
) -> dict[str, Any] | None:
    grid_report = round_record.get(field_name)
    if not isinstance(grid_report, dict):
        return None
    budget = grid_report.get("budget") if isinstance(grid_report.get("budget"), dict) else {}
    candidate_count = max(
        _positive_int(grid_report.get("candidate_count")),
        _positive_int(budget.get("retained_count") if isinstance(budget, dict) else 0),
    )
    stage_id = grid_report.get("stage_id")
    stage_matches_basis = bool(basis_blocking_stage and stage_id == basis_blocking_stage)
    if not grid_report.get("enabled") or candidate_count <= 0 or not stage_matches_basis:
        return None
    return {
        "source": field_name,
        "stage_id": stage_id,
        "optimization_step": grid_report.get("optimization_step"),
        "candidate_count": candidate_count,
        "raw_candidate_budget": _positive_int(
            budget.get("raw_candidate_budget") if isinstance(budget, dict) else 0
        ),
        "retained_count": _positive_int(
            budget.get("retained_count") if isinstance(budget, dict) else 0
        ),
        "stage_matches_basis": stage_matches_basis,
    }


def _patch_direction_source(
    round_record: dict[str, Any],
    *,
    basis_blocking_stage: str | None,
) -> dict[str, Any] | None:
    stage_filter_report = round_record.get("stage_filter_report")
    if not isinstance(stage_filter_report, dict):
        return None
    accepted_count = max(
        _positive_int(stage_filter_report.get("accepted_count")),
        _positive_int(round_record.get("patch_count")),
    )
    stage_id = stage_filter_report.get("stage_id") or round_record.get("basis_blocking_stage")
    stage_matches_basis = bool(basis_blocking_stage and stage_id == basis_blocking_stage)
    if accepted_count <= 0 or not stage_matches_basis:
        return None
    patcher = stage_filter_report.get("patcher")
    optimization_steps: list[str] = []
    if isinstance(patcher, dict):
        optimization_steps = [
            str(step)
            for step in patcher.get("optimization_steps", [])
            if step
        ]
    return {
        "source": "stage_patcher_dispatch",
        "stage_id": stage_id,
        "optimization_step": (
            round_record.get("selected_optimization_step")
            or (optimization_steps[0] if optimization_steps else None)
        ),
        "optimization_steps": optimization_steps,
        "candidate_count": accepted_count,
        "accepted_patch_count": accepted_count,
        "stage_matches_basis": stage_matches_basis,
    }


def revision_round_continuation_contract(
    round_record: dict[str, Any],
    *,
    max_revision_rounds: int,
) -> dict[str, Any]:
    basis_blocking_stage = round_record.get("basis_blocking_stage")
    basis_stage = str(basis_blocking_stage) if basis_blocking_stage else None
    direction_sources = [
        source
        for source in (
            _grid_direction_source(
                round_record,
                "shape_candidate_grid",
                basis_blocking_stage=basis_stage,
            ),
            _grid_direction_source(
                round_record,
                "ink_gray_candidate_grid",
                basis_blocking_stage=basis_stage,
            ),
            _grid_direction_source(
                round_record,
                "photo_texture_candidate_grid",
                basis_blocking_stage=basis_stage,
            ),
            _patch_direction_source(round_record, basis_blocking_stage=basis_stage),
        )
        if source is not None
    ]
    has_stage_specific_direction = bool(direction_sources)
    selected_step = round_record.get("selected_optimization_step")
    return {
        "round": round_record.get("round"),
        "max_revision_rounds": max(1, int(max_revision_rounds or 1)),
        "max_rounds_is_strategy": False,
        "requires_stage_specific_candidate_direction": True,
        "basis_blocking_stage": basis_stage,
        "basis_stage_source": round_record.get("basis_stage_source"),
        "candidate_direction_sources": direction_sources,
        "has_stage_specific_candidate_direction": has_stage_specific_direction,
        "continuation_allowed": has_stage_specific_direction,
        "selected_optimization_step": selected_step,
        "selected_reason": round_record.get("selected_reason"),
        "missing_direction_reason": (
            None
            if has_stage_specific_direction
            else "no stage-specific candidate grid or accepted stage patch for current blocking stage"
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


def vision_candidate_request_payload(
    hard_reports: dict[str, Any],
    *,
    pipeline_profile: str,
    requested_vision_candidate_limit: int,
    total_candidate_count: int,
) -> dict[str, Any]:
    enriched = attach_stage_context_to_rank_report(
        hard_reports,
        pipeline_profile=pipeline_profile,
    )
    requested_limit = int(requested_vision_candidate_limit or 0)
    total_count = max(0, int(total_candidate_count or 0))
    effective_limit = normalize_vision_candidate_limit(requested_limit, total_count)
    candidate_count = int(enriched.get("candidate_count") or 0)
    stage_context_by_candidate = enriched.get("stage_context_by_candidate")
    stage_context_complete = (
        isinstance(stage_context_by_candidate, dict)
        and set(stage_context_by_candidate) == set(enriched.get("candidate_ids") or [])
    )
    return {
        **enriched,
        "requested_vision_candidate_limit": requested_limit,
        "vision_candidate_limit": effective_limit,
        "total_candidate_count": total_count,
        "candidate_count_within_limit": candidate_count <= effective_limit,
        "stage_context_complete": stage_context_complete,
    }
