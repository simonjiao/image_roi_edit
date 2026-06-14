from __future__ import annotations

from pathlib import Path
from typing import Any

from roi_image_edit.prompt_contracts import prompt_io_contract_report
from roi_image_edit.stage_policy import STAGE_ORDER, stage_optimization_summary
from roi_image_edit.stages import prompt_stage_context, stage_gate_for_report


VISION_CANDIDATE_MIN_LIMIT = 3
VISION_CANDIDATE_MAX_LIMIT = 8
EXTERNAL_ARTIFACT_SCHEMA_VERSION = 1


def external_artifact_schema_report() -> dict[str, Any]:
    return {
        "artifact_schema_version": EXTERNAL_ARTIFACT_SCHEMA_VERSION,
        "prompt_io_contract": prompt_io_contract_report(),
        "result_json": {
            "root_required": [
                "artifactSchemaVersion",
                "ok",
                "runDir",
                "artifactManifest",
                "profileResolution",
                "images",
            ],
            "image_required": [
                "id",
                "ok",
                "accepted",
                "applied",
                "instructionDetails",
                "classification",
                "class_key",
                "roi_policy",
                "internal_profile",
                "profile_source",
                "candidates",
                "stage_evidence",
                "regions",
                "artifacts",
            ],
            "candidate_required": [
                "id",
                "stage_context",
                "blocking_stage",
                "patch",
                "model_suggestions",
                "rejection_reason",
            ],
            "region_summary_required": [
                "plan",
                "hard_check",
                "vision",
                "trace",
                "artifacts",
            ],
            "vision_required": [
                "candidate_rank",
                "final_acceptance",
                "vision_target",
                "non_regression_guard",
                "revision_attempts",
                "revision_rounds",
                "vision_prompt_audits",
            ],
            "rejection_required": [
                "accepted",
                "applied",
                "final_is_rejected_candidate",
                "final_blocking_stage",
                "last_round_stop_reason",
            ],
        },
        "progress_jsonl": {
            "record_required": [
                "artifactSchemaVersion",
                "time",
                "event",
            ],
            "stage_fields": [
                "classification",
                "class_key",
                "roi_policy",
                "internal_profile",
                "profile_source",
                "pipeline_profile",
                "stage_order",
                "stage_status",
                "blocking_stage",
                "blocking_stage_reason",
                "allowed_patch_keys",
                "blocked_patch_keys",
            ],
            "candidate_fields": [
                "candidate_count",
                "rendered",
                "best_score",
                "stage_evidence",
            ],
            "patch_fields": [
                "stage_filter_report",
                "selected_optimization_step",
                "revision_continuation_contract",
            ],
            "vision_suggestion_fields": [
                "model_suggestion_filter",
                "model_stage_response_contract",
                "vision_disagreement",
                "vision_target",
                "vision_target_alignment",
                "non_regression_guard",
                "revision_attempts",
            ],
            "rejection_reason_fields": [
                "error",
                "stop_reason",
                "missing_direction_reason",
                "rejection_reason",
            ],
        },
        "artifact_manifest": {
            "root_required": [
                "artifactSchemaVersion",
                "manifest_type",
                "run_dir",
                "global_artifacts",
                "images",
                "all_explainable",
                "missing_required",
            ],
            "image_required": [
                "id",
                "status",
                "accepted",
                "applied",
                "blocking_stage",
                "reports",
                "candidate_images",
                "stage_evidence",
                "explainable",
                "missing_required",
            ],
        },
    }


def progress_record(event: str, fields: dict[str, Any] | None, *, timestamp: str) -> dict[str, Any]:
    return {
        "artifactSchemaVersion": EXTERNAL_ARTIFACT_SCHEMA_VERSION,
        "time": timestamp,
        "event": event,
        **(fields or {}),
    }


def normalize_vision_candidate_limit(requested_limit: int | None, total_candidate_count: int) -> int:
    raw_limit = int(requested_limit or VISION_CANDIDATE_MAX_LIMIT)
    normalized = max(VISION_CANDIDATE_MIN_LIMIT, min(VISION_CANDIDATE_MAX_LIMIT, raw_limit))
    return normalized


def request_audit_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "debugProfile": payload.get("debugProfile"),
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
        "artifactSchemaVersion": EXTERNAL_ARTIFACT_SCHEMA_VERSION,
        "ok": response.get("ok"),
        "runDir": response.get("runDir"),
        "artifactManifest": response.get("artifactManifest"),
        "profileResolution": response.get("profileResolution"),
        "images": images,
    }


def _artifact_path_entry(
    *,
    key: str,
    path: Any,
    purpose: str,
    required: bool,
) -> dict[str, Any]:
    path_str = str(path) if path else None
    exists = bool(path_str and Path(path_str).exists())
    return {
        "key": key,
        "path": path_str,
        "purpose": purpose,
        "required": bool(required),
        "exists": exists,
    }


def _append_path_entry(
    target: list[dict[str, Any]],
    *,
    key: str,
    path: Any,
    purpose: str,
    required: bool = False,
) -> None:
    if not path:
        if required:
            target.append(
                _artifact_path_entry(
                    key=key,
                    path=None,
                    purpose=purpose,
                    required=True,
                )
            )
        return
    target.append(
        _artifact_path_entry(
            key=key,
            path=path,
            purpose=purpose,
            required=required,
        )
    )


def _missing_required(entries: list[dict[str, Any]]) -> list[str]:
    return [
        str(entry.get("key") or "")
        for entry in entries
        if entry.get("required") and not entry.get("exists")
    ]


def _status_for_image(image: dict[str, Any]) -> str:
    if not image.get("ok"):
        return "failed"
    if image.get("accepted") and image.get("applied"):
        return "accepted"
    return "rejected"


def _region_blocking_stage(region: dict[str, Any]) -> str | None:
    summary = region.get("summary") if isinstance(region.get("summary"), dict) else {}
    trace = summary.get("trace") if isinstance(summary.get("trace"), dict) else {}
    if trace.get("final_blocking_stage"):
        return str(trace.get("final_blocking_stage"))
    hard_check = summary.get("hard_check") if isinstance(summary.get("hard_check"), dict) else {}
    stage_gate = hard_check.get("stage_gate") if isinstance(hard_check.get("stage_gate"), dict) else {}
    blocking_stage = stage_gate.get("blocking_stage")
    return str(blocking_stage) if blocking_stage else None


def _collect_region_explanation(region: dict[str, Any]) -> dict[str, Any]:
    summary = region.get("summary") if isinstance(region.get("summary"), dict) else {}
    plan = summary.get("plan") if isinstance(summary.get("plan"), dict) else {}
    vision = summary.get("vision") if isinstance(summary.get("vision"), dict) else {}
    vision_artifacts = (
        vision.get("artifacts")
        if isinstance(vision.get("artifacts"), dict)
        else {}
    )
    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
    stage_evidence = (
        artifacts.get("stage_evidence")
        if isinstance(artifacts.get("stage_evidence"), dict)
        else {}
    )
    reports: list[dict[str, Any]] = []
    candidate_images: list[dict[str, Any]] = []
    embedded_reports: list[dict[str, Any]] = []

    _append_path_entry(
        reports,
        key="roi_plan_report",
        path=artifacts.get("roi_plan_report"),
        purpose="Region search/edit ROI plan, manual exact/anchor policy, and expanded edit ROI diagnostics.",
        required=False,
    )
    _append_path_entry(
        reports,
        key="slot_quality_report",
        path=artifacts.get("slot_quality_report"),
        purpose="File copy of the old source slot completeness and protected-text safety gate.",
        required=False,
    )
    _append_path_entry(
        reports,
        key="pre_candidate_gate_report",
        path=artifacts.get("pre_candidate_gate_report"),
        purpose="File copy of the pre-candidate gate status for this region.",
        required=False,
    )
    _append_path_entry(
        reports,
        key="stage_evidence_summary",
        path=stage_evidence.get("summary"),
        purpose="Per-stage top candidate evidence index for this region.",
        required=False,
    )
    _append_path_entry(
        candidate_images,
        key="selected_candidate",
        path=artifacts.get("selected_candidate"),
        purpose="Final local candidate image selected for this region, even when rejected.",
        required=False,
    )
    _append_path_entry(
        candidate_images,
        key="selected_compare",
        path=artifacts.get("selected_compare"),
        purpose="Magnified comparison of original and selected region candidate.",
        required=bool(region),
    )
    _append_path_entry(
        candidate_images,
        key="vision_candidate_sheet",
        path=vision_artifacts.get("candidate_sheet"),
        purpose="Vision-stage sheet containing the local top candidates.",
        required=False,
    )
    _append_path_entry(
        candidate_images,
        key="vision_final_compare",
        path=vision_artifacts.get("final_compare"),
        purpose="Final visual acceptance comparison image.",
        required=False,
    )
    for idx, audit_path in enumerate(vision_artifacts.get("vision_prompt_audits") or [], start=1):
        _append_path_entry(
            reports,
            key=f"vision_prompt_audit_{idx}",
            path=audit_path,
            purpose="Audit record for the exact prompt text, prompt hashes, image hashes, and model JSON response fields.",
            required=False,
        )

    for preview in vision_artifacts.get("revision_previews") or []:
        if not isinstance(preview, dict):
            continue
        _append_path_entry(
            candidate_images,
            key=f"revision_preview_round_{preview.get('round')}",
            path=preview.get("path"),
            purpose="Selected revision-round comparison candidate.",
            required=False,
        )

    stage_records = (
        stage_evidence.get("stages")
        if isinstance(stage_evidence.get("stages"), dict)
        else {}
    )
    stage_top_candidates: list[dict[str, Any]] = []
    for stage_id in STAGE_ORDER:
        stage_record = stage_records.get(stage_id)
        if not isinstance(stage_record, dict):
            continue
        record = {
            "stage_id": stage_id,
            "available": bool(stage_record.get("available")),
            "reason": stage_record.get("reason"),
        }
        if stage_record.get("available"):
            _append_path_entry(
                candidate_images,
                key=f"{stage_id}_top_compare",
                path=stage_record.get("compare_path"),
                purpose=f"Top local candidate comparison blocked at {stage_id}.",
                required=False,
            )
            _append_path_entry(
                reports,
                key=f"{stage_id}_top_report",
                path=stage_record.get("report_path"),
                purpose=f"Hard/local report for the top {stage_id} candidate.",
                required=False,
            )
            record["compare_path"] = stage_record.get("compare_path")
            record["report_path"] = stage_record.get("report_path")
        stage_top_candidates.append(record)

    if isinstance(plan.get("slot_quality_report"), dict):
        embedded_reports.append(
            {
                "key": "slot_quality_report",
                "purpose": "Old source slot completeness and protected-text safety gate.",
                "available": True,
                "pass": plan["slot_quality_report"].get("pass"),
            }
        )
    if isinstance(plan.get("roi_plan"), dict):
        embedded_reports.append(
            {
                "key": "roi_plan",
                "purpose": "Region ROI policy and expanded edit ROI plan.",
                "available": True,
                "roi_policy": plan.get("roi_policy"),
                "expanded_edit_roi": plan.get("roi_plan", {}).get("expanded_edit_roi"),
            }
        )
    if isinstance(vision.get("vision_target"), dict) and vision["vision_target"].get("active"):
        embedded_reports.append(
            {
                "key": "vision_target",
                "purpose": "Final visual findings mapped back to the public five-stage workflow.",
                "available": True,
                "stage": vision["vision_target"].get("stage"),
                "axes": vision["vision_target"].get("axis_keys") or [],
                "stage_source": vision["vision_target"].get("stage_source"),
                "combo_recipe": vision["vision_target"].get("combo_recipe"),
            }
        )
    if isinstance(vision.get("non_regression_guard"), dict) and vision["non_regression_guard"].get("enabled"):
        embedded_reports.append(
            {
                "key": "non_regression_guard",
                "purpose": "Candidate before/after directions for user-visible visual regression axes.",
                "available": True,
                "pass": vision["non_regression_guard"].get("pass"),
                "axes": vision["non_regression_guard"].get("axes") or [],
                "direction": vision["non_regression_guard"].get("direction"),
            }
        )
    round_targets = [
        round_record.get("vision_target")
        for round_record in vision.get("revision_rounds") or []
        if isinstance(round_record, dict)
        and isinstance(round_record.get("vision_target"), dict)
        and round_record["vision_target"].get("active")
    ]
    if round_targets:
        embedded_reports.append(
            {
                "key": "vision_disagreement",
                "purpose": "Revision rounds where local stages passed but visual acceptance still rejected.",
                "available": True,
                "round_count": len(round_targets),
                "stages": sorted({str(target.get("stage")) for target in round_targets if target.get("stage")}),
            }
        )

    local_missing = _missing_required(reports + candidate_images)
    return {
        "id": region.get("id"),
        "accepted": bool(region.get("accepted")),
        "blocking_stage": _region_blocking_stage(region),
        "reports": reports,
        "candidate_images": candidate_images,
        "embedded_reports": embedded_reports,
        "stage_top_candidates": stage_top_candidates,
        "missing_required": local_missing,
        "explainable": not local_missing
        and (
            bool(reports)
            or bool(candidate_images)
            or bool(embedded_reports)
            or bool(stage_top_candidates)
        ),
    }


def _collect_image_explanation(image: dict[str, Any]) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    candidate_images: list[dict[str, Any]] = []
    embedded_reports: list[dict[str, Any]] = []
    stage_evidence = (
        image.get("stage_evidence")
        if isinstance(image.get("stage_evidence"), dict)
        else {}
    )
    artifacts = image.get("artifacts") if isinstance(image.get("artifacts"), dict) else {}
    status = _status_for_image(image)

    _append_path_entry(
        candidate_images,
        key="final_image",
        path=artifacts.get("final"),
        purpose="Displayed final image for this task.",
        required=status != "failed",
    )
    _append_path_entry(
        reports,
        key="classification_report",
        path=artifacts.get("classification_report"),
        purpose="Per-image automatic workflow classification and internal strategy selection evidence.",
        required=status != "failed",
    )
    _append_path_entry(
        candidate_images,
        key="applied_image",
        path=artifacts.get("applied"),
        purpose="Actually applied image. For rejected tasks this remains the original edit state.",
        required=False,
    )
    _append_path_entry(
        candidate_images,
        key="auto_roi_overlay",
        path=artifacts.get("auto_roi_overlay")
        or (
            (stage_evidence.get("auto_roi") or {}).get("overlay_path")
            if isinstance(stage_evidence.get("auto_roi"), dict)
            else None
        ),
        purpose="Search/edit ROI overlay used to explain automatic region selection.",
        required=False,
    )
    auto_roi_evidence = (
        stage_evidence.get("auto_roi")
        if isinstance(stage_evidence.get("auto_roi"), dict)
        else {}
    )
    _append_path_entry(
        reports,
        key="auto_orientation_report",
        path=artifacts.get("auto_orientation_report")
        or auto_roi_evidence.get("orientation_report_path"),
        purpose="Automatic orientation attempts, target-field quality, and selected orientation reason.",
        required=False,
    )
    _append_path_entry(
        reports,
        key="auto_roi_evidence_report",
        path=artifacts.get("auto_roi_evidence_report") or auto_roi_evidence.get("report_path"),
        purpose="Automatic search/edit ROI evidence and overlay path.",
        required=False,
    )
    _append_path_entry(
        candidate_images,
        key="rejected_input",
        path=artifacts.get("rejected_input"),
        purpose="Original input preserved when processing fails before candidate generation.",
        required=status == "failed",
    )
    _append_path_entry(
        reports,
        key="failure_report",
        path=artifacts.get("failure_report"),
        purpose="Failure report for tasks that stop before candidate generation.",
        required=status == "failed",
    )

    failure = stage_evidence.get("failure") if isinstance(stage_evidence.get("failure"), dict) else None
    if failure and isinstance(failure.get("pre_candidate_gate_report"), dict):
        embedded_reports.append(
            {
                "key": "pre_candidate_gate_report",
                "purpose": "Pre-candidate gate failure reason and gate order.",
                "available": True,
                "failed_gate": failure["pre_candidate_gate_report"].get("failed_gate"),
            }
        )
    if isinstance(image.get("classification"), dict):
        embedded_reports.append(
            {
                "key": "classification",
                "purpose": "Automatic workflow classification embedded in result.json.",
                "available": True,
                "class_key": image.get("class_key") or image["classification"].get("class_key"),
                "internal_profile": image.get("internal_profile") or image["classification"].get("internal_profile"),
                "roi_policy": image.get("roi_policy") or image["classification"].get("roi_policy"),
            }
        )

    regions = [
        _collect_region_explanation(region)
        for region in image.get("regions", [])
        if isinstance(region, dict)
    ]
    for region in regions:
        reports.extend(region.get("reports") or [])
        candidate_images.extend(region.get("candidate_images") or [])
        embedded_reports.extend(region.get("embedded_reports") or [])

    blocking_stages = [
        region.get("blocking_stage")
        for region in regions
        if region.get("blocking_stage")
    ]
    failure_stage = failure.get("failure_stage") if isinstance(failure, dict) else None
    missing = _missing_required(reports + candidate_images)
    if status == "failed" and not embedded_reports and not reports:
        missing.append("failure_explanation")
    if status != "failed" and not regions:
        missing.append("region_explanation")
    has_candidate_evidence = bool(candidate_images) or status == "failed"
    has_report_evidence = bool(reports) or bool(embedded_reports)
    return {
        "id": image.get("id"),
        "status": status,
        "ok": bool(image.get("ok")),
        "accepted": bool(image.get("accepted")),
        "applied": bool(image.get("applied")),
        "blocking_stage": blocking_stages[0] if blocking_stages else failure_stage,
        "reports": reports,
        "candidate_images": candidate_images,
        "embedded_reports": embedded_reports,
        "stage_evidence": {
            "auto_roi": stage_evidence.get("auto_roi"),
            "failure": failure,
            "regions": regions,
        },
        "explanation_sources": [
            source
            for source, available in (
                ("result_json", True),
                ("progress_jsonl", True),
                ("stage_evidence", bool(stage_evidence) or bool(regions)),
                ("candidate_images", has_candidate_evidence),
                ("reports", has_report_evidence),
            )
            if available
        ],
        "missing_required": sorted(set(item for item in missing if item)),
        "explainable": not missing and has_candidate_evidence and has_report_evidence,
    }


def delivery_artifact_manifest(
    response: dict[str, Any],
    *,
    run_dir: Path,
    result_path: Path,
    progress_path: Path,
) -> dict[str, Any]:
    global_artifacts = [
        _artifact_path_entry(
            key="result_json",
            path=result_path,
            purpose="Stable task result and per-image acceptance/rejection summary.",
            required=True,
        ),
        _artifact_path_entry(
            key="progress_jsonl",
            path=progress_path,
            purpose="Stable progress stream showing stage rounds, failures, and stop reasons.",
            required=True,
        ),
    ]
    image_records = [
        _collect_image_explanation(image)
        for image in response.get("images", [])
        if isinstance(image, dict)
    ]
    missing = _missing_required(global_artifacts)
    for image in image_records:
        if not image.get("explainable"):
            missing.append(f"image:{image.get('id')}:explanation")
        for item in image.get("missing_required") or []:
            missing.append(f"image:{image.get('id')}:{item}")
    return {
        "artifactSchemaVersion": EXTERNAL_ARTIFACT_SCHEMA_VERSION,
        "manifest_type": "delivery_explanation",
        "run_dir": str(run_dir),
        "global_artifacts": global_artifacts,
        "images": image_records,
        "all_explainable": not missing,
        "missing_required": sorted(set(item for item in missing if item)),
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
        "classification": report.get("classification"),
        "class_key": report.get("class_key") or (report.get("classification") or {}).get("class_key")
        if isinstance(report.get("classification"), dict)
        else report.get("class_key"),
        "roi_policy": report.get("roi_policy") or (report.get("classification") or {}).get("roi_policy")
        if isinstance(report.get("classification"), dict)
        else report.get("roi_policy"),
        "internal_profile": report.get("internal_profile") or (report.get("classification") or {}).get("internal_profile")
        if isinstance(report.get("classification"), dict)
        else report.get("internal_profile"),
        "profile_source": report.get("profile_source") or (report.get("classification") or {}).get("profile_source")
        if isinstance(report.get("classification"), dict)
        else report.get("profile_source"),
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
    guards_stage = grid_report.get("guards_stage")
    stage_matches_basis = bool(
        basis_blocking_stage
        and (stage_id == basis_blocking_stage or guards_stage == basis_blocking_stage)
    )
    if not grid_report.get("enabled") or candidate_count <= 0 or not stage_matches_basis:
        return None
    return {
        "source": field_name,
        "stage_id": stage_id,
        "guards_stage": guards_stage,
        "guard_mode": grid_report.get("guard_mode"),
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


def _controlled_escape_direction_source(
    round_record: dict[str, Any],
    *,
    basis_blocking_stage: str | None,
) -> dict[str, Any] | None:
    escape_report = round_record.get("controlled_escape_grid")
    if not isinstance(escape_report, dict):
        return None
    candidate_count = _positive_int(escape_report.get("candidate_count"))
    primary_stage = escape_report.get("primary_stage")
    stage_matches_basis = bool(basis_blocking_stage and primary_stage == basis_blocking_stage)
    if not escape_report.get("enabled") or candidate_count <= 0 or not stage_matches_basis:
        return None
    return {
        "source": "controlled_escape_grid",
        "stage_id": primary_stage,
        "secondary_stage": escape_report.get("secondary_stage"),
        "optimization_step": "controlled_escape",
        "escape_strategy": escape_report.get("escape_strategy"),
        "candidate_count": candidate_count,
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
                "ink_guard_candidate_grid",
                basis_blocking_stage=basis_stage,
            ),
            _grid_direction_source(
                round_record,
                "photo_texture_candidate_grid",
                basis_blocking_stage=basis_stage,
            ),
            _controlled_escape_direction_source(
                round_record,
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
        context = prompt_stage_context(
            {"pass": True, "pipeline_profile": pipeline_profile},
            pipeline_profile,
        )
        context["optimization_policy"] = stage_optimization_summary(None)
        return context
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
