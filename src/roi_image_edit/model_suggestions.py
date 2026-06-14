from __future__ import annotations

from copy import deepcopy
from typing import Any

from roi_image_edit.stage_patchers import patch_signature
from roi_image_edit.stage_policy import optimization_policy_audit


def _local_blocking_stage(stage_id: str | None) -> str | None:
    value = str(stage_id or "").strip()
    return value or None


def model_stage_response_contract(
    model_json: dict[str, Any] | None,
    local_stage_id: str | None,
) -> dict[str, Any]:
    local_stage = _local_blocking_stage(local_stage_id)
    if not isinstance(model_json, dict):
        model_json = {}
    assessment = model_json.get("stage_assessment")
    if not isinstance(assessment, dict):
        assessment = {}
    stage_exists_value = assessment.get("blocking_stage_exists")
    stage_exists_declared = stage_exists_value if isinstance(stage_exists_value, bool) else None
    current_stage_declared = assessment.get("current_blocking_stage")
    if current_stage_declared is not None:
        current_stage_declared = str(current_stage_declared or "").strip() or None
    suggestion_target_stage = assessment.get("suggestion_target_stage")
    if suggestion_target_stage is None:
        suggestion_target_stage = model_json.get("blocking_stage")
    suggestion_target_stage = str(suggestion_target_stage or "").strip() or None
    model_blocking_stage = str(model_json.get("blocking_stage") or "").strip() or None
    basis = assessment.get("basis")
    if not isinstance(basis, str) or not basis.strip():
        basis = model_json.get("reason")
    basis_text = str(basis or "").strip()
    local_stage_exists = local_stage is not None
    return {
        "local_blocking_stage": local_stage,
        "local_blocking_stage_exists": local_stage_exists,
        "model_blocking_stage": model_blocking_stage,
        "stage_assessment_present": bool(model_json.get("stage_assessment")),
        "blocking_stage_exists_declared": stage_exists_declared,
        "blocking_stage_exists_matches_local": (
            stage_exists_declared == local_stage_exists if stage_exists_declared is not None else False
        ),
        "current_blocking_stage_declared": current_stage_declared,
        "current_blocking_stage_matches_local": current_stage_declared == local_stage,
        "suggestion_target_stage": suggestion_target_stage,
        "suggestion_targets_current_stage": (
            suggestion_target_stage == local_stage
            if local_stage_exists
            else suggestion_target_stage is None
        ),
        "basis": basis_text,
        "basis_present": bool(basis_text),
        "schema_complete": (
            stage_exists_declared is not None
            and current_stage_declared == local_stage
            and bool(basis_text)
        ),
    }


def _suggestion_attempt_record(record: dict[str, Any]) -> dict[str, Any]:
    policy = record.get("optimization_policy")
    if not isinstance(policy, dict):
        policy = {}
    accepted = bool(record.get("accepted_for_candidate_generation"))
    return {
        "source": record.get("source"),
        "kind": record.get("kind"),
        "source_index": record.get("index"),
        "model_blocking_stage": record.get("blocking_stage"),
        "local_blocking_stage": record.get("local_filter_stage"),
        "patch": record.get("patch") if isinstance(record.get("patch"), dict) else {},
        "optimization_policy": policy,
        "accepted_for_candidate_generation": accepted,
        "rejection_reason": None if accepted else record.get("local_filter_reason"),
    }


def filter_model_patch_records(
    records: list[dict[str, Any]],
    stage_id: str | None,
) -> dict[str, Any]:
    """Apply the local stage policy to model-suggested patches.

    Vision output is advisory. This function is the stable boundary that turns
    model JSON into locally allowed revision patches, while preserving every
    rejected suggestion as auditable evidence for progress/result records.
    """

    local_stage = _local_blocking_stage(stage_id)
    audited_records: list[dict[str, Any]] = []
    allowed_patches: list[dict[str, Any]] = []
    rejected_records: list[dict[str, Any]] = []
    patch_source_lookup: dict[str, list[dict[str, Any]]] = {}

    for index, raw_record in enumerate(records, start=1):
        if not isinstance(raw_record, dict):
            continue
        record = deepcopy(raw_record)
        record["filter_index"] = index
        record["local_filter_stage"] = local_stage
        patch = record.get("patch")
        if not isinstance(patch, dict) or not patch:
            policy_audit = optimization_policy_audit(local_stage, None)
            conversion_reason = str(
                record.get("conversion_reason") or "model suggestion did not contain a usable patch"
            )
            record["optimization_policy"] = policy_audit
            record["accepted_for_candidate_generation"] = False
            record["local_filter_decision"] = "rejected"
            record["local_filter_reason"] = conversion_reason
            audited_records.append(record)
            rejected_records.append(record)
            continue

        policy_audit = optimization_policy_audit(local_stage, patch)
        accepted = bool(policy_audit.get("allowed"))
        record["optimization_policy"] = policy_audit
        record["accepted_for_candidate_generation"] = accepted
        record["local_filter_decision"] = "accepted" if accepted else "rejected"
        record["local_filter_reason"] = policy_audit.get("rejection_reason") or "allowed"
        patch_source_lookup.setdefault(patch_signature(patch), []).append(record)
        audited_records.append(record)
        if accepted:
            allowed_patches.append(patch)
        else:
            rejected_records.append(record)

    attempt_records = [_suggestion_attempt_record(record) for record in audited_records]
    return {
        "stage_id": local_stage,
        "record_count": len(audited_records),
        "accepted_count": len(allowed_patches),
        "rejected_count": len(rejected_records),
        "records": audited_records,
        "allowed_patches": allowed_patches,
        "rejected_records": rejected_records,
        "attempt_records": attempt_records,
        "patch_source_lookup": patch_source_lookup,
    }


def combined_model_suggestion_patch(
    filter_result: dict[str, Any],
    *,
    source: str | None = None,
) -> dict[str, Any]:
    """Merge same-response accepted model parameter suggestions into one patch."""

    local_stage = _local_blocking_stage(filter_result.get("stage_id"))
    records = filter_result.get("records")
    if not isinstance(records, list):
        records = []

    merged: dict[str, Any] = {}
    source_records: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        if source is not None and record.get("source") != source:
            continue
        if not record.get("accepted_for_candidate_generation"):
            continue
        patch = record.get("patch")
        if not isinstance(patch, dict) or not patch:
            continue
        source_records.append(record)
        for key, value in patch.items():
            key_text = str(key)
            if key_text in merged and merged[key_text] != value:
                conflicts.append(
                    {
                        "parameter": key_text,
                        "existing": merged[key_text],
                        "incoming": value,
                        "source": record.get("source"),
                        "index": record.get("index"),
                    }
                )
                continue
            merged[key_text] = value

    if conflicts:
        return {
            "enabled": False,
            "source": source,
            "stage_id": local_stage,
            "patch": {},
            "record_count": len(source_records),
            "conflicts": conflicts,
            "reason": "conflicting suggestions for the same parameter",
        }
    if len(merged) < 2:
        return {
            "enabled": False,
            "source": source,
            "stage_id": local_stage,
            "patch": dict(merged),
            "record_count": len(source_records),
            "conflicts": [],
            "reason": "fewer than two accepted suggestions from the same source",
        }

    policy_audit = optimization_policy_audit(local_stage, merged)
    enabled = bool(policy_audit.get("allowed"))
    return {
        "enabled": enabled,
        "source": source,
        "stage_id": local_stage,
        "patch": dict(merged) if enabled else {},
        "record_count": len(source_records),
        "conflicts": [],
        "optimization_policy": policy_audit,
        "source_records": source_records,
        "reason": "allowed" if enabled else policy_audit.get("rejection_reason"),
    }


def model_suggestion_filter_report(filter_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage_id": filter_result.get("stage_id"),
        "record_count": filter_result.get("record_count", 0),
        "accepted_count": filter_result.get("accepted_count", 0),
        "rejected_count": filter_result.get("rejected_count", 0),
        "attempt_records": filter_result.get("attempt_records") or [],
    }
