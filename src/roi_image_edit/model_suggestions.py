from __future__ import annotations

from copy import deepcopy
from typing import Any

from roi_image_edit.stage_patchers import patch_signature
from roi_image_edit.stage_policy import optimization_policy_audit


def _local_blocking_stage(stage_id: str | None) -> str | None:
    value = str(stage_id or "").strip()
    return value or None


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


def model_suggestion_filter_report(filter_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage_id": filter_result.get("stage_id"),
        "record_count": filter_result.get("record_count", 0),
        "accepted_count": filter_result.get("accepted_count", 0),
        "rejected_count": filter_result.get("rejected_count", 0),
        "attempt_records": filter_result.get("attempt_records") or [],
    }
