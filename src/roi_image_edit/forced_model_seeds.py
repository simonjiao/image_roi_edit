from __future__ import annotations

from typing import Any

from roi_image_edit.iterative_pipeline import CandidateParams
from roi_image_edit.stage_patchers import patch_signature


def _suggestion_key(record: dict[str, Any]) -> str:
    patch = record.get("patch") if isinstance(record.get("patch"), dict) else None
    if patch:
        return f"patch:{patch_signature(patch)}"
    suggestion = record.get("suggestion")
    if isinstance(suggestion, dict):
        to_val = suggestion.get("to")
        delta_val = suggestion.get("delta")
        param = suggestion.get("parameter") or suggestion.get("param", "")
        return f"suggestion:{param}:{to_val}:{delta_val}"
    return f"unknown:{record.get('index', '?')}"


def forced_model_seed_audit(
    model_records: list[dict[str, Any]],
    allowed_patches: list[dict[str, Any]],
    seen_params: set[str],
    stage_filter_report: dict[str, Any] | None,
) -> dict[str, Any]:
    audit_records: list[dict[str, Any]] = []
    forced_seeds: list[dict[str, Any]] = []
    seen_suggestion_keys: set[str] = set()

    rejected_patches = stage_filter_report.get("rejected_patches", []) if isinstance(stage_filter_report, dict) else []
    rejected_sigs = {patch_signature(p) for p in rejected_patches if isinstance(p, dict)}

    for record in model_records:
        if not isinstance(record, dict):
            continue
        source = record.get("source", "unknown")
        kind = record.get("kind", "unknown")
        patch = record.get("patch") if isinstance(record.get("patch"), dict) else None
        suggestion = record.get("suggestion")
        raw_patch = dict(patch) if patch else (dict(suggestion) if isinstance(suggestion, dict) else None)

        record_key = _suggestion_key(record)

        if not patch and not isinstance(suggestion, dict):
            audit_records.append({
                "source": source,
                "kind": kind,
                "index": record.get("index"),
                "raw_suggestion": suggestion,
                "converted": False,
                "conversion_reason": record.get("conversion_reason", "no convertible suggestion"),
                "rendered": False,
                "selectable": False,
                "rejection_reason": record.get("conversion_reason", "unconvertible suggestion"),
            })
            continue

        converted = bool(patch)
        sig = patch_signature(patch) if patch else None

        if record_key in seen_suggestion_keys:
            audit_records.append({
                "source": source,
                "kind": kind,
                "raw_patch": raw_patch,
                "converted": converted,
                "converted_patch": patch,
                "deduped": True,
                "deduped_to_key": record_key,
                "rendered": False,
                "selectable": False,
                "rejection_reason": "deduplicated model suggestion",
            })
            continue
        seen_suggestion_keys.add(record_key)

        audit_entry: dict[str, Any] = {
            "source": source,
            "kind": kind,
            "index": record.get("index"),
            "raw_patch": raw_patch,
            "converted": converted,
        }
        if patch:
            audit_entry["converted_patch"] = patch

        if not converted:
            audit_entry["rendered"] = False
            audit_entry["selectable"] = False
            audit_entry["rejection_reason"] = record.get("conversion_reason", "unconvertible")
            audit_records.append(audit_entry)
            continue

        if sig and sig in rejected_sigs:
            audit_entry["rendered"] = False
            audit_entry["selectable"] = False
            audit_entry["rejection_reason"] = "rejected by stage/profile filter"
            audit_records.append(audit_entry)
            continue

        forced_seeds.append({
            "source": source,
            "kind": kind,
            "index": record.get("index"),
            "raw_suggestion": raw_patch,
            "converted_patch": patch,
            "converted": True,
            "deduped": False,
        })

    return {
        "forced_seed_count": len(forced_seeds),
        "total_model_suggestions": len(model_records),
        "audited_suggestions": audit_records,
        "forced_seeds": forced_seeds,
    }
