from __future__ import annotations

from typing import Any


PRE_CANDIDATE_GATE_ORDER = (
    "orientation_check",
    "field_roi_selection",
    "slot_quality_gate",
    "protected_text_guard",
)

PROTECTED_TEXT_ISSUE_TYPES = {
    "slot_overlaps_protected_text",
    "right_boundary_too_close_to_protected_text",
}


def _issue_types(slot_quality_report: dict[str, Any] | None) -> set[str]:
    if not isinstance(slot_quality_report, dict):
        return set()
    return {
        str(issue.get("type"))
        for issue in slot_quality_report.get("issues") or []
        if isinstance(issue, dict) and issue.get("type")
    }


def classify_pre_candidate_slot_failure(slot_quality_report: dict[str, Any] | None) -> str | None:
    issue_types = _issue_types(slot_quality_report)
    if issue_types & PROTECTED_TEXT_ISSUE_TYPES:
        return "protected_text_guard"
    if isinstance(slot_quality_report, dict) and slot_quality_report.get("pass") is False:
        return "slot_quality_gate"
    return None


def pre_candidate_gate_report(
    *,
    candidate_count: int,
    orientation_summary: dict[str, Any] | None = None,
    regions: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    slot_quality_report: dict[str, Any] | None = None,
    failure_step: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    region_count = len(regions or [])
    slot_failure = classify_pre_candidate_slot_failure(slot_quality_report)
    failed_gate = failure_step or slot_failure

    statuses: dict[str, dict[str, Any]] = {
        "orientation_check": {
            "pass": False if failed_gate == "orientation_check" else True,
            "reason": "orientation_selected_or_not_required",
            "orientation": (orientation_summary or {}).get("orientation"),
            "attempt_count": len((orientation_summary or {}).get("attempts") or []),
        },
        "field_roi_selection": {
            "pass": bool(region_count) and failed_gate != "field_roi_selection",
            "reason": "field_roi_selected" if region_count else "field_roi_not_found",
            "region_count": region_count,
        },
        "slot_quality_gate": {
            "pass": None,
            "reason": "not_checked_until_region_plan",
        },
        "protected_text_guard": {
            "pass": None,
            "reason": "not_checked_until_region_plan",
        },
    }

    if isinstance(slot_quality_report, dict):
        issue_types = sorted(_issue_types(slot_quality_report))
        protected_failed = bool(set(issue_types) & PROTECTED_TEXT_ISSUE_TYPES)
        slot_failed = bool(slot_quality_report.get("pass") is False and not protected_failed)
        statuses["slot_quality_gate"] = {
            "pass": not slot_failed,
            "reason": "slot_quality_failed" if slot_failed else "slot_quality_passed_or_deferred_to_protected_guard",
            "issue_types": issue_types,
        }
        statuses["protected_text_guard"] = {
            "pass": not protected_failed,
            "reason": "protected_text_conflict" if protected_failed else "protected_text_unchanged_before_candidates",
            "issue_types": [item for item in issue_types if item in PROTECTED_TEXT_ISSUE_TYPES],
        }

    if failed_gate is None:
        for gate in PRE_CANDIDATE_GATE_ORDER:
            if statuses[gate].get("pass") is False:
                failed_gate = gate
                break

    return {
        "gate_order": list(PRE_CANDIDATE_GATE_ORDER),
        "pass": failed_gate is None,
        "failed_gate": failed_gate,
        "failure_stage": "pre_candidate_generation" if failed_gate else None,
        "candidate_count": int(candidate_count),
        "statuses": statuses,
        "error": error,
    }
