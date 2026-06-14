from __future__ import annotations

from typing import Any

from roi_image_edit.stage_policy import STAGE_ORDER


TEXT_SHAPE_VISUAL_ARBITRATION_ISSUE_TYPES = frozenset(
    {
        "font_style_score_too_high",
        "font_style_mismatch",
        "font_family_style_score_ratio",
        "font_render_style_score_ratio",
        "changed_char_stroke_body_too_small",
        "changed_char_stroke_body_too_narrow",
        "changed_char_fine_strokes_too_soft",
        "ink_area_ratio_too_low",
    }
)


def _clean(value: Any) -> str:
    return str(value or "").strip().lower()


def text_shape_visual_arbitration_candidate(
    report: dict[str, Any] | None,
    stage_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {"eligible": False, "reason": "missing_report"}
    if not bool(report.get("pass")):
        return {"eligible": False, "reason": "hard_boundary_failed"}
    gate = stage_gate if isinstance(stage_gate, dict) else report.get("stage_gate")
    if not isinstance(gate, dict):
        return {"eligible": False, "reason": "missing_stage_gate"}
    if gate.get("blocking_stage") != "text_shape":
        return {
            "eligible": False,
            "reason": "local_blocking_stage_is_not_text_shape",
            "blocking_stage": gate.get("blocking_stage"),
        }
    stage_status = gate.get("stage_status") if isinstance(gate.get("stage_status"), dict) else {}
    text_shape = stage_status.get("text_shape") if isinstance(stage_status, dict) else {}
    issues = text_shape.get("issues") if isinstance(text_shape, dict) else []
    issue_items = [issue for issue in issues if isinstance(issue, dict)]
    if not issue_items:
        return {"eligible": False, "reason": "missing_text_shape_issues"}
    issue_types = [str(issue.get("type") or "") for issue in issue_items]
    non_arbitrable = [
        issue_type
        for issue_type in issue_types
        if issue_type not in TEXT_SHAPE_VISUAL_ARBITRATION_ISSUE_TYPES
        and not issue_type.startswith("font_")
    ]
    if non_arbitrable:
        return {
            "eligible": False,
            "reason": "non_arbitrable_text_shape_issues",
            "issue_types": issue_types,
            "non_arbitrable_issue_types": non_arbitrable,
        }
    return {
        "eligible": True,
        "reason": "text_shape_issues_are_visual_arbitrable",
        "issue_types": issue_types,
        "issue_count": len(issue_items),
        "deferred_issues": issue_items,
    }


def visual_shape_findings_pass(acceptance: dict[str, Any] | None) -> dict[str, Any]:
    findings = acceptance.get("visual_findings") if isinstance(acceptance, dict) else {}
    if not isinstance(findings, dict):
        return {"pass": False, "reason": "missing_visual_findings"}
    shape_values = {
        "char_positions": _clean(findings.get("char_positions") or findings.get("position")),
        "spacing": _clean(findings.get("spacing")),
        "baseline": _clean(findings.get("baseline")),
        "font_similarity": _clean(findings.get("font_similarity")),
        "size": _clean(findings.get("size")),
        "stroke_weight": _clean(findings.get("stroke_weight")),
    }
    missing = [key for key, value in shape_values.items() if not value]
    bad = {
        key: value
        for key, value in shape_values.items()
        if value and value not in {"ok", "pass"}
    }
    if missing:
        return {
            "pass": False,
            "reason": "missing_shape_visual_fields",
            "missing_fields": missing,
            "shape_values": shape_values,
        }
    if bad:
        return {
            "pass": False,
            "reason": "visual_shape_fields_not_ok",
            "bad_fields": bad,
            "shape_values": shape_values,
        }
    return {
        "pass": True,
        "reason": "visual_shape_fields_ok",
        "shape_values": shape_values,
    }


def local_shape_visual_arbitration_report(
    report: dict[str, Any] | None,
    acceptance: dict[str, Any] | None,
    *,
    stage_gate: dict[str, Any] | None = None,
    ink_issues: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    candidate = text_shape_visual_arbitration_candidate(report, stage_gate)
    visual = visual_shape_findings_pass(acceptance)
    issue_items = [issue for issue in (ink_issues or []) if isinstance(issue, dict)]
    if not candidate.get("eligible"):
        return {"active": False, "reason": candidate.get("reason"), "candidate": candidate, "visual_shape": visual}
    if not visual.get("pass"):
        return {"active": False, "reason": visual.get("reason"), "candidate": candidate, "visual_shape": visual}
    if not issue_items:
        return {
            "active": False,
            "reason": "no_ink_gray_issue_to_advance_to",
            "candidate": candidate,
            "visual_shape": visual,
        }
    return {
        "active": True,
        "reason": "vision_shape_passed_local_text_shape_arbitrable_issues_advance_to_ink",
        "from_stage": "text_shape",
        "advance_to_stage": "ink_gray_balance",
        "candidate": candidate,
        "visual_shape": visual,
        "deferred_issues": candidate.get("deferred_issues") or [],
        "ink_issue_types": [str(issue.get("type") or "") for issue in issue_items],
        "policy": "hard_boundary remains local; only visual-arbitrable text_shape issues can be deferred after explicit Vision shape pass",
    }


def acceptance_shape_arbitration_advance_stage(acceptance: dict[str, Any] | None) -> str | None:
    arbitration = acceptance.get("local_shape_arbitration") if isinstance(acceptance, dict) else None
    if not isinstance(arbitration, dict) or not arbitration.get("active"):
        return None
    stage = str(arbitration.get("advance_to_stage") or "").strip()
    return stage if stage in STAGE_ORDER else None
