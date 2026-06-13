from __future__ import annotations

from typing import Any

from roi_image_edit.iterative_pipeline import RenderPlan, TextRun, is_mostly_cjk


def _text_chars(value: str) -> list[str]:
    return [ch for ch in str(value or "") if not ch.isspace()]


def _shape_change_large(slot_report: dict[str, Any]) -> bool | None:
    shape_report = slot_report.get("shape_change_report")
    if isinstance(shape_report, dict) and isinstance(shape_report.get("shape_change_large"), bool):
        return bool(shape_report["shape_change_large"])
    if isinstance(slot_report.get("shape_change_large"), bool):
        return bool(slot_report["shape_change_large"])
    return None


def choose_placement_strategy(
    *,
    source_text: str,
    target_text: str,
    slots: tuple[TextRun, ...],
    slot_report: dict[str, Any],
    draw_mode: str,
) -> tuple[str, str]:
    source_count = len(_text_chars(source_text))
    target_count = len(_text_chars(target_text))
    if not source_count:
        return "manual_fallback", "source_text_missing"
    if draw_mode == "center" or target_count > source_count:
        return "left_anchor_span", "target_text_longer_than_source"
    if target_count < source_count:
        return "left_anchor_span", "target_text_shorter_than_source"
    if not is_mostly_cjk(source_text or target_text):
        return "baseline_numeric", "non_cjk_value_uses_baseline_priority"
    if not slot_report.get("pass"):
        return "top_left_anchor", "slot_quality_failed_keeps_original_anchor_for_rejection"
    if source_text != target_text:
        shape_change_large = _shape_change_large(slot_report)
        if shape_change_large is False:
            return "top_left_anchor", "same_length_cjk_small_shape_change_uses_top_left_anchor"
        if shape_change_large is True:
            return "center_primary", "same_length_cjk_large_shape_change_uses_slot_center"
        return "center_primary", "same_length_cjk_changed_chars_use_slot_center"
    heights = [max(1, slot.y2 - slot.y1) for slot in slots]
    widths = [max(1, slot.x2 - slot.x1) for slot in slots]
    if heights and max(heights) - min(heights) > max(2, float(_median(heights)) * 0.18):
        return "center_primary", "same_length_cjk_slot_height_variation"
    if widths and max(widths) - min(widths) > max(3, float(_median(widths)) * 0.22):
        return "center_primary", "same_length_cjk_slot_width_variation"
    return "top_left_anchor", "same_length_cjk_compact_slots"


def _median(values: list[int]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[midpoint])
    return float(ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _length_change(source_count: int, target_count: int) -> str:
    if source_count and target_count > source_count:
        return "longer"
    if source_count and target_count < source_count:
        return "shorter"
    if source_count and target_count:
        return "same"
    return "unknown"


def _number(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _max_abs(values: list[float]) -> float | None:
    if not values:
        return None
    return round(max(abs(item) for item in values), 3)


def _strategy_constraints(strategy: str, length_change: str) -> dict[str, Any]:
    common = {
        "max_baseline_dy": 2.5,
        "max_char_spacing_delta": 2.0,
    }
    if strategy == "top_left_anchor":
        return {
            **common,
            "anchor_priority": "top_left",
            "max_char_center_dx": 2.0,
            "max_char_center_dy": 2.5,
        }
    if strategy == "center_primary":
        return {
            **common,
            "anchor_priority": "slot_center",
            "max_char_center_dx": 2.0,
            "max_char_center_dy": 2.5,
            "max_left_boundary_dx": 2.0,
        }
    if strategy == "left_anchor_span":
        constraints = {
            **common,
            "anchor_priority": "left_boundary",
            "max_left_boundary_dx": 2.0,
            "max_span_width_delta": 3.0,
        }
        if length_change == "shorter":
            constraints["cleanup_extra_source_slots_required"] = True
        if length_change == "longer":
            constraints["protected_text_overlap_pixels"] = 0
            constraints["right_boundary_diagnostic_required"] = True
        return constraints
    if strategy == "baseline_numeric":
        return {
            "anchor_priority": "left_baseline",
            "max_left_boundary_dx": 1.5,
            "max_baseline_dy": 1.5,
            "max_rhythm_delta": 1.5,
            "max_field_width_delta": 3.0,
        }
    if strategy == "manual_fallback":
        return {
            "anchor_priority": "manual_roi_conservative",
            "allowed_alignment": ["center", "left"],
            "auto_acceptance_confidence": "reduced",
            "auto_acceptance_confidence_cap": 0.45,
        }
    return common


def _strategy_actual_errors(
    alignment_metrics: dict[str, Any],
    per_char: list[dict[str, Any]],
    slot_report: dict[str, Any],
) -> dict[str, Any]:
    center_dx_values = [
        abs(float(item.get("center_dx")))
        for item in per_char
        if isinstance(item, dict) and item.get("center_dx") is not None
    ]
    center_dy_values = [
        abs(float(item.get("center_dy")))
        for item in per_char
        if isinstance(item, dict) and item.get("center_dy") is not None
    ]
    length_report = slot_report.get("length_change_report") if isinstance(slot_report, dict) else {}
    cleanup_report = length_report.get("cleanup_mask_report") if isinstance(length_report, dict) else {}
    right_boundary = length_report.get("right_boundary") if isinstance(length_report, dict) else {}
    return {
        "alignment_metrics_enabled": bool(alignment_metrics.get("enabled")),
        "max_abs_center_dx": round(max(center_dx_values), 3) if center_dx_values else None,
        "max_abs_center_dy": round(max(center_dy_values), 3) if center_dy_values else None,
        "center_distance_delta": alignment_metrics.get("center_distance_delta"),
        "candidate_center_y_range": alignment_metrics.get("candidate_center_y_range"),
        "left_boundary_dx": alignment_metrics.get("left_boundary_dx"),
        "baseline_dy": alignment_metrics.get("baseline_dy"),
        "char_spacing_delta": alignment_metrics.get("char_spacing_delta"),
        "span_width_delta": alignment_metrics.get("span_width_delta"),
        "rhythm_delta": alignment_metrics.get("rhythm_delta"),
        "field_width_delta": alignment_metrics.get("field_width_delta"),
        "protected_text_overlap_pixels": alignment_metrics.get("protected_text_overlap_pixels"),
        "extra_source_cleanup_enabled": cleanup_report.get("enabled") if isinstance(cleanup_report, dict) else None,
        "right_boundary_pass": right_boundary.get("pass") if isinstance(right_boundary, dict) else None,
        "right_boundary_space_sufficient": right_boundary.get("space_sufficient") if isinstance(right_boundary, dict) else None,
    }


def _constraint_issues(constraints: dict[str, Any], actual_errors: dict[str, Any]) -> list[dict[str, Any]]:
    checks = (
        ("max_char_center_dx", "max_abs_center_dx"),
        ("max_char_center_dy", "max_abs_center_dy"),
        ("max_baseline_dy", "baseline_dy"),
        ("max_char_spacing_delta", "char_spacing_delta"),
        ("max_left_boundary_dx", "left_boundary_dx"),
        ("max_span_width_delta", "span_width_delta"),
        ("max_rhythm_delta", "rhythm_delta"),
        ("max_field_width_delta", "field_width_delta"),
    )
    issues: list[dict[str, Any]] = []
    for limit_key, actual_key in checks:
        limit = _number(constraints.get(limit_key))
        actual = _number(actual_errors.get(actual_key))
        if limit is not None and actual is not None and abs(actual) > limit:
            issues.append(
                {
                    "type": f"{actual_key}_exceeds_strategy_limit",
                    "actual": round(abs(actual), 3),
                    "limit": round(limit, 3),
                }
            )
    if constraints.get("protected_text_overlap_pixels") == 0:
        protected_overlap = _number(actual_errors.get("protected_text_overlap_pixels"))
        if protected_overlap is not None and protected_overlap > 0:
            issues.append(
                {
                    "type": "protected_text_overlap_for_longer_replacement",
                    "actual": round(protected_overlap, 3),
                    "limit": 0,
                }
            )
    if constraints.get("cleanup_extra_source_slots_required") and actual_errors.get("extra_source_cleanup_enabled") is False:
        issues.append({"type": "missing_extra_source_slot_cleanup"})
    return issues


def placement_strategy_report(
    plan: RenderPlan,
    alignment_metrics: dict[str, Any],
    alignment_issues: list[dict[str, Any]],
    *,
    max_char_center_dx: float,
    max_char_center_dy: float,
    max_char_center_distance_delta: float,
    max_replacement_center_y_range: float,
) -> dict[str, Any]:
    source_count = len(_text_chars(plan.source_text or ""))
    target_count = len(_text_chars(plan.target_text))
    length_change = _length_change(source_count, target_count)
    target_slot_count = len(plan.slot_boxes)
    if (
        length_change == "longer"
        and plan.draw_mode == "line_chars"
        and plan.placement_strategy == "left_anchor_span"
    ):
        target_slot_count = max(target_slot_count, target_count)
    slot_report = plan.slot_quality_report if isinstance(plan.slot_quality_report, dict) else {}
    per_char = alignment_metrics.get("per_char") if isinstance(alignment_metrics, dict) else []
    if not isinstance(per_char, list):
        per_char = []
    actual_errors = _strategy_actual_errors(alignment_metrics if isinstance(alignment_metrics, dict) else {}, per_char, slot_report)
    constraints = {
        "max_char_center_dx": max_char_center_dx,
        "max_char_center_dy": max_char_center_dy,
        "max_char_center_distance_delta": max_char_center_distance_delta,
        "max_replacement_center_y_range": max_replacement_center_y_range,
        **_strategy_constraints(plan.placement_strategy, length_change),
    }
    strategy_issues = _constraint_issues(constraints, actual_errors)
    slot_quality_pass = slot_report.get("pass")
    issues = [*alignment_issues, *strategy_issues]
    return {
        "strategy": plan.placement_strategy,
        "reason": plan.placement_strategy_reason,
        "pass": not issues and slot_quality_pass is not False,
        "conditions": {
            "source_count": source_count,
            "target_count": target_count,
            "length_change": length_change,
            "draw_mode": plan.draw_mode,
            "is_cjk": is_mostly_cjk((plan.source_text or "") + plan.target_text),
            "slot_count": len(plan.slot_boxes),
            "source_slot_count": len(plan.slot_boxes),
            "target_slot_count": target_slot_count,
            "slot_quality_pass": slot_quality_pass,
            "shape_change_large": _shape_change_large(slot_report),
            "manual_source_missing": not bool(source_count),
        },
        "constraints": constraints,
        "actual_errors": actual_errors,
        "strategy_contract": {
            "anchor_priority": constraints.get("anchor_priority"),
            "baseline_checked": "max_baseline_dy" in constraints,
            "char_spacing_checked": "max_char_spacing_delta" in constraints or "max_rhythm_delta" in constraints,
            "protected_text_guard_checked": constraints.get("protected_text_overlap_pixels") == 0,
            "longer_text_appends_slots": bool(
                length_change == "longer"
                and plan.draw_mode == "line_chars"
                and plan.placement_strategy == "left_anchor_span"
            ),
            "cleanup_required": bool(constraints.get("cleanup_extra_source_slots_required")),
            "auto_acceptance_confidence": constraints.get("auto_acceptance_confidence"),
            "auto_acceptance_confidence_cap": constraints.get("auto_acceptance_confidence_cap"),
        },
        "issues": issues,
    }
