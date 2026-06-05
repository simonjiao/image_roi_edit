from __future__ import annotations

from pathlib import Path
from typing import Any

from roi_image_edit.iterative_pipeline import RenderPlan
from roi_image_edit.environment import font_missing_text_chars


def shape_score_breakdown(report: dict[str, Any], plan: RenderPlan) -> dict[str, Any]:
    shape = report.get("shape_change_report") if isinstance(report, dict) else None
    if not isinstance(shape, dict) or not shape.get("enabled"):
        return {"enabled": False, "reason": "missing_shape_change_report"}
    per_char = [item for item in shape.get("per_char", []) if isinstance(item, dict)]
    if not per_char:
        return {"enabled": False, "reason": "missing_shape_char_metrics"}

    components = {
        "height_width_spacing_baseline": height_width_spacing_baseline_component(per_char),
        "center_error": center_error_component(per_char),
        "boundary_error": boundary_error_component(per_char),
        "ink_area_complexity": ink_area_complexity_component(per_char, report),
        "pose_inheritance": pose_inheritance_component(report),
        "protected_distance": protected_distance_component(per_char, plan),
        "font_style": font_style_component(report, plan),
    }
    score = sum(float(component.get("score") or 0.0) for component in components.values())
    return {
        "enabled": True,
        "score": round(score, 4),
        "components": components,
        "basis": "shape candidate ranking uses geometry, ink area, pose, protected distance, and font report components",
    }


def height_width_spacing_baseline_component(per_char: list[dict[str, Any]]) -> dict[str, Any]:
    width_errors = [abs_float(item.get("bbox_width_delta_ratio")) for item in per_char]
    height_errors = [abs_float(item.get("bbox_height_delta_ratio")) for item in per_char]
    centers_y = [float(item.get("target_image_metrics", {}).get("center_y")) for item in per_char if isinstance(item.get("target_image_metrics"), dict) and item.get("target_image_metrics", {}).get("center_y") is not None]
    candidate_boxes = [item.get("candidate_box") for item in per_char if isinstance(item.get("candidate_box"), list)]
    slot_boxes = [item.get("slot_box") for item in per_char if isinstance(item.get("slot_box"), list)]
    candidate_gaps = gaps(candidate_boxes)
    slot_gaps = gaps(slot_boxes)
    gap_errors = [abs(candidate - slot) for candidate, slot in zip(candidate_gaps, slot_gaps)]
    baseline_range = max(centers_y) - min(centers_y) if centers_y else 0.0
    score = mean(width_errors) * 120.0 + mean(height_errors) * 120.0 + mean(gap_errors) * 2.0 + baseline_range * 3.0
    return {
        "score": round(score, 4),
        "avg_abs_width_delta_ratio": round(mean(width_errors), 4),
        "avg_abs_height_delta_ratio": round(mean(height_errors), 4),
        "avg_abs_gap_delta_px": round(mean(gap_errors), 3),
        "candidate_baseline_center_y_range": round(baseline_range, 3),
    }


def center_error_component(per_char: list[dict[str, Any]]) -> dict[str, Any]:
    dx = [abs_float(item.get("centroid_dx")) for item in per_char]
    dy = [abs_float(item.get("centroid_dy")) for item in per_char]
    score = max(dx or [0.0]) * 14.0 + max(dy or [0.0]) * 14.0
    return {
        "score": round(score, 4),
        "max_abs_centroid_dx": round(max(dx or [0.0]), 3),
        "max_abs_centroid_dy": round(max(dy or [0.0]), 3),
        "avg_abs_centroid_dx": round(mean(dx), 3),
        "avg_abs_centroid_dy": round(mean(dy), 3),
    }


def boundary_error_component(per_char: list[dict[str, Any]]) -> dict[str, Any]:
    left_errors: list[float] = []
    right_errors: list[float] = []
    for item in per_char:
        slot = item.get("slot_box")
        candidate = item.get("candidate_box")
        if not isinstance(slot, list) or not isinstance(candidate, list) or len(slot) != 4 or len(candidate) != 4:
            continue
        left_errors.append(abs(float(candidate[0]) - float(slot[0])))
        right_errors.append(abs(float(candidate[2]) - float(slot[2])))
    score = max(left_errors or [0.0]) * 7.0 + max(right_errors or [0.0]) * 7.0
    return {
        "score": round(score, 4),
        "max_abs_left_delta_px": round(max(left_errors or [0.0]), 3),
        "max_abs_right_delta_px": round(max(right_errors or [0.0]), 3),
        "avg_abs_left_delta_px": round(mean(left_errors), 3),
        "avg_abs_right_delta_px": round(mean(right_errors), 3),
    }


def ink_area_complexity_component(per_char: list[dict[str, Any]], report: dict[str, Any]) -> dict[str, Any]:
    ink_errors = [abs(float(item.get("ink_area_ratio") or 1.0) - 1.0) for item in per_char]
    strict = report.get("strict_gate") if isinstance(report.get("strict_gate"), dict) else {}
    complexity_ratio = float(strict.get("text_complexity_ratio") or 1.0)
    adjusted = [value / max(0.5, complexity_ratio) for value in ink_errors]
    score = mean(adjusted) * 180.0
    return {
        "score": round(score, 4),
        "avg_abs_ink_area_ratio_delta": round(mean(ink_errors), 4),
        "text_complexity_ratio": round(complexity_ratio, 4),
        "avg_complexity_adjusted_ink_area_delta": round(mean(adjusted), 4),
    }


def pose_inheritance_component(report: dict[str, Any]) -> dict[str, Any]:
    pose = report.get("char_pose_metrics") if isinstance(report.get("char_pose_metrics"), dict) else {}
    gaps_values: list[float] = []
    for item in pose.get("per_char", []) if isinstance(pose, dict) else []:
        if not isinstance(item, dict):
            continue
        reference = item.get("reference_shear")
        applied = item.get("applied_shear")
        if reference is None or applied is None:
            continue
        gaps_values.append(abs(float(applied) - float(reference)))
    score = max(gaps_values or [0.0]) * 1200.0
    return {
        "score": round(score, 4),
        "max_abs_shear_error": round(max(gaps_values or [0.0]), 4),
        "avg_abs_shear_error": round(mean(gaps_values), 4),
    }


def protected_distance_component(per_char: list[dict[str, Any]], plan: RenderPlan) -> dict[str, Any]:
    distances: list[float] = []
    for item in per_char:
        candidate = item.get("candidate_box")
        if not isinstance(candidate, list) or len(candidate) != 4:
            continue
        candidate_box = tuple(int(value) for value in candidate)
        for protected in plan.protected_boxes:
            if vertical_overlap(candidate_box, protected) <= 0:
                continue
            distances.append(horizontal_distance(candidate_box, protected))
    min_distance = min(distances) if distances else None
    score = 0.0 if min_distance is None else max(0.0, 4.0 - min_distance) * 60.0
    return {
        "score": round(score, 4),
        "min_horizontal_gap_px": None if min_distance is None else round(min_distance, 3),
        "protected_box_count": len(plan.protected_boxes),
    }


def font_style_component(report: dict[str, Any], plan: RenderPlan) -> dict[str, Any]:
    font = report.get("font_style_gate") if isinstance(report.get("font_style_gate"), dict) else {}
    candidate = font.get("candidate") if isinstance(font.get("candidate"), dict) else {}
    font_best = font.get("font_best") if isinstance(font.get("font_best"), dict) else {}
    score_ratio = candidate.get("score_ratio_to_best")
    font_family_ratio = font_best.get("score_ratio_to_best")
    params = report.get("params") if isinstance(report.get("params"), dict) else {}
    missing_chars = render_missing_chars(params.get("font_path"), plan.target_text)
    issue_count = len(font.get("issues") or []) if isinstance(font, dict) else 0
    ratio_value = float(score_ratio) if score_ratio is not None else 1.0
    score = max(0.0, ratio_value - 1.0) * 250.0 + issue_count * 80.0 + len(missing_chars) * 500.0
    return {
        "score": round(score, 4),
        "font_style_score_ratio": None if score_ratio is None else round(ratio_value, 4),
        "font_family_score_ratio": None if font_family_ratio is None else round(float(font_family_ratio), 4),
        "font_style_pass": font.get("pass") if isinstance(font, dict) else None,
        "renderable_text_check": {
            "checked": bool(params.get("font_path")),
            "target_text": plan.target_text,
            "missing_chars": missing_chars,
            "pass": not missing_chars,
        },
        "issue_count": issue_count,
    }


def render_missing_chars(font_path: Any, text: str) -> list[str]:
    if not font_path:
        return []
    try:
        return font_missing_text_chars(Path(str(font_path)), text)
    except OSError:
        return list(text)


def gaps(boxes: list[Any]) -> list[float]:
    valid = [box for box in boxes if isinstance(box, list) and len(box) == 4]
    return [float(valid[idx + 1][0]) - float(valid[idx][2]) for idx in range(len(valid) - 1)]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def abs_float(value: Any) -> float:
    try:
        return abs(float(value))
    except (TypeError, ValueError):
        return 0.0


def vertical_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    return max(0, min(a[3], b[3]) - max(a[1], b[1]))


def horizontal_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    if a[2] < b[0]:
        return float(b[0] - a[2])
    if b[2] < a[0]:
        return float(a[0] - b[2])
    return 0.0
