from __future__ import annotations

from typing import Any


Box = tuple[int, int, int, int]


def box_size(box: Box) -> tuple[float, float]:
    return max(1.0, float(box[2] - box[0])), max(1.0, float(box[3] - box[1]))


def box_center(box: Box) -> tuple[float, float]:
    return (float(box[0] + box[2]) / 2.0, float(box[1] + box[3]) / 2.0)


def box_metrics(source_box: Box, target_box: Box) -> dict[str, Any]:
    source_w, source_h = box_size(source_box)
    target_w, target_h = box_size(target_box)
    source_area = source_w * source_h
    target_area = target_w * target_h
    source_cx, source_cy = box_center(source_box)
    target_cx, target_cy = box_center(target_box)
    return {
        "source": {
            "box": list(source_box),
            "width": round(source_w, 3),
            "height": round(source_h, 3),
            "area": round(source_area, 3),
            "center_x": round(source_cx, 3),
            "center_y": round(source_cy, 3),
            "margins": margin_distribution(source_box),
        },
        "target": {
            "box": list(target_box),
            "width": round(target_w, 3),
            "height": round(target_h, 3),
            "area": round(target_area, 3),
            "center_x": round(target_cx, 3),
            "center_y": round(target_cy, 3),
            "margins": margin_distribution(target_box),
        },
        "delta": {
            "bbox_width_delta_ratio": round((target_w - source_w) / source_w, 4),
            "bbox_height_delta_ratio": round((target_h - source_h) / source_h, 4),
            "centroid_dx": round(target_cx - source_cx, 3),
            "centroid_dy": round(target_cy - source_cy, 3),
            "ink_area_ratio": round(target_area / max(1.0, source_area), 4),
            "row_projection_distance": round(projection_distance(source_box, target_box, axis="row"), 4),
            "col_projection_distance": round(projection_distance(source_box, target_box, axis="col"), 4),
            "margin_distribution_delta": round(margin_distribution_delta(source_box, target_box), 4),
        },
    }


def projection_distance(source_box: Box, target_box: Box, *, axis: str) -> float:
    source_start, source_end = (source_box[1], source_box[3]) if axis == "row" else (source_box[0], source_box[2])
    target_start, target_end = (target_box[1], target_box[3]) if axis == "row" else (target_box[0], target_box[2])
    source_len = max(1.0, float(source_end - source_start))
    target_len = max(1.0, float(target_end - target_start))
    union_start = min(source_start, target_start)
    union_end = max(source_end, target_end)
    union_len = max(1.0, float(union_end - union_start))
    overlap = max(0.0, float(min(source_end, target_end) - max(source_start, target_start)))
    source_only = source_len - overlap
    target_only = target_len - overlap
    return (source_only + target_only) / union_len


def margin_distribution(box: Box) -> dict[str, float]:
    width, height = box_size(box)
    perimeter = max(1.0, 2.0 * (width + height))
    return {
        "left": round(width / perimeter, 4),
        "right": round(width / perimeter, 4),
        "top": round(height / perimeter, 4),
        "bottom": round(height / perimeter, 4),
    }


def margin_distribution_delta(source_box: Box, target_box: Box) -> float:
    source = margin_distribution(source_box)
    target = margin_distribution(target_box)
    return sum(abs(float(target[key]) - float(source[key])) for key in ("left", "right", "top", "bottom"))


def default_shape_thresholds(slot_height: float) -> dict[str, Any]:
    center_limit = max(2.0, float(slot_height) * 0.16)
    return {
        "bbox_width_delta_ratio": {"limit": 0.30, "threshold_source": "default"},
        "bbox_height_delta_ratio": {"limit": 0.24, "threshold_source": "default"},
        "centroid_dx": {"limit": round(center_limit, 3), "threshold_source": "default"},
        "centroid_dy": {"limit": round(center_limit, 3), "threshold_source": "default"},
        "ink_area_ratio": {"range": [0.42, 1.95], "threshold_source": "default"},
        "row_projection_distance": {"limit": 0.28, "threshold_source": "default"},
        "col_projection_distance": {"limit": 0.34, "threshold_source": "default"},
        "margin_distribution_delta": {"limit": 0.22, "threshold_source": "default"},
    }


def neighbor_stability(widths: list[float], heights: list[float]) -> dict[str, Any]:
    def ratio(values: list[float]) -> float:
        if not values:
            return 0.0
        median_value = median(values)
        return (max(values) - min(values)) / max(1.0, median_value)

    width_variation = ratio(widths)
    height_variation = ratio(heights)
    variation = max(width_variation, height_variation)
    return {
        "source": "neighbor_slot_geometry",
        "slot_count": len(widths),
        "width_variation_ratio": round(float(width_variation), 4),
        "height_variation_ratio": round(float(height_variation), 4),
        "stable": bool(variation <= 0.18),
        "stability_score": round(max(0.0, 1.0 - variation), 4),
    }


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[midpoint])
    return float(ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def font_candidate_distribution(
    *,
    selected_font_size: float,
    slot_heights: list[float],
    reported_distribution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(reported_distribution, dict) and reported_distribution:
        return {
            "source": "slot_quality_report_font_candidate_distribution",
            **reported_distribution,
        }
    median_slot_height = median(slot_heights)
    return {
        "source": "selected_candidate_font_size_distribution",
        "sample_count": 1,
        "median_font_size": round(float(selected_font_size), 3),
        "median_slot_height": round(float(median_slot_height), 3),
        "font_size_to_slot_height_ratio": round(float(selected_font_size) / max(1.0, median_slot_height), 4),
    }


def dynamic_shape_thresholds(
    *,
    slot_height: float,
    neighbor_stability_report: dict[str, Any],
    font_distribution_report: dict[str, Any],
) -> dict[str, Any]:
    base = default_shape_thresholds(slot_height)
    stable_neighbors = bool(neighbor_stability_report.get("stable"))
    try:
        font_ratio = float(font_distribution_report.get("font_size_to_slot_height_ratio") or 1.0)
    except (TypeError, ValueError):
        font_ratio = 1.0
    font_adjustment = max(0.92, min(1.12, font_ratio))
    neighbor_adjustment = 0.92 if stable_neighbors else 1.08
    center_adjustment = neighbor_adjustment * font_adjustment
    threshold_context = {
        "old_slot_height": round(float(slot_height), 3),
        "neighbor_stability": neighbor_stability_report,
        "font_candidate_distribution": font_distribution_report,
    }

    def limit(name: str, value: float) -> dict[str, Any]:
        return {
            "limit": round(float(value), 4),
            "threshold_source": "dynamic",
            "threshold_context": threshold_context,
            "source_components": ["old_slot_height", "neighbor_stability", "font_candidate_distribution"],
        }

    def range_limit(name: str, lower: float, upper: float) -> dict[str, Any]:
        return {
            "range": [round(float(lower), 4), round(float(upper), 4)],
            "threshold_source": "dynamic",
            "threshold_context": threshold_context,
            "source_components": ["old_slot_height", "neighbor_stability", "font_candidate_distribution"],
        }

    return {
        "bbox_width_delta_ratio": limit(
            "bbox_width_delta_ratio",
            float(base["bbox_width_delta_ratio"]["limit"]) * (1.04 if not stable_neighbors else 0.96),
        ),
        "bbox_height_delta_ratio": limit(
            "bbox_height_delta_ratio",
            float(base["bbox_height_delta_ratio"]["limit"]) * (1.04 if not stable_neighbors else 0.96),
        ),
        "centroid_dx": limit("centroid_dx", float(base["centroid_dx"]["limit"]) * center_adjustment),
        "centroid_dy": limit("centroid_dy", float(base["centroid_dy"]["limit"]) * center_adjustment),
        "ink_area_ratio": range_limit(
            "ink_area_ratio",
            float(base["ink_area_ratio"]["range"][0]) / font_adjustment,
            float(base["ink_area_ratio"]["range"][1]) * font_adjustment,
        ),
        "row_projection_distance": limit(
            "row_projection_distance",
            float(base["row_projection_distance"]["limit"]) * (1.05 if not stable_neighbors else 0.95),
        ),
        "col_projection_distance": limit(
            "col_projection_distance",
            float(base["col_projection_distance"]["limit"]) * (1.05 if not stable_neighbors else 0.95),
        ),
        "margin_distribution_delta": limit(
            "margin_distribution_delta",
            float(base["margin_distribution_delta"]["limit"]) * (1.05 if not stable_neighbors else 0.95),
        ),
    }
