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
