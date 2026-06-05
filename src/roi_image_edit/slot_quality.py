from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image

from roi_image_edit.iterative_pipeline import TextRun, clamp_box, text_chars


def text_run_box(run: TextRun) -> tuple[int, int, int, int]:
    return run.x1, run.y1, run.x2, run.y2


def box_area(box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def box_overlap_area(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> int:
    return box_area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def protected_box_overlaps_row(
    box: tuple[int, int, int, int],
    row: tuple[int, int, int, int],
) -> bool:
    overlap = max(0, min(box[3], row[3]) - max(box[1], row[1]))
    return overlap / max(1, min(box[3] - box[1], row[3] - row[1])) >= 0.30


def union_boxes(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _coverage(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 1.0
    return round(float(numerator / denominator), 4)


def _mask_count(gray: np.ndarray, low: int, high: int | None = None) -> int:
    if gray.size == 0:
        return 0
    if high is None:
        return int(np.count_nonzero(gray < low))
    return int(np.count_nonzero((gray >= low) & (gray < high)))


def _cleanup_mask_report(
    boxes: list[tuple[int, int, int, int]],
) -> dict[str, Any]:
    pixel_count = sum(box_area(box) for box in boxes)
    span = union_boxes(boxes)
    return {
        "enabled": bool(boxes),
        "boxes": [list(box) for box in boxes],
        "span": list(span) if span else None,
        "pixel_count": int(pixel_count),
    }


def slot_quality_report(
    img: Image.Image,
    roi: tuple[int, int, int, int],
    slots: tuple[TextRun, ...],
    *,
    source_text: str,
    target_text: str,
    protected_boxes: tuple[tuple[int, int, int, int], ...],
    threshold: int = 165,
) -> dict[str, Any]:
    source_chars = text_chars(source_text)
    target_chars = text_chars(target_text)
    source_count = len(source_chars)
    target_count = len(target_chars)
    expected_count = len(source_chars) or len(target_chars)
    issues: list[dict[str, Any]] = []
    if expected_count and len(slots) < expected_count:
        issues.append(
            {
                "type": "slot_count_too_low",
                "expected": expected_count,
                "actual": len(slots),
            }
        )
    if source_chars and len(slots) > len(source_chars) + 1:
        issues.append(
            {
                "type": "slot_count_too_high",
                "expected": len(source_chars),
                "actual": len(slots),
            }
        )

    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]
    ordered_slots = tuple(sorted(slots, key=lambda item: item.x1))
    source_slot_limit = source_count if source_count else len(ordered_slots)
    source_slot_boxes = [
        text_run_box(slot)
        for slot in ordered_slots[: min(source_slot_limit, len(ordered_slots))]
    ]
    source_span_box = union_boxes(source_slot_boxes)

    core_threshold = min(95, max(55, threshold - 70))
    per_slot: list[dict[str, Any]] = []
    total_protected_overlap = 0
    total_label_overlap = 0
    total_right_protected_overlap = 0
    for idx, slot in enumerate(ordered_slots):
        x1, y1, x2, y2 = clamp_box((slot.x1, slot.y1, slot.x2, slot.y2), (w, h))
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        crop = gray[y1:y2, x1:x2] if width and height else np.zeros((0, 0), dtype=np.uint8)
        core_pixels = _mask_count(crop, core_threshold)
        gray_edge_pixels = _mask_count(crop, core_threshold, threshold)
        dark_pixels = core_pixels + gray_edge_pixels

        pad_y = max(2, int(round(height * 0.20)))
        check_box = clamp_box((x1, y1 - pad_y, x2, y2 + pad_y), (w, h))
        cx1, cy1, cx2, cy2 = check_box
        check_crop = gray[cy1:cy2, cx1:cx2] if cx2 > cx1 and cy2 > cy1 else np.zeros((0, 0), dtype=np.uint8)
        check_core_pixels = _mask_count(check_crop, core_threshold)
        check_gray_edge_pixels = _mask_count(check_crop, core_threshold, threshold)
        below_y2 = min(roi[3], y2 + pad_y)
        below_crop = gray[y2:below_y2, x1:x2] if below_y2 > y2 and width else np.zeros((0, 0), dtype=np.uint8)
        below_core_pixels = _mask_count(below_crop, core_threshold)
        below_gray_edge_pixels = _mask_count(below_crop, core_threshold, threshold)
        below_dark_pixels = below_core_pixels + below_gray_edge_pixels

        slot_issues: list[dict[str, Any]] = []
        if width < 3 or height < 3:
            slot_issues.append({"type": "slot_too_small", "width": width, "height": height})
        min_dark_pixels = max(4, int(round(width * height * 0.025)))
        if dark_pixels < min_dark_pixels:
            slot_issues.append(
                {
                    "type": "slot_has_too_few_dark_pixels",
                    "dark_pixels": dark_pixels,
                    "minimum": min_dark_pixels,
                }
            )
        if below_dark_pixels >= max(3, int(round(width * max(1, below_y2 - y2) * 0.025))):
            slot_issues.append(
                {
                    "type": "slot_bottom_overflow",
                    "dark_pixels_below": below_dark_pixels,
                    "core_pixels_below": below_core_pixels,
                    "gray_edge_pixels_below": below_gray_edge_pixels,
                    "checked_y": [y2, below_y2],
                }
            )

        slot_box = (x1, y1, x2, y2)
        protected_conflicts: list[list[int]] = []
        protected_overlap_pixels = 0
        label_overlap_pixels = 0
        right_protected_overlap_pixels = 0
        for box in protected_boxes:
            overlap = box_overlap_area(slot_box, box)
            if overlap <= 0:
                continue
            protected_overlap_pixels += overlap
            protected_conflicts.append(list(box))
            if source_span_box and box[2] <= source_span_box[0]:
                label_overlap_pixels += overlap
            elif source_span_box and box[0] >= source_span_box[2]:
                right_protected_overlap_pixels += overlap
        total_protected_overlap += protected_overlap_pixels
        total_label_overlap += label_overlap_pixels
        total_right_protected_overlap += right_protected_overlap_pixels
        if protected_conflicts:
            slot_issues.append(
                {
                    "type": "slot_overlaps_protected_text",
                    "protected_boxes": protected_conflicts,
                    "overlap_pixels": protected_overlap_pixels,
                }
            )

        issues.extend({"index": idx, **issue} for issue in slot_issues)
        per_slot.append(
            {
                "index": idx,
                "box": [x1, y1, x2, y2],
                "width": width,
                "height": height,
                "core_pixels": core_pixels,
                "gray_edge_pixels": gray_edge_pixels,
                "dark_pixels": dark_pixels,
                "coverage": {
                    "core_coverage": _coverage(core_pixels, check_core_pixels),
                    "gray_edge_coverage": _coverage(gray_edge_pixels, check_gray_edge_pixels),
                    "bottom_coverage": _coverage(dark_pixels, dark_pixels + below_dark_pixels),
                    "checked_box": list(check_box),
                    "core_pixels_outside_slot": max(0, check_core_pixels - core_pixels),
                    "gray_edge_pixels_outside_slot": max(0, check_gray_edge_pixels - gray_edge_pixels),
                    "bottom_dark_pixels": below_dark_pixels,
                },
                "overlap": {
                    "protected_overlap_pixels": protected_overlap_pixels,
                    "label_overlap_pixels": label_overlap_pixels,
                    "right_protected_overlap_pixels": right_protected_overlap_pixels,
                },
                "issues": slot_issues,
            }
        )

    if source_count and target_count > source_count:
        length_change = "longer"
    elif source_count and target_count < source_count:
        length_change = "shorter"
    elif source_count and target_count:
        length_change = "same"
    else:
        length_change = "unknown"

    extra_source_slots = (
        ordered_slots[target_count: min(source_count, len(ordered_slots))]
        if length_change == "shorter"
        else ()
    )
    extra_source_boxes = [text_run_box(slot) for slot in extra_source_slots]
    cleanup_report = _cleanup_mask_report(extra_source_boxes)
    if length_change == "shorter" and source_count and target_count and len(extra_source_slots) < source_count - target_count:
        issues.append(
            {
                "type": "missing_extra_source_slots_for_cleanup",
                "expected_extra_slots": source_count - target_count,
                "actual_extra_slots": len(extra_source_slots),
            }
        )

    right_boundary: dict[str, Any] = {
        "enabled": bool(length_change == "longer" and source_span_box),
    }
    if length_change == "longer" and source_span_box is not None:
        row_box = source_span_box
        margin = 3
        right_protected_boxes = [
            box
            for box in protected_boxes
            if protected_box_overlaps_row(box, row_box) and box[0] >= row_box[2]
        ]
        right_limit = min([roi[2], *(box[0] - margin for box in right_protected_boxes)])
        slot_widths = [max(1, slot.x2 - slot.x1) for slot in ordered_slots]
        slot_gaps = [
            max(0, ordered_slots[idx + 1].x1 - ordered_slots[idx].x2)
            for idx in range(len(ordered_slots) - 1)
        ]
        median_width = float(np.median(slot_widths)) if slot_widths else max(1.0, (row_box[2] - row_box[0]) / source_count)
        median_gap = float(np.median(slot_gaps)) if slot_gaps else max(2.0, median_width * 0.14)
        estimated_extra_width = max(0.0, (target_count - source_count) * (median_width + median_gap))
        available_right_px = max(0, right_limit - row_box[2])
        protected_gap_px = (
            min((box[0] - row_box[2]) for box in right_protected_boxes)
            if right_protected_boxes
            else None
        )
        right_boundary = {
            "enabled": True,
            "source_span_box": list(row_box),
            "right_limit": int(right_limit),
            "available_right_px": int(available_right_px),
            "estimated_extra_width": round(float(estimated_extra_width), 3),
            "protected_gap_px": None if protected_gap_px is None else int(protected_gap_px),
            "limited_by_protected_text": bool(right_limit < roi[2]),
            "protected_right_boxes": [list(box) for box in right_protected_boxes],
            "pass": bool(available_right_px >= estimated_extra_width),
        }
        if available_right_px < estimated_extra_width:
            issues.append(
                {
                    "type": "right_boundary_too_close_to_protected_text",
                    "available_right_px": int(available_right_px),
                    "estimated_extra_width": round(float(estimated_extra_width), 3),
                    "protected_gap_px": None if protected_gap_px is None else int(protected_gap_px),
                }
            )

    return {
        "pass": not issues,
        "expected_count": expected_count,
        "actual_count": len(slots),
        "source_count": source_count,
        "target_count": target_count,
        "length_change": length_change,
        "source_text": source_text,
        "target_text": target_text,
        "roi": list(roi),
        "source_span_box": list(source_span_box) if source_span_box else None,
        "slot_coverage_schema": {
            "core_threshold": core_threshold,
            "gray_edge_threshold": threshold,
            "coverage_fields": ["core_coverage", "gray_edge_coverage", "bottom_coverage"],
        },
        "overlap_report": {
            "protected_overlap_pixels": int(total_protected_overlap),
            "label_overlap_pixels": int(total_label_overlap),
            "right_protected_overlap_pixels": int(total_right_protected_overlap),
        },
        "length_change_report": {
            "length_change": length_change,
            "source_count": source_count,
            "target_count": target_count,
            "extra_source_slots_for_cleanup": [list(box) for box in extra_source_boxes],
            "extra_source_cleanup_span": cleanup_report["span"],
            "cleanup_mask_report": cleanup_report,
            "right_boundary": right_boundary,
        },
        "per_slot": per_slot,
        "issues": issues,
    }
