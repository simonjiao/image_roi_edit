from __future__ import annotations

from dataclasses import asdict
from typing import Any

import cv2
import numpy as np
from PIL import Image

from roi_image_edit.iterative_pipeline import RenderPlan, TextRun, clamp_box, hard_check
from roi_image_edit.roi_locator import (
    component_text_runs,
    dominant_text_line,
    merge_overlapping_text_components,
    text_chars,
    text_run_box,
)


def _changed_indices(source_text: str, target_text: str) -> list[int]:
    source_chars = text_chars(source_text)
    target_chars = text_chars(target_text)
    if len(source_chars) != len(target_chars):
        return []
    return [
        idx
        for idx, (source_char, target_char) in enumerate(zip(source_chars, target_chars))
        if source_char != target_char
    ]


def _box_area(box: tuple[int, int, int, int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _union_box(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    valid = [box for box in boxes if _box_area(box) > 0]
    if not valid:
        return None
    return (
        min(box[0] for box in valid),
        min(box[1] for box in valid),
        max(box[2] for box in valid),
        max(box[3] for box in valid),
    )


def _slot_gaps(slots: tuple[TextRun, ...]) -> list[int]:
    ordered = tuple(sorted(slots, key=lambda item: item.x1))
    return [
        max(0, ordered[idx + 1].x1 - ordered[idx].x2)
        for idx in range(len(ordered) - 1)
    ]


def _median_slot_padding(slots: tuple[TextRun, ...]) -> tuple[int, int]:
    widths = [max(1, slot.x2 - slot.x1) for slot in slots]
    heights = [max(1, slot.y2 - slot.y1) for slot in slots]
    gaps = _slot_gaps(slots)
    median_width = float(np.median(widths)) if widths else 8.0
    median_height = float(np.median(heights)) if heights else 12.0
    median_gap = float(np.median(gaps)) if gaps else max(2.0, median_width * 0.20)
    return (
        max(2, int(round(min(median_width * 0.22, median_gap * 0.75 + 1.0)))),
        max(3, int(round(median_height * 0.16))),
    )


def _candidate_slot_windows(
    components: tuple[TextRun, ...],
    *,
    count: int,
    roi: tuple[int, int, int, int],
) -> list[tuple[float, tuple[TextRun, ...]]]:
    if len(components) < count or count <= 0:
        return []
    roi_center_x = (roi[0] + roi[2]) / 2.0
    scored: list[tuple[float, tuple[TextRun, ...]]] = []
    for start in range(0, len(components) - count + 1):
        window = tuple(sorted(components[start : start + count], key=lambda item: item.x1))
        span = max(1, window[-1].x2 - window[0].x1)
        gaps = _slot_gaps(window)
        max_gap = max(gaps) if gaps else 0
        center_x = (window[0].x1 + window[-1].x2) / 2.0
        score = span + max_gap * 2.8 + abs(center_x - roi_center_x) * 0.08
        scored.append((float(score), window))
    return sorted(scored, key=lambda item: item[0])


def detect_source_glyph_slots(
    image: Image.Image,
    roi: tuple[int, int, int, int],
    source_text: str,
) -> tuple[tuple[TextRun, ...], dict[str, Any]]:
    source_chars = text_chars(source_text)
    count = len(source_chars)
    if count <= 0:
        return (), {"pass": False, "reason": "missing_source_text"}

    attempts: list[dict[str, Any]] = []
    for threshold in (180, 165, 150, 135, 120):
        components = merge_overlapping_text_components(
            tuple(
                sorted(
                    dominant_text_line(component_text_runs(image, roi, threshold=threshold), roi),
                    key=lambda item: item.x1,
                )
            )
        )
        attempts.append(
            {
                "threshold": threshold,
                "component_count": len(components),
                "components": [text_run_box(item) for item in components],
            }
        )
        windows = _candidate_slot_windows(components, count=count, roi=roi)
        if not windows:
            continue
        _score, selected = windows[0]
        return selected, {
            "pass": True,
            "threshold": threshold,
            "source_text_count": count,
            "component_count": len(components),
            "selected_slots": [text_run_box(item) for item in selected],
            "attempts": attempts,
        }
    return (), {
        "pass": False,
        "reason": "no_component_window_matches_source_text_count",
        "source_text_count": count,
        "attempts": attempts,
    }


def _background_color(arr: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = roi
    crop = arr[y1:y2, x1:x2]
    if crop.size == 0:
        return np.array([255, 255, 255], dtype=np.uint8)
    gray = np.mean(crop, axis=2)
    pixels = crop[gray > 245]
    if len(pixels) == 0:
        pixels = crop[gray > 232]
    if len(pixels) == 0:
        return np.median(crop.reshape(-1, 3), axis=0).astype(np.uint8)
    return np.median(pixels, axis=0).astype(np.uint8)


def _bounded_box(
    box: tuple[int, int, int, int],
    *,
    image_size: tuple[int, int],
    container: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    return (
        max(container[0], min(image_size[0], box[0])),
        max(container[1], min(image_size[1], box[1])),
        max(container[0], min(image_size[0], box[2])),
        max(container[1], min(image_size[1], box[3])),
    )


def _erase_box_for_slot(
    slot: TextRun,
    *,
    slots: tuple[TextRun, ...],
    index: int,
    pad_x: int,
    pad_y: int,
    image_size: tuple[int, int],
    container: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    left = slot.x1 - pad_x
    right = slot.x2 + pad_x
    if index > 0:
        left = max(left, slots[index - 1].x2 + 1)
    if index + 1 < len(slots):
        right = min(right, slots[index + 1].x1 - 1)
    return _bounded_box(
        (left, slot.y1 - pad_y, right, slot.y2 + pad_y),
        image_size=image_size,
        container=container,
    )


def _crop_box_for_reference(
    slot: TextRun,
    *,
    slots: tuple[TextRun, ...],
    index: int,
    pad_x: int,
    pad_y: int,
    image_size: tuple[int, int],
    container: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    left = slot.x1 - pad_x
    right = slot.x2 + pad_x
    if index > 0:
        left = max(left, slots[index - 1].x2 + 1)
    if index + 1 < len(slots):
        right = min(right, slots[index + 1].x1 - 1)
    return _bounded_box(
        (left, slot.y1 - pad_y, right, slot.y2 + pad_y),
        image_size=image_size,
        container=container,
    )


def _expand_mask(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0:
        return mask
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def _paste_masked(
    target: np.ndarray,
    patch: np.ndarray,
    mask: np.ndarray,
    *,
    x: int,
    y: int,
) -> tuple[int, int, int, int] | None:
    height, width = target.shape[:2]
    patch_h, patch_w = patch.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(width, x + patch_w)
    y2 = min(height, y + patch_h)
    if x2 <= x1 or y2 <= y1:
        return None
    patch_x1 = x1 - x
    patch_y1 = y1 - y
    patch_x2 = patch_x1 + (x2 - x1)
    patch_y2 = patch_y1 + (y2 - y1)
    local_mask = mask[patch_y1:patch_y2, patch_x1:patch_x2]
    if not np.any(local_mask):
        return None
    target_region = target[y1:y2, x1:x2]
    target_region[local_mask] = patch[patch_y1:patch_y2, patch_x1:patch_x2][local_mask]
    target[y1:y2, x1:x2] = target_region
    return x1, y1, x2, y2


def _reference_index_for_target(
    source_chars: list[str],
    target_chars: list[str],
    changed_index: int,
) -> int | None:
    target_char = target_chars[changed_index]
    candidates = [
        idx
        for idx, source_char in enumerate(source_chars)
        if source_char == target_char and idx != changed_index
    ]
    if not candidates:
        return None
    unchanged = [idx for idx in candidates if source_chars[idx] == target_chars[idx]]
    pool = unchanged or candidates
    return min(pool, key=lambda idx: abs(idx - changed_index))


def source_glyph_reuse_candidate(
    original: Image.Image,
    plan: RenderPlan,
    *,
    max_changed_chars: int = 2,
    ink_threshold: int = 242,
) -> tuple[Image.Image | None, dict[str, Any]]:
    source_text = plan.source_text or ""
    target_text = plan.target_text
    source_chars = text_chars(source_text)
    target_chars = text_chars(target_text)
    if not source_chars or len(source_chars) != len(target_chars):
        return None, {
            "enabled": False,
            "strategy": "source_glyph_reuse",
            "reason": "requires_equal_length_source_and_target",
            "source_text": source_text,
            "target_text": target_text,
        }

    changed = _changed_indices(source_text, target_text)
    if not changed:
        return None, {
            "enabled": False,
            "strategy": "source_glyph_reuse",
            "reason": "source_and_target_are_identical",
            "source_text": source_text,
            "target_text": target_text,
        }
    if len(changed) > max_changed_chars:
        return None, {
            "enabled": False,
            "strategy": "source_glyph_reuse",
            "reason": "too_many_changed_chars",
            "changed_indices": changed,
            "max_changed_chars": max_changed_chars,
        }

    roi = tuple(int(value) for value in plan.search_roi)
    slots, slot_report = detect_source_glyph_slots(original, roi, source_text)
    if len(slots) < len(source_chars):
        return None, {
            "enabled": True,
            "strategy": "source_glyph_reuse",
            "pass": False,
            "reason": "source_glyph_slots_not_detected",
            "slot_detection": slot_report,
        }

    reference_indexes: dict[int, int] = {}
    missing_reference: list[dict[str, Any]] = []
    for idx in changed:
        reference_idx = _reference_index_for_target(source_chars, target_chars, idx)
        if reference_idx is None:
            missing_reference.append(
                {
                    "index": idx,
                    "source_char": source_chars[idx],
                    "target_char": target_chars[idx],
                    "reason": "target_char_not_present_in_source_text",
                }
            )
            continue
        reference_indexes[idx] = reference_idx
    if missing_reference:
        return None, {
            "enabled": True,
            "strategy": "source_glyph_reuse",
            "pass": False,
            "reason": "missing_reusable_target_glyph",
            "missing_reference": missing_reference,
            "slot_detection": slot_report,
        }

    original_rgb = original.convert("RGB")
    arr = np.array(original_rgb)
    edited = arr.copy()
    pad_x, pad_y = _median_slot_padding(slots)
    bg_color = _background_color(arr, roi)
    edit_boxes: list[tuple[int, int, int, int]] = []
    replacements: list[dict[str, Any]] = []

    for idx in changed:
        target_slot = slots[idx]
        ref_idx = reference_indexes[idx]
        ref_slot = slots[ref_idx]
        erase_box = _erase_box_for_slot(
            target_slot,
            slots=slots,
            index=idx,
            pad_x=pad_x,
            pad_y=pad_y,
            image_size=original_rgb.size,
            container=roi,
        )
        ex1, ey1, ex2, ey2 = erase_box
        if ex2 <= ex1 or ey2 <= ey1:
            continue
        erase_crop = edited[ey1:ey2, ex1:ex2]
        erase_gray = np.mean(erase_crop, axis=2)
        erase_mask = _expand_mask(erase_gray < ink_threshold)
        erase_crop[erase_mask] = bg_color
        edited[ey1:ey2, ex1:ex2] = erase_crop
        edit_boxes.append(erase_box)

        ref_crop_box = _crop_box_for_reference(
            ref_slot,
            slots=slots,
            index=ref_idx,
            pad_x=pad_x,
            pad_y=pad_y,
            image_size=original_rgb.size,
            container=roi,
        )
        rx1, ry1, rx2, ry2 = ref_crop_box
        ref_patch = arr[ry1:ry2, rx1:rx2].copy()
        ref_gray = np.mean(ref_patch, axis=2)
        glyph_mask = ref_gray < ink_threshold
        glyph_w = max(1, ref_slot.x2 - ref_slot.x1)
        if idx + 1 < len(slots):
            desired_dark_x1 = target_slot.x2 - glyph_w
        elif idx > 0:
            desired_dark_x1 = target_slot.x1
        else:
            desired_dark_x1 = int(round((target_slot.x1 + target_slot.x2 - glyph_w) / 2.0))
        desired_dark_y1 = target_slot.y1
        dx = desired_dark_x1 - ref_slot.x1
        dy = desired_dark_y1 - ref_slot.y1
        pasted_box = _paste_masked(
            edited,
            ref_patch,
            glyph_mask,
            x=rx1 + dx,
            y=ry1 + dy,
        )
        if pasted_box:
            edit_boxes.append(pasted_box)
        replacements.append(
            {
                "index": idx,
                "source_char": source_chars[idx],
                "target_char": target_chars[idx],
                "target_slot": text_run_box(target_slot),
                "reference_index": ref_idx,
                "reference_slot": text_run_box(ref_slot),
                "erase_box": list(erase_box),
                "reference_crop_box": list(ref_crop_box),
                "paste_box": list(pasted_box) if pasted_box else None,
                "alignment": "right_edge" if idx + 1 < len(slots) else "left_edge" if idx > 0 else "center",
            }
        )

    candidate = Image.fromarray(edited).convert("RGB")
    diff = np.any(arr != edited, axis=2)
    changed_y, changed_x = np.where(diff)
    diff_bbox = (
        int(changed_x.min()),
        int(changed_y.min()),
        int(changed_x.max()) + 1,
        int(changed_y.max()) + 1,
    ) if len(changed_x) else None
    expected_mask = np.zeros(diff.shape, dtype=bool)
    for box in edit_boxes:
        x1, y1, x2, y2 = clamp_box(box, original_rgb.size)
        expected_mask[y1:y2, x1:x2] = True
    unexpected_changed = int(np.count_nonzero(diff & ~expected_mask))
    allowed_roi = _union_box(edit_boxes) or plan.target_roi
    hard_report = hard_check(original_rgb, candidate, allowed_roi, plan.protected_boxes)
    passed = bool(hard_report.get("pass")) and unexpected_changed == 0 and bool(replacements)
    return candidate if passed else None, {
        "enabled": True,
        "strategy": "source_glyph_reuse",
        "pass": passed,
        "reason": "applied" if passed else "local_boundary_check_failed",
        "source_text": source_text,
        "target_text": target_text,
        "changed_indices": changed,
        "slot_detection": slot_report,
        "char_slots": [asdict(slot) for slot in slots],
        "replacements": replacements,
        "edit_boxes": [list(box) for box in edit_boxes],
        "allowed_roi": list(allowed_roi),
        "diff_bbox": list(diff_bbox) if diff_bbox else None,
        "changed_pixels": int(np.count_nonzero(diff)),
        "unexpected_changed_pixels": unexpected_changed,
        "hard_check": hard_report,
        "ink_threshold": ink_threshold,
        "background_color": [int(value) for value in bg_color.tolist()],
    }
