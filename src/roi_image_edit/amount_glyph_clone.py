from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import base64
import re
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from roi_image_edit.iterative_pipeline import TextRun, clamp_box, hard_check, text_chars, write_json
from roi_image_edit.source_glyph_reuse import detect_source_glyph_slots
from roi_image_edit.roi_locator import text_run_box


AMOUNT_GLYPH_CLONE_SCENARIOS = frozenset({"amount_glyph_clone"})


@dataclass(frozen=True)
class AmountGlyphSource:
    image: Image.Image
    roi: tuple[int, int, int, int]
    text: str
    label: str = ""


@dataclass(frozen=True)
class AmountGlyphEdit:
    roi: tuple[int, int, int, int]
    source_text: str
    target_text: str
    label: str = ""


@dataclass
class _GlyphPatch:
    char: str
    label: str
    source_text: str
    source_index: int
    slot_index: int
    slot: TextRun
    crop_box: tuple[int, int, int, int]
    patch: np.ndarray
    mask: np.ndarray

    @property
    def slot_width(self) -> int:
        return max(1, self.slot.x2 - self.slot.x1)

    @property
    def slot_height(self) -> int:
        return max(1, self.slot.y2 - self.slot.y1)

    @property
    def offset_x(self) -> int:
        return self.slot.x1 - self.crop_box[0]

    @property
    def offset_y(self) -> int:
        return self.slot.y1 - self.crop_box[1]


def is_amount_glyph_clone_classification(classification: dict[str, Any] | None) -> bool:
    if not isinstance(classification, dict):
        return False
    return str(classification.get("scenario") or "") in AMOUNT_GLYPH_CLONE_SCENARIOS


def _box_area(box: tuple[int, int, int, int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _union_box(boxes: list[tuple[int, int, int, int]] | tuple[tuple[int, int, int, int], ...]) -> tuple[int, int, int, int] | None:
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
    return [max(0, ordered[idx + 1].x1 - ordered[idx].x2) for idx in range(len(ordered) - 1)]


def _median_gap(slots: tuple[TextRun, ...]) -> int:
    gaps = _slot_gaps(slots)
    if gaps:
        return max(1, int(round(float(np.median(gaps)))))
    widths = [max(1, slot.x2 - slot.x1) for slot in slots]
    return max(1, int(round(float(np.median(widths)) * 0.18))) if widths else 2


def _bounded_to_container(
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


def _slot_crop_box(
    slot: TextRun,
    *,
    slots: tuple[TextRun, ...],
    index: int,
    image_size: tuple[int, int],
    container: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    pad_x = 2
    pad_y = 2
    left = slot.x1 - pad_x
    right = slot.x2 + pad_x
    if index > 0:
        left = max(left, slots[index - 1].x2 + 1)
    if index + 1 < len(slots):
        right = min(right, slots[index + 1].x1 - 1)
    return _bounded_to_container(
        (left, slot.y1 - pad_y, right, slot.y2 + pad_y),
        image_size=image_size,
        container=container,
    )


def _glyph_mask(patch: np.ndarray, ink_threshold: int) -> np.ndarray:
    gray = (
        patch[:, :, 0].astype(np.float32) * 0.299
        + patch[:, :, 1].astype(np.float32) * 0.587
        + patch[:, :, 2].astype(np.float32) * 0.114
    )
    mask = gray < ink_threshold
    if not np.any(mask):
        return mask
    kernel = np.ones((2, 2), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def _background_color(arr: np.ndarray, roi: tuple[int, int, int, int], clear_box: tuple[int, int, int, int]) -> np.ndarray:
    rx1, ry1, rx2, ry2 = roi
    cx1, cy1, cx2, cy2 = clear_box
    crop = arr[ry1:ry2, rx1:rx2]
    if crop.size == 0:
        return np.array([255, 255, 255], dtype=np.uint8)
    local_mask = np.ones(crop.shape[:2], dtype=bool)
    local_mask[cy1 - ry1 : cy2 - ry1, cx1 - rx1 : cx2 - rx1] = False
    gray = (
        crop[:, :, 0].astype(np.float32) * 0.299
        + crop[:, :, 1].astype(np.float32) * 0.587
        + crop[:, :, 2].astype(np.float32) * 0.114
    )
    sample_mask = local_mask & (gray > 235)
    samples = crop[sample_mask]
    if samples.size == 0:
        samples = crop[local_mask]
    if samples.size == 0:
        samples = crop.reshape(-1, 3)
    return np.median(samples, axis=0).astype(np.uint8)


def _patch_report(glyph: _GlyphPatch) -> dict[str, Any]:
    return {
        "char": glyph.char,
        "label": glyph.label,
        "source_text": glyph.source_text,
        "source_index": glyph.source_index,
        "slot_index": glyph.slot_index,
        "slot": text_run_box(glyph.slot),
        "crop_box": list(glyph.crop_box),
        "mask_pixels": int(np.count_nonzero(glyph.mask)),
    }


def _collect_glyphs(
    samples: list[AmountGlyphSource],
    *,
    ink_threshold: int,
) -> tuple[dict[str, list[_GlyphPatch]], list[dict[str, Any]]]:
    library: dict[str, list[_GlyphPatch]] = {}
    reports: list[dict[str, Any]] = []
    for source_index, sample in enumerate(samples):
        image = sample.image.convert("RGB")
        roi = clamp_box(sample.roi, image.size)
        chars = text_chars(sample.text)
        slots, slot_report = detect_source_glyph_slots(image, roi, sample.text)
        sample_report: dict[str, Any] = {
            "label": sample.label or f"source_{source_index}",
            "text": sample.text,
            "roi": list(roi),
            "slot_detection": slot_report,
            "glyph_count": 0,
        }
        if len(slots) != len(chars):
            sample_report["reason"] = "slot_count_mismatch"
            reports.append(sample_report)
            continue
        arr = np.array(image)
        for slot_index, (char, slot) in enumerate(zip(chars, slots)):
            crop_box = _slot_crop_box(
                slot,
                slots=slots,
                index=slot_index,
                image_size=image.size,
                container=roi,
            )
            x1, y1, x2, y2 = crop_box
            if x2 <= x1 or y2 <= y1:
                continue
            patch = arr[y1:y2, x1:x2].copy()
            mask = _glyph_mask(patch, ink_threshold)
            if not np.any(mask):
                continue
            glyph = _GlyphPatch(
                char=char,
                label=sample.label or f"source_{source_index}",
                source_text=sample.text,
                source_index=source_index,
                slot_index=slot_index,
                slot=slot,
                crop_box=crop_box,
                patch=patch,
                mask=mask,
            )
            library.setdefault(char, []).append(glyph)
            sample_report["glyph_count"] += 1
        reports.append(sample_report)
    return library, reports


def _select_glyphs(
    target_text: str,
    library: dict[str, list[_GlyphPatch]],
) -> tuple[list[_GlyphPatch], list[dict[str, Any]]]:
    counters: dict[str, int] = {}
    selected: list[_GlyphPatch] = []
    missing: list[dict[str, Any]] = []
    for index, char in enumerate(text_chars(target_text)):
        options = library.get(char) or []
        if not options:
            missing.append({"index": index, "char": char, "reason": "glyph_not_available"})
            continue
        pick = options[counters.get(char, 0) % len(options)]
        counters[char] = counters.get(char, 0) + 1
        selected.append(pick)
    return selected, missing


def _planned_slots(
    selected: list[_GlyphPatch],
    *,
    source_slots: tuple[TextRun, ...],
) -> tuple[list[tuple[int, int, int, int]], dict[str, Any]]:
    if not source_slots:
        raise ValueError("amount glyph clone requires source slots")
    gap = _median_gap(source_slots)
    plus_slot = source_slots[0]
    digit_slots = source_slots[1:] or source_slots
    digit_bottom = int(round(float(np.median([slot.y2 for slot in digit_slots]))))
    plus_center_y = (plus_slot.y1 + plus_slot.y2) / 2.0
    x = 0
    natural: list[tuple[int, int, int, int]] = []
    for idx, glyph in enumerate(selected):
        if glyph.char == "+":
            y1 = int(round(plus_center_y - glyph.slot_height / 2.0))
        else:
            y1 = digit_bottom - glyph.slot_height
        natural.append((x, y1, x + glyph.slot_width, y1 + glyph.slot_height))
        x += glyph.slot_width
        if idx + 1 < len(selected):
            x += gap
    natural_right = natural[-1][2] if natural else 0
    desired_right = max(slot.x2 for slot in source_slots)
    dx = desired_right - natural_right
    slots = [(x1 + dx, y1, x2 + dx, y2) for x1, y1, x2, y2 in natural]
    return slots, {
        "gap": gap,
        "desired_right": desired_right,
        "natural_width": natural_right,
        "x_shift": dx,
        "source_slots": [text_run_box(slot) for slot in source_slots],
    }


def _paste_masked(
    target: np.ndarray,
    glyph: _GlyphPatch,
    *,
    planned_slot: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    x = planned_slot[0] - glyph.offset_x
    y = planned_slot[1] - glyph.offset_y
    height, width = target.shape[:2]
    patch_h, patch_w = glyph.patch.shape[:2]
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
    local_mask = glyph.mask[patch_y1:patch_y2, patch_x1:patch_x2]
    if not np.any(local_mask):
        return None
    region = target[y1:y2, x1:x2]
    region[local_mask] = glyph.patch[patch_y1:patch_y2, patch_x1:patch_x2][local_mask]
    target[y1:y2, x1:x2] = region
    return x1, y1, x2, y2


def replace_amount_with_glyph_sources(
    image: Image.Image,
    roi: tuple[int, int, int, int],
    *,
    source_text: str,
    target_text: str,
    glyph_sources: list[AmountGlyphSource] | tuple[AmountGlyphSource, ...],
    label: str = "",
    ink_threshold: int = 230,
) -> tuple[Image.Image, dict[str, Any]]:
    original = image.convert("RGB")
    roi = clamp_box(roi, original.size)
    source_slots, source_slot_report = detect_source_glyph_slots(original, roi, source_text)
    if len(source_slots) != len(text_chars(source_text)):
        raise ValueError("amount glyph clone could not detect source amount slots")

    samples = [
        AmountGlyphSource(original, roi, source_text, label=label or "target_source"),
        *list(glyph_sources),
    ]
    library, sample_reports = _collect_glyphs(samples, ink_threshold=ink_threshold)
    selected, missing = _select_glyphs(target_text, library)
    if missing:
        return original.copy(), {
            "enabled": True,
            "operation": "amount_glyph_clone",
            "strategy": "cross_amount_glyph_clone",
            "pass": False,
            "reason": "missing_target_glyphs",
            "source_text": source_text,
            "target_text": target_text,
            "missing": missing,
            "source_slot_detection": source_slot_report,
            "glyph_sources": sample_reports,
        }

    planned, layout_report = _planned_slots(selected, source_slots=source_slots)
    source_box = _union_box([text_run_box(slot) for slot in source_slots])
    target_box = _union_box(planned)
    if source_box is None or target_box is None:
        raise ValueError("amount glyph clone could not compute source or target box")
    clear_box_unclamped = (
        min(source_box[0], target_box[0]) - 2,
        min(source_box[1], target_box[1]) - 2,
        max(source_box[2], target_box[2]) + 2,
        max(source_box[3], target_box[3]) + 2,
    )
    clear_box = _bounded_to_container(clear_box_unclamped, image_size=original.size, container=roi)
    if clear_box != clear_box_unclamped:
        return original.copy(), {
            "enabled": True,
            "operation": "amount_glyph_clone",
            "strategy": "cross_amount_glyph_clone",
            "pass": False,
            "reason": "target_layout_exceeds_explicit_roi",
            "source_text": source_text,
            "target_text": target_text,
            "roi": list(roi),
            "target_box": list(target_box),
            "clear_box_unclamped": list(clear_box_unclamped),
            "clear_box": list(clear_box),
            "source_slot_detection": source_slot_report,
            "glyph_sources": sample_reports,
        }

    arr = np.array(original)
    edited = arr.copy()
    background = _background_color(arr, roi, clear_box)
    cx1, cy1, cx2, cy2 = clear_box
    edited[cy1:cy2, cx1:cx2] = background
    paste_boxes: list[tuple[int, int, int, int]] = []
    placement: list[dict[str, Any]] = []
    for idx, (glyph, slot_box) in enumerate(zip(selected, planned)):
        pasted_box = _paste_masked(edited, glyph, planned_slot=slot_box)
        if pasted_box:
            paste_boxes.append(pasted_box)
        placement.append(
            {
                "index": idx,
                "char": glyph.char,
                "planned_slot": list(slot_box),
                "paste_box": list(pasted_box) if pasted_box else None,
                "glyph": _patch_report(glyph),
            }
        )

    candidate = Image.fromarray(edited).convert("RGB")
    diff = np.any(arr != edited, axis=2)
    allowed = np.zeros(diff.shape, dtype=bool)
    allowed[cy1:cy2, cx1:cx2] = True
    unexpected_changed = int(np.count_nonzero(diff & ~allowed))
    hard_report = hard_check(original, candidate, roi)
    passed = bool(hard_report.get("pass")) and unexpected_changed == 0 and bool(paste_boxes)
    report = {
        "enabled": True,
        "operation": "amount_glyph_clone",
        "strategy": "cross_amount_glyph_clone",
        "pass": passed,
        "reason": "applied" if passed else "hard_boundary_failed",
        "source_text": source_text,
        "target_text": target_text,
        "roi": list(roi),
        "source_box": list(source_box),
        "target_box": list(target_box),
        "clear_box": list(clear_box),
        "paste_boxes": [list(box) for box in paste_boxes],
        "background_color": [int(value) for value in background.tolist()],
        "ink_threshold": ink_threshold,
        "layout": layout_report,
        "source_slot_detection": source_slot_report,
        "glyph_sources": sample_reports,
        "placement": placement,
        "changed_pixels": int(np.count_nonzero(diff)),
        "unexpected_changed_pixels": unexpected_changed,
        "hard_check": hard_report,
    }
    return candidate if passed else original.copy(), report


def clone_amounts_with_glyph_sources(
    image: Image.Image,
    *,
    edits: list[AmountGlyphEdit] | tuple[AmountGlyphEdit, ...],
    glyph_sources: list[AmountGlyphSource] | tuple[AmountGlyphSource, ...],
    ink_threshold: int = 230,
) -> tuple[Image.Image, dict[str, Any]]:
    edited = image.convert("RGB")
    reports: list[dict[str, Any]] = []
    accepted = True
    for index, edit in enumerate(edits):
        edited, report = replace_amount_with_glyph_sources(
            edited,
            edit.roi,
            source_text=edit.source_text,
            target_text=edit.target_text,
            glyph_sources=glyph_sources,
            label=edit.label or f"edit_{index + 1}",
            ink_threshold=ink_threshold,
        )
        reports.append(report)
        accepted = accepted and bool(report.get("pass"))
        if not report.get("pass"):
            break
    return edited, {
        "enabled": True,
        "operation": "amount_glyph_clone",
        "strategy": "cross_amount_glyph_clone",
        "pass": accepted,
        "edit_count": len(edits),
        "accepted_count": sum(1 for report in reports if report.get("pass")),
        "edits": reports,
    }


def amount_glyph_clone_preview(
    original: Image.Image,
    edited: Image.Image,
    rois: list[tuple[int, int, int, int]],
    *,
    scale: int = 3,
) -> Image.Image:
    preview = edited.copy()
    draw = ImageDraw.Draw(preview)
    for roi in rois:
        draw.rectangle(roi, outline=(220, 60, 60), width=1)
    if scale != 1:
        preview = preview.resize((preview.width * scale, preview.height * scale), Image.Resampling.NEAREST)
    return preview


def save_amount_glyph_clone_artifacts(
    *,
    original: Image.Image,
    edited: Image.Image,
    report: dict[str, Any],
    output_path: Path,
    report_path: Path | None = None,
) -> dict[str, str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    edited.save(output_path)
    resolved_report_path = report_path or output_path.with_suffix(".amount_glyph_clone_report.json")
    write_json(resolved_report_path, report)
    rois = [
        tuple(int(value) for value in edit.get("roi", []))
        for edit in report.get("edits", [])
        if isinstance(edit, dict) and len(edit.get("roi", [])) == 4
    ]
    preview_path = output_path.with_suffix(".amount_glyph_clone_preview.png")
    amount_glyph_clone_preview(original, edited, rois).save(preview_path)
    return {
        "output": str(output_path),
        "report": str(resolved_report_path),
        "preview": str(preview_path),
    }


def image_to_data_url(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def process_amount_glyph_clone_region(
    original: Image.Image,
    roi: tuple[int, int, int, int],
    *,
    source_text: str,
    target_text: str,
    glyph_sources: list[AmountGlyphSource] | tuple[AmountGlyphSource, ...],
    run_dir: Path,
    region_id: str,
    classification: dict[str, Any] | None = None,
    progress: Any | None = None,
) -> tuple[Image.Image, Image.Image, list[dict[str, Any]], dict[str, Any], bool]:
    region_dir = run_dir / "regions" / re.sub(r"[^A-Za-z0-9_.-]+", "_", region_id or "region")
    region_dir.mkdir(parents=True, exist_ok=True)
    roi = clamp_box(roi, original.size)
    edited, clone_report = replace_amount_with_glyph_sources(
        original,
        roi,
        source_text=source_text,
        target_text=target_text,
        glyph_sources=glyph_sources,
        label=region_id,
    )
    report_path = region_dir / "amount_glyph_clone_report.json"
    selected_path = region_dir / "selected_amount_glyph_clone.png"
    preview_path = region_dir / "selected_amount_glyph_clone_preview.png"
    write_json(report_path, clone_report)
    edited.save(selected_path)
    amount_glyph_clone_preview(original, edited, [roi], scale=4).save(preview_path)
    accepted = bool(clone_report.get("pass"))
    region_classification = classification or {}
    summary = {
        "plan": {
            "search_roi": list(roi),
            "target_roi": list(roi),
            "slot_boxes": [],
            "protected_boxes": [],
            "field_key": "amount",
            "draw_mode": "amount_glyph_clone",
            "pipeline_profile": region_classification.get("internal_profile") or "clean_digital",
            "classification": region_classification,
            "class_key": region_classification.get("class_key"),
            "roi_policy": region_classification.get("roi_policy"),
            "internal_profile": region_classification.get("internal_profile"),
            "profile_source": region_classification.get("profile_source"),
        },
        "score": None,
        "hard_check": clone_report.get("hard_check"),
        "vision": {
            "enabled": False,
            "accepted": accepted,
            "reason": "local_amount_glyph_clone_bypasses_general_replacement_candidate_pipeline",
            "revision_rounds": [],
        },
        "trace": {
            "accepted": accepted,
            "final_is_rejected_candidate": not accepted,
            "final_candidate_id": "amount_glyph_clone",
            "final_blocking_stage": None if accepted else "amount_glyph_clone",
            "final_stage_severity": None,
            "revision_round_count": 0,
            "last_round_stop_reason": "amount_glyph_cloned" if accepted else "amount_glyph_clone_failed",
        },
        "accepted": accepted,
        "applied": accepted,
        "amount_glyph_clone_report": clone_report,
        "stage_evidence": {
            "stage_order": ["hard_boundary", "amount_glyph_clone"],
            "amount_glyph_clone": clone_report,
        },
        "artifacts": {
            "selected_candidate": str(selected_path),
            "selected_compare": str(preview_path),
            "amount_glyph_clone_report": str(report_path),
            "display_image_is_candidate": not accepted,
        },
        "rejected_fonts": [],
    }
    candidate = {
        "index": 1,
        "kind": "amount_glyph_clone",
        "candidate_id": "amount_glyph_clone",
        "label": "amount glyph clone",
        "score": None,
        "class_key": region_classification.get("class_key"),
        "roi_policy": region_classification.get("roi_policy"),
        "internal_profile": region_classification.get("internal_profile"),
        "profile_source": region_classification.get("profile_source"),
        "stage_context": {
            "class_key": region_classification.get("class_key"),
            "stage_order": ["hard_boundary", "amount_glyph_clone"],
        },
        "dataUrl": image_to_data_url(Image.open(preview_path)),
    }
    if progress:
        progress(
            "amount_glyph_clone_finished",
            {
                "region_id": region_id,
                "classification": region_classification,
                "class_key": region_classification.get("class_key"),
                "roi_policy": region_classification.get("roi_policy"),
                "internal_profile": region_classification.get("internal_profile"),
                "profile_source": region_classification.get("profile_source"),
                "accepted": accepted,
                "amount_glyph_clone_report": clone_report,
            },
        )
    return edited, edited, [candidate], summary, accepted
