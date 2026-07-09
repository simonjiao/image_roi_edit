from __future__ import annotations

from io import BytesIO
from pathlib import Path
import base64
import re
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from roi_image_edit.iterative_pipeline import (
    build_old_text_mask,
    clamp_box,
    hard_check,
    mask_bbox,
    text_chars,
    write_json,
)


AMOUNT_VALUE_SCENARIOS = frozenset({"amount_value_replace"})

AMOUNT_FONT_CANDIDATES = (
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/SFCompact.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
)


def is_amount_replacement_classification(classification: dict[str, Any] | None) -> bool:
    if not isinstance(classification, dict):
        return False
    return str(classification.get("scenario") or "") in AMOUNT_VALUE_SCENARIOS


def _mask_report(mask: np.ndarray, roi: tuple[int, int, int, int]) -> dict[str, Any]:
    x1, y1, x2, y2 = roi
    crop = mask[y1:y2, x1:x2] > 0
    return {
        "roi": [int(x1), int(y1), int(x2), int(y2)],
        "mask_pixels": int(np.count_nonzero(crop)),
        "roi_area": int(max(0, x2 - x1) * max(0, y2 - y1)),
    }


def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont | None:
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return None


def _font_bbox(font: ImageFont.FreeTypeFont, text: str) -> tuple[int, int, int, int]:
    bbox = font.getbbox(text)
    return int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])


def _choose_amount_font(
    source_text: str,
    target_text: str,
    source_box: tuple[int, int, int, int],
) -> tuple[ImageFont.FreeTypeFont, dict[str, Any]]:
    sx1, sy1, sx2, sy2 = source_box
    source_w = max(1, sx2 - sx1)
    source_h = max(1, sy2 - sy1)
    best: tuple[float, ImageFont.FreeTypeFont, dict[str, Any]] | None = None
    for path in AMOUNT_FONT_CANDIDATES:
        for size in range(16, 31):
            font = _load_font(path, size)
            if font is None:
                continue
            bbox = _font_bbox(font, source_text)
            width = max(1, bbox[2] - bbox[0])
            height = max(1, bbox[3] - bbox[1])
            score = abs(width - source_w) + abs(height - source_h) * 2.0
            report = {
                "font_path": path,
                "font_size": size,
                "source_bbox": list(bbox),
                "source_width": width,
                "source_height": height,
                "target_bbox": list(_font_bbox(font, target_text)),
                "score": round(float(score), 3),
            }
            if best is None or score < best[0]:
                best = (score, font, report)
    if best is None:
        font = ImageFont.load_default()
        return font, {"font_path": "Pillow default", "font_size": None, "score": None}
    return best[1], best[2]


def _local_background_color(arr: np.ndarray, roi: tuple[int, int, int, int], clear_box: tuple[int, int, int, int]) -> np.ndarray:
    rx1, ry1, rx2, ry2 = roi
    cx1, cy1, cx2, cy2 = clear_box
    crop = arr[ry1:ry2, rx1:rx2]
    if crop.size == 0:
        return np.array([255, 255, 255], dtype=np.uint8)
    local_mask = np.ones(crop.shape[:2], dtype=bool)
    local_mask[cy1 - ry1 : cy2 - ry1, cx1 - rx1 : cx2 - rx1] = False
    samples = crop[local_mask]
    if samples.size == 0:
        samples = crop.reshape(-1, 3)
    return np.median(samples, axis=0).astype(np.uint8)


def _dark_text_color(arr: np.ndarray, mask: np.ndarray, roi: tuple[int, int, int, int]) -> tuple[int, int, int]:
    x1, y1, x2, y2 = roi
    crop = arr[y1:y2, x1:x2]
    gray = (
        crop[:, :, 0].astype(np.float32) * 0.299
        + crop[:, :, 1].astype(np.float32) * 0.587
        + crop[:, :, 2].astype(np.float32) * 0.114
    )
    crop_mask = gray < 115
    samples = crop[crop_mask]
    if samples.size == 0:
        samples = crop[mask[y1:y2, x1:x2] > 0]
    if samples.size == 0:
        return (50, 50, 50)
    color = np.median(samples, axis=0)
    return tuple(int(max(0, min(255, round(value)))) for value in color)


def _text_bbox_at(
    font: ImageFont.FreeTypeFont,
    text: str,
    *,
    desired_right: int,
    desired_center_y: float,
) -> tuple[int, int, int, int, tuple[int, int]]:
    bbox = _font_bbox(font, text)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    target_x1 = int(round(desired_right - width))
    target_y1 = int(round(desired_center_y - height / 2.0))
    draw_xy = (target_x1 - bbox[0], target_y1 - bbox[1])
    return target_x1, target_y1, target_x1 + width, target_y1 + height, draw_xy


def replace_amount_in_roi(
    image: Image.Image,
    roi: tuple[int, int, int, int],
    *,
    source_text: str,
    target_text: str,
    threshold: int = 210,
) -> tuple[Image.Image, dict[str, Any]]:
    original = image.convert("RGB")
    arr = np.array(original)
    roi = clamp_box(roi, original.size)
    x1, y1, x2, y2 = roi
    if x2 <= x1 or y2 <= y1:
        raise ValueError("empty amount replacement roi")
    if not source_text or not target_text:
        raise ValueError("amount replacement requires source and target text")

    mask = build_old_text_mask(
        arr,
        roi,
        threshold=threshold,
        dilate_iterations=1,
    )
    source_box = mask_bbox(mask > 0)
    if source_box is None:
        raise ValueError("amount replacement roi did not contain replaceable dark text")

    font, font_report = _choose_amount_font(source_text, target_text, source_box)
    sx1, sy1, sx2, sy2 = source_box
    desired_right = sx2
    desired_center_y = (sy1 + sy2) / 2.0
    tx1, ty1, tx2, ty2, draw_xy = _text_bbox_at(
        font,
        target_text,
        desired_right=desired_right,
        desired_center_y=desired_center_y,
    )
    pad_x = 2
    pad_y = 2
    clear_box = clamp_box(
        (
            min(sx1, tx1) - pad_x,
            min(sy1, ty1) - pad_y,
            max(sx2, tx2) + pad_x,
            max(sy2, ty2) + pad_y,
        ),
        original.size,
    )
    if clear_box[0] < x1 or clear_box[1] < y1 or clear_box[2] > x2 or clear_box[3] > y2:
        raise ValueError("amount replacement target would exceed explicit ROI")

    edited_arr = arr.copy()
    background = _local_background_color(arr, roi, clear_box)
    cx1, cy1, cx2, cy2 = clear_box
    edited_arr[cy1:cy2, cx1:cx2] = background
    edited = Image.fromarray(edited_arr).convert("RGB")
    draw = ImageDraw.Draw(edited)
    draw.text(draw_xy, target_text, font=font, fill=_dark_text_color(arr, mask, roi))
    report = {
        "enabled": True,
        "operation": "replace_amount_value",
        "source_text": source_text,
        "target_text": target_text,
        "alignment": "right_anchor_preserve_suffix",
        "threshold": threshold,
        "mask": _mask_report(mask, roi),
        "source_box": list(source_box),
        "target_box": [int(tx1), int(ty1), int(tx2), int(ty2)],
        "clear_box": list(clear_box),
        "font": font_report,
        "hard_check": hard_check(original, edited, roi),
    }
    report["pass"] = bool(report["hard_check"].get("pass")) and report["mask"]["mask_pixels"] > 0
    return edited, report


def amount_replacement_preview(
    original: Image.Image,
    edited: Image.Image,
    roi: tuple[int, int, int, int],
    *,
    scale: int = 4,
) -> Image.Image:
    x1, y1, x2, y2 = roi
    pad = 12
    preview_box = clamp_box((x1 - pad, y1 - pad, x2 + pad, y2 + pad), original.size)
    old = original.crop(preview_box)
    new = edited.crop(preview_box)
    width = old.width + new.width + 8
    height = max(old.height, new.height)
    preview = Image.new("RGB", (width, height), (245, 245, 245))
    preview.paste(old, (0, 0))
    preview.paste(new, (old.width + 8, 0))
    draw = ImageDraw.Draw(preview)
    draw.line((old.width + 3, 0, old.width + 3, height), fill=(220, 60, 60), width=1)
    if scale != 1:
        preview = preview.resize((preview.width * scale, preview.height * scale), Image.Resampling.NEAREST)
    return preview


def process_amount_replacement_region(
    original: Image.Image,
    roi: tuple[int, int, int, int],
    *,
    source_text: str,
    target_text: str,
    run_dir: Path,
    region_id: str,
    classification: dict[str, Any] | None = None,
    progress: Any | None = None,
) -> tuple[Image.Image, Image.Image, list[dict[str, Any]], dict[str, Any], bool]:
    region_dir = run_dir / "regions" / re.sub(r"[^A-Za-z0-9_.-]+", "_", region_id or "region")
    region_dir.mkdir(parents=True, exist_ok=True)
    roi = clamp_box(roi, original.size)
    edited, amount_report = replace_amount_in_roi(
        original,
        roi,
        source_text=source_text,
        target_text=target_text,
    )
    preview = amount_replacement_preview(original, edited, roi)

    selected_path = region_dir / "selected_amount_replacement.png"
    compare_path = region_dir / "selected_amount_replacement_compare.png"
    report_path = region_dir / "amount_replacement_report.json"
    roi_plan_path = region_dir / "roi_plan_report.json"
    edited.save(selected_path)
    preview.save(compare_path)

    region_classification = classification or {}
    roi_plan = {
        "search_roi": list(roi),
        "edit_roi": list(roi),
        "expanded_edit_roi": None,
        "roi_policy": region_classification.get("roi_policy") or "auto",
        "source_slot_count": len(text_chars(source_text)),
        "target_slot_count": len(text_chars(target_text)),
        "alignment": "right_anchor_preserve_suffix",
    }
    write_json(report_path, amount_report)
    write_json(
        roi_plan_path,
        {
            **roi_plan,
            "classification": region_classification,
            "class_key": region_classification.get("class_key"),
            "internal_profile": region_classification.get("internal_profile"),
            "profile_source": region_classification.get("profile_source"),
        },
    )

    accepted = bool(amount_report.get("pass"))
    summary = {
        "plan": {
            "search_roi": list(roi),
            "target_roi": list(roi),
            "slot_boxes": [],
            "protected_boxes": [],
            "field_key": "amount",
            "draw_mode": "replace_amount_value",
            "pipeline_profile": region_classification.get("internal_profile") or "clean_digital",
            "classification": region_classification,
            "class_key": region_classification.get("class_key"),
            "roi_policy": region_classification.get("roi_policy"),
            "internal_profile": region_classification.get("internal_profile"),
            "profile_source": region_classification.get("profile_source"),
            "roi_plan": roi_plan,
            "slot_quality_report": {
                "pass": True,
                "operation": "replace_amount_value",
                "source_text": source_text,
                "target_text": target_text,
                "classification": region_classification,
            },
        },
        "score": None,
        "hard_check": amount_report.get("hard_check"),
        "vision": {
            "enabled": False,
            "accepted": accepted,
            "reason": "local_amount_value_class_bypasses_general_replacement_candidate_pipeline",
            "revision_rounds": [],
        },
        "trace": {
            "accepted": accepted,
            "final_is_rejected_candidate": not accepted,
            "final_candidate_id": "amount_replacement_local",
            "final_blocking_stage": None if accepted else "amount_value_replace",
            "final_stage_severity": None,
            "revision_round_count": 0,
            "last_round_stop_reason": "amount_replaced" if accepted else "amount_replacement_failed",
            "next_round_plan": None if accepted else {"blocking_stage": "amount_value_replace"},
        },
        "accepted": accepted,
        "applied": accepted,
        "amount_replacement_report": amount_report,
        "stage_evidence": {
            "stage_order": ["hard_boundary", "amount_value_replace"],
            "amount_value_replace": amount_report,
        },
        "artifacts": {
            "selected_candidate": str(selected_path),
            "selected_compare": str(compare_path),
            "amount_replacement_report": str(report_path),
            "roi_plan_report": str(roi_plan_path),
            "display_image_is_candidate": not accepted,
        },
        "rejected_fonts": [],
    }
    candidate = {
        "index": 1,
        "kind": "amount_replacement_local",
        "candidate_id": "amount_replacement_local",
        "label": "local amount replacement",
        "score": None,
        "class_key": region_classification.get("class_key"),
        "roi_policy": region_classification.get("roi_policy"),
        "internal_profile": region_classification.get("internal_profile"),
        "profile_source": region_classification.get("profile_source"),
        "stage_context": {
            "class_key": region_classification.get("class_key"),
            "stage_order": ["hard_boundary", "amount_value_replace"],
        },
        "dataUrl": _image_to_data_url(preview),
    }
    if progress:
        progress(
            "amount_replacement_finished",
            {
                "region_id": region_id,
                "classification": region_classification,
                "class_key": region_classification.get("class_key"),
                "roi_policy": region_classification.get("roi_policy"),
                "internal_profile": region_classification.get("internal_profile"),
                "profile_source": region_classification.get("profile_source"),
                "accepted": accepted,
                "amount_replacement_report": amount_report,
            },
        )
    return edited, edited, [candidate], summary, accepted


def _image_to_data_url(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
