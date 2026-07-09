from __future__ import annotations

from io import BytesIO
from pathlib import Path
import base64
import re
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from roi_image_edit.iterative_pipeline import (
    build_old_text_mask,
    clamp_box,
    hard_check,
    mask_bbox,
    text_chars,
    write_json,
)


TEXT_REDACTION_SCENARIOS = frozenset({"anchored_text_redaction", "text_redaction"})


def is_text_redaction_classification(classification: dict[str, Any] | None) -> bool:
    if not isinstance(classification, dict):
        return False
    if classification.get("operation") == "redact_text":
        return True
    return str(classification.get("scenario") or "") in TEXT_REDACTION_SCENARIOS


def _mask_report(mask: np.ndarray, roi: tuple[int, int, int, int]) -> dict[str, Any]:
    x1, y1, x2, y2 = roi
    crop = mask[y1:y2, x1:x2] > 0
    return {
        "roi": [int(x1), int(y1), int(x2), int(y2)],
        "mask_pixels": int(np.count_nonzero(crop)),
        "roi_area": int(max(0, x2 - x1) * max(0, y2 - y1)),
    }


def _expanded_redaction_box(
    mask: np.ndarray,
    roi: tuple[int, int, int, int],
    *,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    bbox = mask_bbox(mask > 0)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    rx1, ry1, rx2, ry2 = roi
    text_w = max(1, x2 - x1)
    text_h = max(1, y2 - y1)
    pad_x = max(4, int(round(text_w * 0.10)))
    pad_y = max(2, int(round(text_h * 0.10)))
    return clamp_box(
        (
            max(rx1, x1 - pad_x),
            max(ry1, y1 - pad_y),
            min(rx2, x2 + pad_x),
            min(ry2, y2 + pad_y),
        ),
        image_size,
    )


def _redaction_colors(arr: np.ndarray, roi: tuple[int, int, int, int], fill_box: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    rx1, ry1, rx2, ry2 = roi
    fx1, fy1, fx2, fy2 = fill_box
    crop = arr[ry1:ry2, rx1:rx2]
    if crop.size == 0:
        base = np.array([238, 238, 238], dtype=np.int16)
    else:
        local_mask = np.ones(crop.shape[:2], dtype=bool)
        local_mask[fy1 - ry1 : fy2 - ry1, fx1 - rx1 : fx2 - rx1] = False
        samples = crop[local_mask]
        if samples.size == 0:
            samples = crop.reshape(-1, 3)
        base = np.median(samples, axis=0).astype(np.int16)

    if int(np.mean(base)) < 128:
        first = np.clip(base + 26, 0, 255)
        second = np.clip(base + 48, 0, 255)
    else:
        first = np.clip(base - 8, 0, 255)
        second = np.clip(base - 22, 0, 255)
    return first.astype(np.uint8), second.astype(np.uint8)


def _redaction_pattern(
    width: int,
    height: int,
    *,
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    block = max(3, min(7, int(round(min(width, height) / 3.5))))
    pattern = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(0, height, block):
        for x in range(0, width, block):
            color = first if ((x // block) + (y // block)) % 2 == 0 else second
            pattern[y : y + block, x : x + block] = color
    return pattern


def redact_text_in_roi(
    image: Image.Image,
    roi: tuple[int, int, int, int],
    *,
    threshold: int = 210,
) -> tuple[Image.Image, dict[str, Any]]:
    original = image.convert("RGB")
    arr = np.array(original)
    roi = clamp_box(roi, original.size)
    x1, y1, x2, y2 = roi
    if x2 <= x1 or y2 <= y1:
        raise ValueError("empty text redaction roi")

    mask = build_old_text_mask(
        arr,
        roi,
        threshold=threshold,
        dilate_iterations=1,
    )
    if not np.any(mask):
        raise ValueError("text redaction roi did not contain redactable dark text")

    fill_box = _expanded_redaction_box(mask, roi, image_size=original.size)
    if fill_box is None:
        raise ValueError("text redaction mask did not produce a fill box")

    fx1, fy1, fx2, fy2 = fill_box
    edited_arr = arr.copy()
    first, second = _redaction_colors(arr, roi, fill_box)
    edited_arr[fy1:fy2, fx1:fx2] = _redaction_pattern(
        fx2 - fx1,
        fy2 - fy1,
        first=first,
        second=second,
    )
    edited = Image.fromarray(edited_arr).convert("RGB")
    report = {
        "enabled": True,
        "operation": "redact_text",
        "style": "mosaic_block",
        "threshold": threshold,
        "mask": _mask_report(mask, roi),
        "fill_box": list(fill_box),
        "hard_check": hard_check(original, edited, roi),
    }
    report["pass"] = bool(report["hard_check"].get("pass")) and report["mask"]["mask_pixels"] > 0
    return edited, report


def text_redaction_preview(
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


def process_text_redaction_region(
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
    del target_text
    region_dir = run_dir / "regions" / re.sub(r"[^A-Za-z0-9_.-]+", "_", region_id or "region")
    region_dir.mkdir(parents=True, exist_ok=True)
    roi = clamp_box(roi, original.size)
    edited, redaction_report = redact_text_in_roi(original, roi)
    preview = text_redaction_preview(original, edited, roi)

    selected_path = region_dir / "selected_text_redaction.png"
    compare_path = region_dir / "selected_text_redaction_compare.png"
    report_path = region_dir / "text_redaction_report.json"
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
        "target_slot_count": 0,
    }
    write_json(report_path, redaction_report)
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

    accepted = bool(redaction_report.get("pass"))
    summary = {
        "plan": {
            "search_roi": list(roi),
            "target_roi": list(roi),
            "slot_boxes": [],
            "protected_boxes": [],
            "field_key": None,
            "draw_mode": "redact_text",
            "pipeline_profile": region_classification.get("internal_profile") or "photo_scan",
            "classification": region_classification,
            "class_key": region_classification.get("class_key"),
            "roi_policy": region_classification.get("roi_policy"),
            "internal_profile": region_classification.get("internal_profile"),
            "profile_source": region_classification.get("profile_source"),
            "roi_plan": roi_plan,
            "slot_quality_report": {
                "pass": True,
                "operation": "redact_text",
                "source_text": source_text,
                "target_text": "",
                "classification": region_classification,
            },
        },
        "score": None,
        "hard_check": redaction_report.get("hard_check"),
        "vision": {
            "enabled": False,
            "accepted": accepted,
            "reason": "local_text_redaction_class_bypasses_replacement_candidate_pipeline",
            "revision_rounds": [],
        },
        "trace": {
            "accepted": accepted,
            "final_is_rejected_candidate": not accepted,
            "final_candidate_id": "text_redaction_local",
            "final_blocking_stage": None if accepted else "text_redaction",
            "final_stage_severity": None,
            "revision_round_count": 0,
            "last_round_stop_reason": "text_redacted" if accepted else "text_redaction_failed",
            "next_round_plan": None if accepted else {"blocking_stage": "text_redaction"},
        },
        "accepted": accepted,
        "applied": accepted,
        "text_redaction_report": redaction_report,
        "stage_evidence": {
            "stage_order": ["hard_boundary", "text_redaction"],
            "text_redaction": redaction_report,
        },
        "artifacts": {
            "selected_candidate": str(selected_path),
            "selected_compare": str(compare_path),
            "text_redaction_report": str(report_path),
            "roi_plan_report": str(roi_plan_path),
            "display_image_is_candidate": not accepted,
        },
        "rejected_fonts": [],
    }
    candidate = {
        "index": 1,
        "kind": "text_redaction_local",
        "candidate_id": "text_redaction_local",
        "label": "local text redaction",
        "score": None,
        "class_key": region_classification.get("class_key"),
        "roi_policy": region_classification.get("roi_policy"),
        "internal_profile": region_classification.get("internal_profile"),
        "profile_source": region_classification.get("profile_source"),
        "stage_context": {
            "class_key": region_classification.get("class_key"),
            "stage_order": ["hard_boundary", "text_redaction"],
        },
        "dataUrl": _image_to_data_url(preview),
    }
    if progress:
        progress(
            "text_redaction_finished",
            {
                "region_id": region_id,
                "classification": region_classification,
                "class_key": region_classification.get("class_key"),
                "roi_policy": region_classification.get("roi_policy"),
                "internal_profile": region_classification.get("internal_profile"),
                "profile_source": region_classification.get("profile_source"),
                "accepted": accepted,
                "text_redaction_report": redaction_report,
            },
        )
    return edited, edited, [candidate], summary, accepted


def _image_to_data_url(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
