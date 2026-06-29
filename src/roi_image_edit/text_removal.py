from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import cv2
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
from roi_image_edit.roi_locator import (
    group_page_text_lines,
    merge_overlapping_text_components,
    page_text_components,
)


TEXT_REMOVAL_SCENARIOS = frozenset({"anchored_text_removal", "text_removal"})


def is_text_removal_classification(classification: dict[str, Any] | None) -> bool:
    if not isinstance(classification, dict):
        return False
    if classification.get("operation") == "remove_text":
        return True
    return str(classification.get("scenario") or "") in TEXT_REMOVAL_SCENARIOS


def _line_records(image: Image.Image) -> list[dict[str, Any]]:
    components = page_text_components(image.convert("RGB"))
    records: list[dict[str, Any]] = []
    for index, group in enumerate(group_page_text_lines(components)):
        merged = merge_overlapping_text_components(group)
        if not merged:
            continue
        x1 = min(run.x1 for run in merged)
        y1 = min(run.y1 for run in merged)
        x2 = max(run.x2 for run in merged)
        y2 = max(run.y2 for run in merged)
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        records.append(
            {
                "index": index,
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "width": int(width),
                "height": int(height),
                "component_count": int(len(merged)),
                "center_y": round(float((y1 + y2) / 2.0), 3),
                "aspect_ratio": round(float(width / height), 4),
            }
        )
    return records


def _looks_like_short_left_text_line(
    line: dict[str, Any],
    *,
    image_size: tuple[int, int],
) -> bool:
    width, height = image_size
    x1, y1, x2, y2 = [int(value) for value in line["box"]]
    line_w = x2 - x1
    line_h = y2 - y1
    if y1 < int(height * 0.52) or y2 > int(height * 0.90):
        return False
    if x1 > int(width * 0.34):
        return False
    if line_w > int(width * 0.28):
        return False
    if line_h < 10 or line_h > 48:
        return False
    return int(line.get("component_count") or 0) >= 2


def _choose_text_removal_line(
    lines: list[dict[str, Any]],
    *,
    image_size: tuple[int, int],
    source_text: str,
    removal_context: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = [
        line
        for line in lines
        if _looks_like_short_left_text_line(line, image_size=image_size)
    ]
    if not candidates:
        raise ValueError("无法自动定位待抹除文字行。")

    candidates.sort(key=lambda item: (int(item["box"][1]), int(item["box"][0])))
    relation = str(removal_context.get("anchor_relation") or "")
    source_count = len(text_chars(source_text))
    if relation == "below":
        clustered: list[list[dict[str, Any]]] = []
        for line in candidates:
            if not clustered:
                clustered.append([line])
                continue
            previous = clustered[-1][-1]
            previous_y2 = int(previous["box"][3])
            current_y1 = int(line["box"][1])
            max_gap = max(18, int(round(max(float(previous["height"]), float(line["height"])) * 1.65)))
            if current_y1 - previous_y2 <= max_gap:
                clustered[-1].append(line)
            else:
                clustered.append([line])
        clusters = [cluster for cluster in clustered if len(cluster) >= 2]
        if clusters:
            selected_cluster = max(
                clusters,
                key=lambda cluster: (
                    len(cluster),
                    int(cluster[-1]["box"][1]),
                ),
            )
            # In anchored removal tasks, the heading is often the first short
            # line; the text to remove is one of the following conclusion rows.
            conclusion_rows = selected_cluster[1:] if len(selected_cluster) > 1 else selected_cluster
            chosen = conclusion_rows[-1]
            return chosen, {
                "selection_rule": "below_anchor_short_left_line_cluster_last_conclusion_row",
                "candidate_count": len(candidates),
                "cluster_size": len(selected_cluster),
                "source_char_count": source_count,
                "anchor_relation": relation,
            }

    chosen = candidates[-1]
    return chosen, {
        "selection_rule": "last_short_left_text_line",
        "candidate_count": len(candidates),
        "source_char_count": source_count,
        "anchor_relation": relation,
    }


def auto_select_text_removal_regions(
    image: Image.Image,
    *,
    instruction_details: dict[str, Any],
    classification: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_text = str(instruction_details.get("source_text") or "")
    if not source_text:
        raise ValueError("missing source text for text removal")

    removal_context = (
        instruction_details.get("removal_context")
        if isinstance(instruction_details.get("removal_context"), dict)
        else {}
    )
    lines = _line_records(image)
    chosen, selection = _choose_text_removal_line(
        lines,
        image_size=image.size,
        source_text=source_text,
        removal_context=removal_context,
    )
    x1, y1, x2, y2 = [int(value) for value in chosen["box"]]
    pad_x = max(8, int(round((x2 - x1) * 0.12)))
    pad_y = max(5, int(round((y2 - y1) * 0.32)))
    rx1, ry1, rx2, ry2 = clamp_box((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), image.size)
    region = {
        "id": "auto_text_removal",
        "rect": {"x": rx1, "y": ry1, "w": rx2 - rx1, "h": ry2 - ry1},
        "auto": True,
        "sourceText": source_text,
        "targetText": "",
        "_autoTextRemoval": True,
        "_autoRemovalLine": chosen,
        "_autoRemovalEvidence": selection,
        "_autoSearchRoi": [rx1, ry1, rx2, ry2],
        "_autoTargetRoi": [rx1, ry1, rx2, ry2],
        "_autoEditRoi": [rx1, ry1, rx2, ry2],
        "_autoSlotBoxes": [],
        "_autoProtectedBoxes": [],
        "_autoSlotQualityReport": {"pass": True, "operation": "remove_text"},
    }
    report = {
        "enabled": True,
        "classification": classification or {},
        "source_text": source_text,
        "removal_context": removal_context,
        "selected_line": chosen,
        "selected_roi": [rx1, ry1, rx2, ry2],
        "selection": selection,
        "line_count": len(lines),
    }
    return [region], report


def _mask_report(mask: np.ndarray, roi: tuple[int, int, int, int]) -> dict[str, Any]:
    x1, y1, x2, y2 = roi
    crop = mask[y1:y2, x1:x2] > 0
    return {
        "roi": [int(x1), int(y1), int(x2), int(y2)],
        "mask_pixels": int(np.count_nonzero(crop)),
        "roi_area": int(max(0, x2 - x1) * max(0, y2 - y1)),
    }


def _expanded_fill_box(
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
    pad_x = max(5, int(round(text_w * 0.05)))
    pad_y_top = max(2, int(round(text_h * 0.08)))
    pad_y_bottom = max(4, int(round(text_h * 0.16)))
    return clamp_box(
        (
            max(rx1, x1 - pad_x),
            max(ry1, y1 - pad_y_top),
            min(rx2, x2 + pad_x),
            min(ry2, y2 + pad_y_bottom),
        ),
        image_size,
    )


def erase_text_in_roi(
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
        raise ValueError("empty text removal roi")

    mask = build_old_text_mask(
        arr,
        roi,
        threshold=threshold,
        dilate_iterations=2,
    )
    if not np.any(mask):
        raise ValueError("text removal roi did not contain removable dark text")

    fill_box = _expanded_fill_box(mask, roi, image_size=original.size)
    if fill_box is None:
        raise ValueError("text removal mask did not produce a fill box")
    fx1, fy1, fx2, fy2 = fill_box
    fill_mask = np.zeros(mask.shape, dtype=np.uint8)
    fill_mask[fy1:fy2, fx1:fx2] = 255

    inpainted = cv2.inpaint(arr, fill_mask, 7, cv2.INPAINT_TELEA).astype(np.float32)
    soft_mask = cv2.GaussianBlur(fill_mask, (0, 0), 3.0).astype(np.float32) / 255.0
    soft_mask[:y1, :] = 0.0
    soft_mask[y2:, :] = 0.0
    soft_mask[:, :x1] = 0.0
    soft_mask[:, x2:] = 0.0
    base_arr = (
        arr.astype(np.float32) * (1.0 - soft_mask[:, :, None])
        + inpainted * soft_mask[:, :, None]
    )
    base_arr = np.clip(base_arr, 0, 255).astype(np.uint8)

    outside = np.ones(mask.shape, dtype=bool)
    outside[y1:y2, x1:x2] = False
    base_arr[outside] = arr[outside]
    edited = Image.fromarray(base_arr).convert("RGB")
    report = {
        "enabled": True,
        "operation": "remove_text",
        "threshold": threshold,
        "mask": _mask_report(mask, roi),
        "fill_mask": _mask_report(fill_mask, roi),
        "fill_box": list(fill_box),
        "hard_check": hard_check(original, edited, roi),
    }
    report["pass"] = bool(report["hard_check"].get("pass")) and report["mask"]["mask_pixels"] > 0
    return edited, report


def text_removal_preview(
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


def process_text_removal_region(
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
    edited, removal_report = erase_text_in_roi(original, roi)
    preview = text_removal_preview(original, edited, roi)

    selected_path = region_dir / "selected_text_removal.png"
    compare_path = region_dir / "selected_text_removal_compare.png"
    report_path = region_dir / "text_removal_report.json"
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
    write_json(report_path, removal_report)
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

    accepted = bool(removal_report.get("pass"))
    summary = {
        "plan": {
            "search_roi": list(roi),
            "target_roi": list(roi),
            "slot_boxes": [],
            "protected_boxes": [],
            "field_key": None,
            "draw_mode": "remove_text",
            "pipeline_profile": region_classification.get("internal_profile") or "photo_scan",
            "classification": region_classification,
            "class_key": region_classification.get("class_key"),
            "roi_policy": region_classification.get("roi_policy"),
            "internal_profile": region_classification.get("internal_profile"),
            "profile_source": region_classification.get("profile_source"),
            "roi_plan": roi_plan,
            "slot_quality_report": {
                "pass": True,
                "operation": "remove_text",
                "source_text": source_text,
                "target_text": "",
                "classification": region_classification,
            },
        },
        "score": None,
        "hard_check": removal_report.get("hard_check"),
        "vision": {
            "enabled": False,
            "accepted": accepted,
            "reason": "local_text_removal_class_bypasses_replacement_candidate_pipeline",
            "revision_rounds": [],
        },
        "trace": {
            "accepted": accepted,
            "final_is_rejected_candidate": not accepted,
            "final_candidate_id": "text_removal_local",
            "final_blocking_stage": None if accepted else "background_cleanup",
            "final_stage_severity": None,
            "revision_round_count": 0,
            "last_round_stop_reason": "text_removed" if accepted else "text_removal_failed",
            "next_round_plan": None if accepted else {"blocking_stage": "background_cleanup"},
        },
        "accepted": accepted,
        "applied": accepted,
        "text_removal_report": removal_report,
        "stage_evidence": {
            "stage_order": ["hard_boundary", "background_cleanup"],
            "text_removal": removal_report,
        },
        "artifacts": {
            "selected_candidate": str(selected_path),
            "selected_compare": str(compare_path),
            "text_removal_report": str(report_path),
            "roi_plan_report": str(roi_plan_path),
            "display_image_is_candidate": not accepted,
        },
        "rejected_fonts": [],
    }
    candidate = {
        "index": 1,
        "kind": "text_removal_local",
        "candidate_id": "text_removal_local",
        "label": "local text removal",
        "score": None,
        "class_key": region_classification.get("class_key"),
        "roi_policy": region_classification.get("roi_policy"),
        "internal_profile": region_classification.get("internal_profile"),
        "profile_source": region_classification.get("profile_source"),
        "stage_context": {
            "class_key": region_classification.get("class_key"),
            "stage_order": ["hard_boundary", "background_cleanup"],
        },
        "dataUrl": _image_to_data_url(preview),
    }
    if progress:
        progress(
            "text_removal_finished",
            {
                "region_id": region_id,
                "classification": region_classification,
                "class_key": region_classification.get("class_key"),
                "roi_policy": region_classification.get("roi_policy"),
                "internal_profile": region_classification.get("internal_profile"),
                "profile_source": region_classification.get("profile_source"),
                "accepted": accepted,
                "text_removal_report": removal_report,
            },
        )
    return edited, edited, [candidate], summary, accepted


def _image_to_data_url(img: Image.Image) -> str:
    from io import BytesIO
    import base64

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
