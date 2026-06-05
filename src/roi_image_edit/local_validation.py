from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from roi_image_edit.iterative_pipeline import (
    CandidateParams,
    RenderPlan,
    build_trailing_value_cleanup_mask,
    char_alignment_gate,
    char_gray_band_metrics,
    char_pose_metrics,
    clamp_box,
    draw_replacement_layer,
    extra_source_slot_cleanup_issues,
    extra_source_slot_cleanup_metrics,
    font_style_gate,
    gray_band_counts,
    hard_check,
    is_mostly_cjk,
    local_score,
    params_label,
    replacement_char_bboxes,
    render_candidate,
    strict_gate_issues,
    strict_visual_metrics,
)
from roi_image_edit.roi_locator import (
    clamp_box_to_container,
    protected_box_overlaps_row,
    text_chars,
    text_run_box,
)
from roi_image_edit.stages import stage_gate_for_report as ordered_stage_gate_for_report


def text_complexity_ratio(plan: RenderPlan, params: CandidateParams) -> float:
    source_text = plan.source_text or ""
    target_text = plan.target_text or ""
    if not source_text or not target_text or source_text == target_text:
        return 1.0
    try:
        font = ImageFont.truetype(params.font_path, params.font_size)
    except OSError:
        return 1.0

    def rendered_ink(text: str) -> int:
        bbox = font.getbbox(text)
        width = max(1, bbox[2] - bbox[0] + 8)
        height = max(1, bbox[3] - bbox[1] + 8)
        layer = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(layer)
        draw.text((4 - bbox[0], 4 - bbox[1]), text, fill=255, font=font)
        arr = np.array(layer)
        return int(np.count_nonzero(arr > 24))

    source_ink = rendered_ink(source_text)
    target_ink = rendered_ink(target_text)
    if source_ink <= 0 or target_ink <= 0:
        return 1.0
    return max(0.75, min(1.65, target_ink / source_ink))


def rendered_glyph_complexity(params: dict[str, Any] | CandidateParams, text: str) -> int | None:
    if not text:
        return None
    try:
        if isinstance(params, CandidateParams):
            font_path = params.font_path
            font_size = params.font_size
        else:
            font_path = str(params.get("font_path") or "")
            font_size = int(params.get("font_size") or 0)
        font = ImageFont.truetype(font_path, font_size)
    except (OSError, TypeError, ValueError):
        return None
    bbox = font.getbbox(text)
    width = max(1, bbox[2] - bbox[0] + 8)
    height = max(1, bbox[3] - bbox[1] + 8)
    layer = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(layer)
    draw.text((4 - bbox[0], 4 - bbox[1]), text, fill=255, font=font)
    return int(np.count_nonzero(np.array(layer) > 24))


def gray_band_profile_for_box(
    gray: np.ndarray,
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> dict[str, Any] | None:
    x1, y1, x2, y2 = clamp_box(box, image_size)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = gray[y1:y2, x1:x2]
    if crop.size <= 0:
        return None
    counts = gray_band_counts(crop)
    area = int(crop.size)
    lt165 = max(1, int(counts.get("lt165") or 0))
    return {
        "box": [x1, y1, x2, y2],
        "area": area,
        "mean_gray": round(float(np.mean(crop)), 3),
        "std_gray": round(float(np.std(crop)), 3),
        "counts": counts,
        "density": {
            "lt55": round(float(counts["lt55"]) / max(1, area), 5),
            "lt70": round(float(counts["lt70"]) / max(1, area), 5),
            "lt165": round(float(counts["lt165"]) / max(1, area), 5),
        },
        "share": {
            "lt55_of_lt165": round(float(counts["lt55"]) / lt165, 5),
            "lt70_of_lt165": round(float(counts["lt70"]) / lt165, 5),
            "outer_120_165_of_lt165": round(float(counts["band_120_165"]) / lt165, 5),
        },
    }


def same_row_reference_boxes(plan: RenderPlan) -> tuple[tuple[int, int, int, int], ...]:
    tx1, ty1, tx2, ty2 = plan.target_roi
    row_h = max(1, ty2 - ty1)
    boxes: list[tuple[int, int, int, int]] = []
    for box in plan.protected_boxes:
        if not protected_box_overlaps_row(box, plan.target_roi):
            continue
        if box[2] <= tx1 or box[0] >= tx2:
            if box[3] - box[1] >= max(4, int(round(row_h * 0.35))):
                boxes.append(box)
    return tuple(boxes[:8])


def build_reference_profile(
    original: Image.Image,
    plan: RenderPlan,
    params: CandidateParams,
) -> dict[str, Any]:
    gray = cv2.cvtColor(np.array(original.convert("RGB")), cv2.COLOR_RGB2GRAY)
    source_box = plan.source_reference_box or plan.target_roi
    source_profile = gray_band_profile_for_box(gray, source_box, original.size)
    target_profile = gray_band_profile_for_box(gray, plan.target_roi, original.size)
    slot_profiles = [
        profile
        for profile in (
            gray_band_profile_for_box(gray, text_run_box(slot), original.size)
            for slot in tuple(sorted(plan.slot_boxes, key=lambda item: item.x1))
        )
        if profile is not None
    ]
    neighbor_profiles = [
        profile
        for profile in (
            gray_band_profile_for_box(gray, box, original.size)
            for box in same_row_reference_boxes(plan)
        )
        if profile is not None and (profile.get("counts") or {}).get("lt165", 0) >= 8
    ]

    source_counts = (source_profile or {}).get("counts") or {}
    source_density = (source_profile or {}).get("density") or {}
    try:
        source_lt55 = float(source_counts.get("lt55") or 0.0)
        source_lt165 = float(source_counts.get("lt165") or 0.0)
        source_core_density = float(source_density.get("lt55") or 0.0)
        source_std_gray = float((source_profile or {}).get("std_gray") or 0.0)
    except (TypeError, ValueError):
        source_lt55 = 0.0
        source_lt165 = 0.0
        source_core_density = 0.0
        source_std_gray = 0.0

    neighbor_core_densities: list[float] = []
    neighbor_outer_shares: list[float] = []
    for profile in neighbor_profiles:
        density = profile.get("density") or {}
        share = profile.get("share") or {}
        try:
            neighbor_core_densities.append(float(density.get("lt55") or 0.0))
            neighbor_outer_shares.append(float(share.get("outer_120_165_of_lt165") or 0.0))
        except (TypeError, ValueError):
            continue
    neighbor_core_density = float(np.median(neighbor_core_densities)) if neighbor_core_densities else None
    neighbor_outer_share = float(np.median(neighbor_outer_shares)) if neighbor_outer_shares else None

    source_count = len(text_chars(plan.source_text or ""))
    target_count = len(text_chars(plan.target_text))
    complexity_ratio = text_complexity_ratio(plan, params)
    count_ratio = float(target_count / source_count) if source_count and target_count else 1.0
    count_allowance = 1.0 + max(0.0, min(0.9, count_ratio - 1.0)) * 0.55
    complexity_allowance = 1.0 + max(0.0, min(0.65, complexity_ratio - 1.0)) * 0.45
    core_delta_limit = max(58.0, source_lt55 * 0.32 * count_allowance * complexity_allowance)
    char_core_delta_limit = max(48.0, source_lt55 * 0.70 * complexity_allowance / max(1.0, source_count or 1.0))
    core_mean_lighten_limit = max(2.0, min(8.0, source_std_gray * 0.14))

    reference_core_density = max(
        source_core_density,
        neighbor_core_density if neighbor_core_density is not None else 0.0,
    )
    core_density_conflict = False
    if neighbor_core_density is not None:
        core_density_conflict = abs(source_core_density - neighbor_core_density) >= 0.045
    if neighbor_core_density is None:
        arbitration_rule = "source_only_no_same_row_neighbor"
        selected_core_reference = "source"
    elif source_core_density >= neighbor_core_density:
        arbitration_rule = "use_darker_of_source_and_same_row_neighbor"
        selected_core_reference = "source"
    else:
        arbitration_rule = "use_darker_of_source_and_same_row_neighbor"
        selected_core_reference = "same_row_neighbor"
    if reference_core_density >= 0.18:
        opacity_floor = 0.72
    elif reference_core_density >= 0.12:
        opacity_floor = 0.68
    elif reference_core_density >= 0.08:
        opacity_floor = 0.64
    else:
        opacity_floor = 0.60

    return {
        "enabled": source_profile is not None,
        "source_text": plan.source_text or "",
        "target_text": plan.target_text,
        "source_box": list(source_box),
        "target_roi": list(plan.target_roi),
        "source": source_profile,
        "target": target_profile,
        "slots": slot_profiles,
        "same_row_neighbors": neighbor_profiles,
        "dynamic_ink": {
            "source_lt55": round(source_lt55, 3),
            "source_lt165": round(source_lt165, 3),
            "source_core_density": round(source_core_density, 5),
            "neighbor_core_density": None if neighbor_core_density is None else round(neighbor_core_density, 5),
            "neighbor_outer_share": None if neighbor_outer_share is None else round(neighbor_outer_share, 5),
            "text_complexity_ratio": round(float(complexity_ratio), 4),
            "text_count_ratio": round(float(count_ratio), 4),
            "roi_lt55_delta_limit": round(float(core_delta_limit), 3),
            "char_lt55_delta_limit": round(float(char_core_delta_limit), 3),
            "core_mean_lighten_limit": round(float(core_mean_lighten_limit), 3),
            "opacity_floor_for_excess_core": round(float(opacity_floor), 3),
            "basis": "source_text_region_and_same_row_neighbors",
            "arbitration": {
                "source_core_density": round(float(source_core_density), 5),
                "neighbor_core_density": (
                    None if neighbor_core_density is None else round(float(neighbor_core_density), 5)
                ),
                "conflict_detected": core_density_conflict,
                "selected_core_reference": selected_core_reference,
                "rule": arbitration_rule,
            },
        },
    }


def dynamic_ink_limits(report: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    profile = report.get("reference_profile")
    if not isinstance(profile, dict):
        return {}
    dynamic = profile.get("dynamic_ink")
    return dynamic if isinstance(dynamic, dict) else {}


def opacity_floor_for_excess_black(report: dict[str, Any] | None) -> float:
    dynamic = dynamic_ink_limits(report)
    try:
        return max(0.55, min(0.76, float(dynamic.get("opacity_floor_for_excess_core"))))
    except (TypeError, ValueError):
        return 0.64


def local_ink_balance_issues(report: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    strict_gate = report.get("strict_gate")
    dynamic_limits = dynamic_ink_limits(report)
    complexity_ratio = 1.0
    if isinstance(strict_gate, dict):
        try:
            complexity_ratio = float(strict_gate.get("text_complexity_ratio") or 1.0)
        except (TypeError, ValueError):
            complexity_ratio = 1.0
    char_bands = report.get("char_gray_band_metrics")
    if isinstance(char_bands, dict) and char_bands.get("enabled"):
        per_char = [item for item in char_bands.get("per_char", []) if isinstance(item, dict)]

        def slot_area(item: dict[str, Any]) -> float:
            box = item.get("slot_box") or []
            if len(box) != 4:
                return 0.0
            return max(1.0, float((int(box[2]) - int(box[0])) * (int(box[3]) - int(box[1]))))

        def neighbor_core_bounds_changed_item(item: dict[str, Any]) -> bool:
            if item.get("source_char") == item.get("target_char"):
                return False
            if not is_mostly_cjk(str(item.get("target_char") or "")):
                return False
            try:
                changed_index = int(item.get("index") or 0)
            except (TypeError, ValueError):
                changed_index = 0
            neighbor_items = [
                candidate
                for candidate in per_char
                if candidate.get("source_char") == candidate.get("target_char")
                and is_mostly_cjk(str(candidate.get("target_char") or ""))
            ]
            if not neighbor_items:
                return False
            neighbor = min(
                neighbor_items,
                key=lambda candidate: abs(int(candidate.get("index") or 0) - changed_index),
            )
            changed_counts = item.get("new") or {}
            neighbor_counts = neighbor.get("new") or {}
            if not isinstance(changed_counts, dict) or not isinstance(neighbor_counts, dict):
                return False
            changed_area = slot_area(item)
            neighbor_area = slot_area(neighbor)
            try:
                changed_core_density = float(changed_counts.get("lt55") or 0.0) / changed_area
                neighbor_core_density = float(neighbor_counts.get("lt55") or 0.0) / neighbor_area
                changed_lt70_density = float(changed_counts.get("lt70") or 0.0) / changed_area
                neighbor_lt70_density = float(neighbor_counts.get("lt70") or 0.0) / neighbor_area
                changed_lt165 = max(1.0, float(changed_counts.get("lt165") or 0.0))
                neighbor_lt165 = max(1.0, float(neighbor_counts.get("lt165") or 0.0))
                changed_outer_share = float(changed_counts.get("band_120_165") or 0.0) / changed_lt165
                neighbor_outer_share = float(neighbor_counts.get("band_120_165") or 0.0) / neighbor_lt165
            except (TypeError, ValueError, ZeroDivisionError):
                return False
            return (
                neighbor_core_density >= 0.08
                and changed_core_density <= neighbor_core_density + 0.022
                and changed_lt70_density <= neighbor_lt70_density + 0.040
                and changed_outer_share <= neighbor_outer_share + 0.060
            )

        for item in per_char:
            if not isinstance(item, dict):
                continue
            if item.get("source_char") == item.get("target_char"):
                continue
            delta = item.get("delta") or {}
            old = item.get("old") or {}
            try:
                old_lt55 = float(old.get("lt55") or 0.0)
                old_lt165 = float(old.get("lt165") or 0.0)
                lt55_delta = float(delta.get("lt55") or 0.0)
                lt165_delta = float(delta.get("lt165") or 0.0)
                band_70_90_delta = float(delta.get("band_70_90") or 0.0)
                band_90_120_delta = float(delta.get("band_90_120") or 0.0)
            except (TypeError, ValueError):
                continue
            if old_lt55 <= 0 or old_lt165 <= 0:
                continue

            middle_delta = band_70_90_delta + band_90_120_delta
            complexity_core_allowance = min(0.86, 0.70 + max(0.0, complexity_ratio - 1.0) * 0.55)
            try:
                profile_char_limit = float(dynamic_limits.get("char_lt55_delta_limit") or 0.0)
            except (TypeError, ValueError):
                profile_char_limit = 0.0
            max_core_delta = max(52.0, profile_char_limit, old_lt55 * complexity_core_allowance)
            if lt55_delta > max_core_delta and middle_delta < -20.0:
                if complexity_ratio >= 1.08 and neighbor_core_bounds_changed_item(item):
                    continue
                issues.append(
                    {
                        "type": "changed_char_core_too_black_hard",
                        "index": item.get("index"),
                        "source_char": item.get("source_char"),
                        "target_char": item.get("target_char"),
                        "lt55_delta": lt55_delta,
                        "limit": round(max_core_delta, 3),
                        "middle_gray_delta": middle_delta,
                        "text_complexity_ratio": round(complexity_ratio, 4),
                    }
                )
            elif lt55_delta > max(70.0, old_lt55 * 0.85) and lt165_delta > max(36.0, old_lt165 * 0.12):
                if complexity_ratio >= 1.08 and neighbor_core_bounds_changed_item(item):
                    continue
                issues.append(
                    {
                        "type": "changed_char_core_too_black",
                        "index": item.get("index"),
                        "source_char": item.get("source_char"),
                        "target_char": item.get("target_char"),
                        "lt55_delta": lt55_delta,
                        "limit": round(max(70.0, old_lt55 * 0.85), 3),
                    }
                )

    bands = (report.get("strict_visual_metrics") or {}).get("bands")
    if isinstance(bands, dict):
        try:
            old_lt55 = float(bands.get("old_lt55_pixels") or 0.0)
            lt55_delta = float(bands.get("lt55_delta") or 0.0)
            old_share = bands.get("old_lt55_share_of_lt165")
            new_share = bands.get("new_lt55_share_of_lt165")
            share_delta = None if old_share is None or new_share is None else float(new_share) - float(old_share)
        except (TypeError, ValueError):
            old_lt55 = 0.0
            lt55_delta = 0.0
            share_delta = None
        try:
            roi_core_delta_limit = float(dynamic_limits.get("roi_lt55_delta_limit") or 0.0)
        except (TypeError, ValueError):
            roi_core_delta_limit = 0.0
        roi_core_delta_limit = max(48.0, roi_core_delta_limit or old_lt55 * 0.32)
        if old_lt55 >= 32.0 and lt55_delta > roi_core_delta_limit:
            issues.append(
                {
                    "type": "roi_core_too_black",
                    "lt55_delta": lt55_delta,
                    "limit": round(roi_core_delta_limit, 3),
                    "limit_source": "reference_profile",
                }
            )
        if share_delta is not None and share_delta > 0.085:
            issues.append(
                {
                    "type": "roi_black_core_share_too_high",
                    "share_delta": round(float(share_delta), 4),
                    "limit": 0.085,
                }
            )
    return issues


def local_stroke_body_issues(
    report: dict[str, Any],
    *,
    allow_excess_black_core: bool = False,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if not isinstance(report, dict):
        return issues
    if not allow_excess_black_core and report_has_excess_black_core(report):
        return issues

    char_bands = report.get("char_gray_band_metrics")
    if not isinstance(char_bands, dict) or not char_bands.get("enabled"):
        return issues

    for item in char_bands.get("per_char", []):
        if not isinstance(item, dict):
            continue
        if item.get("source_char") == item.get("target_char"):
            continue
        delta = item.get("delta") or {}
        old = item.get("old") or {}
        try:
            old_lt165 = float(old.get("lt165") or 0.0)
            lt55_delta = float(delta.get("lt55") or 0.0)
            lt165_delta = float(delta.get("lt165") or 0.0)
            band_55_70_delta = float(delta.get("band_55_70") or 0.0)
            band_70_90_delta = float(delta.get("band_70_90") or 0.0)
            band_90_120_delta = float(delta.get("band_90_120") or 0.0)
            band_120_165_delta = float(delta.get("band_120_165") or 0.0)
            old_band_120_165 = float(old.get("band_120_165") or 0.0)
        except (TypeError, ValueError):
            continue
        if old_lt165 <= 0:
            continue

        middle_delta = band_70_90_delta + band_90_120_delta
        gray_body_delta = band_55_70_delta + middle_delta
        min_body_delta = min(12.0, old_lt165 * 0.03)
        strict_gate = report.get("strict_gate")
        complexity_ratio = 1.0
        if isinstance(strict_gate, dict):
            try:
                complexity_ratio = float(strict_gate.get("text_complexity_ratio") or 1.0)
            except (TypeError, ValueError):
                complexity_ratio = 1.0
        params = report.get("params")
        stroke_opacity = 0.0
        blur = 0.0
        if isinstance(params, dict):
            try:
                stroke_opacity = float(params.get("stroke_opacity") or 0.0)
                blur = float(params.get("blur") or 0.0)
            except (TypeError, ValueError):
                stroke_opacity = 0.0
                blur = 0.0
        middle_limit = -18.0 if complexity_ratio >= 1.08 else -30.0
        if lt165_delta < -6.0:
            issues.append(
                {
                    "type": "changed_char_stroke_body_too_small",
                    "index": item.get("index"),
                    "source_char": item.get("source_char"),
                    "target_char": item.get("target_char"),
                    "lt165_delta": round(lt165_delta, 3),
                    "limit": -6.0,
                    "lt55_delta": round(lt55_delta, 3),
                    "middle_gray_delta": round(middle_delta, 3),
                    "gray_body_delta": round(gray_body_delta, 3),
                    "band_120_165_delta": round(band_120_165_delta, 3),
                }
            )
            continue

        core_is_not_deficient = lt55_delta >= -8.0
        middle_is_missing = middle_delta < middle_limit
        body_gain_is_too_small = lt165_delta < min_body_delta and gray_body_delta < -18.0
        hard_core_with_thin_body = lt55_delta > 28.0 and middle_delta < -28.0
        light_edge_substitutes_for_body = (
            middle_delta < -16.0
            and band_120_165_delta > 38.0
            and complexity_ratio >= 1.08
        )
        fine_strokes_too_soft = (
            complexity_ratio >= 1.08
            and band_70_90_delta < -12.0
            and band_120_165_delta > 62.0
            and (stroke_opacity < 0.085 or blur > 0.42)
        )
        if core_is_not_deficient and (
            middle_is_missing
            or body_gain_is_too_small
            or hard_core_with_thin_body
            or light_edge_substitutes_for_body
            or fine_strokes_too_soft
        ):
            neighbor_style_clear = not local_neighbor_style_issues(report, allow_excess_black_core=True)
            clean_edge_body_is_bounded = (
                complexity_ratio >= 1.08
                and neighbor_style_clear
                and lt165_delta >= min_body_delta - 4.0
                and lt55_delta >= 30.0
                and middle_delta >= -42.0
                and not light_edge_substitutes_for_body
                and band_120_165_delta <= max(46.0, old_band_120_165 * 0.42)
            )
            if clean_edge_body_is_bounded:
                continue
            issues.append(
                {
                    "type": "changed_char_fine_strokes_too_soft"
                    if fine_strokes_too_soft
                    else "changed_char_stroke_body_too_narrow",
                    "index": item.get("index"),
                    "source_char": item.get("source_char"),
                    "target_char": item.get("target_char"),
                    "lt165_delta": round(lt165_delta, 3),
                    "min_body_delta": round(min_body_delta, 3),
                    "lt55_delta": round(lt55_delta, 3),
                    "middle_gray_delta": round(middle_delta, 3),
                    "gray_body_delta": round(gray_body_delta, 3),
                    "band_120_165_delta": round(band_120_165_delta, 3),
                    "middle_limit": round(middle_limit, 3),
                    "text_complexity_ratio": round(complexity_ratio, 4),
                    "stroke_opacity": round(stroke_opacity, 3),
                    "blur": round(blur, 3),
                }
            )
    return issues


def local_pose_issues(report: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if not isinstance(report, dict):
        return issues
    pose = report.get("char_pose_metrics")
    if not isinstance(pose, dict) or not pose.get("enabled"):
        return issues
    for item in pose.get("per_char", []):
        if not isinstance(item, dict) or not item.get("changed"):
            continue
        try:
            source_raw = item.get("source_slot_shear")
            source_shear = None if source_raw is None else float(source_raw)
            neighbor_raw = item.get("neighbor_shear")
            neighbor_shear = None if neighbor_raw is None else float(neighbor_raw)
            reference_raw = item.get("reference_shear")
            reference_shear = None if reference_raw is None else float(reference_raw)
            applied_shear = float(item.get("applied_shear") or 0.0)
        except (TypeError, ValueError):
            continue
        if reference_shear is None:
            reference_shear = source_shear
        if reference_raw is None and source_shear is not None and neighbor_shear is not None and abs(neighbor_shear) >= 0.018:
            reference_shear = source_shear * 0.75 + neighbor_shear * 0.25
        if reference_shear is None:
            reference_shear = neighbor_shear
        if reference_shear is None:
            continue
        if abs(reference_shear) < 0.05:
            continue
        min_applied_abs = abs(reference_shear) * 0.78
        if reference_shear * applied_shear <= 0 or abs(applied_shear) < min_applied_abs:
            issues.append(
                {
                    "type": "changed_char_pose_shear_too_weak",
                    "index": item.get("index"),
                    "source_char": item.get("source_char"),
                    "target_char": item.get("target_char"),
                    "source_slot_shear": None if source_shear is None else round(source_shear, 4),
                    "neighbor_shear": None if neighbor_shear is None else round(neighbor_shear, 4),
                    "reference_shear": round(reference_shear, 4),
                    "applied_shear": round(applied_shear, 4),
                    "min_applied_abs_shear": round(min_applied_abs, 4),
                }
            )
    return issues


def report_has_fine_strokes_too_soft(report: dict[str, Any] | None) -> bool:
    if not isinstance(report, dict):
        return False
    for issue in local_stroke_body_issues(report):
        if isinstance(issue, dict) and issue.get("type") == "changed_char_fine_strokes_too_soft":
            return True
    return False


def report_has_outer_gray_halo(report: dict[str, Any] | None) -> bool:
    if not isinstance(report, dict):
        return False
    return bool(local_outer_gray_halo_issues(report, allow_excess_black_core=True))


def local_outer_gray_halo_issues(
    report: dict[str, Any],
    *,
    allow_excess_black_core: bool = False,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if not isinstance(report, dict):
        return issues
    if not allow_excess_black_core and report_has_excess_black_core(report):
        return issues
    char_bands = report.get("char_gray_band_metrics")
    if not isinstance(char_bands, dict) or not char_bands.get("enabled"):
        return issues
    per_char = [item for item in char_bands.get("per_char", []) if isinstance(item, dict)]
    if len(per_char) < 2:
        return issues

    params = report.get("params")
    if not isinstance(params, dict):
        params = {}

    def slot_area(item: dict[str, Any]) -> float:
        box = item.get("slot_box") or []
        if len(box) != 4:
            return 0.0
        return max(1.0, float((int(box[2]) - int(box[0])) * (int(box[3]) - int(box[1]))))

    changed_items = [
        item
        for item in per_char
        if item.get("source_char") != item.get("target_char")
        and is_mostly_cjk(str(item.get("target_char") or ""))
    ]
    neighbor_items = [
        item
        for item in per_char
        if item.get("source_char") == item.get("target_char")
        and is_mostly_cjk(str(item.get("target_char") or ""))
    ]
    if not changed_items or not neighbor_items:
        return issues

    for changed in changed_items:
        try:
            changed_index = int(changed.get("index") or 0)
        except (TypeError, ValueError):
            changed_index = 0
        neighbor = min(
            neighbor_items,
            key=lambda item: abs(int(item.get("index") or 0) - changed_index),
        )
        target_char = str(changed.get("target_char") or "")
        neighbor_char = str(neighbor.get("target_char") or "")
        target_complexity = rendered_glyph_complexity(params, target_char)
        neighbor_complexity = rendered_glyph_complexity(params, neighbor_char)
        if target_complexity is not None and neighbor_complexity:
            if target_complexity < neighbor_complexity * 0.75:
                continue

        changed_counts = changed.get("new") or {}
        neighbor_counts = neighbor.get("new") or {}
        old_counts = changed.get("old") or {}
        delta_counts = changed.get("delta") or {}
        if (
            not isinstance(changed_counts, dict)
            or not isinstance(neighbor_counts, dict)
            or not isinstance(old_counts, dict)
            or not isinstance(delta_counts, dict)
        ):
            continue
        changed_area = slot_area(changed)
        neighbor_area = slot_area(neighbor)
        try:
            changed_lt165 = max(1.0, float(changed_counts.get("lt165") or 0.0))
            neighbor_lt165 = max(1.0, float(neighbor_counts.get("lt165") or 0.0))
            changed_outer = float(changed_counts.get("band_120_165") or 0.0)
            neighbor_outer = float(neighbor_counts.get("band_120_165") or 0.0)
            old_outer = float(old_counts.get("band_120_165") or 0.0)
            outer_delta = float(delta_counts.get("band_120_165") or 0.0)
        except (TypeError, ValueError):
            continue

        changed_outer_share = changed_outer / changed_lt165
        neighbor_outer_share = neighbor_outer / neighbor_lt165
        changed_outer_density = changed_outer / max(1.0, changed_area)
        neighbor_outer_density = neighbor_outer / max(1.0, neighbor_area)
        outer_share_gap = changed_outer_share - neighbor_outer_share
        outer_density_gap = changed_outer_density - neighbor_outer_density
        excess_outer_limit = max(48.0, old_outer * 0.42)
        if outer_share_gap > 0.060 and outer_delta > excess_outer_limit and outer_density_gap > 0.030:
            issues.append(
                {
                    "type": "changed_char_neighbor_outer_gray_halo_too_high",
                    "index": changed.get("index"),
                    "source_char": changed.get("source_char"),
                    "target_char": target_char,
                    "neighbor_index": neighbor.get("index"),
                    "neighbor_char": neighbor_char,
                    "target_complexity": target_complexity,
                    "neighbor_complexity": neighbor_complexity,
                    "changed_outer_share": round(changed_outer_share, 4),
                    "neighbor_outer_share": round(neighbor_outer_share, 4),
                    "outer_share_gap": round(outer_share_gap, 4),
                    "changed_outer_density": round(changed_outer_density, 4),
                    "neighbor_outer_density": round(neighbor_outer_density, 4),
                    "outer_density_gap": round(outer_density_gap, 4),
                    "band_120_165_delta": round(outer_delta, 3),
                    "band_120_165_delta_limit": round(excess_outer_limit, 3),
                }
            )
    return issues


def local_neighbor_style_issues(
    report: dict[str, Any],
    *,
    allow_excess_black_core: bool = False,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if not isinstance(report, dict):
        return issues
    if not allow_excess_black_core and report_has_excess_black_core(report):
        return issues
    issues.extend(local_outer_gray_halo_issues(report, allow_excess_black_core=allow_excess_black_core))
    char_bands = report.get("char_gray_band_metrics")
    if not isinstance(char_bands, dict) or not char_bands.get("enabled"):
        return issues
    per_char = [item for item in char_bands.get("per_char", []) if isinstance(item, dict)]
    if len(per_char) < 2:
        return issues

    params = report.get("params")
    if not isinstance(params, dict):
        params = {}

    def slot_area(item: dict[str, Any]) -> float:
        box = item.get("slot_box") or []
        if len(box) != 4:
            return 0.0
        return max(1.0, float((int(box[2]) - int(box[0])) * (int(box[3]) - int(box[1]))))

    def density(counts: dict[str, Any], key: str, area: float) -> float:
        try:
            return float(counts.get(key) or 0.0) / max(1.0, area)
        except (TypeError, ValueError):
            return 0.0

    changed_items = [
        item
        for item in per_char
        if item.get("source_char") != item.get("target_char")
        and is_mostly_cjk(str(item.get("target_char") or ""))
    ]
    neighbor_items = [
        item
        for item in per_char
        if item.get("source_char") == item.get("target_char")
        and is_mostly_cjk(str(item.get("target_char") or ""))
    ]
    if not changed_items or not neighbor_items:
        return issues

    for changed in changed_items:
        try:
            changed_index = int(changed.get("index") or 0)
        except (TypeError, ValueError):
            changed_index = 0
        neighbor = min(
            neighbor_items,
            key=lambda item: abs(int(item.get("index") or 0) - changed_index),
        )
        target_char = str(changed.get("target_char") or "")
        neighbor_char = str(neighbor.get("target_char") or "")
        target_complexity = rendered_glyph_complexity(params, target_char)
        neighbor_complexity = rendered_glyph_complexity(params, neighbor_char)
        if target_complexity is not None and neighbor_complexity:
            if target_complexity < neighbor_complexity * 0.75:
                continue

        changed_counts = changed.get("new") or {}
        neighbor_counts = neighbor.get("new") or {}
        if not isinstance(changed_counts, dict) or not isinstance(neighbor_counts, dict):
            continue
        changed_area = slot_area(changed)
        neighbor_area = slot_area(neighbor)
        changed_core_density = density(changed_counts, "lt55", changed_area)
        neighbor_core_density = density(neighbor_counts, "lt55", neighbor_area)
        changed_lt70_density = density(changed_counts, "lt70", changed_area)
        neighbor_lt70_density = density(neighbor_counts, "lt70", neighbor_area)
        changed_lt165 = max(1.0, float(changed_counts.get("lt165") or 0.0))
        neighbor_lt165 = max(1.0, float(neighbor_counts.get("lt165") or 0.0))
        changed_outer_share = float(changed_counts.get("band_120_165") or 0.0) / changed_lt165
        neighbor_outer_share = float(neighbor_counts.get("band_120_165") or 0.0) / neighbor_lt165

        core_density_gap = neighbor_core_density - changed_core_density
        lt70_density_gap = neighbor_lt70_density - changed_lt70_density
        outer_share_gap = changed_outer_share - neighbor_outer_share
        neighbor_core_is_meaningful = neighbor_core_density >= 0.08 and changed_core_density >= 0.08
        core_gap_is_visible = core_density_gap > 0.022 and outer_share_gap > 0.045
        gray_edge_is_substituting = core_density_gap > 0.018 and outer_share_gap > 0.050
        if neighbor_core_is_meaningful and (core_gap_is_visible or gray_edge_is_substituting):
            issues.append(
                {
                    "type": "changed_char_neighbor_core_density_too_low",
                    "index": changed.get("index"),
                    "source_char": changed.get("source_char"),
                    "target_char": target_char,
                    "neighbor_index": neighbor.get("index"),
                    "neighbor_char": neighbor_char,
                    "target_complexity": target_complexity,
                    "neighbor_complexity": neighbor_complexity,
                    "changed_core_density": round(changed_core_density, 4),
                    "neighbor_core_density": round(neighbor_core_density, 4),
                    "core_density_gap": round(core_density_gap, 4),
                    "changed_lt70_density": round(changed_lt70_density, 4),
                    "neighbor_lt70_density": round(neighbor_lt70_density, 4),
                    "lt70_density_gap": round(lt70_density_gap, 4),
                    "changed_outer_share": round(changed_outer_share, 4),
                    "neighbor_outer_share": round(neighbor_outer_share, 4),
                    "outer_share_gap": round(outer_share_gap, 4),
                    "core_density_limit": 0.024,
                    "outer_share_limit": 0.045,
                }
            )
    return issues


def changed_texture_boxes(plan: RenderPlan) -> tuple[tuple[int, int, int, int], ...]:
    source_chars = text_chars(plan.source_text or "")
    target_chars = text_chars(plan.target_text)
    if (
        source_chars
        and target_chars
        and len(source_chars) == len(target_chars)
        and plan.slot_boxes
    ):
        slots = tuple(sorted(plan.slot_boxes, key=lambda item: item.x1))
        boxes: list[tuple[int, int, int, int]] = []
        for idx, (source_char, target_char) in enumerate(zip(source_chars, target_chars)):
            if source_char == target_char or idx >= len(slots):
                continue
            slot = slots[idx]
            pad_x = max(3, int(round((slot.x2 - slot.x1) * 0.18)))
            pad_y = max(2, int(round((slot.y2 - slot.y1) * 0.14)))
            boxes.append(
                clamp_box_to_container(
                    (slot.x1 - pad_x, slot.y1 - pad_y, slot.x2 + pad_x, slot.y2 + pad_y),
                    plan.target_roi,
                )
            )
        if boxes:
            return tuple(boxes)
    return (plan.target_roi,)


def photo_texture_metrics(
    original: Image.Image,
    candidate: Image.Image,
    plan: RenderPlan,
    params: CandidateParams,
) -> dict[str, Any]:
    original_gray = cv2.cvtColor(np.array(original.convert("RGB")), cv2.COLOR_RGB2GRAY)
    candidate_gray = cv2.cvtColor(np.array(candidate.convert("RGB")), cv2.COLOR_RGB2GRAY)
    old_lap = np.abs(cv2.Laplacian(original_gray, cv2.CV_32F, ksize=3))
    new_lap = np.abs(cv2.Laplacian(candidate_gray, cv2.CV_32F, ksize=3))
    old_residual = np.abs(original_gray.astype(np.float32) - cv2.GaussianBlur(original_gray, (0, 0), 1.2))
    new_residual = np.abs(candidate_gray.astype(np.float32) - cv2.GaussianBlur(candidate_gray, (0, 0), 1.2))

    per_box: list[dict[str, Any]] = []
    old_edge_values: list[float] = []
    new_edge_values: list[float] = []
    old_residual_values: list[float] = []
    new_residual_values: list[float] = []
    for box in changed_texture_boxes(plan):
        x1, y1, x2, y2 = clamp_box(box, original.size)
        if x2 <= x1 or y2 <= y1:
            continue
        old_crop = original_gray[y1:y2, x1:x2]
        new_crop = candidate_gray[y1:y2, x1:x2]
        old_mask = old_crop < 165
        new_mask = new_crop < 165
        if int(np.count_nonzero(old_mask)) < 12 or int(np.count_nonzero(new_mask)) < 12:
            continue
        kernel = np.ones((3, 3), dtype=np.uint8)
        old_edge = cv2.morphologyEx(old_mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
        new_edge = cv2.morphologyEx(new_mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
        old_near = cv2.dilate(old_mask.astype(np.uint8), kernel, iterations=1) > 0
        new_near = cv2.dilate(new_mask.astype(np.uint8), kernel, iterations=1) > 0
        if int(np.count_nonzero(old_edge)) < 6 or int(np.count_nonzero(new_edge)) < 6:
            continue
        old_edge_lap = old_lap[y1:y2, x1:x2][old_edge]
        new_edge_lap = new_lap[y1:y2, x1:x2][new_edge]
        old_res = old_residual[y1:y2, x1:x2][old_near]
        new_res = new_residual[y1:y2, x1:x2][new_near]
        old_edge_mean = float(np.mean(old_edge_lap))
        new_edge_mean = float(np.mean(new_edge_lap))
        old_res_mean = float(np.mean(old_res)) if old_res.size else 0.0
        new_res_mean = float(np.mean(new_res)) if new_res.size else 0.0
        old_edge_values.append(old_edge_mean)
        new_edge_values.append(new_edge_mean)
        old_residual_values.append(old_res_mean)
        new_residual_values.append(new_res_mean)
        per_box.append(
            {
                "box": [x1, y1, x2, y2],
                "old_edge_laplacian_mean": round(old_edge_mean, 3),
                "new_edge_laplacian_mean": round(new_edge_mean, 3),
                "edge_laplacian_ratio": round(new_edge_mean / max(1.0, old_edge_mean), 4),
                "old_residual_mean": round(old_res_mean, 3),
                "new_residual_mean": round(new_res_mean, 3),
                "residual_ratio": round(new_res_mean / max(1.0, old_res_mean), 4),
            }
        )

    if not per_box:
        return {"enabled": False, "reason": "not enough changed text texture pixels"}

    old_edge_mean = float(np.mean(old_edge_values))
    new_edge_mean = float(np.mean(new_edge_values))
    old_residual_mean = float(np.mean(old_residual_values))
    new_residual_mean = float(np.mean(new_residual_values))
    jpeg_quality = int(params.jpeg_quality or 0)
    jpeg_weight = 0.0
    if 1 <= jpeg_quality <= 99:
        jpeg_weight = max(0.0, min(0.42, (100.0 - float(jpeg_quality)) / 85.0))
    return {
        "enabled": True,
        "old_edge_laplacian_mean": round(old_edge_mean, 3),
        "new_edge_laplacian_mean": round(new_edge_mean, 3),
        "edge_laplacian_ratio": round(new_edge_mean / max(1.0, old_edge_mean), 4),
        "old_residual_mean": round(old_residual_mean, 3),
        "new_residual_mean": round(new_residual_mean, 3),
        "residual_ratio": round(new_residual_mean / max(1.0, old_residual_mean), 4),
        "params": {
            "blur": params.blur,
            "photo_warp": params.photo_warp,
            "edge_breakup": params.edge_breakup,
            "photo_noise": params.photo_noise,
            "jpeg_quality": params.jpeg_quality,
            "jpeg_weight": round(jpeg_weight, 4),
        },
        "per_box": per_box,
    }


def local_photo_texture_issues(report: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    metrics = report.get("photo_texture_metrics") if isinstance(report, dict) else None
    if not isinstance(metrics, dict) or not metrics.get("enabled"):
        return issues
    params = metrics.get("params") if isinstance(metrics.get("params"), dict) else {}
    try:
        blur = float(params.get("blur") or 0.0)
        photo_warp = float(params.get("photo_warp") or 0.0)
        edge_breakup = float(params.get("edge_breakup") or 0.0)
        photo_noise = float(params.get("photo_noise") or 0.0)
        jpeg_weight = float(params.get("jpeg_weight") or 0.0)
        edge_ratio = float(metrics.get("edge_laplacian_ratio") or 1.0)
        residual_ratio = float(metrics.get("residual_ratio") or 1.0)
        old_residual_mean = float(metrics.get("old_residual_mean") or 0.0)
        old_edge_mean = float(metrics.get("old_edge_laplacian_mean") or 0.0)
    except (TypeError, ValueError):
        return issues

    texture_strength = photo_warp + edge_breakup * 3.0 + photo_noise * 2.0 + jpeg_weight
    source_photo_like = old_residual_mean >= 2.4 or old_edge_mean >= 28.0
    if source_photo_like and blur <= 0.04 and texture_strength < 0.018:
        issues.append(
            {
                "type": "photo_texture_not_applied",
                "blur": round(blur, 3),
                "texture_strength": round(texture_strength, 4),
                "old_residual_mean": round(old_residual_mean, 3),
                "old_edge_laplacian_mean": round(old_edge_mean, 3),
            }
        )
    if source_photo_like and edge_ratio > 1.85 and blur < 0.22:
        issues.append(
            {
                "type": "photo_texture_too_sharp",
                "edge_laplacian_ratio": round(edge_ratio, 4),
                "blur": round(blur, 3),
            }
        )
    if source_photo_like and residual_ratio < 0.42 and photo_noise < 0.006 and jpeg_weight < 0.02:
        issues.append(
            {
                "type": "photo_texture_too_clean",
                "residual_ratio": round(residual_ratio, 4),
                "photo_noise": round(photo_noise, 3),
                "jpeg_weight": round(jpeg_weight, 4),
            }
        )
    if edge_ratio < 0.18 and blur > 0.70:
        issues.append(
            {
                "type": "photo_texture_too_blurry",
                "edge_laplacian_ratio": round(edge_ratio, 4),
                "blur": round(blur, 3),
            }
        )
    return issues


def background_texture_metrics(
    original: Image.Image,
    candidate: Image.Image,
    plan: RenderPlan,
    params: CandidateParams,
) -> dict[str, Any]:
    original_arr = np.array(original.convert("RGB"))
    candidate_arr = np.array(candidate.convert("RGB"))
    original_gray = cv2.cvtColor(original_arr, cv2.COLOR_RGB2GRAY)
    candidate_gray = cv2.cvtColor(candidate_arr, cv2.COLOR_RGB2GRAY)
    h, w = original_gray.shape
    tx1, ty1, tx2, ty2 = plan.target_roi
    if tx2 <= tx1 or ty2 <= ty1:
        return {"enabled": False, "reason": "empty target roi"}

    old_dark = (original_gray[ty1:ty2, tx1:tx2] < int(params.mask_threshold)).astype(np.uint8)
    if int(np.count_nonzero(old_dark)) < 8:
        return {"enabled": False, "reason": "not enough old text mask pixels"}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    old_dark_base = old_dark > 0
    old_dark = cv2.dilate(old_dark, kernel, iterations=max(1, int(params.mask_dilate_iterations)))
    replacement_layer = draw_replacement_layer(size=original.size, plan=plan, params=params, original=original)
    alpha = np.array(replacement_layer.getchannel("A"))
    new_alpha = alpha[ty1:ty2, tx1:tx2] > 18
    trailing_cleanup_mask = build_trailing_value_cleanup_mask(plan, replacement_layer, original.size)
    trailing_cleanup_crop = trailing_cleanup_mask[ty1:ty2, tx1:tx2] > 0
    fill_mask = (old_dark > 0) & ~new_alpha
    if int(np.count_nonzero(fill_mask)) < 10:
        fill_mask = old_dark > 0
    if int(np.count_nonzero(fill_mask)) < 10:
        return {"enabled": False, "reason": "not enough fill background pixels"}

    pad_x = max(10, int(round((tx2 - tx1) * 0.35)))
    pad_y = max(6, int(round((ty2 - ty1) * 0.60)))
    rx1, ry1, rx2, ry2 = clamp_box((tx1 - pad_x, ty1 - pad_y, tx2 + pad_x, ty2 + pad_y), original.size)
    reference_mask = np.ones((ry2 - ry1, rx2 - rx1), dtype=bool)
    reference_mask[ty1 - ry1 : ty2 - ry1, tx1 - rx1 : tx2 - rx1] = False
    reference_crop = original_gray[ry1:ry2, rx1:rx2]
    reference_mask &= reference_crop >= 165
    if int(np.count_nonzero(reference_mask)) < 24:
        reference_mask = original_gray[ry1:ry2, rx1:rx2] >= 150
    if int(np.count_nonzero(reference_mask)) < 24:
        return {"enabled": False, "reason": "not enough same-row background reference"}

    old_fill = original_gray[ty1:ty2, tx1:tx2][fill_mask]
    new_fill = candidate_gray[ty1:ty2, tx1:tx2][fill_mask]
    reference_values = reference_crop[reference_mask]

    def residual_values(gray: np.ndarray) -> np.ndarray:
        residual = np.abs(gray.astype(np.float32) - cv2.GaussianBlur(gray, (0, 0), 1.2))
        return residual

    old_residual = residual_values(original_gray[ty1:ty2, tx1:tx2])[fill_mask]
    new_residual = residual_values(candidate_gray[ty1:ty2, tx1:tx2])[fill_mask]
    reference_residual = residual_values(reference_crop)[reference_mask]
    old_mean = float(np.mean(old_fill))
    new_mean = float(np.mean(new_fill))
    reference_mean = float(np.mean(reference_values))
    old_std = float(np.std(old_fill))
    new_std = float(np.std(new_fill))
    reference_std = float(np.std(reference_values))
    old_residual_mean = float(np.mean(old_residual)) if old_residual.size else 0.0
    new_residual_mean = float(np.mean(new_residual)) if new_residual.size else 0.0
    reference_residual_mean = float(np.mean(reference_residual)) if reference_residual.size else 0.0

    trailing_metrics: dict[str, Any] = {
        "enabled": int(np.count_nonzero(trailing_cleanup_crop)) >= 24,
        "pixels": int(np.count_nonzero(trailing_cleanup_crop)),
    }
    if trailing_metrics["enabled"]:
        trailing_values = candidate_gray[ty1:ty2, tx1:tx2][trailing_cleanup_crop]
        trailing_residual = residual_values(candidate_gray[ty1:ty2, tx1:tx2])[trailing_cleanup_crop]
        trailing_std = float(np.std(trailing_values))
        trailing_residual_mean = float(np.mean(trailing_residual)) if trailing_residual.size else 0.0
        trailing_metrics.update(
            {
                "mean_gray": round(float(np.mean(trailing_values)), 3),
                "reference_mean_gray": round(reference_mean, 3),
                "reference_mean_delta": round(float(np.mean(trailing_values) - reference_mean), 3),
                "std_gray": round(trailing_std, 3),
                "reference_std_gray": round(reference_std, 3),
                "std_ratio": round(trailing_std / max(0.8, reference_std), 4),
                "residual_mean": round(trailing_residual_mean, 3),
                "reference_residual_mean": round(reference_residual_mean, 3),
                "residual_ratio": round(trailing_residual_mean / max(0.8, reference_residual_mean), 4),
            }
        )

    ghost_probe = cv2.dilate(
        old_dark_base.astype(np.uint8),
        kernel,
        iterations=max(3, int(params.mask_dilate_iterations) + 1),
    ) > 0
    ghost_probe &= ~new_alpha
    local_background_mask = (~ghost_probe) & ~new_alpha & (original_gray[ty1:ty2, tx1:tx2] >= 165)
    ghost_pixels = int(np.count_nonzero(ghost_probe))
    local_background_pixels = int(np.count_nonzero(local_background_mask))
    ghost_metrics: dict[str, Any] = {
        "enabled": ghost_pixels >= 24 and local_background_pixels >= 24,
        "probe_pixels": ghost_pixels,
        "local_background_pixels": local_background_pixels,
    }
    if ghost_metrics["enabled"]:
        ghost_values = candidate_gray[ty1:ty2, tx1:tx2][ghost_probe]
        local_background_values = candidate_gray[ty1:ty2, tx1:tx2][local_background_mask]
        bg_p95 = float(np.percentile(local_background_values, 95))
        bg_p99 = float(np.percentile(local_background_values, 99))
        bg_p05 = float(np.percentile(local_background_values, 5))
        bg_p10 = float(np.percentile(local_background_values, 10))
        bg_p25 = float(np.percentile(local_background_values, 25))
        ghost_p05 = float(np.percentile(ghost_values, 5))
        ghost_p10 = float(np.percentile(ghost_values, 10))
        ghost_p25 = float(np.percentile(ghost_values, 25))
        ghost_p95 = float(np.percentile(ghost_values, 95))
        ghost_p99 = float(np.percentile(ghost_values, 99))
        ghost_metrics.update(
            {
                "probe_mean_gray": round(float(np.mean(ghost_values)), 3),
                "local_background_mean_gray": round(float(np.mean(local_background_values)), 3),
                "probe_background_mean_delta": round(
                    float(np.mean(ghost_values) - np.mean(local_background_values)),
                    3,
                ),
                "probe_p05_gray": round(ghost_p05, 3),
                "local_background_p05_gray": round(bg_p05, 3),
                "probe_p05_delta": round(ghost_p05 - bg_p05, 3),
                "probe_p10_gray": round(ghost_p10, 3),
                "local_background_p10_gray": round(bg_p10, 3),
                "probe_p10_delta": round(ghost_p10 - bg_p10, 3),
                "probe_p25_gray": round(ghost_p25, 3),
                "local_background_p25_gray": round(bg_p25, 3),
                "probe_p25_delta": round(ghost_p25 - bg_p25, 3),
                "probe_p95_gray": round(ghost_p95, 3),
                "local_background_p95_gray": round(bg_p95, 3),
                "probe_p95_delta": round(ghost_p95 - bg_p95, 3),
                "probe_p99_gray": round(ghost_p99, 3),
                "local_background_p99_gray": round(bg_p99, 3),
                "probe_p99_delta": round(ghost_p99 - bg_p99, 3),
                "bright_over_background_p95_ratio": round(
                    float(np.mean(ghost_values > bg_p95)),
                    5,
                ),
                "bright_over_background_p99_ratio": round(
                    float(np.mean(ghost_values > bg_p99)),
                    5,
                ),
                "dark_under_background_p05_ratio": round(
                    float(np.mean(ghost_values < bg_p05)),
                    5,
                ),
                "dark_under_background_p10_ratio": round(
                    float(np.mean(ghost_values < bg_p10)),
                    5,
                ),
                "dark_under_background_p25_ratio": round(
                    float(np.mean(ghost_values < bg_p25)),
                    5,
                ),
            }
        )
    return {
        "enabled": True,
        "target_roi": [tx1, ty1, tx2, ty2],
        "fill_pixels": int(np.count_nonzero(fill_mask)),
        "reference_pixels": int(np.count_nonzero(reference_mask)),
        "old_fill_mean_gray": round(old_mean, 3),
        "new_fill_mean_gray": round(new_mean, 3),
        "reference_mean_gray": round(reference_mean, 3),
        "new_reference_mean_delta": round(new_mean - reference_mean, 3),
        "old_reference_mean_delta": round(old_mean - reference_mean, 3),
        "old_fill_std_gray": round(old_std, 3),
        "new_fill_std_gray": round(new_std, 3),
        "reference_std_gray": round(reference_std, 3),
        "std_ratio": round(new_std / max(0.8, reference_std), 4),
        "old_residual_mean": round(old_residual_mean, 3),
        "new_residual_mean": round(new_residual_mean, 3),
        "reference_residual_mean": round(reference_residual_mean, 3),
        "residual_ratio": round(new_residual_mean / max(0.8, reference_residual_mean), 4),
        "white_ghost_probe": ghost_metrics,
        "trailing_cleanup_patch": trailing_metrics,
    }


def local_background_texture_issues(report: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = report.get("background_texture_metrics") if isinstance(report, dict) else None
    if not isinstance(metrics, dict) or not metrics.get("enabled"):
        return []
    issues: list[dict[str, Any]] = []
    try:
        mean_delta = float(metrics.get("new_reference_mean_delta") or 0.0)
        old_mean_delta = float(metrics.get("old_reference_mean_delta") or 0.0)
        std_ratio = float(metrics.get("std_ratio") or 1.0)
        residual_ratio = float(metrics.get("residual_ratio") or 1.0)
        reference_residual = float(metrics.get("reference_residual_mean") or 0.0)
    except (TypeError, ValueError):
        return issues
    ghost_probe = metrics.get("white_ghost_probe")
    if isinstance(ghost_probe, dict) and ghost_probe.get("enabled"):
        try:
            bright_p95_ratio = float(ghost_probe.get("bright_over_background_p95_ratio") or 0.0)
            bright_p99_ratio = float(ghost_probe.get("bright_over_background_p99_ratio") or 0.0)
            dark_p10_ratio = float(ghost_probe.get("dark_under_background_p10_ratio") or 0.0)
            dark_p25_ratio = float(ghost_probe.get("dark_under_background_p25_ratio") or 0.0)
            p10_delta = float(ghost_probe.get("probe_p10_delta") or 0.0)
            p25_delta = float(ghost_probe.get("probe_p25_delta") or 0.0)
            p95_delta = float(ghost_probe.get("probe_p95_delta") or 0.0)
            p99_delta = float(ghost_probe.get("probe_p99_delta") or 0.0)
            mean_delta_local = float(ghost_probe.get("probe_background_mean_delta") or 0.0)
            probe_pixels = int(ghost_probe.get("probe_pixels") or 0)
        except (TypeError, ValueError):
            bright_p95_ratio = 0.0
            bright_p99_ratio = 0.0
            dark_p10_ratio = 0.0
            dark_p25_ratio = 0.0
            p10_delta = 0.0
            p25_delta = 0.0
            p95_delta = 0.0
            p99_delta = 0.0
            mean_delta_local = 0.0
            probe_pixels = 0
        if probe_pixels >= 120 and (
            (bright_p95_ratio > 0.14 and p95_delta > 6.0)
            or (bright_p99_ratio > 0.055 and p99_delta > 12.0)
            or (mean_delta_local > 2.6 and bright_p95_ratio > 0.10)
        ):
            issues.append(
                {
                    "type": "background_white_ghost_residual",
                    "bright_over_background_p95_ratio": round(bright_p95_ratio, 5),
                    "p95_delta": round(p95_delta, 3),
                    "bright_over_background_p99_ratio": round(bright_p99_ratio, 5),
                    "p99_delta": round(p99_delta, 3),
                    "mean_delta": round(mean_delta_local, 3),
                    "probe_pixels": probe_pixels,
                }
            )
        if probe_pixels >= 120 and (
            (mean_delta_local < -3.0 and dark_p10_ratio > 0.28)
            or (p10_delta < -5.0 and dark_p25_ratio > 0.45)
            or (p25_delta < -3.5 and dark_p25_ratio > 0.55)
        ):
            issues.append(
                {
                    "type": "background_shadow_ghost_residual",
                    "dark_under_background_p10_ratio": round(dark_p10_ratio, 5),
                    "p10_delta": round(p10_delta, 3),
                    "dark_under_background_p25_ratio": round(dark_p25_ratio, 5),
                    "p25_delta": round(p25_delta, 3),
                    "mean_delta": round(mean_delta_local, 3),
                    "probe_pixels": probe_pixels,
                }
            )
    trailing_patch = metrics.get("trailing_cleanup_patch")
    if isinstance(trailing_patch, dict) and trailing_patch.get("enabled"):
        try:
            trailing_std_ratio = float(trailing_patch.get("std_ratio") or 1.0)
            trailing_residual_ratio = float(trailing_patch.get("residual_ratio") or 1.0)
            trailing_mean_delta = float(trailing_patch.get("reference_mean_delta") or 0.0)
            trailing_pixels = int(trailing_patch.get("pixels") or 0)
        except (TypeError, ValueError):
            trailing_std_ratio = 1.0
            trailing_residual_ratio = 1.0
            trailing_mean_delta = 0.0
            trailing_pixels = 0
        if trailing_pixels >= 120 and (trailing_std_ratio < 0.36 or trailing_residual_ratio < 0.42):
            issues.append(
                {
                    "type": "background_trailing_patch_too_smooth",
                    "std_ratio": round(trailing_std_ratio, 4),
                    "residual_ratio": round(trailing_residual_ratio, 4),
                    "mean_delta": round(trailing_mean_delta, 3),
                    "pixels": trailing_pixels,
                }
            )
    if abs(mean_delta) > max(12.0, abs(old_mean_delta) + 7.0):
        issues.append(
            {
                "type": "background_fill_luminance_mismatch",
                "new_reference_mean_delta": round(mean_delta, 3),
                "old_reference_mean_delta": round(old_mean_delta, 3),
                "limit": round(max(12.0, abs(old_mean_delta) + 7.0), 3),
            }
        )
    if reference_residual >= 1.8 and residual_ratio < 0.42:
        issues.append(
            {
                "type": "background_fill_too_smooth",
                "residual_ratio": round(residual_ratio, 4),
                "reference_residual_mean": round(reference_residual, 3),
                "limit": 0.42,
            }
        )
    texture_variance_limit = 0.62 if reference_residual >= 2.4 else 0.48
    structured_ghost = any(
        issue.get("type") in {"background_white_ghost_residual", "background_shadow_ghost_residual"}
        for issue in issues
    )
    if std_ratio < texture_variance_limit and residual_ratio < 1.05 and not structured_ghost:
        issues.append(
            {
                "type": "background_fill_low_texture_variance",
                "std_ratio": round(std_ratio, 4),
                "limit": round(texture_variance_limit, 3),
            }
        )
    return issues


def strict_gate_stage_issues(report: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    stages = {
        "text_shape": [],
        "ink_gray_balance": [],
        "background_cleanup": [],
    }
    strict_gate = report.get("strict_gate")
    if not isinstance(strict_gate, dict):
        return stages

    for issue in strict_gate.get("issues", []):
        if not isinstance(issue, dict):
            continue
        issue_type = str(issue.get("type") or "")
        if (
            issue_type.startswith("char_")
            or issue_type.startswith("replacement_")
            or issue_type.startswith("font_")
            or issue_type.startswith("bbox_")
            or issue_type.startswith("centroid_")
            or issue_type.startswith("ink_area_ratio_")
            or "font_" in issue_type
        ):
            stages["text_shape"].append(issue)
        elif issue_type.startswith("extra_source_slot_"):
            stages["background_cleanup"].append(issue)
        else:
            stages["ink_gray_balance"].append(issue)
    return stages


def stage_gate_for_report(report: dict[str, Any], profile: str | None = None) -> dict[str, Any]:
    return ordered_stage_gate_for_report(report, profile or str(report.get("pipeline_profile") or "photo_scan"))


def stage_issues(report: dict[str, Any] | None, stage_id: str) -> list[dict[str, Any]]:
    if not isinstance(report, dict):
        return []
    stage_gate = report.get("stage_gate")
    if not isinstance(stage_gate, dict):
        stage_gate = stage_gate_for_report(report)
    for stage in stage_gate.get("stages", []):
        if isinstance(stage, dict) and stage.get("id") == stage_id:
            return [issue for issue in stage.get("issues", []) if isinstance(issue, dict)]
    return []


def stage_selection_penalty(report: dict[str, Any] | None) -> float:
    if not isinstance(report, dict):
        return 0.0
    stage_gate = report.get("stage_gate")
    if not isinstance(stage_gate, dict):
        stage_gate = stage_gate_for_report(report)
    blocking_stage = str(stage_gate.get("blocking_stage") or "")
    if not blocking_stage:
        return 0.0

    penalties = {
        "hard_boundary": 600000.0,
        "text_shape": 9500.0,
        "ink_gray_balance": 2600.0,
        "photo_texture": 1200.0,
        "background_cleanup": 1800.0,
    }
    penalty = penalties.get(blocking_stage, 1000.0)
    issues = stage_issues(report, blocking_stage)
    penalty += len(issues) * (850.0 if blocking_stage == "text_shape" else 260.0)
    if blocking_stage == "text_shape":
        penalty += gray_stroke_balance_penalty(report) * 4.0
        if local_pose_issues(report):
            penalty += 900.0
    return penalty


def ink_stage_issue_severity(report: dict[str, Any] | None) -> float:
    severity = 0.0
    for issue in stage_issues(report, "ink_gray_balance"):
        issue_type = str(issue.get("type") or "")
        try:
            if issue_type == "dark_pixel_ratio_too_high":
                severity += max(0.0, float(issue.get("actual") or 0.0) - float(issue.get("limit") or 0.0)) * 1800.0
            elif issue_type in {"changed_char_core_too_black", "changed_char_core_too_black_hard"}:
                severity += max(0.0, float(issue.get("lt55_delta") or 0.0) - float(issue.get("limit") or 0.0)) * 2.2
            elif issue_type == "roi_core_too_black":
                severity += max(0.0, float(issue.get("lt55_delta") or 0.0) - float(issue.get("limit") or 0.0)) * 2.8
            elif issue_type == "roi_black_core_share_too_high":
                severity += max(0.0, float(issue.get("share_delta") or 0.0) - float(issue.get("limit") or 0.0)) * 2400.0
            elif issue_type == "core_mean_gray_too_light":
                severity += max(0.0, float(issue.get("actual") or 0.0) - float(issue.get("limit") or 0.0)) * 32.0
            elif issue_type == "core_lighten_too_high":
                severity += max(0.0, float(issue.get("actual") or 0.0) - float(issue.get("limit") or 0.0)) * 26.0
            else:
                severity += 80.0
        except (TypeError, ValueError):
            severity += 80.0
    return severity


def stage_issue_severity(report: dict[str, Any] | None, stage_id: str | None) -> float:
    if not stage_id:
        return 0.0
    if stage_id == "ink_gray_balance":
        return ink_stage_issue_severity(report)
    if stage_id == "background_cleanup":
        severity = 0.0
        for issue in stage_issues(report, "background_cleanup"):
            issue_type = str(issue.get("type") or "")
            try:
                if issue_type == "background_white_ghost_residual":
                    severity += max(0.0, float(issue.get("bright_over_background_p95_ratio") or 0.0) - 0.08) * 900.0
                    severity += max(0.0, float(issue.get("p95_delta") or 0.0) - 4.0) * 16.0
                    severity += max(0.0, float(issue.get("p99_delta") or 0.0) - 8.0) * 8.0
                elif issue_type == "background_shadow_ghost_residual":
                    severity += max(0.0, float(issue.get("dark_under_background_p10_ratio") or 0.0) - 0.20) * 720.0
                    severity += max(0.0, abs(float(issue.get("p10_delta") or 0.0)) - 3.0) * 14.0
                    severity += max(0.0, abs(float(issue.get("mean_delta") or 0.0)) - 2.0) * 18.0
                elif issue_type == "background_trailing_patch_too_smooth":
                    severity += max(0.0, 0.36 - float(issue.get("std_ratio") or 0.0)) * 620.0
                    severity += max(0.0, 0.42 - float(issue.get("residual_ratio") or 0.0)) * 760.0
                elif issue_type == "background_fill_luminance_mismatch":
                    severity += max(0.0, abs(float(issue.get("new_reference_mean_delta") or 0.0)) - float(issue.get("limit") or 0.0)) * 35.0
                elif issue_type == "background_fill_too_smooth":
                    severity += max(0.0, float(issue.get("limit") or 0.0) - float(issue.get("residual_ratio") or 0.0)) * 520.0
                elif issue_type == "background_fill_low_texture_variance":
                    severity += max(0.0, float(issue.get("limit") or 0.0) - float(issue.get("std_ratio") or 0.0)) * 420.0
                else:
                    severity += 90.0
            except (TypeError, ValueError):
                severity += 90.0
        return severity
    if stage_id == "photo_texture":
        severity = 0.0
        for issue in stage_issues(report, "photo_texture"):
            issue_type = str(issue.get("type") or "")
            try:
                if issue_type == "photo_texture_too_sharp":
                    severity += max(0.0, float(issue.get("edge_laplacian_ratio") or 0.0) - 1.0) * 160.0
                elif issue_type == "photo_texture_too_clean":
                    severity += max(0.0, 0.42 - float(issue.get("residual_ratio") or 0.0)) * 420.0
                elif issue_type == "photo_texture_too_blurry":
                    severity += max(0.0, 0.18 - float(issue.get("edge_laplacian_ratio") or 0.0)) * 520.0
                else:
                    severity += 90.0
            except (TypeError, ValueError):
                severity += 90.0
        return severity
    return float(len(stage_issues(report, stage_id))) * 100.0


def params_delta(before: CandidateParams, after: CandidateParams) -> dict[str, Any]:
    changes: dict[str, Any] = {}
    for name in before.__dataclass_fields__:
        if name == "candidate_id":
            continue
        old = getattr(before, name)
        new = getattr(after, name)
        if old == new:
            continue
        if isinstance(old, float) or isinstance(new, float):
            old_out = round(float(old), 4)
            new_out = round(float(new), 4)
        else:
            old_out = old
            new_out = new
        changes[name] = {"from": old_out, "to": new_out}
    return changes


def constraint_reason(
    report: dict[str, Any] | None,
    acceptance: dict[str, Any],
) -> str:
    stage = (stage_gate_for_report(report).get("blocking_stage") if isinstance(report, dict) else None)
    if stage == "text_shape":
        return "text_shape_stage_caps_ink_and_photo_side_effects"
    if report_has_excess_black_core(report):
        return "dynamic_reference_profile_caps_true_black_core"
    if report_has_background_white_ghost(report):
        return "background_white_or_shadow_ghost_cleanup_caps"
    if report_has_background_low_texture(report):
        return "background_low_texture_recovery_caps"
    if acceptance_reports_background_patch(acceptance):
        return "vision_background_patch_feedback_caps"
    if report_needs_wider_gray_strokes(report):
        return "stroke_body_recovery_caps"
    if report_needs_thinner_strokes(report):
        return "thin_or_dark_core_recovery_caps"
    return "no_local_constraint"


def constraint_audit(
    raw_params: CandidateParams,
    constrained_params: CandidateParams,
    report: dict[str, Any] | None,
    acceptance: dict[str, Any],
) -> dict[str, Any]:
    changes = params_delta(raw_params, constrained_params)
    return {
        "applied": bool(changes),
        "reason": constraint_reason(report, acceptance) if changes else "none",
        "changes": changes,
    }


def alignment_vertical_penalty(report: dict[str, Any] | None) -> float:
    if not isinstance(report, dict):
        return 0.0
    alignment = report.get("char_alignment_metrics")
    if not isinstance(alignment, dict) or not alignment.get("enabled"):
        return 0.0
    penalty = 0.0
    for item in alignment.get("per_char", []):
        if not isinstance(item, dict) or not item.get("candidate_box"):
            continue
        try:
            center_dy = float(item.get("center_dy") or 0.0)
        except (TypeError, ValueError):
            continue
        penalty += abs(center_dy) * 32.0
        penalty += max(0.0, -center_dy) * 72.0
        penalty += max(0.0, center_dy - 0.75) * 96.0
    return penalty


def report_stage_pass(report: dict[str, Any]) -> bool:
    stage_gate = report.get("stage_gate")
    if not isinstance(stage_gate, dict):
        stage_gate = stage_gate_for_report(report)
    return bool(stage_gate.get("pass"))


def apply_local_acceptance_gate(acceptance: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    stage_gate = stage_gate_for_report(report)
    shape_stage_issues = stage_issues({"stage_gate": stage_gate}, "text_shape")
    ink_issues = local_ink_balance_issues(report)
    shape_blocking = stage_gate.get("blocking_stage") == "text_shape"
    neighbor_issues = local_neighbor_style_issues(report, allow_excess_black_core=True)
    if ink_issues and neighbor_issues:
        ink_issues = [
            issue
            for issue in ink_issues
            if str(issue.get("type") or "")
            not in {
                "changed_char_core_too_black_hard",
                "changed_char_core_too_black",
                "roi_core_too_black",
                "roi_black_core_share_too_high",
            }
        ]
    body_issues = local_stroke_body_issues(
        report,
        allow_excess_black_core=shape_blocking,
    )
    if ink_issues and not shape_blocking:
        body_issues = []
    pose_issues = [] if ink_issues and not shape_blocking else local_pose_issues(report)
    photo_issues = stage_issues({"stage_gate": stage_gate}, "photo_texture")
    background_issues = stage_issues({"stage_gate": stage_gate}, "background_cleanup")
    uncovered_shape_issue_count = max(
        0,
        len(shape_stage_issues) - len(body_issues) - len(neighbor_issues) - len(pose_issues),
    )
    if (
        stage_gate.get("pass")
        and not ink_issues
        and not body_issues
        and not neighbor_issues
        and not pose_issues
        and not uncovered_shape_issue_count
        and not photo_issues
        and not background_issues
    ):
        return acceptance

    gated = dict(acceptance or {})
    gated["stage_gate"] = stage_gate
    findings = gated.get("visual_findings")
    findings = dict(findings) if isinstance(findings, dict) else {}
    if ink_issues and not shape_blocking:
        findings.setdefault("stroke_weight", "too_bold")
        findings.setdefault("darkness", "too_dark")
        findings.setdefault("sharpness", "too_sharp")
    elif body_issues or neighbor_issues:
        findings["stroke_weight"] = "too_thin"
        findings.setdefault("darkness", "ok")
        findings.setdefault("sharpness", "too_sharp")
    else:
        findings.setdefault("font_similarity", "slightly_off")
        findings.setdefault("baseline", "slightly_off")
    gated["visual_findings"] = findings
    gated["pass"] = False
    gated["acceptance_level"] = "marginal"
    gated["final_decision"] = "revise"
    if ink_issues:
        gated["local_ink_balance_issues"] = ink_issues
    if body_issues:
        gated["local_stroke_body_issues"] = body_issues
    if neighbor_issues:
        gated["local_neighbor_style_issues"] = neighbor_issues
    if pose_issues:
        gated["local_pose_issues"] = pose_issues
    if photo_issues:
        gated["local_photo_texture_issues"] = photo_issues
    if background_issues:
        gated["local_background_cleanup_issues"] = background_issues
    if shape_stage_issues:
        gated["local_shape_stage_issues"] = shape_stage_issues
    reason = str(gated.get("reason") or "").strip()
    outer_neighbor_issues = [
        issue
        for issue in neighbor_issues
        if isinstance(issue, dict) and issue.get("type") == "changed_char_neighbor_outer_gray_halo_too_high"
    ]
    core_neighbor_issues = [
        issue
        for issue in neighbor_issues
        if isinstance(issue, dict) and issue.get("type") == "changed_char_neighbor_core_density_too_low"
    ]
    if shape_blocking and body_issues and neighbor_issues:
        local_reason = "本地形态阶段发现新字笔画体量和同一行邻字风格仍未过关；必须先修字体/字号/笔画身体/姿态，再进入黑度和照片质感阶段。"
    elif shape_blocking and body_issues:
        local_reason = "本地形态阶段发现新字笔画身体仍偏窄或中间灰阶不足；必须先修粗细和字形，再进入黑度阶段。"
    elif shape_blocking and neighbor_issues:
        local_reason = "本地形态阶段发现新字相对同一行保留字核心密度或外灰边不一致；必须先修邻字风格匹配。"
    elif ink_issues:
        local_reason = "本地硬指标发现新字深黑核心过量或灰阶过渡不足，需要继续降低核心黑块并补照片质感。"
    elif photo_issues:
        local_reason = "本地照片质感阶段发现新字与原图的拍照模糊、边缘断裂、噪声或压缩质感不一致，需要先补齐照片质感再交付。"
    elif background_issues:
        local_reason = "本地背景阶段发现旧字清理或底色纹理仍不自然，需要继续修背景。"
    elif body_issues and outer_neighbor_issues and not core_neighbor_issues:
        local_reason = "本地硬指标发现新字外层浅灰边相对同一行保留字偏多，底部/外缘有灰雾感；需要压掉外圈灰边，同时保留核心笔画。"
    elif body_issues and neighbor_issues:
        local_reason = "本地硬指标发现新字笔画体量不足，且相对同一行保留字核心暗部密度偏低、外层灰边偏多；需要让细笔画更实，同时避免继续用灰雾撑厚。"
    elif body_issues:
        local_reason = "本地硬指标发现新字笔画体量不足，中间灰阶覆盖偏少；需要加宽笔画身体和扫描灰边，而不是单纯压黑核心。"
    elif outer_neighbor_issues and not core_neighbor_issues:
        local_reason = "本地邻字风格指标发现新字相对同一行保留字外层浅灰边偏多，需要清理外圈灰雾而不是继续加模糊。"
    elif neighbor_issues:
        local_reason = "本地邻字风格指标发现新字相对同一行保留字核心暗部密度偏低、外层灰边偏多，需要让细笔画更实。"
    elif uncovered_shape_issue_count:
        local_reason = "本地形态阶段发现字体、字号、字槽、基线或笔画形态仍未过关，必须先修形态，再处理黑度和照片质感。"
    else:
        local_reason = "本地姿态指标发现新字倾斜继承不足，需要更贴近旧槽位的拍照倾斜。"
    gated["reason"] = f"{reason} {local_reason}".strip()
    must_fix = gated.get("must_fix")
    if not isinstance(must_fix, list):
        must_fix = []
    if ink_issues and not shape_blocking:
        must_fix.append("Reduce excessive <55 core pixels and recover mid-gray scanned edges before accepting.")
    elif body_issues:
        must_fix.append("Increase stroke body and mid-gray coverage while keeping <55 core pixels bounded.")
    if neighbor_issues:
        if outer_neighbor_issues and not core_neighbor_issues:
            must_fix.append("Reduce outer 120-165 gray halo around changed characters while preserving the dark core.")
        else:
            must_fix.append("Match changed character core density to same-row preserved neighbors without adding gray haze.")
    if photo_issues:
        must_fix.append("Match scan/photo texture after shape and ink pass: blur, edge breakup, noise, and compression must be locally consistent.")
    if background_issues:
        must_fix.append("Resolve background cleanup before accepting the final image.")
    if pose_issues and not ink_issues and not body_issues and not neighbor_issues:
        must_fix.append("Increase local slot shear inheritance for changed characters before accepting.")
    if uncovered_shape_issue_count:
        must_fix.append("Resolve text_shape stage before tuning ink, blur, noise, or background texture.")
    gated["must_fix"] = must_fix
    return gated


def report_has_background_white_ghost(report: dict[str, Any] | None) -> bool:
    return any(
        str(issue.get("type") or "") in {"background_white_ghost_residual", "background_shadow_ghost_residual"}
        for issue in stage_issues(report, "background_cleanup")
        if isinstance(issue, dict)
    )


def report_has_background_low_texture(report: dict[str, Any] | None) -> bool:
    return any(
        str(issue.get("type") or "") in {
            "background_fill_too_smooth",
            "background_fill_low_texture_variance",
            "background_trailing_patch_too_smooth",
        }
        for issue in stage_issues(report, "background_cleanup")
        if isinstance(issue, dict)
    )

def report_needs_thinner_strokes(report: dict[str, Any] | None) -> bool:
    if not isinstance(report, dict):
        return False
    char_bands = report.get("char_gray_band_metrics")
    if isinstance(char_bands, dict) and char_bands.get("enabled"):
        for item in char_bands.get("per_char", []):
            if not isinstance(item, dict):
                continue
            source_char = item.get("source_char")
            target_char = item.get("target_char")
            if source_char == target_char:
                continue
            delta = item.get("delta") or {}
            old = item.get("old") or {}
            try:
                lt165_delta = float(delta.get("lt165") or 0.0)
                lt55_delta = float(delta.get("lt55") or 0.0)
                old_lt165 = float(old.get("lt165") or 0.0)
            except (TypeError, ValueError):
                continue
            if old_lt165 > 0 and lt165_delta > max(28.0, old_lt165 * 0.12) and lt55_delta > 70.0:
                return True
    return False


def gray_stroke_balance_penalty(report: dict[str, Any] | None) -> float:
    if not isinstance(report, dict):
        return 0.0
    char_bands = report.get("char_gray_band_metrics")
    if not isinstance(char_bands, dict) or not char_bands.get("enabled"):
        return 0.0

    penalty = 0.0
    for item in char_bands.get("per_char", []):
        if not isinstance(item, dict):
            continue
        if item.get("source_char") == item.get("target_char"):
            continue
        delta = item.get("delta") or {}
        try:
            lt55_delta = float(delta.get("lt55") or 0.0)
            lt165_delta = float(delta.get("lt165") or 0.0)
            band_70_90_delta = float(delta.get("band_70_90") or 0.0)
            band_90_120_delta = float(delta.get("band_90_120") or 0.0)
        except (TypeError, ValueError):
            continue
        penalty += max(0.0, -lt165_delta) * 2.2
        penalty += max(0.0, -band_70_90_delta) * 1.4
        penalty += max(0.0, -band_90_120_delta) * 1.0
        penalty += max(0.0, lt55_delta - 45.0) * 0.8
    return penalty


def report_needs_wider_gray_strokes(report: dict[str, Any] | None) -> bool:
    if not isinstance(report, dict):
        return False
    if report_has_excess_black_core(report):
        return False
    if local_stroke_body_issues(report):
        return True
    if local_neighbor_style_issues(report):
        return True
    char_bands = report.get("char_gray_band_metrics")
    if not isinstance(char_bands, dict) or not char_bands.get("enabled"):
        return False

    for item in char_bands.get("per_char", []):
        if not isinstance(item, dict):
            continue
        if item.get("source_char") == item.get("target_char"):
            continue
        delta = item.get("delta") or {}
        try:
            lt55_delta = float(delta.get("lt55") or 0.0)
            lt165_delta = float(delta.get("lt165") or 0.0)
            band_70_90_delta = float(delta.get("band_70_90") or 0.0)
            band_90_120_delta = float(delta.get("band_90_120") or 0.0)
        except (TypeError, ValueError):
            continue
        middle_deficit = band_70_90_delta + band_90_120_delta
        if lt165_delta < -6.0:
            return True
        if lt55_delta > 35.0 and middle_deficit < -32.0:
            return True
    return False


def report_has_excess_black_core(report: dict[str, Any] | None) -> bool:
    if not isinstance(report, dict):
        return False
    neighbor_core_low = bool(local_neighbor_style_issues(report, allow_excess_black_core=True))
    local_issues = report.get("local_ink_balance_issues")
    if isinstance(local_issues, list):
        for issue in local_issues:
            if not isinstance(issue, dict):
                continue
            issue_type = str(issue.get("type") or "")
            if "too_black" in issue_type or issue_type == "roi_core_too_black":
                return not neighbor_core_low
        return False

    if neighbor_core_low:
        return False

    char_bands = report.get("char_gray_band_metrics")
    if not isinstance(char_bands, dict) or not char_bands.get("enabled"):
        return False
    for item in char_bands.get("per_char", []):
        if not isinstance(item, dict):
            continue
        if item.get("source_char") == item.get("target_char"):
            continue
        delta = item.get("delta") or {}
        try:
            lt55_delta = float(delta.get("lt55") or 0.0)
            middle_delta = float(delta.get("band_70_90") or 0.0) + float(delta.get("band_90_120") or 0.0)
        except (TypeError, ValueError):
            continue
        if lt55_delta > 62.0 and middle_delta < -18.0:
            return True
    return False


def shape_change_report(
    size: tuple[int, int],
    plan: RenderPlan,
    params: CandidateParams,
) -> dict[str, Any]:
    source_chars = text_chars(plan.source_text or "")
    target_chars = text_chars(plan.target_text)
    if not source_chars or not target_chars or len(source_chars) != len(target_chars):
        return {"enabled": False, "reason": "requires_same_length_source_and_target"}
    if not plan.slot_boxes or len(plan.slot_boxes) < len(target_chars):
        return {"enabled": False, "reason": "missing_per_character_slots"}
    candidate_boxes = replacement_char_bboxes(size, plan, params)
    if not candidate_boxes:
        return {"enabled": False, "reason": "missing_candidate_char_boxes"}

    ordered_slots = tuple(sorted(plan.slot_boxes, key=lambda item: item.x1))
    per_char: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for idx, (source_char, target_char) in enumerate(zip(source_chars, target_chars)):
        slot = ordered_slots[idx]
        candidate_box = candidate_boxes[idx] if idx < len(candidate_boxes) else None
        slot_w = max(1.0, float(slot.x2 - slot.x1))
        slot_h = max(1.0, float(slot.y2 - slot.y1))
        slot_area = max(1.0, slot_w * slot_h)
        if candidate_box is None:
            issue = {"type": "candidate_char_missing", "index": idx, "target_char": target_char}
            per_char.append(
                {
                    "index": idx,
                    "source_char": source_char,
                    "target_char": target_char,
                    "slot_box": [slot.x1, slot.y1, slot.x2, slot.y2],
                    "candidate_box": None,
                    "issues": [issue],
                }
            )
            issues.append(issue)
            continue
        x1, y1, x2, y2 = candidate_box
        candidate_w = max(1.0, float(x2 - x1))
        candidate_h = max(1.0, float(y2 - y1))
        candidate_area = candidate_w * candidate_h
        slot_cx = (slot.x1 + slot.x2) / 2.0
        slot_cy = (slot.y1 + slot.y2) / 2.0
        candidate_cx = (x1 + x2) / 2.0
        candidate_cy = (y1 + y2) / 2.0
        width_delta_ratio = (candidate_w - slot_w) / slot_w
        height_delta_ratio = (candidate_h - slot_h) / slot_h
        centroid_dx = candidate_cx - slot_cx
        centroid_dy = candidate_cy - slot_cy
        ink_area_ratio = candidate_area / slot_area
        char_issues: list[dict[str, Any]] = []
        if source_char != target_char:
            if abs(width_delta_ratio) > 0.30:
                char_issues.append(
                    {
                        "type": "bbox_width_delta_ratio_large",
                        "actual": round(float(width_delta_ratio), 4),
                        "limit": 0.30,
                    }
                )
            if abs(height_delta_ratio) > 0.24:
                char_issues.append(
                    {
                        "type": "bbox_height_delta_ratio_large",
                        "actual": round(float(height_delta_ratio), 4),
                        "limit": 0.24,
                    }
                )
            if abs(centroid_dx) > max(2.0, slot_h * 0.16):
                char_issues.append(
                    {
                        "type": "centroid_dx_large",
                        "actual": round(float(centroid_dx), 3),
                        "limit": round(float(max(2.0, slot_h * 0.16)), 3),
                    }
                )
            if abs(centroid_dy) > max(2.0, slot_h * 0.16):
                char_issues.append(
                    {
                        "type": "centroid_dy_large",
                        "actual": round(float(centroid_dy), 3),
                        "limit": round(float(max(2.0, slot_h * 0.16)), 3),
                    }
                )
            if ink_area_ratio < 0.42 or ink_area_ratio > 1.95:
                char_issues.append(
                    {
                        "type": "ink_area_ratio_outside_range",
                        "actual": round(float(ink_area_ratio), 4),
                        "range": [0.42, 1.95],
                    }
                )
        per_char.append(
            {
                "index": idx,
                "source_char": source_char,
                "target_char": target_char,
                "slot_box": [slot.x1, slot.y1, slot.x2, slot.y2],
                "candidate_box": [int(x1), int(y1), int(x2), int(y2)],
                "bbox_width_delta_ratio": round(float(width_delta_ratio), 4),
                "bbox_height_delta_ratio": round(float(height_delta_ratio), 4),
                "centroid_dx": round(float(centroid_dx), 3),
                "centroid_dy": round(float(centroid_dy), 3),
                "ink_area_ratio": round(float(ink_area_ratio), 4),
                "issues": char_issues,
            }
        )
        issues.extend(
            {"index": idx, "target_char": target_char, **issue}
            for issue in char_issues
        )
    return {
        "enabled": True,
        "placement_strategy": plan.placement_strategy,
        "placement_strategy_reason": plan.placement_strategy_reason,
        "shape_change_large": bool(issues),
        "per_char": per_char,
        "issues": issues,
    }


def candidate_report(
    original: Image.Image,
    candidate: Image.Image,
    plan: RenderPlan,
    params: CandidateParams,
    font_style_reference: dict[str, Any],
    *,
    pipeline_profile: str | None = None,
) -> dict[str, Any]:
    report = hard_check(original, candidate, plan.target_roi, plan.protected_boxes)
    reference_profile = build_reference_profile(original, plan, params)
    strict_metrics = strict_visual_metrics(original, candidate, plan.target_roi)
    source_count = len(text_chars(plan.source_text or ""))
    target_count = len(text_chars(plan.target_text))
    mismatch = bool(source_count and target_count and source_count != target_count)
    same_length_changed = bool(
        source_count
        and target_count
        and source_count == target_count
        and (plan.source_text or "") != plan.target_text
    )
    complexity_ratio = text_complexity_ratio(plan, params)
    if source_count and target_count > source_count:
        max_dark_pixel_ratio = min(1.90, max(1.42, 1.25 * target_count / source_count))
    elif same_length_changed:
        max_dark_pixel_ratio = min(1.22, max(1.12, 1.12 + max(0.0, complexity_ratio - 1.0) * 0.45))
    else:
        max_dark_pixel_ratio = 1.35 if mismatch else 1.12
    min_dark_pixel_ratio = 0.55 if mismatch and target_count < source_count else 0.78 if mismatch else 0.88
    shorter_replacement = bool(source_count and target_count and target_count < source_count)
    max_edge_lighten_delta = 6.0 if shorter_replacement else 4.0
    dynamic_limits = (reference_profile.get("dynamic_ink") or {}) if isinstance(reference_profile, dict) else {}
    try:
        max_core_lighten_delta = float(dynamic_limits.get("core_mean_lighten_limit") or 2.0)
    except (TypeError, ValueError):
        max_core_lighten_delta = 2.0
    strict_issues = strict_gate_issues(
        strict_metrics,
        max_dark_pixel_ratio=max_dark_pixel_ratio,
        min_dark_pixel_ratio=min_dark_pixel_ratio,
        max_core_mean_gray_delta=18.0,
        max_edge_mean_gray_delta=16.0,
        max_core_lighten_delta=max_core_lighten_delta,
        max_edge_lighten_delta=max_edge_lighten_delta,
    )
    cleanup_metrics = extra_source_slot_cleanup_metrics(original, candidate, plan, params)
    cleanup_issues = extra_source_slot_cleanup_issues(cleanup_metrics)
    alignment_metrics, alignment_issues = char_alignment_gate(
        original.size,
        plan,
        params,
        max_char_center_dx=2.0,
        max_char_center_distance_delta=2.0,
        max_char_center_dy=2.5,
        max_replacement_center_y_range=2.0,
    )
    font_style = font_style_gate(
        original,
        plan,
        params,
        font_style_reference,
        max_score_ratio=1.25,
    )
    shape_report = shape_change_report(original.size, plan, params)
    shape_issues = (
        shape_report.get("issues")
        if isinstance(shape_report, dict) and isinstance(shape_report.get("issues"), list)
        else []
    )
    report["params"] = asdict(params)
    report["pipeline_profile"] = pipeline_profile or "photo_scan"
    report["placement_strategy"] = plan.placement_strategy
    report["placement_strategy_reason"] = plan.placement_strategy_reason
    report["slot_quality_report"] = plan.slot_quality_report
    report["reference_profile"] = reference_profile
    report["strict_visual_metrics"] = strict_metrics
    report["char_gray_band_metrics"] = char_gray_band_metrics(original, candidate, plan)
    report["char_pose_metrics"] = char_pose_metrics(original, plan, params)
    report["photo_texture_metrics"] = photo_texture_metrics(original, candidate, plan, params)
    report["background_texture_metrics"] = background_texture_metrics(original, candidate, plan, params)
    report["extra_source_slot_cleanup_metrics"] = cleanup_metrics
    report["char_alignment_metrics"] = alignment_metrics
    report["shape_change_report"] = shape_report
    report["font_style_gate"] = font_style
    report["strict_gate"] = {
        "max_dark_pixel_ratio": max_dark_pixel_ratio,
        "min_dark_pixel_ratio": min_dark_pixel_ratio,
        "text_complexity_ratio": round(float(complexity_ratio), 4),
        "max_core_mean_gray_delta": 18.0,
        "max_edge_mean_gray_delta": 16.0,
        "max_core_lighten_delta": round(float(max_core_lighten_delta), 3),
        "max_edge_lighten_delta": max_edge_lighten_delta,
        "max_char_center_dx": 2.0,
        "max_char_center_dy": 2.5,
        "max_replacement_center_y_range": 2.0,
        "max_char_center_distance_delta": 2.0,
        "max_font_style_score_ratio": 1.25,
        "pass": not strict_issues
        and not cleanup_issues
        and not alignment_issues
        and not shape_issues
        and bool(font_style.get("pass", True)),
        "issues": strict_issues
        + cleanup_issues
        + alignment_issues
        + list(shape_issues)
        + list(font_style.get("issues", [])),
    }
    report["local_ink_balance_issues"] = local_ink_balance_issues(report)
    report["local_stroke_body_issues"] = local_stroke_body_issues(report)
    report["local_neighbor_style_issues"] = local_neighbor_style_issues(report)
    report["local_pose_issues"] = local_pose_issues(report)
    report["local_photo_texture_issues"] = local_photo_texture_issues(report)
    report["local_background_texture_issues"] = local_background_texture_issues(report)
    report["stage_gate"] = stage_gate_for_report(report, report["pipeline_profile"])
    return report


def processing_candidate_score(report: dict[str, Any]) -> float:
    if not report.get("pass"):
        return 1_000_000.0
    score = 0.0
    score += stage_selection_penalty(report)
    strict_gate = report.get("strict_gate")
    longer_replacement = (
        isinstance(strict_gate, dict)
        and float(strict_gate.get("max_dark_pixel_ratio") or 0.0) > 1.42
    )
    if isinstance(strict_gate, dict) and not strict_gate.get("pass", True):
        score += len(strict_gate.get("issues", [])) * 220.0

    metrics = report.get("strict_visual_metrics", {})
    thresholds = metrics.get("thresholds", {})
    for threshold, values in thresholds.items():
        ratio = values.get("dark_pixel_ratio")
        mean_delta = values.get("mean_gray_delta")
        if ratio is not None:
            if longer_replacement:
                desired_ratio = {
                    "120": 1.28,
                    "140": 1.16,
                    "150": 1.08,
                    "160": 0.98,
                    "165": 0.95,
                }.get(str(threshold), 1.0)
            else:
                desired_ratio = 1.0
            score += abs(float(ratio) - desired_ratio) * (80.0 if str(threshold) == "165" else 45.0)
        if mean_delta is not None:
            score += abs(float(mean_delta)) * (2.6 if str(threshold) == "165" else 1.8)

    bands = metrics.get("bands", {})
    if isinstance(bands, dict):
        old_lt55 = float(bands.get("old_lt55_pixels") or 0.0)
        new_lt55 = float(bands.get("new_lt55_pixels") or 0.0)
        old_lt70 = float(bands.get("old_lt70_pixels") or 0.0)
        new_lt70 = float(bands.get("new_lt70_pixels") or 0.0)
        old_gray_edge = float(bands.get("old_120_165_pixels") or 0.0)
        new_gray_edge = float(bands.get("new_120_165_pixels") or 0.0)
        if old_lt55 >= 16:
            score += max(0.0, (new_lt55 - old_lt55) - max(58.0, old_lt55 * 0.28)) * 5.0
        if old_lt70 >= 32:
            score += max(0.0, (new_lt70 - old_lt70) - max(70.0, old_lt70 * 0.24)) * 2.8
        if old_lt55 < 16:
            score += max(0.0, new_lt55 - old_lt55 - 24.0) * 18.0
        if old_lt70 < 32:
            score += max(0.0, new_lt70 - old_lt70 - 36.0) * 8.0
        score += abs(new_gray_edge - old_gray_edge) * 0.10

    cleanup_metrics = report.get("extra_source_slot_cleanup_metrics")
    if isinstance(cleanup_metrics, dict) and cleanup_metrics.get("enabled"):
        for item in cleanup_metrics.get("per_box", []):
            lt150_ratio = item.get("new_lt150_ratio")
            retention = item.get("lt150_retention_ratio")
            column_deviation = item.get("max_column_mean_deviation")
            if lt150_ratio is not None:
                score += max(0.0, float(lt150_ratio) - 0.030) * 900.0
            if retention is not None:
                score += max(0.0, float(retention) - 0.12) * 120.0
            if column_deviation is not None:
                score += max(0.0, float(column_deviation) - 3.0) * 18.0

    font_style = report.get("font_style_gate")
    if isinstance(font_style, dict):
        ratio = font_style.get("score_ratio_to_best")
        if ratio is not None:
            score += max(0.0, float(ratio) - 1.0) * 180.0

    body_issues = report.get("local_stroke_body_issues")
    if isinstance(body_issues, list) and body_issues:
        score += len(body_issues) * 260.0
    if report_needs_wider_gray_strokes(report):
        score += gray_stroke_balance_penalty(report) * 2.4

    params = report.get("params")
    if longer_replacement and isinstance(params, dict):
        blur = float(params.get("blur") or 0.0)
        score += max(0.0, blur - 0.62) * 120.0
        score += max(0.0, 0.42 - blur) * 80.0
    return score


def region_candidate_score(
    original: Image.Image,
    candidate: Image.Image,
    plan: RenderPlan,
    report: dict[str, Any],
) -> float:
    if plan.draw_mode == "center":
        return processing_candidate_score(report)
    return local_score(original, candidate, plan, report) + stage_selection_penalty(report)


def old_region_lt55_pixels(img: Image.Image, roi: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = roi
    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return int(np.count_nonzero(gray[y1:y2, x1:x2] < 55))


def region_context_box(
    roi: tuple[int, int, int, int],
    image_size: tuple[int, int],
    *,
    pad_ratio: float = 0.60,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi
    pad = max(24, int(round(max(x2 - x1, y2 - y1) * pad_ratio)))
    return clamp_box((x1 - pad, y1 - pad, x2 + pad, y2 + pad), image_size)


def save_region_context(
    img: Image.Image,
    box: tuple[int, int, int, int],
    path: Path,
    *,
    scale: int = 4,
) -> None:
    crop = img.crop(box)
    if scale > 1:
        crop = crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(path)


def save_region_compare(
    original: Image.Image,
    candidate: Image.Image,
    box: tuple[int, int, int, int],
    path: Path,
    *,
    scale: int = 4,
) -> None:
    old = original.crop(box)
    new = candidate.crop(box)
    w, h = old.size
    label_h = 24
    sheet = Image.new("RGB", (w * scale * 2, h * scale + label_h), (248, 248, 248))
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 5), "original", fill=(20, 20, 20))
    draw.text((w * scale + 6, 5), "candidate", fill=(20, 20, 20))
    sheet.paste(old.resize((w * scale, h * scale), Image.Resampling.NEAREST), (0, label_h))
    sheet.paste(new.resize((w * scale, h * scale), Image.Resampling.NEAREST), (w * scale, label_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def compact_hard_reports(
    rendered: list[tuple[CandidateParams, Image.Image, dict[str, Any], float]],
    plan: RenderPlan,
) -> dict[str, Any]:
    reports: dict[str, Any] = {
        "task": {
            "source_text": plan.source_text,
            "target_text": plan.target_text,
            "search_roi": list(plan.search_roi),
            "target_roi": list(plan.target_roi),
            "draw_mode": plan.draw_mode,
            "text_angle_degrees": round(float(plan.text_angle_degrees), 3),
            "slot_boxes": [asdict(item) for item in plan.slot_boxes],
            "protected_boxes": [list(item) for item in plan.protected_boxes],
        },
        "candidates": {},
    }
    for params, _candidate, report, score in rendered:
        reports["candidates"][params.candidate_id] = {
            "label": params_label(params),
            "params": asdict(params),
            "score": round(float(score), 3),
            "hard_check": report,
        }
    return reports
