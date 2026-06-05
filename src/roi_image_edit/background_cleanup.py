from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image

from roi_image_edit.iterative_pipeline import (
    CandidateParams,
    RenderPlan,
    clamp_box,
    draw_replacement_layer,
    extra_source_slot_cleanup_boxes,
    gray_array,
    text_chars,
)


def _ratio(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(float(count / total), 5)


def _local_background_gray(gray: np.ndarray, box: tuple[int, int, int, int]) -> float:
    height, width = gray.shape[:2]
    x1, y1, x2, y2 = box
    px1 = max(0, x1 - 8)
    py1 = max(0, y1 - 5)
    px2 = min(width, x2 + 8)
    py2 = min(height, y2 + 5)
    if px2 <= px1 or py2 <= py1:
        return float(np.percentile(gray[y1:y2, x1:x2], 75))

    yy, xx = np.mgrid[py1:py2, px1:px2]
    slot_mask = (xx >= x1) & (xx < x2) & (yy >= y1) & (yy < y2)
    ring_values = gray[py1:py2, px1:px2][~slot_mask]
    if ring_values.size == 0:
        return float(np.percentile(gray[y1:y2, x1:x2], 75))

    background_values = ring_values[ring_values >= 125]
    if background_values.size:
        return float(np.percentile(background_values, 50))
    return float(np.percentile(ring_values, 75))


def _gray_residual_threshold(local_background_gray: float) -> int:
    return int(round(max(118.0, min(155.0, local_background_gray - 12.0))))


def _dark_residual_counts(
    old_gray: np.ndarray,
    new_gray: np.ndarray,
    mask: np.ndarray,
    *,
    candidate_gray_threshold: int = 165,
) -> dict[str, Any]:
    total = int(np.count_nonzero(mask))
    if total <= 0:
        return {
            "old_core_pixels": 0,
            "old_gray_edge_pixels": 0,
            "candidate_core_residual_pixels": 0,
            "candidate_gray_residual_pixels": 0,
            "candidate_core_residual_ratio": 0.0,
            "candidate_gray_residual_ratio": 0.0,
        }
    old_values = old_gray[mask]
    new_values = new_gray[mask]
    old_core = int(np.count_nonzero(old_values < 120))
    old_gray_edge = int(np.count_nonzero((old_values >= 120) & (old_values < 165)))
    candidate_core = int(np.count_nonzero(new_values < 120))
    candidate_gray = int(np.count_nonzero(new_values < candidate_gray_threshold))
    return {
        "old_core_pixels": old_core,
        "old_gray_edge_pixels": old_gray_edge,
        "candidate_core_residual_pixels": candidate_core,
        "candidate_gray_residual_pixels": candidate_gray,
        "candidate_core_residual_ratio": _ratio(candidate_core, max(1, old_core)),
        "candidate_gray_residual_ratio": _ratio(candidate_gray, max(1, old_core + old_gray_edge)),
    }


def source_slot_precleanup_report(
    original: Image.Image,
    candidate: Image.Image,
    plan: RenderPlan,
    params: CandidateParams | None = None,
    *,
    max_core_residual_ratio: float = 0.025,
    max_gray_residual_ratio: float = 0.080,
) -> dict[str, Any]:
    source_chars = text_chars(plan.source_text or "")
    if not source_chars or not plan.slot_boxes:
        return {"enabled": False, "reason": "missing source slots"}

    old_gray = gray_array(original)
    new_gray = gray_array(candidate)
    alpha = np.zeros_like(old_gray, dtype=np.uint8)
    alpha_status: dict[str, Any] = {"available": False, "reason": "params_missing"}
    if params is not None:
        try:
            alpha = np.array(draw_replacement_layer(size=original.size, plan=plan, params=params).getchannel("A"))
            alpha_status = {"available": True, "reason": "replacement_layer_rendered"}
        except OSError as exc:
            alpha_status = {"available": False, "reason": "replacement_layer_font_unavailable", "error": str(exc)}
    ignore_new_text = alpha > 0
    ordered_slots = tuple(sorted(plan.slot_boxes, key=lambda item: item.x1)[: len(source_chars)])
    per_slot: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for index, slot in enumerate(ordered_slots):
        x1, y1, x2, y2 = clamp_box((slot.x1, slot.y1, slot.x2, slot.y2), original.size)
        if x2 <= x1 or y2 <= y1:
            continue
        old_crop = old_gray[y1:y2, x1:x2]
        cleanup_mask = old_crop < 165
        cleanup_mask &= ~ignore_new_text[y1:y2, x1:x2]
        local_background = _local_background_gray(new_gray, (x1, y1, x2, y2))
        gray_threshold = _gray_residual_threshold(local_background)
        counts = _dark_residual_counts(
            old_gray[y1:y2, x1:x2],
            new_gray[y1:y2, x1:x2],
            cleanup_mask,
            candidate_gray_threshold=gray_threshold,
        )
        item = {
            "index": index,
            "source_char": source_chars[index] if index < len(source_chars) else None,
            "box": [x1, y1, x2, y2],
            "cleanup_pixels": int(np.count_nonzero(cleanup_mask)),
            "excluded_by_new_text_alpha_pixels": int(np.count_nonzero((old_crop < 165) & ignore_new_text[y1:y2, x1:x2])),
            "local_background_gray": round(local_background, 3),
            "candidate_gray_residual_threshold": gray_threshold,
            **counts,
        }
        per_slot.append(item)
        if item["candidate_core_residual_ratio"] > max_core_residual_ratio:
            issues.append(
                {
                    "type": "source_slot_core_residue",
                    "index": index,
                    "box": item["box"],
                    "actual": item["candidate_core_residual_ratio"],
                    "limit": max_core_residual_ratio,
                    "candidate_core_residual_pixels": item["candidate_core_residual_pixels"],
                }
            )
        if item["candidate_gray_residual_ratio"] > max_gray_residual_ratio:
            issues.append(
                {
                    "type": "source_slot_gray_edge_residue",
                    "index": index,
                    "box": item["box"],
                    "actual": item["candidate_gray_residual_ratio"],
                    "limit": max_gray_residual_ratio,
                    "candidate_gray_residual_pixels": item["candidate_gray_residual_pixels"],
                }
            )
    return {
        "enabled": True,
        "mask_scope": "source_slots_excluding_new_text_alpha",
        "replacement_alpha": alpha_status,
        "source_slot_boxes": [[slot.x1, slot.y1, slot.x2, slot.y2] for slot in ordered_slots],
        "thresholds": {
            "old_core_threshold": 120,
            "old_gray_edge_threshold": 165,
            "candidate_core_threshold": 120,
            "candidate_gray_threshold_rule": "max(118, min(155, local_background_gray - 12))",
            "max_core_residual_ratio": max_core_residual_ratio,
            "max_gray_residual_ratio": max_gray_residual_ratio,
        },
        "per_slot": per_slot,
        "issues": issues,
        "pass": not issues,
    }


def extra_source_cleanup_coverage_report(plan: RenderPlan) -> dict[str, Any]:
    source_count = len(text_chars(plan.source_text or ""))
    target_count = len(text_chars(plan.target_text))
    expected_extra_slots = max(0, source_count - target_count)
    boxes = [list(box) for box in extra_source_slot_cleanup_boxes(plan)]
    slot_report = plan.slot_quality_report if isinstance(plan.slot_quality_report, dict) else {}
    length_report = slot_report.get("length_change_report") if isinstance(slot_report, dict) else {}
    cleanup_mask = length_report.get("cleanup_mask_report") if isinstance(length_report, dict) else {}
    slot_quality_boxes = cleanup_mask.get("boxes") if isinstance(cleanup_mask, dict) else []
    covered = bool(expected_extra_slots == 0 or boxes or slot_quality_boxes)
    return {
        "enabled": expected_extra_slots > 0,
        "source_count": source_count,
        "target_count": target_count,
        "expected_extra_slots": expected_extra_slots,
        "extra_source_cleanup_boxes": boxes,
        "slot_quality_cleanup_mask_boxes": slot_quality_boxes if isinstance(slot_quality_boxes, list) else [],
        "covered": covered,
        "pass": covered,
        "issues": [] if covered else [{"type": "extra_source_slot_cleanup_missing", "expected_extra_slots": expected_extra_slots}],
    }


def pre_cleanup_report(
    original: Image.Image,
    candidate: Image.Image,
    plan: RenderPlan,
    params: CandidateParams | None = None,
) -> dict[str, Any]:
    source_report = source_slot_precleanup_report(original, candidate, plan, params)
    extra_report = extra_source_cleanup_coverage_report(plan)
    issues = [
        *(source_report.get("issues") or [] if isinstance(source_report, dict) else []),
        *(extra_report.get("issues") or [] if isinstance(extra_report, dict) else []),
    ]
    return {
        "enabled": bool(source_report.get("enabled") or extra_report.get("enabled")),
        "stage": "pre_cleanup",
        "purpose": "remove_old_value_core_and_gray_before_post_blend",
        "source_slot_cleanup": source_report,
        "extra_source_slot_cleanup": extra_report,
        "issues": issues,
        "pass": not issues,
    }


def post_blend_report(
    plan: RenderPlan,
    background_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    metrics = background_metrics if isinstance(background_metrics, dict) else {}
    target_roi = tuple(metrics.get("target_roi") or plan.target_roi)
    scope_box = clamp_box(target_roi, (max(plan.search_roi[2], plan.target_roi[2]), max(plan.search_roi[3], plan.target_roi[3])))
    protected_overlap = 0
    sx1, sy1, sx2, sy2 = scope_box
    for px1, py1, px2, py2 in plan.protected_boxes:
        protected_overlap += max(0, min(sx2, px2) - max(sx1, px1)) * max(0, min(sy2, py2) - max(sy1, py1))
    white_probe = metrics.get("white_ghost_probe") if isinstance(metrics.get("white_ghost_probe"), dict) else {}
    trailing = metrics.get("trailing_cleanup_patch") if isinstance(metrics.get("trailing_cleanup_patch"), dict) else {}
    residual_ratio = metrics.get("residual_ratio")
    std_ratio = metrics.get("std_ratio")
    mean_delta = metrics.get("new_reference_mean_delta")
    axes = {
        "patch_visible": {"value": abs(float(mean_delta or 0.0)), "limit": 12.0},
        "white_ghost": {"value": float(white_probe.get("bright_over_background_p95_ratio") or 0.0), "limit": 0.14},
        "dark_shadow": {"value": float(white_probe.get("dark_under_background_p10_ratio") or 0.0), "limit": 0.28},
        "smooth_smear": {"value": float(trailing.get("residual_ratio") or residual_ratio or 1.0), "min": 0.42},
        "texture_break": {"value": abs(1.0 - float(residual_ratio or 1.0)), "limit": 0.72},
        "roi_edge_seam": {"value": abs(float(mean_delta or 0.0)) + max(0.0, 0.48 - float(std_ratio or 1.0)), "limit": 13.0},
    }
    issues: list[dict[str, Any]] = []
    if protected_overlap:
        issues.append({"type": "post_blend_scope_overlaps_protected_text", "protected_overlap_pixels": protected_overlap})
    if axes["patch_visible"]["value"] > axes["patch_visible"]["limit"]:
        issues.append({"type": "post_blend_patch_visible", **axes["patch_visible"]})
    if axes["white_ghost"]["value"] > axes["white_ghost"]["limit"]:
        issues.append({"type": "post_blend_white_ghost", **axes["white_ghost"]})
    if axes["dark_shadow"]["value"] > axes["dark_shadow"]["limit"]:
        issues.append({"type": "post_blend_dark_shadow", **axes["dark_shadow"]})
    if axes["smooth_smear"]["value"] < axes["smooth_smear"]["min"]:
        issues.append({"type": "post_blend_smooth_smear", **axes["smooth_smear"]})
    if axes["texture_break"]["value"] > axes["texture_break"]["limit"]:
        issues.append({"type": "post_blend_texture_break", **axes["texture_break"]})
    if axes["roi_edge_seam"]["value"] > axes["roi_edge_seam"]["limit"]:
        issues.append({"type": "post_blend_roi_edge_seam", **axes["roi_edge_seam"]})
    return {
        "enabled": bool(metrics.get("enabled")),
        "stage": "post_blend",
        "scope": {
            "scope_box": list(scope_box),
            "target_roi": list(plan.target_roi),
            "outside_target_roi_pixels": 0,
            "protected_overlap_pixels": protected_overlap,
        },
        "artifact_axes": axes,
        "issues": issues,
        "pass": not issues,
    }


def background_cleanup_stage_report(
    pre_cleanup: dict[str, Any],
    post_blend: dict[str, Any],
) -> dict[str, Any]:
    pre_pass = bool(pre_cleanup.get("pass"))
    post_pass = bool(post_blend.get("pass"))
    if not pre_pass:
        blocking_step = "pre_cleanup"
        reason = "pre_cleanup_failed"
    elif not post_pass:
        blocking_step = "post_blend"
        reason = "post_blend_failed"
    else:
        blocking_step = None
        reason = None
    return {
        "stage": "background_cleanup",
        "priority_order": ["pre_cleanup", "post_blend"],
        "pre_cleanup": pre_cleanup,
        "post_blend": post_blend,
        "post_blend_can_deliver": pre_pass,
        "blocking_step": blocking_step,
        "blocking_reason": reason,
        "issues": [*(pre_cleanup.get("issues") or []), *(post_blend.get("issues") or [])],
        "pass": pre_pass and post_pass,
    }
