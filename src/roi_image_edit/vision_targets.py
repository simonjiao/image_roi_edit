from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from roi_image_edit.stage_policy import STAGE_ORDER, optimization_policy_audit
from roi_image_edit.stages import stage_gate_for_report


@dataclass(frozen=True)
class VisionAxis:
    axis: str
    stage: str
    value: str
    source_field: str


def _clean(value: Any) -> str:
    return str(value or "").strip().lower()


def _final_acceptance_delivers(acceptance: dict[str, Any]) -> bool:
    final_level = _clean(acceptance.get("acceptance_level"))
    final_decision = _clean(acceptance.get("final_decision"))
    return bool(acceptance.get("pass")) and final_level == "pass" and final_decision == "deliver"


def _visual_axes(acceptance: dict[str, Any] | None) -> list[VisionAxis]:
    findings = acceptance.get("visual_findings") if isinstance(acceptance, dict) else {}
    if not isinstance(findings, dict):
        findings = {}
    axes: list[VisionAxis] = []

    background = _clean(findings.get("background"))
    if background in {"patch_visible", "ghost_visible", "seam_visible", "too_smooth"}:
        axes.append(VisionAxis(background, "background_cleanup", background, "visual_findings.background"))
    if background in {"white_specks", "white_dots", "white_ghost"}:
        axes.append(VisionAxis("white_specks", "background_cleanup", background, "visual_findings.background"))

    sharpness = _clean(findings.get("sharpness") or findings.get("blur"))
    if sharpness in {"too_sharp", "too_blurry"}:
        axes.append(VisionAxis(sharpness, "photo_texture", sharpness, "visual_findings.sharpness"))

    darkness = _clean(findings.get("darkness"))
    if darkness in {"too_dark", "slightly_dark", "too_light", "slightly_light"}:
        axes.append(VisionAxis(darkness, "ink_gray_balance", darkness, "visual_findings.darkness"))
    stroke_weight = _clean(findings.get("stroke_weight"))
    if stroke_weight in {"too_bold", "slightly_bold", "too_thin", "slightly_thin"}:
        axes.append(VisionAxis(stroke_weight, "ink_gray_balance", stroke_weight, "visual_findings.stroke_weight"))

    for field in ("char_positions", "spacing", "baseline", "font_similarity", "size"):
        value = _clean(findings.get(field))
        if value and value not in {"ok", "pass"}:
            axes.append(VisionAxis(field, "text_shape", value, f"visual_findings.{field}"))

    seen: set[tuple[str, str]] = set()
    unique: list[VisionAxis] = []
    for axis in axes:
        key = (axis.axis, axis.stage)
        if key in seen:
            continue
        seen.add(key)
        unique.append(axis)
    return unique


def _primary_stage(axes: list[VisionAxis]) -> str | None:
    for axis in axes:
        if axis.stage in STAGE_ORDER:
            return axis.stage
    return None


def _bounds_for_axis(axis: VisionAxis) -> list[dict[str, Any]]:
    if axis.axis == "too_sharp":
        return [
            {"parameter": "blur", "direction": "increase", "min_delta": 0.04, "max_delta": 0.16},
            {"parameter": "edge_breakup", "direction": "increase_limited", "min_delta": 0.004, "max_delta": 0.020},
            {"parameter": "photo_noise", "direction": "increase_limited", "max_delta": 0.025},
            {"parameter": "core_ink_gain", "direction": "not_increase"},
        ]
    if axis.axis == "too_blurry":
        return [
            {"parameter": "blur", "direction": "decrease", "min_delta": 0.04, "max_delta": 0.12},
            {"parameter": "alpha_contrast", "direction": "not_decrease"},
        ]
    if axis.axis in {"patch_visible", "ghost_visible", "seam_visible", "too_smooth"}:
        return [
            {"parameter": "mask_threshold", "direction": "maintain_or_increase"},
            {"parameter": "inpaint_radius", "direction": "maintain_or_increase"},
            {"parameter": "photo_noise", "direction": "increase_limited", "max_delta": 0.020},
            {"parameter": "edge_breakup", "direction": "increase_limited", "max_delta": 0.012},
        ]
    if axis.axis == "white_specks":
        return [
            {"parameter": "photo_noise", "direction": "decrease", "min_delta": 0.004, "max_delta": 0.030},
            {"parameter": "edge_breakup", "direction": "decrease", "min_delta": 0.002, "max_delta": 0.016},
            {"parameter": "jpeg_quality", "direction": "increase"},
        ]
    if axis.axis in {"too_dark", "slightly_dark", "too_bold", "slightly_bold"}:
        return [
            {"parameter": "opacity", "direction": "decrease", "min_delta": 0.010, "max_delta": 0.080},
            {"parameter": "core_ink_gain", "direction": "decrease", "max_delta": 0.100},
            {"parameter": "core_darken_strength", "direction": "decrease", "max_delta": 0.100},
            {"parameter": "blur", "direction": "increase_limited", "max_delta": 0.120},
        ]
    if axis.axis in {"too_light", "slightly_light", "too_thin", "slightly_thin"}:
        return [
            {"parameter": "opacity", "direction": "increase", "max_delta": 0.060},
            {"parameter": "core_ink_gain", "direction": "increase", "max_delta": 0.080},
            {"parameter": "core_darken_strength", "direction": "increase", "max_delta": 0.080},
        ]
    return []


def _prior_axis_counts(prior_targets: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for target in prior_targets or []:
        if not isinstance(target, dict) or not target.get("active"):
            continue
        for axis in target.get("axes") or []:
            if isinstance(axis, dict):
                key = str(axis.get("axis") or "")
            else:
                key = str(axis or "")
            if key:
                counts[key] = counts.get(key, 0) + 1
    return counts


def vision_target_from_acceptance(
    report: dict[str, Any] | None,
    acceptance: dict[str, Any] | None,
    *,
    prior_targets: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    round_index: int | None = None,
    basis_candidate_id: str | None = None,
) -> dict[str, Any]:
    stage_gate = stage_gate_for_report(report) if isinstance(report, dict) else {}
    local_stage = stage_gate.get("blocking_stage") if isinstance(stage_gate, dict) else None
    axes = _visual_axes(acceptance)
    visual_rejected = bool(
        isinstance(acceptance, dict)
        and not _final_acceptance_delivers(acceptance)
        and (
            _clean(acceptance.get("final_decision")) in {"revise", "reject"}
            or _clean(acceptance.get("acceptance_level")) in {"marginal", "fail", "rejected"}
            or not bool(acceptance.get("pass"))
        )
    )
    active = bool(not local_stage and visual_rejected and axes)
    primary_stage = _primary_stage(axes) if active else None
    prior_counts = _prior_axis_counts(prior_targets)
    axis_records: list[dict[str, Any]] = []
    bounds: list[dict[str, Any]] = []
    for axis in axes:
        count = prior_counts.get(axis.axis, 0) + 1
        record = {
            **asdict(axis),
            "repeat_count": count,
            "escalated": count >= 2,
        }
        axis_records.append(record)
        for bound in _bounds_for_axis(axis):
            item = {**bound, "axis": axis.axis, "stage": axis.stage}
            if item not in bounds:
                bounds.append(item)
    combo_axes = {axis.axis for axis in axes}
    combo_recipe = None
    if {"too_sharp", "patch_visible"} <= combo_axes:
        combo_recipe = "too_sharp_plus_patch_visible"
    elif combo_axes & {"too_dark", "slightly_dark", "too_bold", "slightly_bold"} and "too_sharp" in combo_axes:
        combo_recipe = "too_dark_plus_too_sharp"
    elif "patch_visible" in combo_axes and "white_specks" in combo_axes:
        combo_recipe = "patch_visible_plus_white_specks"
    return {
        "active": active,
        "stage": primary_stage,
        "stage_source": "vision_acceptance" if active else None,
        "source": "final_acceptance",
        "round": round_index,
        "basis_candidate_id": basis_candidate_id,
        "local_blocking_stage": local_stage,
        "visual_rejected": visual_rejected,
        "axes": axis_records if active else [],
        "axis_keys": [axis["axis"] for axis in axis_records] if active else [],
        "stage_targets": sorted({axis.stage for axis in axes}) if active else [],
        "bounds": bounds if active else [],
        "combo_recipe": combo_recipe if active else None,
        "repeated": any(axis["repeat_count"] >= 2 for axis in axis_records) if active else False,
        "reason": (
            "local stages passed but final visual acceptance rejected with concrete findings"
            if active
            else "local blocking stage exists or visual acceptance has no actionable findings"
        ),
    }


def vision_target_recipe_patches(target: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(target, dict) or not target.get("active"):
        return []
    recipe = target.get("combo_recipe")
    if recipe == "too_sharp_plus_patch_visible":
        return [
            {
                "blur_delta": 0.08,
                "edge_breakup_delta": 0.006,
                "photo_noise_delta": 0.012,
                "mask_threshold_delta": 6,
                "inpaint_radius_delta": 1,
            },
            {
                "blur_delta": 0.10,
                "edge_breakup_delta": 0.008,
                "photo_noise_delta": 0.014,
                "mask_dilate_iterations_delta": 1,
            },
            {
                "blur_delta": 0.06,
                "edge_breakup_delta": 0.004,
                "photo_noise_delta": 0.008,
                "mask_threshold_delta": 10,
                "jpeg_quality_delta": -2,
            },
        ]
    if recipe == "too_dark_plus_too_sharp":
        return [
            {"opacity_delta": -0.025, "core_ink_gain_delta": -0.030, "blur_delta": 0.08},
            {"core_darken_strength_delta": -0.035, "blur_delta": 0.10, "edge_breakup_delta": 0.004},
        ]
    if recipe == "patch_visible_plus_white_specks":
        return [
            {"photo_noise_delta": -0.012, "edge_breakup_delta": -0.006, "jpeg_quality_delta": 6, "mask_threshold_delta": 6},
            {"photo_noise_delta": -0.018, "edge_breakup_delta": -0.008, "inpaint_radius_delta": 1},
        ]
    return []


def vision_target_recipe_report(target: dict[str, Any] | None) -> dict[str, Any]:
    patches = vision_target_recipe_patches(target)
    stage = target.get("stage") if isinstance(target, dict) else None
    return {
        "enabled": bool(patches),
        "vision_target_stage": stage,
        "combo_recipe": target.get("combo_recipe") if isinstance(target, dict) else None,
        "candidate_count": len(patches),
        "recipe_limit": 3,
        "cross_stage_cartesian_search": False,
        "patches": patches,
        "patch_audits": [
            optimization_policy_audit(str(stage) if stage else None, patch)
            for patch in patches
        ],
    }


def vision_target_alignment(
    target: dict[str, Any] | None,
    params: Any,
    basis_params: Any,
) -> dict[str, Any]:
    if not isinstance(target, dict) or not target.get("active"):
        return {"enabled": False, "score_adjustment": 0.0, "direction": "neutral", "axis_results": []}
    axis_results: list[dict[str, Any]] = []
    adjustment = 0.0

    def delta(name: str) -> float:
        return float(getattr(params, name)) - float(getattr(basis_params, name))

    axes = [str(axis.get("axis") or "") for axis in target.get("axes") or [] if isinstance(axis, dict)]
    repeat_multiplier = 1.0 + (0.5 if target.get("repeated") else 0.0)

    for axis in axes:
        axis_adjustment = 0.0
        aligned = False
        opposite = False
        details: dict[str, Any] = {}
        if axis == "too_sharp":
            blur_delta = delta("blur")
            edge_delta = delta("edge_breakup")
            noise_delta = delta("photo_noise")
            aligned = blur_delta >= 0.035 or edge_delta >= 0.004
            opposite = blur_delta <= -0.010 or edge_delta < -0.004 or noise_delta > 0.035
            axis_adjustment -= min(max(blur_delta, 0.0), 0.14) * 1800.0
            axis_adjustment -= min(max(edge_delta, 0.0), 0.020) * 5000.0
            if blur_delta <= 0.0:
                axis_adjustment += 220.0
            if opposite:
                axis_adjustment += 420.0
            details = {"blur_delta": round(blur_delta, 4), "edge_breakup_delta": round(edge_delta, 4), "photo_noise_delta": round(noise_delta, 4)}
        elif axis in {"patch_visible", "ghost_visible", "seam_visible", "too_smooth"}:
            mask_delta = delta("mask_threshold")
            inpaint_delta = delta("inpaint_radius")
            noise_delta = delta("photo_noise")
            edge_delta = delta("edge_breakup")
            aligned = mask_delta > 0 or inpaint_delta > 0 or 0.0 < noise_delta <= 0.020 or 0.0 < edge_delta <= 0.012
            opposite = mask_delta < 0 or inpaint_delta < 0 or noise_delta > 0.035 or edge_delta > 0.020
            if mask_delta >= 0:
                axis_adjustment -= min(mask_delta, 16.0) * 5.0
            if inpaint_delta >= 0:
                axis_adjustment -= min(inpaint_delta, 2.0) * 70.0
            if 0.0 < noise_delta <= 0.020:
                axis_adjustment -= noise_delta * 2500.0
            if opposite:
                axis_adjustment += 520.0
            details = {"mask_threshold_delta": round(mask_delta, 4), "inpaint_radius_delta": round(inpaint_delta, 4), "photo_noise_delta": round(noise_delta, 4), "edge_breakup_delta": round(edge_delta, 4)}
        elif axis == "white_specks":
            noise_drop = -delta("photo_noise")
            edge_drop = -delta("edge_breakup")
            jpeg_gain = delta("jpeg_quality")
            aligned = noise_drop >= 0.004 or edge_drop >= 0.002 or jpeg_gain > 0
            opposite = noise_drop < -0.004 or edge_drop < -0.002
            axis_adjustment -= min(max(noise_drop, 0.0), 0.030) * 3000.0
            axis_adjustment -= min(max(edge_drop, 0.0), 0.016) * 3600.0
            axis_adjustment -= min(max(jpeg_gain, 0.0), 10.0) * 5.0
            if opposite:
                axis_adjustment += 520.0
            details = {"photo_noise_delta": round(delta("photo_noise"), 4), "edge_breakup_delta": round(delta("edge_breakup"), 4), "jpeg_quality_delta": round(jpeg_gain, 4)}
        elif axis in {"too_dark", "slightly_dark", "too_bold", "slightly_bold"}:
            opacity_drop = -delta("opacity")
            core_drop = -delta("core_ink_gain")
            darken_drop = -delta("core_darken_strength")
            blur_delta = delta("blur")
            aligned = opacity_drop >= 0.010 or core_drop > 0.0 or darken_drop > 0.0
            opposite = opacity_drop < -0.005 or core_drop < -0.004 or darken_drop < -0.004
            axis_adjustment -= min(max(opacity_drop, 0.0), 0.08) * 1600.0
            axis_adjustment -= min(max(core_drop, 0.0), 0.10) * 650.0
            axis_adjustment -= min(max(darken_drop, 0.0), 0.10) * 650.0
            axis_adjustment -= min(max(blur_delta, 0.0), 0.12) * 180.0
            if opposite:
                axis_adjustment += 560.0
            details = {"opacity_delta": round(delta("opacity"), 4), "core_ink_gain_delta": round(delta("core_ink_gain"), 4), "core_darken_strength_delta": round(delta("core_darken_strength"), 4), "blur_delta": round(blur_delta, 4)}
        elif axis in {"too_light", "slightly_light", "too_thin", "slightly_thin"}:
            opacity_gain = delta("opacity")
            core_gain = delta("core_ink_gain")
            darken_gain = delta("core_darken_strength")
            aligned = opacity_gain >= 0.010 or core_gain > 0.0 or darken_gain > 0.0
            opposite = opacity_gain < -0.005 or core_gain < -0.004 or darken_gain < -0.004
            axis_adjustment -= min(max(opacity_gain, 0.0), 0.06) * 1300.0
            axis_adjustment -= min(max(core_gain, 0.0), 0.08) * 620.0
            axis_adjustment -= min(max(darken_gain, 0.0), 0.08) * 620.0
            if opposite:
                axis_adjustment += 460.0
            details = {"opacity_delta": round(opacity_gain, 4), "core_ink_gain_delta": round(core_gain, 4), "core_darken_strength_delta": round(darken_gain, 4)}
        axis_adjustment *= repeat_multiplier
        adjustment += axis_adjustment
        axis_results.append(
            {
                "axis": axis,
                "aligned": aligned,
                "opposite": opposite,
                "score_adjustment": round(float(axis_adjustment), 3),
                "deltas": details,
            }
        )

    opposite_any = any(item["opposite"] for item in axis_results)
    aligned_any = any(item["aligned"] for item in axis_results)
    return {
        "enabled": True,
        "vision_target_stage": target.get("stage"),
        "axis_results": axis_results,
        "score_adjustment": round(float(adjustment), 3),
        "direction": "opposite" if opposite_any else "aligned" if aligned_any else "neutral",
        "aligned": aligned_any and not opposite_any,
        "opposite": opposite_any,
    }


def vision_target_alignment_complete(
    target: dict[str, Any] | None,
    alignment: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(target, dict) or not target.get("active"):
        return {"enabled": False}
    if not isinstance(alignment, dict) or not alignment.get("enabled"):
        return {"enabled": False}
    axis_results = [
        item
        for item in alignment.get("axis_results") or []
        if isinstance(item, dict)
    ]
    results_by_axis = {str(item.get("axis") or ""): item for item in axis_results}
    axes = [
        axis
        for axis in target.get("axes") or []
        if isinstance(axis, dict) and axis.get("axis")
    ]
    required_axes = [
        str(axis.get("axis"))
        for axis in axes
        if target.get("stage") != "background_cleanup"
        or axis.get("stage") in {"background_cleanup", "photo_texture"}
    ]
    if not required_axes:
        required_axes = [str(axis.get("axis")) for axis in axes]
    missing_axes = [
        axis
        for axis in required_axes
        if not bool((results_by_axis.get(axis) or {}).get("aligned"))
    ]
    opposite_axes = [
        str(item.get("axis") or "")
        for item in axis_results
        if item.get("opposite")
    ]
    return {
        "enabled": True,
        "required_axes": required_axes,
        "missing_axes": missing_axes,
        "opposite_axes": opposite_axes,
        "complete": not missing_axes and not opposite_axes,
        "reason": (
            "all required visual target axes are covered"
            if not missing_axes and not opposite_axes
            else "candidate does not cover every required repeated visual target axis"
        ),
    }


def non_regression_guard_report(
    target: dict[str, Any] | None,
    alignment: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(target, dict) or not target.get("active"):
        return {"enabled": False}
    return {
        "enabled": True,
        "axes": target.get("axis_keys") or [],
        "vision_target_stage": target.get("stage"),
        "pass": not bool((alignment or {}).get("opposite")),
        "direction": (alignment or {}).get("direction"),
        "axis_results": (alignment or {}).get("axis_results") or [],
        "reason": (
            "candidate moves opposite to repeated visual target"
            if bool((alignment or {}).get("opposite"))
            else "candidate does not regress against visual target axes"
        ),
    }
