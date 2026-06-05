from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from roi_image_edit.iterative_pipeline import (
    CandidateParams,
    RenderPlan,
    default_char_offsets,
    dedupe_params,
    mutate_params,
)
from roi_image_edit.local_validation import (
    alignment_vertical_penalty,
    gray_stroke_balance_penalty,
    local_neighbor_style_issues,
    local_outer_gray_halo_issues,
    local_stroke_body_issues,
    opacity_floor_for_excess_black,
    report_has_background_low_texture,
    report_has_background_white_ghost,
    report_has_excess_black_core,
    report_has_fine_strokes_too_soft,
    report_has_outer_gray_halo,
    report_needs_thinner_strokes,
    report_needs_wider_gray_strokes,
    stage_issues,
)
from roi_image_edit.roi_locator import max_font_size_for_plan, text_chars
from roi_image_edit.stage_patchers import (
    acceptance_blocking_stage,
    acceptance_reports_background_patch,
    acceptance_reports_too_dark_or_bold,
)
from roi_image_edit.stages import stage_gate_for_report


TEXT_SHAPE_GRID_ALLOWED_DELTA_KEYS = frozenset(
    {
        "font_name",
        "font_path",
        "font_size",
        "text_dx",
        "text_dy",
        "char_offsets",
        "stroke_opacity",
        "ink_gain",
        "alpha_contrast",
        "core_ink_gain",
        "core_darken_strength",
        "core_darken_threshold",
        "core_darken_target_gray",
    }
)
TEXT_SHAPE_GRID_BLOCKED_DELTA_KEYS = frozenset(
    {
        "opacity",
        "blur",
        "mask_threshold",
        "mask_dilate_iterations",
        "inpaint_radius",
        "photo_warp",
        "edge_breakup",
        "photo_noise",
        "jpeg_quality",
    }
)
TEXT_SHAPE_GRID_TOP_LIMIT = 48
TEXT_SHAPE_GRID_BUDGET_RANGE = (300, 1500)

INK_GRAY_GRID_ALLOWED_DELTA_KEYS = frozenset(
    {
        "opacity",
        "stroke_opacity",
        "ink_gain",
        "alpha_contrast",
        "core_ink_gain",
        "core_darken_strength",
        "core_darken_threshold",
        "core_darken_target_gray",
    }
)
INK_GRAY_GRID_BLOCKED_DELTA_KEYS = frozenset(
    {
        "font_name",
        "font_path",
        "font_size",
        "blur",
        "text_dx",
        "text_dy",
        "char_offsets",
        "mask_threshold",
        "mask_dilate_iterations",
        "inpaint_radius",
        "photo_warp",
        "edge_breakup",
        "photo_noise",
        "jpeg_quality",
    }
)
INK_GRAY_GRID_TOP_LIMIT = 16
INK_GRAY_GRID_BUDGET_RANGE = (100, 800)

PHOTO_TEXTURE_GRID_ALLOWED_DELTA_KEYS = frozenset(
    {
        "blur",
        "alpha_contrast",
        "photo_warp",
        "edge_breakup",
        "photo_noise",
        "jpeg_quality",
    }
)
PHOTO_TEXTURE_GRID_BLOCKED_DELTA_KEYS = frozenset(
    {
        "font_name",
        "font_path",
        "font_size",
        "opacity",
        "stroke_opacity",
        "ink_gain",
        "core_ink_gain",
        "core_darken_strength",
        "core_darken_threshold",
        "core_darken_target_gray",
        "text_dx",
        "text_dy",
        "char_offsets",
        "mask_threshold",
        "mask_dilate_iterations",
        "inpaint_radius",
    }
)
PHOTO_TEXTURE_GRID_TOP_LIMIT = 6
PHOTO_TEXTURE_GRID_BUDGET_RANGE = (30, 200)


@dataclass(frozen=True)
class ShapeCandidateGrid:
    candidates: list[CandidateParams]
    report: dict[str, Any]


@dataclass(frozen=True)
class InkGrayCandidateGrid:
    candidates: list[CandidateParams]
    report: dict[str, Any]


@dataclass(frozen=True)
class PhotoTextureCandidateGrid:
    candidates: list[CandidateParams]
    report: dict[str, Any]


def layered_candidate_search_report(*grid_reports: dict[str, Any]) -> dict[str, Any]:
    stages: list[dict[str, Any]] = []
    for report in grid_reports:
        if not isinstance(report, dict):
            continue
        budget = report.get("budget") if isinstance(report.get("budget"), dict) else {}
        stage_record = {
            "stage_id": report.get("stage_id"),
            "optimization_step": report.get("optimization_step"),
            "enabled": bool(report.get("enabled")),
            "candidate_count": int(report.get("candidate_count") or 0),
        }
        if budget:
            stage_record["raw_candidate_budget"] = int(budget.get("raw_candidate_budget") or 0)
            stage_record["retained_count"] = int(budget.get("retained_count") or 0)
            stage_record["pruned_count"] = int(budget.get("pruned_count") or 0)
            stage_record["within_budget"] = bool(budget.get("within_budget"))
        else:
            stage_record["reason"] = report.get("reason")
        stages.append(stage_record)

    return {
        "strategy": "layered_stage_search",
        "cross_stage_cartesian_search": False,
        "contract": "Generate and prune candidates within each blocking stage; do not multiply shape, ink-gray, and photo-texture axes into a single full Cartesian search.",
        "stage_order": [stage.get("stage_id") for stage in stages],
        "enabled_stage_ids": [
            stage.get("stage_id")
            for stage in stages
            if stage.get("enabled")
        ],
        "stages": stages,
        "raw_candidate_budget_by_stage": {
            str(stage.get("stage_id")): stage.get("raw_candidate_budget")
            for stage in stages
            if stage.get("enabled")
        },
        "retained_count_by_stage": {
            str(stage.get("stage_id")): stage.get("retained_count")
            for stage in stages
            if stage.get("enabled")
        },
        "pruned_count_by_stage": {
            str(stage.get("stage_id")): stage.get("pruned_count")
            for stage in stages
            if stage.get("enabled")
        },
    }


def final_acceptance_delivers(acceptance: dict[str, Any]) -> bool:
    final_level = str(acceptance.get("acceptance_level", "")).strip().lower()
    final_decision = str(acceptance.get("final_decision", "")).strip().lower()
    return bool(acceptance.get("pass")) and final_level == "pass" and final_decision == "deliver"

def revision_selection_score(
    score: float,
    params: CandidateParams,
    basis_params: CandidateParams,
    acceptance: dict[str, Any],
    report: dict[str, Any] | None = None,
    candidate_report: dict[str, Any] | None = None,
) -> float:
    adjusted = float(score)
    adjusted += (
        alignment_vertical_penalty(candidate_report)
        - alignment_vertical_penalty(report)
    ) * 3.0
    findings = acceptance.get("visual_findings") if isinstance(acceptance, dict) else {}
    if not isinstance(findings, dict):
        findings = {}

    stage_gate = stage_gate_for_report(report) if isinstance(report, dict) else {}
    if stage_gate.get("blocking_stage") == "text_shape":
        stroke_gain = max(0.0, float(params.stroke_opacity) - float(basis_params.stroke_opacity))
        ink_gain = max(0.0, float(params.ink_gain) - float(basis_params.ink_gain))
        blur_gain = max(0.0, float(params.blur) - float(basis_params.blur))
        core_gain = max(0.0, float(params.core_ink_gain) - float(basis_params.core_ink_gain))
        darken_gain = max(0.0, float(params.core_darken_strength) - float(basis_params.core_darken_strength))
        alpha_drop = max(0.0, float(basis_params.alpha_contrast) - float(params.alpha_contrast))
        size_gain = max(0, int(params.font_size) - int(basis_params.font_size))
        adjusted += gray_stroke_balance_penalty(candidate_report) * 5.0
        candidate_stage = stage_gate_for_report(candidate_report) if isinstance(candidate_report, dict) else {}
        if candidate_stage.get("blocking_stage") == "text_shape":
            adjusted += 1200.0
        elif candidate_stage.get("blocking_stage"):
            adjusted -= 900.0
        else:
            adjusted -= 1600.0
        body_issues = local_stroke_body_issues(candidate_report or {}, allow_excess_black_core=True)
        neighbor_issues = local_neighbor_style_issues(candidate_report or {}, allow_excess_black_core=True)
        halo_issues = local_outer_gray_halo_issues(candidate_report or {}, allow_excess_black_core=True)
        adjusted += len(body_issues) * 520.0
        adjusted += len(neighbor_issues) * 460.0
        adjusted += len(halo_issues) * 420.0
        adjusted += max(0.0, float(params.stroke_opacity) - 0.12) * 6000.0
        adjusted += max(0.0, float(params.core_ink_gain) - 0.30) * 3600.0
        adjusted += max(0.0, float(params.core_darken_strength) - 0.24) * 3200.0
        adjusted += max(0.0, float(params.photo_warp) - 0.12) * 4200.0
        adjusted += max(0.0, float(params.blur) - 0.30) * 1600.0
        adjusted -= min(stroke_gain, 0.08) * 3800.0
        adjusted -= min(blur_gain, 0.12) * 420.0
        adjusted -= min(alpha_drop, 0.20) * 260.0
        adjusted -= min(size_gain, 1) * 80.0
        adjusted -= min(ink_gain, 0.04) * 160.0
        adjusted -= min(core_gain, 0.08) * 220.0
        adjusted -= min(darken_gain, 0.08) * 180.0
        return adjusted

    if stage_gate.get("blocking_stage") == "photo_texture":
        candidate_stage = stage_gate_for_report(candidate_report) if isinstance(candidate_report, dict) else {}
        if candidate_stage.get("blocking_stage") == "photo_texture":
            adjusted += 600.0
        elif candidate_stage.get("blocking_stage"):
            adjusted -= 260.0
        else:
            adjusted -= 900.0
        blur_change = abs(float(params.blur) - float(basis_params.blur))
        noise_change = abs(float(params.photo_noise) - float(basis_params.photo_noise))
        edge_change = abs(float(params.edge_breakup) - float(basis_params.edge_breakup))
        warp_change = abs(float(params.photo_warp) - float(basis_params.photo_warp))
        jpeg_change = abs(int(params.jpeg_quality) - int(basis_params.jpeg_quality))
        adjusted -= min(blur_change, 0.12) * 220.0
        adjusted -= min(noise_change, 0.025) * 1800.0
        adjusted -= min(edge_change, 0.010) * 1400.0
        adjusted -= min(warp_change, 0.030) * 280.0
        adjusted -= min(jpeg_change, 8) * 8.0
        return adjusted

    if report_has_excess_black_core(report):
        candidate_stage = stage_gate_for_report(candidate_report) if isinstance(candidate_report, dict) else {}
        candidate_block = candidate_stage.get("blocking_stage")
        if candidate_block == "ink_gray_balance":
            adjusted += 480.0
        elif candidate_block:
            adjusted += 320.0
        else:
            adjusted -= 900.0
        opacity_drop = max(0.0, float(basis_params.opacity) - float(params.opacity))
        stroke_drop = max(0.0, float(basis_params.stroke_opacity) - float(params.stroke_opacity))
        ink_drop = max(0.0, float(basis_params.ink_gain) - float(params.ink_gain))
        alpha_drop = max(0.0, float(basis_params.alpha_contrast) - float(params.alpha_contrast))
        core_drop = max(0.0, float(basis_params.core_ink_gain) - float(params.core_ink_gain))
        darken_drop = max(0.0, float(basis_params.core_darken_strength) - float(params.core_darken_strength))
        blur_increase = max(0.0, float(params.blur) - float(basis_params.blur))
        adjusted -= opacity_drop * 980.0
        adjusted -= stroke_drop * 1300.0
        adjusted -= ink_drop * 240.0
        adjusted -= alpha_drop * 260.0
        adjusted -= core_drop * 180.0
        adjusted -= darken_drop * 160.0
        adjusted -= blur_increase * 90.0
        return adjusted

    darkness = str(findings.get("darkness", "")).strip().lower()
    stroke_weight = str(findings.get("stroke_weight", "")).strip().lower()
    if darkness == "too_dark" or stroke_weight in {"too_bold", "slightly_bold"}:
        opacity_drop = max(0.0, float(basis_params.opacity) - float(params.opacity))
        ink_drop = max(0.0, float(basis_params.ink_gain) - float(params.ink_gain))
        core_drop = max(0.0, float(basis_params.core_ink_gain) - float(params.core_ink_gain))
        blur_increase = max(0.0, float(params.blur) - float(basis_params.blur))
        adjusted -= opacity_drop * 320.0
        adjusted -= ink_drop * 160.0
        adjusted -= core_drop * 120.0
        adjusted -= blur_increase * 70.0
        return adjusted

    if report_needs_wider_gray_strokes(report):
        opacity_drop = max(0.0, float(basis_params.opacity) - float(params.opacity))
        stroke_gain = max(0.0, float(params.stroke_opacity) - float(basis_params.stroke_opacity))
        stroke_drop = max(0.0, float(basis_params.stroke_opacity) - float(params.stroke_opacity))
        ink_gain = max(0.0, float(params.ink_gain) - float(basis_params.ink_gain))
        core_gain = max(0.0, float(params.core_ink_gain) - float(basis_params.core_ink_gain))
        darken_gain = max(0.0, float(params.core_darken_strength) - float(basis_params.core_darken_strength))
        blur_drop = max(0.0, float(basis_params.blur) - float(params.blur))
        blur_gain = max(0.0, float(params.blur) - float(basis_params.blur))
        alpha_gain = max(0.0, float(params.alpha_contrast) - float(basis_params.alpha_contrast))
        size_gain = max(0, int(params.font_size) - int(basis_params.font_size))
        photo_gain = max(0.0, float(params.photo_noise) - float(basis_params.photo_noise))
        photo_drop = max(0.0, float(basis_params.photo_noise) - float(params.photo_noise))
        edge_drop = max(0.0, float(basis_params.edge_breakup) - float(params.edge_breakup))
        neighbor_style_issue = bool(local_neighbor_style_issues(report))
        if report_has_outer_gray_halo(report):
            if local_outer_gray_halo_issues(candidate_report, allow_excess_black_core=True):
                adjusted += 260.0
            if local_stroke_body_issues(candidate_report):
                adjusted += 140.0
            adjusted -= blur_drop * 520.0
            adjusted -= stroke_drop * 2300.0
            adjusted -= photo_drop * 1100.0
            adjusted -= edge_drop * 1400.0
            adjusted -= alpha_gain * 260.0
            adjusted -= core_gain * 180.0
            adjusted -= darken_gain * 160.0
            adjusted += blur_gain * 520.0
            adjusted += photo_gain * 900.0
            adjusted += stroke_gain * 1500.0
            adjusted += opacity_drop * 600.0
            return adjusted
        adjusted += opacity_drop * 900.0
        adjusted += gray_stroke_balance_penalty(candidate_report) * 6.0
        if local_stroke_body_issues(candidate_report):
            adjusted += 160.0
        if neighbor_style_issue and local_neighbor_style_issues(candidate_report):
            adjusted += 190.0
        if stroke_gain <= 0.0:
            adjusted += 190.0
        if stroke_gain <= 0.0 and ink_gain <= 0.0 and blur_gain <= 0.0 and photo_gain <= 0.0:
            adjusted += 240.0
        adjusted -= min(stroke_gain, 0.07) * 4600.0
        adjusted += max(0.0, stroke_gain - 0.08) * 1400.0
        adjusted -= blur_gain * 95.0
        adjusted -= ink_gain * 150.0
        adjusted -= photo_gain * 170.0
        if neighbor_style_issue:
            adjusted -= core_gain * 420.0
            adjusted -= darken_gain * 320.0
            adjusted += blur_gain * 180.0
            adjusted += photo_gain * 160.0
        adjusted -= min(size_gain, 1) * 70.0
        return adjusted

    if not report_needs_thinner_strokes(report):
        return adjusted
    if acceptance_reports_too_dark_or_bold(acceptance):
        opacity_drop = max(0.0, float(basis_params.opacity) - float(params.opacity))
        opacity_raise = max(0.0, float(params.opacity) - float(basis_params.opacity))
        blur_increase = max(0.0, float(params.blur) - float(basis_params.blur))
        alpha_drop = max(0.0, float(basis_params.alpha_contrast) - float(params.alpha_contrast))
        core_drop = max(0.0, float(basis_params.core_ink_gain) - float(params.core_ink_gain))
        darken_drop = max(0.0, float(basis_params.core_darken_strength) - float(params.core_darken_strength))

        adjusted -= opacity_drop * 1800.0
        adjusted -= blur_increase * 260.0
        adjusted -= alpha_drop * 320.0
        adjusted -= core_drop * 180.0
        adjusted -= darken_drop * 160.0
        adjusted += opacity_raise * 2600.0
        return adjusted

    opacity_drop = max(0.0, float(basis_params.opacity) - float(params.opacity))
    blur_increase = max(0.0, float(params.blur) - float(basis_params.blur))
    alpha_contrast_gain = max(0.0, float(params.alpha_contrast) - float(basis_params.alpha_contrast))
    font_size_drop = max(0, int(basis_params.font_size) - int(params.font_size))
    threshold_gain = max(0, int(params.core_darken_threshold) - int(basis_params.core_darken_threshold))

    adjusted += opacity_drop * 1500.0
    adjusted += blur_increase * 260.0
    adjusted -= alpha_contrast_gain * 180.0
    adjusted -= min(font_size_drop, 1) * 55.0
    adjusted -= min(threshold_gain, 20) * 2.5
    return adjusted


def constrained_revision_params(
    params: CandidateParams,
    basis_params: CandidateParams,
    acceptance: dict[str, Any],
    report: dict[str, Any] | None = None,
    *,
    round_idx: int,
) -> CandidateParams:
    stage_gate = stage_gate_for_report(report) if isinstance(report, dict) else {}
    if stage_gate.get("blocking_stage") == "text_shape":
        has_outer_halo = report_has_outer_gray_halo(report)
        stroke_cap = 0.10 if has_outer_halo else 0.14
        blur_cap = 0.22 if has_outer_halo else 0.30
        return mutate_params(
            params,
            opacity=max(0.82, min(1.0, params.opacity)),
            blur=max(0.08, min(blur_cap, params.blur)),
            stroke_opacity=min(stroke_cap, params.stroke_opacity),
            alpha_contrast=min(0.35, params.alpha_contrast),
            photo_warp=min(0.12, params.photo_warp),
            edge_breakup=min(0.012, params.edge_breakup),
            photo_noise=min(0.030, params.photo_noise),
            core_ink_gain=min(0.30, params.core_ink_gain),
            core_darken_strength=min(0.24, params.core_darken_strength),
            core_darken_threshold=min(150, params.core_darken_threshold),
        )

    if report_has_excess_black_core(report):
        opacity_floor = opacity_floor_for_excess_black(report)
        return mutate_params(
            params,
            opacity=max(opacity_floor, min(1.0, params.opacity)),
            blur=max(0.18, min(0.65, params.blur)),
            stroke_opacity=min(0.06, params.stroke_opacity),
            alpha_contrast=min(0.35, params.alpha_contrast),
            core_ink_gain=min(0.22, params.core_ink_gain),
            core_darken_strength=min(0.18, params.core_darken_strength),
        )

    if report_has_background_white_ghost(report):
        return mutate_params(
            params,
            photo_noise=max(0.0, min(0.038, params.photo_noise)),
            edge_breakup=max(0.0, min(0.014, params.edge_breakup)),
            jpeg_quality=max(94, params.jpeg_quality),
            mask_threshold=max(params.mask_threshold, min(215, basis_params.mask_threshold + 12)),
            mask_dilate_iterations=max(3, min(5, params.mask_dilate_iterations)),
            inpaint_radius=max(2, min(3, params.inpaint_radius)),
        )

    if report_has_background_low_texture(report):
        return mutate_params(
            params,
            photo_noise=min(0.14, params.photo_noise),
            edge_breakup=min(0.060, params.edge_breakup),
            jpeg_quality=max(82, params.jpeg_quality),
            mask_dilate_iterations=max(2, params.mask_dilate_iterations),
            inpaint_radius=max(1, min(3, params.inpaint_radius)),
        )

    if acceptance_reports_background_patch(acceptance):
        return mutate_params(
            params,
            photo_noise=min(0.120, params.photo_noise),
            edge_breakup=min(0.050, params.edge_breakup),
            jpeg_quality=max(82, params.jpeg_quality),
            mask_dilate_iterations=max(2, params.mask_dilate_iterations),
            inpaint_radius=max(1, min(3, params.inpaint_radius)),
        )

    if not report_needs_thinner_strokes(report):
        if not report_needs_wider_gray_strokes(report):
            return params
        if report_has_outer_gray_halo(report):
            return mutate_params(
                params,
                opacity=max(max(0.82, basis_params.opacity - 0.02), params.opacity),
                blur=min(params.blur, max(0.08, basis_params.blur - 0.03)),
                stroke_opacity=min(params.stroke_opacity, max(0.0, basis_params.stroke_opacity - 0.01)),
                alpha_contrast=min(0.45, params.alpha_contrast),
                photo_warp=min(params.photo_warp, max(0.0, basis_params.photo_warp - 0.02)),
                edge_breakup=min(params.edge_breakup, max(0.0, basis_params.edge_breakup - 0.006)),
                photo_noise=min(params.photo_noise, max(0.0, basis_params.photo_noise - 0.018)),
                core_ink_gain=max(params.core_ink_gain, basis_params.core_ink_gain),
                core_darken_strength=max(params.core_darken_strength, basis_params.core_darken_strength),
            )
        if local_neighbor_style_issues(report):
            return mutate_params(
                params,
                opacity=max(max(0.82, basis_params.opacity - 0.04), params.opacity),
                blur=max(max(0.14, basis_params.blur - 0.08), params.blur),
                stroke_opacity=max(min(0.16, basis_params.stroke_opacity + 0.02), params.stroke_opacity),
                alpha_contrast=min(0.45, params.alpha_contrast),
                photo_warp=max(max(0.0, basis_params.photo_warp - 0.04), params.photo_warp),
                edge_breakup=max(max(0.0, basis_params.edge_breakup - 0.010), params.edge_breakup),
                photo_noise=max(max(0.0, basis_params.photo_noise - 0.020), params.photo_noise),
                core_ink_gain=min(0.34, params.core_ink_gain),
                core_darken_strength=min(0.30, params.core_darken_strength),
            )
        if report_has_fine_strokes_too_soft(report):
            return mutate_params(
                params,
                opacity=max(max(0.82, basis_params.opacity - 0.05), params.opacity),
                blur=max(max(0.18, basis_params.blur - 0.08), params.blur),
                stroke_opacity=max(min(0.16, basis_params.stroke_opacity + 0.03), params.stroke_opacity),
                alpha_contrast=min(0.45, params.alpha_contrast),
                photo_warp=max(max(0.0, basis_params.photo_warp - 0.04), params.photo_warp),
                edge_breakup=max(max(0.0, basis_params.edge_breakup - 0.010), params.edge_breakup),
                photo_noise=max(max(0.0, basis_params.photo_noise - 0.020), params.photo_noise),
                core_ink_gain=min(max(0.0, basis_params.core_ink_gain), params.core_ink_gain),
                core_darken_strength=min(max(0.0, basis_params.core_darken_strength), params.core_darken_strength),
            )
        return mutate_params(
            params,
            opacity=max(max(0.82, basis_params.opacity - 0.04), params.opacity),
            blur=max(max(0.12, basis_params.blur), params.blur),
            alpha_contrast=min(0.45, params.alpha_contrast),
            core_ink_gain=min(max(0.0, basis_params.core_ink_gain), params.core_ink_gain),
            core_darken_strength=min(max(0.0, basis_params.core_darken_strength), params.core_darken_strength),
        )

    alpha_cap = 0.35
    threshold_cap = 166
    target_gray_floor = 20
    blur_floor = 0.08
    opacity_floor = 0.92
    core_darken_cap = 0.46
    if acceptance_reports_too_dark_or_bold(acceptance):
        return mutate_params(
            params,
            opacity=max(0.70, min(basis_params.opacity, params.opacity)),
            blur=max(0.08, min(0.65, params.blur)),
            stroke_opacity=min(basis_params.stroke_opacity, params.stroke_opacity),
            alpha_contrast=min(basis_params.alpha_contrast, params.alpha_contrast),
            core_ink_gain=min(basis_params.core_ink_gain, params.core_ink_gain),
            core_darken_strength=min(basis_params.core_darken_strength, params.core_darken_strength),
        )

    if round_idx >= 3:
        # Once the stroke footprint has been narrowed, later rounds may only
        # soften or tone it; they must not keep hardening the glyph core.
        alpha_cap = min(alpha_cap, max(0.20, basis_params.alpha_contrast))
        threshold_cap = min(threshold_cap, max(140, basis_params.core_darken_threshold + 4))

    return mutate_params(
        params,
        opacity=max(opacity_floor, params.opacity),
        blur=max(blur_floor, params.blur),
        alpha_contrast=min(alpha_cap, params.alpha_contrast),
        core_darken_strength=min(core_darken_cap, params.core_darken_strength),
        core_darken_threshold=min(threshold_cap, params.core_darken_threshold),
        core_darken_target_gray=max(target_gray_floor, params.core_darken_target_gray),
    )





def report_blocks_text_shape(report: dict[str, Any] | None) -> bool:
    if not isinstance(report, dict):
        return False
    stage_gate = report.get("stage_gate")
    if not isinstance(stage_gate, dict):
        stage_gate = stage_gate_for_report(report)
    return stage_gate.get("blocking_stage") == "text_shape"


def shape_font_items(
    params: CandidateParams,
    font_style_reference: dict[str, Any],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    ranked_fonts = font_style_reference.get("ranked_fonts", [])
    ranked = [item for item in ranked_fonts if isinstance(item, dict)]
    selected: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    def add_item(item: dict[str, Any]) -> None:
        font_path = str(item.get("font_path") or "")
        if not font_path or font_path in seen_paths:
            return
        selected.append(item)
        seen_paths.add(font_path)

    preferred_order = ("Songti", "GBSN", "SimSun", "FangSong", "NotoSerif", "UMing")
    for preferred_name in preferred_order:
        for item in ranked:
            if str(item.get("font_name") or "") == preferred_name:
                add_item(item)
                break
        if len(selected) >= limit:
            break

    for item in ranked:
        if len(selected) >= limit:
            break
        add_item(item)

    add_item(
        {
            "font_name": params.font_name,
            "font_path": params.font_path,
            "font_size": params.font_size,
        }
    )
    return selected[:limit]


def normalized_offset_candidates(plan: RenderPlan, params: CandidateParams) -> tuple[tuple[tuple[int, int], ...], ...]:
    target_count = len(text_chars(plan.target_text))
    if target_count <= 0 or plan.draw_mode == "center":
        return ((),)

    candidates: list[tuple[tuple[int, int], ...]] = []

    def add_offsets(value: Any) -> None:
        if not value:
            return
        try:
            offsets = tuple((int(item[0]), int(item[1])) for item in value)
        except (TypeError, ValueError, IndexError):
            return
        if len(offsets) != target_count:
            return
        if offsets not in candidates:
            candidates.append(offsets)

    add_offsets(params.char_offsets)
    add_offsets(default_char_offsets(plan.target_text))
    add_offsets(tuple((0, 0) for _ in range(target_count)))
    if target_count == 2:
        add_offsets(((3, 0), (-2, 1)))
        add_offsets(((4, 0), (-2, 1)))
        add_offsets(((5, 0), (-1, 0)))
    if not candidates:
        candidates.append(default_char_offsets(plan.target_text))
    return tuple(candidates[:3])


def params_delta_keys(base: CandidateParams, candidate: CandidateParams) -> frozenset[str]:
    base_data = asdict(base)
    candidate_data = asdict(candidate)
    changed = {
        key
        for key, value in candidate_data.items()
        if key != "candidate_id" and value != base_data.get(key)
    }
    return frozenset(changed)


def shape_issue_flags(report: dict[str, Any] | None) -> dict[str, bool]:
    shape_issues = stage_issues(report, "text_shape")
    has_outer_halo = any(
        str(issue.get("type") or "") == "changed_char_neighbor_outer_gray_halo_too_high"
        for issue in shape_issues
        if isinstance(issue, dict)
    )
    has_body_gap = any(
        "stroke_body" in str(issue.get("type") or "")
        or "fine_strokes" in str(issue.get("type") or "")
        for issue in shape_issues
        if isinstance(issue, dict)
    )
    has_pose_gap = any(
        "pose" in str(issue.get("type") or "") or "shear" in str(issue.get("type") or "")
        for issue in shape_issues
        if isinstance(issue, dict)
    )
    return {
        "outer_halo": has_outer_halo,
        "body_gap": has_body_gap,
        "pose_gap": has_pose_gap,
    }


def shape_stroke_body_grid(flags: dict[str, bool]) -> tuple[tuple[float, float, float, float, float], ...]:
    if flags.get("outer_halo"):
        return (
            (0.02, 0.02, 0.16, 0.12, 0.10),
            (0.04, 0.02, 0.14, 0.14, 0.12),
            (0.04, 0.03, 0.12, 0.16, 0.12),
            (0.06, 0.01, 0.20, 0.12, 0.10),
        )
    if flags.get("body_gap"):
        return (
            (0.03, 0.00, 0.18, 0.00, 0.00),
            (0.04, 0.00, 0.16, 0.04, 0.04),
            (0.04, 0.01, 0.14, 0.08, 0.06),
            (0.04, 0.04, 0.10, 0.18, 0.14),
        )
    return (
        (0.04, 0.03, 0.12, 0.16, 0.12),
        (0.04, 0.03, 0.10, 0.18, 0.14),
        (0.06, 0.02, 0.16, 0.14, 0.12),
    )


def text_shape_reset_candidate_grid(
    params: CandidateParams,
    font_style_reference: dict[str, Any],
    plan: RenderPlan,
    report: dict[str, Any] | None,
    *,
    limit: int = TEXT_SHAPE_GRID_TOP_LIMIT,
) -> ShapeCandidateGrid:
    if not report_blocks_text_shape(report):
        return ShapeCandidateGrid(
            candidates=[],
            report={
                "enabled": False,
                "reason": "text_shape_not_blocking",
                "stage_id": "text_shape",
                "candidate_count": 0,
            },
        )

    flags = shape_issue_flags(report)
    stroke_body_grid = shape_stroke_body_grid(flags)
    size_deltas = (0, -1, 1)
    if flags.get("body_gap"):
        size_deltas = (0, 1, -1)
    if flags.get("pose_gap"):
        size_deltas = tuple(dict.fromkeys(size_deltas + (0,)))

    max_font_size = max_font_size_for_plan(plan)
    font_items = shape_font_items(params, font_style_reference, limit=4)
    offset_candidates = normalized_offset_candidates(plan, params)
    text_dx_candidates = tuple(dict.fromkeys((params.text_dx, 0, -1, 1)))[:3]
    text_dy_candidates = tuple(dict.fromkeys((params.text_dy, 0, -1, 1)))[:3]
    raw_budget = (
        len(font_items)
        * len(size_deltas)
        * len(offset_candidates)
        * len(text_dx_candidates)
        * len(text_dy_candidates)
        * len(stroke_body_grid)
    )

    variants: list[CandidateParams] = []
    for font_item in font_items:
        font_name = str(font_item.get("font_name") or params.font_name)
        font_path = str(font_item.get("font_path") or params.font_path)
        try:
            base_size = int(font_item.get("font_size") or params.font_size)
        except (TypeError, ValueError):
            base_size = params.font_size
        for size_delta in size_deltas:
            font_size = max(8, min(max_font_size, base_size + int(size_delta)))
            for offsets in offset_candidates:
                for text_dx in text_dx_candidates:
                    for text_dy in text_dy_candidates:
                        for (
                            stroke_opacity,
                            ink_gain,
                            alpha_contrast,
                            core_ink_gain,
                            core_darken_strength,
                        ) in stroke_body_grid:
                            variants.append(
                                mutate_params(
                                    params,
                                    font_name=font_name,
                                    font_path=font_path,
                                    font_size=font_size,
                                    stroke_opacity=stroke_opacity,
                                    ink_gain=ink_gain,
                                    alpha_contrast=alpha_contrast,
                                    core_ink_gain=core_ink_gain,
                                    core_darken_strength=core_darken_strength,
                                    core_darken_threshold=130,
                                    core_darken_target_gray=28,
                                    text_dx=text_dx,
                                    text_dy=text_dy,
                                    char_offsets=offsets,
                                    opacity=params.opacity,
                                    blur=params.blur,
                                    mask_threshold=params.mask_threshold,
                                    mask_dilate_iterations=params.mask_dilate_iterations,
                                    inpaint_radius=params.inpaint_radius,
                                    photo_warp=params.photo_warp,
                                    edge_breakup=params.edge_breakup,
                                    photo_noise=params.photo_noise,
                                    jpeg_quality=params.jpeg_quality,
                                )
                            )
    candidates = dedupe_params(variants, min(limit, TEXT_SHAPE_GRID_TOP_LIMIT))
    candidate_records: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    for candidate in candidates:
        delta_keys = params_delta_keys(params, candidate)
        blocked_delta_keys = sorted(delta_keys & TEXT_SHAPE_GRID_BLOCKED_DELTA_KEYS)
        undeclared_delta_keys = sorted(delta_keys - TEXT_SHAPE_GRID_ALLOWED_DELTA_KEYS)
        record = {
            "candidate_id": candidate.candidate_id,
            "delta_keys": sorted(delta_keys),
            "allowed_delta_keys_only": not blocked_delta_keys and not undeclared_delta_keys,
            "blocked_delta_keys": blocked_delta_keys,
            "undeclared_delta_keys": undeclared_delta_keys,
        }
        candidate_records.append(record)
        if blocked_delta_keys or undeclared_delta_keys:
            violations.append(record)

    budget_min, budget_max = TEXT_SHAPE_GRID_BUDGET_RANGE
    report_payload = {
        "enabled": True,
        "stage_id": "text_shape",
        "optimization_step": "shape_reset",
        "budget": {
            "raw_candidate_budget": raw_budget,
            "budget_min": budget_min,
            "budget_max": budget_max,
            "within_budget": budget_min <= raw_budget <= budget_max,
            "retained_top_limit": min(limit, TEXT_SHAPE_GRID_TOP_LIMIT),
            "retained_count": len(candidates),
            "pruned_count": max(0, raw_budget - len(candidates)),
        },
        "axes": {
            "font_count": len(font_items),
            "font_size_delta_count": len(size_deltas),
            "placement_strategy": plan.placement_strategy,
            "placement_strategy_reason": plan.placement_strategy_reason,
            "text_dx_count": len(text_dx_candidates),
            "text_dy_count": len(text_dy_candidates),
            "char_offsets_count": len(offset_candidates),
            "stroke_body_grid_count": len(stroke_body_grid),
            "pose_shear_source": "renderer_reference_slot_shear_from_source_slots_and_neighbors",
        },
        "issue_flags": flags,
        "allowed_delta_keys": sorted(TEXT_SHAPE_GRID_ALLOWED_DELTA_KEYS),
        "blocked_delta_keys": sorted(TEXT_SHAPE_GRID_BLOCKED_DELTA_KEYS),
        "candidate_count": len(candidates),
        "candidate_delta_audit": candidate_records,
        "violations": violations,
    }
    return ShapeCandidateGrid(candidates=candidates, report=report_payload)


def text_shape_reset_candidates(
    params: CandidateParams,
    font_style_reference: dict[str, Any],
    plan: RenderPlan,
    report: dict[str, Any] | None,
    *,
    limit: int = TEXT_SHAPE_GRID_TOP_LIMIT,
) -> list[CandidateParams]:
    return text_shape_reset_candidate_grid(
        params,
        font_style_reference,
        plan,
        report,
        limit=limit,
    ).candidates


def report_blocks_ink_gray(report: dict[str, Any] | None) -> bool:
    if not isinstance(report, dict):
        return False
    stage_gate = report.get("stage_gate")
    if not isinstance(stage_gate, dict):
        stage_gate = stage_gate_for_report(report)
    return stage_gate.get("blocking_stage") == "ink_gray_balance"


def _float_axis(
    base: float,
    deltas: tuple[float, ...],
    *,
    low: float,
    high: float,
    target_len: int,
) -> tuple[float, ...]:
    values: list[float] = []

    def add(value: float) -> None:
        bounded = round(max(low, min(high, float(value))), 3)
        if bounded not in values:
            values.append(bounded)

    for delta in deltas:
        add(base + delta)
    step = 0.01
    probe = 1
    while len(values) < target_len and probe <= 80:
        add(base + probe * step)
        if len(values) >= target_len:
            break
        add(base - probe * step)
        probe += 1
    return tuple(values[:target_len])


def _int_axis(
    base: int,
    deltas: tuple[int, ...],
    *,
    low: int,
    high: int,
    target_len: int,
) -> tuple[int, ...]:
    values: list[int] = []

    def add(value: int) -> None:
        bounded = max(low, min(high, int(value)))
        if bounded not in values:
            values.append(bounded)

    for delta in deltas:
        add(base + delta)
    probe = 1
    while len(values) < target_len and probe <= 80:
        add(base + probe)
        if len(values) >= target_len:
            break
        add(base - probe)
        probe += 1
    return tuple(values[:target_len])


def ink_gray_issue_flags(report: dict[str, Any] | None) -> dict[str, bool]:
    issue_types = {
        str(issue.get("type") or "")
        for issue in stage_issues(report, "ink_gray_balance")
        if isinstance(issue, dict)
    }
    return {
        "excess_black_core": report_has_excess_black_core(report),
        "outer_gray_halo": report_has_outer_gray_halo(report),
        "needs_wider_gray_strokes": report_needs_wider_gray_strokes(report),
        "needs_thinner_strokes": report_needs_thinner_strokes(report),
        "fine_strokes_too_soft": report_has_fine_strokes_too_soft(report),
        "core_too_light": bool(
            issue_types
            & {
                "core_mean_gray_too_light",
                "core_lighten_too_high",
                "changed_char_core_too_light",
                "ink_too_light",
            }
        ),
    }


def ink_gray_axes(params: CandidateParams, report: dict[str, Any] | None) -> dict[str, tuple[Any, ...]]:
    flags = ink_gray_issue_flags(report)
    if flags["excess_black_core"] or flags["needs_thinner_strokes"]:
        opacity_deltas = (0.0, -0.02, -0.04, -0.06)
        stroke_deltas = (0.0, -0.02, -0.04, 0.01)
        ink_deltas = (0.0, -0.02)
        alpha_deltas = (0.0, -0.04)
        core_deltas = ((0.0, 0.0, 0, 0), (-0.04, -0.04, -6, 4), (-0.08, -0.06, -10, 8), (-0.02, -0.08, -4, 10))
    elif flags["outer_gray_halo"]:
        opacity_deltas = (0.0, 0.01, -0.02, 0.02)
        stroke_deltas = (0.0, -0.01, -0.03, 0.01)
        ink_deltas = (0.0, 0.01)
        alpha_deltas = (0.0, -0.04)
        core_deltas = ((0.0, 0.0, 0, 0), (0.02, 0.02, 3, -2), (-0.02, -0.02, -4, 4), (0.04, 0.01, 6, -4))
    else:
        opacity_deltas = (0.0, 0.02, 0.04, -0.02)
        stroke_deltas = (0.0, 0.02, 0.04, -0.01)
        ink_deltas = (0.0, 0.02)
        alpha_deltas = (0.0, 0.04)
        core_deltas = ((0.0, 0.0, 0, 0), (0.04, 0.04, 6, -4), (0.08, 0.06, 10, -8), (0.02, 0.08, 4, -10))

    core_axis = tuple(
        (
            _float_axis(params.core_ink_gain, (core_delta,), low=0.0, high=1.0, target_len=1)[0],
            _float_axis(params.core_darken_strength, (darken_delta,), low=0.0, high=1.0, target_len=1)[0],
            _int_axis(params.core_darken_threshold, (threshold_delta,), low=0, high=254, target_len=1)[0],
            _int_axis(params.core_darken_target_gray, (target_delta,), low=0, high=120, target_len=1)[0],
        )
        for core_delta, darken_delta, threshold_delta, target_delta in core_deltas
    )
    return {
        "opacity": _float_axis(params.opacity, opacity_deltas, low=0.2, high=1.0, target_len=4),
        "stroke_opacity": _float_axis(params.stroke_opacity, stroke_deltas, low=0.0, high=1.0, target_len=4),
        "ink_gain": _float_axis(params.ink_gain, ink_deltas, low=0.0, high=1.0, target_len=2),
        "alpha_contrast": _float_axis(params.alpha_contrast, alpha_deltas, low=0.0, high=2.0, target_len=2),
        "core_tone": core_axis,
    }


def ink_gray_candidate_grid(
    params: CandidateParams,
    report: dict[str, Any] | None,
    *,
    limit: int = INK_GRAY_GRID_TOP_LIMIT,
) -> InkGrayCandidateGrid:
    if not report_blocks_ink_gray(report):
        return InkGrayCandidateGrid(
            candidates=[],
            report={
                "enabled": False,
                "reason": "ink_gray_balance_not_blocking",
                "stage_id": "ink_gray_balance",
                "candidate_count": 0,
            },
        )

    axes = ink_gray_axes(params, report)
    opacity_values = axes["opacity"]
    stroke_values = axes["stroke_opacity"]
    ink_values = axes["ink_gain"]
    alpha_values = axes["alpha_contrast"]
    core_values = axes["core_tone"]
    raw_budget = (
        len(opacity_values)
        * len(stroke_values)
        * len(ink_values)
        * len(alpha_values)
        * len(core_values)
    )

    variants: list[CandidateParams] = []
    for opacity in opacity_values:
        for stroke_opacity in stroke_values:
            for ink_gain in ink_values:
                for alpha_contrast in alpha_values:
                    for (
                        core_ink_gain,
                        core_darken_strength,
                        core_darken_threshold,
                        core_darken_target_gray,
                    ) in core_values:
                        variants.append(
                            mutate_params(
                                params,
                                opacity=opacity,
                                stroke_opacity=stroke_opacity,
                                ink_gain=ink_gain,
                                alpha_contrast=alpha_contrast,
                                core_ink_gain=core_ink_gain,
                                core_darken_strength=core_darken_strength,
                                core_darken_threshold=core_darken_threshold,
                                core_darken_target_gray=core_darken_target_gray,
                            )
                        )
    retained_limit = min(limit, INK_GRAY_GRID_TOP_LIMIT)
    candidates = dedupe_params(variants, retained_limit)
    candidate_records: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    for candidate in candidates:
        delta_keys = params_delta_keys(params, candidate)
        blocked_delta_keys = sorted(delta_keys & INK_GRAY_GRID_BLOCKED_DELTA_KEYS)
        undeclared_delta_keys = sorted(delta_keys - INK_GRAY_GRID_ALLOWED_DELTA_KEYS)
        record = {
            "candidate_id": candidate.candidate_id,
            "parent_candidate_id": params.candidate_id,
            "delta_keys": sorted(delta_keys),
            "allowed_delta_keys_only": not blocked_delta_keys and not undeclared_delta_keys,
            "blocked_delta_keys": blocked_delta_keys,
            "undeclared_delta_keys": undeclared_delta_keys,
        }
        candidate_records.append(record)
        if blocked_delta_keys or undeclared_delta_keys:
            violations.append(record)

    budget_min, budget_max = INK_GRAY_GRID_BUDGET_RANGE
    return InkGrayCandidateGrid(
        candidates=candidates,
        report={
            "enabled": True,
            "stage_id": "ink_gray_balance",
            "optimization_step": "ink_gray_balance",
            "parent_candidate_id": params.candidate_id,
            "parent_shape_candidate_id": params.candidate_id,
            "budget": {
                "raw_candidate_budget": raw_budget,
                "budget_min": budget_min,
                "budget_max": budget_max,
                "within_budget": budget_min <= raw_budget <= budget_max,
                "retained_top_limit": retained_limit,
                "retained_count": len(candidates),
                "pruned_count": max(0, raw_budget - len(candidates)),
            },
            "axes": {
                "opacity_count": len(opacity_values),
                "stroke_opacity_count": len(stroke_values),
                "ink_gain_count": len(ink_values),
                "alpha_contrast_count": len(alpha_values),
                "core_tone_count": len(core_values),
                "ranking_method": "local_issue_ordered_axis_priority",
            },
            "issue_flags": ink_gray_issue_flags(report),
            "allowed_delta_keys": sorted(INK_GRAY_GRID_ALLOWED_DELTA_KEYS),
            "blocked_delta_keys": sorted(INK_GRAY_GRID_BLOCKED_DELTA_KEYS),
            "preserved_shape_keys": [
                "font_name",
                "font_path",
                "font_size",
                "text_dx",
                "text_dy",
                "char_offsets",
            ],
            "shape_key_changes_require_stage": "text_shape",
            "candidate_count": len(candidates),
            "candidate_delta_audit": candidate_records,
            "violations": violations,
        },
    )


def report_blocks_photo_texture(report: dict[str, Any] | None) -> bool:
    if not isinstance(report, dict):
        return False
    stage_gate = report.get("stage_gate")
    if not isinstance(stage_gate, dict):
        stage_gate = stage_gate_for_report(report)
    return stage_gate.get("blocking_stage") == "photo_texture"


def photo_texture_issue_flags(report: dict[str, Any] | None) -> dict[str, bool]:
    issue_types = {
        str(issue.get("type") or "")
        for issue in stage_issues(report, "photo_texture")
        if isinstance(issue, dict)
    }
    return {
        "too_sharp": "photo_texture_too_sharp" in issue_types,
        "too_clean": "photo_texture_too_clean" in issue_types,
        "too_blurry": "photo_texture_too_blurry" in issue_types,
        "edge_breakup_missing": "photo_texture_edge_breakup_missing" in issue_types,
        "not_applied": "photo_texture_not_applied" in issue_types,
    }


def _jpeg_axis(base: int, deltas: tuple[int, ...], *, target_len: int) -> tuple[int, ...]:
    start = int(base or 94)
    return _int_axis(start, deltas, low=35, high=99, target_len=target_len)


def photo_texture_axes(params: CandidateParams, report: dict[str, Any] | None) -> dict[str, tuple[Any, ...]]:
    flags = photo_texture_issue_flags(report)
    if flags["too_blurry"]:
        blur_deltas = (0.0, -0.06, -0.10)
        alpha_deltas = (0.0, 0.02)
        warp_deltas = (0.0, -0.02)
        breakup_deltas = (0.0, -0.004, 0.004)
        noise_deltas = (0.0, -0.008, -0.014)
        jpeg_deltas = (0, 6)
    else:
        blur_deltas = (0.0, 0.04, 0.08)
        alpha_deltas = (0.0, -0.02)
        warp_deltas = (0.0, 0.02)
        breakup_deltas = (0.0, 0.006, 0.012)
        noise_deltas = (0.0, 0.014, 0.024)
        jpeg_deltas = (0, -6)
        if flags["edge_breakup_missing"] or flags["not_applied"]:
            breakup_deltas = (0.006, 0.012, 0.018)
        if flags["too_clean"] or flags["not_applied"]:
            noise_deltas = (0.012, 0.020, 0.030)

    return {
        "blur": _float_axis(params.blur, blur_deltas, low=0.0, high=2.0, target_len=2),
        "alpha_contrast": _float_axis(params.alpha_contrast, alpha_deltas, low=0.0, high=2.0, target_len=2),
        "photo_warp": _float_axis(params.photo_warp, warp_deltas, low=0.0, high=1.0, target_len=2),
        "edge_breakup": _float_axis(params.edge_breakup, breakup_deltas, low=0.0, high=0.2, target_len=3),
        "photo_noise": _float_axis(params.photo_noise, noise_deltas, low=0.0, high=0.35, target_len=3),
        "jpeg_quality": _jpeg_axis(params.jpeg_quality, jpeg_deltas, target_len=2),
    }


def photo_texture_candidate_grid(
    params: CandidateParams,
    report: dict[str, Any] | None,
    *,
    limit: int = PHOTO_TEXTURE_GRID_TOP_LIMIT,
) -> PhotoTextureCandidateGrid:
    if not report_blocks_photo_texture(report):
        return PhotoTextureCandidateGrid(
            candidates=[],
            report={
                "enabled": False,
                "reason": "photo_texture_not_blocking",
                "stage_id": "photo_texture",
                "candidate_count": 0,
            },
        )

    axes = photo_texture_axes(params, report)
    blur_values = axes["blur"]
    alpha_values = axes["alpha_contrast"]
    warp_values = axes["photo_warp"]
    breakup_values = axes["edge_breakup"]
    noise_values = axes["photo_noise"]
    jpeg_values = axes["jpeg_quality"]
    raw_budget = (
        len(blur_values)
        * len(alpha_values)
        * len(warp_values)
        * len(breakup_values)
        * len(noise_values)
        * len(jpeg_values)
    )

    variants: list[CandidateParams] = []
    for blur in blur_values:
        for alpha_contrast in alpha_values:
            for photo_warp in warp_values:
                for edge_breakup in breakup_values:
                    for photo_noise in noise_values:
                        for jpeg_quality in jpeg_values:
                            variants.append(
                                mutate_params(
                                    params,
                                    blur=blur,
                                    alpha_contrast=alpha_contrast,
                                    photo_warp=photo_warp,
                                    edge_breakup=edge_breakup,
                                    photo_noise=photo_noise,
                                    jpeg_quality=jpeg_quality,
                                )
                            )
    retained_limit = min(limit, PHOTO_TEXTURE_GRID_TOP_LIMIT)
    candidates = dedupe_params(variants, retained_limit)
    candidate_records: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    for candidate in candidates:
        delta_keys = params_delta_keys(params, candidate)
        blocked_delta_keys = sorted(delta_keys & PHOTO_TEXTURE_GRID_BLOCKED_DELTA_KEYS)
        undeclared_delta_keys = sorted(delta_keys - PHOTO_TEXTURE_GRID_ALLOWED_DELTA_KEYS)
        record = {
            "candidate_id": candidate.candidate_id,
            "parent_candidate_id": params.candidate_id,
            "delta_keys": sorted(delta_keys),
            "allowed_delta_keys_only": not blocked_delta_keys and not undeclared_delta_keys,
            "blocked_delta_keys": blocked_delta_keys,
            "undeclared_delta_keys": undeclared_delta_keys,
        }
        candidate_records.append(record)
        if blocked_delta_keys or undeclared_delta_keys:
            violations.append(record)

    budget_min, budget_max = PHOTO_TEXTURE_GRID_BUDGET_RANGE
    return PhotoTextureCandidateGrid(
        candidates=candidates,
        report={
            "enabled": True,
            "stage_id": "photo_texture",
            "optimization_step": "photo_texture",
            "parent_candidate_id": params.candidate_id,
            "budget": {
                "raw_candidate_budget": raw_budget,
                "budget_min": budget_min,
                "budget_max": budget_max,
                "within_budget": budget_min <= raw_budget <= budget_max,
                "retained_top_limit": retained_limit,
                "retained_count": len(candidates),
                "pruned_count": max(0, raw_budget - len(candidates)),
            },
            "axes": {
                "blur_count": len(blur_values),
                "alpha_contrast_count": len(alpha_values),
                "photo_warp_count": len(warp_values),
                "edge_breakup_count": len(breakup_values),
                "photo_noise_count": len(noise_values),
                "jpeg_quality_count": len(jpeg_values),
                "ranking_method": "local_photo_texture_issue_axis_priority",
                "alpha_adjustment_scope": "small_alpha_degradation_or_recovery_only",
                "residual_retexture_keys": ["edge_breakup", "photo_noise", "jpeg_quality"],
            },
            "issue_flags": photo_texture_issue_flags(report),
            "allowed_delta_keys": sorted(PHOTO_TEXTURE_GRID_ALLOWED_DELTA_KEYS),
            "blocked_delta_keys": sorted(PHOTO_TEXTURE_GRID_BLOCKED_DELTA_KEYS),
            "candidate_count": len(candidates),
            "candidate_delta_audit": candidate_records,
            "violations": violations,
        },
    )




def final_font_revision_candidates(
    params: CandidateParams,
    font_style_reference: dict[str, Any],
    plan: RenderPlan,
    report: dict[str, Any] | None,
) -> list[CandidateParams]:
    shape_reset = text_shape_reset_candidates(
        params,
        font_style_reference,
        plan,
        report,
        limit=24,
    )
    if shape_reset:
        return shape_reset

    ranked_fonts = font_style_reference.get("ranked_fonts", [])
    if not isinstance(ranked_fonts, list):
        return []

    preferred_order = ("SimSun", "Songti", "GBSN", "FangSong", "NotoSerif", "UMing")
    selected: list[dict[str, Any]] = []
    seen_paths: set[str] = {params.font_path}
    for preferred_name in preferred_order:
        for item in ranked_fonts:
            if not isinstance(item, dict):
                continue
            font_path = str(item.get("font_path") or "")
            font_name = str(item.get("font_name") or "")
            if not font_path or font_path in seen_paths:
                continue
            if font_name != preferred_name:
                continue
            selected.append(item)
            seen_paths.add(font_path)
            break

    variants: list[CandidateParams] = []
    for item in selected[:4]:
        font_name = str(item.get("font_name") or params.font_name)
        font_path = str(item.get("font_path") or params.font_path)
        try:
            base_size = int(item.get("font_size") or params.font_size)
        except (TypeError, ValueError):
            base_size = params.font_size
        tuning_grid = (
            (0, 0.98, 0.14, 0.04, 0.03, 0.12, 0.18, 0.14),
            (0, 1.00, 0.12, 0.06, 0.02, 0.16, 0.14, 0.12),
            (1, 0.96, 0.16, 0.06, 0.03, 0.10, 0.18, 0.14),
            (-1, 1.00, 0.10, 0.04, 0.02, 0.18, 0.12, 0.10),
        )
        for (
            size_delta,
            opacity,
            blur,
            stroke_opacity,
            ink_gain,
            alpha_contrast,
            core_ink_gain,
            core_darken_strength,
        ) in tuning_grid:
            variants.append(
                mutate_params(
                    params,
                    font_name=font_name,
                    font_path=font_path,
                    font_size=max(8, base_size + size_delta),
                    opacity=opacity,
                    blur=blur,
                    stroke_opacity=stroke_opacity,
                    core_ink_gain=core_ink_gain,
                    core_darken_strength=core_darken_strength,
                    ink_gain=ink_gain,
                    alpha_contrast=alpha_contrast,
                    photo_warp=min(0.10, params.photo_warp),
                    edge_breakup=min(0.010, params.edge_breakup),
                    photo_noise=min(0.020, params.photo_noise),
                    jpeg_quality=max(94, params.jpeg_quality),
                )
            )
    return dedupe_params(variants, 8)
