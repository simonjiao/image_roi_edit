from __future__ import annotations

from typing import Any

from roi_image_edit.iterative_pipeline import CandidateParams, mutate_params
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
)
from roi_image_edit.stage_patchers import (
    acceptance_reports_background_patch,
    acceptance_reports_too_dark_or_bold,
)
from roi_image_edit.stages import stage_gate_for_report


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
