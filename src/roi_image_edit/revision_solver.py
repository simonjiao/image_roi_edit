from __future__ import annotations

from dataclasses import asdict
import json
import re
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
    stage_gate_for_report,
    stage_issues,
)
from roi_image_edit.roi_locator import max_font_size_for_plan, text_chars
from roi_image_edit.stage_policy import STAGE_ORDER


def final_acceptance_delivers(acceptance: dict[str, Any]) -> bool:
    final_level = str(acceptance.get("acceptance_level", "")).strip().lower()
    final_decision = str(acceptance.get("final_decision", "")).strip().lower()
    return bool(acceptance.get("pass")) and final_level == "pass" and final_decision == "deliver"


def acceptance_blocking_stage(acceptance: dict[str, Any] | None) -> str | None:
    if not isinstance(acceptance, dict):
        return None
    stage = str(acceptance.get("blocking_stage") or "").strip()
    if stage in STAGE_ORDER:
        return stage
    findings = acceptance.get("visual_findings")
    if isinstance(findings, dict):
        background = str(findings.get("background") or "").strip().lower()
        if background in {"patch_visible", "ghost_visible", "seam_visible", "too_smooth"}:
            return "background_cleanup"
        sharpness = str(findings.get("sharpness") or "").strip().lower()
        if sharpness in {"too_sharp", "too_blurry"}:
            return "photo_texture"
        darkness = str(findings.get("darkness") or "").strip().lower()
        stroke_weight = str(findings.get("stroke_weight") or "").strip().lower()
        if darkness in {"too_dark", "too_light"} or stroke_weight in {"too_bold", "too_thin", "slightly_bold"}:
            return "ink_gray_balance"
        text_shape_values = {
            str(findings.get("char_positions") or "").strip().lower(),
            str(findings.get("spacing") or "").strip().lower(),
            str(findings.get("baseline") or "").strip().lower(),
            str(findings.get("font_similarity") or "").strip().lower(),
            str(findings.get("size") or "").strip().lower(),
        }
        if any(value and value not in {"ok", "pass"} for value in text_shape_values):
            return "text_shape"
    return None


def effective_blocking_stage(
    report: dict[str, Any] | None,
    acceptance: dict[str, Any] | None,
) -> tuple[str | None, bool]:
    local_stage = None
    if isinstance(report, dict):
        local_stage = (stage_gate_for_report(report) or {}).get("blocking_stage")
    if local_stage:
        return str(local_stage), True
    visual_stage = acceptance_blocking_stage(acceptance)
    return visual_stage, False


def acceptance_text_fragments(acceptance: dict[str, Any]) -> list[str]:
    fragments: list[str] = []
    if not isinstance(acceptance, dict):
        return fragments
    for key in ("reason", "summary"):
        value = acceptance.get(key)
        if isinstance(value, str) and value.strip():
            fragments.append(value)
    for key in ("must_fix", "optional_tuning"):
        entries = acceptance.get(key)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, str) and entry.strip():
                fragments.append(entry)
            elif isinstance(entry, dict):
                for sub_key in ("issue", "suggestion"):
                    value = entry.get(sub_key)
                    if isinstance(value, str) and value.strip():
                        fragments.append(value)
    suggested_patch = acceptance.get("suggested_patch")
    if isinstance(suggested_patch, dict):
        fragments.append(json.dumps(suggested_patch, ensure_ascii=False))
    return fragments


def patch_signature(patch: dict[str, Any]) -> str:
    rounded: dict[str, Any] = {}
    for key, value in patch.items():
        if isinstance(value, float):
            rounded[key] = round(value, 4)
        else:
            rounded[key] = value
    return json.dumps(rounded, ensure_ascii=False, sort_keys=True)


def params_signature(params: CandidateParams) -> str:
    data = asdict(params)
    data.pop("candidate_id", None)
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def dedupe_patches(patches: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for patch in patches:
        clean = {
            key: value
            for key, value in patch.items()
            if value is not None and not (isinstance(value, float) and abs(value) < 0.0001)
        }
        if not clean:
            continue
        key = patch_signature(clean)
        if key in seen:
            continue
        seen.add(key)
        unique.append(clean)
        if len(unique) >= limit:
            break
    return unique


def delta_patch_for_target(params: CandidateParams, name: str, target: float) -> dict[str, Any] | None:
    mapping = {
        "opacity": "opacity_delta",
        "blur": "blur_delta",
        "stroke_opacity": "stroke_opacity_delta",
        "ink_gain": "ink_gain_delta",
        "alpha_contrast": "alpha_contrast_delta",
        "core_ink_gain": "core_ink_gain_delta",
        "core_darken_strength": "core_darken_strength_delta",
        "core_darken_threshold": "core_darken_threshold_delta",
        "core_darken_target_gray": "core_darken_target_gray_delta",
        "photo_warp": "photo_warp_delta",
        "edge_breakup": "edge_breakup_delta",
        "photo_noise": "photo_noise_delta",
        "jpeg_quality": "jpeg_quality_delta",
        "text_dx": "text_dx_delta",
        "text_dy": "text_dy_delta",
    }
    if name == "font_size":
        delta = int(round(target - float(params.font_size)))
        return {"font_size_delta": delta} if delta else None
    if name in {"text_dx", "text_dy"}:
        delta = int(round(target - float(getattr(params, name))))
        return {mapping[name]: delta} if delta else None
    if name not in mapping:
        return None
    current = float(getattr(params, name))
    delta = float(target) - current
    if abs(delta) < 0.005:
        return None
    if name in {"core_darken_threshold", "core_darken_target_gray"}:
        rounded_delta: float | int = int(round(delta))
        if not rounded_delta:
            return None
    else:
        rounded_delta = round(delta, 4)
    return {mapping[name]: rounded_delta}


def patch_from_parameter_suggestion(
    params: CandidateParams,
    suggestion: dict[str, Any],
) -> dict[str, Any] | None:
    name = str(suggestion.get("name") or suggestion.get("parameter") or "").strip()
    if not name:
        return None
    if "to" in suggestion:
        try:
            return delta_patch_for_target(params, name, float(suggestion["to"]))
        except (TypeError, ValueError):
            return None

    delta_key = f"{name}_delta"
    delta = suggestion.get("delta", suggestion.get(delta_key))
    if delta is None:
        return None
    mapping = {
        "font_size": "font_size_delta",
        "opacity": "opacity_delta",
        "blur": "blur_delta",
        "stroke_opacity": "stroke_opacity_delta",
        "ink_gain": "ink_gain_delta",
        "alpha_contrast": "alpha_contrast_delta",
        "core_ink_gain": "core_ink_gain_delta",
        "core_darken_strength": "core_darken_strength_delta",
        "core_darken_threshold": "core_darken_threshold_delta",
        "core_darken_target_gray": "core_darken_target_gray_delta",
        "photo_warp": "photo_warp_delta",
        "edge_breakup": "edge_breakup_delta",
        "photo_noise": "photo_noise_delta",
        "jpeg_quality": "jpeg_quality_delta",
        "text_dx": "text_dx_delta",
        "text_dy": "text_dy_delta",
    }
    patch_key = mapping.get(name)
    if not patch_key:
        return None
    try:
        if name in {"font_size", "core_darken_threshold", "core_darken_target_gray", "jpeg_quality", "text_dx", "text_dy"}:
            value: float | int = int(round(float(delta)))
        else:
            value = round(float(delta), 4)
    except (TypeError, ValueError):
        return None
    return {patch_key: value} if value else None


def model_patch_records(
    params: CandidateParams,
    model_json: dict[str, Any],
    *,
    source: str,
) -> list[dict[str, Any]]:
    if not isinstance(model_json, dict):
        return []
    records: list[dict[str, Any]] = []
    suggested_patch = model_json.get("suggested_patch")
    if isinstance(suggested_patch, dict):
        records.append(
            {
                "source": source,
                "kind": "suggested_patch",
                "direction": model_json.get("direction"),
                "blocking_stage": model_json.get("blocking_stage"),
                "patch": suggested_patch,
            }
        )

    suggestions = model_json.get("parameter_suggestions")
    if isinstance(suggestions, list):
        for idx, suggestion in enumerate(suggestions, start=1):
            if not isinstance(suggestion, dict):
                continue
            patch = patch_from_parameter_suggestion(params, suggestion)
            if not patch:
                continue
            records.append(
                {
                    "source": source,
                    "kind": "parameter_suggestion",
                    "index": idx,
                    "direction": model_json.get("direction"),
                    "blocking_stage": model_json.get("blocking_stage"),
                    "suggestion": suggestion,
                    "patch": patch,
                }
            )
    return records


def numeric_revision_patches(params: CandidateParams, acceptance: dict[str, Any]) -> list[dict[str, Any]]:
    patches: list[dict[str, Any]] = []
    patches.extend(
        record["patch"]
        for record in model_patch_records(params, acceptance, source="model_json")
        if isinstance(record.get("patch"), dict)
    )
    param_names = (
        "core_darken_strength",
        "core_ink_gain",
        "core_darken_threshold",
        "core_darken_target_gray",
        "photo_warp",
        "edge_breakup",
        "photo_noise",
        "jpeg_quality",
        "alpha_contrast",
        "stroke_opacity",
        "ink_gain",
        "opacity",
        "blur",
        "font_size",
        "text_dx",
        "text_dy",
    )
    for text in acceptance_text_fragments(acceptance):
        for name in param_names:
            escaped = re.escape(name)
            patterns = (
                rf"{escaped}\s*(?:[=:：]\s*|从\s*)?[0-9]+(?:\.[0-9]+)?\s*(?:->|→)\s*([0-9]+(?:\.[0-9]+)?)",
                rf"{escaped}\s*(?:从\s*)?[0-9]+(?:\.[0-9]+)?\s*(?:到|调到|降到|下调到|增到|增加到|小幅增到|小幅降到)\s*([0-9]+(?:\.[0-9]+)?)",
                rf"{escaped}\s*[=:：]\s*([0-9]+(?:\.[0-9]+)?)",
            )
            for pattern in patterns:
                match = re.search(pattern, text, flags=re.IGNORECASE)
                if not match:
                    continue
                patch = delta_patch_for_target(params, name, float(match.group(1)))
                if patch:
                    patches.append(patch)
                    break
    return dedupe_patches(patches, 8)


def thin_dark_core_patches(acceptance: dict[str, Any]) -> list[dict[str, Any]]:
    findings = acceptance.get("visual_findings") if isinstance(acceptance, dict) else {}
    if not isinstance(findings, dict):
        findings = {}
    text = "\n".join(acceptance_text_fragments(acceptance)).lower()
    stroke_weight = str(findings.get("stroke_weight", "")).strip().lower()
    darkness = str(findings.get("darkness", "")).strip().lower()
    font_size = str(findings.get("size", "")).strip().lower()
    wants_thinner = acceptance_wants_thinner_strokes(acceptance)
    wants_darker_core = acceptance_wants_darker_core(acceptance)
    if not wants_thinner and font_size != "too_large":
        return []
    if acceptance_reports_too_dark_or_bold(acceptance) and not wants_darker_core:
        return [
            {"opacity_delta": -0.04, "alpha_contrast_delta": -0.04},
            {"opacity_delta": -0.06, "alpha_contrast_delta": -0.06, "blur_delta": 0.04},
            {"opacity_delta": -0.06, "blur_delta": 0.08},
            {"font_size_delta": -1, "opacity_delta": -0.03, "blur_delta": 0.04},
        ]

    patches = [
        {
            "font_size_delta": -1,
            "blur_delta": -0.08,
            "alpha_contrast_delta": 0.20,
            "core_ink_gain_delta": -0.10,
            "core_darken_strength_delta": 0.04,
            "core_darken_threshold_delta": 10,
        },
        {
            "font_size_delta": -1,
            "blur_delta": -0.06,
            "alpha_contrast_delta": 0.25,
            "core_ink_gain_delta": -0.14,
            "core_darken_strength_delta": 0.06,
            "core_darken_threshold_delta": 16,
            "core_darken_target_gray_delta": -4,
        },
        {
            "blur_delta": -0.10,
            "alpha_contrast_delta": 0.25,
            "core_ink_gain_delta": -0.12,
            "core_darken_threshold_delta": 18,
            "core_darken_target_gray_delta": -6,
        },
        {
            "font_size_delta": -1,
            "opacity_delta": 0.02,
            "blur_delta": -0.06,
            "alpha_contrast_delta": 0.18,
            "core_ink_gain_delta": -0.12,
            "core_darken_strength_delta": 0.08,
            "core_darken_threshold_delta": 14,
        },
    ]
    if darkness == "too_dark" and not wants_darker_core:
        patches.append(
            {
                "font_size_delta": -1,
                "blur_delta": -0.06,
                "alpha_contrast_delta": 0.22,
                "core_ink_gain_delta": -0.16,
                "core_darken_strength_delta": -0.02,
                "core_darken_threshold_delta": 18,
            }
        )
    return patches


def acceptance_wants_darker_core(acceptance: dict[str, Any]) -> bool:
    text = "\n".join(acceptance_text_fragments(acceptance)).lower()
    findings = acceptance.get("visual_findings") if isinstance(acceptance, dict) else {}
    if not isinstance(findings, dict):
        findings = {}
    darkness = str(findings.get("darkness", "")).strip().lower()
    stroke_weight = str(findings.get("stroke_weight", "")).strip().lower()
    if darkness == "too_light" or stroke_weight == "too_thin":
        return True
    if any(token in text for token in ("too_dark", "too_bold", "过黑", "偏黑", "过重", "偏重", "太黑", "太粗", "黑度偏重", "核心过量")):
        return False
    return any(
        token in text
        for token in ("不够黑", "偏浅", "太浅", "过淡", "偏淡", "核心不足", "核心不够", "too_light", "too_thin")
    )


def acceptance_wants_thinner_strokes(acceptance: dict[str, Any]) -> bool:
    findings = acceptance.get("visual_findings") if isinstance(acceptance, dict) else {}
    if not isinstance(findings, dict):
        findings = {}
    text = "\n".join(acceptance_text_fragments(acceptance)).lower()
    stroke_weight = str(findings.get("stroke_weight", "")).strip().lower()
    if any(token in text for token in ("不够粗", "偏细", "太细", "更粗", "更重", "加粗", "描黑", "too_thin")):
        return False
    return (
        stroke_weight == "too_bold"
        or "too_bold" in text
        or "偏重" in text
        or "过重" in text
        or ("笔画" in text and ("粗" in text or "重" in text))
    )


def acceptance_reports_too_dark_or_bold(acceptance: dict[str, Any]) -> bool:
    findings = acceptance.get("visual_findings") if isinstance(acceptance, dict) else {}
    if not isinstance(findings, dict):
        findings = {}
    text = "\n".join(acceptance_text_fragments(acceptance)).lower()
    darkness = str(findings.get("darkness", "")).strip().lower()
    stroke_weight = str(findings.get("stroke_weight", "")).strip().lower()
    return (
        darkness == "too_dark"
        or stroke_weight in {"too_bold", "slightly_bold"}
        or "too_dark" in text
        or "too_bold" in text
        or "偏黑" in text
        or "过黑" in text
        or "偏重" in text
        or "过重" in text
    )


def acceptance_reports_background_patch(acceptance: dict[str, Any]) -> bool:
    findings = acceptance.get("visual_findings") if isinstance(acceptance, dict) else {}
    if not isinstance(findings, dict):
        findings = {}
    background = str(findings.get("background", "")).strip().lower()
    text = "\n".join(acceptance_text_fragments(acceptance)).lower()
    return (
        background in {"patch_visible", "ghost_visible", "too_smooth"}
        or "补丁" in text
        or "平滑" in text
        or "涂抹" in text
        or "残影" in text
        or "ghost_visible" in text
        or "patch_visible" in text
    )



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



def black_core_reduction_patches() -> list[dict[str, Any]]:
    return [
        {"opacity_delta": -0.06, "stroke_opacity_delta": -0.02, "alpha_contrast_delta": -0.05},
        {"opacity_delta": -0.06, "stroke_opacity_delta": -0.04, "alpha_contrast_delta": -0.08},
        {"opacity_delta": -0.06, "blur_delta": -0.02, "stroke_opacity_delta": -0.04},
        {"opacity_delta": -0.06, "blur_delta": 0.02, "stroke_opacity_delta": -0.02},
        {"opacity_delta": -0.04, "blur_delta": 0.06, "core_ink_gain_delta": -0.03},
        {"opacity_delta": -0.06, "blur_delta": 0.08, "core_darken_strength_delta": -0.03},
        {"core_ink_gain_delta": -0.06, "core_darken_strength_delta": -0.04},
        {"core_darken_strength_delta": -0.06, "blur_delta": 0.04},
        {"ink_gain_delta": -0.04, "core_ink_gain_delta": -0.04, "blur_delta": 0.06},
        {"opacity_delta": -0.03, "core_ink_gain_delta": -0.05, "photo_noise_delta": 0.018},
        {
            "core_ink_gain_delta": -0.08,
            "core_darken_strength_delta": -0.06,
            "edge_breakup_delta": 0.008,
            "photo_noise_delta": 0.020,
            "jpeg_quality_delta": -4,
        },
    ]


def alignment_centering_patches(report: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(report, dict):
        return []
    alignment = report.get("char_alignment_metrics")
    if not isinstance(alignment, dict) or not alignment.get("enabled"):
        return []
    center_dys: list[float] = []
    for item in alignment.get("per_char", []):
        if not isinstance(item, dict) or not item.get("candidate_box"):
            continue
        try:
            center_dys.append(float(item.get("center_dy") or 0.0))
        except (TypeError, ValueError):
            continue
    if not center_dys:
        return []
    mean_center_dy = sum(center_dys) / len(center_dys)
    if mean_center_dy < -0.25:
        return [{"text_dy_delta": 1}]
    if mean_center_dy > 0.90:
        return [{"text_dy_delta": -1}]
    return []


def neighbor_outer_gray_cleanup_patches() -> list[dict[str, Any]]:
    return [
        {
            "alpha_contrast_delta": 0.16,
            "blur_delta": -0.06,
            "stroke_opacity_delta": -0.04,
            "core_ink_gain_delta": 0.02,
            "core_darken_strength_delta": 0.03,
            "edge_breakup_delta": -0.010,
            "photo_noise_delta": -0.030,
            "jpeg_quality_delta": 8,
        },
        {
            "alpha_contrast_delta": 0.12,
            "blur_delta": -0.05,
            "stroke_opacity_delta": -0.03,
            "core_ink_gain_delta": 0.02,
            "core_darken_strength_delta": 0.02,
            "edge_breakup_delta": -0.008,
            "photo_noise_delta": -0.025,
            "jpeg_quality_delta": 6,
        },
        {
            "alpha_contrast_delta": 0.20,
            "blur_delta": -0.07,
            "stroke_opacity_delta": -0.05,
            "opacity_delta": 0.01,
            "core_ink_gain_delta": 0.02,
            "edge_breakup_delta": -0.010,
            "photo_noise_delta": -0.035,
            "jpeg_quality_delta": 8,
        },
        {
            "blur_delta": -0.07,
            "stroke_opacity_delta": -0.03,
            "core_ink_gain_delta": 0.02,
            "core_darken_strength_delta": 0.02,
            "edge_breakup_delta": -0.008,
            "photo_noise_delta": -0.030,
            "jpeg_quality_delta": 6,
        },
        {
            "blur_delta": -0.05,
            "stroke_opacity_delta": -0.04,
            "opacity_delta": 0.01,
            "core_ink_gain_delta": 0.02,
            "edge_breakup_delta": -0.008,
            "photo_noise_delta": -0.025,
            "jpeg_quality_delta": 6,
        },
        {
            "blur_delta": -0.09,
            "core_darken_strength_delta": 0.03,
            "edge_breakup_delta": -0.010,
            "photo_noise_delta": -0.035,
            "jpeg_quality_delta": 8,
        },
        {
            "blur_delta": -0.06,
            "stroke_opacity_delta": -0.02,
            "ink_gain_delta": -0.02,
            "core_ink_gain_delta": 0.03,
            "core_darken_strength_delta": 0.02,
            "edge_breakup_delta": -0.008,
            "photo_noise_delta": -0.025,
            "jpeg_quality_delta": 6,
        },
    ]


def neighbor_core_density_recovery_patches() -> list[dict[str, Any]]:
    return [
        {
            "stroke_opacity_delta": 0.03,
            "blur_delta": -0.04,
            "core_ink_gain_delta": 0.04,
            "core_darken_strength_delta": 0.04,
            "core_darken_threshold_delta": 8,
            "photo_warp_delta": -0.02,
            "edge_breakup_delta": -0.004,
            "photo_noise_delta": -0.012,
            "jpeg_quality_delta": 4,
        },
        {
            "stroke_opacity_delta": 0.04,
            "opacity_delta": 0.02,
            "blur_delta": -0.04,
            "core_ink_gain_delta": 0.03,
            "core_darken_strength_delta": 0.03,
            "photo_warp_delta": -0.02,
            "edge_breakup_delta": -0.004,
            "photo_noise_delta": -0.012,
            "jpeg_quality_delta": 4,
        },
        {
            "stroke_opacity_delta": 0.05,
            "blur_delta": -0.03,
            "ink_gain_delta": 0.01,
            "core_ink_gain_delta": 0.02,
            "core_darken_strength_delta": 0.02,
            "photo_warp_delta": -0.02,
            "edge_breakup_delta": -0.004,
            "photo_noise_delta": -0.010,
            "jpeg_quality_delta": 4,
        },
        {
            "stroke_opacity_delta": 0.03,
            "font_size_delta": 1,
            "blur_delta": -0.04,
            "core_ink_gain_delta": 0.03,
            "core_darken_strength_delta": 0.03,
            "photo_warp_delta": -0.02,
            "edge_breakup_delta": -0.004,
            "photo_noise_delta": -0.012,
            "jpeg_quality_delta": 4,
        },
    ]


def gray_stroke_recovery_patches() -> list[dict[str, Any]]:
    return [
        {
            "stroke_opacity_delta": 0.03,
            "blur_delta": -0.04,
            "ink_gain_delta": -0.02,
            "core_ink_gain_delta": -0.02,
            "photo_warp_delta": -0.02,
            "edge_breakup_delta": -0.004,
            "photo_noise_delta": -0.012,
            "jpeg_quality_delta": 4,
        },
        {
            "stroke_opacity_delta": 0.04,
            "blur_delta": -0.05,
            "ink_gain_delta": -0.03,
            "core_ink_gain_delta": -0.03,
            "photo_warp_delta": -0.02,
            "edge_breakup_delta": -0.006,
            "photo_noise_delta": -0.014,
            "jpeg_quality_delta": 4,
        },
        {
            "stroke_opacity_delta": 0.06,
            "opacity_delta": -0.02,
            "blur_delta": 0.03,
            "ink_gain_delta": 0.02,
            "core_ink_gain_delta": -0.05,
            "core_darken_strength_delta": -0.05,
            "photo_warp_delta": 0.04,
            "edge_breakup_delta": 0.006,
            "photo_noise_delta": 0.016,
            "jpeg_quality_delta": -4,
        },
        {
            "stroke_opacity_delta": 0.08,
            "opacity_delta": -0.03,
            "blur_delta": 0.02,
            "ink_gain_delta": 0.02,
            "core_ink_gain_delta": -0.06,
            "core_darken_strength_delta": -0.06,
            "photo_warp_delta": 0.04,
            "edge_breakup_delta": 0.006,
            "photo_noise_delta": 0.016,
            "jpeg_quality_delta": -4,
        },
        {
            "stroke_opacity_delta": 0.10,
            "blur_delta": 0.06,
            "ink_gain_delta": 0.04,
            "alpha_contrast_delta": -0.06,
            "core_ink_gain_delta": -0.05,
            "core_darken_strength_delta": -0.04,
            "photo_warp_delta": 0.06,
            "edge_breakup_delta": 0.012,
            "photo_noise_delta": 0.030,
            "jpeg_quality_delta": -8,
        },
        {
            "stroke_opacity_delta": 0.12,
            "font_size_delta": 1,
            "blur_delta": 0.07,
            "ink_gain_delta": 0.03,
            "alpha_contrast_delta": -0.08,
            "core_ink_gain_delta": -0.07,
            "core_darken_strength_delta": -0.05,
            "photo_warp_delta": 0.08,
            "edge_breakup_delta": 0.014,
            "photo_noise_delta": 0.035,
            "jpeg_quality_delta": -10,
        },
        {
            "stroke_opacity_delta": 0.05,
            "blur_delta": 0.08,
            "ink_gain_delta": 0.04,
            "core_ink_gain_delta": -0.04,
            "core_darken_strength_delta": -0.04,
            "photo_warp_delta": 0.06,
            "edge_breakup_delta": 0.010,
            "photo_noise_delta": 0.025,
            "jpeg_quality_delta": -6,
        },
        {
            "stroke_opacity_delta": 0.06,
            "font_size_delta": 1,
            "blur_delta": 0.06,
            "ink_gain_delta": 0.03,
            "core_ink_gain_delta": -0.06,
            "core_darken_strength_delta": -0.04,
            "photo_warp_delta": 0.08,
            "edge_breakup_delta": 0.012,
            "photo_noise_delta": 0.030,
            "jpeg_quality_delta": -8,
        },
        {
            "opacity_delta": 0.03,
            "blur_delta": 0.08,
            "ink_gain_delta": 0.03,
            "core_darken_strength_delta": -0.06,
            "photo_warp_delta": 0.05,
            "edge_breakup_delta": 0.010,
            "photo_noise_delta": 0.025,
            "jpeg_quality_delta": -6,
        },
        {
            "blur_delta": 0.12,
            "ink_gain_delta": 0.06,
            "alpha_contrast_delta": -0.10,
            "core_ink_gain_delta": -0.08,
            "photo_warp_delta": 0.08,
            "edge_breakup_delta": 0.014,
            "photo_noise_delta": 0.035,
            "jpeg_quality_delta": -10,
        },
    ]


def photo_texture_recovery_patches(report: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    issue_types = {
        str(issue.get("type") or "")
        for issue in stage_issues(report, "photo_texture")
        if isinstance(issue, dict)
    }
    if "photo_texture_too_blurry" in issue_types:
        return [
            {
                "blur_delta": -0.08,
                "photo_noise_delta": -0.006,
                "edge_breakup_delta": -0.002,
                "jpeg_quality_delta": 4,
            },
            {
                "blur_delta": -0.12,
                "alpha_contrast_delta": 0.08,
                "photo_warp_delta": -0.010,
                "jpeg_quality_delta": 6,
            },
            {
                "blur_delta": -0.06,
                "photo_noise_delta": -0.004,
                "jpeg_quality_delta": 4,
            },
        ]
    return [
        {
            "blur_delta": 0.08,
            "edge_breakup_delta": 0.006,
            "photo_noise_delta": 0.014,
            "jpeg_quality_delta": -6,
        },
        {
            "blur_delta": 0.06,
            "photo_warp_delta": 0.020,
            "edge_breakup_delta": 0.004,
            "photo_noise_delta": 0.010,
            "jpeg_quality_delta": -4,
        },
        {
            "blur_delta": 0.10,
            "photo_noise_delta": 0.018,
            "jpeg_quality_delta": -8,
        },
        {
            "edge_breakup_delta": 0.008,
            "photo_noise_delta": 0.012,
            "jpeg_quality_delta": -6,
        },
        {
            "photo_warp_delta": 0.025,
            "edge_breakup_delta": 0.006,
            "photo_noise_delta": 0.010,
        },
    ]


def background_cleanup_recovery_patches(report: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    issue_types = {
        str(issue.get("type") or "")
        for issue in stage_issues(report, "background_cleanup")
        if isinstance(issue, dict)
    }
    patches: list[dict[str, Any]] = []
    has_white_ghost = "background_white_ghost_residual" in issue_types
    if has_white_ghost:
        patches.extend(
            [
                {
                    "mask_threshold_delta": 12,
                    "mask_dilate_iterations_delta": 1,
                    "inpaint_radius_delta": 1,
                    "photo_noise_delta": -0.018,
                    "edge_breakup_delta": -0.010,
                    "jpeg_quality_delta": 8,
                },
                {
                    "mask_threshold_delta": 20,
                    "mask_dilate_iterations_delta": 1,
                    "inpaint_radius_delta": 1,
                    "photo_noise_delta": -0.024,
                    "edge_breakup_delta": -0.012,
                    "jpeg_quality_delta": 10,
                },
                {
                    "mask_threshold_delta": 8,
                    "mask_dilate_iterations_delta": 1,
                    "inpaint_radius_delta": 1,
                    "photo_noise_delta": -0.012,
                    "edge_breakup_delta": -0.008,
                    "jpeg_quality_delta": 6,
                },
                {
                    "mask_dilate_iterations_delta": 1,
                    "inpaint_radius_delta": 1,
                    "photo_noise_delta": -0.020,
                    "edge_breakup_delta": -0.010,
                    "jpeg_quality_delta": 8,
                },
            ]
        )
    if "background_fill_luminance_mismatch" in issue_types and not has_white_ghost:
        patches.extend(
            [
                {"mask_threshold_delta": -10, "inpaint_radius_delta": -1, "photo_noise_delta": 0.012},
                {"mask_threshold_delta": 10, "inpaint_radius_delta": 1, "photo_noise_delta": 0.010},
                {"inpaint_radius_delta": -1, "edge_breakup_delta": 0.006, "photo_noise_delta": 0.016},
            ]
        )
    if (
        not has_white_ghost
        and (
            "background_fill_too_smooth" in issue_types
            or "background_fill_low_texture_variance" in issue_types
            or "background_trailing_patch_too_smooth" in issue_types
        )
    ):
        patches.extend(
            [
                {"photo_noise_delta": 0.020, "edge_breakup_delta": 0.006, "jpeg_quality_delta": -6},
                {"photo_warp_delta": 0.020, "photo_noise_delta": 0.014, "jpeg_quality_delta": -4},
                {"mask_dilate_iterations_delta": -1, "photo_noise_delta": 0.018, "edge_breakup_delta": 0.006},
                {"photo_noise_delta": 0.032, "edge_breakup_delta": 0.010, "jpeg_quality_delta": -8},
                {"inpaint_radius_delta": -1, "photo_noise_delta": 0.028, "edge_breakup_delta": 0.010},
                {"photo_noise_delta": 0.052, "edge_breakup_delta": 0.018, "jpeg_quality_delta": -10},
                {"photo_noise_delta": 0.070, "edge_breakup_delta": 0.024, "jpeg_quality_delta": -12},
                {"inpaint_radius_delta": 1, "photo_noise_delta": 0.045, "edge_breakup_delta": 0.014},
            ]
        )
    if not patches:
        patches.extend(
            [
                {"photo_noise_delta": 0.014, "edge_breakup_delta": 0.004},
                {"mask_threshold_delta": -8, "inpaint_radius_delta": -1},
            ]
        )
    return dedupe_patches(patches, 8)


def visual_background_cleanup_patches(acceptance: dict[str, Any]) -> list[dict[str, Any]]:
    if not acceptance_reports_background_patch(acceptance):
        return []
    return [
        {"photo_noise_delta": 0.030, "edge_breakup_delta": 0.010, "jpeg_quality_delta": -6},
        {"photo_noise_delta": 0.045, "edge_breakup_delta": 0.014, "jpeg_quality_delta": -10},
        {"inpaint_radius_delta": -1, "photo_noise_delta": 0.035, "edge_breakup_delta": 0.012},
        {"mask_threshold_delta": -8, "inpaint_radius_delta": -1, "photo_noise_delta": 0.030},
        {"mask_dilate_iterations_delta": -1, "photo_noise_delta": 0.035, "edge_breakup_delta": 0.010},
    ]


def ink_balance_recovery_patches(report: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    issue_types = {
        str(issue.get("type") or "")
        for issue in stage_issues(report, "ink_gray_balance")
        if isinstance(issue, dict)
    }
    patches: list[dict[str, Any]] = []
    if "core_mean_gray_too_light" in issue_types or "core_lighten_too_high" in issue_types:
        patches.extend(
            [
                {"opacity_delta": 0.015},
                {"alpha_contrast_delta": 0.05},
                {"opacity_delta": 0.015, "blur_delta": -0.04},
                {"core_darken_strength_delta": 0.02},
                {"opacity_delta": 0.01, "alpha_contrast_delta": 0.04},
                {"opacity_delta": 0.02, "core_darken_strength_delta": 0.02},
            ]
        )
    if any("too_black" in issue_type or issue_type == "roi_core_too_black" for issue_type in issue_types):
        patches.extend(black_core_reduction_patches()[:6])
    if not patches:
        patches.extend(
            [
                {"opacity_delta": -0.02},
                {"opacity_delta": 0.02},
                {"blur_delta": 0.04},
                {"blur_delta": -0.04},
            ]
        )
    return dedupe_patches(patches, 12)


def keep_patch_for_gray_stroke_recovery(patch: dict[str, Any]) -> bool:
    try:
        opacity_delta = float(patch.get("opacity_delta") or 0.0)
        blur_delta = float(patch.get("blur_delta") or 0.0)
        alpha_contrast_delta = float(patch.get("alpha_contrast_delta") or 0.0)
        stroke_opacity_delta = float(patch.get("stroke_opacity_delta") or 0.0)
        ink_gain_delta = float(patch.get("ink_gain_delta") or 0.0)
        photo_noise_delta = float(patch.get("photo_noise_delta") or 0.0)
        edge_breakup_delta = float(patch.get("edge_breakup_delta") or 0.0)
        font_size_delta = int(round(float(patch.get("font_size_delta") or 0.0)))
    except (TypeError, ValueError):
        return False
    if opacity_delta < 0 and stroke_opacity_delta <= 0:
        return False
    if blur_delta < 0 and stroke_opacity_delta <= 0:
        return False
    if alpha_contrast_delta > 0:
        return False
    if font_size_delta < 0:
        return False
    widens_body = (
        stroke_opacity_delta > 0.0
        or ink_gain_delta > 0.0
        or blur_delta > 0.0
        or photo_noise_delta > 0.0
        or edge_breakup_delta > 0.0
        or font_size_delta > 0
    )
    if not widens_body:
        return False
    return True


def keep_patch_for_outer_gray_cleanup(patch: dict[str, Any]) -> bool:
    try:
        opacity_delta = float(patch.get("opacity_delta") or 0.0)
        blur_delta = float(patch.get("blur_delta") or 0.0)
        stroke_opacity_delta = float(patch.get("stroke_opacity_delta") or 0.0)
        ink_gain_delta = float(patch.get("ink_gain_delta") or 0.0)
        alpha_contrast_delta = float(patch.get("alpha_contrast_delta") or 0.0)
        core_ink_gain_delta = float(patch.get("core_ink_gain_delta") or 0.0)
        core_darken_strength_delta = float(patch.get("core_darken_strength_delta") or 0.0)
        photo_noise_delta = float(patch.get("photo_noise_delta") or 0.0)
        edge_breakup_delta = float(patch.get("edge_breakup_delta") or 0.0)
        font_size_delta = int(round(float(patch.get("font_size_delta") or 0.0)))
    except (TypeError, ValueError):
        return False
    if opacity_delta < -0.025:
        return False
    if blur_delta > 0.0 or photo_noise_delta > 0.0 or edge_breakup_delta > 0.0:
        return False
    if core_ink_gain_delta < 0.0 or core_darken_strength_delta < 0.0:
        return False
    if font_size_delta != 0:
        return False
    trims_outer_gray = (
        blur_delta < 0.0
        or stroke_opacity_delta < 0.0
        or ink_gain_delta < 0.0
        or alpha_contrast_delta > 0.0
        or photo_noise_delta < 0.0
        or edge_breakup_delta < 0.0
    )
    preserves_core = core_ink_gain_delta > 0.0 or core_darken_strength_delta > 0.0 or opacity_delta > 0.0
    return trims_outer_gray and preserves_core


def keep_patch_for_neighbor_core_recovery(patch: dict[str, Any]) -> bool:
    try:
        opacity_delta = float(patch.get("opacity_delta") or 0.0)
        blur_delta = float(patch.get("blur_delta") or 0.0)
        stroke_opacity_delta = float(patch.get("stroke_opacity_delta") or 0.0)
        ink_gain_delta = float(patch.get("ink_gain_delta") or 0.0)
        core_ink_gain_delta = float(patch.get("core_ink_gain_delta") or 0.0)
        core_darken_strength_delta = float(patch.get("core_darken_strength_delta") or 0.0)
        photo_noise_delta = float(patch.get("photo_noise_delta") or 0.0)
        edge_breakup_delta = float(patch.get("edge_breakup_delta") or 0.0)
        font_size_delta = int(round(float(patch.get("font_size_delta") or 0.0)))
    except (TypeError, ValueError):
        return False
    if opacity_delta < -0.035:
        return False
    if blur_delta > 0.02:
        return False
    if photo_noise_delta > 0.0 or edge_breakup_delta > 0.0:
        return False
    if core_ink_gain_delta < 0.0 or core_darken_strength_delta < 0.0:
        return False
    if font_size_delta < 0:
        return False
    improves_core = (
        stroke_opacity_delta > 0.0
        or ink_gain_delta > 0.0
        or core_ink_gain_delta > 0.0
        or core_darken_strength_delta > 0.0
        or blur_delta < 0.0
    )
    return improves_core


def revision_patches_for_round(
    params: CandidateParams,
    acceptance: dict[str, Any],
    report: dict[str, Any] | None = None,
    *,
    rank_patch: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    patches: list[dict[str, Any]] = []
    stage_gate = stage_gate_for_report(report) if isinstance(report, dict) else {}
    blocking_stage = stage_gate.get("blocking_stage") or acceptance_blocking_stage(acceptance)
    if blocking_stage == "text_shape":
        if report_has_outer_gray_halo(report):
            patches.extend(neighbor_outer_gray_cleanup_patches())
            patches.extend(
                patch
                for patch in numeric_revision_patches(params, acceptance)
                if keep_patch_for_outer_gray_cleanup(patch)
            )
            if isinstance(rank_patch, dict) and keep_patch_for_outer_gray_cleanup(rank_patch):
                patches.append(rank_patch)
            return dedupe_patches(patches, 12)
        if local_neighbor_style_issues(report or {}, allow_excess_black_core=True):
            patches.extend(neighbor_core_density_recovery_patches())
            patches.extend(
                patch
                for patch in numeric_revision_patches(params, acceptance)
                if keep_patch_for_neighbor_core_recovery(patch)
            )
            if isinstance(rank_patch, dict) and keep_patch_for_neighbor_core_recovery(rank_patch):
                patches.append(rank_patch)
            return dedupe_patches(patches, 12)
        patches.extend(gray_stroke_recovery_patches())
        patches.extend(
            patch
            for patch in numeric_revision_patches(params, acceptance)
            if keep_patch_for_gray_stroke_recovery(patch)
        )
        if isinstance(rank_patch, dict) and keep_patch_for_gray_stroke_recovery(rank_patch):
            patches.append(rank_patch)
        return dedupe_patches(patches, 12)

    if blocking_stage == "photo_texture":
        patches.extend(alignment_centering_patches(report))
        patches.extend(photo_texture_recovery_patches(report))
        patches.extend(final_revision_patches(acceptance))
        if isinstance(rank_patch, dict):
            patches.append(rank_patch)
        return dedupe_patches(patches, 12)

    if blocking_stage == "background_cleanup":
        patches.extend(background_cleanup_recovery_patches(report))
        patches.extend(visual_background_cleanup_patches(acceptance))
        patches.extend(numeric_revision_patches(params, acceptance))
        patches.extend(photo_texture_recovery_patches(report))
        if isinstance(rank_patch, dict):
            patches.append(rank_patch)
        return dedupe_patches(patches, 12)

    if report_has_excess_black_core(report):
        patches.extend(alignment_centering_patches(report))
        patches.extend(black_core_reduction_patches())
        patches.extend(numeric_revision_patches(params, acceptance))
        if isinstance(rank_patch, dict):
            patches.append(rank_patch)
        patches.extend(final_revision_patches(acceptance))
        return dedupe_patches(patches, 12)

    if blocking_stage == "ink_gray_balance":
        patches.extend(alignment_centering_patches(report))
        patches.extend(ink_balance_recovery_patches(report))
        patches.extend(numeric_revision_patches(params, acceptance))
        if isinstance(rank_patch, dict):
            patches.append(rank_patch)
        patches.extend(final_revision_patches(acceptance))
        return dedupe_patches(patches, 12)

    needs_wider_gray_strokes = report_needs_wider_gray_strokes(report)
    if needs_wider_gray_strokes:
        patches.extend(alignment_centering_patches(report))
        if report_has_outer_gray_halo(report):
            patches.extend(neighbor_outer_gray_cleanup_patches())
            patches.extend(
                patch
                for patch in numeric_revision_patches(params, acceptance)
                if keep_patch_for_outer_gray_cleanup(patch)
            )
            if isinstance(rank_patch, dict) and keep_patch_for_outer_gray_cleanup(rank_patch):
                patches.append(rank_patch)
            return dedupe_patches(patches, 12)
        if local_neighbor_style_issues(report, allow_excess_black_core=True):
            patches.extend(neighbor_core_density_recovery_patches())
            patches.extend(
                patch
                for patch in numeric_revision_patches(params, acceptance)
                if keep_patch_for_neighbor_core_recovery(patch)
            )
            if isinstance(rank_patch, dict) and keep_patch_for_neighbor_core_recovery(rank_patch):
                patches.append(rank_patch)
            return dedupe_patches(patches, 12)
        patches.extend(gray_stroke_recovery_patches())
        patches.extend(
            patch
            for patch in numeric_revision_patches(params, acceptance)
            if keep_patch_for_gray_stroke_recovery(patch)
        )
        return dedupe_patches(patches, 12)

    patches.extend(alignment_centering_patches(report))
    if report_needs_thinner_strokes(report):
        patches.extend(thin_dark_core_patches(acceptance))
    patches.extend(numeric_revision_patches(params, acceptance))
    if isinstance(rank_patch, dict):
        patches.append(rank_patch)
    patches.extend(final_revision_patches(acceptance))
    return dedupe_patches(patches, 12)


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
    return tuple(candidates[:4])


def text_shape_reset_candidates(
    params: CandidateParams,
    font_style_reference: dict[str, Any],
    plan: RenderPlan,
    report: dict[str, Any] | None,
    *,
    limit: int = 48,
) -> list[CandidateParams]:
    if not report_blocks_text_shape(report):
        return []

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

    if has_outer_halo:
        shape_grid = (
            (1.00, 0.10, 0.02, 0.02, 0.16, 0.12, 0.10, 0.06, 0.004, 0.008, 98),
            (0.98, 0.12, 0.04, 0.02, 0.14, 0.14, 0.12, 0.06, 0.004, 0.010, 98),
            (0.96, 0.14, 0.04, 0.03, 0.12, 0.16, 0.12, 0.07, 0.006, 0.012, 96),
            (1.00, 0.08, 0.06, 0.01, 0.20, 0.12, 0.10, 0.05, 0.002, 0.006, 99),
        )
        size_deltas = (0, -1, 1)
    elif has_body_gap:
        shape_grid = (
            (0.90, 0.16, 0.03, 0.00, 0.18, 0.00, 0.00, 0.06, 0.004, 0.008, 99),
            (0.92, 0.14, 0.04, 0.00, 0.16, 0.04, 0.04, 0.06, 0.004, 0.008, 99),
            (0.94, 0.16, 0.04, 0.01, 0.14, 0.08, 0.06, 0.06, 0.004, 0.010, 98),
            (1.00, 0.12, 0.04, 0.04, 0.10, 0.18, 0.14, 0.07, 0.006, 0.012, 96),
            (0.98, 0.14, 0.06, 0.03, 0.10, 0.18, 0.14, 0.07, 0.006, 0.014, 96),
            (0.96, 0.16, 0.08, 0.02, 0.12, 0.16, 0.12, 0.08, 0.008, 0.016, 95),
            (1.00, 0.10, 0.08, 0.02, 0.16, 0.14, 0.12, 0.06, 0.004, 0.010, 98),
            (0.94, 0.18, 0.06, 0.03, 0.08, 0.20, 0.14, 0.08, 0.008, 0.018, 94),
            (1.00, 0.14, 0.02, 0.05, 0.08, 0.22, 0.16, 0.08, 0.006, 0.014, 96),
        )
        size_deltas = (0, 1, -1, 2)
    else:
        shape_grid = (
            (1.00, 0.12, 0.04, 0.03, 0.12, 0.16, 0.12, 0.07, 0.006, 0.012, 96),
            (0.98, 0.16, 0.04, 0.03, 0.10, 0.18, 0.14, 0.08, 0.006, 0.014, 96),
            (1.00, 0.10, 0.06, 0.02, 0.16, 0.14, 0.12, 0.06, 0.004, 0.010, 98),
        )
        size_deltas = (0, -1, 1)

    if has_pose_gap:
        size_deltas = tuple(dict.fromkeys(size_deltas + (0,)))

    max_font_size = max_font_size_for_plan(plan)
    offset_candidates = normalized_offset_candidates(plan, params)
    text_dy_candidates = tuple(dict.fromkeys((params.text_dy, 0, -1, 1)))[:3]
    variants: list[CandidateParams] = []
    for font_item in shape_font_items(params, font_style_reference, limit=5):
        font_name = str(font_item.get("font_name") or params.font_name)
        font_path = str(font_item.get("font_path") or params.font_path)
        try:
            base_size = int(font_item.get("font_size") or params.font_size)
        except (TypeError, ValueError):
            base_size = params.font_size
        for size_delta in size_deltas:
            font_size = max(8, min(max_font_size, base_size + int(size_delta)))
            for offsets in offset_candidates:
                for text_dy in text_dy_candidates:
                    for (
                        opacity,
                        blur,
                        stroke_opacity,
                        ink_gain,
                        alpha_contrast,
                        core_ink_gain,
                        core_darken_strength,
                        photo_warp,
                        edge_breakup,
                        photo_noise,
                        jpeg_quality,
                    ) in shape_grid:
                        variants.append(
                            mutate_params(
                                params,
                                font_name=font_name,
                                font_path=font_path,
                                font_size=font_size,
                                opacity=opacity,
                                blur=blur,
                                stroke_opacity=stroke_opacity,
                                ink_gain=ink_gain,
                                alpha_contrast=alpha_contrast,
                                core_ink_gain=core_ink_gain,
                                core_darken_strength=core_darken_strength,
                                core_darken_threshold=130,
                                core_darken_target_gray=28,
                                text_dy=text_dy,
                                char_offsets=offsets,
                                photo_warp=photo_warp,
                                edge_breakup=edge_breakup,
                                photo_noise=photo_noise,
                                jpeg_quality=jpeg_quality,
                            )
                        )
    return dedupe_params(variants, limit)


def final_revision_patches(acceptance: dict[str, Any]) -> list[dict[str, Any]]:
    findings = acceptance.get("visual_findings") if isinstance(acceptance, dict) else {}
    if not isinstance(findings, dict):
        findings = {}

    darkness = str(findings.get("darkness", "")).strip().lower()
    stroke_weight = str(findings.get("stroke_weight", "")).strip().lower()
    sharpness = str(findings.get("sharpness", "")).strip().lower()
    background = str(findings.get("background", "")).strip().lower()
    text = "\n".join(acceptance_text_fragments(acceptance)).lower()
    patches: list[dict[str, Any]] = []

    if darkness == "too_dark" or stroke_weight == "too_bold":
        patches.extend(
            [
                {
                    "font_size_delta": -1,
                    "blur_delta": -0.08,
                    "alpha_contrast_delta": 0.20,
                    "core_ink_gain_delta": -0.10,
                    "core_darken_strength_delta": 0.04,
                    "core_darken_threshold_delta": 10,
                },
                {"core_darken_strength_delta": -0.04},
                {"core_ink_gain_delta": -0.06, "core_darken_strength_delta": -0.04},
                {"opacity_delta": -0.05, "core_ink_gain_delta": -0.06, "core_darken_strength_delta": -0.04},
                {"opacity_delta": -0.06, "core_ink_gain_delta": -0.10, "core_darken_strength_delta": -0.08},
                {
                    "opacity_delta": -0.06,
                    "core_ink_gain_delta": -0.15,
                    "core_darken_strength_delta": -0.12,
                    "blur_delta": 0.10,
                },
                {
                    "opacity_delta": -0.06,
                    "core_ink_gain_delta": -0.15,
                    "core_darken_strength_delta": -0.15,
                    "blur_delta": 0.15,
                },
            ]
        )
    elif darkness == "too_light" or stroke_weight == "too_thin":
        patches.extend(
            [
                {"core_darken_strength_delta": 0.04},
                {"core_ink_gain_delta": 0.06, "core_darken_strength_delta": 0.04},
                {"opacity_delta": 0.04, "core_ink_gain_delta": 0.06},
            ]
        )

    if sharpness == "too_sharp":
        patches.append({"blur_delta": 0.08, "opacity_delta": -0.03})
        patches.append(
            {
                "blur_delta": 0.08,
                "edge_breakup_delta": 0.012,
                "photo_noise_delta": 0.030,
                "jpeg_quality_delta": -8,
            }
        )
    elif sharpness == "too_blurry":
        patches.append({"blur_delta": -0.08, "opacity_delta": 0.03})

    if (
        background == "ghost_visible"
        or "残影" in text
        or "旧字" in text
        or "ghost_visible" in text
    ):
        patches.extend(
            [
                {
                    "mask_threshold_delta": 12,
                    "mask_dilate_iterations_delta": 1,
                    "inpaint_radius_delta": 1,
                    "photo_noise_delta": -0.018,
                    "edge_breakup_delta": -0.010,
                    "jpeg_quality_delta": 8,
                },
                {
                    "mask_threshold_delta": 20,
                    "mask_dilate_iterations_delta": 1,
                    "inpaint_radius_delta": 1,
                    "photo_noise_delta": -0.024,
                    "edge_breakup_delta": -0.012,
                    "jpeg_quality_delta": 10,
                },
                {
                    "mask_dilate_iterations_delta": 1,
                    "inpaint_radius_delta": 1,
                    "photo_noise_delta": -0.020,
                    "edge_breakup_delta": -0.010,
                    "jpeg_quality_delta": 8,
                },
            ]
        )
    elif (
        background in {"patch_visible", "too_smooth"}
        or "补丁" in text
        or "平滑" in text
        or "涂抹" in text
        or "patch_visible" in text
    ):
        patches.extend(
            [
                {"photo_noise_delta": 0.012, "edge_breakup_delta": 0.006},
                {"photo_noise_delta": 0.018, "edge_breakup_delta": 0.008, "jpeg_quality_delta": -4},
                {"inpaint_radius_delta": -1, "photo_noise_delta": 0.014, "edge_breakup_delta": 0.006},
                {"mask_dilate_iterations_delta": -1, "photo_noise_delta": 0.016, "edge_breakup_delta": 0.006},
                {"photo_noise_delta": 0.030, "edge_breakup_delta": 0.010, "jpeg_quality_delta": -8},
            ]
        )

    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for patch in patches:
        key = json.dumps(patch, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(patch)
    return unique[:7]


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
