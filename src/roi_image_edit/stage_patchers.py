from __future__ import annotations

from dataclasses import asdict
import json
import re
from typing import Any

from roi_image_edit.iterative_pipeline import CandidateParams
from roi_image_edit.local_validation import (
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
from roi_image_edit.stage_policy import (
    STAGE_ORDER,
    optimization_policy_audit,
    optimization_steps_for_patch,
)


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


def patch_allowed_for_stage(patch: dict[str, Any] | None, stage_id: str | None) -> dict[str, Any]:
    audit = optimization_policy_audit(stage_id, patch)
    steps = optimization_steps_for_patch(patch)
    return {
        **audit,
        "stage_id": stage_id,
        "patch": patch or {},
        "optimization_steps": steps,
    }


def filter_patches_for_stage(
    patches: list[dict[str, Any]],
    stage_id: str | None,
    *,
    limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for patch in dedupe_patches(patches, 128):
        audit = patch_allowed_for_stage(patch, stage_id)
        if audit["allowed"]:
            accepted.append(patch)
        else:
            rejected.append(audit)
        if len(accepted) >= limit:
            break
    return accepted, rejected


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


def text_shape_patches(
    params: CandidateParams,
    acceptance: dict[str, Any],
    report: dict[str, Any] | None = None,
    *,
    rank_patch: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    patches: list[dict[str, Any]] = []
    if report_has_outer_gray_halo(report):
        patches.extend(neighbor_outer_gray_cleanup_patches())
        patches.extend(
            patch
            for patch in numeric_revision_patches(params, acceptance)
            if keep_patch_for_outer_gray_cleanup(patch)
        )
        if isinstance(rank_patch, dict) and keep_patch_for_outer_gray_cleanup(rank_patch):
            patches.append(rank_patch)
        return patches
    if local_neighbor_style_issues(report or {}, allow_excess_black_core=True):
        patches.extend(neighbor_core_density_recovery_patches())
        patches.extend(
            patch
            for patch in numeric_revision_patches(params, acceptance)
            if keep_patch_for_neighbor_core_recovery(patch)
        )
        if isinstance(rank_patch, dict) and keep_patch_for_neighbor_core_recovery(rank_patch):
            patches.append(rank_patch)
        return patches
    patches.extend(gray_stroke_recovery_patches())
    patches.extend(
        patch
        for patch in numeric_revision_patches(params, acceptance)
        if keep_patch_for_gray_stroke_recovery(patch)
    )
    if isinstance(rank_patch, dict) and keep_patch_for_gray_stroke_recovery(rank_patch):
        patches.append(rank_patch)
    return patches


def ink_gray_balance_patches(
    params: CandidateParams,
    acceptance: dict[str, Any],
    report: dict[str, Any] | None = None,
    *,
    rank_patch: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    patches: list[dict[str, Any]] = []
    patches.extend(alignment_centering_patches(report))
    if report_has_excess_black_core(report):
        patches.extend(black_core_reduction_patches())
    else:
        patches.extend(ink_balance_recovery_patches(report))
    patches.extend(numeric_revision_patches(params, acceptance))
    if isinstance(rank_patch, dict):
        patches.append(rank_patch)
    patches.extend(final_revision_patches(acceptance))
    return patches


def photo_texture_patches(
    params: CandidateParams,
    acceptance: dict[str, Any],
    report: dict[str, Any] | None = None,
    *,
    rank_patch: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    patches: list[dict[str, Any]] = []
    patches.extend(alignment_centering_patches(report))
    patches.extend(photo_texture_recovery_patches(report))
    patches.extend(final_revision_patches(acceptance))
    if isinstance(rank_patch, dict):
        patches.append(rank_patch)
    return patches


def background_cleanup_patches(
    params: CandidateParams,
    acceptance: dict[str, Any],
    report: dict[str, Any] | None = None,
    *,
    rank_patch: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    patches: list[dict[str, Any]] = []
    patches.extend(background_cleanup_recovery_patches(report))
    patches.extend(visual_background_cleanup_patches(acceptance))
    patches.extend(numeric_revision_patches(params, acceptance))
    patches.extend(photo_texture_recovery_patches(report))
    if isinstance(rank_patch, dict):
        patches.append(rank_patch)
    return patches


def final_acceptance_patches(
    params: CandidateParams,
    acceptance: dict[str, Any],
    report: dict[str, Any] | None = None,
    *,
    rank_patch: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    patches: list[dict[str, Any]] = []
    patches.extend(alignment_centering_patches(report))
    if report_needs_thinner_strokes(report):
        patches.extend(thin_dark_core_patches(acceptance))
    patches.extend(numeric_revision_patches(params, acceptance))
    if isinstance(rank_patch, dict):
        patches.append(rank_patch)
    patches.extend(final_revision_patches(acceptance))
    return patches


STAGE_PATCHERS = {
    "text_shape": text_shape_patches,
    "ink_gray_balance": ink_gray_balance_patches,
    "photo_texture": photo_texture_patches,
    "background_cleanup": background_cleanup_patches,
}


def revision_patches_for_round(
    params: CandidateParams,
    acceptance: dict[str, Any],
    report: dict[str, Any] | None = None,
    *,
    rank_patch: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    stage_gate = stage_gate_for_report(report) if isinstance(report, dict) else {}
    blocking_stage = stage_gate.get("blocking_stage") or acceptance_blocking_stage(acceptance)
    patcher_stage = str(blocking_stage) if blocking_stage in STAGE_PATCHERS else None
    if patcher_stage is None and report_has_excess_black_core(report):
        patcher_stage = "ink_gray_balance"
    if patcher_stage is None and report_needs_wider_gray_strokes(report):
        patcher_stage = "text_shape"
    patcher = STAGE_PATCHERS.get(patcher_stage) or final_acceptance_patches
    patches = patcher(params, acceptance, report, rank_patch=rank_patch)
    accepted, _rejected = filter_patches_for_stage(patches, patcher_stage, limit=12)
    return accepted


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
