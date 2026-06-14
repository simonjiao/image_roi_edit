from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from roi_image_edit.stage_policy import STAGE_ORDER


DATA_DIR = Path(__file__).resolve().parent / "historical_cases"
DEFAULT_FALSE_PASS_CASES = DATA_DIR / "false_pass_cases.jsonl"
SHA256_HEX_LENGTH = 64

AXIS_STAGE = {
    "too_gray": "ink_gray_balance",
    "too_light": "ink_gray_balance",
    "too_dark": "ink_gray_balance",
    "stroke_too_bold": "text_shape",
    "font_mismatch": "text_shape",
    "shape_unnatural": "text_shape",
    "too_blurry": "photo_texture",
    "too_sharp": "photo_texture",
    "patch_visible": "background_cleanup",
    "ghost_visible": "background_cleanup",
    "white_specks": "background_cleanup",
}


def _clean(value: Any) -> str:
    return str(value or "").strip().lower()


def _text_count(value: str | None) -> int:
    return len([ch for ch in (value or "") if not ch.isspace()])


def _length_change(source_count: int, target_count: int) -> str:
    if target_count > source_count:
        return "longer"
    if target_count < source_count:
        return "shorter"
    return "same"


def normalize_history_axis(axis: str) -> str:
    key = _clean(axis).replace("-", "_")
    aliases = {
        "gray": "too_gray",
        "slightly_gray": "too_gray",
        "grey": "too_gray",
        "too_grey": "too_gray",
        "dark": "too_dark",
        "black": "too_dark",
        "too_black": "too_dark",
        "overblack": "too_dark",
        "over_black": "too_dark",
        "bold": "stroke_too_bold",
        "too_bold": "stroke_too_bold",
        "stroke_too_thick": "stroke_too_bold",
        "too_thick": "stroke_too_bold",
        "thick_stroke": "stroke_too_bold",
        "font_style_mismatch": "font_mismatch",
        "font_family_mismatch": "font_mismatch",
        "unnatural_shape": "shape_unnatural",
        "unnatural": "shape_unnatural",
        "glyph_unnatural": "shape_unnatural",
        "blurred": "too_blurry",
        "too_blur": "too_blurry",
        "visible_patch": "patch_visible",
        "white_dots": "white_specks",
    }
    return aliases.get(key, key)


def _hash_is_valid(value: Any) -> bool:
    text = str(value or "")
    return len(text) == SHA256_HEX_LENGTH and all(ch in "0123456789abcdef" for ch in text)


def validate_false_pass_case(case: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = (
        "case_id",
        "class_key",
        "length_change",
        "artifact_hashes",
        "observed",
        "user_rejection",
        "normalized_axes",
        "generalization_constraints",
        "expected",
    )
    for key in required:
        if key not in case:
            errors.append(f"missing:{key}")
    if _clean(case.get("length_change")) not in {"same", "longer", "shorter"}:
        errors.append("invalid:length_change")
    hashes = case.get("artifact_hashes")
    if not isinstance(hashes, dict):
        errors.append("invalid:artifact_hashes")
    else:
        for key in ("input_sha256", "output_sha256"):
            if not _hash_is_valid(hashes.get(key)):
                errors.append(f"invalid:{key}")
    axes = case.get("normalized_axes")
    if not isinstance(axes, list) or not axes:
        errors.append("invalid:normalized_axes")
    else:
        for axis in axes:
            normalized = normalize_history_axis(str(axis))
            if normalized not in AXIS_STAGE:
                errors.append(f"invalid:axis:{axis}")
    expected = case.get("expected")
    if not isinstance(expected, dict):
        errors.append("invalid:expected")
    else:
        stage = _clean(expected.get("blocking_stage"))
        if stage not in STAGE_ORDER:
            errors.append("invalid:expected.blocking_stage")
        try:
            max_steps = int(expected.get("max_steps"))
        except (TypeError, ValueError):
            max_steps = 0
        if max_steps <= 0:
            errors.append("invalid:expected.max_steps")
    constraints = case.get("generalization_constraints")
    if not isinstance(constraints, dict):
        errors.append("invalid:generalization_constraints")
    elif constraints.get("forbid_specific_text") is not True:
        errors.append("invalid:generalization_constraints.forbid_specific_text")
    return errors


def load_false_pass_cases(path: Path | None = None) -> list[dict[str, Any]]:
    case_path = path or DEFAULT_FALSE_PASS_CASES
    if not case_path.exists():
        return []
    cases: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(case_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{case_path}:{line_no}: false-pass case must be an object")
        errors = validate_false_pass_case(payload)
        if errors:
            raise ValueError(f"{case_path}:{line_no}: invalid false-pass case: {', '.join(errors)}")
        cases.append(payload)
    return cases


def _report_class_key(report: dict[str, Any] | None) -> str:
    if not isinstance(report, dict):
        return ""
    classification = report.get("classification")
    if isinstance(classification, dict) and classification.get("class_key"):
        return str(classification.get("class_key"))
    return str(report.get("class_key") or "")


def _report_counts(report: dict[str, Any] | None, plan: Any | None) -> tuple[int, int]:
    roi_plan = report.get("roi_plan") if isinstance(report, dict) else None
    if isinstance(roi_plan, dict):
        try:
            source = int(roi_plan.get("source_slot_count") or 0)
            target = int(roi_plan.get("target_slot_count") or 0)
            if source or target:
                return source, target
        except (TypeError, ValueError):
            pass
    return _text_count(getattr(plan, "source_text", "")), _text_count(getattr(plan, "target_text", ""))


def false_pass_case_matches(case: dict[str, Any], report: dict[str, Any] | None, plan: Any | None = None) -> bool:
    if _report_class_key(report) != str(case.get("class_key") or ""):
        return False
    source_count, target_count = _report_counts(report, plan)
    if _length_change(source_count, target_count) != _clean(case.get("length_change")):
        return False
    constraints = case.get("generalization_constraints")
    if isinstance(constraints, dict):
        expected_source = constraints.get("source_text_count")
        expected_target = constraints.get("target_text_count")
        if expected_source is not None and int(expected_source) != source_count:
            return False
        if expected_target is not None and int(expected_target) != target_count:
            return False
    return True


def matching_false_pass_cases(
    report: dict[str, Any] | None,
    plan: Any | None = None,
    *,
    cases: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    source_cases = cases if cases is not None else load_false_pass_cases()
    return [case for case in source_cases if false_pass_case_matches(case, report, plan)]


def historical_false_pass_target(
    report: dict[str, Any] | None,
    plan: Any | None = None,
    *,
    cases: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    matches = matching_false_pass_cases(report, plan, cases=cases)
    if not matches:
        return {"active": False, "source": "historical_false_pass", "cases": []}
    axes: list[dict[str, Any]] = []
    seen_axes: set[str] = set()
    stage_targets: set[str] = set()
    max_steps = 0
    case_ids: list[str] = []
    for case in matches:
        case_ids.append(str(case.get("case_id")))
        expected = case.get("expected") if isinstance(case.get("expected"), dict) else {}
        try:
            max_steps = max(max_steps, int(expected.get("max_steps") or 0))
        except (TypeError, ValueError):
            pass
        for axis_value in case.get("normalized_axes") or []:
            axis = normalize_history_axis(str(axis_value))
            if axis in seen_axes:
                continue
            seen_axes.add(axis)
            stage = AXIS_STAGE.get(axis, str(expected.get("blocking_stage") or "ink_gray_balance"))
            stage_targets.add(stage)
            axes.append(
                {
                    "axis": axis,
                    "stage": stage,
                    "source": "historical_false_pass",
                    "required_closure": True,
                }
            )
    primary_stage = next((stage for stage in STAGE_ORDER if stage in stage_targets), None)
    return {
        "active": True,
        "source": "historical_false_pass",
        "stage": primary_stage,
        "stage_source": "historical_false_pass",
        "case_ids": case_ids,
        "axes": axes,
        "axis_keys": [axis["axis"] for axis in axes],
        "stage_targets": sorted(stage_targets),
        "max_steps": max_steps or 3,
        "reason": "versioned human-rejected false-pass case matched this workflow class",
    }


def _closure_by_axis(acceptance: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(acceptance, dict):
        return {}
    raw = acceptance.get("historical_target_closure")
    if raw is None:
        raw = acceptance.get("vision_target_closure")
    if not isinstance(raw, dict):
        return {}
    axis_items = raw.get("axes")
    if isinstance(axis_items, dict):
        return {
            normalize_history_axis(str(axis)): dict(value)
            for axis, value in axis_items.items()
            if isinstance(value, dict)
        }
    if isinstance(axis_items, list):
        result: dict[str, dict[str, Any]] = {}
        for item in axis_items:
            if not isinstance(item, dict):
                continue
            axis = normalize_history_axis(str(item.get("axis") or ""))
            if axis:
                result[axis] = dict(item)
        return result
    return {}


def _report_issue_types(report: dict[str, Any] | None) -> set[str]:
    if not isinstance(report, dict):
        return set()
    issues: list[Any] = []
    for key in (
        "local_stroke_body_issues",
        "local_neighbor_style_issues",
        "local_pose_issues",
        "local_ink_balance_issues",
        "local_photo_texture_issues",
        "local_background_texture_issues",
        "local_background_cleanup_issues",
    ):
        value = report.get(key)
        if isinstance(value, list):
            issues.extend(value)
    strict_gate = report.get("strict_gate")
    if isinstance(strict_gate, dict) and isinstance(strict_gate.get("issues"), list):
        issues.extend(strict_gate["issues"])
    stage_gate = report.get("stage_gate")
    if isinstance(stage_gate, dict):
        for stage in stage_gate.get("stages", []):
            if isinstance(stage, dict) and isinstance(stage.get("issues"), list):
                issues.extend(stage["issues"])
    return {str(issue.get("type") or "") for issue in issues if isinstance(issue, dict)}


def _nested_dict(report: dict[str, Any] | None, key: str) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    value = report.get(key)
    return value if isinstance(value, dict) else {}


def _roi_counts(report: dict[str, Any] | None) -> tuple[int, int]:
    roi_plan = _nested_dict(report, "roi_plan")
    try:
        return int(roi_plan.get("source_slot_count") or 0), int(roi_plan.get("target_slot_count") or 0)
    except (TypeError, ValueError):
        return 0, 0


def _is_cjk_text(value: Any) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in str(value or ""))


def _axis_local_gray_status(report: dict[str, Any] | None) -> dict[str, Any]:
    issue_types = _report_issue_types(report)
    blocking_issue_types = {
        "changed_char_core_too_gray",
        "changed_char_core_too_light",
        "core_mean_gray_too_light",
        "core_lighten_too_high",
        "ink_too_light",
        "longer_mid_gray_body_too_black",
    }
    if issue_types & blocking_issue_types:
        return {
            "available": True,
            "closed": False,
            "reason": "local ink-gray issue is still open",
            "issue_types": sorted(issue_types & blocking_issue_types),
        }

    metrics = _nested_dict(report, "strict_visual_metrics")
    bands = metrics.get("bands") if isinstance(metrics.get("bands"), dict) else {}
    if not bands:
        return {"available": False, "closed": False, "reason": "missing strict gray-band metrics"}

    try:
        old_lt55 = float(bands.get("old_lt55_pixels") or 0.0)
        old_lt90 = float(bands.get("old_lt90_pixels") or 0.0)
        lt55_delta = float(bands.get("lt55_delta") or 0.0)
        lt90_delta = float(bands.get("lt90_delta") or 0.0)
        old_share = float(bands.get("old_lt55_share_of_lt165") or 0.0)
        new_share = float(bands.get("new_lt55_share_of_lt165") or 0.0)
    except (TypeError, ValueError):
        return {"available": False, "closed": False, "reason": "invalid strict gray-band metrics"}

    source_count, target_count = _roi_counts(report)
    if source_count and target_count > source_count and old_lt90 >= 64.0:
        text_count_ratio = target_count / float(source_count)
        expected_lt90_delta = old_lt90 * (text_count_ratio - 1.0)
        excess_lt90_delta = lt90_delta - expected_lt90_delta
        excess_limit = max(72.0, old_lt90 * 0.20)
        share_drop = old_share - new_share
        core_not_recovered = lt55_delta < max(16.0, old_lt55 * 0.10)
        if excess_lt90_delta > excess_limit and core_not_recovered:
            return {
                "available": True,
                "closed": False,
                "reason": "mid-gray body increased beyond length-adjusted expectation without matching true-black core",
                "lt90_delta": round(lt90_delta, 3),
                "expected_lt90_delta": round(expected_lt90_delta, 3),
                "excess_lt90_delta": round(excess_lt90_delta, 3),
                "limit": round(excess_limit, 3),
                "lt55_delta": round(lt55_delta, 3),
                "core_share_drop": round(share_drop, 4),
            }
        if share_drop > 0.025 and excess_lt90_delta > max(56.0, old_lt90 * 0.16):
            return {
                "available": True,
                "closed": False,
                "reason": "true-black share dropped while mid-gray body expanded",
                "core_share_drop": round(share_drop, 4),
                "excess_lt90_delta": round(excess_lt90_delta, 3),
            }

    char_bands = _nested_dict(report, "char_gray_band_metrics")
    per_char = char_bands.get("per_char") if isinstance(char_bands.get("per_char"), list) else []
    for item in per_char:
        if not isinstance(item, dict):
            continue
        if item.get("source_char") == item.get("target_char"):
            continue
        if not _is_cjk_text(item.get("target_char")):
            continue
        old = item.get("old") if isinstance(item.get("old"), dict) else {}
        delta = item.get("delta") if isinstance(item.get("delta"), dict) else {}
        try:
            old_lt55 = float(old.get("lt55") or 0.0)
            core_deficit = -float(delta.get("lt55") or 0.0)
            gray_haze_delta = float(delta.get("band_55_70") or 0.0) + max(
                0.0,
                float(delta.get("band_70_90") or 0.0),
            )
        except (TypeError, ValueError):
            continue
        if old_lt55 >= 32.0 and core_deficit > max(36.0, old_lt55 * 0.35) and gray_haze_delta > max(
            28.0, old_lt55 * 0.24
        ):
            return {
                "available": True,
                "closed": False,
                "reason": "changed CJK glyph still has gray haze and insufficient true-black core",
                "index": item.get("index"),
                "core_deficit": round(core_deficit, 3),
                "gray_haze_delta": round(gray_haze_delta, 3),
            }

    return {"available": True, "closed": True, "reason": "local gray-band metrics closed the historical gray axis"}


def _axis_local_dark_status(report: dict[str, Any] | None) -> dict[str, Any]:
    issue_types = _report_issue_types(report)
    blocking_issue_types = {
        "changed_char_core_too_black",
        "changed_char_core_too_black_hard",
        "changed_char_deep_gray_too_dark",
        "changed_char_mid_gray_too_black",
        "longer_mid_gray_body_too_black",
        "roi_black_core_share_too_high",
        "roi_core_too_black",
    }
    if issue_types & blocking_issue_types:
        return {
            "available": True,
            "closed": False,
            "reason": "local ink-darkness issue is still open",
            "issue_types": sorted(issue_types & blocking_issue_types),
        }

    metrics = _nested_dict(report, "strict_visual_metrics")
    bands = metrics.get("bands") if isinstance(metrics.get("bands"), dict) else {}
    if isinstance(bands, dict) and bands:
        try:
            old_lt55 = float(bands.get("old_lt55_pixels") or 0.0)
            lt55_delta = float(bands.get("lt55_delta") or 0.0)
            old_share = float(bands.get("old_lt55_share_of_lt165") or 0.0)
            new_share = float(bands.get("new_lt55_share_of_lt165") or 0.0)
        except (TypeError, ValueError):
            return {"available": False, "closed": False, "reason": "invalid strict darkness metrics"}
        if old_lt55 >= 32.0 and lt55_delta > max(58.0, old_lt55 * 0.36):
            return {
                "available": True,
                "closed": False,
                "reason": "ROI true-black core still exceeds local darkness limit",
                "lt55_delta": round(lt55_delta, 3),
                "limit": round(max(58.0, old_lt55 * 0.36), 3),
            }
        share_delta = new_share - old_share
        if share_delta > 0.085:
            return {
                "available": True,
                "closed": False,
                "reason": "ROI true-black share still exceeds local darkness limit",
                "share_delta": round(share_delta, 4),
                "limit": 0.085,
            }

    char_bands = _nested_dict(report, "char_gray_band_metrics")
    per_char = char_bands.get("per_char") if isinstance(char_bands.get("per_char"), list) else []
    source_count, target_count = _roi_counts(report)
    longer_replacement = bool(source_count and target_count > source_count)
    for item in per_char:
        if not isinstance(item, dict):
            continue
        if item.get("source_char") == item.get("target_char"):
            continue
        if not _is_cjk_text(item.get("target_char")):
            continue
        old = item.get("old") if isinstance(item.get("old"), dict) else {}
        delta = item.get("delta") if isinstance(item.get("delta"), dict) else {}
        try:
            old_lt55 = float(old.get("lt55") or 0.0)
            old_lt165 = float(old.get("lt165") or 0.0)
            lt55_delta = float(delta.get("lt55") or 0.0)
            band_55_70_delta = float(delta.get("band_55_70") or 0.0)
        except (TypeError, ValueError):
            continue
        lt70_delta = lt55_delta + band_55_70_delta
        deep_gray_limit = max(48.0, old_lt55 * 0.62)
        lt70_delta_limit = max(62.0, old_lt55 * 0.72)
        if (
            longer_replacement
            and old_lt55 >= 32.0
            and old_lt165 > 0.0
            and lt55_delta >= -8.0
            and band_55_70_delta > deep_gray_limit
            and lt70_delta > lt70_delta_limit
        ):
            return {
                "available": True,
                "closed": False,
                "reason": "changed CJK glyph still has excessive 55-70 near-black body",
                "index": item.get("index"),
                "band_55_70_delta": round(band_55_70_delta, 3),
                "limit": round(deep_gray_limit, 3),
                "lt70_delta": round(lt70_delta, 3),
                "lt70_delta_limit": round(lt70_delta_limit, 3),
            }

    if not bands and not per_char:
        return {"available": False, "closed": False, "reason": "missing local darkness metrics"}
    return {"available": True, "closed": True, "reason": "local darkness metrics closed the historical dark axis"}


def _axis_local_blur_status(report: dict[str, Any] | None) -> dict[str, Any]:
    issue_types = _report_issue_types(report)
    if "photo_texture_too_blurry" in issue_types:
        return {
            "available": True,
            "closed": False,
            "reason": "local photo texture issue is still too blurry",
            "issue_types": ["photo_texture_too_blurry"],
        }
    metrics = _nested_dict(report, "photo_texture_metrics")
    if not metrics:
        return {"available": False, "closed": False, "reason": "missing photo texture metrics"}
    try:
        edge_ratio = float(metrics.get("edge_laplacian_ratio") or 1.0)
        old_edge_mean = float(metrics.get("old_edge_laplacian_mean") or 0.0)
        params = metrics.get("params") if isinstance(metrics.get("params"), dict) else {}
        blur = float(params.get("blur") or 0.0)
    except (TypeError, ValueError):
        return {"available": False, "closed": False, "reason": "invalid photo texture metrics"}
    if old_edge_mean >= 60.0:
        limit = 0.72
    elif old_edge_mean >= 45.0:
        limit = 0.62
    else:
        limit = 0.24
    if old_edge_mean >= 45.0 and edge_ratio < limit and blur >= 0.45:
        return {
            "available": True,
            "closed": False,
            "reason": "edge sharpness ratio remains below historical false-pass closure limit",
            "edge_laplacian_ratio": round(edge_ratio, 4),
            "limit": round(limit, 4),
            "blur": round(blur, 3),
            "old_edge_laplacian_mean": round(old_edge_mean, 3),
        }
    return {"available": True, "closed": True, "reason": "local photo texture metrics closed the historical blur axis"}


TEXT_SHAPE_ISSUE_TYPES = {
    "font_category_not_preferred",
    "font_family_style_score_ratio",
    "font_render_style_score_ratio",
    "changed_char_alpha_stroke_body_too_bold",
    "changed_char_alpha_stroke_body_too_thin",
    "changed_char_alpha_stroke_body_slightly_bold",
    "changed_char_stroke_body_too_bold",
    "changed_char_stroke_body_too_small",
    "char_center_dx",
    "char_center_dy",
    "char_center_distance_delta",
    "replacement_center_y_range",
    "row_baseline_top_too_high",
    "row_baseline_bottom_too_low",
    "pose_mismatch",
}

STROKE_TOO_BOLD_ISSUE_TYPES = {
    "changed_char_alpha_stroke_body_too_bold",
    "changed_char_alpha_stroke_body_slightly_bold",
    "changed_char_stroke_body_too_bold",
}

FONT_MISMATCH_ISSUE_TYPES = {
    "font_category_not_preferred",
    "font_family_style_score_ratio",
    "font_render_style_score_ratio",
}


def _axis_local_stroke_too_bold_status(report: dict[str, Any] | None) -> dict[str, Any]:
    issue_types = _report_issue_types(report)
    if issue_types & STROKE_TOO_BOLD_ISSUE_TYPES:
        return {
            "available": True,
            "closed": False,
            "reason": "local text-shape issue still reports excessive stroke body weight",
            "issue_types": sorted(issue_types & STROKE_TOO_BOLD_ISSUE_TYPES),
        }
    metrics = _nested_dict(report, "stroke_body_shape_metrics")
    per_char = metrics.get("per_char") if isinstance(metrics.get("per_char"), list) else []
    ratios: list[float] = []
    for item in per_char:
        if not isinstance(item, dict) or not item.get("changed"):
            continue
        try:
            ratios.append(float(item.get("body_area_ratio") or 0.0))
        except (TypeError, ValueError):
            continue
    if ratios:
        max_ratio = max(ratios)
        mean_ratio = sum(ratios) / len(ratios)
        if max_ratio > 0.70 or mean_ratio > 0.64:
            return {
                "available": True,
                "closed": False,
                "reason": "changed glyph alpha body ratio remains above historical bold-stroke closure limit",
                "max_body_area_ratio": round(max_ratio, 4),
                "mean_body_area_ratio": round(mean_ratio, 4),
                "limits": {"max_body_area_ratio": 0.70, "mean_body_area_ratio": 0.64},
            }
        return {
            "available": True,
            "closed": True,
            "reason": "local stroke-body metrics closed the historical bold-stroke axis",
            "max_body_area_ratio": round(max_ratio, 4),
            "mean_body_area_ratio": round(mean_ratio, 4),
        }
    if report is None:
        return {"available": False, "closed": False, "reason": "missing local report for stroke shape axis"}
    return {"available": False, "closed": False, "reason": "missing stroke-body shape metrics"}


def _axis_local_font_mismatch_status(report: dict[str, Any] | None) -> dict[str, Any]:
    issue_types = _report_issue_types(report)
    if issue_types & FONT_MISMATCH_ISSUE_TYPES:
        return {
            "available": True,
            "closed": False,
            "reason": "local text-shape issue still reports font/style mismatch",
            "issue_types": sorted(issue_types & FONT_MISMATCH_ISSUE_TYPES),
        }
    font_gate = _nested_dict(report, "font_style_gate")
    if font_gate:
        if font_gate.get("pass") is False:
            return {
                "available": True,
                "closed": False,
                "reason": "font style gate is still open",
                "font_style_gate": font_gate,
            }
        return {"available": True, "closed": True, "reason": "font style gate closed the historical font mismatch axis"}
    if report is None:
        return {"available": False, "closed": False, "reason": "missing local report for font mismatch axis"}
    return {"available": True, "closed": True, "reason": "no local font mismatch issue remains open"}


def _axis_local_shape_unnatural_status(report: dict[str, Any] | None) -> dict[str, Any]:
    issue_types = _report_issue_types(report)
    open_shape_issues = issue_types & TEXT_SHAPE_ISSUE_TYPES
    if open_shape_issues:
        return {
            "available": True,
            "closed": False,
            "reason": "local text-shape issue remains open",
            "issue_types": sorted(open_shape_issues),
        }
    stage_gate = _nested_dict(report, "stage_gate")
    if stage_gate:
        stage_status = stage_gate.get("stage_status")
        if isinstance(stage_status, dict):
            text_shape = stage_status.get("text_shape")
            if isinstance(text_shape, dict) and text_shape.get("pass") is False:
                return {
                    "available": True,
                    "closed": False,
                    "reason": "text_shape stage is still open",
                    "text_shape": text_shape,
                }
    if report is None:
        return {"available": False, "closed": False, "reason": "missing local report for shape axis"}
    return {"available": True, "closed": True, "reason": "local text-shape metrics closed the historical shape axis"}


def _axis_local_status(axis: str, report: dict[str, Any] | None) -> dict[str, Any]:
    normalized = normalize_history_axis(axis)
    if normalized == "too_gray":
        return _axis_local_gray_status(report)
    if normalized == "too_dark":
        return _axis_local_dark_status(report)
    if normalized == "stroke_too_bold":
        return _axis_local_stroke_too_bold_status(report)
    if normalized == "font_mismatch":
        return _axis_local_font_mismatch_status(report)
    if normalized == "shape_unnatural":
        return _axis_local_shape_unnatural_status(report)
    if normalized == "too_blurry":
        return _axis_local_blur_status(report)
    if report is None:
        return {"available": False, "closed": True, "reason": "no local rule for this axis"}
    return {"available": True, "closed": True, "reason": "no local rule for this axis"}


def historical_target_completion(
    target: dict[str, Any] | None,
    acceptance: dict[str, Any] | None,
    *,
    report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(target, dict) or not target.get("active"):
        return {"enabled": False}
    closure = _closure_by_axis(acceptance)
    required = [
        str(axis.get("axis"))
        for axis in target.get("axes") or []
        if isinstance(axis, dict) and axis.get("axis")
    ]
    local_closure: dict[str, dict[str, Any]] = {}
    for axis in required:
        if report is not None:
            local_closure[axis] = _axis_local_status(axis, report)
    closed_axes = []
    for axis in required:
        vision_closed = bool((closure.get(axis) or {}).get("closed"))
        local = local_closure.get(axis)
        local_closed = True if local is None else bool(local.get("closed"))
        if vision_closed and local_closed:
            closed_axes.append(axis)
    missing_axes = [axis for axis in required if axis not in closed_axes]
    return {
        "enabled": True,
        "source": "historical_false_pass",
        "case_ids": target.get("case_ids") or [],
        "required_axes": required,
        "closed_axes": closed_axes,
        "missing_axes": missing_axes,
        "complete": not missing_axes,
        "closure": closure,
        "local_closure": local_closure,
        "reason": (
            "all historical false-pass axes are explicitly closed"
            if not missing_axes
            else "final acceptance did not explicitly close historical false-pass axes with local evidence"
        ),
    }


def apply_historical_false_pass_gate(
    acceptance: dict[str, Any] | None,
    target: dict[str, Any] | None,
    *,
    report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gated = copy.deepcopy(acceptance) if isinstance(acceptance, dict) else {}
    if not isinstance(target, dict) or not target.get("active"):
        return gated
    completion = historical_target_completion(target, gated, report=report)
    gated["historical_false_pass_target"] = target
    gated["historical_target_completion"] = completion
    if completion.get("complete"):
        return gated
    gated["pass"] = False
    gated["acceptance_level"] = "marginal"
    gated["final_decision"] = "revise"
    gated["blocking_stage"] = target.get("stage") or "ink_gray_balance"
    findings = gated.get("visual_findings")
    findings = dict(findings) if isinstance(findings, dict) else {}
    if "too_gray" in completion.get("missing_axes", []):
        findings["darkness"] = "too_light"
    if "too_dark" in completion.get("missing_axes", []):
        findings["darkness"] = "too_dark"
    if "stroke_too_bold" in completion.get("missing_axes", []):
        findings["stroke_weight"] = "too_bold"
    if "font_mismatch" in completion.get("missing_axes", []):
        findings["font_similarity"] = "mismatch"
    if "shape_unnatural" in completion.get("missing_axes", []):
        findings["shape"] = "unnatural"
    if "too_blurry" in completion.get("missing_axes", []):
        findings["sharpness"] = "too_blurry"
    if "too_sharp" in completion.get("missing_axes", []):
        findings["sharpness"] = "too_sharp"
    if any(axis in completion.get("missing_axes", []) for axis in ("patch_visible", "ghost_visible")):
        findings["background"] = "patch_visible"
    gated["visual_findings"] = findings
    must_fix = list(gated.get("must_fix") or [])
    must_fix.append(
        {
            "stage": gated["blocking_stage"],
            "issue": "historical_false_pass_axes_not_closed",
            "evidence": {
                "case_ids": target.get("case_ids") or [],
                "missing_axes": completion.get("missing_axes") or [],
            },
        }
    )
    gated["must_fix"] = must_fix
    reason = str(gated.get("reason") or "").strip()
    prefix = "Historical false-pass target is still open; local gate changed deliver to revise."
    gated["reason"] = f"{prefix} {reason}".strip()
    return gated
