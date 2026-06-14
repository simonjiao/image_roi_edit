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


def historical_target_completion(
    target: dict[str, Any] | None,
    acceptance: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(target, dict) or not target.get("active"):
        return {"enabled": False}
    closure = _closure_by_axis(acceptance)
    required = [
        str(axis.get("axis"))
        for axis in target.get("axes") or []
        if isinstance(axis, dict) and axis.get("axis")
    ]
    closed_axes = [axis for axis in required if bool((closure.get(axis) or {}).get("closed"))]
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
        "reason": (
            "all historical false-pass axes are explicitly closed"
            if not missing_axes
            else "final acceptance did not explicitly close historical false-pass axes"
        ),
    }


def apply_historical_false_pass_gate(
    acceptance: dict[str, Any] | None,
    target: dict[str, Any] | None,
) -> dict[str, Any]:
    gated = copy.deepcopy(acceptance) if isinstance(acceptance, dict) else {}
    if not isinstance(target, dict) or not target.get("active"):
        return gated
    completion = historical_target_completion(target, gated)
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
