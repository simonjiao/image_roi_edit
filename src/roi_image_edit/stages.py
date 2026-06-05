from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

from roi_image_edit.stage_policy import (
    OPTIMIZATION_STEP_KEYS,
    STAGE_LABELS,
    optimization_policy_for_stage,
)
from roi_image_edit.stage_profiles import StageProfile, stage_profile


StageDetector = Callable[[dict[str, Any]], "StageResult"]


@dataclass(frozen=True)
class StageResult:
    stage_id: str
    display_name: str
    passed: bool
    blocks_next: bool
    severity: str
    issues: tuple[dict[str, Any], ...]
    reason: str
    allowed_patch_keys: tuple[str, ...]
    blocked_patch_keys: tuple[str, ...]

    def as_report(self) -> dict[str, Any]:
        return {
            "id": self.stage_id,
            "label": self.display_name,
            "pass": self.passed,
            "blocks_next": self.blocks_next,
            "severity": self.severity,
            "issues": list(self.issues),
            "reason": self.reason,
            "allowed_patch_keys": list(self.allowed_patch_keys),
            "blocked_patch_keys": list(self.blocked_patch_keys),
        }


@dataclass(frozen=True)
class StageSpec:
    id: str
    display_name: str
    blocks_next: bool
    detect: StageDetector
    optimization_steps: tuple[str, ...]
    allowed_patch_keys: frozenset[str]
    blocked_patch_keys: frozenset[str]

    def as_report(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("detect", None)
        data["allowed_patch_keys"] = sorted(self.allowed_patch_keys)
        data["blocked_patch_keys"] = sorted(self.blocked_patch_keys)
        return data


def patch_keys_for_steps(steps: tuple[str, ...] | list[str]) -> frozenset[str]:
    keys: set[str] = set()
    for step in steps:
        keys.update(OPTIMIZATION_STEP_KEYS.get(str(step), set()))
    return frozenset(keys)


def _strict_stage_issues(report: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    from roi_image_edit.local_validation import strict_gate_stage_issues

    return strict_gate_stage_issues(report)


def _issue_tuple(issues: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    return tuple(issue for issue in issues if isinstance(issue, dict))


def _severity_label(issues: tuple[dict[str, Any], ...]) -> str:
    if not issues:
        return "pass"
    if len(issues) >= 3:
        return "high"
    return "medium"


def _stage_result(
    stage_id: str,
    issues: list[dict[str, Any]],
    *,
    allowed_patch_keys: frozenset[str],
    blocked_patch_keys: frozenset[str],
) -> StageResult:
    issue_items = _issue_tuple(issues)
    return StageResult(
        stage_id=stage_id,
        display_name=STAGE_LABELS[stage_id],
        passed=not issue_items,
        blocks_next=stage_spec(stage_id).blocks_next,
        severity=_severity_label(issue_items),
        issues=issue_items,
        reason="pass" if not issue_items else str(issue_items[0].get("type") or "stage_failed"),
        allowed_patch_keys=tuple(sorted(allowed_patch_keys)),
        blocked_patch_keys=tuple(sorted(blocked_patch_keys)),
    )


def detect_hard_boundary(report: dict[str, Any]) -> StageResult:
    spec = stage_spec("hard_boundary")
    issues: list[dict[str, Any]] = []
    if not report.get("pass"):
        hard_issues = report.get("issues")
        if isinstance(hard_issues, list):
            issues.extend(issue for issue in hard_issues if isinstance(issue, dict))
        if not issues:
            issues.append({"type": "hard_check_failed"})
    return _stage_result(
        "hard_boundary",
        issues,
        allowed_patch_keys=spec.allowed_patch_keys,
        blocked_patch_keys=spec.blocked_patch_keys,
    )


def detect_text_shape(report: dict[str, Any]) -> StageResult:
    from roi_image_edit.local_validation import (
        local_neighbor_style_issues,
        local_pose_issues,
        local_stroke_body_issues,
    )

    spec = stage_spec("text_shape")
    strict = _strict_stage_issues(report)
    font_style = report.get("font_style_gate")
    issues = list(strict.get("text_shape") or [])
    if isinstance(font_style, dict):
        issues.extend(issue for issue in font_style.get("issues", []) if isinstance(issue, dict))
    issues.extend(local_stroke_body_issues(report, allow_excess_black_core=True))
    issues.extend(local_neighbor_style_issues(report, allow_excess_black_core=True))
    issues.extend(local_pose_issues(report))
    return _stage_result(
        "text_shape",
        issues,
        allowed_patch_keys=spec.allowed_patch_keys,
        blocked_patch_keys=spec.blocked_patch_keys,
    )


def detect_ink_gray_balance(report: dict[str, Any]) -> StageResult:
    spec = stage_spec("ink_gray_balance")
    strict = _strict_stage_issues(report)
    issues = list(strict.get("ink_gray_balance") or [])
    issues.extend(issue for issue in report.get("local_ink_balance_issues", []) if isinstance(issue, dict))
    return _stage_result(
        "ink_gray_balance",
        issues,
        allowed_patch_keys=spec.allowed_patch_keys,
        blocked_patch_keys=spec.blocked_patch_keys,
    )


def detect_photo_texture(report: dict[str, Any]) -> StageResult:
    spec = stage_spec("photo_texture")
    issues = [issue for issue in report.get("local_photo_texture_issues", []) if isinstance(issue, dict)]
    return _stage_result(
        "photo_texture",
        issues,
        allowed_patch_keys=spec.allowed_patch_keys,
        blocked_patch_keys=spec.blocked_patch_keys,
    )


def detect_background_cleanup(report: dict[str, Any]) -> StageResult:
    spec = stage_spec("background_cleanup")
    strict = _strict_stage_issues(report)
    issues = list(strict.get("background_cleanup") or [])
    issues.extend(issue for issue in report.get("local_background_texture_issues", []) if isinstance(issue, dict))
    return _stage_result(
        "background_cleanup",
        issues,
        allowed_patch_keys=spec.allowed_patch_keys,
        blocked_patch_keys=spec.blocked_patch_keys,
    )


def _build_spec(stage_id: str, detect: StageDetector) -> StageSpec:
    policy = optimization_policy_for_stage(stage_id)
    allowed_steps = tuple(str(step) for step in policy.get("allowed_steps") or [])
    forbidden_steps = tuple(str(step) for step in policy.get("forbidden_steps") or [])
    return StageSpec(
        id=stage_id,
        display_name=STAGE_LABELS[stage_id],
        blocks_next=True,
        detect=detect,
        optimization_steps=allowed_steps,
        allowed_patch_keys=patch_keys_for_steps(allowed_steps),
        blocked_patch_keys=patch_keys_for_steps(forbidden_steps),
    )


STAGE_SPECS = {
    "hard_boundary": _build_spec("hard_boundary", detect_hard_boundary),
    "text_shape": _build_spec("text_shape", detect_text_shape),
    "ink_gray_balance": _build_spec("ink_gray_balance", detect_ink_gray_balance),
    "photo_texture": _build_spec("photo_texture", detect_photo_texture),
    "background_cleanup": _build_spec("background_cleanup", detect_background_cleanup),
}


def stage_spec(stage_id: str) -> StageSpec:
    try:
        return STAGE_SPECS[stage_id]
    except KeyError as exc:
        raise ValueError(f"unknown stage id: {stage_id}") from exc


def stage_specs(profile: StageProfile | str | None = None) -> tuple[StageSpec, ...]:
    profile_obj = profile if isinstance(profile, StageProfile) else stage_profile(profile)
    return tuple(STAGE_SPECS[stage_id] for stage_id in profile_obj.stage_order if stage_id in profile_obj.enabled_stage_ids)


def stage_results_for_report(
    report: dict[str, Any],
    profile: StageProfile | str | None = None,
) -> tuple[StageResult, ...]:
    return tuple(spec.detect(report) for spec in stage_specs(profile))


def blocking_stage_result(
    report: dict[str, Any],
    profile: StageProfile | str | None = None,
) -> StageResult | None:
    for result in stage_results_for_report(report, profile):
        if not result.passed and stage_spec(result.stage_id).blocks_next:
            return result
    return None


def stage_gate_for_report(
    report: dict[str, Any],
    profile: StageProfile | str | None = None,
) -> dict[str, Any]:
    profile_obj = profile if isinstance(profile, StageProfile) else stage_profile(profile)
    results = stage_results_for_report(report, profile_obj)
    blocking = next((result for result in results if not result.passed), None)
    stages = [result.as_report() for result in results]
    stage_status = {stage["id"]: stage for stage in stages}
    return {
        "profile": profile_obj.id,
        "profile_summary": profile_obj.as_report(),
        "order": [result.stage_id for result in results],
        "blocking_stage": blocking.stage_id if blocking else None,
        "blocking_stage_blocks_next": blocking.blocks_next if blocking else False,
        "pass": blocking is None,
        "stage_status": stage_status,
        "stages": stages,
    }


def prompt_stage_context(report: dict[str, Any], profile: StageProfile | str | None = None) -> dict[str, Any]:
    gate = stage_gate_for_report(report, profile)
    blocking_stage = gate.get("blocking_stage")
    spec = stage_spec(str(blocking_stage)) if blocking_stage else None
    return {
        "pipeline_profile": gate.get("profile"),
        "stage_order": gate.get("order"),
        "blocking_stage": blocking_stage,
        "blocking_stage_blocks_next": spec.blocks_next if spec else False,
        "stage_status": gate.get("stage_status"),
        "allowed_patch_keys": sorted(spec.allowed_patch_keys) if spec else [],
        "blocked_patch_keys": sorted(spec.blocked_patch_keys) if spec else [],
    }
