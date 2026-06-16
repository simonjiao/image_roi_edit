from __future__ import annotations

from typing import Any


STAGE_ORDER = (
    "hard_boundary",
    "text_shape",
    "ink_gray_balance",
    "photo_texture",
    "background_cleanup",
)

STAGE_LABELS = {
    "hard_boundary": "ROI boundary and protected text",
    "text_shape": "font, size, slot, baseline, stroke body, local pose",
    "ink_gray_balance": "true-black core, mid-gray body, outer gray edge",
    "photo_texture": "scan blur, edge breakup, compression/noise texture",
    "background_cleanup": "inpaint texture and removed old slots",
}

# These are not stages. They are within-stage optimization steps used to
# classify model patches and locally generated candidate mutations.
OPTIMIZATION_STEP_KEYS = {
    "text_shape": {
        "font_size_delta",
        "text_dx_delta",
        "text_dy_delta",
        "char_offsets_delta",
    },
    "stroke_body_shape": {
        "stroke_opacity_delta",
    },
    "ink_gray_balance": {
        "opacity_delta",
        "ink_gain_delta",
        "alpha_contrast_delta",
        "core_ink_gain_delta",
        "core_darken_strength_delta",
        "core_darken_threshold_delta",
        "core_darken_target_gray_delta",
    },
    "photo_texture": {
        "blur_delta",
        "photo_warp_delta",
        "edge_breakup_delta",
        "photo_noise_delta",
        "jpeg_quality_delta",
    },
    "background_cleanup": {
        "mask_threshold_delta",
        "mask_dilate_iterations_delta",
        "inpaint_radius_delta",
    },
}


def patch_keys_for_steps(steps: tuple[str, ...] | list[str] | set[str] | frozenset[str]) -> frozenset[str]:
    keys: set[str] = set()
    for step in steps:
        keys.update(OPTIMIZATION_STEP_KEYS.get(str(step), set()))
    return frozenset(keys)


# Stage gates define ordering and blocking. Optimization policy defines which
# candidate mutation steps may run while a given stage is blocking.
STAGE_OPTIMIZATION_POLICY = {
    "hard_boundary": {
        "allowed_steps": [],
        "forbidden_steps": [
            "text_shape",
            "stroke_body_shape",
            "ink_gray_balance",
            "photo_texture",
            "background_cleanup",
        ],
        "reason": "Hard boundary failures must stop before visual or revision tuning.",
    },
    "text_shape": {
        "allowed_steps": ["text_shape", "stroke_body_shape"],
        "forbidden_steps": ["background_cleanup"],
        "secondary_only_steps": ["photo_texture"],
        "reason": "Shape must be solved by font, slot, baseline, pose, and stroke body before texture becomes dominant.",
    },
    "ink_gray_balance": {
        "allowed_steps": ["ink_gray_balance"],
        "forbidden_steps": ["text_shape", "background_cleanup"],
        "secondary_only_steps": ["photo_texture"],
        "reason": "Ink balance must solve true-black core, mid-gray body, and gray edge before photo texture dominates.",
    },
    "photo_texture": {
        "allowed_steps": ["photo_texture"],
        "forbidden_steps": ["text_shape", "stroke_body_shape", "ink_gray_balance", "background_cleanup"],
        "reason": "Photo texture is only allowed after shape and ink stages pass.",
    },
    "background_cleanup": {
        "allowed_steps": ["background_cleanup", "photo_texture"],
        "forbidden_steps": ["text_shape", "stroke_body_shape", "ink_gray_balance"],
        "reason": "Background cleanup must repair mask, inpaint, texture, ghosting, and seams without using new text to hide residue.",
    },
    "none": {
        "allowed_steps": [
            "text_shape",
            "stroke_body_shape",
            "ink_gray_balance",
            "photo_texture",
            "background_cleanup",
        ],
        "forbidden_steps": [],
        "reason": "No local blocking stage remains.",
    },
}


def optimization_steps_for_patch(patch: dict[str, Any] | None) -> list[str]:
    if not isinstance(patch, dict):
        return []
    keys = {str(key) for key, value in patch.items() if value is not None}
    steps: list[str] = []
    for step, step_keys in OPTIMIZATION_STEP_KEYS.items():
        if keys & step_keys:
            steps.append(step)
    return steps


def optimization_policy_for_stage(stage_id: str | None) -> dict[str, Any]:
    policy = STAGE_OPTIMIZATION_POLICY.get(stage_id or "none") or STAGE_OPTIMIZATION_POLICY["none"]
    return {
        "stage_id": stage_id or None,
        "stage_label": STAGE_LABELS.get(str(stage_id or "")),
        "allowed_steps": list(policy.get("allowed_steps") or []),
        "forbidden_steps": list(policy.get("forbidden_steps") or []),
        "secondary_only_steps": list(policy.get("secondary_only_steps") or []),
        "reason": policy.get("reason"),
    }


def optimization_policy_audit(stage_id: str | None, patch: dict[str, Any] | None) -> dict[str, Any]:
    policy = optimization_policy_for_stage(stage_id)
    steps = optimization_steps_for_patch(patch)
    allowed = set(policy["allowed_steps"])
    forbidden = set(policy["forbidden_steps"])
    secondary_only = set(policy.get("secondary_only_steps") or [])
    effective_steps = list(steps)
    if "photo_texture" in allowed and "photo_texture" in effective_steps:
        patch_keys = {str(key) for key, value in (patch or {}).items() if value is not None}
        if patch_keys and patch_keys <= OPTIMIZATION_STEP_KEYS["photo_texture"]:
            effective_steps = [
                step
                for step in effective_steps
                if step not in {"stroke_body_shape", "ink_gray_balance"}
            ]
    primary_steps = [step for step in effective_steps if step not in secondary_only]
    forbidden_hits = [step for step in effective_steps if step in forbidden]
    disallowed_primary = [
        step
        for step in primary_steps
        if step not in allowed and step not in secondary_only
    ]
    secondary_only_without_primary = bool(steps and not primary_steps and secondary_only.intersection(steps))
    is_allowed = not forbidden_hits and not disallowed_primary and not secondary_only_without_primary
    if not steps:
        is_allowed = True
    reason = "allowed"
    if forbidden_hits:
        reason = f"forbidden optimization steps for current stage: {', '.join(forbidden_hits)}"
    elif disallowed_primary:
        reason = f"primary optimization steps outside current stage: {', '.join(disallowed_primary)}"
    elif secondary_only_without_primary:
        reason = "secondary-only photo texture patch cannot be the main adjustment for this stage"
    return {
        **policy,
        "optimization_steps": steps,
        "effective_optimization_steps": effective_steps,
        "primary_optimization_steps": primary_steps,
        "optimization_step": selected_optimization_step(
            {
                "primary_optimization_steps": primary_steps,
                "effective_optimization_steps": effective_steps,
                "optimization_steps": steps,
            }
        ),
        "allowed": is_allowed,
        "rejection_reason": None if is_allowed else reason,
    }


def selected_optimization_step(optimization_report: dict[str, Any] | None) -> str | None:
    if not isinstance(optimization_report, dict):
        return None
    for key in ("primary_optimization_steps", "effective_optimization_steps", "optimization_steps", "allowed_steps"):
        steps = optimization_report.get(key)
        if not isinstance(steps, (list, tuple)):
            continue
        for step in steps:
            step_text = str(step or "").strip()
            if step_text:
                return step_text
    return None


def stage_optimization_summary(stage_id: str | None) -> dict[str, Any]:
    summary = optimization_policy_for_stage(stage_id)
    summary["optimization_step"] = selected_optimization_step(summary)
    return summary
