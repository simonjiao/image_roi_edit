from __future__ import annotations

import json
from typing import Any

from roi_image_edit.prompt_assets import PROMPT_NAMES, load_prompt


ACTIVE_VISION_PROMPTS = (
    "candidate_rank_prompt.txt",
    "tuning_prompt.txt",
    "final_acceptance_prompt.txt",
)

RESERVED_DIAGNOSTIC_PROMPTS = (
    "font_size_prompt.txt",
    "darkness_blur_prompt.txt",
)

PROMPT_INPUT_CONTRACTS: dict[str, dict[str, tuple[str, ...]]] = {
    "candidate_rank_prompt.txt": {
        "formatted_payloads": ("hard_check_report",),
        "runtime_context_fields": (
            "field_key",
            "field_label_text",
            "field_separator_text",
            "protected_texts",
            "source_text",
            "target_text",
            "search_roi",
            "target_roi",
            "slot_boxes",
            "protected_boxes",
            "text_angle_degrees",
        ),
        "stage_context_fields": (
            "stage_context_by_candidate",
            "blocking_stage",
            "stage_status",
            "pass_with_deferred",
            "deferred_issues",
            "deferred_to_stage",
            "allowed_patch_keys",
            "blocked_patch_keys",
            "stage_filter_contract",
        ),
    },
    "tuning_prompt.txt": {
        "formatted_payloads": ("current_params", "hard_check_report"),
        "runtime_context_fields": (
            "field_key",
            "field_label_text",
            "field_separator_text",
            "protected_texts",
            "source_text",
            "target_text",
            "search_roi",
            "target_roi",
            "slot_boxes",
            "protected_boxes",
            "text_angle_degrees",
        ),
        "stage_context_fields": (
            "stage_context",
            "blocking_stage",
            "stage_status",
            "pass_with_deferred",
            "deferred_issues",
            "deferred_to_stage",
            "allowed_patch_keys",
            "blocked_patch_keys",
            "profile_constraints",
        ),
    },
    "final_acceptance_prompt.txt": {
        "formatted_payloads": ("final_params", "hard_check_report"),
        "runtime_context_fields": (
            "field_key",
            "field_label_text",
            "field_separator_text",
            "protected_texts",
            "source_text",
            "target_text",
            "search_roi",
            "target_roi",
            "slot_boxes",
            "protected_boxes",
            "text_angle_degrees",
        ),
        "stage_context_fields": (
            "stage_context",
            "blocking_stage",
            "stage_status",
            "pass_with_deferred",
            "deferred_issues",
            "deferred_to_stage",
            "allowed_patch_keys",
            "blocked_patch_keys",
            "profile_constraints",
        ),
    },
    "font_size_prompt.txt": {
        "formatted_payloads": ("hard_check_report",),
        "runtime_context_fields": (),
        "stage_context_fields": ("stage_context", "blocking_stage"),
    },
    "darkness_blur_prompt.txt": {
        "formatted_payloads": ("hard_check_report",),
        "runtime_context_fields": (),
        "stage_context_fields": (
            "stage_context",
            "blocking_stage",
            "stage_status",
            "pass_with_deferred",
            "deferred_issues",
            "deferred_to_stage",
        ),
    },
    "master_prompt.txt": {
        "formatted_payloads": (),
        "runtime_context_fields": (),
        "stage_context_fields": (),
    },
}

PROMPT_OUTPUT_FIELD_HANDLING: dict[str, dict[str, str]] = {
    "candidate_rank_prompt.txt": {
        "pass": "record",
        "best_candidate": "execute",
        "blocking_stage": "validate",
        "stage_assessment": "validate",
        "stage_assessment.blocking_stage_exists": "validate",
        "stage_assessment.current_blocking_stage": "validate",
        "stage_assessment.suggestion_target_stage": "validate",
        "stage_assessment.basis": "validate",
        "direction": "record",
        "reason": "record",
        "rejected_candidates": "record",
        "rejected_candidates.候选编号或参数标签": "record_dynamic_map",
        "visual_findings": "record",
        "visual_findings.font_similarity": "record",
        "visual_findings.position": "record",
        "visual_findings.size": "record",
        "visual_findings.spacing": "record",
        "visual_findings.darkness": "record",
        "visual_findings.blur": "record",
        "visual_findings.background": "record",
        "suggested_patch": "execute",
        "suggested_patch.font_size_delta": "execute",
        "suggested_patch.text_dx_delta": "execute",
        "suggested_patch.text_dy_delta": "execute",
        "suggested_patch.char_offsets_delta": "execute",
        "suggested_patch.char_offsets_delta[].dx": "execute",
        "suggested_patch.char_offsets_delta[].dy": "execute",
        "suggested_patch.opacity_delta": "execute",
        "suggested_patch.blur_delta": "execute",
        "parameter_suggestions": "execute",
        "parameter_suggestions[].name": "execute",
        "parameter_suggestions[].from": "record",
        "parameter_suggestions[].to": "execute",
        "parameter_suggestions[].reason": "record",
        "local_metric_conflicts": "record",
    },
    "tuning_prompt.txt": {
        "pass": "validate",
        "blocking_stage": "validate",
        "stage_assessment": "validate",
        "stage_assessment.blocking_stage_exists": "validate",
        "stage_assessment.current_blocking_stage": "validate",
        "stage_assessment.suggestion_target_stage": "validate",
        "stage_assessment.basis": "validate",
        "direction": "record",
        "reason": "record",
        "visual_findings": "record",
        "visual_findings.char_positions": "record",
        "visual_findings.font_similarity": "record",
        "visual_findings.size": "record",
        "visual_findings.spacing": "record",
        "visual_findings.darkness": "record",
        "visual_findings.blur": "record",
        "visual_findings.background": "record",
        "suggested_patch": "execute",
        "suggested_patch.font_size_delta": "execute",
        "suggested_patch.text_dx_delta": "execute",
        "suggested_patch.text_dy_delta": "execute",
        "suggested_patch.char_offsets_delta": "execute",
        "suggested_patch.char_offsets_delta[].dx": "execute",
        "suggested_patch.char_offsets_delta[].dy": "execute",
        "suggested_patch.opacity_delta": "execute",
        "suggested_patch.blur_delta": "execute",
        "stop_iteration": "validate",
    },
    "final_acceptance_prompt.txt": {
        "pass": "validate",
        "acceptance_level": "validate",
        "reason": "record",
        "blocking_stage": "validate",
        "stage_assessment": "validate",
        "stage_assessment.blocking_stage_exists": "validate",
        "stage_assessment.current_blocking_stage": "validate",
        "stage_assessment.suggestion_target_stage": "validate",
        "stage_assessment.basis": "validate",
        "direction": "record",
        "hard_check": "ignore_model_claim",
        "hard_check.size_match": "ignore_model_claim",
        "hard_check.outside_roi_unchanged": "ignore_model_claim",
        "hard_check.border_unchanged": "ignore_model_claim",
        "hard_check.protected_text_unchanged": "ignore_model_claim",
        "visual_findings": "record",
        "visual_findings.char_positions": "record",
        "visual_findings.spacing": "record",
        "visual_findings.baseline": "record",
        "visual_findings.font_similarity": "record",
        "visual_findings.size": "record",
        "visual_findings.stroke_weight": "record",
        "visual_findings.darkness": "record",
        "visual_findings.sharpness": "record",
        "visual_findings.background": "record",
        "must_fix": "record",
        "optional_tuning": "record",
        "optional_tuning[].issue": "record",
        "optional_tuning[].suggestion": "record",
        "parameter_suggestions": "execute",
        "parameter_suggestions[].name": "execute",
        "parameter_suggestions[].from": "record",
        "parameter_suggestions[].to": "execute",
        "parameter_suggestions[].reason": "record",
        "local_metric_conflicts": "record",
        "historical_target_closure": "validate",
        "historical_target_closure.axes": "validate",
        "historical_target_closure.axes[].axis": "validate",
        "historical_target_closure.axes[].closed": "validate",
        "historical_target_closure.axes[].basis": "record",
        "final_decision": "validate",
    },
    "font_size_prompt.txt": {
        "blocking_stage": "reserved_not_called",
        "direction": "reserved_not_called",
        "best_font": "reserved_not_called",
        "best_font_size": "reserved_not_called",
        "reason": "reserved_not_called",
        "font_rejections": "reserved_not_called",
        "font_rejections.字体名称": "reserved_not_called_dynamic_map",
        "font_size_findings": "reserved_not_called",
        "font_size_findings.候选标签或字号": "reserved_not_called_dynamic_map",
        "suggested_next": "reserved_not_called",
        "suggested_next.font": "reserved_not_called",
        "suggested_next.font_size": "reserved_not_called",
    },
    "darkness_blur_prompt.txt": {
        "blocking_stage": "reserved_not_called",
        "direction": "reserved_not_called",
        "best_opacity": "reserved_not_called",
        "best_blur": "reserved_not_called",
        "reason": "reserved_not_called",
        "visual_findings": "reserved_not_called",
        "visual_findings.darkness": "reserved_not_called",
        "visual_findings.edge_gray": "reserved_not_called",
        "visual_findings.sharpness": "reserved_not_called",
        "visual_findings.integration": "reserved_not_called",
        "suggested_next": "reserved_not_called",
        "suggested_next.opacity": "reserved_not_called",
        "suggested_next.blur": "reserved_not_called",
    },
    "master_prompt.txt": {},
}


def extract_output_json_block(prompt_text: str) -> str | None:
    marker = prompt_text.find("输出 JSON")
    if marker < 0:
        return None
    start = prompt_text.find("{", marker)
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(prompt_text[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return prompt_text[start : index + 1]
    return None


def output_field_paths(value: Any, prefix: str = "") -> tuple[str, ...]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            paths.append(path)
            paths.extend(output_field_paths(child, path))
    elif isinstance(value, list) and value:
        paths.extend(output_field_paths(value[0], f"{prefix}[]"))
    return tuple(paths)


def prompt_output_field_paths(prompt_name: str) -> tuple[str, ...]:
    block = extract_output_json_block(load_prompt(prompt_name))
    if not block:
        return ()
    return output_field_paths(json.loads(block))


def prompt_io_contract_report() -> dict[str, Any]:
    prompts: dict[str, Any] = {}
    for name in PROMPT_NAMES:
        prompt_fields = set(prompt_output_field_paths(name))
        registered_fields = set(PROMPT_OUTPUT_FIELD_HANDLING.get(name, {}))
        prompts[name] = {
            "active": name in ACTIVE_VISION_PROMPTS,
            "reserved_diagnostic": name in RESERVED_DIAGNOSTIC_PROMPTS,
            "formatted_input_contract": PROMPT_INPUT_CONTRACTS.get(name, {}),
            "output_fields": sorted(prompt_fields),
            "registered_output_fields": sorted(registered_fields),
            "unregistered_output_fields": sorted(prompt_fields - registered_fields),
            "stale_registered_fields": sorted(registered_fields - prompt_fields),
            "output_field_handling": PROMPT_OUTPUT_FIELD_HANDLING.get(name, {}),
        }
    return {
        "active_vision_prompts": list(ACTIVE_VISION_PROMPTS),
        "reserved_diagnostic_prompts": list(RESERVED_DIAGNOSTIC_PROMPTS),
        "prompts": prompts,
    }
