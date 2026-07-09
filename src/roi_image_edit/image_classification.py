from __future__ import annotations

import re
from typing import Any

import cv2
import numpy as np
from PIL import Image

from roi_image_edit.iterative_pipeline import is_mostly_cjk
from roi_image_edit.roi_locator import (
    group_page_text_lines,
    page_text_components,
    text_chars,
)


NUMERIC_DATE_RE = re.compile(r"^[\d\s:：/\\.,，.年月日号编号第-]+$")


def length_change_for_texts(source_text: str, target_text: str) -> str:
    source_count = len(text_chars(source_text))
    target_count = len(text_chars(target_text))
    if source_count and target_count > source_count:
        return "longer"
    if source_count and target_count == 0:
        return "removed"
    if source_count and target_count < source_count:
        return "shorter"
    if source_count and target_count:
        return "same"
    return "unknown"


def script_for_texts(source_text: str, target_text: str, field_key: str | None = None) -> str:
    combined = f"{source_text}{target_text}".strip()
    if field_key in {"date", "age", "number", "receive_time", "amount"}:
        return "numeric_or_date"
    if combined and NUMERIC_DATE_RE.match(combined):
        return "numeric_or_date"
    if is_mostly_cjk(combined):
        return "cjk"
    if any("a" <= char.lower() <= "z" for char in combined):
        return "latin"
    return "mixed"


def image_layout_evidence(image: Image.Image) -> dict[str, Any]:
    rgb = image.convert("RGB")
    arr = np.array(rgb)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    components = page_text_components(rgb)
    lines = group_page_text_lines(components)
    dark_mask = gray < 180
    dark_ratio = float(np.count_nonzero(dark_mask)) / max(1, gray.size)
    channel_spread = np.max(arr.astype(np.int16), axis=2) - np.min(arr.astype(np.int16), axis=2)
    background = gray[~dark_mask]
    background_std = float(np.std(background)) if background.size else float(np.std(gray))
    return {
        "width": int(rgb.width),
        "height": int(rgb.height),
        "area": int(rgb.width * rgb.height),
        "component_count": int(len(components)),
        "line_count": int(len(lines)),
        "dark_pixel_ratio": round(dark_ratio, 5),
        "background_gray_std": round(background_std, 3),
        "mean_channel_spread": round(float(np.mean(channel_spread)), 3),
    }


def _looks_low_res(evidence: dict[str, Any]) -> bool:
    return (
        min(int(evidence["width"]), int(evidence["height"])) <= 160
        or int(evidence["area"]) <= 30_000
    )


def _looks_clean_digital(evidence: dict[str, Any], script: str, field_key: str | None) -> bool:
    return (
        script == "numeric_or_date"
        and field_key in {"date", "age", "number", "receive_time", None}
        and float(evidence["background_gray_std"]) <= 16.0
        and float(evidence["mean_channel_spread"]) <= 8.0
        and int(evidence["line_count"]) <= 3
    )


def _scenario_for(
    *,
    image_type: str,
    script: str,
    field_key: str | None,
    evidence: dict[str, Any],
    source_explicit: bool,
    operation: str,
    removal_context: dict[str, Any] | None,
    redaction_context: dict[str, Any] | None = None,
) -> str:
    if operation == "amount_glyph_clone":
        return "amount_glyph_clone"
    if operation == "remove_text":
        context = removal_context if isinstance(removal_context, dict) else {}
        if context.get("anchor_relation") == "below" or context.get("anchor_text"):
            return "anchored_text_removal"
        return "text_removal"
    if operation == "redact_text":
        context = redaction_context if isinstance(redaction_context, dict) else {}
        if context.get("anchor_relation") == "below" or context.get("anchor_text"):
            return "anchored_text_redaction"
        return "text_redaction"
    if field_key == "amount" and script == "numeric_or_date":
        return "amount_value_replace"
    if image_type == "clean_digital" and script == "numeric_or_date":
        return "numeric_or_date_replace"
    if int(evidence.get("line_count") or 0) >= 4 and not field_key:
        return "dense_paragraph_replace"
    if field_key or source_explicit:
        return "form_field_value_replace"
    return "inline_text_replace"


def _class_key(image_type: str, scenario: str, script: str) -> str:
    if image_type == "clean_digital" and scenario == "numeric_or_date_replace":
        return "clean_digital.numeric_or_date_replace"
    return f"{image_type}.{scenario}.{script}"


def _internal_profile(image_type: str, roi_policy: str) -> str:
    if roi_policy == "manual_exact":
        return "manual_roi_quick"
    if roi_policy == "manual_anchor":
        return "photo_scan"
    if image_type == "clean_digital":
        return "clean_digital"
    if image_type == "low_res_thumbnail":
        return "low_res_thumbnail"
    return "photo_scan"


def _prompt_pack(image_type: str, scenario: str, script: str) -> str:
    if scenario in {"anchored_text_removal", "text_removal"}:
        return scenario
    if scenario in {"anchored_text_redaction", "text_redaction"}:
        return scenario
    if scenario == "amount_value_replace":
        return "amount_value_replace"
    if scenario == "amount_glyph_clone":
        return "amount_glyph_clone"
    if image_type == "clean_digital" and script == "numeric_or_date":
        return "clean_numeric_or_date_replace"
    if image_type == "low_res_thumbnail":
        return f"low_res_{scenario}"
    if scenario == "form_field_value_replace":
        return "form_field_value_replace"
    if scenario == "dense_paragraph_replace":
        return "dense_paragraph_replace"
    return "inline_text_replace"


def _parameter_family(image_type: str, roi_policy: str, scenario: str) -> str:
    if scenario in {"anchored_text_removal", "text_removal"}:
        if image_type == "clean_digital":
            return "clean_digital_text_removal"
        return "photo_document_text_removal"
    if scenario in {"anchored_text_redaction", "text_redaction"}:
        if image_type == "clean_digital":
            return "clean_digital_text_redaction"
        return "photo_document_text_redaction"
    if scenario == "amount_value_replace":
        return "clean_digital_amount_value_replace"
    if scenario == "amount_glyph_clone":
        return "clean_digital_amount_glyph_clone"
    if roi_policy == "manual_exact":
        return "manual_roi_conservative"
    if image_type == "clean_digital":
        return "clean_digital_no_photo_texture"
    if image_type == "low_res_thumbnail":
        return "low_res_magnified_conservative"
    return "photo_document_scan"


def classify_image_workflow(
    image: Image.Image,
    *,
    instruction_details: dict[str, Any],
    regions: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
) -> dict[str, Any]:
    source_text = str(instruction_details.get("source_text") or "")
    target_text = str(instruction_details.get("target_text") or "")
    field_key = instruction_details.get("field_key")
    field_key = str(field_key) if field_key else None
    source_explicit = bool(instruction_details.get("source_explicit"))
    operation = str(instruction_details.get("operation") or "replace_text")
    removal_context = (
        instruction_details.get("removal_context")
        if isinstance(instruction_details.get("removal_context"), dict)
        else {}
    )
    redaction_context = (
        instruction_details.get("redaction_context")
        if isinstance(instruction_details.get("redaction_context"), dict)
        else {}
    )
    region_count = len(regions or [])
    roi_input = "manual" if region_count else "auto"
    initial_roi_policy = (
        "auto"
        if roi_input == "auto"
        else "manual_anchor"
        if source_explicit
        else "manual_exact"
    )
    evidence = image_layout_evidence(image)
    script = script_for_texts(source_text, target_text, field_key)
    if _looks_low_res(evidence):
        image_type = "low_res_thumbnail"
    elif _looks_clean_digital(evidence, script, field_key):
        image_type = "clean_digital"
    else:
        image_type = "photo_document"
    scenario = _scenario_for(
        image_type=image_type,
        script=script,
        field_key=field_key,
        evidence=evidence,
        source_explicit=source_explicit,
        operation=operation,
        removal_context=removal_context,
        redaction_context=redaction_context,
    )
    roi_policy = initial_roi_policy
    internal_profile = _internal_profile(image_type, roi_policy)
    if scenario in {"amount_value_replace", "amount_glyph_clone"}:
        internal_profile = "clean_digital"
    confidence = 0.88
    if image_type == "low_res_thumbnail":
        confidence = 0.82
    if roi_input == "manual":
        confidence = min(confidence, 0.78 if roi_policy == "manual_exact" else 0.84)
    return {
        "image_type": image_type,
        "scenario": scenario,
        "script": script,
        "operation": operation,
        "length_change": "redacted" if operation == "redact_text" else length_change_for_texts(source_text, target_text),
        "roi_input": roi_input,
        "roi_policy": roi_policy,
        "class_key": _class_key(image_type, scenario, script),
        "prompt_pack": _prompt_pack(image_type, scenario, script),
        "parameter_family": _parameter_family(image_type, roi_policy, scenario),
        "confidence": round(confidence, 3),
        "evidence": {
            **evidence,
            "field_key": field_key,
            "source_explicit": source_explicit,
            "manual_region_count": int(region_count),
            "classifier_version": 1,
            "operation": operation,
            "removal_context": removal_context,
            "redaction_context": redaction_context,
        },
        "internal_profile": internal_profile,
        "profile_source": "classification",
    }


def classify_region_roi_policy(
    *,
    image_classification: dict[str, Any],
    search_roi: tuple[int, int, int, int],
    edit_roi: tuple[int, int, int, int],
    source_text: str,
) -> str:
    if image_classification.get("roi_input") != "manual":
        return "auto"
    if not text_chars(source_text):
        return "manual_exact"
    search_area = max(1, (search_roi[2] - search_roi[0]) * (search_roi[3] - search_roi[1]))
    edit_area = max(1, (edit_roi[2] - edit_roi[0]) * (edit_roi[3] - edit_roi[1]))
    if edit_area / search_area <= 0.82 or edit_roi != search_roi:
        return "manual_anchor"
    return "manual_exact"


def with_region_roi_policy(
    image_classification: dict[str, Any],
    *,
    roi_policy: str,
    internal_profile: str | None = None,
) -> dict[str, Any]:
    profile = internal_profile or str(image_classification.get("internal_profile") or "photo_scan")
    if roi_policy == "manual_exact":
        profile = "manual_roi_quick"
    elif profile == "manual_roi_quick" and roi_policy == "manual_anchor":
        profile = "photo_scan"
    return {
        **image_classification,
        "roi_policy": roi_policy,
        "internal_profile": profile,
        "profile_source": "classification",
    }
