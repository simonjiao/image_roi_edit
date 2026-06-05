from __future__ import annotations

import json
from typing import Any


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
