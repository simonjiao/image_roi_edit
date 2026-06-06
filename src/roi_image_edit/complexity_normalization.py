from __future__ import annotations

from typing import Any


def text_complexity(text: str) -> float:
    """Estimate CJK character complexity by stroke density proxy.

    Returns a per-character complexity score where higher values mean more
    complex (denser) characters.
    """
    if not text:
        return 0.0
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return 0.0
    cjk_score = 0.0
    for ch in chars:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF:
            cjk_score += 1.0
            if 0x8000 <= cp <= 0x9FFF:
                cjk_score += 0.2
        elif 0x3400 <= cp <= 0x4DBF:
            cjk_score += 1.0
        else:
            cjk_score += 0.8
    return round(cjk_score / len(chars), 4)


def complexity_normalized_ink_limits(
    source_text: str,
    target_text: str,
    base_core_black_limit: float = 48.0,
) -> dict[str, Any]:
    source_count = len([ch for ch in source_text if not ch.isspace()])
    target_count = len([ch for ch in target_text if not ch.isspace()])
    source_complexity = text_complexity(source_text)
    target_complexity = text_complexity(target_text)
    text_count_ratio = target_count / max(source_count, 1)
    complexity_ratio = target_complexity / max(source_complexity, 0.01)

    normalized_core_black_limit = base_core_black_limit
    normalization_reason = "no_normalization_needed"

    if text_count_ratio > 1.0:
        normalized_core_black_limit = base_core_black_limit * text_count_ratio
        normalization_reason = "target_has_more_characters"
    if complexity_ratio > 1.05:
        normalized_core_black_limit = max(
            normalized_core_black_limit,
            base_core_black_limit * complexity_ratio,
        )
        normalization_reason = (
            f"{normalization_reason};target_is_more_complex"
            if normalization_reason != "no_normalization_needed"
            else "target_is_more_complex"
        )

    normalized_core_black_limit = round(min(normalized_core_black_limit, base_core_black_limit * 2.5), 3)

    return {
        "enabled": text_count_ratio > 1.0 or complexity_ratio > 1.05,
        "source_text_count": source_count,
        "target_text_count": target_count,
        "text_count_ratio": round(text_count_ratio, 4),
        "source_complexity": source_complexity,
        "target_complexity": target_complexity,
        "complexity_ratio": round(complexity_ratio, 4),
        "base_core_black_limit": base_core_black_limit,
        "normalized_core_black_limit": normalized_core_black_limit,
        "normalization_reason": normalization_reason,
        "hard_boundary_not_relaxed": True,
        "protected_text_not_relaxed": True,
        "slot_quality_not_relaxed": True,
        "outside_roi_not_relaxed": True,
        "old_text_residual_not_relaxed": True,
        "affects_only_ink_quality_thresholds": True,
    }
