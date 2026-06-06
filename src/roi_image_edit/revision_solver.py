from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from roi_image_edit.iterative_pipeline import (
    CandidateParams,
    RenderPlan,
    default_char_offsets,
    dedupe_params,
    mutate_params,
)
from roi_image_edit.local_validation import (
    report_has_excess_black_core,
    report_has_fine_strokes_too_soft,
    report_has_outer_gray_halo,
    report_needs_thinner_strokes,
    report_needs_wider_gray_strokes,
    stage_issues,
)
from roi_image_edit.roi_locator import max_font_size_for_plan, text_chars
from roi_image_edit.stages import stage_gate_for_report


TEXT_SHAPE_GRID_ALLOWED_DELTA_KEYS = frozenset(
    {
        "font_name",
        "font_path",
        "font_size",
        "text_dx",
        "text_dy",
        "char_offsets",
        "stroke_opacity",
        "ink_gain",
        "alpha_contrast",
        "core_ink_gain",
        "core_darken_strength",
        "core_darken_threshold",
        "core_darken_target_gray",
    }
)
TEXT_SHAPE_GRID_BLOCKED_DELTA_KEYS = frozenset(
    {
        "opacity",
        "blur",
        "mask_threshold",
        "mask_dilate_iterations",
        "inpaint_radius",
        "photo_warp",
        "edge_breakup",
        "photo_noise",
        "jpeg_quality",
    }
)
TEXT_SHAPE_GRID_TOP_LIMIT = 48
TEXT_SHAPE_GRID_BUDGET_RANGE = (300, 1500)

INK_GRAY_GRID_ALLOWED_DELTA_KEYS = frozenset(
    {
        "opacity",
        "stroke_opacity",
        "ink_gain",
        "alpha_contrast",
        "core_ink_gain",
        "core_darken_strength",
        "core_darken_threshold",
        "core_darken_target_gray",
    }
)
INK_GRAY_GRID_BLOCKED_DELTA_KEYS = frozenset(
    {
        "font_name",
        "font_path",
        "font_size",
        "blur",
        "text_dx",
        "text_dy",
        "char_offsets",
        "mask_threshold",
        "mask_dilate_iterations",
        "inpaint_radius",
        "photo_warp",
        "edge_breakup",
        "photo_noise",
        "jpeg_quality",
    }
)
INK_GRAY_GRID_TOP_LIMIT = 16
INK_GRAY_GRID_BUDGET_RANGE = (100, 800)

PHOTO_TEXTURE_GRID_ALLOWED_DELTA_KEYS = frozenset(
    {
        "blur",
        "alpha_contrast",
        "photo_warp",
        "edge_breakup",
        "photo_noise",
        "jpeg_quality",
    }
)
PHOTO_TEXTURE_GRID_BLOCKED_DELTA_KEYS = frozenset(
    {
        "font_name",
        "font_path",
        "font_size",
        "opacity",
        "stroke_opacity",
        "ink_gain",
        "core_ink_gain",
        "core_darken_strength",
        "core_darken_threshold",
        "core_darken_target_gray",
        "text_dx",
        "text_dy",
        "char_offsets",
        "mask_threshold",
        "mask_dilate_iterations",
        "inpaint_radius",
    }
)
PHOTO_TEXTURE_GRID_TOP_LIMIT = 6
PHOTO_TEXTURE_GRID_BUDGET_RANGE = (30, 200)

TEXT_SHAPE_PRUNE_REASON_CATEGORIES = (
    "glyph_height",
    "center_alignment",
    "baseline_alignment",
    "character_spacing",
    "protected_distance",
    "font_style",
    "stroke_body",
    "pose_inheritance",
)
INK_GRAY_PRUNE_REASON_CATEGORIES = (
    "true_black_core",
    "deep_core_density",
    "outer_gray_edge",
    "mid_gray_body",
    "complexity_adjustment",
)
PHOTO_TEXTURE_PRUNE_REASON_CATEGORIES = (
    "over_sharp",
    "over_blurry",
    "missing_edge_breakup",
    "background_too_smooth",
    "white_or_shadow_ghost",
    "old_residual",
    "roi_gradient_break",
)

_PRUNE_REASON_SOURCES = {
    "text_shape": {
        "glyph_height": ("font_size", "slot_height", "max_font_size_for_plan"),
        "center_alignment": ("text_dx", "text_dy", "char_alignment_metrics.center"),
        "baseline_alignment": ("text_dy", "char_offsets", "char_alignment_metrics.baseline"),
        "character_spacing": ("char_offsets", "slot_boxes", "placement_strategy"),
        "protected_distance": ("protected_boxes", "target_roi", "right_boundary"),
        "font_style": ("font_name", "font_path", "font_style_reference.ranked_fonts"),
        "stroke_body": ("stroke_opacity", "ink_gain", "alpha_contrast", "core_ink_gain"),
        "pose_inheritance": ("char_offsets", "source_slot_shear", "neighbor_shear"),
    },
    "ink_gray_balance": {
        "true_black_core": ("opacity", "core_ink_gain", "core_darken_strength", "lt55"),
        "deep_core_density": ("core_darken_threshold", "core_darken_target_gray", "lt90"),
        "outer_gray_edge": ("stroke_opacity", "alpha_contrast", "outer_gray_halo"),
        "mid_gray_body": ("stroke_opacity", "ink_gain", "gray_band_metrics"),
        "complexity_adjustment": ("reference_profile", "target_source_complexity_ratio"),
    },
    "photo_texture": {
        "over_sharp": ("blur", "alpha_contrast", "photo_texture_too_sharp"),
        "over_blurry": ("blur", "photo_texture_too_blurry"),
        "missing_edge_breakup": ("edge_breakup", "photo_texture_edge_breakup_missing"),
        "background_too_smooth": ("photo_noise", "background_fill_too_smooth"),
        "white_or_shadow_ghost": ("background_white_ghost_residual", "background_shadow_ghost_residual"),
        "old_residual": ("background_trailing_patch_too_smooth", "cleanup_mask_report"),
        "roi_gradient_break": ("roi_gradient", "background_texture_metrics"),
    },
}


@dataclass(frozen=True)
class ShapeCandidateGrid:
    candidates: list[CandidateParams]
    report: dict[str, Any]


@dataclass(frozen=True)
class InkGrayCandidateGrid:
    candidates: list[CandidateParams]
    report: dict[str, Any]


@dataclass(frozen=True)
class PhotoTextureCandidateGrid:
    candidates: list[CandidateParams]
    report: dict[str, Any]


def prune_reason_contract(stage_id: str, raw_budget: int, retained_count: int) -> dict[str, Any]:
    sources = _PRUNE_REASON_SOURCES[stage_id]
    return {
        "stage_id": stage_id,
        "retention_method": "dedupe_then_axis_priority_top_n",
        "raw_candidate_budget": int(raw_budget),
        "retained_count": int(retained_count),
        "pruned_count": max(0, int(raw_budget) - int(retained_count)),
        "required_categories": list(sources),
        "category_sources": {category: list(value) for category, value in sources.items()},
    }


def _candidate_reason_categories(stage_id: str, delta_keys: frozenset[str]) -> list[str]:
    if stage_id == "text_shape":
        mapping = {
            "font_name": ("font_style",),
            "font_path": ("font_style",),
            "font_size": ("glyph_height", "protected_distance"),
            "text_dx": ("center_alignment",),
            "text_dy": ("center_alignment", "baseline_alignment"),
            "char_offsets": ("baseline_alignment", "character_spacing", "pose_inheritance"),
            "stroke_opacity": ("stroke_body",),
            "ink_gain": ("stroke_body",),
            "alpha_contrast": ("stroke_body",),
            "core_ink_gain": ("stroke_body",),
            "core_darken_strength": ("stroke_body",),
            "core_darken_threshold": ("stroke_body",),
            "core_darken_target_gray": ("stroke_body",),
        }
    elif stage_id == "ink_gray_balance":
        mapping = {
            "opacity": ("true_black_core", "deep_core_density"),
            "stroke_opacity": ("mid_gray_body", "outer_gray_edge"),
            "ink_gain": ("mid_gray_body",),
            "alpha_contrast": ("outer_gray_edge",),
            "core_ink_gain": ("true_black_core", "complexity_adjustment"),
            "core_darken_strength": ("true_black_core", "complexity_adjustment"),
            "core_darken_threshold": ("deep_core_density", "complexity_adjustment"),
            "core_darken_target_gray": ("deep_core_density", "complexity_adjustment"),
        }
    else:
        mapping = {
            "blur": ("over_sharp", "over_blurry"),
            "alpha_contrast": ("over_sharp",),
            "photo_warp": ("roi_gradient_break",),
            "edge_breakup": ("missing_edge_breakup", "roi_gradient_break"),
            "photo_noise": ("background_too_smooth", "old_residual"),
            "jpeg_quality": ("background_too_smooth", "old_residual"),
        }
    categories: set[str] = set()
    for key in delta_keys:
        categories.update(mapping.get(key, ()))
    return sorted(categories)


def layered_candidate_search_report(*grid_reports: dict[str, Any]) -> dict[str, Any]:
    stages: list[dict[str, Any]] = []
    parent_shape_trace: dict[str, Any] = {}
    for report in grid_reports:
        if not isinstance(report, dict):
            continue
        budget = report.get("budget") if isinstance(report.get("budget"), dict) else {}
        stage_id = report.get("stage_id")
        stage_record = {
            "stage_id": stage_id,
            "optimization_step": report.get("optimization_step"),
            "enabled": bool(report.get("enabled")),
            "candidate_count": int(report.get("candidate_count") or 0),
        }
        parent_shape_contract = report.get("parent_shape_contract")
        if isinstance(parent_shape_contract, dict) and stage_id:
            parent_shape_trace[str(stage_id)] = parent_shape_contract
            stage_record["parent_shape_candidate_id"] = parent_shape_contract.get(
                "parent_shape_candidate_id"
            )
            stage_record["parent_shape_stage_passed"] = parent_shape_contract.get(
                "parent_shape_stage_passed"
            )
        if budget:
            stage_record["raw_candidate_budget"] = int(budget.get("raw_candidate_budget") or 0)
            stage_record["retained_count"] = int(budget.get("retained_count") or 0)
            stage_record["pruned_count"] = int(budget.get("pruned_count") or 0)
            stage_record["within_budget"] = bool(budget.get("within_budget"))
        else:
            stage_record["reason"] = report.get("reason")
        stages.append(stage_record)

    return {
        "strategy": "layered_stage_search",
        "cross_stage_cartesian_search": False,
        "contract": "Generate and prune candidates within each blocking stage; do not multiply shape, ink-gray, and photo-texture axes into a single full Cartesian search.",
        "stage_order": [stage.get("stage_id") for stage in stages],
        "enabled_stage_ids": [
            stage.get("stage_id")
            for stage in stages
            if stage.get("enabled")
        ],
        "stages": stages,
        "raw_candidate_budget_by_stage": {
            str(stage.get("stage_id")): stage.get("raw_candidate_budget")
            for stage in stages
            if stage.get("enabled")
        },
        "retained_count_by_stage": {
            str(stage.get("stage_id")): stage.get("retained_count")
            for stage in stages
            if stage.get("enabled")
        },
        "parent_shape_trace": parent_shape_trace,
        "pruned_count_by_stage": {
            str(stage.get("stage_id")): stage.get("pruned_count")
            for stage in stages
            if stage.get("enabled")
        },
    }


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
    return tuple(candidates[:3])


def params_delta_keys(base: CandidateParams, candidate: CandidateParams) -> frozenset[str]:
    base_data = asdict(base)
    candidate_data = asdict(candidate)
    changed = {
        key
        for key, value in candidate_data.items()
        if key != "candidate_id" and value != base_data.get(key)
    }
    return frozenset(changed)


def shape_issue_flags(report: dict[str, Any] | None) -> dict[str, bool]:
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
    return {
        "outer_halo": has_outer_halo,
        "body_gap": has_body_gap,
        "pose_gap": has_pose_gap,
    }


def shape_stroke_body_grid(flags: dict[str, bool]) -> tuple[tuple[float, float, float, float, float], ...]:
    if flags.get("outer_halo"):
        return (
            (0.02, 0.02, 0.16, 0.12, 0.10),
            (0.04, 0.02, 0.14, 0.14, 0.12),
            (0.04, 0.03, 0.12, 0.16, 0.12),
            (0.06, 0.01, 0.20, 0.12, 0.10),
        )
    if flags.get("body_gap"):
        return (
            (0.03, 0.00, 0.18, 0.00, 0.00),
            (0.04, 0.00, 0.16, 0.04, 0.04),
            (0.04, 0.01, 0.14, 0.08, 0.06),
            (0.04, 0.04, 0.10, 0.18, 0.14),
        )
    return (
        (0.04, 0.03, 0.12, 0.16, 0.12),
        (0.04, 0.03, 0.10, 0.18, 0.14),
        (0.06, 0.02, 0.16, 0.14, 0.12),
    )


def text_shape_reset_candidate_grid(
    params: CandidateParams,
    font_style_reference: dict[str, Any],
    plan: RenderPlan,
    report: dict[str, Any] | None,
    *,
    limit: int = TEXT_SHAPE_GRID_TOP_LIMIT,
) -> ShapeCandidateGrid:
    if not report_blocks_text_shape(report):
        return ShapeCandidateGrid(
            candidates=[],
            report={
                "enabled": False,
                "reason": "text_shape_not_blocking",
                "stage_id": "text_shape",
                "candidate_count": 0,
            },
        )

    flags = shape_issue_flags(report)
    stroke_body_grid = shape_stroke_body_grid(flags)
    size_deltas = (0, -1, 1)
    if flags.get("body_gap"):
        size_deltas = (0, 1, -1)
    if flags.get("pose_gap"):
        size_deltas = tuple(dict.fromkeys(size_deltas + (0,)))

    max_font_size = max_font_size_for_plan(plan)
    font_items = shape_font_items(params, font_style_reference, limit=4)
    offset_candidates = normalized_offset_candidates(plan, params)
    text_dx_candidates = tuple(dict.fromkeys((params.text_dx, 0, -1, 1)))[:3]
    text_dy_candidates = tuple(dict.fromkeys((params.text_dy, 0, -1, 1)))[:3]
    raw_budget = (
        len(font_items)
        * len(size_deltas)
        * len(offset_candidates)
        * len(text_dx_candidates)
        * len(text_dy_candidates)
        * len(stroke_body_grid)
    )

    variants: list[CandidateParams] = []
    for font_item in font_items:
        font_name = str(font_item.get("font_name") or params.font_name)
        font_path = str(font_item.get("font_path") or params.font_path)
        try:
            base_size = int(font_item.get("font_size") or params.font_size)
        except (TypeError, ValueError):
            base_size = params.font_size
        for size_delta in size_deltas:
            font_size = max(8, min(max_font_size, base_size + int(size_delta)))
            for offsets in offset_candidates:
                for text_dx in text_dx_candidates:
                    for text_dy in text_dy_candidates:
                        for (
                            stroke_opacity,
                            ink_gain,
                            alpha_contrast,
                            core_ink_gain,
                            core_darken_strength,
                        ) in stroke_body_grid:
                            variants.append(
                                mutate_params(
                                    params,
                                    font_name=font_name,
                                    font_path=font_path,
                                    font_size=font_size,
                                    stroke_opacity=stroke_opacity,
                                    ink_gain=ink_gain,
                                    alpha_contrast=alpha_contrast,
                                    core_ink_gain=core_ink_gain,
                                    core_darken_strength=core_darken_strength,
                                    core_darken_threshold=130,
                                    core_darken_target_gray=28,
                                    text_dx=text_dx,
                                    text_dy=text_dy,
                                    char_offsets=offsets,
                                    opacity=params.opacity,
                                    blur=params.blur,
                                    mask_threshold=params.mask_threshold,
                                    mask_dilate_iterations=params.mask_dilate_iterations,
                                    inpaint_radius=params.inpaint_radius,
                                    photo_warp=params.photo_warp,
                                    edge_breakup=params.edge_breakup,
                                    photo_noise=params.photo_noise,
                                    jpeg_quality=params.jpeg_quality,
                                )
                            )
    candidates = dedupe_params(variants, min(limit, TEXT_SHAPE_GRID_TOP_LIMIT))
    candidate_records: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    for candidate in candidates:
        delta_keys = params_delta_keys(params, candidate)
        blocked_delta_keys = sorted(delta_keys & TEXT_SHAPE_GRID_BLOCKED_DELTA_KEYS)
        undeclared_delta_keys = sorted(delta_keys - TEXT_SHAPE_GRID_ALLOWED_DELTA_KEYS)
        record = {
            "candidate_id": candidate.candidate_id,
            "delta_keys": sorted(delta_keys),
            "reason_categories": _candidate_reason_categories("text_shape", delta_keys),
            "allowed_delta_keys_only": not blocked_delta_keys and not undeclared_delta_keys,
            "blocked_delta_keys": blocked_delta_keys,
            "undeclared_delta_keys": undeclared_delta_keys,
        }
        candidate_records.append(record)
        if blocked_delta_keys or undeclared_delta_keys:
            violations.append(record)

    budget_min, budget_max = TEXT_SHAPE_GRID_BUDGET_RANGE
    report_payload = {
        "enabled": True,
        "stage_id": "text_shape",
        "optimization_step": "shape_reset",
        "budget": {
            "raw_candidate_budget": raw_budget,
            "budget_min": budget_min,
            "budget_max": budget_max,
            "within_budget": budget_min <= raw_budget <= budget_max,
            "retained_top_limit": min(limit, TEXT_SHAPE_GRID_TOP_LIMIT),
            "retained_count": len(candidates),
            "pruned_count": max(0, raw_budget - len(candidates)),
        },
        "prune_reason_contract": prune_reason_contract("text_shape", raw_budget, len(candidates)),
        "axes": {
            "font_count": len(font_items),
            "font_size_delta_count": len(size_deltas),
            "placement_strategy": plan.placement_strategy,
            "placement_strategy_reason": plan.placement_strategy_reason,
            "text_dx_count": len(text_dx_candidates),
            "text_dy_count": len(text_dy_candidates),
            "char_offsets_count": len(offset_candidates),
            "protected_box_count": len(plan.protected_boxes),
            "stroke_body_grid_count": len(stroke_body_grid),
            "pose_shear_source": "renderer_reference_slot_shear_from_source_slots_and_neighbors",
        },
        "issue_flags": flags,
        "allowed_delta_keys": sorted(TEXT_SHAPE_GRID_ALLOWED_DELTA_KEYS),
        "blocked_delta_keys": sorted(TEXT_SHAPE_GRID_BLOCKED_DELTA_KEYS),
        "candidate_count": len(candidates),
        "candidate_delta_audit": candidate_records,
        "violations": violations,
    }
    return ShapeCandidateGrid(candidates=candidates, report=report_payload)


def text_shape_reset_candidates(
    params: CandidateParams,
    font_style_reference: dict[str, Any],
    plan: RenderPlan,
    report: dict[str, Any] | None,
    *,
    limit: int = TEXT_SHAPE_GRID_TOP_LIMIT,
) -> list[CandidateParams]:
    return text_shape_reset_candidate_grid(
        params,
        font_style_reference,
        plan,
        report,
        limit=limit,
    ).candidates


def report_blocks_ink_gray(report: dict[str, Any] | None) -> bool:
    if not isinstance(report, dict):
        return False
    stage_gate = report.get("stage_gate")
    if not isinstance(stage_gate, dict):
        stage_gate = stage_gate_for_report(report)
    return stage_gate.get("blocking_stage") == "ink_gray_balance"


def _float_axis(
    base: float,
    deltas: tuple[float, ...],
    *,
    low: float,
    high: float,
    target_len: int,
) -> tuple[float, ...]:
    values: list[float] = []

    def add(value: float) -> None:
        bounded = round(max(low, min(high, float(value))), 3)
        if bounded not in values:
            values.append(bounded)

    for delta in deltas:
        add(base + delta)
    step = 0.01
    probe = 1
    while len(values) < target_len and probe <= 80:
        add(base + probe * step)
        if len(values) >= target_len:
            break
        add(base - probe * step)
        probe += 1
    return tuple(values[:target_len])


def _int_axis(
    base: int,
    deltas: tuple[int, ...],
    *,
    low: int,
    high: int,
    target_len: int,
) -> tuple[int, ...]:
    values: list[int] = []

    def add(value: int) -> None:
        bounded = max(low, min(high, int(value)))
        if bounded not in values:
            values.append(bounded)

    for delta in deltas:
        add(base + delta)
    probe = 1
    while len(values) < target_len and probe <= 80:
        add(base + probe)
        if len(values) >= target_len:
            break
        add(base - probe)
        probe += 1
    return tuple(values[:target_len])


def ink_gray_issue_flags(report: dict[str, Any] | None) -> dict[str, bool]:
    issue_types = {
        str(issue.get("type") or "")
        for issue in stage_issues(report, "ink_gray_balance")
        if isinstance(issue, dict)
    }
    return {
        "excess_black_core": report_has_excess_black_core(report),
        "outer_gray_halo": report_has_outer_gray_halo(report)
        or "changed_char_neighbor_outer_gray_halo_too_high" in issue_types,
        "needs_wider_gray_strokes": report_needs_wider_gray_strokes(report),
        "needs_thinner_strokes": report_needs_thinner_strokes(report),
        "fine_strokes_too_soft": report_has_fine_strokes_too_soft(report),
        "core_too_light": bool(
            issue_types
            & {
                "core_mean_gray_too_light",
                "core_lighten_too_high",
                "changed_char_core_too_light",
                "ink_too_light",
            }
        ),
    }


def _build_micro_tuning_report(
    micro_tuning: dict[str, Any],
    micro_variants: list[CandidateParams],
    micro_candidate_ids: list[str],
) -> dict[str, Any]:
    enabled = bool(micro_tuning.get("enabled"))
    family = micro_tuning.get("candidate_family", "")
    report: dict[str, Any] = {
        "enabled": enabled,
        "family": family,
        "stage_id": "ink_gray_balance",
        "candidate_count": len(micro_variants) if enabled else 0,
        "candidate_ids": micro_candidate_ids if enabled else [],
    }
    if enabled:
        for field in ("metric", "actual", "limit_value", "gap", "gap_ratio"):
            if field in micro_tuning:
                key = "limit" if field == "limit_value" else field
                report[key] = micro_tuning[field]
    else:
        report["disabled_reason"] = micro_tuning.get("reason", "unknown")
    return report
    issue_types = {
        str(issue.get("type") or "")
        for issue in stage_issues(report, "ink_gray_balance")
        if isinstance(issue, dict)
    }
    return {
        "excess_black_core": report_has_excess_black_core(report),
        "outer_gray_halo": report_has_outer_gray_halo(report)
        or "changed_char_neighbor_outer_gray_halo_too_high" in issue_types,
        "needs_wider_gray_strokes": report_needs_wider_gray_strokes(report),
        "needs_thinner_strokes": report_needs_thinner_strokes(report),
        "fine_strokes_too_soft": report_has_fine_strokes_too_soft(report),
        "core_too_light": bool(
            issue_types
            & {
                "core_mean_gray_too_light",
                "core_lighten_too_high",
                "changed_char_core_too_light",
                "ink_too_light",
            }
        ),
    }


def _numeric_issue_gap(issue: dict[str, Any]) -> float | None:
    try:
        actual = float(issue.get("actual"))
        limit = float(issue.get("limit"))
    except (TypeError, ValueError):
        return None
    return actual - limit


def _core_light_near_threshold(report: dict[str, Any] | None) -> dict[str, Any]:
    allowed_issue_types = {"core_mean_gray_too_light", "core_lighten_too_high"}
    issues = stage_issues(report, "ink_gray_balance")
    issue_types = {str(issue.get("type") or "") for issue in issues}
    if not issue_types:
        return {"enabled": False, "reason": "no_ink_gray_issues"}
    if issue_types - allowed_issue_types:
        return {
            "enabled": False,
            "reason": "other_ink_gray_issues_present",
            "issue_types": sorted(issue_types),
        }
    gaps: list[float] = []
    for issue in issues:
        gap = _numeric_issue_gap(issue)
        if gap is None or gap <= 0:
            return {
                "enabled": False,
                "reason": "missing_or_nonpositive_issue_gap",
                "issue_types": sorted(issue_types),
            }
        gaps.append(gap)
    max_gap = max(gaps)
    near_threshold_limit = 0.75
    if max_gap > near_threshold_limit:
        return {
            "enabled": False,
            "reason": "issue_gap_not_near_threshold",
            "max_gap": round(max_gap, 3),
            "limit": near_threshold_limit,
            "issue_types": sorted(issue_types),
        }
    return {
        "enabled": True,
        "reason": "near_threshold_core_light_micro_tuning",
        "max_gap": round(max_gap, 3),
        "limit": near_threshold_limit,
        "issue_types": sorted(issue_types),
        "candidate_family": "core_only_micro_recovery",
    }


def _core_overblack_near_threshold(report: dict[str, Any] | None) -> dict[str, Any]:
    allowed_issue_types = {"roi_core_too_black", "changed_char_core_too_black"}
    issues = stage_issues(report, "ink_gray_balance")
    if not issues:
        return {"enabled": False, "reason": "no_ink_gray_issues"}
    issue_types = {str(issue.get("type") or "") for issue in issues}
    overblack_issues = [issue for issue in issues if str(issue.get("type") or "") in allowed_issue_types]
    if not overblack_issues:
        return {"enabled": False, "reason": "no_overblack_issues_present", "issue_types": sorted(issue_types)}
    if issue_types - allowed_issue_types:
        return {
            "enabled": False,
            "reason": "other_ink_gray_issues_present",
            "issue_types": sorted(issue_types),
        }
    gaps: list[float] = []
    for issue in overblack_issues:
        actual = None
        limit = None
        try:
            actual = float(issue.get("actual"))
            limit = float(issue.get("limit"))
        except (TypeError, ValueError):
            pass
        if actual is None or limit is None:
            continue
        excess = actual - limit
        if excess <= 0:
            continue
        gap_ratio = limit / max(actual, 1.0)
        gaps.append(1.0 - gap_ratio)
    if not gaps:
        return {
            "enabled": False,
            "reason": "no_overblack_gap_computed",
            "issue_types": sorted(issue_types),
        }
    mean_gap_ratio = sum(gaps) / len(gaps)
    near_threshold_limit = 0.12
    if mean_gap_ratio > near_threshold_limit:
        return {
            "enabled": False,
            "reason": "overblack_gap_not_near_threshold",
            "mean_gap_ratio": round(mean_gap_ratio, 4),
            "limit": near_threshold_limit,
            "issue_types": sorted(issue_types),
        }
    metric_actual = None
    metric_limit = None
    for issue in overblack_issues:
        try:
            metric_actual = float(issue.get("actual"))
            metric_limit = float(issue.get("limit"))
        except (TypeError, ValueError):
            pass
        if metric_actual is not None and metric_limit is not None:
            break
    return {
        "enabled": True,
        "reason": "near_threshold_overblack_micro_tuning",
        "family": "overblack_micro_reduction",
        "mean_gap_ratio": round(mean_gap_ratio, 4),
        "limit": near_threshold_limit,
        "issue_types": sorted(issue_types),
        "candidate_family": "core_only_micro_reduction",
        "metric": "ink_gray_balance_core_black",
        "actual": round(float(metric_actual or 0), 3),
        "limit_value": round(float(metric_limit or 0), 3),
        "gap": round(float((metric_actual or 0) - (metric_limit or 0)), 3),
        "gap_ratio": round(mean_gap_ratio, 4),
    }


def ink_gray_near_threshold_micro_tuning(report: dict[str, Any] | None) -> dict[str, Any]:
    core_light = _core_light_near_threshold(report)
    if core_light.get("enabled"):
        return core_light
    issue_types_light = set(core_light.get("issue_types") or [])
    allowed_light = {"core_mean_gray_too_light", "core_lighten_too_high"}
    if issue_types_light and issue_types_light.issubset(allowed_light):
        return core_light
    overblack = _core_overblack_near_threshold(report)
    if overblack.get("enabled"):
        return overblack
    issue_types_overblack = set(overblack.get("issue_types") or [])
    allowed_overblack = {"roi_core_too_black", "changed_char_core_too_black"}
    if issue_types_overblack and issue_types_overblack.issubset(allowed_overblack):
        return overblack
    issues = stage_issues(report, "ink_gray_balance")
    issue_types = sorted({str(issue.get("type") or "") for issue in issues if isinstance(issue, dict)})
    if not issues:
        return {"enabled": False, "reason": "no_ink_gray_issues"}
    overblack_issue_types = {"roi_core_too_black", "changed_char_core_too_black"}
    if issue_types and set(issue_types) & overblack_issue_types:
        return {
            "enabled": False,
            "reason": "other_ink_gray_issues_present",
            "issue_types": issue_types,
        }
    return {"enabled": False, "reason": "no_near_threshold_condition", "issue_types": issue_types}


def _core_light_micro_variants(params: CandidateParams) -> list[CandidateParams]:
    core_gain = params.core_ink_gain
    darken = params.core_darken_strength
    target = params.core_darken_target_gray
    threshold = params.core_darken_threshold
    return [
        mutate_params(params, core_darken_strength=darken + 0.003),
        mutate_params(params, core_darken_strength=darken + 0.006),
        mutate_params(params, core_ink_gain=core_gain + 0.003),
        mutate_params(params, core_ink_gain=core_gain + 0.006),
        mutate_params(params, core_ink_gain=core_gain + 0.004, core_darken_strength=darken + 0.004),
        mutate_params(params, core_darken_threshold=threshold + 1),
        mutate_params(params, core_darken_target_gray=target - 1),
        mutate_params(params, core_darken_strength=darken + 0.004, core_darken_target_gray=target - 1),
        mutate_params(params, alpha_contrast=params.alpha_contrast + 0.003),
        mutate_params(params, opacity=params.opacity + 0.003),
    ]


def _core_overblack_micro_variants(params: CandidateParams) -> list[CandidateParams]:
    return [
        mutate_params(params, opacity=params.opacity - 0.005),
        mutate_params(params, opacity=params.opacity - 0.010),
        mutate_params(params, core_ink_gain=params.core_ink_gain - 0.005),
        mutate_params(params, core_ink_gain=params.core_ink_gain - 0.010),
        mutate_params(params, core_darken_strength=params.core_darken_strength - 0.005),
        mutate_params(params, core_darken_strength=params.core_darken_strength - 0.010),
        mutate_params(params, core_ink_gain=params.core_ink_gain - 0.006, core_darken_strength=params.core_darken_strength - 0.006),
        mutate_params(params, alpha_contrast=params.alpha_contrast - 0.005),
        mutate_params(params, opacity=params.opacity - 0.007, core_ink_gain=params.core_ink_gain - 0.005),
        mutate_params(params, core_darken_strength=params.core_darken_strength - 0.007, alpha_contrast=params.alpha_contrast - 0.005),
    ]


def ink_gray_micro_tuning_candidates(
    params: CandidateParams,
    report: dict[str, Any] | None,
) -> list[CandidateParams]:
    micro = ink_gray_near_threshold_micro_tuning(report)
    if not micro.get("enabled"):
        return []
    family = micro.get("candidate_family", "")
    if family == "core_only_micro_reduction":
        return _core_overblack_micro_variants(params)
    return _core_light_micro_variants(params)


def ink_gray_axes(params: CandidateParams, report: dict[str, Any] | None) -> dict[str, tuple[Any, ...]]:
    flags = ink_gray_issue_flags(report)
    combined_core_light_outer_halo = flags["core_too_light"] and flags["outer_gray_halo"]
    if flags["excess_black_core"] or flags["needs_thinner_strokes"]:
        opacity_deltas = (0.0, -0.02, -0.04, -0.06)
        stroke_deltas = (0.0, -0.02, -0.04, 0.01)
        ink_deltas = (0.0, -0.02)
        alpha_deltas = (0.0, -0.04)
        core_deltas = ((0.0, 0.0, 0, 0), (-0.04, -0.04, -6, 4), (-0.08, -0.06, -10, 8), (-0.02, -0.08, -4, 10))
    elif combined_core_light_outer_halo:
        opacity_deltas = (0.0, 0.01, 0.02, 0.03)
        stroke_deltas = (0.0, -0.01, -0.02, -0.03)
        ink_deltas = (0.0, -0.01)
        alpha_deltas = (0.04, 0.08)
        core_deltas = ((0.02, 0.03, 4, -2), (0.04, 0.04, 6, -4), (0.06, 0.06, 10, -6), (0.03, 0.08, 5, -8))
    elif flags["outer_gray_halo"]:
        opacity_deltas = (0.0, 0.01, -0.02, 0.02)
        stroke_deltas = (0.0, -0.01, -0.03, 0.01)
        ink_deltas = (0.0, 0.01)
        alpha_deltas = (0.0, -0.04)
        core_deltas = ((0.0, 0.0, 0, 0), (0.02, 0.02, 3, -2), (-0.02, -0.02, -4, 4), (0.04, 0.01, 6, -4))
    else:
        opacity_deltas = (0.0, 0.02, 0.04, -0.02)
        stroke_deltas = (0.0, 0.02, 0.04, -0.01)
        ink_deltas = (0.0, 0.02)
        alpha_deltas = (0.0, 0.04)
        core_deltas = ((0.0, 0.0, 0, 0), (0.04, 0.04, 6, -4), (0.08, 0.06, 10, -8), (0.02, 0.08, 4, -10))

    core_axis = tuple(
        (
            _float_axis(params.core_ink_gain, (core_delta,), low=0.0, high=1.0, target_len=1)[0],
            _float_axis(params.core_darken_strength, (darken_delta,), low=0.0, high=1.0, target_len=1)[0],
            _int_axis(params.core_darken_threshold, (threshold_delta,), low=0, high=254, target_len=1)[0],
            _int_axis(params.core_darken_target_gray, (target_delta,), low=0, high=120, target_len=1)[0],
        )
        for core_delta, darken_delta, threshold_delta, target_delta in core_deltas
    )
    return {
        "opacity": _float_axis(params.opacity, opacity_deltas, low=0.2, high=1.0, target_len=4),
        "stroke_opacity": _float_axis(params.stroke_opacity, stroke_deltas, low=0.0, high=1.0, target_len=4),
        "ink_gain": _float_axis(params.ink_gain, ink_deltas, low=0.0, high=1.0, target_len=2),
        "alpha_contrast": _float_axis(params.alpha_contrast, alpha_deltas, low=0.0, high=2.0, target_len=2),
        "core_tone": core_axis,
    }


def ink_gray_candidate_grid(
    params: CandidateParams,
    report: dict[str, Any] | None,
    *,
    limit: int = INK_GRAY_GRID_TOP_LIMIT,
    parent_shape_candidate_id: str | None = None,
    allow_text_shape_guard: bool = False,
) -> InkGrayCandidateGrid:
    guard_for_text_shape = (
        bool(allow_text_shape_guard)
        and report_blocks_text_shape(report)
        and report_has_excess_black_core(report)
    )
    if not report_blocks_ink_gray(report) and not guard_for_text_shape:
        reason = "ink_gray_balance_not_blocking"
        if allow_text_shape_guard and report_blocks_text_shape(report):
            reason = "text_shape_without_excess_black_core"
        return InkGrayCandidateGrid(
            candidates=[],
            report={
                "enabled": False,
                "reason": reason,
                "stage_id": "ink_gray_balance",
                "guards_stage": "text_shape" if allow_text_shape_guard else None,
                "guard_mode": "text_shape_excess_black_core" if allow_text_shape_guard else None,
                "candidate_count": 0,
            },
        )

    micro_tuning = ink_gray_near_threshold_micro_tuning(report)
    micro_variants = ink_gray_micro_tuning_candidates(params, report)
    axes = ink_gray_axes(params, report)
    opacity_values = axes["opacity"]
    stroke_values = axes["stroke_opacity"]
    ink_values = axes["ink_gain"]
    alpha_values = axes["alpha_contrast"]
    core_values = axes["core_tone"]
    axis_raw_budget = (
        len(opacity_values)
        * len(stroke_values)
        * len(ink_values)
        * len(alpha_values)
        * len(core_values)
    )
    raw_budget = axis_raw_budget + len(micro_variants)

    axis_variants: list[CandidateParams] = []
    for opacity in opacity_values:
        for stroke_opacity in stroke_values:
            for ink_gain in ink_values:
                for alpha_contrast in alpha_values:
                    for (
                        core_ink_gain,
                        core_darken_strength,
                        core_darken_threshold,
                        core_darken_target_gray,
                    ) in core_values:
                        axis_variants.append(
                            mutate_params(
                                params,
                                opacity=opacity,
                                stroke_opacity=stroke_opacity,
                                ink_gain=ink_gain,
                                alpha_contrast=alpha_contrast,
                                core_ink_gain=core_ink_gain,
                                core_darken_strength=core_darken_strength,
                                core_darken_threshold=core_darken_threshold,
                                core_darken_target_gray=core_darken_target_gray,
                            )
                        )
    retained_limit = min(limit, INK_GRAY_GRID_TOP_LIMIT)
    max_axis_retain = max(0, retained_limit - len(micro_variants))
    retained_axis = dedupe_params(axis_variants, max(1, max_axis_retain))
    retained_micro = dedupe_params(micro_variants, len(micro_variants))
    candidates = retained_micro + retained_axis[:max_axis_retain]
    micro_candidate_ids = [c.candidate_id for c in retained_micro]
    candidate_records: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    shape_parent_id = str(parent_shape_candidate_id or params.candidate_id)
    for candidate in candidates:
        delta_keys = params_delta_keys(params, candidate)
        blocked_delta_keys = sorted(delta_keys & INK_GRAY_GRID_BLOCKED_DELTA_KEYS)
        undeclared_delta_keys = sorted(delta_keys - INK_GRAY_GRID_ALLOWED_DELTA_KEYS)
        record = {
            "candidate_id": candidate.candidate_id,
            "parent_candidate_id": params.candidate_id,
            "parent_shape_candidate_id": shape_parent_id,
            "delta_keys": sorted(delta_keys),
            "reason_categories": _candidate_reason_categories("ink_gray_balance", delta_keys),
            "allowed_delta_keys_only": not blocked_delta_keys and not undeclared_delta_keys,
            "blocked_delta_keys": blocked_delta_keys,
            "undeclared_delta_keys": undeclared_delta_keys,
        }
        candidate_records.append(record)
        if blocked_delta_keys or undeclared_delta_keys:
            violations.append(record)

    stage_gate = stage_gate_for_report(report or {})
    text_shape_status = (
        stage_gate.get("stage_status", {}).get("text_shape")
        if isinstance(stage_gate.get("stage_status"), dict)
        else {}
    )
    parent_shape_contract = {
        "required_prior_stage": "text_shape",
        "required_parent_state": (
            "text_shape_not_yet_passed_but_ink_must_not_regress"
            if guard_for_text_shape
            else "text_shape_passed_before_ink_gray"
        ),
        "current_blocking_stage": stage_gate.get("blocking_stage"),
        "parent_candidate_id": params.candidate_id,
        "parent_shape_candidate_id": shape_parent_id,
        "parent_shape_source": (
            "current_text_shape_candidate_with_excess_black_core"
            if guard_for_text_shape
            else "current_candidate_after_text_shape_pass"
        ),
        "parent_shape_stage_passed": bool(
            isinstance(text_shape_status, dict) and text_shape_status.get("pass")
        ),
        "guard_allows_unpassed_text_shape": bool(guard_for_text_shape),
        "candidate_parent_trace_complete": all(
            record.get("parent_shape_candidate_id") == shape_parent_id
            for record in candidate_records
        ),
        "candidate_parent_shape_ids": sorted(
            {
                str(record.get("parent_shape_candidate_id"))
                for record in candidate_records
                if record.get("parent_shape_candidate_id")
            }
        ),
    }
    issue_flags = ink_gray_issue_flags(report)
    combined_core_light_outer_halo = issue_flags["core_too_light"] and issue_flags["outer_gray_halo"]
    budget_min, budget_max = INK_GRAY_GRID_BUDGET_RANGE
    overblack_micro_report: dict[str, Any] = _build_micro_tuning_report(
        micro_tuning, micro_variants, micro_candidate_ids
    )
    return InkGrayCandidateGrid(
        candidates=candidates,
        report={
            "enabled": True,
            "stage_id": "ink_gray_balance",
            "optimization_step": "ink_guard" if guard_for_text_shape else "ink_gray_balance",
            "guards_stage": "text_shape" if guard_for_text_shape else None,
            "guard_mode": "text_shape_excess_black_core" if guard_for_text_shape else None,
            "guard_contract": (
                {
                    "purpose": "protect ink-gray balance while text_shape remains the primary blocking stage",
                    "primary_blocking_stage": "text_shape",
                    "protected_stage": "ink_gray_balance",
                    "allowed_delta_keys": sorted(INK_GRAY_GRID_ALLOWED_DELTA_KEYS),
                    "blocked_delta_keys": sorted(INK_GRAY_GRID_BLOCKED_DELTA_KEYS),
                    "selection_rule": "candidate must not regress text_shape and must reduce ink_gray_balance severity",
                }
                if guard_for_text_shape
                else None
            ),
            "parent_candidate_id": params.candidate_id,
            "parent_shape_candidate_id": shape_parent_id,
            "parent_shape_contract": parent_shape_contract,
            "budget": {
                "raw_candidate_budget": raw_budget,
                "budget_min": budget_min,
                "budget_max": budget_max,
                "within_budget": budget_min <= raw_budget <= budget_max,
                "retained_top_limit": retained_limit,
                "retained_count": len(candidates),
                "pruned_count": max(0, raw_budget - len(candidates)),
            },
            "prune_reason_contract": prune_reason_contract("ink_gray_balance", raw_budget, len(candidates)),
            "axes": {
                "opacity_count": len(opacity_values),
                "stroke_opacity_count": len(stroke_values),
                "ink_gain_count": len(ink_values),
                "alpha_contrast_count": len(alpha_values),
                "core_tone_count": len(core_values),
                "near_threshold_micro_tuning": micro_tuning,
                "near_threshold_micro_candidate_count": len(micro_variants),
                "near_threshold_micro_candidate_ids": micro_candidate_ids,
                "micro_candidates_retained_separately": True,
                "micro_retention_rule": "micro tuning candidates are retained independently and prepended ahead of axis-priority top-N; they are not subject to axis-priority pruning",
                "ranking_method": "local_issue_ordered_axis_priority",
                "combined_core_light_outer_gray_halo_strategy": (
                    "recover_core_density_and_trim_outer_gray"
                    if combined_core_light_outer_halo else None
                ),
                "forbidden_combined_strategy_directions": (
                    ["increase_blur", "increase_photo_noise", "expand_outer_gray_halo"]
                    if combined_core_light_outer_halo else []
                ),
            },
            "issue_flags": issue_flags,
            "allowed_delta_keys": sorted(INK_GRAY_GRID_ALLOWED_DELTA_KEYS),
            "blocked_delta_keys": sorted(INK_GRAY_GRID_BLOCKED_DELTA_KEYS),
            "overblack_micro_tuning_report": overblack_micro_report,
            "preserved_shape_keys": [
                "font_name",
                "font_path",
                "font_size",
                "text_dx",
                "text_dy",
                "char_offsets",
            ],
            "shape_key_changes_require_stage": "text_shape",
            "candidate_count": len(candidates),
            "candidate_delta_audit": candidate_records,
            "violations": violations,
        },
    )


def report_blocks_photo_texture(report: dict[str, Any] | None) -> bool:
    if not isinstance(report, dict):
        return False
    stage_gate = report.get("stage_gate")
    if not isinstance(stage_gate, dict):
        stage_gate = stage_gate_for_report(report)
    return stage_gate.get("blocking_stage") == "photo_texture"


def photo_texture_issue_flags(report: dict[str, Any] | None) -> dict[str, bool]:
    issue_types = {
        str(issue.get("type") or "")
        for issue in stage_issues(report, "photo_texture")
        if isinstance(issue, dict)
    }
    return {
        "too_sharp": "photo_texture_too_sharp" in issue_types,
        "too_clean": "photo_texture_too_clean" in issue_types,
        "too_blurry": "photo_texture_too_blurry" in issue_types,
        "edge_breakup_missing": "photo_texture_edge_breakup_missing" in issue_types,
        "not_applied": "photo_texture_not_applied" in issue_types,
    }


def _jpeg_axis(base: int, deltas: tuple[int, ...], *, target_len: int) -> tuple[int, ...]:
    start = int(base or 94)
    return _int_axis(start, deltas, low=35, high=99, target_len=target_len)


def photo_texture_axes(params: CandidateParams, report: dict[str, Any] | None) -> dict[str, tuple[Any, ...]]:
    flags = photo_texture_issue_flags(report)
    if flags["too_blurry"]:
        blur_deltas = (0.0, -0.06, -0.10)
        alpha_deltas = (0.0, 0.02)
        warp_deltas = (0.0, -0.02)
        breakup_deltas = (0.0, -0.004, 0.004)
        noise_deltas = (0.0, -0.008, -0.014)
        jpeg_deltas = (0, 6)
    else:
        blur_deltas = (0.0, 0.04, 0.08)
        alpha_deltas = (0.0, -0.02)
        warp_deltas = (0.0, 0.02)
        breakup_deltas = (0.0, 0.006, 0.012)
        noise_deltas = (0.0, 0.014, 0.024)
        jpeg_deltas = (0, -6)
        if flags["edge_breakup_missing"] or flags["not_applied"]:
            breakup_deltas = (0.006, 0.012, 0.018)
        if flags["too_clean"] or flags["not_applied"]:
            noise_deltas = (0.012, 0.020, 0.030)

    return {
        "blur": _float_axis(params.blur, blur_deltas, low=0.0, high=2.0, target_len=2),
        "alpha_contrast": _float_axis(params.alpha_contrast, alpha_deltas, low=0.0, high=2.0, target_len=2),
        "photo_warp": _float_axis(params.photo_warp, warp_deltas, low=0.0, high=1.0, target_len=2),
        "edge_breakup": _float_axis(params.edge_breakup, breakup_deltas, low=0.0, high=0.2, target_len=3),
        "photo_noise": _float_axis(params.photo_noise, noise_deltas, low=0.0, high=0.35, target_len=3),
        "jpeg_quality": _jpeg_axis(params.jpeg_quality, jpeg_deltas, target_len=2),
    }


def photo_texture_candidate_grid(
    params: CandidateParams,
    report: dict[str, Any] | None,
    *,
    limit: int = PHOTO_TEXTURE_GRID_TOP_LIMIT,
) -> PhotoTextureCandidateGrid:
    if not report_blocks_photo_texture(report):
        return PhotoTextureCandidateGrid(
            candidates=[],
            report={
                "enabled": False,
                "reason": "photo_texture_not_blocking",
                "stage_id": "photo_texture",
                "candidate_count": 0,
            },
        )

    axes = photo_texture_axes(params, report)
    blur_values = axes["blur"]
    alpha_values = axes["alpha_contrast"]
    warp_values = axes["photo_warp"]
    breakup_values = axes["edge_breakup"]
    noise_values = axes["photo_noise"]
    jpeg_values = axes["jpeg_quality"]
    raw_budget = (
        len(blur_values)
        * len(alpha_values)
        * len(warp_values)
        * len(breakup_values)
        * len(noise_values)
        * len(jpeg_values)
    )

    variants: list[CandidateParams] = []
    for blur in blur_values:
        for alpha_contrast in alpha_values:
            for photo_warp in warp_values:
                for edge_breakup in breakup_values:
                    for photo_noise in noise_values:
                        for jpeg_quality in jpeg_values:
                            variants.append(
                                mutate_params(
                                    params,
                                    blur=blur,
                                    alpha_contrast=alpha_contrast,
                                    photo_warp=photo_warp,
                                    edge_breakup=edge_breakup,
                                    photo_noise=photo_noise,
                                    jpeg_quality=jpeg_quality,
                                )
                            )
    retained_limit = min(limit, PHOTO_TEXTURE_GRID_TOP_LIMIT)
    candidates = dedupe_params(variants, retained_limit)
    candidate_records: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    for candidate in candidates:
        delta_keys = params_delta_keys(params, candidate)
        blocked_delta_keys = sorted(delta_keys & PHOTO_TEXTURE_GRID_BLOCKED_DELTA_KEYS)
        undeclared_delta_keys = sorted(delta_keys - PHOTO_TEXTURE_GRID_ALLOWED_DELTA_KEYS)
        record = {
            "candidate_id": candidate.candidate_id,
            "parent_candidate_id": params.candidate_id,
            "delta_keys": sorted(delta_keys),
            "reason_categories": _candidate_reason_categories("photo_texture", delta_keys),
            "allowed_delta_keys_only": not blocked_delta_keys and not undeclared_delta_keys,
            "blocked_delta_keys": blocked_delta_keys,
            "undeclared_delta_keys": undeclared_delta_keys,
        }
        candidate_records.append(record)
        if blocked_delta_keys or undeclared_delta_keys:
            violations.append(record)

    budget_min, budget_max = PHOTO_TEXTURE_GRID_BUDGET_RANGE
    return PhotoTextureCandidateGrid(
        candidates=candidates,
        report={
            "enabled": True,
            "stage_id": "photo_texture",
            "optimization_step": "photo_texture",
            "parent_candidate_id": params.candidate_id,
            "budget": {
                "raw_candidate_budget": raw_budget,
                "budget_min": budget_min,
                "budget_max": budget_max,
                "within_budget": budget_min <= raw_budget <= budget_max,
                "retained_top_limit": retained_limit,
                "retained_count": len(candidates),
                "pruned_count": max(0, raw_budget - len(candidates)),
            },
            "prune_reason_contract": prune_reason_contract("photo_texture", raw_budget, len(candidates)),
            "axes": {
                "blur_count": len(blur_values),
                "alpha_contrast_count": len(alpha_values),
                "photo_warp_count": len(warp_values),
                "edge_breakup_count": len(breakup_values),
                "photo_noise_count": len(noise_values),
                "jpeg_quality_count": len(jpeg_values),
                "ranking_method": "local_photo_texture_issue_axis_priority",
                "alpha_adjustment_scope": "small_alpha_degradation_or_recovery_only",
                "residual_retexture_keys": ["edge_breakup", "photo_noise", "jpeg_quality"],
            },
            "issue_flags": photo_texture_issue_flags(report),
            "allowed_delta_keys": sorted(PHOTO_TEXTURE_GRID_ALLOWED_DELTA_KEYS),
            "blocked_delta_keys": sorted(PHOTO_TEXTURE_GRID_BLOCKED_DELTA_KEYS),
            "candidate_count": len(candidates),
            "candidate_delta_audit": candidate_records,
            "violations": violations,
        },
    )




def final_font_revision_candidates(
    params: CandidateParams,
    font_style_reference: dict[str, Any],
    plan: RenderPlan,
    report: dict[str, Any] | None,
) -> list[CandidateParams]:
    return text_shape_reset_candidates(
        params,
        font_style_reference,
        plan,
        report,
        limit=24,
    )
