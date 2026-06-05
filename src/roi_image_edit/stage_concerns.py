from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from roi_image_edit.stage_policy import STAGE_ORDER


@dataclass(frozen=True)
class DiagnosticConcernMapping:
    concern_id: str
    current_stages: tuple[str, ...]
    optimization_steps: tuple[str, ...]
    report_fields: tuple[str, ...]
    notes: str

    def as_report(self) -> dict[str, Any]:
        return asdict(self)


DIAGNOSTIC_CONCERN_MAPPINGS: tuple[DiagnosticConcernMapping, ...] = (
    DiagnosticConcernMapping(
        concern_id="slot_alignment",
        current_stages=("hard_boundary", "text_shape"),
        optimization_steps=("field_roi_selection", "slot_quality_gate", "slot_alignment_search"),
        report_fields=("slot_quality_report", "placement_strategy_report", "search_roi", "target_roi"),
        notes="ROI and slot safety gate before rendering; slot alignment search remains a text_shape optimization step.",
    ),
    DiagnosticConcernMapping(
        concern_id="font_structure",
        current_stages=("text_shape",),
        optimization_steps=("font_style_search", "font_size_search"),
        report_fields=("font_style_gate", "shape_change_report", "font_style_score"),
        notes="Font family and size are shape work; they must not be repaired by ink or texture stages.",
    ),
    DiagnosticConcernMapping(
        concern_id="pose_geometry",
        current_stages=("text_shape",),
        optimization_steps=("pose_shear_search",),
        report_fields=("local_pose_issues", "text_angle_degrees", "shape_change_report"),
        notes="Pose is inferred from slots, neighbors, and projection metrics; no per-image direction rule is valid.",
    ),
    DiagnosticConcernMapping(
        concern_id="stroke_body",
        current_stages=("text_shape",),
        optimization_steps=("stroke_body_search",),
        report_fields=("local_stroke_body_issues", "shape_change_report"),
        notes="Stroke body is shape work and must be solved before edge or texture cleanup dominates.",
    ),
    DiagnosticConcernMapping(
        concern_id="tone_gray",
        current_stages=("ink_gray_balance",),
        optimization_steps=("core_black_search", "mid_gray_body_search", "opacity_search"),
        report_fields=("local_ink_balance_issues", "gray_band_metrics", "reference_profile"),
        notes="Ink balance separates true-black core, mid-gray body, outer gray, and opacity.",
    ),
    DiagnosticConcernMapping(
        concern_id="edge_quality",
        current_stages=("ink_gray_balance", "photo_texture"),
        optimization_steps=("outer_gray_control", "edge_breakup_match"),
        report_fields=("local_outer_gray_halo_issues", "photo_texture_metrics", "local_photo_texture_issues"),
        notes="Outer gray belongs to ink balance; photographed edge breakup belongs to photo texture.",
    ),
    DiagnosticConcernMapping(
        concern_id="photo_texture",
        current_stages=("photo_texture",),
        optimization_steps=(
            "blur_match",
            "edge_breakup_match",
            "noise_texture_match",
            "jpeg_texture_match",
            "residual_retexture",
        ),
        report_fields=("photo_texture_metrics", "local_photo_texture_issues"),
        notes="Texture matching runs only after text_shape and ink_gray_balance pass.",
    ),
)


def diagnostic_concern_mapping_report() -> list[dict[str, Any]]:
    stage_ids = set(STAGE_ORDER)
    report: list[dict[str, Any]] = []
    for mapping in DIAGNOSTIC_CONCERN_MAPPINGS:
        item = mapping.as_report()
        item["public_stage_ids"] = list(STAGE_ORDER)
        item["current_stages_valid"] = all(stage_id in stage_ids for stage_id in mapping.current_stages)
        item["optimization_step_scope"] = "within_stage_not_public_stage"
        report.append(item)
    return report


def mapping_for_concern(concern_id: str) -> dict[str, Any]:
    for item in diagnostic_concern_mapping_report():
        if item["concern_id"] == concern_id:
            return item
    raise ValueError(f"unknown diagnostic concern: {concern_id}")
