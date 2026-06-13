from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from roi_image_edit.stage_policy import STAGE_ORDER


@dataclass(frozen=True)
class StageProfile:
    id: str
    display_name: str
    stage_order: tuple[str, ...]
    enabled_stage_ids: frozenset[str]
    description: str
    enable_photo_texture: bool = True
    enable_pose: bool = True
    enable_photo_warp: bool = True
    vision_context_scale: str = "standard"
    shape_priority: tuple[str, ...] = ()
    revision_complexity: str = "full"
    preserve_rejected_candidate: bool = True
    edge_policy: str = "match_source_texture"
    manual_roi: bool = False

    def as_report(self) -> dict[str, Any]:
        data = asdict(self)
        data["enabled_stage_ids"] = list(self.enabled_stage_ids)
        data["shape_priority"] = list(self.shape_priority)
        return data


PHOTO_SCAN = StageProfile(
    id="photo_scan",
    display_name="Photo or scanned document",
    stage_order=STAGE_ORDER,
    enabled_stage_ids=frozenset(STAGE_ORDER),
    description="Default profile for photographed or scanned small text.",
    enable_photo_texture=True,
    enable_pose=True,
    enable_photo_warp=True,
    vision_context_scale="standard",
    shape_priority=("font_family_similarity", "stroke_body_weight", "local_pose_match", "photo_texture_match"),
    edge_policy="match_photo_scan_texture",
)

CLEAN_DIGITAL = StageProfile(
    id="clean_digital",
    display_name="Clean digital image",
    stage_order=STAGE_ORDER,
    enabled_stage_ids=frozenset(stage for stage in STAGE_ORDER if stage != "photo_texture"),
    description="Digital screenshots or clean raster text; disables photo texture tuning.",
    enable_photo_texture=False,
    enable_pose=True,
    enable_photo_warp=False,
    vision_context_scale="standard",
    shape_priority=("font_family_similarity", "stroke_body_weight", "clean_edge"),
    edge_policy="clean_edges_no_photo_warp",
)

LOW_RES_THUMBNAIL = StageProfile(
    id="low_res_thumbnail",
    display_name="Low-resolution thumbnail",
    stage_order=STAGE_ORDER,
    enabled_stage_ids=frozenset(STAGE_ORDER),
    description="Very small ROI text where stage thresholds must remain explicit and conservative.",
    enable_photo_texture=True,
    enable_pose=True,
    enable_photo_warp=True,
    vision_context_scale="magnified",
    shape_priority=("font_family_similarity", "stroke_body_weight", "slot_geometry", "ink_gray_density"),
    edge_policy="magnified_low_res_edges",
)

MANUAL_ROI_QUICK = StageProfile(
    id="manual_roi_quick",
    display_name="Manual ROI quick check",
    stage_order=STAGE_ORDER,
    enabled_stage_ids=frozenset(
        ("hard_boundary", "text_shape", "ink_gray_balance")
    ),
    description="Manual rectangles; runs the minimal local gates and preserves rejected candidates instead of auto-running complex texture stages.",
    enable_photo_texture=False,
    enable_pose=True,
    enable_photo_warp=False,
    vision_context_scale="standard",
    shape_priority=("slot_geometry", "font_family_similarity", "stroke_body_weight"),
    revision_complexity="minimal",
    edge_policy="manual_roi_no_complex_texture",
    manual_roi=True,
)

STAGE_PROFILES = {
    profile.id: profile
    for profile in (
        PHOTO_SCAN,
        CLEAN_DIGITAL,
        LOW_RES_THUMBNAIL,
        MANUAL_ROI_QUICK,
    )
}

DEFAULT_STAGE_PROFILE_ID = PHOTO_SCAN.id


def stage_profile(profile_id: str | None = None) -> StageProfile:
    requested = str(profile_id or DEFAULT_STAGE_PROFILE_ID).strip() or DEFAULT_STAGE_PROFILE_ID
    if requested not in STAGE_PROFILES:
        raise ValueError(
            "unknown stage profile "
            f"{requested!r}; expected one of {', '.join(sorted(STAGE_PROFILES))}"
        )
    return STAGE_PROFILES[requested]


def stage_profile_choices() -> tuple[str, ...]:
    return tuple(sorted(STAGE_PROFILES))


def default_stage_profile() -> StageProfile:
    return stage_profile(DEFAULT_STAGE_PROFILE_ID)


def resolve_stage_profile(
    requested_profile: str | None = None,
    suggested_profile: str | None = None,
) -> dict[str, Any]:
    requested = str(requested_profile or "").strip()
    suggested = str(suggested_profile or "").strip()
    if requested:
        profile = stage_profile(requested)
        source = "explicit_request"
    elif suggested:
        profile = stage_profile(suggested)
        source = "auto_suggestion"
    else:
        profile = default_stage_profile()
        source = "default"
    return {
        "id": profile.id,
        "source": source,
        "requested_profile": requested or None,
        "suggested_profile": suggested or None,
        "profile": profile.as_report(),
    }


def resolve_internal_stage_profile(
    classification: dict[str, Any] | None = None,
    *,
    debug_profile: str | None = None,
) -> dict[str, Any]:
    debug = str(debug_profile or "").strip()
    if debug:
        profile = stage_profile(debug)
        return {
            "id": profile.id,
            "source": "debug_override",
            "requested_profile": debug,
            "suggested_profile": (classification or {}).get("internal_profile"),
            "profile_source": "debug_override",
            "classification": classification or {},
            "profile": profile.as_report(),
        }
    profile_id = str((classification or {}).get("internal_profile") or DEFAULT_STAGE_PROFILE_ID)
    profile = stage_profile(profile_id)
    return {
        "id": profile.id,
        "source": "classification",
        "requested_profile": None,
        "suggested_profile": profile.id,
        "profile_source": "classification",
        "classification": classification or {},
        "profile": profile.as_report(),
    }


def stage_profile_summary(profile_id: str | None = None) -> dict[str, Any]:
    return stage_profile(profile_id).as_report()
