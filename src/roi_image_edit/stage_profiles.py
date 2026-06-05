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
    manual_roi: bool = False

    def as_report(self) -> dict[str, Any]:
        data = asdict(self)
        data["enabled_stage_ids"] = list(self.enabled_stage_ids)
        return data


PHOTO_SCAN = StageProfile(
    id="photo_scan",
    display_name="Photo or scanned document",
    stage_order=STAGE_ORDER,
    enabled_stage_ids=frozenset(STAGE_ORDER),
    description="Default profile for photographed or scanned small text.",
    enable_photo_texture=True,
)

CLEAN_DIGITAL = StageProfile(
    id="clean_digital",
    display_name="Clean digital image",
    stage_order=STAGE_ORDER,
    enabled_stage_ids=frozenset(stage for stage in STAGE_ORDER if stage != "photo_texture"),
    description="Digital screenshots or clean raster text; disables photo texture tuning.",
    enable_photo_texture=False,
)

LOW_RES_THUMBNAIL = StageProfile(
    id="low_res_thumbnail",
    display_name="Low-resolution thumbnail",
    stage_order=STAGE_ORDER,
    enabled_stage_ids=frozenset(STAGE_ORDER),
    description="Very small ROI text where stage thresholds must remain explicit and conservative.",
    enable_photo_texture=True,
)

MANUAL_ROI_QUICK = StageProfile(
    id="manual_roi_quick",
    display_name="Manual ROI quick check",
    stage_order=STAGE_ORDER,
    enabled_stage_ids=frozenset(STAGE_ORDER),
    description="Manual rectangles; still reports all gates but keeps ROI selection outside the profile.",
    enable_photo_texture=True,
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


def stage_profile_summary(profile_id: str | None = None) -> dict[str, Any]:
    return stage_profile(profile_id).as_report()
