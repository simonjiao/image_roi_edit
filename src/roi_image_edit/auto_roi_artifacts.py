from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from roi_image_edit.roi_locator import auto_region_evidence


def auto_roi_evidence_payload(regions: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> dict[str, Any]:
    evidence_items = [auto_region_evidence(region) for region in regions if isinstance(region, dict) and region.get("auto")]
    return {
        "region_count": len(evidence_items),
        "regions": evidence_items,
        "all_have_search_roi": all(bool(item.get("search_roi")) for item in evidence_items) if evidence_items else False,
        "all_have_edit_roi": all(bool(item.get("edit_roi")) for item in evidence_items) if evidence_items else False,
        "all_edit_roi_within_search_roi": all(edit_roi_within_search_roi(item) for item in evidence_items) if evidence_items else False,
        "all_search_area_gte_edit_area": all(
            bool((item.get("roi_geometry") or {}).get("search_area_gte_edit_area"))
            for item in evidence_items
        )
        if evidence_items
        else False,
        "all_edit_roi_avoid_protected_text": all(
            bool((item.get("roi_geometry") or {}).get("edit_avoids_protected_text"))
            for item in evidence_items
        )
        if evidence_items
        else False,
    }


def edit_roi_within_search_roi(item: dict[str, Any]) -> bool:
    search = item.get("search_roi")
    edit = item.get("edit_roi")
    if not isinstance(search, list) or not isinstance(edit, list) or len(search) != 4 or len(edit) != 4:
        return False
    sx1, sy1, sx2, sy2 = [int(value) for value in search]
    ex1, ey1, ex2, ey2 = [int(value) for value in edit]
    return sx1 <= ex1 <= ex2 <= sx2 and sy1 <= ey1 <= ey2 <= sy2


def draw_auto_roi_overlay(
    image: Image.Image,
    regions: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> Image.Image:
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    for region in regions:
        if not isinstance(region, dict) or not region.get("auto"):
            continue
        evidence = auto_region_evidence(region)
        draw_box(draw, evidence.get("search_roi"), outline=(0, 130, 0), width=2)
        draw_box(draw, evidence.get("target_roi"), outline=(0, 86, 210), width=2)
        draw_box(draw, evidence.get("edit_roi"), outline=(210, 28, 28), width=2)
    return overlay


def save_auto_roi_overlay(
    image: Image.Image,
    regions: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    path: Path,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    draw_auto_roi_overlay(image, regions).save(path)
    return str(path)


def draw_box(
    draw: ImageDraw.ImageDraw,
    box: object,
    *,
    outline: tuple[int, int, int],
    width: int,
) -> None:
    if not isinstance(box, list) or len(box) != 4:
        return
    x1, y1, x2, y2 = [int(value) for value in box]
    if x2 <= x1 or y2 <= y1:
        return
    for inset in range(width):
        draw.rectangle((x1 + inset, y1 + inset, x2 - inset, y2 - inset), outline=outline)
