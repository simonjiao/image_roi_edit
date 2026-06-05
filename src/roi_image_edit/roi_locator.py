from __future__ import annotations

import re
from typing import Any

import cv2
import numpy as np
from PIL import Image

from roi_image_edit.iterative_pipeline import (
    RenderPlan,
    TextRun,
    clamp_box,
    dark_runs,
    filter_fonts_by_required_text,
    find_font_candidates,
    font_style_score,
    is_mostly_cjk,
    split_run_by_projection,
)


def rotated_auto_orientation_variants(img: Image.Image) -> tuple[tuple[str, Image.Image], ...]:
    return (
        ("none", img),
        ("rotate_90_ccw", img.rotate(90, expand=True)),
        ("rotate_90_cw", img.rotate(270, expand=True)),
        ("rotate_180", img.rotate(180, expand=True)),
    )


def document_orientation_quality(img: Image.Image) -> dict[str, Any]:
    image_w, image_h = img.size
    components = page_text_components(img)
    line_count = 0
    long_line_count = 0
    top_line_count = 0
    left_header_count = 0
    total_line_width = 0.0
    for group in group_page_text_lines(components):
        merged = merge_overlapping_text_components(group)
        if len(merged) < 3:
            continue
        x1 = min(run.x1 for run in merged)
        y1 = min(run.y1 for run in merged)
        x2 = max(run.x2 for run in merged)
        y2 = max(run.y2 for run in merged)
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        if width < max(42, int(round(image_w * 0.06))):
            continue
        if width < height * 3.2:
            continue
        line_count += 1
        total_line_width += min(1.0, width / max(1, image_w))
        center_y = (y1 + y2) / 2.0
        if width >= image_w * 0.28:
            long_line_count += 1
        if center_y <= image_h * 0.34:
            top_line_count += 1
            if x1 <= image_w * 0.42:
                left_header_count += 1
    score = (
        line_count * 4.0
        + long_line_count * 8.0
        + top_line_count * 6.0
        + left_header_count * 10.0
        + total_line_width * 5.0
    )
    if image_w > image_h and top_line_count >= 2:
        score += 12.0
    return {
        "score": round(float(score), 3),
        "component_count": len(components),
        "line_count": line_count,
        "long_line_count": long_line_count,
        "top_line_count": top_line_count,
        "left_header_count": left_header_count,
    }


def auto_orient_for_instruction(
    img: Image.Image,
    *,
    instruction: str,
    source_text: str,
    target_text: str,
) -> tuple[Image.Image, list[dict[str, Any]], dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    first_error: ValueError | None = None
    successes: list[tuple[float, float, int, str, Image.Image, list[dict[str, Any]], dict[str, Any]]] = []
    for orientation_index, (orientation, candidate) in enumerate(rotated_auto_orientation_variants(img)):
        orientation_quality = document_orientation_quality(candidate)
        try:
            regions = auto_select_regions_for_instruction(
                candidate,
                instruction=instruction,
                source_text=source_text,
                target_text=target_text,
            )
        except ValueError as exc:
            if first_error is None:
                first_error = exc
            attempts.append(
                {
                    "orientation": orientation,
                    "size": list(candidate.size),
                    "ok": False,
                    "error": str(exc),
                    "direction_quality": orientation_quality,
                }
            )
            continue
        auto_score = min(float(region.get("_autoScore", 0.0)) for region in regions)
        selection_score = auto_score - float(orientation_quality.get("score") or 0.0)
        attempt = {
            "orientation": orientation,
            "size": list(candidate.size),
            "ok": True,
            "auto_score": round(float(auto_score), 3),
            "direction_score": orientation_quality.get("score"),
            "selection_score": round(float(selection_score), 3),
            "direction_quality": orientation_quality,
            "regions": [
                {
                    "id": str(region.get("id") or ""),
                    "rect": region.get("rect") or {},
                    "auto_score": region.get("_autoScore"),
                    "target_roi": region.get("_autoTargetRoi"),
                }
                for region in regions
            ],
        }
        attempts.append(
            attempt
        )
        successes.append(
            (
                -float(orientation_quality.get("score") or 0.0),
                selection_score,
                orientation_index,
                orientation,
                candidate,
                regions,
                attempt,
            )
        )

    if successes:
        successes.sort(key=lambda item: (item[1], item[0]))
        selected_direction_key, selected_score, _orientation_index, orientation, candidate, regions, selected_attempt = successes[0]
        selection_margin = None
        direction_margin = None
        if len(successes) > 1:
            selection_margin = round(float(successes[1][1] - selected_score), 3)
            direction_margin = round(float(successes[1][0] - selected_direction_key), 3)
        return candidate, regions, {
            "applied": orientation != "none",
            "orientation": orientation,
            "attempts": attempts,
            "selected_score": round(float(selected_score), 3),
            "selection_margin": selection_margin,
            "direction_margin": direction_margin,
            "selected_attempt": selected_attempt,
        }

    if first_error is not None:
        raise first_error
    raise ValueError("无法自动定位旧文字。")


def parse_instruction(text: str) -> tuple[str, str]:
    details = parse_instruction_details(text)
    return details["source_text"], details["target_text"]


def parse_instruction_details(text: str) -> dict[str, Any]:
    raw = " ".join(str(text or "").strip().split())
    op_pattern = r"调整为|调整成|更改为|更改成|变更为|变更成|修改为|修改成|替换为|替换成|改为|改成|换为|换成|->|=>|→"
    field_aliases = sorted(
        (alias for aliases in FIELD_ALIASES.values() for alias in aliases),
        key=len,
        reverse=True,
    )
    field_alias_pattern = "|".join(re.escape(alias) for alias in field_aliases)
    field_only_pattern = (
        rf"^(?P<field>{field_alias_pattern})\s*"
        r"(?:修改|调整|更改|变更|替换|改|换)\s*(?:为|成|到)?\s*(?P<dst>.+)$"
        if field_alias_pattern
        else ""
    )
    if field_only_pattern:
        field_match = re.search(field_only_pattern, raw, flags=re.IGNORECASE)
        if field_match:
            return {
                "raw": raw,
                "field_key": infer_instruction_field(field_match.group("field")),
                "source_text": "",
                "target_text": cleanup_instruction_part(field_match.group("dst")),
                "source_explicit": False,
            }
    patterns = (
        rf"把\s*(?P<src>.+?)\s*(?:{op_pattern})\s*(?P<dst>.+)",
        rf"(?P<src>.+?)\s*(?:{op_pattern})\s*(?P<dst>.+)",
    )
    for pattern in patterns:
        match = re.search(pattern, raw)
        if match:
            raw_source = cleanup_instruction_part(match.group("src"))
            source = cleanup_instruction_part(strip_field_prefix(raw_source))
            target = cleanup_instruction_part(strip_field_prefix(match.group("dst")))
            if source and target:
                return {
                    "raw": raw,
                    "field_key": infer_instruction_field(raw) or infer_instruction_field(raw_source),
                    "source_text": source,
                    "target_text": target,
                    "source_explicit": True,
                }
    if raw:
        return {
            "raw": raw,
            "field_key": infer_instruction_field(raw),
            "source_text": "",
            "target_text": cleanup_instruction_part(raw),
            "source_explicit": False,
        }
    return {"raw": "", "field_key": None, "source_text": "", "target_text": "", "source_explicit": False}


def cleanup_instruction_part(value: str) -> str:
    return str(value or "").strip(" ：:，,。.;；\"'`[]()（）")


FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "name": ("姓名", "名字", "患者姓名", "姓名名", "name"),
    "department": ("科室", "科别", "就诊科室", "就诊科别", "department", "dept"),
    "receive_time": ("接收时间", "接受时间", "采样时间", "收样时间", "receive time"),
    "date": ("日期", "时间", "检查日期", "报告日期", "出生日期", "date"),
    "age": ("年龄", "岁数", "age"),
}


def infer_instruction_field(value: str) -> str | None:
    raw = str(value or "").lower()
    compact = re.sub(r"\s+", "", raw)
    for field, aliases in FIELD_ALIASES.items():
        if any(alias.lower() in compact for alias in aliases):
            return field
    return None


def strip_field_prefix(value: str) -> str:
    text = str(value or "").strip()
    compact_aliases = sorted(
        (alias for aliases in FIELD_ALIASES.values() for alias in aliases),
        key=len,
        reverse=True,
    )
    for alias in compact_aliases:
        alias_pattern = r"\s*".join(re.escape(char) for char in alias)
        pattern = rf"^\s*(?:字段|field)?\s*{alias_pattern}\s*[:：,，;；-]*\s*"
        stripped = re.sub(pattern, "", text, flags=re.IGNORECASE)
        if stripped != text:
            return stripped
    return text


def text_chars(value: str) -> list[str]:
    return [ch for ch in str(value or "") if not ch.isspace()]


def text_run_center_y(run: TextRun) -> float:
    return (run.y1 + run.y2) / 2.0


def text_run_center_x(run: TextRun) -> float:
    return (run.x1 + run.x2) / 2.0


def estimate_text_angle_degrees(slots: tuple[TextRun, ...]) -> float:
    if len(slots) < 2:
        return 0.0
    ordered = tuple(sorted(slots, key=lambda item: item.x1))
    xs = np.array([text_run_center_x(slot) for slot in ordered], dtype=np.float32)
    ys = np.array([text_run_center_y(slot) for slot in ordered], dtype=np.float32)
    if float(xs.max() - xs.min()) < 8.0:
        return 0.0
    try:
        slope = float(np.polyfit(xs, ys, 1)[0])
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return 0.0
    angle = float(np.degrees(np.arctan(slope)))
    if not np.isfinite(angle):
        return 0.0
    return max(-8.0, min(8.0, angle))


def text_run_height(run: TextRun) -> int:
    return max(1, run.y2 - run.y1)


def vertical_overlap_ratio(a: TextRun, b: TextRun) -> float:
    overlap = max(0, min(a.y2, b.y2) - max(a.y1, b.y1))
    return overlap / max(1, min(text_run_height(a), text_run_height(b)))


def component_text_runs(
    img: Image.Image,
    roi: tuple[int, int, int, int],
    *,
    threshold: int,
) -> tuple[TextRun, ...]:
    x1, y1, x2, y2 = roi
    roi_w = max(1, x2 - x1)
    roi_h = max(1, y2 - y1)
    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    mask = (gray[y1:y2, x1:x2] < threshold).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    runs: list[TextRun] = []
    min_area = max(6, int(round(roi_w * roi_h * 0.00035)))
    min_h = max(4, int(round(roi_h * 0.08)))
    for idx in range(1, num):
        cx, cy, cw, ch, area = (int(v) for v in stats[idx])
        if area < min_area or cw <= 1 or ch <= 1:
            continue
        if ch < min_h:
            continue
        if cw >= roi_w * 0.45 and ch <= max(4, int(round(roi_h * 0.25))):
            continue
        runs.append(
            TextRun(
                x1=x1 + cx,
                y1=y1 + cy,
                x2=x1 + cx + cw,
                y2=y1 + cy + ch,
                area=area,
            )
        )
    return tuple(sorted(runs, key=lambda item: (item.x1, item.y1)))


def dominant_text_line(
    runs: tuple[TextRun, ...],
    roi: tuple[int, int, int, int],
) -> tuple[TextRun, ...]:
    if not runs:
        return ()
    roi_cy = (roi[1] + roi[3]) / 2.0
    groups: list[tuple[float, int, tuple[TextRun, ...]]] = []
    for seed in runs:
        seed_h = text_run_height(seed)
        group = tuple(
            run
            for run in runs
            if vertical_overlap_ratio(seed, run) >= 0.30
            or abs(text_run_center_y(seed) - text_run_center_y(run))
            <= max(5.0, max(seed_h, text_run_height(run)) * 0.55)
        )
        total_area = sum(run.area for run in group)
        group_cy = sum(text_run_center_y(run) * run.area for run in group) / max(1, total_area)
        center_penalty = abs(group_cy - roi_cy) * 80.0
        groups.append((len(group) * 300.0 + total_area - center_penalty, total_area, group))
    _, _, selected = max(groups, key=lambda item: (item[0], item[1]))
    return tuple(sorted(selected, key=lambda item: item.x1))


def compact_run_window(runs: tuple[TextRun, ...], count: int) -> tuple[TextRun, ...]:
    if count <= 0 or len(runs) <= count:
        return runs
    ordered = tuple(sorted(runs, key=lambda item: item.x1))
    best: tuple[float, tuple[TextRun, ...]] | None = None
    for start in range(0, len(ordered) - count + 1):
        window = ordered[start : start + count]
        gaps = [window[idx + 1].x1 - window[idx].x2 for idx in range(len(window) - 1)]
        span = window[-1].x2 - window[0].x1
        max_gap = max(gaps) if gaps else 0
        total_area = sum(run.area for run in window)
        score = max_gap * 5.0 + span - total_area * 0.03
        if best is None or score < best[0]:
            best = (score, window)
    return best[1] if best else ordered[-count:]


def glyph_like_runs(
    runs: tuple[TextRun, ...],
    *,
    count: int,
    reference_text: str,
) -> tuple[TextRun, ...]:
    if count <= 0 or len(runs) <= count or not is_mostly_cjk(reference_text):
        return runs
    heights = sorted((text_run_height(run) for run in runs), reverse=True)
    areas = sorted((run.area for run in runs), reverse=True)
    sample_size = min(len(runs), max(1, count))
    ref_height = float(np.median(heights[:sample_size]))
    ref_area = float(np.median(areas[:sample_size]))
    filtered = tuple(
        run
        for run in runs
        if text_run_height(run) >= ref_height * 0.55
        and run.area >= ref_area * 0.22
    )
    return filtered if len(filtered) >= count else runs


def merge_text_runs(runs: tuple[TextRun, ...]) -> TextRun | None:
    if not runs:
        return None
    return TextRun(
        x1=min(run.x1 for run in runs),
        y1=min(run.y1 for run in runs),
        x2=max(run.x2 for run in runs),
        y2=max(run.y2 for run in runs),
        area=sum(run.area for run in runs),
    )


def is_horizontal_rule_run(run: TextRun, roi: tuple[int, int, int, int]) -> bool:
    width = max(1, run.x2 - run.x1)
    height = max(1, run.y2 - run.y1)
    roi_w = max(1, roi[2] - roi[0])
    roi_h = max(1, roi[3] - roi[1])
    max_rule_h = max(3, int(round(roi_h * 0.12)))
    min_rule_w = max(12, int(round(roi_w * 0.12)), height * 6)
    return height <= max_rule_h and width >= min_rule_w


def text_like_dark_runs(
    img: Image.Image,
    roi: tuple[int, int, int, int],
    *,
    threshold: int,
    min_area: int,
) -> tuple[TextRun, ...]:
    runs = tuple(dark_runs(img, roi, threshold=threshold, min_area=min_area))
    filtered = tuple(run for run in runs if not is_horizontal_rule_run(run, roi))
    return filtered if filtered else runs


def merge_overlapping_text_components(runs: tuple[TextRun, ...]) -> tuple[TextRun, ...]:
    if not runs:
        return ()
    ordered = sorted(runs, key=lambda item: (item.x1, item.y1))
    merged: list[TextRun] = []
    current = ordered[0]
    for run in ordered[1:]:
        x_overlap = min(current.x2, run.x2) - max(current.x1, run.x1)
        x_gap = run.x1 - current.x2
        y_overlap = min(current.y2, run.y2) - max(current.y1, run.y1)
        if x_overlap >= -1 and (y_overlap > 0 or x_gap <= 1):
            current = TextRun(
                x1=min(current.x1, run.x1),
                y1=min(current.y1, run.y1),
                x2=max(current.x2, run.x2),
                y2=max(current.y2, run.y2),
                area=current.area + run.area,
            )
            continue
        merged.append(current)
        current = run
    merged.append(current)
    return tuple(merged)


def slots_are_compact_source(
    slots: tuple[TextRun, ...],
    roi: tuple[int, int, int, int],
    *,
    desired_count: int,
) -> bool:
    if desired_count <= 1 or len(slots) < desired_count:
        return True
    ordered = tuple(sorted(slots[:desired_count], key=lambda item: item.x1))
    widths = [max(1, slot.x2 - slot.x1) for slot in ordered]
    heights = [max(1, slot.y2 - slot.y1) for slot in ordered]
    span = ordered[-1].x2 - ordered[0].x1
    inner_gaps = [
        ordered[idx + 1].x1 - ordered[idx].x2
        for idx in range(len(ordered) - 1)
    ]
    median_width = float(np.median(widths))
    median_height = float(np.median(heights))
    roi_w = max(1, roi[2] - roi[0])
    max_span = min(roi_w * 0.52, median_width * desired_count * 3.2 + median_height)
    max_gap = max(14.0, median_width * 1.25, median_height * 0.65)
    return span <= max_span and (not inner_gaps or max(inner_gaps) <= max_gap)


def source_slots_after_label_components(
    img: Image.Image,
    roi: tuple[int, int, int, int],
    *,
    source_text: str,
    target_text: str,
) -> tuple[TextRun, ...]:
    basis_text = source_text or target_text
    desired_count = len(text_chars(source_text)) or len(text_chars(target_text))
    if desired_count < 2 or not is_mostly_cjk(basis_text):
        return ()

    roi_w = max(1, roi[2] - roi[0])
    for threshold in (165, 150, 135):
        components = merge_overlapping_text_components(
            tuple(
                sorted(
                    dominant_text_line(component_text_runs(img, roi, threshold=threshold), roi),
                    key=lambda item: item.x1,
                )
            )
        )
        if len(components) < desired_count + 1:
            continue

        heights = [text_run_height(run) for run in components]
        median_h = float(np.median(heights)) if heights else 1.0
        max_label_x = roi[0] + roi_w * 0.45
        max_value_start_x = roi[0] + roi_w * 0.70
        max_prev_gap = max(24.0, median_h * 1.35)
        max_inner_gap = max(12.0, median_h * 0.65)
        min_glyph_w = max(4.0, median_h * 0.30)
        best: tuple[float, tuple[TextRun, ...]] | None = None

        for start in range(1, len(components) - desired_count + 1):
            window = components[start : start + desired_count]
            if window[0].x1 > max_value_start_x:
                continue
            prev = components[start - 1]
            if prev.x1 > max_label_x:
                continue
            prev_gap = window[0].x1 - prev.x2
            if prev_gap < 0 or prev_gap > max_prev_gap:
                continue
            widths = [max(1, run.x2 - run.x1) for run in window]
            if any(width < min_glyph_w for width in widths):
                continue
            window_heights = [text_run_height(run) for run in window]
            if any(height < max(window_heights) * 0.55 for height in window_heights):
                continue
            inner_gaps = [
                window[idx + 1].x1 - window[idx].x2
                for idx in range(len(window) - 1)
            ]
            if any(gap < -2 for gap in inner_gaps):
                continue
            if inner_gaps and max(inner_gaps) > max_inner_gap:
                continue
            span = window[-1].x2 - window[0].x1
            if span < median_h * desired_count * 0.62:
                continue
            if span > max(roi_w * 0.45, median_h * desired_count * 2.0):
                continue
            score = (
                sum(run.area for run in window) * 0.12
                + span * 0.25
                - abs(prev_gap - median_h * 0.45) * 0.55
                - start * 0.15
                - abs(threshold - 165) * 0.02
            )
            if best is None or score > best[0]:
                best = (score, tuple(window))
        if best is not None:
            return best[1]
    return ()


def source_slots_after_label_gap(
    img: Image.Image,
    roi: tuple[int, int, int, int],
    *,
    source_text: str,
    target_text: str,
) -> tuple[TextRun, ...]:
    basis_text = source_text or target_text
    desired_count = len(text_chars(source_text)) or len(text_chars(target_text))
    if desired_count < 2 or not is_mostly_cjk(basis_text):
        return ()

    roi_w = max(1, roi[2] - roi[0])
    for threshold in (120, 105, 135):
        runs = text_like_dark_runs(img, roi, threshold=threshold, min_area=4)
        ordered = tuple(sorted(dominant_text_line(runs, roi), key=lambda item: item.x1))
        if len(ordered) < desired_count + 2:
            continue

        heights = [text_run_height(run) for run in ordered]
        median_h = float(np.median(heights)) if heights else 1.0
        row_width = max(1, ordered[-1].x2 - ordered[0].x1)
        separator_max_w = max(4.0, median_h * 0.42)
        separator_max_h = max(10.0, median_h * 1.05)
        nearby_gap = max(14.0, median_h * 0.90)
        source_start_gap = max(28.0, median_h * 1.45, row_width * 0.12)
        source_inner_gap = max(12.0, median_h * 0.70, row_width * 0.06)
        best: tuple[float, tuple[TextRun, ...]] | None = None

        for idx in range(1, len(ordered) - 1):
            separator = ordered[idx]
            sep_w = separator.x2 - separator.x1
            sep_h = separator.y2 - separator.y1
            prev_gap = separator.x1 - ordered[idx - 1].x2
            next_gap = ordered[idx + 1].x1 - separator.x2
            if sep_w > separator_max_w or sep_h > separator_max_h:
                continue
            if prev_gap < 0 or prev_gap > nearby_gap:
                continue
            if next_gap < 0 or next_gap > source_start_gap:
                continue

            for component_threshold in (125, 120, 130, 115):
                components = tuple(
                    sorted(
                        dominant_text_line(
                            component_text_runs(img, roi, threshold=component_threshold),
                            roi,
                        ),
                        key=lambda item: item.x1,
                    )
                )
                cluster: list[TextRun] = []
                for run in components:
                    if run.x1 <= separator.x2:
                        continue
                    if not cluster and run.x1 - separator.x2 > source_start_gap:
                        break
                    if cluster:
                        gap = run.x1 - cluster[-1].x2
                        if gap >= source_inner_gap:
                            break
                    cluster.append(run)
                if len(cluster) < 1:
                    continue
                cluster_box = merge_text_runs(tuple(cluster))
                if cluster_box is None:
                    continue
                cluster_w = cluster_box.x2 - cluster_box.x1
                if cluster_w < median_h * desired_count * 0.45:
                    continue
                if cluster_w > max(roi_w * 0.55, median_h * desired_count * 3.2):
                    continue

                split = split_run_by_projection(
                    img,
                    cluster_box,
                    parts=desired_count,
                    threshold=component_threshold,
                )
                if len(split) < desired_count:
                    continue
                selected = tuple(sorted(split[:desired_count], key=lambda item: item.x1))
                selected_span = selected[-1].x2 - selected[0].x1
                score = (
                    cluster_box.area * 0.08
                    + selected_span * 0.25
                    - abs(next_gap - median_h * 0.8) * 0.4
                    - idx * 0.2
                    - abs(component_threshold - 125) * 0.03
                )
                if best is None or score > best[0]:
                    best = (score, selected)

        if best is not None:
            return best[1]
    return ()


def source_slots_after_label_run(
    img: Image.Image,
    roi: tuple[int, int, int, int],
    *,
    source_text: str,
    target_text: str,
) -> tuple[TextRun, ...]:
    basis_text = source_text or target_text
    desired_count = len(text_chars(source_text)) or len(text_chars(target_text))
    if desired_count < 2 or not is_mostly_cjk(basis_text):
        return ()

    roi_w = max(1, roi[2] - roi[0])
    for threshold in (105, 120, 135):
        runs = text_like_dark_runs(img, roi, threshold=threshold, min_area=4)
        ordered = tuple(sorted(dominant_text_line(runs, roi), key=lambda item: item.x1))
        if len(ordered) < 2:
            continue
        heights = [text_run_height(run) for run in ordered]
        median_h = float(np.median(heights)) if heights else 1.0
        max_label_x = roi[0] + roi_w * 0.40
        max_label_to_value_gap = max(24.0, median_h * 0.85)
        min_value_w = median_h * desired_count * 0.45
        max_value_w = median_h * desired_count * 2.05
        min_next_gap = max(12.0, median_h * 0.45)
        best: tuple[float, tuple[TextRun, ...]] | None = None

        for idx in range(1, len(ordered)):
            label = ordered[idx - 1]
            value = ordered[idx]
            if label.x1 > max_label_x:
                continue
            label_to_value_gap = value.x1 - label.x2
            if label_to_value_gap < 0 or label_to_value_gap > max_label_to_value_gap:
                continue
            value_w = value.x2 - value.x1
            if value_w < min_value_w or value_w > max_value_w:
                continue
            if idx + 1 < len(ordered):
                next_gap = ordered[idx + 1].x1 - value.x2
                if next_gap < min_next_gap:
                    continue
            else:
                next_gap = roi[2] - value.x2

            split = split_run_by_projection(
                img,
                value,
                parts=desired_count,
                threshold=threshold,
            )
            if len(split) < desired_count:
                continue
            selected = tuple(sorted(split[:desired_count], key=lambda item: item.x1))
            score = (
                value.area * 0.15
                + value_w * 0.35
                + min(next_gap, median_h * 1.4) * 0.40
                - abs(label_to_value_gap - median_h * 0.20) * 0.30
            )
            if best is None or score > best[0]:
                best = (score, selected)
        if best is not None:
            return best[1]
    return ()


def source_slots_from_projection(
    img: Image.Image,
    roi: tuple[int, int, int, int],
    *,
    source_text: str,
    target_text: str,
) -> tuple[TextRun, ...]:
    basis_text = source_text or target_text
    desired_count = len(text_chars(source_text)) or len(text_chars(target_text))
    if desired_count <= 0 or not is_mostly_cjk(basis_text):
        return ()

    for threshold in (105, 120, 135):
        runs = text_like_dark_runs(img, roi, threshold=threshold, min_area=8)
        if len(runs) < desired_count:
            continue
        line_runs = dominant_text_line(runs, roi)
        glyph_runs = glyph_like_runs(
            line_runs,
            count=desired_count,
            reference_text=basis_text,
        )
        if len(glyph_runs) < desired_count:
            continue
        if len(glyph_runs) == desired_count:
            selected = tuple(sorted(glyph_runs, key=lambda item: item.x1))
            if slots_are_compact_source(selected, roi, desired_count=desired_count):
                return selected
            continue

        selected = slots_after_separator_gap(glyph_runs, desired_count)
        if selected and slots_are_compact_source(selected, roi, desired_count=desired_count):
            return selected

    return ()


def slots_after_separator_gap(
    runs: tuple[TextRun, ...],
    count: int,
) -> tuple[TextRun, ...]:
    if count <= 0 or len(runs) <= count:
        return ()

    ordered = tuple(sorted(runs, key=lambda item: item.x1))
    widths = [max(1, run.x2 - run.x1) for run in ordered]
    median_width = float(np.median(widths))
    min_separator_gap = max(6.0, median_width * 0.45)
    best: tuple[float, tuple[TextRun, ...]] | None = None
    for idx in range(len(ordered) - 1):
        gap = ordered[idx + 1].x1 - ordered[idx].x2
        suffix = ordered[idx + 1 :]
        if gap < min_separator_gap or len(suffix) < count:
            continue
        window = suffix[:count]
        inner_gaps = [window[item + 1].x1 - window[item].x2 for item in range(len(window) - 1)]
        max_inner_gap = max(inner_gaps) if inner_gaps else 0
        span = window[-1].x2 - window[0].x1
        score = gap * 8.0 - max_inner_gap * 3.0 - span * 0.12
        if best is None or score > best[0]:
            best = (score, window)
    return best[1] if best else ()


def text_run_box(run: TextRun) -> tuple[int, int, int, int]:
    return run.x1, run.y1, run.x2, run.y2


def clamp_box_to_container(
    box: tuple[int, int, int, int],
    container: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    x1 = max(container[0], box[0])
    y1 = max(container[1], box[1])
    x2 = min(container[2], box[2])
    y2 = min(container[3], box[3])
    if x2 <= x1 or y2 <= y1:
        return container
    return x1, y1, x2, y2


def box_area(box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def box_overlap_area(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> int:
    return box_area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def run_overlaps_slots(run: TextRun, slots: tuple[TextRun, ...]) -> bool:
    run_box = text_run_box(run)
    for slot in slots:
        slot_box = text_run_box(slot)
        overlap = box_overlap_area(run_box, slot_box)
        if overlap / max(1, min(box_area(run_box), box_area(slot_box))) >= 0.45:
            return True
    return False


def context_text_runs(
    img: Image.Image,
    roi: tuple[int, int, int, int],
) -> tuple[TextRun, ...]:
    best: tuple[TextRun, ...] = ()
    for threshold in (105, 120, 135):
        runs = tuple(dark_runs(img, roi, threshold=threshold, min_area=8))
        if not runs:
            continue
        line_runs = dominant_text_line(runs, roi)
        if len(line_runs) > len(best):
            best = line_runs
    return tuple(sorted(best, key=lambda item: item.x1))


def protected_boxes_for_region(
    img: Image.Image,
    roi: tuple[int, int, int, int],
    slots: tuple[TextRun, ...],
) -> tuple[tuple[int, int, int, int], ...]:
    if not slots:
        return ()
    boxes: list[tuple[int, int, int, int]] = []
    for run in context_text_runs(img, roi):
        if run_overlaps_slots(run, slots):
            continue
        pad = 1
        boxes.append(
            clamp_box(
                (run.x1 - pad, run.y1 - pad, run.x2 + pad, run.y2 + pad),
                img.size,
            )
        )
    return tuple(boxes)


def protected_box_overlaps_row(
    box: tuple[int, int, int, int],
    row: tuple[int, int, int, int],
) -> bool:
    overlap = max(0, min(box[3], row[3]) - max(box[1], row[1]))
    return overlap / max(1, min(box[3] - box[1], row[3] - row[1])) >= 0.30


def expand_roi_for_longer_replacement(
    target_roi: tuple[int, int, int, int],
    search_roi: tuple[int, int, int, int],
    *,
    slots: tuple[TextRun, ...],
    protected_boxes: tuple[tuple[int, int, int, int], ...],
    source_text: str,
    target_text: str,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    source_count = len(text_chars(source_text))
    target_count = len(text_chars(target_text))
    if source_count <= 0 or target_count <= source_count or not slots:
        return target_roi

    x1, y1, x2, y2 = target_roi
    current_width = max(1, x2 - x1)
    slot_widths = [max(1, slot.x2 - slot.x1) for slot in slots]
    slot_gaps = [
        max(0, slots[idx + 1].x1 - slots[idx].x2)
        for idx in range(len(slots) - 1)
    ]
    median_width = float(np.median(slot_widths)) if slot_widths else current_width / source_count
    median_gap = float(np.median(slot_gaps)) if slot_gaps else max(2.0, median_width * 0.14)
    desired_width = max(
        current_width,
        int(round(current_width * target_count / source_count)),
        int(round(median_width * target_count + median_gap * max(0, target_count - 1) + median_width * 0.45)),
    )

    margin = 3
    left_limit = search_roi[0]
    right_limit = search_roi[2]
    for box in protected_boxes:
        if not protected_box_overlaps_row(box, target_roi):
            continue
        if box[0] < x1:
            left_limit = max(left_limit, box[2] + margin)
        elif box[0] >= x2:
            right_limit = min(right_limit, box[0] - margin)

    new_x1 = max(x1, left_limit)
    new_x2 = min(right_limit, max(x2, new_x1 + desired_width))
    if new_x2 <= new_x1:
        return target_roi
    return clamp_box((new_x1, y1, new_x2, y2), image_size)


def expand_roi_for_shorter_replacement(
    target_roi: tuple[int, int, int, int],
    search_roi: tuple[int, int, int, int],
    *,
    slots: tuple[TextRun, ...],
    source_text: str,
    target_text: str,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    source_count = len(text_chars(source_text))
    target_count = len(text_chars(target_text))
    if source_count <= target_count or target_count <= 0 or not slots:
        return target_roi
    slot_widths = [max(1, slot.x2 - slot.x1) for slot in slots]
    slot_heights = [max(1, slot.y2 - slot.y1) for slot in slots]
    pad_x = max(8, int(round(float(np.median(slot_widths)) * 1.15)))
    pad_y = max(4, int(round(float(np.median(slot_heights)) * 0.18)))
    expanded = (
        max(search_roi[0], target_roi[0] - 1),
        max(search_roi[1], target_roi[1] - pad_y),
        min(search_roi[2], target_roi[2] + pad_x),
        min(search_roi[3], target_roi[3] + pad_y),
    )
    return clamp_box(expanded, image_size)


def expand_auto_search_roi(
    img: Image.Image,
    roi: tuple[int, int, int, int],
    plan: RenderPlan,
    *,
    source_text: str,
    target_text: str,
) -> tuple[int, int, int, int]:
    source_count = len(text_chars(source_text))
    target_count = len(text_chars(target_text))
    if source_count <= 0 or not plan.slot_boxes:
        return roi
    slot_widths = [max(1, slot.x2 - slot.x1) for slot in plan.slot_boxes]
    slot_heights = [max(1, slot.y2 - slot.y1) for slot in plan.slot_boxes]
    median_width = float(np.median(slot_widths)) if slot_widths else 1.0
    median_height = float(np.median(slot_heights)) if slot_heights else 1.0
    right_pad = max(28, int(round(median_width * (2.7 if source_count > target_count else 0.9))))
    bottom_pad = max(4, int(round(median_height * 0.20))) if source_count > target_count else 0

    row = TextRun(
        x1=plan.target_roi[0],
        y1=min(slot.y1 for slot in plan.slot_boxes),
        x2=plan.target_roi[2],
        y2=max(slot.y2 for slot in plan.slot_boxes),
        area=1,
    )
    next_text_x = img.size[0]
    for run in page_text_components(img):
        if run.x1 <= plan.target_roi[2] + 2:
            continue
        if vertical_overlap_ratio(run, row) < 0.25:
            continue
        next_text_x = min(next_text_x, run.x1)
    right_limit = min(img.size[0], next_text_x - 3)
    new_x2 = min(right_limit, roi[2] + right_pad)
    new_y2 = min(img.size[1], roi[3] + bottom_pad)
    if new_x2 <= roi[2] and new_y2 <= roi[3]:
        return roi
    return clamp_box((roi[0], roi[1], new_x2, new_y2), img.size)


def auto_edit_roi_from_plan(
    img: Image.Image,
    roi: tuple[int, int, int, int],
    plan: RenderPlan,
    *,
    source_text: str,
    target_text: str,
    field_key: str | None,
) -> tuple[int, int, int, int]:
    if not plan.slot_boxes:
        return roi
    if field_key == "name":
        return expand_auto_search_roi(
            img,
            roi,
            plan,
            source_text=source_text,
            target_text=target_text,
        )

    source_count = len(text_chars(source_text))
    target_count = len(text_chars(target_text))
    if source_count <= 0:
        return roi

    slots = tuple(sorted(plan.slot_boxes, key=lambda item: item.x1)[:source_count])
    if not slots:
        return roi
    value_box = slots_roi(slots, img.size) or plan.target_roi
    slot_widths = [max(1, slot.x2 - slot.x1) for slot in slots]
    slot_heights = [max(1, slot.y2 - slot.y1) for slot in slots]
    median_width = float(np.median(slot_widths)) if slot_widths else max(1, value_box[2] - value_box[0])
    median_height = float(np.median(slot_heights)) if slot_heights else max(1, value_box[3] - value_box[1])
    left_pad = max(3, int(round(median_width * 0.16)))
    right_pad = max(8, int(round(median_width * (0.75 if target_count <= source_count else 1.35))))
    vertical_pad = max(7, int(round(median_height * 0.38)))

    row = (value_box[0], value_box[1], value_box[2], value_box[3])
    right_limit = roi[2]
    for box in plan.protected_boxes:
        if box[0] <= value_box[2]:
            continue
        if not protected_box_overlaps_row(box, row):
            continue
        right_limit = min(right_limit, box[0] - 2)
    if right_limit <= value_box[2]:
        right_limit = value_box[2] + right_pad

    desired_right = value_box[2] + right_pad
    if target_count > source_count:
        desired_width = int(round((value_box[2] - value_box[0]) * target_count / source_count + median_width * 0.5))
        desired_right = max(desired_right, value_box[0] + desired_width)

    top_limit = 0
    bottom_limit = img.size[1]
    for run in page_text_components(img):
        horizontal_gap = max(value_box[0] - run.x2, run.x1 - value_box[2], 0)
        if horizontal_gap > max(80.0, median_width * max(2, source_count) * 0.80):
            continue
        if run.y2 <= value_box[1] - 3:
            top_limit = max(top_limit, run.y2 + 1)
        elif run.y1 >= value_box[3] + 3:
            bottom_limit = min(bottom_limit, run.y1 - 1)
    return clamp_box(
        (
            max(roi[0], value_box[0] - left_pad),
            max(top_limit, value_box[1] - vertical_pad),
            min(right_limit, desired_right),
            min(bottom_limit, value_box[3] + vertical_pad),
        ),
        img.size,
    )


def slots_roi(
    slots: tuple[TextRun, ...],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    if not slots:
        return None
    widths = [max(1, slot.x2 - slot.x1) for slot in slots]
    heights = [max(1, slot.y2 - slot.y1) for slot in slots]
    pad_x = max(2, int(round(float(np.median(widths)) * 0.18)))
    pad_y = max(2, int(round(float(np.median(heights)) * 0.18)))
    return clamp_box(
        (
            min(slot.x1 for slot in slots) - pad_x,
            min(slot.y1 for slot in slots) - pad_y,
            max(slot.x2 for slot in slots) + pad_x,
            max(slot.y2 for slot in slots) + pad_y,
        ),
        image_size,
    )


def source_text_body_y_bounds(
    img: Image.Image,
    roi: tuple[int, int, int, int],
    slots: tuple[TextRun, ...],
    *,
    threshold: int = 165,
) -> tuple[int, int] | None:
    if not slots:
        return None
    rx1, ry1, rx2, ry2 = roi
    sx1 = max(rx1, min(slot.x1 for slot in slots))
    sx2 = min(rx2, max(slot.x2 for slot in slots))
    if sx2 <= sx1 or ry2 <= ry1:
        return None

    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    local = gray[ry1:ry2, rx1:rx2] < threshold
    if not np.any(local):
        return None

    row_counts = np.count_nonzero(local, axis=1)
    rule_cutoff = max(24, int(round((rx2 - rx1) * 0.28)))
    rule_rows = row_counts >= rule_cutoff
    if np.any(rule_rows):
        expanded = rule_rows.copy()
        for offset in (-2, -1, 1, 2):
            shifted = np.zeros_like(rule_rows)
            if offset < 0:
                shifted[:offset] = rule_rows[-offset:]
            else:
                shifted[offset:] = rule_rows[:-offset]
            expanded |= shifted
        local = local.copy()
        local[expanded, :] = False

    slot_mask = local[:, sx1 - rx1 : sx2 - rx1]
    row_body_counts = np.count_nonzero(slot_mask, axis=1)
    rows = np.where(row_body_counts >= 2)[0]
    if len(rows) == 0:
        rows = np.where(row_body_counts > 0)[0]
    if len(rows) == 0:
        return None

    body_y1 = int(ry1 + rows.min())
    body_y2 = int(ry1 + rows.max() + 1)
    if body_y2 - body_y1 < 4:
        return None
    return body_y1, body_y2


def non_cjk_value_slot_after_label(
    img: Image.Image,
    roi: tuple[int, int, int, int],
    *,
    source_text: str,
    target_text: str,
) -> tuple[TextRun, ...]:
    basis_text = source_text or target_text
    if not basis_text or is_mostly_cjk(basis_text):
        return ()

    best: tuple[float, TextRun] | None = None
    best_line_fallback: tuple[float, TextRun] | None = None
    for threshold in (150, 135, 120, 165):
        components = merge_overlapping_text_components(
            tuple(
                sorted(
                    dominant_text_line(component_text_runs(img, roi, threshold=threshold), roi),
                    key=lambda item: item.x1,
                )
            )
        )
        if len(components) < 2:
            continue
        heights = [text_run_height(run) for run in components]
        median_h = float(np.median(heights)) if heights else 1.0
        line_value = merge_text_runs(tuple(components))
        if line_value is not None:
            line_w = line_value.x2 - line_value.x1
            line_h = line_value.y2 - line_value.y1
            if line_w >= max(12.0, median_h * 0.8) and line_h >= max(6.0, median_h * 0.35):
                line_score = line_value.area * 0.06 + line_w * 0.22 - abs(threshold - 150) * 0.03
                if best_line_fallback is None or line_score > best_line_fallback[0]:
                    best_line_fallback = (line_score, line_value)
        for idx, separator in enumerate(components):
            if idx == 0 or not is_colon_like_run(separator, median_h):
                continue
            label = components[idx - 1]
            if separator.x1 - label.x2 > max(22.0, median_h * 0.95):
                continue

            value_runs: list[TextRun] = []
            last_x2 = separator.x2
            for run in components[idx + 1 :]:
                gap = run.x1 - last_x2
                if value_runs:
                    current_center_y = float(np.median([(item.y1 + item.y2) / 2.0 for item in value_runs]))
                    run_center_y = (run.y1 + run.y2) / 2.0
                    if abs(run_center_y - current_center_y) > max(8.0, median_h * 0.50):
                        break
                if value_runs and gap > max(30.0, median_h * 1.45):
                    break
                if not value_runs and gap > max(38.0, median_h * 1.85):
                    break
                if is_colon_like_run(run, median_h) and value_runs and gap > median_h * 0.45:
                    break
                value_runs.append(run)
                last_x2 = run.x2
            if not value_runs:
                continue

            value = merge_text_runs(tuple(value_runs))
            if value is None:
                continue
            value_w = value.x2 - value.x1
            value_h = value.y2 - value.y1
            if value_w < max(12.0, median_h * 0.8) or value_h < max(6.0, median_h * 0.35):
                continue
            score = value.area * 0.08 + value_w * 0.28 - abs((value.y1 + value.y2) / 2.0 - (roi[1] + roi[3]) / 2.0)
            score -= abs(threshold - 150) * 0.03
            if best is None or score > best[0]:
                best = (score, value)
    if best is not None:
        return (best[1],)
    return (best_line_fallback[1],) if best_line_fallback is not None else ()


def split_cjk_value_cluster(
    img: Image.Image,
    runs: tuple[TextRun, ...],
    *,
    desired_count: int,
    threshold: int,
    median_h: float,
) -> tuple[TextRun, ...]:
    if desired_count <= 0 or not runs:
        return ()
    value = merge_text_runs(tuple(sorted(runs, key=lambda item: item.x1)))
    if value is None:
        return ()
    value_w = value.x2 - value.x1
    if value_w < max(12.0, median_h * desired_count * 0.62):
        return tuple(sorted(runs[:desired_count], key=lambda item: item.x1))
    split = split_run_by_projection(
        img,
        value,
        parts=desired_count,
        threshold=threshold,
    )
    if len(split) >= desired_count:
        return tuple(sorted(split[:desired_count], key=lambda item: item.x1))
    return tuple(sorted(runs[:desired_count], key=lambda item: item.x1))


def cjk_value_slots_after_colon(
    img: Image.Image,
    roi: tuple[int, int, int, int],
    *,
    source_text: str,
    target_text: str,
) -> tuple[TextRun, ...]:
    basis_text = source_text or target_text
    desired_count = len(text_chars(source_text)) or len(text_chars(target_text))
    if desired_count < 2 or not is_mostly_cjk(basis_text):
        return ()

    roi_w = max(1, roi[2] - roi[0])
    for threshold in (150, 135, 165, 120):
        components = merge_overlapping_text_components(
            tuple(sorted(component_text_runs(img, roi, threshold=threshold), key=lambda item: item.x1))
        )
        if len(components) < desired_count + 2:
            continue
        heights = [text_run_height(run) for run in components]
        median_h = float(np.median(heights)) if heights else 1.0
        for idx, separator in enumerate(components):
            if idx < 2 or not is_colon_like_run(separator, median_h):
                continue
            if separator.x1 < roi[0] + roi_w * 0.26 or separator.x1 > roi[0] + roi_w * 0.76:
                continue
            left_runs = components[:idx]
            if len(left_runs) < 2:
                continue
            value_runs: list[TextRun] = []
            last_x2 = separator.x2
            for run in components[idx + 1 :]:
                gap = run.x1 - last_x2
                if not value_runs and gap > max(38.0, median_h * 1.45):
                    break
                if value_runs and gap > max(14.0, median_h * 0.65):
                    break
                if is_colon_like_run(run, median_h):
                    break
                if text_run_height(run) < max(8.0, median_h * 0.45):
                    continue
                value_runs.append(run)
                last_x2 = run.x2
            if len(value_runs) >= desired_count:
                return split_cjk_value_cluster(
                    img,
                    tuple(value_runs),
                    desired_count=desired_count,
                    threshold=threshold,
                    median_h=median_h,
                )
            if len(value_runs) == 1:
                value = value_runs[0]
                value_w = value.x2 - value.x1
                if value_w >= median_h * desired_count * 0.70:
                    split = split_run_by_projection(
                        img,
                        value,
                        parts=desired_count,
                        threshold=threshold,
                    )
                    if len(split) >= desired_count:
                        return tuple(sorted(split[:desired_count], key=lambda item: item.x1))
    return ()


def cjk_value_slots_without_label(
    img: Image.Image,
    roi: tuple[int, int, int, int],
    *,
    source_text: str,
    target_text: str,
) -> tuple[TextRun, ...]:
    basis_text = source_text or target_text
    desired_count = len(text_chars(source_text)) or len(text_chars(target_text))
    if desired_count < 2 or not is_mostly_cjk(basis_text):
        return ()

    roi_w = max(1, roi[2] - roi[0])
    for threshold in (150, 135, 120, 165):
        components = merge_overlapping_text_components(
            tuple(
                sorted(
                    dominant_text_line(component_text_runs(img, roi, threshold=threshold), roi),
                    key=lambda item: item.x1,
                )
            )
        )
        if not components:
            continue
        heights = [text_run_height(run) for run in components]
        median_h = float(np.median(heights)) if heights else 1.0
        if any(is_colon_like_run(run, median_h) for run in components):
            continue
        glyphs = tuple(
            run
            for run in components
            if text_run_height(run) >= max(8.0, median_h * 0.45)
        )
        if not glyphs:
            continue
        value = merge_text_runs(glyphs)
        if value is None:
            continue
        value_w = value.x2 - value.x1
        if value_w < median_h * desired_count * 0.62:
            continue
        if value.x1 > roi[0] + roi_w * 0.32:
            continue
        return split_cjk_value_cluster(
            img,
            glyphs,
            desired_count=desired_count,
            threshold=threshold,
            median_h=median_h,
        )
    return ()


def component_slots_for_region(
    img: Image.Image,
    roi: tuple[int, int, int, int],
    *,
    source_text: str,
    target_text: str,
) -> tuple[TextRun, ...]:
    source_count = len(text_chars(source_text))
    target_count = len(text_chars(target_text))
    desired_count = source_count or target_count
    if desired_count <= 0:
        return ()
    non_cjk_slot = non_cjk_value_slot_after_label(
        img,
        roi,
        source_text=source_text,
        target_text=target_text,
    )
    if non_cjk_slot:
        return non_cjk_slot
    cjk_unlabeled_slots = cjk_value_slots_without_label(
        img,
        roi,
        source_text=source_text,
        target_text=target_text,
    )
    if cjk_unlabeled_slots:
        return cjk_unlabeled_slots
    cjk_after_colon_slots = cjk_value_slots_after_colon(
        img,
        roi,
        source_text=source_text,
        target_text=target_text,
    )
    if cjk_after_colon_slots:
        return cjk_after_colon_slots
    component_label_slots = source_slots_after_label_components(
        img,
        roi,
        source_text=source_text,
        target_text=target_text,
    )
    if component_label_slots:
        return component_label_slots
    label_slots = source_slots_after_label_gap(
        img,
        roi,
        source_text=source_text,
        target_text=target_text,
    )
    if label_slots:
        return label_slots
    label_run_slots = source_slots_after_label_run(
        img,
        roi,
        source_text=source_text,
        target_text=target_text,
    )
    if label_run_slots:
        return label_run_slots
    projection_slots = source_slots_from_projection(
        img,
        roi,
        source_text=source_text,
        target_text=target_text,
    )
    if len(projection_slots) >= min(desired_count, 1) and slots_are_compact_source(
        projection_slots,
        roi,
        desired_count=desired_count,
    ):
        return projection_slots

    if source_count:
        return ()

    for threshold in (150, 135, 120, 165):
        line_runs = dominant_text_line(component_text_runs(img, roi, threshold=threshold), roi)
        if not line_runs:
            continue
        selected = tuple(sorted(line_runs, key=lambda item: item.x1))
        selected = glyph_like_runs(
            selected,
            count=desired_count,
            reference_text=source_text or target_text,
        )
        if len(selected) > desired_count:
            selected = compact_run_window(selected, desired_count)
        if len(selected) == 1 and desired_count > 1 and is_mostly_cjk(source_text or target_text):
            selected = split_run_by_projection(
                img,
                selected[0],
                parts=desired_count,
                threshold=threshold,
            )
        if len(selected) >= min(desired_count, 1):
            return tuple(sorted(selected, key=lambda item: item.x1))
    return ()


def synthesize_slots(roi: tuple[int, int, int, int], count: int) -> tuple[TextRun, ...]:
    if count <= 0:
        return ()
    x1, y1, x2, y2 = roi
    width = max(1, x2 - x1)
    pad_x = max(1, round(width * 0.06))
    pad_y = max(1, round((y2 - y1) * 0.08))
    usable_x1 = min(x2 - 1, x1 + pad_x)
    usable_x2 = max(usable_x1 + 1, x2 - pad_x)
    usable_w = max(1, usable_x2 - usable_x1)
    slots: list[TextRun] = []
    for idx in range(count):
        sx1 = int(round(usable_x1 + usable_w * idx / count))
        sx2 = int(round(usable_x1 + usable_w * (idx + 1) / count))
        if idx == count - 1:
            sx2 = usable_x2
        slots.append(
            TextRun(
                x1=sx1,
                y1=min(y2 - 1, y1 + pad_y),
                x2=max(sx1 + 1, sx2),
                y2=max(y1 + pad_y + 1, y2 - pad_y),
                area=0,
            )
        )
    return tuple(slots)


def page_text_components(
    img: Image.Image,
    *,
    threshold: int = 165,
) -> tuple[TextRun, ...]:
    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    mask = (gray < threshold).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    image_w, image_h = img.size
    runs: list[TextRun] = []
    for idx in range(1, num):
        x, y, w, h, area = (int(v) for v in stats[idx])
        if area < 6 or w <= 1 or h <= 2:
            continue
        if h > max(90, int(round(image_h * 0.08))):
            continue
        if w > image_w * 0.72:
            continue
        if area > max(8000, image_w * image_h * 0.01):
            continue
        run = TextRun(x1=x, y1=y, x2=x + w, y2=y + h, area=area)
        if is_horizontal_rule_run(run, (0, 0, image_w, image_h)):
            continue
        runs.append(run)
    return tuple(sorted(runs, key=lambda item: (item.y1, item.x1)))


def group_page_text_lines(runs: tuple[TextRun, ...]) -> tuple[tuple[TextRun, ...], ...]:
    groups: list[list[TextRun]] = []
    for run in sorted(runs, key=lambda item: (text_run_center_y(item), item.x1)):
        placed = False
        run_cy = text_run_center_y(run)
        run_h = text_run_height(run)
        for group in groups:
            group_area = max(1, sum(item.area for item in group))
            group_cy = sum(text_run_center_y(item) * item.area for item in group) / group_area
            group_h = float(np.median([text_run_height(item) for item in group]))
            if abs(run_cy - group_cy) <= max(8.0, min(28.0, max(run_h, group_h) * 0.75)):
                group.append(run)
                placed = True
                break
        if not placed:
            groups.append([run])
    return tuple(tuple(sorted(group, key=lambda item: item.x1)) for group in groups)


def line_candidate_rois(img: Image.Image) -> tuple[tuple[int, int, int, int], ...]:
    image_w, image_h = img.size
    rois: list[tuple[int, int, int, int]] = []
    for group in group_page_text_lines(page_text_components(img)):
        merged = merge_overlapping_text_components(group)
        if not merged:
            continue
        x1 = min(run.x1 for run in merged)
        y1 = min(run.y1 for run in merged)
        x2 = max(run.x2 for run in merged)
        y2 = max(run.y2 for run in merged)
        width = x2 - x1
        height = y2 - y1
        if width < 18 or height < 8 or height > 90:
            continue
        heights = [text_run_height(run) for run in merged]
        median_h = float(np.median(heights)) if heights else height
        pad_x = max(8, int(round(median_h * 0.55)))
        pad_y = max(4, int(round(median_h * 0.28)))
        rois.append(clamp_box((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), (image_w, image_h)))
    deduped: list[tuple[int, int, int, int]] = []
    for roi in sorted(rois, key=lambda item: (item[1], item[0], item[2] - item[0])):
        if roi not in deduped:
            deduped.append(roi)
    return tuple(deduped)


def is_colon_like_run(run: TextRun, median_h: float) -> bool:
    width = max(1, run.x2 - run.x1)
    height = max(1, run.y2 - run.y1)
    return width <= max(7.0, median_h * 0.38) and height <= max(14.0, median_h * 0.85)


def name_anchor_score(
    label: TextRun,
    separator: TextRun,
    *,
    median_h: float,
    image_size: tuple[int, int],
) -> float | None:
    image_w, image_h = image_size
    label_w = max(1, label.x2 - label.x1)
    label_h = text_run_height(label)
    sep_gap = separator.x1 - label.x2
    if label_h < max(9.0, median_h * 0.55):
        return None
    if label_w < max(12.0, median_h * 0.85):
        return None
    label_ratio = label_w / max(1.0, label_h)
    # "姓名" and "姓 名" are usually a compact two-character label. A very wide
    # label near a colon is often another field or table text, not the name label.
    if label_ratio < 0.55 or label_ratio > 3.40:
        return None
    spaced_name_label = label_ratio >= 2.0
    max_label_w = max(110.0 if spaced_name_label else 68.0, median_h * (5.40 if spaced_name_label else 3.80))
    if label_w > max_label_w:
        return None
    max_sep_gap = max(24.0, median_h * (3.20 if spaced_name_label else 0.85))
    if sep_gap < -2 or sep_gap > max_sep_gap:
        return None
    if label.x1 > image_w * 0.32:
        return None
    if text_run_center_y(label) > image_h * 0.30:
        return None
    score = 0.0
    center_y = text_run_center_y(label)
    expected_sep_gap = median_h * (2.45 if spaced_name_label else 0.25)
    score += abs(center_y - image_h * 0.105) * 0.75
    score += max(0.0, label.x1 - image_w * 0.12) * 0.35
    score += abs(sep_gap - expected_sep_gap) * 0.5
    score += abs(label_ratio - 1.55) * 4.0
    return score


def name_label_variants_before_separator(
    components: tuple[TextRun, ...],
    separator_index: int,
    *,
    median_h: float,
) -> tuple[TextRun, ...]:
    if separator_index <= 0:
        return ()
    variants = [components[separator_index - 1]]
    if separator_index >= 2:
        left = components[separator_index - 2]
        right = components[separator_index - 1]
        gap = right.x1 - left.x2
        if -2 <= gap <= max(22.0, median_h * 1.25):
            variants.append(
                TextRun(
                    x1=min(left.x1, right.x1),
                    y1=min(left.y1, right.y1),
                    x2=max(left.x2, right.x2),
                    y2=max(left.y2, right.y2),
                    area=left.area + right.area,
                )
            )
    return tuple(variants)


def coarse_name_field_candidate_rois(
    img: Image.Image,
    *,
    source_text: str,
    target_text: str,
) -> tuple[tuple[float, tuple[int, int, int, int]], ...]:
    source_count = len(text_chars(source_text))
    target_count = len(text_chars(target_text))
    if source_count <= 0:
        return ()
    image_w, image_h = img.size
    components = page_text_components(img)
    if not components:
        return ()
    heights = [text_run_height(run) for run in components if text_run_height(run) >= 8]
    median_h = float(np.median(heights)) if heights else 18.0
    scored: list[tuple[float, tuple[int, int, int, int]]] = []
    for separator in components:
        sep_cy = text_run_center_y(separator)
        if separator.x1 > image_w * 0.28 or sep_cy > image_h * 0.30:
            continue
        if not is_colon_like_run(separator, median_h):
            continue
        row_tol = max(24.0, median_h * 1.10)
        same_row = tuple(
            run
            for run in components
            if abs(text_run_center_y(run) - sep_cy) <= row_tol
        )
        left_runs = tuple(
            run
            for run in same_row
            if run.x2 <= separator.x1
            and run.x1 <= image_w * 0.28
            and not is_colon_like_run(run, median_h)
        )
        if not left_runs:
            continue
        label_parts = tuple(sorted(left_runs, key=lambda item: item.x1)[-2:])
        if len(label_parts) < 2:
            continue
        label = TextRun(
            x1=min(run.x1 for run in label_parts),
            y1=min(run.y1 for run in label_parts),
            x2=max(run.x2 for run in label_parts),
            y2=max(run.y2 for run in label_parts),
            area=sum(run.area for run in label_parts),
        )
        anchor_score = name_anchor_score(
            label,
            separator,
            median_h=median_h,
            image_size=img.size,
        )
        if anchor_score is None:
            continue
        value_runs: list[TextRun] = []
        last_x2 = separator.x2
        max_value_gap = max(42.0, median_h * 4.6)
        for run in sorted(same_row, key=lambda item: item.x1):
            if run.x1 <= separator.x2:
                continue
            gap = run.x1 - last_x2
            if value_runs and gap > max(34.0, median_h * 1.55):
                break
            if not value_runs and gap > max_value_gap:
                break
            if is_colon_like_run(run, median_h):
                continue
            if text_run_height(run) < max(8.0, median_h * 0.45):
                continue
            value_runs.append(run)
            last_x2 = run.x2
            if len(value_runs) >= source_count:
                break
        if not value_runs:
            continue
        if len(value_runs) < source_count and not (
            is_mostly_cjk(source_text)
            and len(value_runs) == 1
            and (value_runs[0].x2 - value_runs[0].x1) >= median_h * 0.70
        ):
            continue
        source_end = value_runs[min(source_count, len(value_runs)) - 1].x2
        desired_right = int(round(separator.x2 + median_h * (max(source_count, target_count) * 1.6 + 1.0)))
        right = max(source_end + int(round(median_h * 0.55)), desired_right)
        roi = clamp_box(
            (
                max(0, label.x1 - max(8, int(round(median_h * 0.40)))),
                max(0, min(label.y1, separator.y1, *(run.y1 for run in value_runs)) - max(5, int(round(median_h * 0.25)))),
                min(image_w, right),
                min(image_h, max(label.y2, separator.y2, *(run.y2 for run in value_runs)) + max(5, int(round(median_h * 0.25)))),
            ),
            img.size,
        )
        if roi[2] - roi[0] < 24 or roi[3] - roi[1] < 12:
            continue
        score = anchor_score + abs(text_run_center_y(label) - image_h * 0.11) * 0.35
        scored.append((score, roi))
    deduped: list[tuple[float, tuple[int, int, int, int]]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for score, roi in sorted(scored, key=lambda item: (item[0], item[1][1], item[1][0])):
        if roi in seen:
            continue
        seen.add(roi)
        deduped.append((score, roi))
    return tuple(deduped)


def name_field_candidate_rois(
    img: Image.Image,
    *,
    source_text: str,
    target_text: str,
) -> tuple[tuple[float, tuple[int, int, int, int]], ...]:
    source_count = len(text_chars(source_text))
    target_count = len(text_chars(target_text))
    if source_count <= 0:
        return ()

    image_w, image_h = img.size
    rois: list[tuple[float, tuple[int, int, int, int]]] = []
    for group in group_page_text_lines(page_text_components(img)):
        components = merge_overlapping_text_components(group)
        if len(components) < 3:
            continue
        heights = [text_run_height(run) for run in components]
        median_h = float(np.median(heights)) if heights else 1.0
        label_min_w = max(7.0, median_h * 0.35)
        label_min_h = max(9.0, median_h * 0.55)
        for idx, separator in enumerate(components):
            if idx == 0 or not is_colon_like_run(separator, median_h):
                continue
            label_options: list[tuple[float, TextRun]] = []
            for label in name_label_variants_before_separator(components, idx, median_h=median_h):
                anchor_score = name_anchor_score(
                    label,
                    separator,
                    median_h=median_h,
                    image_size=img.size,
                )
                if anchor_score is None:
                    continue
                if label.x2 > separator.x1 or separator.x1 - label.x2 > max(22.0, median_h * 1.25):
                    continue
                if label.x2 - label.x1 < label_min_w or text_run_height(label) < label_min_h:
                    continue
                label_options.append((anchor_score, label))
            if not label_options:
                continue
            anchor_score, label = min(label_options, key=lambda item: item[0])

            value_runs: list[TextRun] = []
            next_field_x = image_w
            last_x2 = separator.x2
            for run in components[idx + 1 :]:
                gap = run.x1 - last_x2
                if value_runs and gap > max(36.0, median_h * 1.55):
                    next_field_x = run.x1
                    break
                if not value_runs and gap > max(34.0, median_h * 1.45):
                    break
                if is_colon_like_run(run, median_h) and value_runs:
                    next_field_x = run.x1
                    break
                if text_run_height(run) >= label_min_h * 0.60 and run.x2 - run.x1 >= max(3.0, median_h * 0.18):
                    value_runs.append(run)
                    last_x2 = run.x2
                if len(value_runs) >= source_count:
                    break

            if len(value_runs) < source_count and not (
                is_mostly_cjk(source_text)
                and len(value_runs) == 1
                and (value_runs[0].x2 - value_runs[0].x1) >= median_h * 0.70
            ):
                continue
            source_end = value_runs[min(source_count, len(value_runs)) - 1].x2
            desired_right = int(round(separator.x2 + median_h * (max(source_count, target_count) * 1.55 + 1.0)))
            right = min(next_field_x - 3, max(source_end + int(round(median_h * 0.7)), desired_right))
            if right <= separator.x2:
                continue
            left = max(0, label.x1 - max(8, int(round(median_h * 0.45))))
            top = max(0, min(run.y1 for run in [label, separator, *value_runs]) - max(5, int(round(median_h * 0.25))))
            bottom = min(image_h, max(run.y2 for run in [label, separator, *value_runs]) + max(5, int(round(median_h * 0.25))))
            roi = clamp_box((left, top, right, bottom), (image_w, image_h))
            if roi[2] - roi[0] >= 24 and roi[3] - roi[1] >= 12:
                value_start_gap = value_runs[0].x1 - separator.x2
                value_span = value_runs[min(source_count, len(value_runs)) - 1].x2 - value_runs[0].x1
                score = (
                    anchor_score
                    + abs(value_start_gap - median_h * 0.55) * 0.35
                    + max(0.0, value_span - median_h * source_count * 2.0) * 0.25
                )
                rois.append((score, roi))
    deduped: list[tuple[float, tuple[int, int, int, int]]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for score, roi in sorted(rois, key=lambda item: (item[0], item[1][1], item[1][0])):
        if roi in seen:
            continue
        seen.add(roi)
        deduped.append((score, roi))
    for score, roi in coarse_name_field_candidate_rois(img, source_text=source_text, target_text=target_text):
        if roi in seen:
            continue
        seen.add(roi)
        deduped.append((score, roi))
    deduped.sort(key=lambda item: (item[0], item[1][1], item[1][0]))
    return tuple(deduped)


def plain_rois_as_scored(
    rois: tuple[tuple[int, int, int, int], ...],
) -> tuple[tuple[float, tuple[int, int, int, int]], ...]:
    return tuple((0.0, roi) for roi in rois)


def inferred_source_placeholder_for_field(
    field_key: str | None,
    target_text: str,
    value_span_width: int,
    median_h: float,
) -> str:
    compact_target = re.sub(r"\s+", "", str(target_text or ""))
    if field_key == "age":
        return "00岁" if "岁" in compact_target else "00"
    if field_key in {"receive_time", "date"}:
        looks_datetime_width = value_span_width >= max(140, int(round(median_h * 7.8)))
        if field_key == "receive_time" or looks_datetime_width:
            return "0000-00-00 00:00:00"
        return "0000-00-00"
    return "0" * max(1, len(text_chars(target_text)))


def field_value_candidate_rois(
    img: Image.Image,
    *,
    field_key: str | None,
    target_text: str,
) -> tuple[tuple[float, tuple[int, int, int, int], str], ...]:
    if not field_key:
        return ()

    image_w, image_h = img.size
    scored: list[tuple[float, tuple[int, int, int, int], str]] = []
    for group in group_page_text_lines(page_text_components(img)):
        components = merge_overlapping_text_components(group)
        if len(components) < 3:
            continue
        heights = [text_run_height(run) for run in components]
        median_h = float(np.median(heights)) if heights else 1.0
        min_label_h = max(8.0, median_h * 0.48)
        max_first_gap = max(36.0, median_h * 1.75)
        max_inner_gap = max(28.0, median_h * 1.35)
        for idx, separator in enumerate(components):
            if idx == 0 or not is_colon_like_run(separator, median_h):
                continue
            label = components[idx - 1]
            if text_run_height(label) < min_label_h:
                continue
            if separator.x1 - label.x2 > max(22.0, median_h * 0.95):
                continue

            value_runs: list[TextRun] = []
            next_field_x = image_w
            last_x2 = separator.x2
            for run in components[idx + 1 :]:
                gap = run.x1 - last_x2
                if value_runs:
                    current_centers = [(item.y1 + item.y2) / 2.0 for item in value_runs]
                    current_center_y = float(np.median(current_centers))
                    run_center_y = (run.y1 + run.y2) / 2.0
                    if abs(run_center_y - current_center_y) > max(8.0, median_h * 0.50):
                        next_field_x = run.x1
                        break
                if value_runs and (
                    gap > max_inner_gap
                    or (is_colon_like_run(run, median_h) and run.x1 - value_runs[-1].x2 > median_h * 0.45)
                ):
                    next_field_x = run.x1
                    break
                if not value_runs and gap > max_first_gap:
                    break
                if text_run_height(run) < max(5.0, median_h * 0.28) and (run.x2 - run.x1) < median_h * 0.55:
                    continue
                value_runs.append(run)
                last_x2 = run.x2

            if not value_runs:
                continue
            value_x1 = min(run.x1 for run in value_runs)
            value_y1 = min(run.y1 for run in value_runs)
            value_x2 = max(run.x2 for run in value_runs)
            value_y2 = max(run.y2 for run in value_runs)
            value_w = value_x2 - value_x1
            value_h = value_y2 - value_y1
            if value_w < max(12.0, median_h * 0.8) or value_h < max(6.0, median_h * 0.35):
                continue

            source_placeholder = inferred_source_placeholder_for_field(
                field_key,
                target_text,
                value_w,
                median_h,
            )
            if field_key in {"receive_time", "date"}:
                min_date_width = max(70.0, median_h * 3.8)
                if value_w < min_date_width:
                    continue
            if field_key == "age" and value_w > max(95.0, median_h * 4.2):
                continue

            pad_x = max(8, int(round(median_h * 0.45)))
            pad_y = max(4, int(round(median_h * 0.24)))
            right = min(image_w, value_x2 + max(4, int(round(median_h * 0.24))))
            if field_key == "receive_time" and source_placeholder.startswith("0000-00-00 00"):
                estimated_full_width = int(round(value_w * 1.58))
                right = max(right, min(image_w, value_x1 + estimated_full_width))
            if next_field_x < right:
                right = max(value_x2, next_field_x - 2)
                if field_key == "receive_time" and source_placeholder.startswith("0000-00-00 00"):
                    right = max(right, min(image_w, value_x1 + estimated_full_width))
            roi = clamp_box(
                (
                    max(0, value_x1 - max(3, int(round(median_h * 0.16)))),
                    max(0, value_y1 - pad_y),
                    min(image_w, right),
                    min(image_h, value_y2 + pad_y),
                ),
                img.size,
            )
            if roi[2] - roi[0] < 24 or roi[3] - roi[1] < 12:
                continue

            score = 0.0
            line_y = (roi[1] + roi[3]) / 2.0
            if field_key == "receive_time":
                score -= min(80.0, max(0.0, separator.x1 - image_w * 0.50) * 0.18)
                score += abs(line_y - image_h * 0.12) * 0.55
                if line_y > image_h * 0.25:
                    score += 100.0
                if line_y > image_h * 0.40:
                    score += 120.0
                score -= min(value_w, median_h * 11.5) * 0.22
                if separator.x1 < image_w * 0.50:
                    score += 80.0
            elif field_key == "date":
                score += abs(line_y - image_h * 0.16) * 0.05
                score -= min(value_w, median_h * 8.0) * 0.12
            elif field_key == "age":
                score += abs(separator.x1 - image_w * 0.12) * 0.08
                score += value_w * 0.18
            score += (roi[2] - roi[0]) * 0.02
            scored.append((score, roi, source_placeholder))

    deduped: list[tuple[float, tuple[int, int, int, int], str]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for item in sorted(scored, key=lambda value: value[0]):
        if item[1] in seen:
            continue
        seen.add(item[1])
        deduped.append(item)
    return tuple(deduped[:12])


def auto_region_score(
    plan: RenderPlan,
    *,
    source_text: str,
    target_text: str,
    field_key: str | None,
    image_size: tuple[int, int],
    source_shape_score: float | None = None,
) -> float | None:
    source_count = len(text_chars(source_text))
    target_count = len(text_chars(target_text))
    if source_count <= 0 or not plan.slot_boxes:
        return None
    slots = tuple(sorted(plan.slot_boxes, key=lambda item: item.x1))
    if len(slots) < min(source_count, 1):
        return None
    if not slots_are_compact_source(slots, plan.search_roi, desired_count=source_count):
        return None
    slot_span = max(1, slots[-1].x2 - slots[0].x1)
    target_w = max(1, plan.target_roi[2] - plan.target_roi[0])
    search_w = max(1, plan.search_roi[2] - plan.search_roi[0])
    if not is_mostly_cjk(source_text or target_text) and len(slots) == 1:
        max_target_w = search_w
    else:
        max_target_w = min(search_w * 0.72, max(slot_span * 2.5, slot_span * target_count / max(1, source_count) + 40))
    if target_w > max_target_w:
        return None

    slot_area = sum(slot.area for slot in slots)
    left_context = any(
        box[2] <= slots[0].x1 and protected_box_overlaps_row(box, plan.target_roi)
        for box in plan.protected_boxes
    )
    right_context = any(
        box[0] >= slots[-1].x2 and protected_box_overlaps_row(box, plan.target_roi)
        for box in plan.protected_boxes
    )
    score = target_w * 0.35 + search_w * 0.08 - slot_area * 0.04
    if left_context:
        score -= 18.0
    if right_context:
        score -= 8.0
    if field_key == "name" and left_context:
        score -= 12.0
    if field_key == "department":
        image_w, image_h = image_size
        target_center_x = (plan.target_roi[0] + plan.target_roi[2]) / 2.0
        target_center_y = (plan.target_roi[1] + plan.target_roi[3]) / 2.0
        score += abs(target_center_y - image_h * 0.11) * 1.0
        score += abs(target_center_x - image_w * 0.36) * 0.20
        if target_center_y > image_h * 0.22:
            score += 260.0
        if target_center_x > image_w * 0.62:
            score += 180.0
        if target_center_x < image_w * 0.18:
            score += 80.0
    if source_shape_score is not None:
        score += source_shape_score * 1.7
    return score


def source_text_shape_score(
    img: Image.Image,
    plan: RenderPlan,
    font_candidates: list[tuple[str, str]],
) -> float | None:
    if not plan.source_reference_box or not plan.source_text or not plan.slot_boxes:
        return None
    base_size = initial_font_size(plan)
    best_score: float | None = None
    for font_name, font_path in font_candidates[:6]:
        for font_size in range(max(8, base_size - 2), base_size + 3):
            score = font_style_score(
                img,
                reference_box=plan.source_reference_box,
                reference_text=plan.source_text,
                reference_kind="auto_source_match",
                slot_boxes=plan.slot_boxes,
                font_name=font_name,
                font_path=font_path,
                font_size=font_size,
                opacity=0.86,
                blur=0.35,
            )
            if score and (best_score is None or float(score["score"]) < best_score):
                best_score = float(score["score"])
    return best_score


def auto_select_regions_for_instruction(
    img: Image.Image,
    *,
    instruction: str,
    source_text: str,
    target_text: str,
) -> list[dict[str, Any]]:
    field_key = infer_instruction_field(instruction)
    font_candidates = find_font_candidates(font_source="recommended")
    font_filter_text = source_text or target_text
    font_candidates, _rejected = filter_fonts_by_required_text(font_candidates, font_filter_text)

    def score_candidate_rois(
        candidate_rois: tuple[tuple[float, tuple[int, int, int, int]], ...],
        *,
        source_override: str | None = None,
    ) -> list[tuple[float, tuple[int, int, int, int], RenderPlan, str]]:
        scored: list[tuple[float, tuple[int, int, int, int], RenderPlan, str]] = []
        effective_source = source_override if source_override is not None else source_text
        for roi_bias, roi in candidate_rois:
            try:
                plan = build_region_plan(
                    img,
                    roi,
                    source_text=effective_source,
                    target_text=target_text,
                )
            except ValueError:
                continue
            shape_score = (
                source_text_shape_score(img, plan, font_candidates)
                if text_chars(effective_source) and is_mostly_cjk(effective_source)
                else None
            )
            effective_shape_limit = (
                220.0
                if field_key == "name" and is_mostly_cjk(effective_source)
                else 70.0
                if is_mostly_cjk(effective_source)
                else 105.0
            )
            if shape_score is not None and shape_score > effective_shape_limit:
                continue
            score = auto_region_score(
                plan,
                source_text=effective_source,
                target_text=target_text,
                field_key=field_key,
                image_size=img.size,
                source_shape_score=shape_score,
            )
            if score is None:
                continue
            roi_bias_weight = 10.0 if field_key == "name" else 2.5
            scored.append((score + roi_bias * roi_bias_weight, roi, plan, effective_source))
        return scored

    candidates: list[tuple[float, tuple[int, int, int, int], RenderPlan, str]] = []
    if text_chars(source_text):
        if field_key == "name":
            candidates = score_candidate_rois(
                name_field_candidate_rois(img, source_text=source_text, target_text=target_text)
            )
        else:
            candidates = score_candidate_rois(plain_rois_as_scored(line_candidate_rois(img)))
    elif field_key:
        for field_score, roi, inferred_source in field_value_candidate_rois(
            img,
            field_key=field_key,
            target_text=target_text,
        ):
            for auto_score, auto_roi, plan, effective_source in score_candidate_rois(
                ((field_score, roi),),
                source_override=inferred_source,
            ):
                candidates.append((auto_score, auto_roi, plan, effective_source))
    else:
        raise ValueError(
            "自动选择 ROI 需要明确旧文字，或写明字段名称，例如：姓名旧值调整为新值、接收时间修改为新值。"
        )

    if not candidates:
        field_label = f"{field_key or '指定字段'}" if field_key else "指定字段"
        source_label = source_text or "字段当前值"
        raise ValueError(
            f"无法自动定位{field_label}中的旧文字：{source_label}。"
            "请手动画一个只包含旧文字和少量右侧空白的矩形，或把指令写成“字段 旧文字调整为新文字”。"
        )

    selected_score, search_roi, selected_plan, selected_source_text = min(candidates, key=lambda item: item[0])
    roi = auto_edit_roi_from_plan(
        img,
        search_roi,
        selected_plan,
        source_text=selected_source_text,
        target_text=target_text,
        field_key=field_key,
    )
    x1, y1, x2, y2 = roi
    return [
        {
            "id": f"auto_{field_key or 'field'}",
            "rect": {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1},
            "auto": True,
            "sourceText": selected_source_text,
            "targetText": target_text,
            "_autoScore": round(float(selected_score), 3),
            "_autoTargetRoi": list(selected_plan.target_roi),
            "_autoSearchRoi": list(search_roi),
        }
    ]


def slots_for_region(
    img: Image.Image,
    roi: tuple[int, int, int, int],
    source_text: str,
    target_text: str,
    *,
    threshold: int = 165,
) -> tuple[TextRun, ...]:
    component_slots = component_slots_for_region(
        img,
        roi,
        source_text=source_text,
        target_text=target_text,
    )
    if component_slots:
        return component_slots

    basis_text = source_text or target_text
    basis_chars = text_chars(basis_text)
    if not basis_chars:
        return ()
    if text_chars(source_text):
        raise ValueError(
            "无法在选区内可靠定位旧文字："
            f"{source_text}。请重新框选只包含旧文字及少量右侧空白的区域，"
            "不要包含左侧标签或右侧其它字段。"
        )
    runs = dark_runs(img, roi, threshold=threshold)
    if is_mostly_cjk(basis_text) and len(basis_chars) > 1 and len(runs) == 1:
        runs = list(
            split_run_by_projection(
                img,
                runs[0],
                parts=len(basis_chars),
                threshold=threshold,
            )
        )
    if len(runs) >= len(basis_chars):
        return tuple(sorted(runs, key=lambda item: item.x1)[-len(basis_chars):])
    if runs:
        return tuple(runs)
    return synthesize_slots(roi, len(basis_chars))


def slot_quality_report(
    img: Image.Image,
    roi: tuple[int, int, int, int],
    slots: tuple[TextRun, ...],
    *,
    source_text: str,
    target_text: str,
    protected_boxes: tuple[tuple[int, int, int, int], ...],
    threshold: int = 165,
) -> dict[str, Any]:
    source_chars = text_chars(source_text)
    target_chars = text_chars(target_text)
    source_count = len(source_chars)
    target_count = len(target_chars)
    expected_count = len(source_chars) or len(target_chars)
    issues: list[dict[str, Any]] = []
    if expected_count and len(slots) < expected_count:
        issues.append(
            {
                "type": "slot_count_too_low",
                "expected": expected_count,
                "actual": len(slots),
            }
        )
    if source_chars and len(slots) > len(source_chars) + 1:
        issues.append(
            {
                "type": "slot_count_too_high",
                "expected": len(source_chars),
                "actual": len(slots),
            }
        )

    def union_boxes(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
        if not boxes:
            return None
        return (
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        )

    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]
    per_slot: list[dict[str, Any]] = []
    ordered_slots = tuple(sorted(slots, key=lambda item: item.x1))
    for idx, slot in enumerate(ordered_slots):
        x1, y1, x2, y2 = clamp_box((slot.x1, slot.y1, slot.x2, slot.y2), (w, h))
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        crop = gray[y1:y2, x1:x2] if width and height else np.zeros((0, 0), dtype=np.uint8)
        dark_pixels = int(np.count_nonzero(crop < threshold))
        slot_issues: list[dict[str, Any]] = []
        if width < 3 or height < 3:
            slot_issues.append({"type": "slot_too_small", "width": width, "height": height})
        min_dark_pixels = max(4, int(round(width * height * 0.025)))
        if dark_pixels < min_dark_pixels:
            slot_issues.append(
                {
                    "type": "slot_has_too_few_dark_pixels",
                    "dark_pixels": dark_pixels,
                    "minimum": min_dark_pixels,
                }
            )
        overflow_pad = max(2, int(round(height * 0.20)))
        bottom_y2 = min(roi[3], y2 + overflow_pad)
        if bottom_y2 > y2 and width > 0:
            below = gray[y2:bottom_y2, x1:x2] < threshold
            below_pixels = int(np.count_nonzero(below))
            if below_pixels >= max(3, int(round(width * overflow_pad * 0.025))):
                slot_issues.append(
                    {
                        "type": "slot_bottom_overflow",
                        "dark_pixels_below": below_pixels,
                        "checked_y": [y2, bottom_y2],
                    }
                )
        slot_box = (x1, y1, x2, y2)
        protected_conflicts = [
            list(box)
            for box in protected_boxes
            if box_overlap_area(slot_box, box) > 0
        ]
        if protected_conflicts:
            slot_issues.append(
                {
                    "type": "slot_overlaps_protected_text",
                    "protected_boxes": protected_conflicts,
                }
            )
        issues.extend({"index": idx, **issue} for issue in slot_issues)
        per_slot.append(
            {
                "index": idx,
                "box": [x1, y1, x2, y2],
                "width": width,
                "height": height,
                "dark_pixels": dark_pixels,
                "issues": slot_issues,
            }
        )

    source_slot_limit = source_count if source_count else len(ordered_slots)
    source_slot_boxes = [
        text_run_box(slot)
        for slot in ordered_slots[: min(source_slot_limit, len(ordered_slots))]
    ]
    source_span_box = union_boxes(source_slot_boxes)
    if source_count and target_count > source_count:
        length_change = "longer"
    elif source_count and target_count < source_count:
        length_change = "shorter"
    elif source_count and target_count:
        length_change = "same"
    else:
        length_change = "unknown"

    extra_source_slots = (
        ordered_slots[target_count: min(source_count, len(ordered_slots))]
        if length_change == "shorter"
        else ()
    )
    extra_source_boxes = [text_run_box(slot) for slot in extra_source_slots]
    cleanup_span = union_boxes(extra_source_boxes)
    if length_change == "shorter" and source_count and target_count and len(extra_source_slots) < source_count - target_count:
        issues.append(
            {
                "type": "missing_extra_source_slots_for_cleanup",
                "expected_extra_slots": source_count - target_count,
                "actual_extra_slots": len(extra_source_slots),
            }
        )

    right_boundary: dict[str, Any] = {
        "enabled": bool(length_change == "longer" and source_span_box),
    }
    if length_change == "longer" and source_span_box is not None:
        row_box = source_span_box
        margin = 3
        right_protected_boxes = [
            box
            for box in protected_boxes
            if protected_box_overlaps_row(box, row_box) and box[0] >= row_box[2]
        ]
        right_limit = min([roi[2], *(box[0] - margin for box in right_protected_boxes)])
        slot_widths = [max(1, slot.x2 - slot.x1) for slot in ordered_slots]
        slot_gaps = [
            max(0, ordered_slots[idx + 1].x1 - ordered_slots[idx].x2)
            for idx in range(len(ordered_slots) - 1)
        ]
        median_width = float(np.median(slot_widths)) if slot_widths else max(1.0, (row_box[2] - row_box[0]) / source_count)
        median_gap = float(np.median(slot_gaps)) if slot_gaps else max(2.0, median_width * 0.14)
        estimated_extra_width = max(0.0, (target_count - source_count) * (median_width + median_gap))
        available_right_px = max(0, right_limit - row_box[2])
        right_boundary = {
            "enabled": True,
            "source_span_box": list(row_box),
            "right_limit": int(right_limit),
            "available_right_px": int(available_right_px),
            "estimated_extra_width": round(float(estimated_extra_width), 3),
            "limited_by_protected_text": bool(right_limit < roi[2]),
            "protected_right_boxes": [list(box) for box in right_protected_boxes],
        }

    return {
        "pass": not issues,
        "expected_count": expected_count,
        "actual_count": len(slots),
        "source_count": source_count,
        "target_count": target_count,
        "length_change": length_change,
        "source_text": source_text,
        "target_text": target_text,
        "roi": list(roi),
        "source_span_box": list(source_span_box) if source_span_box else None,
        "length_change_report": {
            "length_change": length_change,
            "source_count": source_count,
            "target_count": target_count,
            "extra_source_slots_for_cleanup": [list(box) for box in extra_source_boxes],
            "extra_source_cleanup_span": list(cleanup_span) if cleanup_span else None,
            "right_boundary": right_boundary,
        },
        "per_slot": per_slot,
        "issues": issues,
    }


def choose_placement_strategy(
    *,
    source_text: str,
    target_text: str,
    slots: tuple[TextRun, ...],
    slot_report: dict[str, Any],
    draw_mode: str,
) -> tuple[str, str]:
    source_count = len(text_chars(source_text))
    target_count = len(text_chars(target_text))
    if not source_count:
        return "manual_fallback", "source_text_missing"
    if draw_mode == "center" or target_count > source_count:
        return "left_anchor_span", "target_text_longer_than_source"
    if target_count < source_count:
        return "left_anchor_span", "target_text_shorter_than_source"
    if not is_mostly_cjk(source_text or target_text):
        return "baseline_numeric", "non_cjk_value_uses_baseline_priority"
    if not slot_report.get("pass"):
        return "top_left_anchor", "slot_quality_failed_keeps_original_anchor_for_rejection"
    if source_text != target_text:
        return "center_primary", "same_length_cjk_changed_chars_use_slot_center"
    heights = [max(1, slot.y2 - slot.y1) for slot in slots]
    widths = [max(1, slot.x2 - slot.x1) for slot in slots]
    if heights and max(heights) - min(heights) > max(2, float(np.median(heights)) * 0.18):
        return "center_primary", "same_length_cjk_slot_height_variation"
    if widths and max(widths) - min(widths) > max(3, float(np.median(widths)) * 0.22):
        return "center_primary", "same_length_cjk_slot_width_variation"
    return "top_left_anchor", "same_length_cjk_compact_slots"


def build_region_plan(
    img: Image.Image,
    roi: tuple[int, int, int, int],
    *,
    source_text: str,
    target_text: str,
    threshold: int = 165,
) -> RenderPlan:
    slots = slots_for_region(
        img,
        roi,
        source_text=source_text,
        target_text=target_text,
        threshold=threshold,
    )
    target_roi = slots_roi(slots, img.size) or roi
    target_roi = clamp_box_to_container(target_roi, roi)
    if slots:
        slot_widths = [max(1, slot.x2 - slot.x1) for slot in slots]
        slot_heights = [max(1, slot.y2 - slot.y1) for slot in slots]
        if is_mostly_cjk(source_text or target_text):
            pad_x = max(2, int(round(float(np.median(slot_widths)) * 0.07)))
            pad_y = max(6, int(round(float(np.median(slot_heights)) * 0.30)))
        else:
            pad_x = max(3, int(round(float(np.median(slot_widths)) * 0.035)))
            pad_y = max(2, int(round(float(np.median(slot_heights)) * 0.12)))
        target_roi = clamp_box_to_container(
            (
                min(slot.x1 for slot in slots) - pad_x,
                min(slot.y1 for slot in slots) - pad_y,
                max(slot.x2 for slot in slots) + pad_x,
                max(slot.y2 for slot in slots) + pad_y,
            ),
            roi,
        )
    source_reference_box = target_roi
    protected_boxes = protected_boxes_for_region(img, roi, slots)
    slot_report = slot_quality_report(
        img,
        roi,
        slots,
        source_text=source_text,
        target_text=target_text,
        protected_boxes=protected_boxes,
        threshold=threshold,
    )
    source_count = len(text_chars(source_text))
    target_count = len(text_chars(target_text))
    if target_count and source_count and target_count > source_count:
        target_roi = expand_roi_for_longer_replacement(
            target_roi,
            roi,
            slots=slots,
            protected_boxes=protected_boxes,
            source_text=source_text,
            target_text=target_text,
            image_size=img.size,
        )
        target_roi = clamp_box_to_container(target_roi, roi)
        if slots and target_roi[1] <= roi[1] + 1:
            body_bounds = source_text_body_y_bounds(img, roi, slots)
            adjusted_y1 = max(
                roi[1],
                (body_bounds[0] - 1) if body_bounds else min(slot.y1 for slot in slots),
            )
            if adjusted_y1 < target_roi[3]:
                target_roi = (target_roi[0], adjusted_y1, target_roi[2], target_roi[3])
    elif target_count and source_count and target_count < source_count:
        target_roi = expand_roi_for_shorter_replacement(
            target_roi,
            roi,
            slots=slots,
            source_text=source_text,
            target_text=target_text,
            image_size=img.size,
        )
        target_roi = clamp_box_to_container(target_roi, roi)
        source_reference_box = target_roi
    slot_report = {
        **slot_report,
        "target_roi_after_length_policy": list(target_roi),
        "source_reference_box_after_length_policy": list(source_reference_box),
        "protected_boxes": [list(box) for box in protected_boxes],
    }
    length_report = dict(slot_report.get("length_change_report") or {})
    if length_report:
        length_report["target_roi_after_length_policy"] = list(target_roi)
        length_report["source_reference_box_after_length_policy"] = list(source_reference_box)
        slot_report["length_change_report"] = length_report
    draw_mode = "auto"
    if target_count and (source_count and target_count > source_count):
        draw_mode = "center"
    elif target_count and (source_count and target_count < source_count):
        draw_mode = "line_chars" if is_mostly_cjk(target_text) else "auto"
    elif target_count and len(slots) < target_count:
        draw_mode = "center"
    text_angle_degrees = estimate_text_angle_degrees(slots)
    if draw_mode == "line_chars":
        text_angle_degrees = max(-1.5, min(1.5, text_angle_degrees))
    placement_strategy, placement_reason = choose_placement_strategy(
        source_text=source_text,
        target_text=target_text,
        slots=slots,
        slot_report=slot_report,
        draw_mode=draw_mode,
    )
    return RenderPlan(
        target_text=target_text,
        source_text=source_text,
        search_roi=roi,
        target_roi=target_roi,
        slot_boxes=slots,
        protected_boxes=protected_boxes,
        source_reference_box=source_reference_box,
        style_reference_box=None,
        style_reference_text=None,
        draw_mode=draw_mode,
        text_angle_degrees=text_angle_degrees,
        placement_strategy=placement_strategy,
        placement_strategy_reason=placement_reason,
        slot_quality_report=slot_report,
    )


def initial_font_size(plan: RenderPlan) -> int:
    if plan.slot_boxes:
        heights = [box.y2 - box.y1 for box in plan.slot_boxes if box.y2 > box.y1]
        if heights:
            return max(8, min(96, int(round(max(heights) + 1))))
    _, y1, _, y2 = plan.target_roi
    return max(8, min(96, int(round((y2 - y1) * 0.85))))


def max_font_size_for_plan(plan: RenderPlan) -> int:
    base_size = initial_font_size(plan)
    source_count = len(text_chars(plan.source_text or ""))
    target_count = len(text_chars(plan.target_text))
    if plan.draw_mode == "center" and source_count and target_count > source_count:
        x1, _, x2, _ = plan.target_roi
        width_cap = int(round((x2 - x1) / max(1, target_count) * 0.96))
        return max(8, min(base_size + 2, width_cap))
    return max(18, min(72, base_size + 8))
