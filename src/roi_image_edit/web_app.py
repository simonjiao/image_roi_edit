from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
import re
import threading
import time
import uuid
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from roi_image_edit.iterative_pipeline import (
    STRICT_ACCEPTANCE_APPENDIX,
    CandidateParams,
    RenderPlan,
    TextRun,
    VisionClient,
    apply_suggested_patch,
    build_font_style_reference,
    build_trailing_value_cleanup_mask,
    char_alignment_gate,
    char_gray_band_metrics,
    char_pose_metrics,
    clamp_box,
    dark_runs,
    dedupe_params,
    default_char_offsets,
    draw_replacement_layer,
    extra_source_slot_cleanup_issues,
    extra_source_slot_cleanup_metrics,
    find_best_candidate_from_model,
    filter_fonts_by_required_text,
    find_font_candidates,
    font_style_gate,
    font_style_score,
    generate_candidates,
    gray_band_counts,
    hard_check,
    is_mostly_cjk,
    local_score,
    make_contact_sheet,
    mutate_params,
    params_label,
    rank_fonts_by_style_reference,
    render_candidate,
    report_strict_pass,
    split_run_by_projection,
    strict_gate_issues,
    strict_visual_metrics,
    write_json,
)
from roi_image_edit.prompt_assets import load_prompt, missing_prompt_names


ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT / "web"
OUTPUT_DIR = ROOT / "output" / "web"
ENV_PATH = ROOT / ".env"
ProgressCallback = Callable[[str, dict[str, Any]], None]
MAX_WEB_JOB_EVENTS = 800
WEB_JOB_LOCK = threading.Lock()
WEB_JOBS: dict[str, dict[str, Any]] = {}


PATCH_FAMILY_KEYS = {
    "text_shape": {
        "font_size_delta",
        "text_dx_delta",
        "text_dy_delta",
        "char_offsets_delta",
    },
    "stroke_body_shape": {
        "stroke_opacity_delta",
        "ink_gain_delta",
        "alpha_contrast_delta",
        "core_ink_gain_delta",
        "core_darken_strength_delta",
        "core_darken_threshold_delta",
        "core_darken_target_gray_delta",
    },
    "ink_gray_balance": {
        "opacity_delta",
        "stroke_opacity_delta",
        "ink_gain_delta",
        "alpha_contrast_delta",
        "core_ink_gain_delta",
        "core_darken_strength_delta",
        "core_darken_threshold_delta",
        "core_darken_target_gray_delta",
    },
    "photo_texture": {
        "blur_delta",
        "photo_warp_delta",
        "edge_breakup_delta",
        "photo_noise_delta",
        "jpeg_quality_delta",
    },
    "background_cleanup": {
        "mask_threshold_delta",
        "mask_dilate_iterations_delta",
        "inpaint_radius_delta",
    },
}

STAGE_PATCH_POLICY = {
    "hard_boundary": {
        "allowed_families": [],
        "forbidden_families": [
            "text_shape",
            "stroke_body_shape",
            "ink_gray_balance",
            "photo_texture",
            "background_cleanup",
        ],
        "reason": "Hard boundary failures must stop before visual or revision tuning.",
    },
    "text_shape": {
        "allowed_families": ["text_shape", "stroke_body_shape"],
        "forbidden_families": ["background_cleanup"],
        "secondary_only_families": ["photo_texture"],
        "reason": "Shape must be solved by font, slot, baseline, pose, and stroke body before texture becomes dominant.",
    },
    "ink_gray_balance": {
        "allowed_families": ["ink_gray_balance"],
        "forbidden_families": ["text_shape", "background_cleanup"],
        "secondary_only_families": ["photo_texture"],
        "reason": "Ink balance must solve true-black core, mid-gray body, and gray edge before photo texture dominates.",
    },
    "photo_texture": {
        "allowed_families": ["photo_texture"],
        "forbidden_families": ["text_shape", "stroke_body_shape", "ink_gray_balance", "background_cleanup"],
        "reason": "Photo texture is only allowed after shape and ink stages pass.",
    },
    "background_cleanup": {
        "allowed_families": ["background_cleanup", "photo_texture"],
        "forbidden_families": ["text_shape", "stroke_body_shape", "ink_gray_balance"],
        "reason": "Background cleanup must repair mask, inpaint, texture, ghosting, and seams without using new text to hide residue.",
    },
    "none": {
        "allowed_families": [
            "text_shape",
            "stroke_body_shape",
            "ink_gray_balance",
            "photo_texture",
            "background_cleanup",
        ],
        "forbidden_families": [],
        "reason": "No local blocking stage remains.",
    },
}


def request_audit_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "maxCandidates": payload.get("maxCandidates"),
        "images": [
            {
                "id": str(item.get("id") or ""),
                "filename": str(item.get("filename") or ""),
                "instruction": str(item.get("instruction") or ""),
                "regions": [
                    {
                        "id": str(region.get("id") or ""),
                        "rect": region.get("rect") or {},
                    }
                    for region in item.get("regions", [])
                ],
            }
            for item in payload.get("images", [])
        ],
    }


def result_audit_payload(response: dict[str, Any]) -> dict[str, Any]:
    images: list[dict[str, Any]] = []
    for item in response.get("images", []):
        image_record = {
            key: value
            for key, value in item.items()
            if key not in {"sourceDataUrl", "resultDataUrl", "candidates"}
        }
        image_record["candidates"] = [
            {key: value for key, value in candidate.items() if key != "dataUrl"}
            for candidate in item.get("candidates", [])
        ]
        images.append(image_record)
    return {"ok": response.get("ok"), "runDir": response.get("runDir"), "images": images}


def load_web_prompts() -> tuple[str, str, str]:
    required = ("master_prompt.txt", "candidate_rank_prompt.txt", "final_acceptance_prompt.txt")
    missing = missing_prompt_names(required)
    if missing:
        raise FileNotFoundError(
            f"package vision prompts are missing: {', '.join(missing)}"
        )
    return (
        load_prompt("master_prompt.txt"),
        load_prompt("candidate_rank_prompt.txt"),
        load_prompt("final_acceptance_prompt.txt"),
    )


def web_prompt_context(plan: RenderPlan) -> str:
    target_chars = [ch for ch in plan.target_text if not ch.isspace()]
    return (
        "\n\nWeb 动态任务补充：\n"
        f"- 旧文字 source_text: {plan.source_text or ''}\n"
        f"- 新文字 target_text: {plan.target_text}\n"
        f"- 本次真实目标字符序列: {target_chars}\n"
        "- prompt 模板不得假设固定姓名；本次必须按 source_text 和 target_text 的真实字符逐个判断。\n"
        f"- 用户画出的 search_roi: {list(plan.search_roi)}\n"
        f"- 本地脚本选择的 target_roi: {list(plan.target_roi)}\n"
        f"- 本地估计的局部文字倾角 text_angle_degrees: {round(float(plan.text_angle_degrees), 3)}\n"
        f"- 必须保持不变的 protected_boxes: {[list(box) for box in plan.protected_boxes]}\n"
        "- 必须按阶段验收：先看字体/字号/字槽/基线/笔画粗细/局部倾斜姿态，再看黑度和灰边，再看照片质感和背景修补。\n"
        "- 如果 text_shape 阶段未通过，不能用降黑、加模糊、加噪声或背景解释来判定通过。\n"
        "- 必须检查 target_roi 是否覆盖完整旧文字，而不是覆盖“姓名:”标签或冒号碎片。\n"
        "- 如果 source_text 和 target_text 字数不同，也只能改 source_text 的姓名区域以及可用空白，不能改名前标签或名后其它文字。\n"
        "- 如果旧字擦除后的底色明显发白、过于平滑、像涂抹补丁，必须 pass=false。\n"
        "- 如果旧文字没有被完整清除、仍有任何旧字残留，必须 pass=false。\n"
        "- 如果新文字不在旧文字原位置，而是偏到标签、冒号或其他字段位置，必须 pass=false。\n"
        "- 如果 hard_check_report 中 font_style_gate.pass=false 或 strict_gate.pass=false，必须 pass=false。\n"
    )


def image_to_data_url(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def image_from_data_url(value: str) -> Image.Image:
    if "," in value and value.lstrip().startswith("data:"):
        value = value.split(",", 1)[1]
    raw = base64.b64decode(value)
    return ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")


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
        successes.sort(key=lambda item: (item[0], item[1]))
        selected_direction_key, selected_score, _orientation_index, orientation, candidate, regions, selected_attempt = successes[0]
        margin = None
        if len(successes) > 1:
            margin = round(float(successes[1][0] - selected_direction_key), 3)
        return candidate, regions, {
            "applied": orientation != "none",
            "orientation": orientation,
            "attempts": attempts,
            "selected_score": round(float(selected_score), 3),
            "direction_margin": margin,
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
        pattern = rf"^\s*(?:字段|field)?\s*{re.escape(alias)}\s*[:：,，;；-]*\s*"
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
                    break
            if len(value_runs) >= desired_count:
                return tuple(value_runs[:desired_count])
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

    selected_score, roi, selected_plan, selected_source_text = min(candidates, key=lambda item: item[0])
    roi = expand_auto_search_roi(
        img,
        roi,
        selected_plan,
        source_text=selected_source_text,
        target_text=target_text,
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
    if slots and not is_mostly_cjk(source_text or target_text):
        slot_widths = [max(1, slot.x2 - slot.x1) for slot in slots]
        slot_heights = [max(1, slot.y2 - slot.y1) for slot in slots]
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


def text_complexity_ratio(plan: RenderPlan, params: CandidateParams) -> float:
    source_text = plan.source_text or ""
    target_text = plan.target_text or ""
    if not source_text or not target_text or source_text == target_text:
        return 1.0
    try:
        font = ImageFont.truetype(params.font_path, params.font_size)
    except OSError:
        return 1.0

    def rendered_ink(text: str) -> int:
        bbox = font.getbbox(text)
        width = max(1, bbox[2] - bbox[0] + 8)
        height = max(1, bbox[3] - bbox[1] + 8)
        layer = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(layer)
        draw.text((4 - bbox[0], 4 - bbox[1]), text, fill=255, font=font)
        arr = np.array(layer)
        return int(np.count_nonzero(arr > 24))

    source_ink = rendered_ink(source_text)
    target_ink = rendered_ink(target_text)
    if source_ink <= 0 or target_ink <= 0:
        return 1.0
    return max(0.75, min(1.65, target_ink / source_ink))


def rendered_glyph_complexity(params: dict[str, Any] | CandidateParams, text: str) -> int | None:
    if not text:
        return None
    try:
        if isinstance(params, CandidateParams):
            font_path = params.font_path
            font_size = params.font_size
        else:
            font_path = str(params.get("font_path") or "")
            font_size = int(params.get("font_size") or 0)
        font = ImageFont.truetype(font_path, font_size)
    except (OSError, TypeError, ValueError):
        return None
    bbox = font.getbbox(text)
    width = max(1, bbox[2] - bbox[0] + 8)
    height = max(1, bbox[3] - bbox[1] + 8)
    layer = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(layer)
    draw.text((4 - bbox[0], 4 - bbox[1]), text, fill=255, font=font)
    return int(np.count_nonzero(np.array(layer) > 24))


def gray_band_profile_for_box(
    gray: np.ndarray,
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> dict[str, Any] | None:
    x1, y1, x2, y2 = clamp_box(box, image_size)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = gray[y1:y2, x1:x2]
    if crop.size <= 0:
        return None
    counts = gray_band_counts(crop)
    area = int(crop.size)
    lt165 = max(1, int(counts.get("lt165") or 0))
    return {
        "box": [x1, y1, x2, y2],
        "area": area,
        "mean_gray": round(float(np.mean(crop)), 3),
        "std_gray": round(float(np.std(crop)), 3),
        "counts": counts,
        "density": {
            "lt55": round(float(counts["lt55"]) / max(1, area), 5),
            "lt70": round(float(counts["lt70"]) / max(1, area), 5),
            "lt165": round(float(counts["lt165"]) / max(1, area), 5),
        },
        "share": {
            "lt55_of_lt165": round(float(counts["lt55"]) / lt165, 5),
            "lt70_of_lt165": round(float(counts["lt70"]) / lt165, 5),
            "outer_120_165_of_lt165": round(float(counts["band_120_165"]) / lt165, 5),
        },
    }


def same_row_reference_boxes(plan: RenderPlan) -> tuple[tuple[int, int, int, int], ...]:
    tx1, ty1, tx2, ty2 = plan.target_roi
    row_h = max(1, ty2 - ty1)
    boxes: list[tuple[int, int, int, int]] = []
    for box in plan.protected_boxes:
        if not protected_box_overlaps_row(box, plan.target_roi):
            continue
        if box[2] <= tx1 or box[0] >= tx2:
            if box[3] - box[1] >= max(4, int(round(row_h * 0.35))):
                boxes.append(box)
    return tuple(boxes[:8])


def build_reference_profile(
    original: Image.Image,
    plan: RenderPlan,
    params: CandidateParams,
) -> dict[str, Any]:
    gray = cv2.cvtColor(np.array(original.convert("RGB")), cv2.COLOR_RGB2GRAY)
    source_box = plan.source_reference_box or plan.target_roi
    source_profile = gray_band_profile_for_box(gray, source_box, original.size)
    target_profile = gray_band_profile_for_box(gray, plan.target_roi, original.size)
    slot_profiles = [
        profile
        for profile in (
            gray_band_profile_for_box(gray, text_run_box(slot), original.size)
            for slot in tuple(sorted(plan.slot_boxes, key=lambda item: item.x1))
        )
        if profile is not None
    ]
    neighbor_profiles = [
        profile
        for profile in (
            gray_band_profile_for_box(gray, box, original.size)
            for box in same_row_reference_boxes(plan)
        )
        if profile is not None and (profile.get("counts") or {}).get("lt165", 0) >= 8
    ]

    source_counts = (source_profile or {}).get("counts") or {}
    source_density = (source_profile or {}).get("density") or {}
    try:
        source_lt55 = float(source_counts.get("lt55") or 0.0)
        source_lt165 = float(source_counts.get("lt165") or 0.0)
        source_core_density = float(source_density.get("lt55") or 0.0)
        source_std_gray = float((source_profile or {}).get("std_gray") or 0.0)
    except (TypeError, ValueError):
        source_lt55 = 0.0
        source_lt165 = 0.0
        source_core_density = 0.0
        source_std_gray = 0.0

    neighbor_core_densities: list[float] = []
    neighbor_outer_shares: list[float] = []
    for profile in neighbor_profiles:
        density = profile.get("density") or {}
        share = profile.get("share") or {}
        try:
            neighbor_core_densities.append(float(density.get("lt55") or 0.0))
            neighbor_outer_shares.append(float(share.get("outer_120_165_of_lt165") or 0.0))
        except (TypeError, ValueError):
            continue
    neighbor_core_density = float(np.median(neighbor_core_densities)) if neighbor_core_densities else None
    neighbor_outer_share = float(np.median(neighbor_outer_shares)) if neighbor_outer_shares else None

    source_count = len(text_chars(plan.source_text or ""))
    target_count = len(text_chars(plan.target_text))
    complexity_ratio = text_complexity_ratio(plan, params)
    count_ratio = float(target_count / source_count) if source_count and target_count else 1.0
    count_allowance = 1.0 + max(0.0, min(0.9, count_ratio - 1.0)) * 0.55
    complexity_allowance = 1.0 + max(0.0, min(0.65, complexity_ratio - 1.0)) * 0.45
    core_delta_limit = max(58.0, source_lt55 * 0.32 * count_allowance * complexity_allowance)
    char_core_delta_limit = max(48.0, source_lt55 * 0.70 * complexity_allowance / max(1.0, source_count or 1.0))
    core_mean_lighten_limit = max(2.0, min(8.0, source_std_gray * 0.14))

    reference_core_density = max(
        source_core_density,
        neighbor_core_density if neighbor_core_density is not None else 0.0,
    )
    core_density_conflict = False
    if neighbor_core_density is not None:
        core_density_conflict = abs(source_core_density - neighbor_core_density) >= 0.045
    if neighbor_core_density is None:
        arbitration_rule = "source_only_no_same_row_neighbor"
        selected_core_reference = "source"
    elif source_core_density >= neighbor_core_density:
        arbitration_rule = "use_darker_of_source_and_same_row_neighbor"
        selected_core_reference = "source"
    else:
        arbitration_rule = "use_darker_of_source_and_same_row_neighbor"
        selected_core_reference = "same_row_neighbor"
    if reference_core_density >= 0.18:
        opacity_floor = 0.72
    elif reference_core_density >= 0.12:
        opacity_floor = 0.68
    elif reference_core_density >= 0.08:
        opacity_floor = 0.64
    else:
        opacity_floor = 0.60

    return {
        "enabled": source_profile is not None,
        "source_text": plan.source_text or "",
        "target_text": plan.target_text,
        "source_box": list(source_box),
        "target_roi": list(plan.target_roi),
        "source": source_profile,
        "target": target_profile,
        "slots": slot_profiles,
        "same_row_neighbors": neighbor_profiles,
        "dynamic_ink": {
            "source_lt55": round(source_lt55, 3),
            "source_lt165": round(source_lt165, 3),
            "source_core_density": round(source_core_density, 5),
            "neighbor_core_density": None if neighbor_core_density is None else round(neighbor_core_density, 5),
            "neighbor_outer_share": None if neighbor_outer_share is None else round(neighbor_outer_share, 5),
            "text_complexity_ratio": round(float(complexity_ratio), 4),
            "text_count_ratio": round(float(count_ratio), 4),
            "roi_lt55_delta_limit": round(float(core_delta_limit), 3),
            "char_lt55_delta_limit": round(float(char_core_delta_limit), 3),
            "core_mean_lighten_limit": round(float(core_mean_lighten_limit), 3),
            "opacity_floor_for_excess_core": round(float(opacity_floor), 3),
            "basis": "source_text_region_and_same_row_neighbors",
            "arbitration": {
                "source_core_density": round(float(source_core_density), 5),
                "neighbor_core_density": (
                    None if neighbor_core_density is None else round(float(neighbor_core_density), 5)
                ),
                "conflict_detected": core_density_conflict,
                "selected_core_reference": selected_core_reference,
                "rule": arbitration_rule,
            },
        },
    }


def dynamic_ink_limits(report: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    profile = report.get("reference_profile")
    if not isinstance(profile, dict):
        return {}
    dynamic = profile.get("dynamic_ink")
    return dynamic if isinstance(dynamic, dict) else {}


def opacity_floor_for_excess_black(report: dict[str, Any] | None) -> float:
    dynamic = dynamic_ink_limits(report)
    try:
        return max(0.55, min(0.76, float(dynamic.get("opacity_floor_for_excess_core"))))
    except (TypeError, ValueError):
        return 0.64


def local_ink_balance_issues(report: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    strict_gate = report.get("strict_gate")
    dynamic_limits = dynamic_ink_limits(report)
    complexity_ratio = 1.0
    if isinstance(strict_gate, dict):
        try:
            complexity_ratio = float(strict_gate.get("text_complexity_ratio") or 1.0)
        except (TypeError, ValueError):
            complexity_ratio = 1.0
    char_bands = report.get("char_gray_band_metrics")
    if isinstance(char_bands, dict) and char_bands.get("enabled"):
        per_char = [item for item in char_bands.get("per_char", []) if isinstance(item, dict)]

        def slot_area(item: dict[str, Any]) -> float:
            box = item.get("slot_box") or []
            if len(box) != 4:
                return 0.0
            return max(1.0, float((int(box[2]) - int(box[0])) * (int(box[3]) - int(box[1]))))

        def neighbor_core_bounds_changed_item(item: dict[str, Any]) -> bool:
            if item.get("source_char") == item.get("target_char"):
                return False
            if not is_mostly_cjk(str(item.get("target_char") or "")):
                return False
            try:
                changed_index = int(item.get("index") or 0)
            except (TypeError, ValueError):
                changed_index = 0
            neighbor_items = [
                candidate
                for candidate in per_char
                if candidate.get("source_char") == candidate.get("target_char")
                and is_mostly_cjk(str(candidate.get("target_char") or ""))
            ]
            if not neighbor_items:
                return False
            neighbor = min(
                neighbor_items,
                key=lambda candidate: abs(int(candidate.get("index") or 0) - changed_index),
            )
            changed_counts = item.get("new") or {}
            neighbor_counts = neighbor.get("new") or {}
            if not isinstance(changed_counts, dict) or not isinstance(neighbor_counts, dict):
                return False
            changed_area = slot_area(item)
            neighbor_area = slot_area(neighbor)
            try:
                changed_core_density = float(changed_counts.get("lt55") or 0.0) / changed_area
                neighbor_core_density = float(neighbor_counts.get("lt55") or 0.0) / neighbor_area
                changed_lt70_density = float(changed_counts.get("lt70") or 0.0) / changed_area
                neighbor_lt70_density = float(neighbor_counts.get("lt70") or 0.0) / neighbor_area
                changed_lt165 = max(1.0, float(changed_counts.get("lt165") or 0.0))
                neighbor_lt165 = max(1.0, float(neighbor_counts.get("lt165") or 0.0))
                changed_outer_share = float(changed_counts.get("band_120_165") or 0.0) / changed_lt165
                neighbor_outer_share = float(neighbor_counts.get("band_120_165") or 0.0) / neighbor_lt165
            except (TypeError, ValueError, ZeroDivisionError):
                return False
            return (
                neighbor_core_density >= 0.08
                and changed_core_density <= neighbor_core_density + 0.022
                and changed_lt70_density <= neighbor_lt70_density + 0.040
                and changed_outer_share <= neighbor_outer_share + 0.060
            )

        for item in per_char:
            if not isinstance(item, dict):
                continue
            if item.get("source_char") == item.get("target_char"):
                continue
            delta = item.get("delta") or {}
            old = item.get("old") or {}
            try:
                old_lt55 = float(old.get("lt55") or 0.0)
                old_lt165 = float(old.get("lt165") or 0.0)
                lt55_delta = float(delta.get("lt55") or 0.0)
                lt165_delta = float(delta.get("lt165") or 0.0)
                band_70_90_delta = float(delta.get("band_70_90") or 0.0)
                band_90_120_delta = float(delta.get("band_90_120") or 0.0)
            except (TypeError, ValueError):
                continue
            if old_lt55 <= 0 or old_lt165 <= 0:
                continue

            middle_delta = band_70_90_delta + band_90_120_delta
            complexity_core_allowance = min(0.86, 0.70 + max(0.0, complexity_ratio - 1.0) * 0.55)
            try:
                profile_char_limit = float(dynamic_limits.get("char_lt55_delta_limit") or 0.0)
            except (TypeError, ValueError):
                profile_char_limit = 0.0
            max_core_delta = max(52.0, profile_char_limit, old_lt55 * complexity_core_allowance)
            if lt55_delta > max_core_delta and middle_delta < -20.0:
                if complexity_ratio >= 1.08 and neighbor_core_bounds_changed_item(item):
                    continue
                issues.append(
                    {
                        "type": "changed_char_core_too_black_hard",
                        "index": item.get("index"),
                        "source_char": item.get("source_char"),
                        "target_char": item.get("target_char"),
                        "lt55_delta": lt55_delta,
                        "limit": round(max_core_delta, 3),
                        "middle_gray_delta": middle_delta,
                        "text_complexity_ratio": round(complexity_ratio, 4),
                    }
                )
            elif lt55_delta > max(70.0, old_lt55 * 0.85) and lt165_delta > max(36.0, old_lt165 * 0.12):
                if complexity_ratio >= 1.08 and neighbor_core_bounds_changed_item(item):
                    continue
                issues.append(
                    {
                        "type": "changed_char_core_too_black",
                        "index": item.get("index"),
                        "source_char": item.get("source_char"),
                        "target_char": item.get("target_char"),
                        "lt55_delta": lt55_delta,
                        "limit": round(max(70.0, old_lt55 * 0.85), 3),
                    }
                )

    bands = (report.get("strict_visual_metrics") or {}).get("bands")
    if isinstance(bands, dict):
        try:
            old_lt55 = float(bands.get("old_lt55_pixels") or 0.0)
            lt55_delta = float(bands.get("lt55_delta") or 0.0)
            old_share = bands.get("old_lt55_share_of_lt165")
            new_share = bands.get("new_lt55_share_of_lt165")
            share_delta = None if old_share is None or new_share is None else float(new_share) - float(old_share)
        except (TypeError, ValueError):
            old_lt55 = 0.0
            lt55_delta = 0.0
            share_delta = None
        try:
            roi_core_delta_limit = float(dynamic_limits.get("roi_lt55_delta_limit") or 0.0)
        except (TypeError, ValueError):
            roi_core_delta_limit = 0.0
        roi_core_delta_limit = max(48.0, roi_core_delta_limit or old_lt55 * 0.32)
        if old_lt55 >= 32.0 and lt55_delta > roi_core_delta_limit:
            issues.append(
                {
                    "type": "roi_core_too_black",
                    "lt55_delta": lt55_delta,
                    "limit": round(roi_core_delta_limit, 3),
                    "limit_source": "reference_profile",
                }
            )
        if share_delta is not None and share_delta > 0.085:
            issues.append(
                {
                    "type": "roi_black_core_share_too_high",
                    "share_delta": round(float(share_delta), 4),
                    "limit": 0.085,
                }
            )
    return issues


def local_stroke_body_issues(
    report: dict[str, Any],
    *,
    allow_excess_black_core: bool = False,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if not isinstance(report, dict):
        return issues
    if not allow_excess_black_core and report_has_excess_black_core(report):
        return issues

    char_bands = report.get("char_gray_band_metrics")
    if not isinstance(char_bands, dict) or not char_bands.get("enabled"):
        return issues

    for item in char_bands.get("per_char", []):
        if not isinstance(item, dict):
            continue
        if item.get("source_char") == item.get("target_char"):
            continue
        delta = item.get("delta") or {}
        old = item.get("old") or {}
        try:
            old_lt165 = float(old.get("lt165") or 0.0)
            lt55_delta = float(delta.get("lt55") or 0.0)
            lt165_delta = float(delta.get("lt165") or 0.0)
            band_55_70_delta = float(delta.get("band_55_70") or 0.0)
            band_70_90_delta = float(delta.get("band_70_90") or 0.0)
            band_90_120_delta = float(delta.get("band_90_120") or 0.0)
            band_120_165_delta = float(delta.get("band_120_165") or 0.0)
            old_band_120_165 = float(old.get("band_120_165") or 0.0)
        except (TypeError, ValueError):
            continue
        if old_lt165 <= 0:
            continue

        middle_delta = band_70_90_delta + band_90_120_delta
        gray_body_delta = band_55_70_delta + middle_delta
        min_body_delta = min(12.0, old_lt165 * 0.03)
        strict_gate = report.get("strict_gate")
        complexity_ratio = 1.0
        if isinstance(strict_gate, dict):
            try:
                complexity_ratio = float(strict_gate.get("text_complexity_ratio") or 1.0)
            except (TypeError, ValueError):
                complexity_ratio = 1.0
        params = report.get("params")
        stroke_opacity = 0.0
        blur = 0.0
        if isinstance(params, dict):
            try:
                stroke_opacity = float(params.get("stroke_opacity") or 0.0)
                blur = float(params.get("blur") or 0.0)
            except (TypeError, ValueError):
                stroke_opacity = 0.0
                blur = 0.0
        middle_limit = -18.0 if complexity_ratio >= 1.08 else -30.0
        if lt165_delta < -6.0:
            issues.append(
                {
                    "type": "changed_char_stroke_body_too_small",
                    "index": item.get("index"),
                    "source_char": item.get("source_char"),
                    "target_char": item.get("target_char"),
                    "lt165_delta": round(lt165_delta, 3),
                    "limit": -6.0,
                    "lt55_delta": round(lt55_delta, 3),
                    "middle_gray_delta": round(middle_delta, 3),
                    "gray_body_delta": round(gray_body_delta, 3),
                    "band_120_165_delta": round(band_120_165_delta, 3),
                }
            )
            continue

        core_is_not_deficient = lt55_delta >= -8.0
        middle_is_missing = middle_delta < middle_limit
        body_gain_is_too_small = lt165_delta < min_body_delta and gray_body_delta < -18.0
        hard_core_with_thin_body = lt55_delta > 28.0 and middle_delta < -28.0
        light_edge_substitutes_for_body = (
            middle_delta < -16.0
            and band_120_165_delta > 38.0
            and complexity_ratio >= 1.08
        )
        fine_strokes_too_soft = (
            complexity_ratio >= 1.08
            and band_70_90_delta < -12.0
            and band_120_165_delta > 62.0
            and (stroke_opacity < 0.085 or blur > 0.42)
        )
        if core_is_not_deficient and (
            middle_is_missing
            or body_gain_is_too_small
            or hard_core_with_thin_body
            or light_edge_substitutes_for_body
            or fine_strokes_too_soft
        ):
            neighbor_style_clear = not local_neighbor_style_issues(report, allow_excess_black_core=True)
            clean_edge_body_is_bounded = (
                complexity_ratio >= 1.08
                and neighbor_style_clear
                and lt165_delta >= min_body_delta - 4.0
                and lt55_delta >= 30.0
                and middle_delta >= -42.0
                and not light_edge_substitutes_for_body
                and band_120_165_delta <= max(46.0, old_band_120_165 * 0.42)
            )
            if clean_edge_body_is_bounded:
                continue
            issues.append(
                {
                    "type": "changed_char_fine_strokes_too_soft"
                    if fine_strokes_too_soft
                    else "changed_char_stroke_body_too_narrow",
                    "index": item.get("index"),
                    "source_char": item.get("source_char"),
                    "target_char": item.get("target_char"),
                    "lt165_delta": round(lt165_delta, 3),
                    "min_body_delta": round(min_body_delta, 3),
                    "lt55_delta": round(lt55_delta, 3),
                    "middle_gray_delta": round(middle_delta, 3),
                    "gray_body_delta": round(gray_body_delta, 3),
                    "band_120_165_delta": round(band_120_165_delta, 3),
                    "middle_limit": round(middle_limit, 3),
                    "text_complexity_ratio": round(complexity_ratio, 4),
                    "stroke_opacity": round(stroke_opacity, 3),
                    "blur": round(blur, 3),
                }
            )
    return issues


def local_pose_issues(report: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if not isinstance(report, dict):
        return issues
    pose = report.get("char_pose_metrics")
    if not isinstance(pose, dict) or not pose.get("enabled"):
        return issues
    for item in pose.get("per_char", []):
        if not isinstance(item, dict) or not item.get("changed"):
            continue
        try:
            source_raw = item.get("source_slot_shear")
            source_shear = None if source_raw is None else float(source_raw)
            neighbor_raw = item.get("neighbor_shear")
            neighbor_shear = None if neighbor_raw is None else float(neighbor_raw)
            reference_raw = item.get("reference_shear")
            reference_shear = None if reference_raw is None else float(reference_raw)
            applied_shear = float(item.get("applied_shear") or 0.0)
        except (TypeError, ValueError):
            continue
        if reference_shear is None:
            reference_shear = source_shear
        if reference_raw is None and source_shear is not None and neighbor_shear is not None and abs(neighbor_shear) >= 0.018:
            reference_shear = source_shear * 0.75 + neighbor_shear * 0.25
        if reference_shear is None:
            reference_shear = neighbor_shear
        if reference_shear is None:
            continue
        if abs(reference_shear) < 0.05:
            continue
        min_applied_abs = abs(reference_shear) * 0.78
        if reference_shear * applied_shear <= 0 or abs(applied_shear) < min_applied_abs:
            issues.append(
                {
                    "type": "changed_char_pose_shear_too_weak",
                    "index": item.get("index"),
                    "source_char": item.get("source_char"),
                    "target_char": item.get("target_char"),
                    "source_slot_shear": None if source_shear is None else round(source_shear, 4),
                    "neighbor_shear": None if neighbor_shear is None else round(neighbor_shear, 4),
                    "reference_shear": round(reference_shear, 4),
                    "applied_shear": round(applied_shear, 4),
                    "min_applied_abs_shear": round(min_applied_abs, 4),
                }
            )
    return issues


def report_has_fine_strokes_too_soft(report: dict[str, Any] | None) -> bool:
    if not isinstance(report, dict):
        return False
    for issue in local_stroke_body_issues(report):
        if isinstance(issue, dict) and issue.get("type") == "changed_char_fine_strokes_too_soft":
            return True
    return False


def report_has_outer_gray_halo(report: dict[str, Any] | None) -> bool:
    if not isinstance(report, dict):
        return False
    return bool(local_outer_gray_halo_issues(report, allow_excess_black_core=True))


def local_outer_gray_halo_issues(
    report: dict[str, Any],
    *,
    allow_excess_black_core: bool = False,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if not isinstance(report, dict):
        return issues
    if not allow_excess_black_core and report_has_excess_black_core(report):
        return issues
    char_bands = report.get("char_gray_band_metrics")
    if not isinstance(char_bands, dict) or not char_bands.get("enabled"):
        return issues
    per_char = [item for item in char_bands.get("per_char", []) if isinstance(item, dict)]
    if len(per_char) < 2:
        return issues

    params = report.get("params")
    if not isinstance(params, dict):
        params = {}

    def slot_area(item: dict[str, Any]) -> float:
        box = item.get("slot_box") or []
        if len(box) != 4:
            return 0.0
        return max(1.0, float((int(box[2]) - int(box[0])) * (int(box[3]) - int(box[1]))))

    changed_items = [
        item
        for item in per_char
        if item.get("source_char") != item.get("target_char")
        and is_mostly_cjk(str(item.get("target_char") or ""))
    ]
    neighbor_items = [
        item
        for item in per_char
        if item.get("source_char") == item.get("target_char")
        and is_mostly_cjk(str(item.get("target_char") or ""))
    ]
    if not changed_items or not neighbor_items:
        return issues

    for changed in changed_items:
        try:
            changed_index = int(changed.get("index") or 0)
        except (TypeError, ValueError):
            changed_index = 0
        neighbor = min(
            neighbor_items,
            key=lambda item: abs(int(item.get("index") or 0) - changed_index),
        )
        target_char = str(changed.get("target_char") or "")
        neighbor_char = str(neighbor.get("target_char") or "")
        target_complexity = rendered_glyph_complexity(params, target_char)
        neighbor_complexity = rendered_glyph_complexity(params, neighbor_char)
        if target_complexity is not None and neighbor_complexity:
            if target_complexity < neighbor_complexity * 0.75:
                continue

        changed_counts = changed.get("new") or {}
        neighbor_counts = neighbor.get("new") or {}
        old_counts = changed.get("old") or {}
        delta_counts = changed.get("delta") or {}
        if (
            not isinstance(changed_counts, dict)
            or not isinstance(neighbor_counts, dict)
            or not isinstance(old_counts, dict)
            or not isinstance(delta_counts, dict)
        ):
            continue
        changed_area = slot_area(changed)
        neighbor_area = slot_area(neighbor)
        try:
            changed_lt165 = max(1.0, float(changed_counts.get("lt165") or 0.0))
            neighbor_lt165 = max(1.0, float(neighbor_counts.get("lt165") or 0.0))
            changed_outer = float(changed_counts.get("band_120_165") or 0.0)
            neighbor_outer = float(neighbor_counts.get("band_120_165") or 0.0)
            old_outer = float(old_counts.get("band_120_165") or 0.0)
            outer_delta = float(delta_counts.get("band_120_165") or 0.0)
        except (TypeError, ValueError):
            continue

        changed_outer_share = changed_outer / changed_lt165
        neighbor_outer_share = neighbor_outer / neighbor_lt165
        changed_outer_density = changed_outer / max(1.0, changed_area)
        neighbor_outer_density = neighbor_outer / max(1.0, neighbor_area)
        outer_share_gap = changed_outer_share - neighbor_outer_share
        outer_density_gap = changed_outer_density - neighbor_outer_density
        excess_outer_limit = max(48.0, old_outer * 0.42)
        if outer_share_gap > 0.060 and outer_delta > excess_outer_limit and outer_density_gap > 0.030:
            issues.append(
                {
                    "type": "changed_char_neighbor_outer_gray_halo_too_high",
                    "index": changed.get("index"),
                    "source_char": changed.get("source_char"),
                    "target_char": target_char,
                    "neighbor_index": neighbor.get("index"),
                    "neighbor_char": neighbor_char,
                    "target_complexity": target_complexity,
                    "neighbor_complexity": neighbor_complexity,
                    "changed_outer_share": round(changed_outer_share, 4),
                    "neighbor_outer_share": round(neighbor_outer_share, 4),
                    "outer_share_gap": round(outer_share_gap, 4),
                    "changed_outer_density": round(changed_outer_density, 4),
                    "neighbor_outer_density": round(neighbor_outer_density, 4),
                    "outer_density_gap": round(outer_density_gap, 4),
                    "band_120_165_delta": round(outer_delta, 3),
                    "band_120_165_delta_limit": round(excess_outer_limit, 3),
                }
            )
    return issues


def local_neighbor_style_issues(
    report: dict[str, Any],
    *,
    allow_excess_black_core: bool = False,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if not isinstance(report, dict):
        return issues
    if not allow_excess_black_core and report_has_excess_black_core(report):
        return issues
    issues.extend(local_outer_gray_halo_issues(report, allow_excess_black_core=allow_excess_black_core))
    char_bands = report.get("char_gray_band_metrics")
    if not isinstance(char_bands, dict) or not char_bands.get("enabled"):
        return issues
    per_char = [item for item in char_bands.get("per_char", []) if isinstance(item, dict)]
    if len(per_char) < 2:
        return issues

    params = report.get("params")
    if not isinstance(params, dict):
        params = {}

    def slot_area(item: dict[str, Any]) -> float:
        box = item.get("slot_box") or []
        if len(box) != 4:
            return 0.0
        return max(1.0, float((int(box[2]) - int(box[0])) * (int(box[3]) - int(box[1]))))

    def density(counts: dict[str, Any], key: str, area: float) -> float:
        try:
            return float(counts.get(key) or 0.0) / max(1.0, area)
        except (TypeError, ValueError):
            return 0.0

    changed_items = [
        item
        for item in per_char
        if item.get("source_char") != item.get("target_char")
        and is_mostly_cjk(str(item.get("target_char") or ""))
    ]
    neighbor_items = [
        item
        for item in per_char
        if item.get("source_char") == item.get("target_char")
        and is_mostly_cjk(str(item.get("target_char") or ""))
    ]
    if not changed_items or not neighbor_items:
        return issues

    for changed in changed_items:
        try:
            changed_index = int(changed.get("index") or 0)
        except (TypeError, ValueError):
            changed_index = 0
        neighbor = min(
            neighbor_items,
            key=lambda item: abs(int(item.get("index") or 0) - changed_index),
        )
        target_char = str(changed.get("target_char") or "")
        neighbor_char = str(neighbor.get("target_char") or "")
        target_complexity = rendered_glyph_complexity(params, target_char)
        neighbor_complexity = rendered_glyph_complexity(params, neighbor_char)
        if target_complexity is not None and neighbor_complexity:
            if target_complexity < neighbor_complexity * 0.75:
                continue

        changed_counts = changed.get("new") or {}
        neighbor_counts = neighbor.get("new") or {}
        if not isinstance(changed_counts, dict) or not isinstance(neighbor_counts, dict):
            continue
        changed_area = slot_area(changed)
        neighbor_area = slot_area(neighbor)
        changed_core_density = density(changed_counts, "lt55", changed_area)
        neighbor_core_density = density(neighbor_counts, "lt55", neighbor_area)
        changed_lt70_density = density(changed_counts, "lt70", changed_area)
        neighbor_lt70_density = density(neighbor_counts, "lt70", neighbor_area)
        changed_lt165 = max(1.0, float(changed_counts.get("lt165") or 0.0))
        neighbor_lt165 = max(1.0, float(neighbor_counts.get("lt165") or 0.0))
        changed_outer_share = float(changed_counts.get("band_120_165") or 0.0) / changed_lt165
        neighbor_outer_share = float(neighbor_counts.get("band_120_165") or 0.0) / neighbor_lt165

        core_density_gap = neighbor_core_density - changed_core_density
        lt70_density_gap = neighbor_lt70_density - changed_lt70_density
        outer_share_gap = changed_outer_share - neighbor_outer_share
        neighbor_core_is_meaningful = neighbor_core_density >= 0.08 and changed_core_density >= 0.08
        core_gap_is_visible = core_density_gap > 0.022 and outer_share_gap > 0.045
        gray_edge_is_substituting = core_density_gap > 0.018 and outer_share_gap > 0.050
        if neighbor_core_is_meaningful and (core_gap_is_visible or gray_edge_is_substituting):
            issues.append(
                {
                    "type": "changed_char_neighbor_core_density_too_low",
                    "index": changed.get("index"),
                    "source_char": changed.get("source_char"),
                    "target_char": target_char,
                    "neighbor_index": neighbor.get("index"),
                    "neighbor_char": neighbor_char,
                    "target_complexity": target_complexity,
                    "neighbor_complexity": neighbor_complexity,
                    "changed_core_density": round(changed_core_density, 4),
                    "neighbor_core_density": round(neighbor_core_density, 4),
                    "core_density_gap": round(core_density_gap, 4),
                    "changed_lt70_density": round(changed_lt70_density, 4),
                    "neighbor_lt70_density": round(neighbor_lt70_density, 4),
                    "lt70_density_gap": round(lt70_density_gap, 4),
                    "changed_outer_share": round(changed_outer_share, 4),
                    "neighbor_outer_share": round(neighbor_outer_share, 4),
                    "outer_share_gap": round(outer_share_gap, 4),
                    "core_density_limit": 0.024,
                    "outer_share_limit": 0.045,
                }
            )
    return issues


def changed_texture_boxes(plan: RenderPlan) -> tuple[tuple[int, int, int, int], ...]:
    source_chars = text_chars(plan.source_text or "")
    target_chars = text_chars(plan.target_text)
    if (
        source_chars
        and target_chars
        and len(source_chars) == len(target_chars)
        and plan.slot_boxes
    ):
        slots = tuple(sorted(plan.slot_boxes, key=lambda item: item.x1))
        boxes: list[tuple[int, int, int, int]] = []
        for idx, (source_char, target_char) in enumerate(zip(source_chars, target_chars)):
            if source_char == target_char or idx >= len(slots):
                continue
            slot = slots[idx]
            pad_x = max(3, int(round((slot.x2 - slot.x1) * 0.18)))
            pad_y = max(2, int(round((slot.y2 - slot.y1) * 0.14)))
            boxes.append(
                clamp_box_to_container(
                    (slot.x1 - pad_x, slot.y1 - pad_y, slot.x2 + pad_x, slot.y2 + pad_y),
                    plan.target_roi,
                )
            )
        if boxes:
            return tuple(boxes)
    return (plan.target_roi,)


def photo_texture_metrics(
    original: Image.Image,
    candidate: Image.Image,
    plan: RenderPlan,
    params: CandidateParams,
) -> dict[str, Any]:
    original_gray = cv2.cvtColor(np.array(original.convert("RGB")), cv2.COLOR_RGB2GRAY)
    candidate_gray = cv2.cvtColor(np.array(candidate.convert("RGB")), cv2.COLOR_RGB2GRAY)
    old_lap = np.abs(cv2.Laplacian(original_gray, cv2.CV_32F, ksize=3))
    new_lap = np.abs(cv2.Laplacian(candidate_gray, cv2.CV_32F, ksize=3))
    old_residual = np.abs(original_gray.astype(np.float32) - cv2.GaussianBlur(original_gray, (0, 0), 1.2))
    new_residual = np.abs(candidate_gray.astype(np.float32) - cv2.GaussianBlur(candidate_gray, (0, 0), 1.2))

    per_box: list[dict[str, Any]] = []
    old_edge_values: list[float] = []
    new_edge_values: list[float] = []
    old_residual_values: list[float] = []
    new_residual_values: list[float] = []
    for box in changed_texture_boxes(plan):
        x1, y1, x2, y2 = clamp_box(box, original.size)
        if x2 <= x1 or y2 <= y1:
            continue
        old_crop = original_gray[y1:y2, x1:x2]
        new_crop = candidate_gray[y1:y2, x1:x2]
        old_mask = old_crop < 165
        new_mask = new_crop < 165
        if int(np.count_nonzero(old_mask)) < 12 or int(np.count_nonzero(new_mask)) < 12:
            continue
        kernel = np.ones((3, 3), dtype=np.uint8)
        old_edge = cv2.morphologyEx(old_mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
        new_edge = cv2.morphologyEx(new_mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
        old_near = cv2.dilate(old_mask.astype(np.uint8), kernel, iterations=1) > 0
        new_near = cv2.dilate(new_mask.astype(np.uint8), kernel, iterations=1) > 0
        if int(np.count_nonzero(old_edge)) < 6 or int(np.count_nonzero(new_edge)) < 6:
            continue
        old_edge_lap = old_lap[y1:y2, x1:x2][old_edge]
        new_edge_lap = new_lap[y1:y2, x1:x2][new_edge]
        old_res = old_residual[y1:y2, x1:x2][old_near]
        new_res = new_residual[y1:y2, x1:x2][new_near]
        old_edge_mean = float(np.mean(old_edge_lap))
        new_edge_mean = float(np.mean(new_edge_lap))
        old_res_mean = float(np.mean(old_res)) if old_res.size else 0.0
        new_res_mean = float(np.mean(new_res)) if new_res.size else 0.0
        old_edge_values.append(old_edge_mean)
        new_edge_values.append(new_edge_mean)
        old_residual_values.append(old_res_mean)
        new_residual_values.append(new_res_mean)
        per_box.append(
            {
                "box": [x1, y1, x2, y2],
                "old_edge_laplacian_mean": round(old_edge_mean, 3),
                "new_edge_laplacian_mean": round(new_edge_mean, 3),
                "edge_laplacian_ratio": round(new_edge_mean / max(1.0, old_edge_mean), 4),
                "old_residual_mean": round(old_res_mean, 3),
                "new_residual_mean": round(new_res_mean, 3),
                "residual_ratio": round(new_res_mean / max(1.0, old_res_mean), 4),
            }
        )

    if not per_box:
        return {"enabled": False, "reason": "not enough changed text texture pixels"}

    old_edge_mean = float(np.mean(old_edge_values))
    new_edge_mean = float(np.mean(new_edge_values))
    old_residual_mean = float(np.mean(old_residual_values))
    new_residual_mean = float(np.mean(new_residual_values))
    jpeg_quality = int(params.jpeg_quality or 0)
    jpeg_weight = 0.0
    if 1 <= jpeg_quality <= 99:
        jpeg_weight = max(0.0, min(0.42, (100.0 - float(jpeg_quality)) / 85.0))
    return {
        "enabled": True,
        "old_edge_laplacian_mean": round(old_edge_mean, 3),
        "new_edge_laplacian_mean": round(new_edge_mean, 3),
        "edge_laplacian_ratio": round(new_edge_mean / max(1.0, old_edge_mean), 4),
        "old_residual_mean": round(old_residual_mean, 3),
        "new_residual_mean": round(new_residual_mean, 3),
        "residual_ratio": round(new_residual_mean / max(1.0, old_residual_mean), 4),
        "params": {
            "blur": params.blur,
            "photo_warp": params.photo_warp,
            "edge_breakup": params.edge_breakup,
            "photo_noise": params.photo_noise,
            "jpeg_quality": params.jpeg_quality,
            "jpeg_weight": round(jpeg_weight, 4),
        },
        "per_box": per_box,
    }


def local_photo_texture_issues(report: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    metrics = report.get("photo_texture_metrics") if isinstance(report, dict) else None
    if not isinstance(metrics, dict) or not metrics.get("enabled"):
        return issues
    params = metrics.get("params") if isinstance(metrics.get("params"), dict) else {}
    try:
        blur = float(params.get("blur") or 0.0)
        photo_warp = float(params.get("photo_warp") or 0.0)
        edge_breakup = float(params.get("edge_breakup") or 0.0)
        photo_noise = float(params.get("photo_noise") or 0.0)
        jpeg_weight = float(params.get("jpeg_weight") or 0.0)
        edge_ratio = float(metrics.get("edge_laplacian_ratio") or 1.0)
        residual_ratio = float(metrics.get("residual_ratio") or 1.0)
        old_residual_mean = float(metrics.get("old_residual_mean") or 0.0)
        old_edge_mean = float(metrics.get("old_edge_laplacian_mean") or 0.0)
    except (TypeError, ValueError):
        return issues

    texture_strength = photo_warp + edge_breakup * 3.0 + photo_noise * 2.0 + jpeg_weight
    source_photo_like = old_residual_mean >= 2.4 or old_edge_mean >= 28.0
    if source_photo_like and blur <= 0.04 and texture_strength < 0.018:
        issues.append(
            {
                "type": "photo_texture_not_applied",
                "blur": round(blur, 3),
                "texture_strength": round(texture_strength, 4),
                "old_residual_mean": round(old_residual_mean, 3),
                "old_edge_laplacian_mean": round(old_edge_mean, 3),
            }
        )
    if source_photo_like and edge_ratio > 1.85 and blur < 0.22:
        issues.append(
            {
                "type": "photo_texture_too_sharp",
                "edge_laplacian_ratio": round(edge_ratio, 4),
                "blur": round(blur, 3),
            }
        )
    if source_photo_like and residual_ratio < 0.42 and photo_noise < 0.006 and jpeg_weight < 0.02:
        issues.append(
            {
                "type": "photo_texture_too_clean",
                "residual_ratio": round(residual_ratio, 4),
                "photo_noise": round(photo_noise, 3),
                "jpeg_weight": round(jpeg_weight, 4),
            }
        )
    if edge_ratio < 0.18 and blur > 0.70:
        issues.append(
            {
                "type": "photo_texture_too_blurry",
                "edge_laplacian_ratio": round(edge_ratio, 4),
                "blur": round(blur, 3),
            }
        )
    return issues


def background_texture_metrics(
    original: Image.Image,
    candidate: Image.Image,
    plan: RenderPlan,
    params: CandidateParams,
) -> dict[str, Any]:
    original_arr = np.array(original.convert("RGB"))
    candidate_arr = np.array(candidate.convert("RGB"))
    original_gray = cv2.cvtColor(original_arr, cv2.COLOR_RGB2GRAY)
    candidate_gray = cv2.cvtColor(candidate_arr, cv2.COLOR_RGB2GRAY)
    h, w = original_gray.shape
    tx1, ty1, tx2, ty2 = plan.target_roi
    if tx2 <= tx1 or ty2 <= ty1:
        return {"enabled": False, "reason": "empty target roi"}

    old_dark = (original_gray[ty1:ty2, tx1:tx2] < int(params.mask_threshold)).astype(np.uint8)
    if int(np.count_nonzero(old_dark)) < 8:
        return {"enabled": False, "reason": "not enough old text mask pixels"}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    old_dark_base = old_dark > 0
    old_dark = cv2.dilate(old_dark, kernel, iterations=max(1, int(params.mask_dilate_iterations)))
    replacement_layer = draw_replacement_layer(size=original.size, plan=plan, params=params, original=original)
    alpha = np.array(replacement_layer.getchannel("A"))
    new_alpha = alpha[ty1:ty2, tx1:tx2] > 18
    trailing_cleanup_mask = build_trailing_value_cleanup_mask(plan, replacement_layer, original.size)
    trailing_cleanup_crop = trailing_cleanup_mask[ty1:ty2, tx1:tx2] > 0
    fill_mask = (old_dark > 0) & ~new_alpha
    if int(np.count_nonzero(fill_mask)) < 10:
        fill_mask = old_dark > 0
    if int(np.count_nonzero(fill_mask)) < 10:
        return {"enabled": False, "reason": "not enough fill background pixels"}

    pad_x = max(10, int(round((tx2 - tx1) * 0.35)))
    pad_y = max(6, int(round((ty2 - ty1) * 0.60)))
    rx1, ry1, rx2, ry2 = clamp_box((tx1 - pad_x, ty1 - pad_y, tx2 + pad_x, ty2 + pad_y), original.size)
    reference_mask = np.ones((ry2 - ry1, rx2 - rx1), dtype=bool)
    reference_mask[ty1 - ry1 : ty2 - ry1, tx1 - rx1 : tx2 - rx1] = False
    reference_crop = original_gray[ry1:ry2, rx1:rx2]
    reference_mask &= reference_crop >= 165
    if int(np.count_nonzero(reference_mask)) < 24:
        reference_mask = original_gray[ry1:ry2, rx1:rx2] >= 150
    if int(np.count_nonzero(reference_mask)) < 24:
        return {"enabled": False, "reason": "not enough same-row background reference"}

    old_fill = original_gray[ty1:ty2, tx1:tx2][fill_mask]
    new_fill = candidate_gray[ty1:ty2, tx1:tx2][fill_mask]
    reference_values = reference_crop[reference_mask]

    def residual_values(gray: np.ndarray) -> np.ndarray:
        residual = np.abs(gray.astype(np.float32) - cv2.GaussianBlur(gray, (0, 0), 1.2))
        return residual

    old_residual = residual_values(original_gray[ty1:ty2, tx1:tx2])[fill_mask]
    new_residual = residual_values(candidate_gray[ty1:ty2, tx1:tx2])[fill_mask]
    reference_residual = residual_values(reference_crop)[reference_mask]
    old_mean = float(np.mean(old_fill))
    new_mean = float(np.mean(new_fill))
    reference_mean = float(np.mean(reference_values))
    old_std = float(np.std(old_fill))
    new_std = float(np.std(new_fill))
    reference_std = float(np.std(reference_values))
    old_residual_mean = float(np.mean(old_residual)) if old_residual.size else 0.0
    new_residual_mean = float(np.mean(new_residual)) if new_residual.size else 0.0
    reference_residual_mean = float(np.mean(reference_residual)) if reference_residual.size else 0.0

    trailing_metrics: dict[str, Any] = {
        "enabled": int(np.count_nonzero(trailing_cleanup_crop)) >= 24,
        "pixels": int(np.count_nonzero(trailing_cleanup_crop)),
    }
    if trailing_metrics["enabled"]:
        trailing_values = candidate_gray[ty1:ty2, tx1:tx2][trailing_cleanup_crop]
        trailing_residual = residual_values(candidate_gray[ty1:ty2, tx1:tx2])[trailing_cleanup_crop]
        trailing_std = float(np.std(trailing_values))
        trailing_residual_mean = float(np.mean(trailing_residual)) if trailing_residual.size else 0.0
        trailing_metrics.update(
            {
                "mean_gray": round(float(np.mean(trailing_values)), 3),
                "reference_mean_gray": round(reference_mean, 3),
                "reference_mean_delta": round(float(np.mean(trailing_values) - reference_mean), 3),
                "std_gray": round(trailing_std, 3),
                "reference_std_gray": round(reference_std, 3),
                "std_ratio": round(trailing_std / max(0.8, reference_std), 4),
                "residual_mean": round(trailing_residual_mean, 3),
                "reference_residual_mean": round(reference_residual_mean, 3),
                "residual_ratio": round(trailing_residual_mean / max(0.8, reference_residual_mean), 4),
            }
        )

    ghost_probe = cv2.dilate(
        old_dark_base.astype(np.uint8),
        kernel,
        iterations=max(3, int(params.mask_dilate_iterations) + 1),
    ) > 0
    ghost_probe &= ~new_alpha
    local_background_mask = (~ghost_probe) & ~new_alpha & (original_gray[ty1:ty2, tx1:tx2] >= 165)
    ghost_pixels = int(np.count_nonzero(ghost_probe))
    local_background_pixels = int(np.count_nonzero(local_background_mask))
    ghost_metrics: dict[str, Any] = {
        "enabled": ghost_pixels >= 24 and local_background_pixels >= 24,
        "probe_pixels": ghost_pixels,
        "local_background_pixels": local_background_pixels,
    }
    if ghost_metrics["enabled"]:
        ghost_values = candidate_gray[ty1:ty2, tx1:tx2][ghost_probe]
        local_background_values = candidate_gray[ty1:ty2, tx1:tx2][local_background_mask]
        bg_p95 = float(np.percentile(local_background_values, 95))
        bg_p99 = float(np.percentile(local_background_values, 99))
        bg_p05 = float(np.percentile(local_background_values, 5))
        bg_p10 = float(np.percentile(local_background_values, 10))
        bg_p25 = float(np.percentile(local_background_values, 25))
        ghost_p05 = float(np.percentile(ghost_values, 5))
        ghost_p10 = float(np.percentile(ghost_values, 10))
        ghost_p25 = float(np.percentile(ghost_values, 25))
        ghost_p95 = float(np.percentile(ghost_values, 95))
        ghost_p99 = float(np.percentile(ghost_values, 99))
        ghost_metrics.update(
            {
                "probe_mean_gray": round(float(np.mean(ghost_values)), 3),
                "local_background_mean_gray": round(float(np.mean(local_background_values)), 3),
                "probe_background_mean_delta": round(
                    float(np.mean(ghost_values) - np.mean(local_background_values)),
                    3,
                ),
                "probe_p05_gray": round(ghost_p05, 3),
                "local_background_p05_gray": round(bg_p05, 3),
                "probe_p05_delta": round(ghost_p05 - bg_p05, 3),
                "probe_p10_gray": round(ghost_p10, 3),
                "local_background_p10_gray": round(bg_p10, 3),
                "probe_p10_delta": round(ghost_p10 - bg_p10, 3),
                "probe_p25_gray": round(ghost_p25, 3),
                "local_background_p25_gray": round(bg_p25, 3),
                "probe_p25_delta": round(ghost_p25 - bg_p25, 3),
                "probe_p95_gray": round(ghost_p95, 3),
                "local_background_p95_gray": round(bg_p95, 3),
                "probe_p95_delta": round(ghost_p95 - bg_p95, 3),
                "probe_p99_gray": round(ghost_p99, 3),
                "local_background_p99_gray": round(bg_p99, 3),
                "probe_p99_delta": round(ghost_p99 - bg_p99, 3),
                "bright_over_background_p95_ratio": round(
                    float(np.mean(ghost_values > bg_p95)),
                    5,
                ),
                "bright_over_background_p99_ratio": round(
                    float(np.mean(ghost_values > bg_p99)),
                    5,
                ),
                "dark_under_background_p05_ratio": round(
                    float(np.mean(ghost_values < bg_p05)),
                    5,
                ),
                "dark_under_background_p10_ratio": round(
                    float(np.mean(ghost_values < bg_p10)),
                    5,
                ),
                "dark_under_background_p25_ratio": round(
                    float(np.mean(ghost_values < bg_p25)),
                    5,
                ),
            }
        )
    return {
        "enabled": True,
        "target_roi": [tx1, ty1, tx2, ty2],
        "fill_pixels": int(np.count_nonzero(fill_mask)),
        "reference_pixels": int(np.count_nonzero(reference_mask)),
        "old_fill_mean_gray": round(old_mean, 3),
        "new_fill_mean_gray": round(new_mean, 3),
        "reference_mean_gray": round(reference_mean, 3),
        "new_reference_mean_delta": round(new_mean - reference_mean, 3),
        "old_reference_mean_delta": round(old_mean - reference_mean, 3),
        "old_fill_std_gray": round(old_std, 3),
        "new_fill_std_gray": round(new_std, 3),
        "reference_std_gray": round(reference_std, 3),
        "std_ratio": round(new_std / max(0.8, reference_std), 4),
        "old_residual_mean": round(old_residual_mean, 3),
        "new_residual_mean": round(new_residual_mean, 3),
        "reference_residual_mean": round(reference_residual_mean, 3),
        "residual_ratio": round(new_residual_mean / max(0.8, reference_residual_mean), 4),
        "white_ghost_probe": ghost_metrics,
        "trailing_cleanup_patch": trailing_metrics,
    }


def local_background_texture_issues(report: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = report.get("background_texture_metrics") if isinstance(report, dict) else None
    if not isinstance(metrics, dict) or not metrics.get("enabled"):
        return []
    issues: list[dict[str, Any]] = []
    try:
        mean_delta = float(metrics.get("new_reference_mean_delta") or 0.0)
        old_mean_delta = float(metrics.get("old_reference_mean_delta") or 0.0)
        std_ratio = float(metrics.get("std_ratio") or 1.0)
        residual_ratio = float(metrics.get("residual_ratio") or 1.0)
        reference_residual = float(metrics.get("reference_residual_mean") or 0.0)
    except (TypeError, ValueError):
        return issues
    ghost_probe = metrics.get("white_ghost_probe")
    if isinstance(ghost_probe, dict) and ghost_probe.get("enabled"):
        try:
            bright_p95_ratio = float(ghost_probe.get("bright_over_background_p95_ratio") or 0.0)
            bright_p99_ratio = float(ghost_probe.get("bright_over_background_p99_ratio") or 0.0)
            dark_p10_ratio = float(ghost_probe.get("dark_under_background_p10_ratio") or 0.0)
            dark_p25_ratio = float(ghost_probe.get("dark_under_background_p25_ratio") or 0.0)
            p10_delta = float(ghost_probe.get("probe_p10_delta") or 0.0)
            p25_delta = float(ghost_probe.get("probe_p25_delta") or 0.0)
            p95_delta = float(ghost_probe.get("probe_p95_delta") or 0.0)
            p99_delta = float(ghost_probe.get("probe_p99_delta") or 0.0)
            mean_delta_local = float(ghost_probe.get("probe_background_mean_delta") or 0.0)
            probe_pixels = int(ghost_probe.get("probe_pixels") or 0)
        except (TypeError, ValueError):
            bright_p95_ratio = 0.0
            bright_p99_ratio = 0.0
            dark_p10_ratio = 0.0
            dark_p25_ratio = 0.0
            p10_delta = 0.0
            p25_delta = 0.0
            p95_delta = 0.0
            p99_delta = 0.0
            mean_delta_local = 0.0
            probe_pixels = 0
        if probe_pixels >= 120 and (
            (bright_p95_ratio > 0.14 and p95_delta > 6.0)
            or (bright_p99_ratio > 0.055 and p99_delta > 12.0)
            or (mean_delta_local > 2.6 and bright_p95_ratio > 0.10)
        ):
            issues.append(
                {
                    "type": "background_white_ghost_residual",
                    "bright_over_background_p95_ratio": round(bright_p95_ratio, 5),
                    "p95_delta": round(p95_delta, 3),
                    "bright_over_background_p99_ratio": round(bright_p99_ratio, 5),
                    "p99_delta": round(p99_delta, 3),
                    "mean_delta": round(mean_delta_local, 3),
                    "probe_pixels": probe_pixels,
                }
            )
        if probe_pixels >= 120 and (
            (mean_delta_local < -3.0 and dark_p10_ratio > 0.28)
            or (p10_delta < -5.0 and dark_p25_ratio > 0.45)
            or (p25_delta < -3.5 and dark_p25_ratio > 0.55)
        ):
            issues.append(
                {
                    "type": "background_shadow_ghost_residual",
                    "dark_under_background_p10_ratio": round(dark_p10_ratio, 5),
                    "p10_delta": round(p10_delta, 3),
                    "dark_under_background_p25_ratio": round(dark_p25_ratio, 5),
                    "p25_delta": round(p25_delta, 3),
                    "mean_delta": round(mean_delta_local, 3),
                    "probe_pixels": probe_pixels,
                }
            )
    trailing_patch = metrics.get("trailing_cleanup_patch")
    if isinstance(trailing_patch, dict) and trailing_patch.get("enabled"):
        try:
            trailing_std_ratio = float(trailing_patch.get("std_ratio") or 1.0)
            trailing_residual_ratio = float(trailing_patch.get("residual_ratio") or 1.0)
            trailing_mean_delta = float(trailing_patch.get("reference_mean_delta") or 0.0)
            trailing_pixels = int(trailing_patch.get("pixels") or 0)
        except (TypeError, ValueError):
            trailing_std_ratio = 1.0
            trailing_residual_ratio = 1.0
            trailing_mean_delta = 0.0
            trailing_pixels = 0
        if trailing_pixels >= 120 and (trailing_std_ratio < 0.36 or trailing_residual_ratio < 0.42):
            issues.append(
                {
                    "type": "background_trailing_patch_too_smooth",
                    "std_ratio": round(trailing_std_ratio, 4),
                    "residual_ratio": round(trailing_residual_ratio, 4),
                    "mean_delta": round(trailing_mean_delta, 3),
                    "pixels": trailing_pixels,
                }
            )
    if abs(mean_delta) > max(12.0, abs(old_mean_delta) + 7.0):
        issues.append(
            {
                "type": "background_fill_luminance_mismatch",
                "new_reference_mean_delta": round(mean_delta, 3),
                "old_reference_mean_delta": round(old_mean_delta, 3),
                "limit": round(max(12.0, abs(old_mean_delta) + 7.0), 3),
            }
        )
    if reference_residual >= 1.8 and residual_ratio < 0.42:
        issues.append(
            {
                "type": "background_fill_too_smooth",
                "residual_ratio": round(residual_ratio, 4),
                "reference_residual_mean": round(reference_residual, 3),
                "limit": 0.42,
            }
        )
    texture_variance_limit = 0.62 if reference_residual >= 2.4 else 0.48
    structured_ghost = any(
        issue.get("type") in {"background_white_ghost_residual", "background_shadow_ghost_residual"}
        for issue in issues
    )
    if std_ratio < texture_variance_limit and residual_ratio < 1.05 and not structured_ghost:
        issues.append(
            {
                "type": "background_fill_low_texture_variance",
                "std_ratio": round(std_ratio, 4),
                "limit": round(texture_variance_limit, 3),
            }
        )
    return issues


def strict_gate_stage_issues(report: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    stages = {
        "text_shape": [],
        "ink_gray_balance": [],
        "background_cleanup": [],
    }
    strict_gate = report.get("strict_gate")
    if not isinstance(strict_gate, dict):
        return stages

    for issue in strict_gate.get("issues", []):
        if not isinstance(issue, dict):
            continue
        issue_type = str(issue.get("type") or "")
        if (
            issue_type.startswith("char_")
            or issue_type.startswith("replacement_")
            or issue_type.startswith("font_")
            or "font_" in issue_type
        ):
            stages["text_shape"].append(issue)
        elif issue_type.startswith("extra_source_slot_"):
            stages["background_cleanup"].append(issue)
        else:
            stages["ink_gray_balance"].append(issue)
    return stages


def stage_gate_for_report(report: dict[str, Any]) -> dict[str, Any]:
    strict_stage_issues = strict_gate_stage_issues(report)
    font_style = report.get("font_style_gate")
    if isinstance(font_style, dict):
        strict_stage_issues["text_shape"].extend(
            issue for issue in font_style.get("issues", []) if isinstance(issue, dict)
        )

    text_shape_issues = (
        strict_stage_issues["text_shape"]
        + local_stroke_body_issues(report, allow_excess_black_core=True)
        + local_neighbor_style_issues(report, allow_excess_black_core=True)
        + list(report.get("local_pose_issues") or [])
    )
    ink_issues = strict_stage_issues["ink_gray_balance"] + list(report.get("local_ink_balance_issues") or [])
    photo_issues = list(report.get("local_photo_texture_issues") or [])
    background_issues = strict_stage_issues["background_cleanup"] + list(
        report.get("local_background_texture_issues") or []
    )

    stages = [
        {
            "id": "hard_boundary",
            "label": "ROI boundary and protected text",
            "pass": bool(report.get("pass")),
            "issues": [] if report.get("pass") else [{"type": "hard_check_failed"}],
        },
        {
            "id": "text_shape",
            "label": "font, size, slot, baseline, stroke body, local pose",
            "pass": not text_shape_issues,
            "issues": text_shape_issues,
        },
        {
            "id": "ink_gray_balance",
            "label": "true-black core, mid-gray body, outer gray edge",
            "pass": not ink_issues,
            "issues": ink_issues,
        },
        {
            "id": "photo_texture",
            "label": "scan blur, edge breakup, compression/noise texture",
            "pass": not photo_issues,
            "issues": photo_issues,
        },
        {
            "id": "background_cleanup",
            "label": "inpaint texture and removed old slots",
            "pass": not background_issues,
            "issues": background_issues,
        },
    ]

    blocking_stage = None
    for stage in stages:
        if not stage["pass"]:
            blocking_stage = stage["id"]
            break
    return {
        "order": [stage["id"] for stage in stages],
        "blocking_stage": blocking_stage,
        "pass": blocking_stage is None,
        "stages": stages,
    }


def stage_issues(report: dict[str, Any] | None, stage_id: str) -> list[dict[str, Any]]:
    if not isinstance(report, dict):
        return []
    stage_gate = report.get("stage_gate")
    if not isinstance(stage_gate, dict):
        stage_gate = stage_gate_for_report(report)
    for stage in stage_gate.get("stages", []):
        if isinstance(stage, dict) and stage.get("id") == stage_id:
            return [issue for issue in stage.get("issues", []) if isinstance(issue, dict)]
    return []


def stage_selection_penalty(report: dict[str, Any] | None) -> float:
    if not isinstance(report, dict):
        return 0.0
    stage_gate = report.get("stage_gate")
    if not isinstance(stage_gate, dict):
        stage_gate = stage_gate_for_report(report)
    blocking_stage = str(stage_gate.get("blocking_stage") or "")
    if not blocking_stage:
        return 0.0

    penalties = {
        "hard_boundary": 600000.0,
        "text_shape": 9500.0,
        "ink_gray_balance": 2600.0,
        "photo_texture": 1200.0,
        "background_cleanup": 1800.0,
    }
    penalty = penalties.get(blocking_stage, 1000.0)
    issues = stage_issues(report, blocking_stage)
    penalty += len(issues) * (850.0 if blocking_stage == "text_shape" else 260.0)
    if blocking_stage == "text_shape":
        penalty += gray_stroke_balance_penalty(report) * 4.0
        if local_pose_issues(report):
            penalty += 900.0
    return penalty


def ink_stage_issue_severity(report: dict[str, Any] | None) -> float:
    severity = 0.0
    for issue in stage_issues(report, "ink_gray_balance"):
        issue_type = str(issue.get("type") or "")
        try:
            if issue_type == "dark_pixel_ratio_too_high":
                severity += max(0.0, float(issue.get("actual") or 0.0) - float(issue.get("limit") or 0.0)) * 1800.0
            elif issue_type in {"changed_char_core_too_black", "changed_char_core_too_black_hard"}:
                severity += max(0.0, float(issue.get("lt55_delta") or 0.0) - float(issue.get("limit") or 0.0)) * 2.2
            elif issue_type == "roi_core_too_black":
                severity += max(0.0, float(issue.get("lt55_delta") or 0.0) - float(issue.get("limit") or 0.0)) * 2.8
            elif issue_type == "roi_black_core_share_too_high":
                severity += max(0.0, float(issue.get("share_delta") or 0.0) - float(issue.get("limit") or 0.0)) * 2400.0
            elif issue_type == "core_mean_gray_too_light":
                severity += max(0.0, float(issue.get("actual") or 0.0) - float(issue.get("limit") or 0.0)) * 32.0
            elif issue_type == "core_lighten_too_high":
                severity += max(0.0, float(issue.get("actual") or 0.0) - float(issue.get("limit") or 0.0)) * 26.0
            else:
                severity += 80.0
        except (TypeError, ValueError):
            severity += 80.0
    return severity


def stage_issue_severity(report: dict[str, Any] | None, stage_id: str | None) -> float:
    if not stage_id:
        return 0.0
    if stage_id == "ink_gray_balance":
        return ink_stage_issue_severity(report)
    if stage_id == "background_cleanup":
        severity = 0.0
        for issue in stage_issues(report, "background_cleanup"):
            issue_type = str(issue.get("type") or "")
            try:
                if issue_type == "background_white_ghost_residual":
                    severity += max(0.0, float(issue.get("bright_over_background_p95_ratio") or 0.0) - 0.08) * 900.0
                    severity += max(0.0, float(issue.get("p95_delta") or 0.0) - 4.0) * 16.0
                    severity += max(0.0, float(issue.get("p99_delta") or 0.0) - 8.0) * 8.0
                elif issue_type == "background_shadow_ghost_residual":
                    severity += max(0.0, float(issue.get("dark_under_background_p10_ratio") or 0.0) - 0.20) * 720.0
                    severity += max(0.0, abs(float(issue.get("p10_delta") or 0.0)) - 3.0) * 14.0
                    severity += max(0.0, abs(float(issue.get("mean_delta") or 0.0)) - 2.0) * 18.0
                elif issue_type == "background_trailing_patch_too_smooth":
                    severity += max(0.0, 0.36 - float(issue.get("std_ratio") or 0.0)) * 620.0
                    severity += max(0.0, 0.42 - float(issue.get("residual_ratio") or 0.0)) * 760.0
                elif issue_type == "background_fill_luminance_mismatch":
                    severity += max(0.0, abs(float(issue.get("new_reference_mean_delta") or 0.0)) - float(issue.get("limit") or 0.0)) * 35.0
                elif issue_type == "background_fill_too_smooth":
                    severity += max(0.0, float(issue.get("limit") or 0.0) - float(issue.get("residual_ratio") or 0.0)) * 520.0
                elif issue_type == "background_fill_low_texture_variance":
                    severity += max(0.0, float(issue.get("limit") or 0.0) - float(issue.get("std_ratio") or 0.0)) * 420.0
                else:
                    severity += 90.0
            except (TypeError, ValueError):
                severity += 90.0
        return severity
    if stage_id == "photo_texture":
        severity = 0.0
        for issue in stage_issues(report, "photo_texture"):
            issue_type = str(issue.get("type") or "")
            try:
                if issue_type == "photo_texture_too_sharp":
                    severity += max(0.0, float(issue.get("edge_laplacian_ratio") or 0.0) - 1.0) * 160.0
                elif issue_type == "photo_texture_too_clean":
                    severity += max(0.0, 0.42 - float(issue.get("residual_ratio") or 0.0)) * 420.0
                elif issue_type == "photo_texture_too_blurry":
                    severity += max(0.0, 0.18 - float(issue.get("edge_laplacian_ratio") or 0.0)) * 520.0
                else:
                    severity += 90.0
            except (TypeError, ValueError):
                severity += 90.0
        return severity
    return float(len(stage_issues(report, stage_id))) * 100.0


def patch_families(patch: dict[str, Any] | None) -> list[str]:
    if not isinstance(patch, dict):
        return []
    keys = {str(key) for key, value in patch.items() if value is not None}
    families: list[str] = []
    for family, family_keys in PATCH_FAMILY_KEYS.items():
        if keys & family_keys:
            families.append(family)
    return families


def patch_policy_for_stage(stage_id: str | None) -> dict[str, Any]:
    policy = STAGE_PATCH_POLICY.get(stage_id or "none") or STAGE_PATCH_POLICY["none"]
    return {
        "stage": stage_id or None,
        "allowed_families": list(policy.get("allowed_families") or []),
        "forbidden_families": list(policy.get("forbidden_families") or []),
        "secondary_only_families": list(policy.get("secondary_only_families") or []),
        "reason": policy.get("reason"),
    }


def patch_policy_audit(stage_id: str | None, patch: dict[str, Any] | None) -> dict[str, Any]:
    policy = patch_policy_for_stage(stage_id)
    families = patch_families(patch)
    allowed = set(policy["allowed_families"])
    forbidden = set(policy["forbidden_families"])
    secondary_only = set(policy.get("secondary_only_families") or [])
    effective_families = list(families)
    if stage_id == "text_shape" and "stroke_body_shape" in effective_families:
        effective_families = [
            family
            for family in effective_families
            if family != "ink_gray_balance"
        ]
    if stage_id == "ink_gray_balance" and "ink_gray_balance" in effective_families:
        effective_families = [
            family
            for family in effective_families
            if family != "stroke_body_shape"
        ]
    primary_families = [family for family in effective_families if family not in secondary_only]
    forbidden_hits = [family for family in families if family in forbidden]
    disallowed_primary = [
        family
        for family in primary_families
        if family not in allowed and family not in secondary_only
    ]
    secondary_only_without_primary = bool(families and not primary_families and secondary_only.intersection(families))
    is_allowed = not forbidden_hits and not disallowed_primary and not secondary_only_without_primary
    if not families:
        is_allowed = True
    reason = "allowed"
    if forbidden_hits:
        reason = f"forbidden families for current stage: {', '.join(forbidden_hits)}"
    elif disallowed_primary:
        reason = f"primary families outside current stage: {', '.join(disallowed_primary)}"
    elif secondary_only_without_primary:
        reason = "secondary-only photo texture patch cannot be the main adjustment for this stage"
    return {
        **policy,
        "families": families,
        "effective_families": effective_families,
        "primary_families": primary_families,
        "allowed": is_allowed,
        "rejection_reason": None if is_allowed else reason,
    }


def stage_policy_summary(stage_id: str | None) -> dict[str, Any]:
    return patch_policy_for_stage(stage_id)


def params_delta(before: CandidateParams, after: CandidateParams) -> dict[str, Any]:
    changes: dict[str, Any] = {}
    for name in before.__dataclass_fields__:
        if name == "candidate_id":
            continue
        old = getattr(before, name)
        new = getattr(after, name)
        if old == new:
            continue
        if isinstance(old, float) or isinstance(new, float):
            old_out = round(float(old), 4)
            new_out = round(float(new), 4)
        else:
            old_out = old
            new_out = new
        changes[name] = {"from": old_out, "to": new_out}
    return changes


def constraint_reason(
    report: dict[str, Any] | None,
    acceptance: dict[str, Any],
) -> str:
    stage = (stage_gate_for_report(report).get("blocking_stage") if isinstance(report, dict) else None)
    if stage == "text_shape":
        return "text_shape_stage_caps_ink_and_photo_side_effects"
    if report_has_excess_black_core(report):
        return "dynamic_reference_profile_caps_true_black_core"
    if report_has_background_white_ghost(report):
        return "background_white_or_shadow_ghost_cleanup_caps"
    if report_has_background_low_texture(report):
        return "background_low_texture_recovery_caps"
    if acceptance_reports_background_patch(acceptance):
        return "vision_background_patch_feedback_caps"
    if report_needs_wider_gray_strokes(report):
        return "stroke_body_recovery_caps"
    if report_needs_thinner_strokes(report):
        return "thin_or_dark_core_recovery_caps"
    return "no_local_constraint"


def constraint_audit(
    raw_params: CandidateParams,
    constrained_params: CandidateParams,
    report: dict[str, Any] | None,
    acceptance: dict[str, Any],
) -> dict[str, Any]:
    changes = params_delta(raw_params, constrained_params)
    return {
        "applied": bool(changes),
        "reason": constraint_reason(report, acceptance) if changes else "none",
        "changes": changes,
    }


def alignment_vertical_penalty(report: dict[str, Any] | None) -> float:
    if not isinstance(report, dict):
        return 0.0
    alignment = report.get("char_alignment_metrics")
    if not isinstance(alignment, dict) or not alignment.get("enabled"):
        return 0.0
    penalty = 0.0
    for item in alignment.get("per_char", []):
        if not isinstance(item, dict) or not item.get("candidate_box"):
            continue
        try:
            center_dy = float(item.get("center_dy") or 0.0)
        except (TypeError, ValueError):
            continue
        penalty += abs(center_dy) * 32.0
        penalty += max(0.0, -center_dy) * 72.0
        penalty += max(0.0, center_dy - 0.75) * 96.0
    return penalty


def report_stage_pass(report: dict[str, Any]) -> bool:
    stage_gate = report.get("stage_gate")
    if not isinstance(stage_gate, dict):
        stage_gate = stage_gate_for_report(report)
    return bool(stage_gate.get("pass"))


def apply_local_acceptance_gate(acceptance: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    stage_gate = stage_gate_for_report(report)
    shape_stage_issues = stage_issues({"stage_gate": stage_gate}, "text_shape")
    ink_issues = local_ink_balance_issues(report)
    shape_blocking = stage_gate.get("blocking_stage") == "text_shape"
    neighbor_issues = local_neighbor_style_issues(report, allow_excess_black_core=True)
    if ink_issues and neighbor_issues:
        ink_issues = [
            issue
            for issue in ink_issues
            if str(issue.get("type") or "")
            not in {
                "changed_char_core_too_black_hard",
                "changed_char_core_too_black",
                "roi_core_too_black",
                "roi_black_core_share_too_high",
            }
        ]
    body_issues = local_stroke_body_issues(
        report,
        allow_excess_black_core=shape_blocking,
    )
    if ink_issues and not shape_blocking:
        body_issues = []
    pose_issues = [] if ink_issues and not shape_blocking else local_pose_issues(report)
    photo_issues = stage_issues({"stage_gate": stage_gate}, "photo_texture")
    background_issues = stage_issues({"stage_gate": stage_gate}, "background_cleanup")
    uncovered_shape_issue_count = max(
        0,
        len(shape_stage_issues) - len(body_issues) - len(neighbor_issues) - len(pose_issues),
    )
    if (
        stage_gate.get("pass")
        and not ink_issues
        and not body_issues
        and not neighbor_issues
        and not pose_issues
        and not uncovered_shape_issue_count
        and not photo_issues
        and not background_issues
    ):
        return acceptance

    gated = dict(acceptance or {})
    gated["stage_gate"] = stage_gate
    findings = gated.get("visual_findings")
    findings = dict(findings) if isinstance(findings, dict) else {}
    if ink_issues and not shape_blocking:
        findings.setdefault("stroke_weight", "too_bold")
        findings.setdefault("darkness", "too_dark")
        findings.setdefault("sharpness", "too_sharp")
    elif body_issues or neighbor_issues:
        findings["stroke_weight"] = "too_thin"
        findings.setdefault("darkness", "ok")
        findings.setdefault("sharpness", "too_sharp")
    else:
        findings.setdefault("font_similarity", "slightly_off")
        findings.setdefault("baseline", "slightly_off")
    gated["visual_findings"] = findings
    gated["pass"] = False
    gated["acceptance_level"] = "marginal"
    gated["final_decision"] = "revise"
    if ink_issues:
        gated["local_ink_balance_issues"] = ink_issues
    if body_issues:
        gated["local_stroke_body_issues"] = body_issues
    if neighbor_issues:
        gated["local_neighbor_style_issues"] = neighbor_issues
    if pose_issues:
        gated["local_pose_issues"] = pose_issues
    if photo_issues:
        gated["local_photo_texture_issues"] = photo_issues
    if background_issues:
        gated["local_background_cleanup_issues"] = background_issues
    if shape_stage_issues:
        gated["local_shape_stage_issues"] = shape_stage_issues
    reason = str(gated.get("reason") or "").strip()
    outer_neighbor_issues = [
        issue
        for issue in neighbor_issues
        if isinstance(issue, dict) and issue.get("type") == "changed_char_neighbor_outer_gray_halo_too_high"
    ]
    core_neighbor_issues = [
        issue
        for issue in neighbor_issues
        if isinstance(issue, dict) and issue.get("type") == "changed_char_neighbor_core_density_too_low"
    ]
    if shape_blocking and body_issues and neighbor_issues:
        local_reason = "本地形态阶段发现新字笔画体量和同一行邻字风格仍未过关；必须先修字体/字号/笔画身体/姿态，再进入黑度和照片质感阶段。"
    elif shape_blocking and body_issues:
        local_reason = "本地形态阶段发现新字笔画身体仍偏窄或中间灰阶不足；必须先修粗细和字形，再进入黑度阶段。"
    elif shape_blocking and neighbor_issues:
        local_reason = "本地形态阶段发现新字相对同一行保留字核心密度或外灰边不一致；必须先修邻字风格匹配。"
    elif ink_issues:
        local_reason = "本地硬指标发现新字深黑核心过量或灰阶过渡不足，需要继续降低核心黑块并补照片质感。"
    elif photo_issues:
        local_reason = "本地照片质感阶段发现新字与原图的拍照模糊、边缘断裂、噪声或压缩质感不一致，需要先补齐照片质感再交付。"
    elif background_issues:
        local_reason = "本地背景阶段发现旧字清理或底色纹理仍不自然，需要继续修背景。"
    elif body_issues and outer_neighbor_issues and not core_neighbor_issues:
        local_reason = "本地硬指标发现新字外层浅灰边相对同一行保留字偏多，底部/外缘有灰雾感；需要压掉外圈灰边，同时保留核心笔画。"
    elif body_issues and neighbor_issues:
        local_reason = "本地硬指标发现新字笔画体量不足，且相对同一行保留字核心暗部密度偏低、外层灰边偏多；需要让细笔画更实，同时避免继续用灰雾撑厚。"
    elif body_issues:
        local_reason = "本地硬指标发现新字笔画体量不足，中间灰阶覆盖偏少；需要加宽笔画身体和扫描灰边，而不是单纯压黑核心。"
    elif outer_neighbor_issues and not core_neighbor_issues:
        local_reason = "本地邻字风格指标发现新字相对同一行保留字外层浅灰边偏多，需要清理外圈灰雾而不是继续加模糊。"
    elif neighbor_issues:
        local_reason = "本地邻字风格指标发现新字相对同一行保留字核心暗部密度偏低、外层灰边偏多，需要让细笔画更实。"
    elif uncovered_shape_issue_count:
        local_reason = "本地形态阶段发现字体、字号、字槽、基线或笔画形态仍未过关，必须先修形态，再处理黑度和照片质感。"
    else:
        local_reason = "本地姿态指标发现新字倾斜继承不足，需要更贴近旧槽位的拍照倾斜。"
    gated["reason"] = f"{reason} {local_reason}".strip()
    must_fix = gated.get("must_fix")
    if not isinstance(must_fix, list):
        must_fix = []
    if ink_issues and not shape_blocking:
        must_fix.append("Reduce excessive <55 core pixels and recover mid-gray scanned edges before accepting.")
    elif body_issues:
        must_fix.append("Increase stroke body and mid-gray coverage while keeping <55 core pixels bounded.")
    if neighbor_issues:
        if outer_neighbor_issues and not core_neighbor_issues:
            must_fix.append("Reduce outer 120-165 gray halo around changed characters while preserving the dark core.")
        else:
            must_fix.append("Match changed character core density to same-row preserved neighbors without adding gray haze.")
    if photo_issues:
        must_fix.append("Match scan/photo texture after shape and ink pass: blur, edge breakup, noise, and compression must be locally consistent.")
    if background_issues:
        must_fix.append("Resolve background cleanup before accepting the final image.")
    if pose_issues and not ink_issues and not body_issues and not neighbor_issues:
        must_fix.append("Increase local slot shear inheritance for changed characters before accepting.")
    if uncovered_shape_issue_count:
        must_fix.append("Resolve text_shape stage before tuning ink, blur, noise, or background texture.")
    gated["must_fix"] = must_fix
    return gated


def candidate_report(
    original: Image.Image,
    candidate: Image.Image,
    plan: RenderPlan,
    params: CandidateParams,
    font_style_reference: dict[str, Any],
) -> dict[str, Any]:
    report = hard_check(original, candidate, plan.target_roi, plan.protected_boxes)
    reference_profile = build_reference_profile(original, plan, params)
    strict_metrics = strict_visual_metrics(original, candidate, plan.target_roi)
    source_count = len(text_chars(plan.source_text or ""))
    target_count = len(text_chars(plan.target_text))
    mismatch = bool(source_count and target_count and source_count != target_count)
    same_length_changed = bool(
        source_count
        and target_count
        and source_count == target_count
        and (plan.source_text or "") != plan.target_text
    )
    complexity_ratio = text_complexity_ratio(plan, params)
    if source_count and target_count > source_count:
        max_dark_pixel_ratio = min(1.90, max(1.42, 1.25 * target_count / source_count))
    elif same_length_changed:
        max_dark_pixel_ratio = min(1.22, max(1.12, 1.12 + max(0.0, complexity_ratio - 1.0) * 0.45))
    else:
        max_dark_pixel_ratio = 1.35 if mismatch else 1.12
    min_dark_pixel_ratio = 0.55 if mismatch and target_count < source_count else 0.78 if mismatch else 0.88
    shorter_replacement = bool(source_count and target_count and target_count < source_count)
    max_edge_lighten_delta = 6.0 if shorter_replacement else 4.0
    dynamic_limits = (reference_profile.get("dynamic_ink") or {}) if isinstance(reference_profile, dict) else {}
    try:
        max_core_lighten_delta = float(dynamic_limits.get("core_mean_lighten_limit") or 2.0)
    except (TypeError, ValueError):
        max_core_lighten_delta = 2.0
    strict_issues = strict_gate_issues(
        strict_metrics,
        max_dark_pixel_ratio=max_dark_pixel_ratio,
        min_dark_pixel_ratio=min_dark_pixel_ratio,
        max_core_mean_gray_delta=18.0,
        max_edge_mean_gray_delta=16.0,
        max_core_lighten_delta=max_core_lighten_delta,
        max_edge_lighten_delta=max_edge_lighten_delta,
    )
    cleanup_metrics = extra_source_slot_cleanup_metrics(original, candidate, plan, params)
    cleanup_issues = extra_source_slot_cleanup_issues(cleanup_metrics)
    alignment_metrics, alignment_issues = char_alignment_gate(
        original.size,
        plan,
        params,
        max_char_center_dx=2.0,
        max_char_center_distance_delta=2.0,
        max_char_center_dy=2.5,
        max_replacement_center_y_range=2.0,
    )
    font_style = font_style_gate(
        original,
        plan,
        params,
        font_style_reference,
        max_score_ratio=1.25,
    )
    report["params"] = asdict(params)
    report["reference_profile"] = reference_profile
    report["strict_visual_metrics"] = strict_metrics
    report["char_gray_band_metrics"] = char_gray_band_metrics(original, candidate, plan)
    report["char_pose_metrics"] = char_pose_metrics(original, plan, params)
    report["photo_texture_metrics"] = photo_texture_metrics(original, candidate, plan, params)
    report["background_texture_metrics"] = background_texture_metrics(original, candidate, plan, params)
    report["extra_source_slot_cleanup_metrics"] = cleanup_metrics
    report["char_alignment_metrics"] = alignment_metrics
    report["font_style_gate"] = font_style
    report["strict_gate"] = {
        "max_dark_pixel_ratio": max_dark_pixel_ratio,
        "min_dark_pixel_ratio": min_dark_pixel_ratio,
        "text_complexity_ratio": round(float(complexity_ratio), 4),
        "max_core_mean_gray_delta": 18.0,
        "max_edge_mean_gray_delta": 16.0,
        "max_core_lighten_delta": round(float(max_core_lighten_delta), 3),
        "max_edge_lighten_delta": max_edge_lighten_delta,
        "max_char_center_dx": 2.0,
        "max_char_center_dy": 2.5,
        "max_replacement_center_y_range": 2.0,
        "max_char_center_distance_delta": 2.0,
        "max_font_style_score_ratio": 1.25,
        "pass": not strict_issues
        and not cleanup_issues
        and not alignment_issues
        and bool(font_style.get("pass", True)),
        "issues": strict_issues
        + cleanup_issues
        + alignment_issues
        + list(font_style.get("issues", [])),
    }
    report["local_ink_balance_issues"] = local_ink_balance_issues(report)
    report["local_stroke_body_issues"] = local_stroke_body_issues(report)
    report["local_neighbor_style_issues"] = local_neighbor_style_issues(report)
    report["local_pose_issues"] = local_pose_issues(report)
    report["local_photo_texture_issues"] = local_photo_texture_issues(report)
    report["local_background_texture_issues"] = local_background_texture_issues(report)
    report["stage_gate"] = stage_gate_for_report(report)
    return report


def web_candidate_score(report: dict[str, Any]) -> float:
    if not report.get("pass"):
        return 1_000_000.0
    score = 0.0
    score += stage_selection_penalty(report)
    strict_gate = report.get("strict_gate")
    longer_replacement = (
        isinstance(strict_gate, dict)
        and float(strict_gate.get("max_dark_pixel_ratio") or 0.0) > 1.42
    )
    if isinstance(strict_gate, dict) and not strict_gate.get("pass", True):
        score += len(strict_gate.get("issues", [])) * 220.0

    metrics = report.get("strict_visual_metrics", {})
    thresholds = metrics.get("thresholds", {})
    for threshold, values in thresholds.items():
        ratio = values.get("dark_pixel_ratio")
        mean_delta = values.get("mean_gray_delta")
        if ratio is not None:
            if longer_replacement:
                desired_ratio = {
                    "120": 1.28,
                    "140": 1.16,
                    "150": 1.08,
                    "160": 0.98,
                    "165": 0.95,
                }.get(str(threshold), 1.0)
            else:
                desired_ratio = 1.0
            score += abs(float(ratio) - desired_ratio) * (80.0 if str(threshold) == "165" else 45.0)
        if mean_delta is not None:
            score += abs(float(mean_delta)) * (2.6 if str(threshold) == "165" else 1.8)

    bands = metrics.get("bands", {})
    if isinstance(bands, dict):
        old_lt55 = float(bands.get("old_lt55_pixels") or 0.0)
        new_lt55 = float(bands.get("new_lt55_pixels") or 0.0)
        old_lt70 = float(bands.get("old_lt70_pixels") or 0.0)
        new_lt70 = float(bands.get("new_lt70_pixels") or 0.0)
        old_gray_edge = float(bands.get("old_120_165_pixels") or 0.0)
        new_gray_edge = float(bands.get("new_120_165_pixels") or 0.0)
        if old_lt55 >= 16:
            score += max(0.0, (new_lt55 - old_lt55) - max(58.0, old_lt55 * 0.28)) * 5.0
        if old_lt70 >= 32:
            score += max(0.0, (new_lt70 - old_lt70) - max(70.0, old_lt70 * 0.24)) * 2.8
        if old_lt55 < 16:
            score += max(0.0, new_lt55 - old_lt55 - 24.0) * 18.0
        if old_lt70 < 32:
            score += max(0.0, new_lt70 - old_lt70 - 36.0) * 8.0
        score += abs(new_gray_edge - old_gray_edge) * 0.10

    cleanup_metrics = report.get("extra_source_slot_cleanup_metrics")
    if isinstance(cleanup_metrics, dict) and cleanup_metrics.get("enabled"):
        for item in cleanup_metrics.get("per_box", []):
            lt150_ratio = item.get("new_lt150_ratio")
            retention = item.get("lt150_retention_ratio")
            column_deviation = item.get("max_column_mean_deviation")
            if lt150_ratio is not None:
                score += max(0.0, float(lt150_ratio) - 0.030) * 900.0
            if retention is not None:
                score += max(0.0, float(retention) - 0.12) * 120.0
            if column_deviation is not None:
                score += max(0.0, float(column_deviation) - 3.0) * 18.0

    font_style = report.get("font_style_gate")
    if isinstance(font_style, dict):
        ratio = font_style.get("score_ratio_to_best")
        if ratio is not None:
            score += max(0.0, float(ratio) - 1.0) * 180.0

    body_issues = report.get("local_stroke_body_issues")
    if isinstance(body_issues, list) and body_issues:
        score += len(body_issues) * 260.0
    if report_needs_wider_gray_strokes(report):
        score += gray_stroke_balance_penalty(report) * 2.4

    params = report.get("params")
    if longer_replacement and isinstance(params, dict):
        blur = float(params.get("blur") or 0.0)
        score += max(0.0, blur - 0.62) * 120.0
        score += max(0.0, 0.42 - blur) * 80.0
    return score


def region_candidate_score(
    original: Image.Image,
    candidate: Image.Image,
    plan: RenderPlan,
    report: dict[str, Any],
) -> float:
    if plan.draw_mode == "center":
        return web_candidate_score(report)
    return local_score(original, candidate, plan, report) + stage_selection_penalty(report)


def old_region_lt55_pixels(img: Image.Image, roi: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = roi
    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return int(np.count_nonzero(gray[y1:y2, x1:x2] < 55))


def region_context_box(
    roi: tuple[int, int, int, int],
    image_size: tuple[int, int],
    *,
    pad_ratio: float = 0.60,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi
    pad = max(24, int(round(max(x2 - x1, y2 - y1) * pad_ratio)))
    return clamp_box((x1 - pad, y1 - pad, x2 + pad, y2 + pad), image_size)


def save_region_context(
    img: Image.Image,
    box: tuple[int, int, int, int],
    path: Path,
    *,
    scale: int = 4,
) -> None:
    crop = img.crop(box)
    if scale > 1:
        crop = crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(path)


def save_region_compare(
    original: Image.Image,
    candidate: Image.Image,
    box: tuple[int, int, int, int],
    path: Path,
    *,
    scale: int = 4,
) -> None:
    old = original.crop(box)
    new = candidate.crop(box)
    w, h = old.size
    label_h = 24
    sheet = Image.new("RGB", (w * scale * 2, h * scale + label_h), (248, 248, 248))
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 5), "original", fill=(20, 20, 20))
    draw.text((w * scale + 6, 5), "candidate", fill=(20, 20, 20))
    sheet.paste(old.resize((w * scale, h * scale), Image.Resampling.NEAREST), (0, label_h))
    sheet.paste(new.resize((w * scale, h * scale), Image.Resampling.NEAREST), (w * scale, label_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def compact_hard_reports(
    rendered: list[tuple[CandidateParams, Image.Image, dict[str, Any], float]],
    plan: RenderPlan,
) -> dict[str, Any]:
    reports: dict[str, Any] = {
        "task": {
            "source_text": plan.source_text,
            "target_text": plan.target_text,
            "search_roi": list(plan.search_roi),
            "target_roi": list(plan.target_roi),
            "draw_mode": plan.draw_mode,
            "text_angle_degrees": round(float(plan.text_angle_degrees), 3),
            "slot_boxes": [asdict(item) for item in plan.slot_boxes],
            "protected_boxes": [list(item) for item in plan.protected_boxes],
        },
        "candidates": {},
    }
    for params, _candidate, report, score in rendered:
        reports["candidates"][params.candidate_id] = {
            "label": params_label(params),
            "params": asdict(params),
            "score": round(float(score), 3),
            "hard_check": report,
        }
    return reports


def final_acceptance_delivers(acceptance: dict[str, Any]) -> bool:
    final_level = str(acceptance.get("acceptance_level", "")).strip().lower()
    final_decision = str(acceptance.get("final_decision", "")).strip().lower()
    return bool(acceptance.get("pass")) and final_level == "pass" and final_decision == "deliver"


def acceptance_blocking_stage(acceptance: dict[str, Any] | None) -> str | None:
    if not isinstance(acceptance, dict):
        return None
    stage = str(acceptance.get("blocking_stage") or "").strip()
    if stage in STAGE_PATCH_POLICY and stage != "none":
        return stage
    findings = acceptance.get("visual_findings")
    if isinstance(findings, dict):
        background = str(findings.get("background") or "").strip().lower()
        if background in {"patch_visible", "ghost_visible", "seam_visible", "too_smooth"}:
            return "background_cleanup"
        sharpness = str(findings.get("sharpness") or "").strip().lower()
        if sharpness in {"too_sharp", "too_blurry"}:
            return "photo_texture"
        darkness = str(findings.get("darkness") or "").strip().lower()
        stroke_weight = str(findings.get("stroke_weight") or "").strip().lower()
        if darkness in {"too_dark", "too_light"} or stroke_weight in {"too_bold", "too_thin", "slightly_bold"}:
            return "ink_gray_balance"
        text_shape_values = {
            str(findings.get("char_positions") or "").strip().lower(),
            str(findings.get("spacing") or "").strip().lower(),
            str(findings.get("baseline") or "").strip().lower(),
            str(findings.get("font_similarity") or "").strip().lower(),
            str(findings.get("size") or "").strip().lower(),
        }
        if any(value and value not in {"ok", "pass"} for value in text_shape_values):
            return "text_shape"
    return None


def effective_blocking_stage(
    report: dict[str, Any] | None,
    acceptance: dict[str, Any] | None,
) -> tuple[str | None, bool]:
    local_stage = None
    if isinstance(report, dict):
        local_stage = (stage_gate_for_report(report) or {}).get("blocking_stage")
    if local_stage:
        return str(local_stage), True
    visual_stage = acceptance_blocking_stage(acceptance)
    return visual_stage, False


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


def patch_signature(patch: dict[str, Any]) -> str:
    rounded: dict[str, Any] = {}
    for key, value in patch.items():
        if isinstance(value, float):
            rounded[key] = round(value, 4)
        else:
            rounded[key] = value
    return json.dumps(rounded, ensure_ascii=False, sort_keys=True)


def params_signature(params: CandidateParams) -> str:
    data = asdict(params)
    data.pop("candidate_id", None)
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def dedupe_patches(patches: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for patch in patches:
        clean = {
            key: value
            for key, value in patch.items()
            if value is not None and not (isinstance(value, float) and abs(value) < 0.0001)
        }
        if not clean:
            continue
        key = patch_signature(clean)
        if key in seen:
            continue
        seen.add(key)
        unique.append(clean)
        if len(unique) >= limit:
            break
    return unique


def delta_patch_for_target(params: CandidateParams, name: str, target: float) -> dict[str, Any] | None:
    mapping = {
        "opacity": "opacity_delta",
        "blur": "blur_delta",
        "stroke_opacity": "stroke_opacity_delta",
        "ink_gain": "ink_gain_delta",
        "alpha_contrast": "alpha_contrast_delta",
        "core_ink_gain": "core_ink_gain_delta",
        "core_darken_strength": "core_darken_strength_delta",
        "core_darken_threshold": "core_darken_threshold_delta",
        "core_darken_target_gray": "core_darken_target_gray_delta",
        "photo_warp": "photo_warp_delta",
        "edge_breakup": "edge_breakup_delta",
        "photo_noise": "photo_noise_delta",
        "jpeg_quality": "jpeg_quality_delta",
        "text_dx": "text_dx_delta",
        "text_dy": "text_dy_delta",
    }
    if name == "font_size":
        delta = int(round(target - float(params.font_size)))
        return {"font_size_delta": delta} if delta else None
    if name in {"text_dx", "text_dy"}:
        delta = int(round(target - float(getattr(params, name))))
        return {mapping[name]: delta} if delta else None
    if name not in mapping:
        return None
    current = float(getattr(params, name))
    delta = float(target) - current
    if abs(delta) < 0.005:
        return None
    if name in {"core_darken_threshold", "core_darken_target_gray"}:
        rounded_delta: float | int = int(round(delta))
        if not rounded_delta:
            return None
    else:
        rounded_delta = round(delta, 4)
    return {mapping[name]: rounded_delta}


def patch_from_parameter_suggestion(
    params: CandidateParams,
    suggestion: dict[str, Any],
) -> dict[str, Any] | None:
    name = str(suggestion.get("name") or suggestion.get("parameter") or "").strip()
    if not name:
        return None
    if "to" in suggestion:
        try:
            return delta_patch_for_target(params, name, float(suggestion["to"]))
        except (TypeError, ValueError):
            return None

    delta_key = f"{name}_delta"
    delta = suggestion.get("delta", suggestion.get(delta_key))
    if delta is None:
        return None
    mapping = {
        "font_size": "font_size_delta",
        "opacity": "opacity_delta",
        "blur": "blur_delta",
        "stroke_opacity": "stroke_opacity_delta",
        "ink_gain": "ink_gain_delta",
        "alpha_contrast": "alpha_contrast_delta",
        "core_ink_gain": "core_ink_gain_delta",
        "core_darken_strength": "core_darken_strength_delta",
        "core_darken_threshold": "core_darken_threshold_delta",
        "core_darken_target_gray": "core_darken_target_gray_delta",
        "photo_warp": "photo_warp_delta",
        "edge_breakup": "edge_breakup_delta",
        "photo_noise": "photo_noise_delta",
        "jpeg_quality": "jpeg_quality_delta",
        "text_dx": "text_dx_delta",
        "text_dy": "text_dy_delta",
    }
    patch_key = mapping.get(name)
    if not patch_key:
        return None
    try:
        if name in {"font_size", "core_darken_threshold", "core_darken_target_gray", "jpeg_quality", "text_dx", "text_dy"}:
            value: float | int = int(round(float(delta)))
        else:
            value = round(float(delta), 4)
    except (TypeError, ValueError):
        return None
    return {patch_key: value} if value else None


def model_patch_records(
    params: CandidateParams,
    model_json: dict[str, Any],
    *,
    source: str,
) -> list[dict[str, Any]]:
    if not isinstance(model_json, dict):
        return []
    records: list[dict[str, Any]] = []
    suggested_patch = model_json.get("suggested_patch")
    if isinstance(suggested_patch, dict):
        records.append(
            {
                "source": source,
                "kind": "suggested_patch",
                "direction": model_json.get("direction"),
                "blocking_stage": model_json.get("blocking_stage"),
                "patch": suggested_patch,
            }
        )

    suggestions = model_json.get("parameter_suggestions")
    if isinstance(suggestions, list):
        for idx, suggestion in enumerate(suggestions, start=1):
            if not isinstance(suggestion, dict):
                continue
            patch = patch_from_parameter_suggestion(params, suggestion)
            if not patch:
                continue
            records.append(
                {
                    "source": source,
                    "kind": "parameter_suggestion",
                    "index": idx,
                    "direction": model_json.get("direction"),
                    "blocking_stage": model_json.get("blocking_stage"),
                    "suggestion": suggestion,
                    "patch": patch,
                }
            )
    return records


def numeric_revision_patches(params: CandidateParams, acceptance: dict[str, Any]) -> list[dict[str, Any]]:
    patches: list[dict[str, Any]] = []
    patches.extend(
        record["patch"]
        for record in model_patch_records(params, acceptance, source="model_json")
        if isinstance(record.get("patch"), dict)
    )
    param_names = (
        "core_darken_strength",
        "core_ink_gain",
        "core_darken_threshold",
        "core_darken_target_gray",
        "photo_warp",
        "edge_breakup",
        "photo_noise",
        "jpeg_quality",
        "alpha_contrast",
        "stroke_opacity",
        "ink_gain",
        "opacity",
        "blur",
        "font_size",
        "text_dx",
        "text_dy",
    )
    for text in acceptance_text_fragments(acceptance):
        for name in param_names:
            escaped = re.escape(name)
            patterns = (
                rf"{escaped}\s*(?:[=:：]\s*|从\s*)?[0-9]+(?:\.[0-9]+)?\s*(?:->|→)\s*([0-9]+(?:\.[0-9]+)?)",
                rf"{escaped}\s*(?:从\s*)?[0-9]+(?:\.[0-9]+)?\s*(?:到|调到|降到|下调到|增到|增加到|小幅增到|小幅降到)\s*([0-9]+(?:\.[0-9]+)?)",
                rf"{escaped}\s*[=:：]\s*([0-9]+(?:\.[0-9]+)?)",
            )
            for pattern in patterns:
                match = re.search(pattern, text, flags=re.IGNORECASE)
                if not match:
                    continue
                patch = delta_patch_for_target(params, name, float(match.group(1)))
                if patch:
                    patches.append(patch)
                    break
    return dedupe_patches(patches, 8)


def thin_dark_core_patches(acceptance: dict[str, Any]) -> list[dict[str, Any]]:
    findings = acceptance.get("visual_findings") if isinstance(acceptance, dict) else {}
    if not isinstance(findings, dict):
        findings = {}
    text = "\n".join(acceptance_text_fragments(acceptance)).lower()
    stroke_weight = str(findings.get("stroke_weight", "")).strip().lower()
    darkness = str(findings.get("darkness", "")).strip().lower()
    font_size = str(findings.get("size", "")).strip().lower()
    wants_thinner = acceptance_wants_thinner_strokes(acceptance)
    wants_darker_core = acceptance_wants_darker_core(acceptance)
    if not wants_thinner and font_size != "too_large":
        return []
    if acceptance_reports_too_dark_or_bold(acceptance) and not wants_darker_core:
        return [
            {"opacity_delta": -0.04, "alpha_contrast_delta": -0.04},
            {"opacity_delta": -0.06, "alpha_contrast_delta": -0.06, "blur_delta": 0.04},
            {"opacity_delta": -0.06, "blur_delta": 0.08},
            {"font_size_delta": -1, "opacity_delta": -0.03, "blur_delta": 0.04},
        ]

    patches = [
        {
            "font_size_delta": -1,
            "blur_delta": -0.08,
            "alpha_contrast_delta": 0.20,
            "core_ink_gain_delta": -0.10,
            "core_darken_strength_delta": 0.04,
            "core_darken_threshold_delta": 10,
        },
        {
            "font_size_delta": -1,
            "blur_delta": -0.06,
            "alpha_contrast_delta": 0.25,
            "core_ink_gain_delta": -0.14,
            "core_darken_strength_delta": 0.06,
            "core_darken_threshold_delta": 16,
            "core_darken_target_gray_delta": -4,
        },
        {
            "blur_delta": -0.10,
            "alpha_contrast_delta": 0.25,
            "core_ink_gain_delta": -0.12,
            "core_darken_threshold_delta": 18,
            "core_darken_target_gray_delta": -6,
        },
        {
            "font_size_delta": -1,
            "opacity_delta": 0.02,
            "blur_delta": -0.06,
            "alpha_contrast_delta": 0.18,
            "core_ink_gain_delta": -0.12,
            "core_darken_strength_delta": 0.08,
            "core_darken_threshold_delta": 14,
        },
    ]
    if darkness == "too_dark" and not wants_darker_core:
        patches.append(
            {
                "font_size_delta": -1,
                "blur_delta": -0.06,
                "alpha_contrast_delta": 0.22,
                "core_ink_gain_delta": -0.16,
                "core_darken_strength_delta": -0.02,
                "core_darken_threshold_delta": 18,
            }
        )
    return patches


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


def report_has_background_white_ghost(report: dict[str, Any] | None) -> bool:
    return any(
        str(issue.get("type") or "") in {"background_white_ghost_residual", "background_shadow_ghost_residual"}
        for issue in stage_issues(report, "background_cleanup")
        if isinstance(issue, dict)
    )


def report_has_background_low_texture(report: dict[str, Any] | None) -> bool:
    return any(
        str(issue.get("type") or "") in {
            "background_fill_too_smooth",
            "background_fill_low_texture_variance",
            "background_trailing_patch_too_smooth",
        }
        for issue in stage_issues(report, "background_cleanup")
        if isinstance(issue, dict)
    )


def revision_selection_score(
    score: float,
    params: CandidateParams,
    basis_params: CandidateParams,
    acceptance: dict[str, Any],
    report: dict[str, Any] | None = None,
    candidate_report: dict[str, Any] | None = None,
) -> float:
    adjusted = float(score)
    adjusted += (
        alignment_vertical_penalty(candidate_report)
        - alignment_vertical_penalty(report)
    ) * 3.0
    findings = acceptance.get("visual_findings") if isinstance(acceptance, dict) else {}
    if not isinstance(findings, dict):
        findings = {}

    stage_gate = stage_gate_for_report(report) if isinstance(report, dict) else {}
    if stage_gate.get("blocking_stage") == "text_shape":
        stroke_gain = max(0.0, float(params.stroke_opacity) - float(basis_params.stroke_opacity))
        ink_gain = max(0.0, float(params.ink_gain) - float(basis_params.ink_gain))
        blur_gain = max(0.0, float(params.blur) - float(basis_params.blur))
        core_gain = max(0.0, float(params.core_ink_gain) - float(basis_params.core_ink_gain))
        darken_gain = max(0.0, float(params.core_darken_strength) - float(basis_params.core_darken_strength))
        alpha_drop = max(0.0, float(basis_params.alpha_contrast) - float(params.alpha_contrast))
        size_gain = max(0, int(params.font_size) - int(basis_params.font_size))
        adjusted += gray_stroke_balance_penalty(candidate_report) * 5.0
        candidate_stage = stage_gate_for_report(candidate_report) if isinstance(candidate_report, dict) else {}
        if candidate_stage.get("blocking_stage") == "text_shape":
            adjusted += 1200.0
        elif candidate_stage.get("blocking_stage"):
            adjusted -= 900.0
        else:
            adjusted -= 1600.0
        body_issues = local_stroke_body_issues(candidate_report or {}, allow_excess_black_core=True)
        neighbor_issues = local_neighbor_style_issues(candidate_report or {}, allow_excess_black_core=True)
        halo_issues = local_outer_gray_halo_issues(candidate_report or {}, allow_excess_black_core=True)
        adjusted += len(body_issues) * 520.0
        adjusted += len(neighbor_issues) * 460.0
        adjusted += len(halo_issues) * 420.0
        adjusted += max(0.0, float(params.stroke_opacity) - 0.12) * 6000.0
        adjusted += max(0.0, float(params.core_ink_gain) - 0.30) * 3600.0
        adjusted += max(0.0, float(params.core_darken_strength) - 0.24) * 3200.0
        adjusted += max(0.0, float(params.photo_warp) - 0.12) * 4200.0
        adjusted += max(0.0, float(params.blur) - 0.30) * 1600.0
        adjusted -= min(stroke_gain, 0.08) * 3800.0
        adjusted -= min(blur_gain, 0.12) * 420.0
        adjusted -= min(alpha_drop, 0.20) * 260.0
        adjusted -= min(size_gain, 1) * 80.0
        adjusted -= min(ink_gain, 0.04) * 160.0
        adjusted -= min(core_gain, 0.08) * 220.0
        adjusted -= min(darken_gain, 0.08) * 180.0
        return adjusted

    if stage_gate.get("blocking_stage") == "photo_texture":
        candidate_stage = stage_gate_for_report(candidate_report) if isinstance(candidate_report, dict) else {}
        if candidate_stage.get("blocking_stage") == "photo_texture":
            adjusted += 600.0
        elif candidate_stage.get("blocking_stage"):
            adjusted -= 260.0
        else:
            adjusted -= 900.0
        blur_change = abs(float(params.blur) - float(basis_params.blur))
        noise_change = abs(float(params.photo_noise) - float(basis_params.photo_noise))
        edge_change = abs(float(params.edge_breakup) - float(basis_params.edge_breakup))
        warp_change = abs(float(params.photo_warp) - float(basis_params.photo_warp))
        jpeg_change = abs(int(params.jpeg_quality) - int(basis_params.jpeg_quality))
        adjusted -= min(blur_change, 0.12) * 220.0
        adjusted -= min(noise_change, 0.025) * 1800.0
        adjusted -= min(edge_change, 0.010) * 1400.0
        adjusted -= min(warp_change, 0.030) * 280.0
        adjusted -= min(jpeg_change, 8) * 8.0
        return adjusted

    if report_has_excess_black_core(report):
        candidate_stage = stage_gate_for_report(candidate_report) if isinstance(candidate_report, dict) else {}
        candidate_block = candidate_stage.get("blocking_stage")
        if candidate_block == "ink_gray_balance":
            adjusted += 480.0
        elif candidate_block:
            adjusted += 320.0
        else:
            adjusted -= 900.0
        opacity_drop = max(0.0, float(basis_params.opacity) - float(params.opacity))
        stroke_drop = max(0.0, float(basis_params.stroke_opacity) - float(params.stroke_opacity))
        ink_drop = max(0.0, float(basis_params.ink_gain) - float(params.ink_gain))
        alpha_drop = max(0.0, float(basis_params.alpha_contrast) - float(params.alpha_contrast))
        core_drop = max(0.0, float(basis_params.core_ink_gain) - float(params.core_ink_gain))
        darken_drop = max(0.0, float(basis_params.core_darken_strength) - float(params.core_darken_strength))
        blur_increase = max(0.0, float(params.blur) - float(basis_params.blur))
        adjusted -= opacity_drop * 980.0
        adjusted -= stroke_drop * 1300.0
        adjusted -= ink_drop * 240.0
        adjusted -= alpha_drop * 260.0
        adjusted -= core_drop * 180.0
        adjusted -= darken_drop * 160.0
        adjusted -= blur_increase * 90.0
        return adjusted

    darkness = str(findings.get("darkness", "")).strip().lower()
    stroke_weight = str(findings.get("stroke_weight", "")).strip().lower()
    if darkness == "too_dark" or stroke_weight in {"too_bold", "slightly_bold"}:
        opacity_drop = max(0.0, float(basis_params.opacity) - float(params.opacity))
        ink_drop = max(0.0, float(basis_params.ink_gain) - float(params.ink_gain))
        core_drop = max(0.0, float(basis_params.core_ink_gain) - float(params.core_ink_gain))
        blur_increase = max(0.0, float(params.blur) - float(basis_params.blur))
        adjusted -= opacity_drop * 320.0
        adjusted -= ink_drop * 160.0
        adjusted -= core_drop * 120.0
        adjusted -= blur_increase * 70.0
        return adjusted

    if report_needs_wider_gray_strokes(report):
        opacity_drop = max(0.0, float(basis_params.opacity) - float(params.opacity))
        stroke_gain = max(0.0, float(params.stroke_opacity) - float(basis_params.stroke_opacity))
        stroke_drop = max(0.0, float(basis_params.stroke_opacity) - float(params.stroke_opacity))
        ink_gain = max(0.0, float(params.ink_gain) - float(basis_params.ink_gain))
        core_gain = max(0.0, float(params.core_ink_gain) - float(basis_params.core_ink_gain))
        darken_gain = max(0.0, float(params.core_darken_strength) - float(basis_params.core_darken_strength))
        blur_drop = max(0.0, float(basis_params.blur) - float(params.blur))
        blur_gain = max(0.0, float(params.blur) - float(basis_params.blur))
        alpha_gain = max(0.0, float(params.alpha_contrast) - float(basis_params.alpha_contrast))
        size_gain = max(0, int(params.font_size) - int(basis_params.font_size))
        photo_gain = max(0.0, float(params.photo_noise) - float(basis_params.photo_noise))
        photo_drop = max(0.0, float(basis_params.photo_noise) - float(params.photo_noise))
        edge_drop = max(0.0, float(basis_params.edge_breakup) - float(params.edge_breakup))
        neighbor_style_issue = bool(local_neighbor_style_issues(report))
        if report_has_outer_gray_halo(report):
            if local_outer_gray_halo_issues(candidate_report, allow_excess_black_core=True):
                adjusted += 260.0
            if local_stroke_body_issues(candidate_report):
                adjusted += 140.0
            adjusted -= blur_drop * 520.0
            adjusted -= stroke_drop * 2300.0
            adjusted -= photo_drop * 1100.0
            adjusted -= edge_drop * 1400.0
            adjusted -= alpha_gain * 260.0
            adjusted -= core_gain * 180.0
            adjusted -= darken_gain * 160.0
            adjusted += blur_gain * 520.0
            adjusted += photo_gain * 900.0
            adjusted += stroke_gain * 1500.0
            adjusted += opacity_drop * 600.0
            return adjusted
        adjusted += opacity_drop * 900.0
        adjusted += gray_stroke_balance_penalty(candidate_report) * 6.0
        if local_stroke_body_issues(candidate_report):
            adjusted += 160.0
        if neighbor_style_issue and local_neighbor_style_issues(candidate_report):
            adjusted += 190.0
        if stroke_gain <= 0.0:
            adjusted += 190.0
        if stroke_gain <= 0.0 and ink_gain <= 0.0 and blur_gain <= 0.0 and photo_gain <= 0.0:
            adjusted += 240.0
        adjusted -= min(stroke_gain, 0.07) * 4600.0
        adjusted += max(0.0, stroke_gain - 0.08) * 1400.0
        adjusted -= blur_gain * 95.0
        adjusted -= ink_gain * 150.0
        adjusted -= photo_gain * 170.0
        if neighbor_style_issue:
            adjusted -= core_gain * 420.0
            adjusted -= darken_gain * 320.0
            adjusted += blur_gain * 180.0
            adjusted += photo_gain * 160.0
        adjusted -= min(size_gain, 1) * 70.0
        return adjusted

    if not report_needs_thinner_strokes(report):
        return adjusted
    if acceptance_reports_too_dark_or_bold(acceptance):
        opacity_drop = max(0.0, float(basis_params.opacity) - float(params.opacity))
        opacity_raise = max(0.0, float(params.opacity) - float(basis_params.opacity))
        blur_increase = max(0.0, float(params.blur) - float(basis_params.blur))
        alpha_drop = max(0.0, float(basis_params.alpha_contrast) - float(params.alpha_contrast))
        core_drop = max(0.0, float(basis_params.core_ink_gain) - float(params.core_ink_gain))
        darken_drop = max(0.0, float(basis_params.core_darken_strength) - float(params.core_darken_strength))

        adjusted -= opacity_drop * 1800.0
        adjusted -= blur_increase * 260.0
        adjusted -= alpha_drop * 320.0
        adjusted -= core_drop * 180.0
        adjusted -= darken_drop * 160.0
        adjusted += opacity_raise * 2600.0
        return adjusted

    opacity_drop = max(0.0, float(basis_params.opacity) - float(params.opacity))
    blur_increase = max(0.0, float(params.blur) - float(basis_params.blur))
    alpha_contrast_gain = max(0.0, float(params.alpha_contrast) - float(basis_params.alpha_contrast))
    font_size_drop = max(0, int(basis_params.font_size) - int(params.font_size))
    threshold_gain = max(0, int(params.core_darken_threshold) - int(basis_params.core_darken_threshold))

    adjusted += opacity_drop * 1500.0
    adjusted += blur_increase * 260.0
    adjusted -= alpha_contrast_gain * 180.0
    adjusted -= min(font_size_drop, 1) * 55.0
    adjusted -= min(threshold_gain, 20) * 2.5
    return adjusted


def constrained_revision_params(
    params: CandidateParams,
    basis_params: CandidateParams,
    acceptance: dict[str, Any],
    report: dict[str, Any] | None = None,
    *,
    round_idx: int,
) -> CandidateParams:
    stage_gate = stage_gate_for_report(report) if isinstance(report, dict) else {}
    if stage_gate.get("blocking_stage") == "text_shape":
        has_outer_halo = report_has_outer_gray_halo(report)
        stroke_cap = 0.10 if has_outer_halo else 0.14
        blur_cap = 0.22 if has_outer_halo else 0.30
        return mutate_params(
            params,
            opacity=max(0.82, min(1.0, params.opacity)),
            blur=max(0.08, min(blur_cap, params.blur)),
            stroke_opacity=min(stroke_cap, params.stroke_opacity),
            alpha_contrast=min(0.35, params.alpha_contrast),
            photo_warp=min(0.12, params.photo_warp),
            edge_breakup=min(0.012, params.edge_breakup),
            photo_noise=min(0.030, params.photo_noise),
            core_ink_gain=min(0.30, params.core_ink_gain),
            core_darken_strength=min(0.24, params.core_darken_strength),
            core_darken_threshold=min(150, params.core_darken_threshold),
        )

    if report_has_excess_black_core(report):
        opacity_floor = opacity_floor_for_excess_black(report)
        return mutate_params(
            params,
            opacity=max(opacity_floor, min(1.0, params.opacity)),
            blur=max(0.18, min(0.65, params.blur)),
            stroke_opacity=min(0.06, params.stroke_opacity),
            alpha_contrast=min(0.35, params.alpha_contrast),
            core_ink_gain=min(0.22, params.core_ink_gain),
            core_darken_strength=min(0.18, params.core_darken_strength),
        )

    if report_has_background_white_ghost(report):
        return mutate_params(
            params,
            photo_noise=max(0.0, min(0.038, params.photo_noise)),
            edge_breakup=max(0.0, min(0.014, params.edge_breakup)),
            jpeg_quality=max(94, params.jpeg_quality),
            mask_threshold=max(params.mask_threshold, min(215, basis_params.mask_threshold + 12)),
            mask_dilate_iterations=max(3, min(5, params.mask_dilate_iterations)),
            inpaint_radius=max(2, min(3, params.inpaint_radius)),
        )

    if report_has_background_low_texture(report):
        return mutate_params(
            params,
            photo_noise=min(0.14, params.photo_noise),
            edge_breakup=min(0.060, params.edge_breakup),
            jpeg_quality=max(82, params.jpeg_quality),
            mask_dilate_iterations=max(2, params.mask_dilate_iterations),
            inpaint_radius=max(1, min(3, params.inpaint_radius)),
        )

    if acceptance_reports_background_patch(acceptance):
        return mutate_params(
            params,
            photo_noise=min(0.120, params.photo_noise),
            edge_breakup=min(0.050, params.edge_breakup),
            jpeg_quality=max(82, params.jpeg_quality),
            mask_dilate_iterations=max(2, params.mask_dilate_iterations),
            inpaint_radius=max(1, min(3, params.inpaint_radius)),
        )

    if not report_needs_thinner_strokes(report):
        if not report_needs_wider_gray_strokes(report):
            return params
        if report_has_outer_gray_halo(report):
            return mutate_params(
                params,
                opacity=max(max(0.82, basis_params.opacity - 0.02), params.opacity),
                blur=min(params.blur, max(0.08, basis_params.blur - 0.03)),
                stroke_opacity=min(params.stroke_opacity, max(0.0, basis_params.stroke_opacity - 0.01)),
                alpha_contrast=min(0.45, params.alpha_contrast),
                photo_warp=min(params.photo_warp, max(0.0, basis_params.photo_warp - 0.02)),
                edge_breakup=min(params.edge_breakup, max(0.0, basis_params.edge_breakup - 0.006)),
                photo_noise=min(params.photo_noise, max(0.0, basis_params.photo_noise - 0.018)),
                core_ink_gain=max(params.core_ink_gain, basis_params.core_ink_gain),
                core_darken_strength=max(params.core_darken_strength, basis_params.core_darken_strength),
            )
        if local_neighbor_style_issues(report):
            return mutate_params(
                params,
                opacity=max(max(0.82, basis_params.opacity - 0.04), params.opacity),
                blur=max(max(0.14, basis_params.blur - 0.08), params.blur),
                stroke_opacity=max(min(0.16, basis_params.stroke_opacity + 0.02), params.stroke_opacity),
                alpha_contrast=min(0.45, params.alpha_contrast),
                photo_warp=max(max(0.0, basis_params.photo_warp - 0.04), params.photo_warp),
                edge_breakup=max(max(0.0, basis_params.edge_breakup - 0.010), params.edge_breakup),
                photo_noise=max(max(0.0, basis_params.photo_noise - 0.020), params.photo_noise),
                core_ink_gain=min(0.34, params.core_ink_gain),
                core_darken_strength=min(0.30, params.core_darken_strength),
            )
        if report_has_fine_strokes_too_soft(report):
            return mutate_params(
                params,
                opacity=max(max(0.82, basis_params.opacity - 0.05), params.opacity),
                blur=max(max(0.18, basis_params.blur - 0.08), params.blur),
                stroke_opacity=max(min(0.16, basis_params.stroke_opacity + 0.03), params.stroke_opacity),
                alpha_contrast=min(0.45, params.alpha_contrast),
                photo_warp=max(max(0.0, basis_params.photo_warp - 0.04), params.photo_warp),
                edge_breakup=max(max(0.0, basis_params.edge_breakup - 0.010), params.edge_breakup),
                photo_noise=max(max(0.0, basis_params.photo_noise - 0.020), params.photo_noise),
                core_ink_gain=min(max(0.0, basis_params.core_ink_gain), params.core_ink_gain),
                core_darken_strength=min(max(0.0, basis_params.core_darken_strength), params.core_darken_strength),
            )
        return mutate_params(
            params,
            opacity=max(max(0.82, basis_params.opacity - 0.04), params.opacity),
            blur=max(max(0.12, basis_params.blur), params.blur),
            alpha_contrast=min(0.45, params.alpha_contrast),
            core_ink_gain=min(max(0.0, basis_params.core_ink_gain), params.core_ink_gain),
            core_darken_strength=min(max(0.0, basis_params.core_darken_strength), params.core_darken_strength),
        )

    alpha_cap = 0.35
    threshold_cap = 166
    target_gray_floor = 20
    blur_floor = 0.08
    opacity_floor = 0.92
    core_darken_cap = 0.46
    if acceptance_reports_too_dark_or_bold(acceptance):
        return mutate_params(
            params,
            opacity=max(0.70, min(basis_params.opacity, params.opacity)),
            blur=max(0.08, min(0.65, params.blur)),
            stroke_opacity=min(basis_params.stroke_opacity, params.stroke_opacity),
            alpha_contrast=min(basis_params.alpha_contrast, params.alpha_contrast),
            core_ink_gain=min(basis_params.core_ink_gain, params.core_ink_gain),
            core_darken_strength=min(basis_params.core_darken_strength, params.core_darken_strength),
        )

    if round_idx >= 3:
        # Once the stroke footprint has been narrowed, later rounds may only
        # soften or tone it; they must not keep hardening the glyph core.
        alpha_cap = min(alpha_cap, max(0.20, basis_params.alpha_contrast))
        threshold_cap = min(threshold_cap, max(140, basis_params.core_darken_threshold + 4))

    return mutate_params(
        params,
        opacity=max(opacity_floor, params.opacity),
        blur=max(blur_floor, params.blur),
        alpha_contrast=min(alpha_cap, params.alpha_contrast),
        core_darken_strength=min(core_darken_cap, params.core_darken_strength),
        core_darken_threshold=min(threshold_cap, params.core_darken_threshold),
        core_darken_target_gray=max(target_gray_floor, params.core_darken_target_gray),
    )


def report_needs_thinner_strokes(report: dict[str, Any] | None) -> bool:
    if not isinstance(report, dict):
        return False
    char_bands = report.get("char_gray_band_metrics")
    if isinstance(char_bands, dict) and char_bands.get("enabled"):
        for item in char_bands.get("per_char", []):
            if not isinstance(item, dict):
                continue
            source_char = item.get("source_char")
            target_char = item.get("target_char")
            if source_char == target_char:
                continue
            delta = item.get("delta") or {}
            old = item.get("old") or {}
            try:
                lt165_delta = float(delta.get("lt165") or 0.0)
                lt55_delta = float(delta.get("lt55") or 0.0)
                old_lt165 = float(old.get("lt165") or 0.0)
            except (TypeError, ValueError):
                continue
            if old_lt165 > 0 and lt165_delta > max(28.0, old_lt165 * 0.12) and lt55_delta > 70.0:
                return True
    return False


def gray_stroke_balance_penalty(report: dict[str, Any] | None) -> float:
    if not isinstance(report, dict):
        return 0.0
    char_bands = report.get("char_gray_band_metrics")
    if not isinstance(char_bands, dict) or not char_bands.get("enabled"):
        return 0.0

    penalty = 0.0
    for item in char_bands.get("per_char", []):
        if not isinstance(item, dict):
            continue
        if item.get("source_char") == item.get("target_char"):
            continue
        delta = item.get("delta") or {}
        try:
            lt55_delta = float(delta.get("lt55") or 0.0)
            lt165_delta = float(delta.get("lt165") or 0.0)
            band_70_90_delta = float(delta.get("band_70_90") or 0.0)
            band_90_120_delta = float(delta.get("band_90_120") or 0.0)
        except (TypeError, ValueError):
            continue
        penalty += max(0.0, -lt165_delta) * 2.2
        penalty += max(0.0, -band_70_90_delta) * 1.4
        penalty += max(0.0, -band_90_120_delta) * 1.0
        penalty += max(0.0, lt55_delta - 45.0) * 0.8
    return penalty


def report_needs_wider_gray_strokes(report: dict[str, Any] | None) -> bool:
    if not isinstance(report, dict):
        return False
    if report_has_excess_black_core(report):
        return False
    if local_stroke_body_issues(report):
        return True
    if local_neighbor_style_issues(report):
        return True
    char_bands = report.get("char_gray_band_metrics")
    if not isinstance(char_bands, dict) or not char_bands.get("enabled"):
        return False

    for item in char_bands.get("per_char", []):
        if not isinstance(item, dict):
            continue
        if item.get("source_char") == item.get("target_char"):
            continue
        delta = item.get("delta") or {}
        try:
            lt55_delta = float(delta.get("lt55") or 0.0)
            lt165_delta = float(delta.get("lt165") or 0.0)
            band_70_90_delta = float(delta.get("band_70_90") or 0.0)
            band_90_120_delta = float(delta.get("band_90_120") or 0.0)
        except (TypeError, ValueError):
            continue
        middle_deficit = band_70_90_delta + band_90_120_delta
        if lt165_delta < -6.0:
            return True
        if lt55_delta > 35.0 and middle_deficit < -32.0:
            return True
    return False


def report_has_excess_black_core(report: dict[str, Any] | None) -> bool:
    if not isinstance(report, dict):
        return False
    neighbor_core_low = bool(local_neighbor_style_issues(report, allow_excess_black_core=True))
    local_issues = report.get("local_ink_balance_issues")
    if isinstance(local_issues, list):
        for issue in local_issues:
            if not isinstance(issue, dict):
                continue
            issue_type = str(issue.get("type") or "")
            if "too_black" in issue_type or issue_type == "roi_core_too_black":
                return not neighbor_core_low
        return False

    if neighbor_core_low:
        return False

    char_bands = report.get("char_gray_band_metrics")
    if not isinstance(char_bands, dict) or not char_bands.get("enabled"):
        return False
    for item in char_bands.get("per_char", []):
        if not isinstance(item, dict):
            continue
        if item.get("source_char") == item.get("target_char"):
            continue
        delta = item.get("delta") or {}
        try:
            lt55_delta = float(delta.get("lt55") or 0.0)
            middle_delta = float(delta.get("band_70_90") or 0.0) + float(delta.get("band_90_120") or 0.0)
        except (TypeError, ValueError):
            continue
        if lt55_delta > 62.0 and middle_delta < -18.0:
            return True
    return False


def black_core_reduction_patches() -> list[dict[str, Any]]:
    return [
        {"opacity_delta": -0.06, "stroke_opacity_delta": -0.02, "alpha_contrast_delta": -0.05},
        {"opacity_delta": -0.06, "stroke_opacity_delta": -0.04, "alpha_contrast_delta": -0.08},
        {"opacity_delta": -0.06, "blur_delta": -0.02, "stroke_opacity_delta": -0.04},
        {"opacity_delta": -0.06, "blur_delta": 0.02, "stroke_opacity_delta": -0.02},
        {"opacity_delta": -0.04, "blur_delta": 0.06, "core_ink_gain_delta": -0.03},
        {"opacity_delta": -0.06, "blur_delta": 0.08, "core_darken_strength_delta": -0.03},
        {"core_ink_gain_delta": -0.06, "core_darken_strength_delta": -0.04},
        {"core_darken_strength_delta": -0.06, "blur_delta": 0.04},
        {"ink_gain_delta": -0.04, "core_ink_gain_delta": -0.04, "blur_delta": 0.06},
        {"opacity_delta": -0.03, "core_ink_gain_delta": -0.05, "photo_noise_delta": 0.018},
        {
            "core_ink_gain_delta": -0.08,
            "core_darken_strength_delta": -0.06,
            "edge_breakup_delta": 0.008,
            "photo_noise_delta": 0.020,
            "jpeg_quality_delta": -4,
        },
    ]


def alignment_centering_patches(report: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(report, dict):
        return []
    alignment = report.get("char_alignment_metrics")
    if not isinstance(alignment, dict) or not alignment.get("enabled"):
        return []
    center_dys: list[float] = []
    for item in alignment.get("per_char", []):
        if not isinstance(item, dict) or not item.get("candidate_box"):
            continue
        try:
            center_dys.append(float(item.get("center_dy") or 0.0))
        except (TypeError, ValueError):
            continue
    if not center_dys:
        return []
    mean_center_dy = sum(center_dys) / len(center_dys)
    if mean_center_dy < -0.25:
        return [{"text_dy_delta": 1}]
    if mean_center_dy > 0.90:
        return [{"text_dy_delta": -1}]
    return []


def neighbor_outer_gray_cleanup_patches() -> list[dict[str, Any]]:
    return [
        {
            "alpha_contrast_delta": 0.16,
            "blur_delta": -0.06,
            "stroke_opacity_delta": -0.04,
            "core_ink_gain_delta": 0.02,
            "core_darken_strength_delta": 0.03,
            "edge_breakup_delta": -0.010,
            "photo_noise_delta": -0.030,
            "jpeg_quality_delta": 8,
        },
        {
            "alpha_contrast_delta": 0.12,
            "blur_delta": -0.05,
            "stroke_opacity_delta": -0.03,
            "core_ink_gain_delta": 0.02,
            "core_darken_strength_delta": 0.02,
            "edge_breakup_delta": -0.008,
            "photo_noise_delta": -0.025,
            "jpeg_quality_delta": 6,
        },
        {
            "alpha_contrast_delta": 0.20,
            "blur_delta": -0.07,
            "stroke_opacity_delta": -0.05,
            "opacity_delta": 0.01,
            "core_ink_gain_delta": 0.02,
            "edge_breakup_delta": -0.010,
            "photo_noise_delta": -0.035,
            "jpeg_quality_delta": 8,
        },
        {
            "blur_delta": -0.07,
            "stroke_opacity_delta": -0.03,
            "core_ink_gain_delta": 0.02,
            "core_darken_strength_delta": 0.02,
            "edge_breakup_delta": -0.008,
            "photo_noise_delta": -0.030,
            "jpeg_quality_delta": 6,
        },
        {
            "blur_delta": -0.05,
            "stroke_opacity_delta": -0.04,
            "opacity_delta": 0.01,
            "core_ink_gain_delta": 0.02,
            "edge_breakup_delta": -0.008,
            "photo_noise_delta": -0.025,
            "jpeg_quality_delta": 6,
        },
        {
            "blur_delta": -0.09,
            "core_darken_strength_delta": 0.03,
            "edge_breakup_delta": -0.010,
            "photo_noise_delta": -0.035,
            "jpeg_quality_delta": 8,
        },
        {
            "blur_delta": -0.06,
            "stroke_opacity_delta": -0.02,
            "ink_gain_delta": -0.02,
            "core_ink_gain_delta": 0.03,
            "core_darken_strength_delta": 0.02,
            "edge_breakup_delta": -0.008,
            "photo_noise_delta": -0.025,
            "jpeg_quality_delta": 6,
        },
    ]


def neighbor_core_density_recovery_patches() -> list[dict[str, Any]]:
    return [
        {
            "stroke_opacity_delta": 0.03,
            "blur_delta": -0.04,
            "core_ink_gain_delta": 0.04,
            "core_darken_strength_delta": 0.04,
            "core_darken_threshold_delta": 8,
            "photo_warp_delta": -0.02,
            "edge_breakup_delta": -0.004,
            "photo_noise_delta": -0.012,
            "jpeg_quality_delta": 4,
        },
        {
            "stroke_opacity_delta": 0.04,
            "opacity_delta": 0.02,
            "blur_delta": -0.04,
            "core_ink_gain_delta": 0.03,
            "core_darken_strength_delta": 0.03,
            "photo_warp_delta": -0.02,
            "edge_breakup_delta": -0.004,
            "photo_noise_delta": -0.012,
            "jpeg_quality_delta": 4,
        },
        {
            "stroke_opacity_delta": 0.05,
            "blur_delta": -0.03,
            "ink_gain_delta": 0.01,
            "core_ink_gain_delta": 0.02,
            "core_darken_strength_delta": 0.02,
            "photo_warp_delta": -0.02,
            "edge_breakup_delta": -0.004,
            "photo_noise_delta": -0.010,
            "jpeg_quality_delta": 4,
        },
        {
            "stroke_opacity_delta": 0.03,
            "font_size_delta": 1,
            "blur_delta": -0.04,
            "core_ink_gain_delta": 0.03,
            "core_darken_strength_delta": 0.03,
            "photo_warp_delta": -0.02,
            "edge_breakup_delta": -0.004,
            "photo_noise_delta": -0.012,
            "jpeg_quality_delta": 4,
        },
    ]


def gray_stroke_recovery_patches() -> list[dict[str, Any]]:
    return [
        {
            "stroke_opacity_delta": 0.03,
            "blur_delta": -0.04,
            "ink_gain_delta": -0.02,
            "core_ink_gain_delta": -0.02,
            "photo_warp_delta": -0.02,
            "edge_breakup_delta": -0.004,
            "photo_noise_delta": -0.012,
            "jpeg_quality_delta": 4,
        },
        {
            "stroke_opacity_delta": 0.04,
            "blur_delta": -0.05,
            "ink_gain_delta": -0.03,
            "core_ink_gain_delta": -0.03,
            "photo_warp_delta": -0.02,
            "edge_breakup_delta": -0.006,
            "photo_noise_delta": -0.014,
            "jpeg_quality_delta": 4,
        },
        {
            "stroke_opacity_delta": 0.06,
            "opacity_delta": -0.02,
            "blur_delta": 0.03,
            "ink_gain_delta": 0.02,
            "core_ink_gain_delta": -0.05,
            "core_darken_strength_delta": -0.05,
            "photo_warp_delta": 0.04,
            "edge_breakup_delta": 0.006,
            "photo_noise_delta": 0.016,
            "jpeg_quality_delta": -4,
        },
        {
            "stroke_opacity_delta": 0.08,
            "opacity_delta": -0.03,
            "blur_delta": 0.02,
            "ink_gain_delta": 0.02,
            "core_ink_gain_delta": -0.06,
            "core_darken_strength_delta": -0.06,
            "photo_warp_delta": 0.04,
            "edge_breakup_delta": 0.006,
            "photo_noise_delta": 0.016,
            "jpeg_quality_delta": -4,
        },
        {
            "stroke_opacity_delta": 0.10,
            "blur_delta": 0.06,
            "ink_gain_delta": 0.04,
            "alpha_contrast_delta": -0.06,
            "core_ink_gain_delta": -0.05,
            "core_darken_strength_delta": -0.04,
            "photo_warp_delta": 0.06,
            "edge_breakup_delta": 0.012,
            "photo_noise_delta": 0.030,
            "jpeg_quality_delta": -8,
        },
        {
            "stroke_opacity_delta": 0.12,
            "font_size_delta": 1,
            "blur_delta": 0.07,
            "ink_gain_delta": 0.03,
            "alpha_contrast_delta": -0.08,
            "core_ink_gain_delta": -0.07,
            "core_darken_strength_delta": -0.05,
            "photo_warp_delta": 0.08,
            "edge_breakup_delta": 0.014,
            "photo_noise_delta": 0.035,
            "jpeg_quality_delta": -10,
        },
        {
            "stroke_opacity_delta": 0.05,
            "blur_delta": 0.08,
            "ink_gain_delta": 0.04,
            "core_ink_gain_delta": -0.04,
            "core_darken_strength_delta": -0.04,
            "photo_warp_delta": 0.06,
            "edge_breakup_delta": 0.010,
            "photo_noise_delta": 0.025,
            "jpeg_quality_delta": -6,
        },
        {
            "stroke_opacity_delta": 0.06,
            "font_size_delta": 1,
            "blur_delta": 0.06,
            "ink_gain_delta": 0.03,
            "core_ink_gain_delta": -0.06,
            "core_darken_strength_delta": -0.04,
            "photo_warp_delta": 0.08,
            "edge_breakup_delta": 0.012,
            "photo_noise_delta": 0.030,
            "jpeg_quality_delta": -8,
        },
        {
            "opacity_delta": 0.03,
            "blur_delta": 0.08,
            "ink_gain_delta": 0.03,
            "core_darken_strength_delta": -0.06,
            "photo_warp_delta": 0.05,
            "edge_breakup_delta": 0.010,
            "photo_noise_delta": 0.025,
            "jpeg_quality_delta": -6,
        },
        {
            "blur_delta": 0.12,
            "ink_gain_delta": 0.06,
            "alpha_contrast_delta": -0.10,
            "core_ink_gain_delta": -0.08,
            "photo_warp_delta": 0.08,
            "edge_breakup_delta": 0.014,
            "photo_noise_delta": 0.035,
            "jpeg_quality_delta": -10,
        },
    ]


def photo_texture_recovery_patches(report: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    issue_types = {
        str(issue.get("type") or "")
        for issue in stage_issues(report, "photo_texture")
        if isinstance(issue, dict)
    }
    if "photo_texture_too_blurry" in issue_types:
        return [
            {
                "blur_delta": -0.08,
                "photo_noise_delta": -0.006,
                "edge_breakup_delta": -0.002,
                "jpeg_quality_delta": 4,
            },
            {
                "blur_delta": -0.12,
                "alpha_contrast_delta": 0.08,
                "photo_warp_delta": -0.010,
                "jpeg_quality_delta": 6,
            },
            {
                "blur_delta": -0.06,
                "photo_noise_delta": -0.004,
                "jpeg_quality_delta": 4,
            },
        ]
    return [
        {
            "blur_delta": 0.08,
            "edge_breakup_delta": 0.006,
            "photo_noise_delta": 0.014,
            "jpeg_quality_delta": -6,
        },
        {
            "blur_delta": 0.06,
            "photo_warp_delta": 0.020,
            "edge_breakup_delta": 0.004,
            "photo_noise_delta": 0.010,
            "jpeg_quality_delta": -4,
        },
        {
            "blur_delta": 0.10,
            "photo_noise_delta": 0.018,
            "jpeg_quality_delta": -8,
        },
        {
            "edge_breakup_delta": 0.008,
            "photo_noise_delta": 0.012,
            "jpeg_quality_delta": -6,
        },
        {
            "photo_warp_delta": 0.025,
            "edge_breakup_delta": 0.006,
            "photo_noise_delta": 0.010,
        },
    ]


def background_cleanup_recovery_patches(report: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    issue_types = {
        str(issue.get("type") or "")
        for issue in stage_issues(report, "background_cleanup")
        if isinstance(issue, dict)
    }
    patches: list[dict[str, Any]] = []
    has_white_ghost = "background_white_ghost_residual" in issue_types
    if has_white_ghost:
        patches.extend(
            [
                {
                    "mask_threshold_delta": 12,
                    "mask_dilate_iterations_delta": 1,
                    "inpaint_radius_delta": 1,
                    "photo_noise_delta": -0.018,
                    "edge_breakup_delta": -0.010,
                    "jpeg_quality_delta": 8,
                },
                {
                    "mask_threshold_delta": 20,
                    "mask_dilate_iterations_delta": 1,
                    "inpaint_radius_delta": 1,
                    "photo_noise_delta": -0.024,
                    "edge_breakup_delta": -0.012,
                    "jpeg_quality_delta": 10,
                },
                {
                    "mask_threshold_delta": 8,
                    "mask_dilate_iterations_delta": 1,
                    "inpaint_radius_delta": 1,
                    "photo_noise_delta": -0.012,
                    "edge_breakup_delta": -0.008,
                    "jpeg_quality_delta": 6,
                },
                {
                    "mask_dilate_iterations_delta": 1,
                    "inpaint_radius_delta": 1,
                    "photo_noise_delta": -0.020,
                    "edge_breakup_delta": -0.010,
                    "jpeg_quality_delta": 8,
                },
            ]
        )
    if "background_fill_luminance_mismatch" in issue_types and not has_white_ghost:
        patches.extend(
            [
                {"mask_threshold_delta": -10, "inpaint_radius_delta": -1, "photo_noise_delta": 0.012},
                {"mask_threshold_delta": 10, "inpaint_radius_delta": 1, "photo_noise_delta": 0.010},
                {"inpaint_radius_delta": -1, "edge_breakup_delta": 0.006, "photo_noise_delta": 0.016},
            ]
        )
    if (
        not has_white_ghost
        and (
            "background_fill_too_smooth" in issue_types
            or "background_fill_low_texture_variance" in issue_types
            or "background_trailing_patch_too_smooth" in issue_types
        )
    ):
        patches.extend(
            [
                {"photo_noise_delta": 0.020, "edge_breakup_delta": 0.006, "jpeg_quality_delta": -6},
                {"photo_warp_delta": 0.020, "photo_noise_delta": 0.014, "jpeg_quality_delta": -4},
                {"mask_dilate_iterations_delta": -1, "photo_noise_delta": 0.018, "edge_breakup_delta": 0.006},
                {"photo_noise_delta": 0.032, "edge_breakup_delta": 0.010, "jpeg_quality_delta": -8},
                {"inpaint_radius_delta": -1, "photo_noise_delta": 0.028, "edge_breakup_delta": 0.010},
                {"photo_noise_delta": 0.052, "edge_breakup_delta": 0.018, "jpeg_quality_delta": -10},
                {"photo_noise_delta": 0.070, "edge_breakup_delta": 0.024, "jpeg_quality_delta": -12},
                {"inpaint_radius_delta": 1, "photo_noise_delta": 0.045, "edge_breakup_delta": 0.014},
            ]
        )
    if not patches:
        patches.extend(
            [
                {"photo_noise_delta": 0.014, "edge_breakup_delta": 0.004},
                {"mask_threshold_delta": -8, "inpaint_radius_delta": -1},
            ]
        )
    return dedupe_patches(patches, 8)


def visual_background_cleanup_patches(acceptance: dict[str, Any]) -> list[dict[str, Any]]:
    if not acceptance_reports_background_patch(acceptance):
        return []
    return [
        {"photo_noise_delta": 0.030, "edge_breakup_delta": 0.010, "jpeg_quality_delta": -6},
        {"photo_noise_delta": 0.045, "edge_breakup_delta": 0.014, "jpeg_quality_delta": -10},
        {"inpaint_radius_delta": -1, "photo_noise_delta": 0.035, "edge_breakup_delta": 0.012},
        {"mask_threshold_delta": -8, "inpaint_radius_delta": -1, "photo_noise_delta": 0.030},
        {"mask_dilate_iterations_delta": -1, "photo_noise_delta": 0.035, "edge_breakup_delta": 0.010},
    ]


def ink_balance_recovery_patches(report: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    issue_types = {
        str(issue.get("type") or "")
        for issue in stage_issues(report, "ink_gray_balance")
        if isinstance(issue, dict)
    }
    patches: list[dict[str, Any]] = []
    if "core_mean_gray_too_light" in issue_types or "core_lighten_too_high" in issue_types:
        patches.extend(
            [
                {"opacity_delta": 0.015},
                {"alpha_contrast_delta": 0.05},
                {"opacity_delta": 0.015, "blur_delta": -0.04},
                {"core_darken_strength_delta": 0.02},
                {"opacity_delta": 0.01, "alpha_contrast_delta": 0.04},
                {"opacity_delta": 0.02, "core_darken_strength_delta": 0.02},
            ]
        )
    if any("too_black" in issue_type or issue_type == "roi_core_too_black" for issue_type in issue_types):
        patches.extend(black_core_reduction_patches()[:6])
    if not patches:
        patches.extend(
            [
                {"opacity_delta": -0.02},
                {"opacity_delta": 0.02},
                {"blur_delta": 0.04},
                {"blur_delta": -0.04},
            ]
        )
    return dedupe_patches(patches, 12)


def keep_patch_for_gray_stroke_recovery(patch: dict[str, Any]) -> bool:
    try:
        opacity_delta = float(patch.get("opacity_delta") or 0.0)
        blur_delta = float(patch.get("blur_delta") or 0.0)
        alpha_contrast_delta = float(patch.get("alpha_contrast_delta") or 0.0)
        stroke_opacity_delta = float(patch.get("stroke_opacity_delta") or 0.0)
        ink_gain_delta = float(patch.get("ink_gain_delta") or 0.0)
        photo_noise_delta = float(patch.get("photo_noise_delta") or 0.0)
        edge_breakup_delta = float(patch.get("edge_breakup_delta") or 0.0)
        font_size_delta = int(round(float(patch.get("font_size_delta") or 0.0)))
    except (TypeError, ValueError):
        return False
    if opacity_delta < 0 and stroke_opacity_delta <= 0:
        return False
    if blur_delta < 0 and stroke_opacity_delta <= 0:
        return False
    if alpha_contrast_delta > 0:
        return False
    if font_size_delta < 0:
        return False
    widens_body = (
        stroke_opacity_delta > 0.0
        or ink_gain_delta > 0.0
        or blur_delta > 0.0
        or photo_noise_delta > 0.0
        or edge_breakup_delta > 0.0
        or font_size_delta > 0
    )
    if not widens_body:
        return False
    return True


def keep_patch_for_outer_gray_cleanup(patch: dict[str, Any]) -> bool:
    try:
        opacity_delta = float(patch.get("opacity_delta") or 0.0)
        blur_delta = float(patch.get("blur_delta") or 0.0)
        stroke_opacity_delta = float(patch.get("stroke_opacity_delta") or 0.0)
        ink_gain_delta = float(patch.get("ink_gain_delta") or 0.0)
        alpha_contrast_delta = float(patch.get("alpha_contrast_delta") or 0.0)
        core_ink_gain_delta = float(patch.get("core_ink_gain_delta") or 0.0)
        core_darken_strength_delta = float(patch.get("core_darken_strength_delta") or 0.0)
        photo_noise_delta = float(patch.get("photo_noise_delta") or 0.0)
        edge_breakup_delta = float(patch.get("edge_breakup_delta") or 0.0)
        font_size_delta = int(round(float(patch.get("font_size_delta") or 0.0)))
    except (TypeError, ValueError):
        return False
    if opacity_delta < -0.025:
        return False
    if blur_delta > 0.0 or photo_noise_delta > 0.0 or edge_breakup_delta > 0.0:
        return False
    if core_ink_gain_delta < 0.0 or core_darken_strength_delta < 0.0:
        return False
    if font_size_delta != 0:
        return False
    trims_outer_gray = (
        blur_delta < 0.0
        or stroke_opacity_delta < 0.0
        or ink_gain_delta < 0.0
        or alpha_contrast_delta > 0.0
        or photo_noise_delta < 0.0
        or edge_breakup_delta < 0.0
    )
    preserves_core = core_ink_gain_delta > 0.0 or core_darken_strength_delta > 0.0 or opacity_delta > 0.0
    return trims_outer_gray and preserves_core


def keep_patch_for_neighbor_core_recovery(patch: dict[str, Any]) -> bool:
    try:
        opacity_delta = float(patch.get("opacity_delta") or 0.0)
        blur_delta = float(patch.get("blur_delta") or 0.0)
        stroke_opacity_delta = float(patch.get("stroke_opacity_delta") or 0.0)
        ink_gain_delta = float(patch.get("ink_gain_delta") or 0.0)
        core_ink_gain_delta = float(patch.get("core_ink_gain_delta") or 0.0)
        core_darken_strength_delta = float(patch.get("core_darken_strength_delta") or 0.0)
        photo_noise_delta = float(patch.get("photo_noise_delta") or 0.0)
        edge_breakup_delta = float(patch.get("edge_breakup_delta") or 0.0)
        font_size_delta = int(round(float(patch.get("font_size_delta") or 0.0)))
    except (TypeError, ValueError):
        return False
    if opacity_delta < -0.035:
        return False
    if blur_delta > 0.02:
        return False
    if photo_noise_delta > 0.0 or edge_breakup_delta > 0.0:
        return False
    if core_ink_gain_delta < 0.0 or core_darken_strength_delta < 0.0:
        return False
    if font_size_delta < 0:
        return False
    improves_core = (
        stroke_opacity_delta > 0.0
        or ink_gain_delta > 0.0
        or core_ink_gain_delta > 0.0
        or core_darken_strength_delta > 0.0
        or blur_delta < 0.0
    )
    return improves_core


def revision_patches_for_round(
    params: CandidateParams,
    acceptance: dict[str, Any],
    report: dict[str, Any] | None = None,
    *,
    rank_patch: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    patches: list[dict[str, Any]] = []
    stage_gate = stage_gate_for_report(report) if isinstance(report, dict) else {}
    blocking_stage = stage_gate.get("blocking_stage") or acceptance_blocking_stage(acceptance)
    if blocking_stage == "text_shape":
        if report_has_outer_gray_halo(report):
            patches.extend(neighbor_outer_gray_cleanup_patches())
            patches.extend(
                patch
                for patch in numeric_revision_patches(params, acceptance)
                if keep_patch_for_outer_gray_cleanup(patch)
            )
            if isinstance(rank_patch, dict) and keep_patch_for_outer_gray_cleanup(rank_patch):
                patches.append(rank_patch)
            return dedupe_patches(patches, 12)
        if local_neighbor_style_issues(report or {}, allow_excess_black_core=True):
            patches.extend(neighbor_core_density_recovery_patches())
            patches.extend(
                patch
                for patch in numeric_revision_patches(params, acceptance)
                if keep_patch_for_neighbor_core_recovery(patch)
            )
            if isinstance(rank_patch, dict) and keep_patch_for_neighbor_core_recovery(rank_patch):
                patches.append(rank_patch)
            return dedupe_patches(patches, 12)
        patches.extend(gray_stroke_recovery_patches())
        patches.extend(
            patch
            for patch in numeric_revision_patches(params, acceptance)
            if keep_patch_for_gray_stroke_recovery(patch)
        )
        if isinstance(rank_patch, dict) and keep_patch_for_gray_stroke_recovery(rank_patch):
            patches.append(rank_patch)
        return dedupe_patches(patches, 12)

    if blocking_stage == "photo_texture":
        patches.extend(alignment_centering_patches(report))
        patches.extend(photo_texture_recovery_patches(report))
        patches.extend(final_revision_patches(acceptance))
        if isinstance(rank_patch, dict):
            patches.append(rank_patch)
        return dedupe_patches(patches, 12)

    if blocking_stage == "background_cleanup":
        patches.extend(background_cleanup_recovery_patches(report))
        patches.extend(visual_background_cleanup_patches(acceptance))
        patches.extend(numeric_revision_patches(params, acceptance))
        patches.extend(photo_texture_recovery_patches(report))
        if isinstance(rank_patch, dict):
            patches.append(rank_patch)
        return dedupe_patches(patches, 12)

    if report_has_excess_black_core(report):
        patches.extend(alignment_centering_patches(report))
        patches.extend(black_core_reduction_patches())
        patches.extend(numeric_revision_patches(params, acceptance))
        if isinstance(rank_patch, dict):
            patches.append(rank_patch)
        patches.extend(final_revision_patches(acceptance))
        return dedupe_patches(patches, 12)

    if blocking_stage == "ink_gray_balance":
        patches.extend(alignment_centering_patches(report))
        patches.extend(ink_balance_recovery_patches(report))
        patches.extend(numeric_revision_patches(params, acceptance))
        if isinstance(rank_patch, dict):
            patches.append(rank_patch)
        patches.extend(final_revision_patches(acceptance))
        return dedupe_patches(patches, 12)

    needs_wider_gray_strokes = report_needs_wider_gray_strokes(report)
    if needs_wider_gray_strokes:
        patches.extend(alignment_centering_patches(report))
        if report_has_outer_gray_halo(report):
            patches.extend(neighbor_outer_gray_cleanup_patches())
            patches.extend(
                patch
                for patch in numeric_revision_patches(params, acceptance)
                if keep_patch_for_outer_gray_cleanup(patch)
            )
            if isinstance(rank_patch, dict) and keep_patch_for_outer_gray_cleanup(rank_patch):
                patches.append(rank_patch)
            return dedupe_patches(patches, 12)
        if local_neighbor_style_issues(report, allow_excess_black_core=True):
            patches.extend(neighbor_core_density_recovery_patches())
            patches.extend(
                patch
                for patch in numeric_revision_patches(params, acceptance)
                if keep_patch_for_neighbor_core_recovery(patch)
            )
            if isinstance(rank_patch, dict) and keep_patch_for_neighbor_core_recovery(rank_patch):
                patches.append(rank_patch)
            return dedupe_patches(patches, 12)
        patches.extend(gray_stroke_recovery_patches())
        patches.extend(
            patch
            for patch in numeric_revision_patches(params, acceptance)
            if keep_patch_for_gray_stroke_recovery(patch)
        )
        return dedupe_patches(patches, 12)

    patches.extend(alignment_centering_patches(report))
    if report_needs_thinner_strokes(report):
        patches.extend(thin_dark_core_patches(acceptance))
    patches.extend(numeric_revision_patches(params, acceptance))
    if isinstance(rank_patch, dict):
        patches.append(rank_patch)
    patches.extend(final_revision_patches(acceptance))
    return dedupe_patches(patches, 12)


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
    return tuple(candidates[:4])


def text_shape_reset_candidates(
    params: CandidateParams,
    font_style_reference: dict[str, Any],
    plan: RenderPlan,
    report: dict[str, Any] | None,
    *,
    limit: int = 48,
) -> list[CandidateParams]:
    if not report_blocks_text_shape(report):
        return []

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

    if has_outer_halo:
        shape_grid = (
            (1.00, 0.10, 0.02, 0.02, 0.16, 0.12, 0.10, 0.06, 0.004, 0.008, 98),
            (0.98, 0.12, 0.04, 0.02, 0.14, 0.14, 0.12, 0.06, 0.004, 0.010, 98),
            (0.96, 0.14, 0.04, 0.03, 0.12, 0.16, 0.12, 0.07, 0.006, 0.012, 96),
            (1.00, 0.08, 0.06, 0.01, 0.20, 0.12, 0.10, 0.05, 0.002, 0.006, 99),
        )
        size_deltas = (0, -1, 1)
    elif has_body_gap:
        shape_grid = (
            (0.90, 0.16, 0.03, 0.00, 0.18, 0.00, 0.00, 0.06, 0.004, 0.008, 99),
            (0.92, 0.14, 0.04, 0.00, 0.16, 0.04, 0.04, 0.06, 0.004, 0.008, 99),
            (0.94, 0.16, 0.04, 0.01, 0.14, 0.08, 0.06, 0.06, 0.004, 0.010, 98),
            (1.00, 0.12, 0.04, 0.04, 0.10, 0.18, 0.14, 0.07, 0.006, 0.012, 96),
            (0.98, 0.14, 0.06, 0.03, 0.10, 0.18, 0.14, 0.07, 0.006, 0.014, 96),
            (0.96, 0.16, 0.08, 0.02, 0.12, 0.16, 0.12, 0.08, 0.008, 0.016, 95),
            (1.00, 0.10, 0.08, 0.02, 0.16, 0.14, 0.12, 0.06, 0.004, 0.010, 98),
            (0.94, 0.18, 0.06, 0.03, 0.08, 0.20, 0.14, 0.08, 0.008, 0.018, 94),
            (1.00, 0.14, 0.02, 0.05, 0.08, 0.22, 0.16, 0.08, 0.006, 0.014, 96),
        )
        size_deltas = (0, 1, -1, 2)
    else:
        shape_grid = (
            (1.00, 0.12, 0.04, 0.03, 0.12, 0.16, 0.12, 0.07, 0.006, 0.012, 96),
            (0.98, 0.16, 0.04, 0.03, 0.10, 0.18, 0.14, 0.08, 0.006, 0.014, 96),
            (1.00, 0.10, 0.06, 0.02, 0.16, 0.14, 0.12, 0.06, 0.004, 0.010, 98),
        )
        size_deltas = (0, -1, 1)

    if has_pose_gap:
        size_deltas = tuple(dict.fromkeys(size_deltas + (0,)))

    max_font_size = max_font_size_for_plan(plan)
    offset_candidates = normalized_offset_candidates(plan, params)
    text_dy_candidates = tuple(dict.fromkeys((params.text_dy, 0, -1, 1)))[:3]
    variants: list[CandidateParams] = []
    for font_item in shape_font_items(params, font_style_reference, limit=5):
        font_name = str(font_item.get("font_name") or params.font_name)
        font_path = str(font_item.get("font_path") or params.font_path)
        try:
            base_size = int(font_item.get("font_size") or params.font_size)
        except (TypeError, ValueError):
            base_size = params.font_size
        for size_delta in size_deltas:
            font_size = max(8, min(max_font_size, base_size + int(size_delta)))
            for offsets in offset_candidates:
                for text_dy in text_dy_candidates:
                    for (
                        opacity,
                        blur,
                        stroke_opacity,
                        ink_gain,
                        alpha_contrast,
                        core_ink_gain,
                        core_darken_strength,
                        photo_warp,
                        edge_breakup,
                        photo_noise,
                        jpeg_quality,
                    ) in shape_grid:
                        variants.append(
                            mutate_params(
                                params,
                                font_name=font_name,
                                font_path=font_path,
                                font_size=font_size,
                                opacity=opacity,
                                blur=blur,
                                stroke_opacity=stroke_opacity,
                                ink_gain=ink_gain,
                                alpha_contrast=alpha_contrast,
                                core_ink_gain=core_ink_gain,
                                core_darken_strength=core_darken_strength,
                                core_darken_threshold=130,
                                core_darken_target_gray=28,
                                text_dy=text_dy,
                                char_offsets=offsets,
                                photo_warp=photo_warp,
                                edge_breakup=edge_breakup,
                                photo_noise=photo_noise,
                                jpeg_quality=jpeg_quality,
                            )
                        )
    return dedupe_params(variants, limit)


def final_revision_patches(acceptance: dict[str, Any]) -> list[dict[str, Any]]:
    findings = acceptance.get("visual_findings") if isinstance(acceptance, dict) else {}
    if not isinstance(findings, dict):
        findings = {}

    darkness = str(findings.get("darkness", "")).strip().lower()
    stroke_weight = str(findings.get("stroke_weight", "")).strip().lower()
    sharpness = str(findings.get("sharpness", "")).strip().lower()
    background = str(findings.get("background", "")).strip().lower()
    text = "\n".join(acceptance_text_fragments(acceptance)).lower()
    patches: list[dict[str, Any]] = []

    if darkness == "too_dark" or stroke_weight == "too_bold":
        patches.extend(
            [
                {
                    "font_size_delta": -1,
                    "blur_delta": -0.08,
                    "alpha_contrast_delta": 0.20,
                    "core_ink_gain_delta": -0.10,
                    "core_darken_strength_delta": 0.04,
                    "core_darken_threshold_delta": 10,
                },
                {"core_darken_strength_delta": -0.04},
                {"core_ink_gain_delta": -0.06, "core_darken_strength_delta": -0.04},
                {"opacity_delta": -0.05, "core_ink_gain_delta": -0.06, "core_darken_strength_delta": -0.04},
                {"opacity_delta": -0.06, "core_ink_gain_delta": -0.10, "core_darken_strength_delta": -0.08},
                {
                    "opacity_delta": -0.06,
                    "core_ink_gain_delta": -0.15,
                    "core_darken_strength_delta": -0.12,
                    "blur_delta": 0.10,
                },
                {
                    "opacity_delta": -0.06,
                    "core_ink_gain_delta": -0.15,
                    "core_darken_strength_delta": -0.15,
                    "blur_delta": 0.15,
                },
            ]
        )
    elif darkness == "too_light" or stroke_weight == "too_thin":
        patches.extend(
            [
                {"core_darken_strength_delta": 0.04},
                {"core_ink_gain_delta": 0.06, "core_darken_strength_delta": 0.04},
                {"opacity_delta": 0.04, "core_ink_gain_delta": 0.06},
            ]
        )

    if sharpness == "too_sharp":
        patches.append({"blur_delta": 0.08, "opacity_delta": -0.03})
        patches.append(
            {
                "blur_delta": 0.08,
                "edge_breakup_delta": 0.012,
                "photo_noise_delta": 0.030,
                "jpeg_quality_delta": -8,
            }
        )
    elif sharpness == "too_blurry":
        patches.append({"blur_delta": -0.08, "opacity_delta": 0.03})

    if (
        background == "ghost_visible"
        or "残影" in text
        or "旧字" in text
        or "ghost_visible" in text
    ):
        patches.extend(
            [
                {
                    "mask_threshold_delta": 12,
                    "mask_dilate_iterations_delta": 1,
                    "inpaint_radius_delta": 1,
                    "photo_noise_delta": -0.018,
                    "edge_breakup_delta": -0.010,
                    "jpeg_quality_delta": 8,
                },
                {
                    "mask_threshold_delta": 20,
                    "mask_dilate_iterations_delta": 1,
                    "inpaint_radius_delta": 1,
                    "photo_noise_delta": -0.024,
                    "edge_breakup_delta": -0.012,
                    "jpeg_quality_delta": 10,
                },
                {
                    "mask_dilate_iterations_delta": 1,
                    "inpaint_radius_delta": 1,
                    "photo_noise_delta": -0.020,
                    "edge_breakup_delta": -0.010,
                    "jpeg_quality_delta": 8,
                },
            ]
        )
    elif (
        background in {"patch_visible", "too_smooth"}
        or "补丁" in text
        or "平滑" in text
        or "涂抹" in text
        or "patch_visible" in text
    ):
        patches.extend(
            [
                {"photo_noise_delta": 0.012, "edge_breakup_delta": 0.006},
                {"photo_noise_delta": 0.018, "edge_breakup_delta": 0.008, "jpeg_quality_delta": -4},
                {"inpaint_radius_delta": -1, "photo_noise_delta": 0.014, "edge_breakup_delta": 0.006},
                {"mask_dilate_iterations_delta": -1, "photo_noise_delta": 0.016, "edge_breakup_delta": 0.006},
                {"photo_noise_delta": 0.030, "edge_breakup_delta": 0.010, "jpeg_quality_delta": -8},
            ]
        )

    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for patch in patches:
        key = json.dumps(patch, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(patch)
    return unique[:7]


def final_font_revision_candidates(
    params: CandidateParams,
    font_style_reference: dict[str, Any],
    plan: RenderPlan,
    report: dict[str, Any] | None,
) -> list[CandidateParams]:
    shape_reset = text_shape_reset_candidates(
        params,
        font_style_reference,
        plan,
        report,
        limit=24,
    )
    if shape_reset:
        return shape_reset

    ranked_fonts = font_style_reference.get("ranked_fonts", [])
    if not isinstance(ranked_fonts, list):
        return []

    preferred_order = ("SimSun", "Songti", "GBSN", "FangSong", "NotoSerif", "UMing")
    selected: list[dict[str, Any]] = []
    seen_paths: set[str] = {params.font_path}
    for preferred_name in preferred_order:
        for item in ranked_fonts:
            if not isinstance(item, dict):
                continue
            font_path = str(item.get("font_path") or "")
            font_name = str(item.get("font_name") or "")
            if not font_path or font_path in seen_paths:
                continue
            if font_name != preferred_name:
                continue
            selected.append(item)
            seen_paths.add(font_path)
            break

    variants: list[CandidateParams] = []
    for item in selected[:4]:
        font_name = str(item.get("font_name") or params.font_name)
        font_path = str(item.get("font_path") or params.font_path)
        try:
            base_size = int(item.get("font_size") or params.font_size)
        except (TypeError, ValueError):
            base_size = params.font_size
        tuning_grid = (
            (0, 0.98, 0.14, 0.04, 0.03, 0.12, 0.18, 0.14),
            (0, 1.00, 0.12, 0.06, 0.02, 0.16, 0.14, 0.12),
            (1, 0.96, 0.16, 0.06, 0.03, 0.10, 0.18, 0.14),
            (-1, 1.00, 0.10, 0.04, 0.02, 0.18, 0.12, 0.10),
        )
        for (
            size_delta,
            opacity,
            blur,
            stroke_opacity,
            ink_gain,
            alpha_contrast,
            core_ink_gain,
            core_darken_strength,
        ) in tuning_grid:
            variants.append(
                mutate_params(
                    params,
                    font_name=font_name,
                    font_path=font_path,
                    font_size=max(8, base_size + size_delta),
                    opacity=opacity,
                    blur=blur,
                    stroke_opacity=stroke_opacity,
                    core_ink_gain=core_ink_gain,
                    core_darken_strength=core_darken_strength,
                    ink_gain=ink_gain,
                    alpha_contrast=alpha_contrast,
                    photo_warp=min(0.10, params.photo_warp),
                    edge_breakup=min(0.010, params.edge_breakup),
                    photo_noise=min(0.020, params.photo_noise),
                    jpeg_quality=max(94, params.jpeg_quality),
                )
            )
    return dedupe_params(variants, 8)


def run_region_vision_checks(
    *,
    original: Image.Image,
    rendered: list[tuple[CandidateParams, Image.Image, dict[str, Any], float]],
    plan: RenderPlan,
    region_dir: Path,
    vision_client: VisionClient,
    prompts: tuple[str, str, str],
    candidate_limit: int,
    font_style_reference: dict[str, Any],
    progress: ProgressCallback | None = None,
) -> tuple[CandidateParams | None, dict[str, Any]]:
    if not rendered:
        return None, {"enabled": True, "error": "no candidates for vision review"}

    master_prompt, candidate_prompt_template, final_prompt_template = prompts
    region_dir.mkdir(parents=True, exist_ok=True)
    context_box = region_context_box(plan.search_roi, original.size)
    original_context_path = region_dir / "vision_original_context.png"
    save_region_context(original, context_box, original_context_path)

    vision_rendered = rendered[: max(1, candidate_limit)]
    vision_sheet_path = region_dir / "vision_candidate_sheet.png"
    sheet_items = [
        (
            params_label(params),
            compare_region_preview(original, candidate, plan.search_roi, scale=3),
        )
        for params, candidate, _report, _score in vision_rendered
    ]
    make_contact_sheet(sheet_items, vision_sheet_path, scale=1, cols=1)

    hard_reports = compact_hard_reports(vision_rendered, plan)
    prompt = candidate_prompt_template.replace(
        "{hard_check_report}",
        json.dumps(hard_reports, ensure_ascii=False, indent=2),
    )
    prompt += web_prompt_context(plan)
    prompt += STRICT_ACCEPTANCE_APPENDIX
    candidate_rank_json = vision_client.call_json(
        system_prompt=master_prompt,
        user_prompt=prompt,
        image_paths=[original_context_path, vision_sheet_path],
    )
    write_json(region_dir / "visual_eval_candidate_rank.json", candidate_rank_json)

    model_best = find_best_candidate_from_model(
        candidate_rank_json,
        [(params, image, report) for params, image, report, _score in vision_rendered],
    )
    if model_best is not None:
        model_tuple = next(
            (item for item in rendered if item[0].candidate_id == model_best.candidate_id),
            None,
        )
        if model_tuple is None or not report_strict_pass(model_tuple[2]) or not report_stage_pass(model_tuple[2]):
            candidate_rank_json["model_choice_overridden"] = {
                "candidate_id": model_best.candidate_id,
                "reason": "model selected a candidate that failed hard_check, strict_gate, or ordered stage_gate",
            }
            model_best = None
    strict_fallback = next(
        (item[0] for item in rendered if report_strict_pass(item[2]) and report_stage_pass(item[2])),
        next((item[0] for item in rendered if report_strict_pass(item[2])), rendered[0][0]),
    )
    chosen_params = model_best or strict_fallback
    write_json(region_dir / "visual_eval_candidate_rank.json", candidate_rank_json)
    chosen_tuple = next(
        (item for item in rendered if item[0].candidate_id == chosen_params.candidate_id),
        rendered[0],
    )
    final_params, final_image, final_report, final_score = chosen_tuple
    final_context_path = region_dir / "vision_final_context.png"
    final_compare_path = region_dir / "vision_final_compare.png"

    def evaluate_final(
        *,
        params: CandidateParams,
        image: Image.Image,
        report: dict[str, Any],
        score: float,
        context_path: Path,
        compare_path: Path,
        out_path: Path,
    ) -> dict[str, Any]:
        save_region_context(image, context_box, context_path)
        save_region_compare(original, image, context_box, compare_path)
        hard_payload = {
            "task": {
                "source_text": plan.source_text,
                "target_text": plan.target_text,
                "search_roi": list(plan.search_roi),
                "target_roi": list(plan.target_roi),
                "draw_mode": plan.draw_mode,
                "text_angle_degrees": round(float(plan.text_angle_degrees), 3),
                "slot_boxes": [asdict(item) for item in plan.slot_boxes],
                "protected_boxes": [list(item) for item in plan.protected_boxes],
            },
            "final_score": round(float(score), 3),
            "hard_check": report,
        }
        final_prompt = (
            final_prompt_template.replace(
                "{final_params}",
                json.dumps(asdict(params), ensure_ascii=False, indent=2),
            ).replace(
                "{hard_check_report}",
                json.dumps(hard_payload, ensure_ascii=False, indent=2),
            )
        )
        final_prompt += web_prompt_context(plan)
        final_prompt += STRICT_ACCEPTANCE_APPENDIX
        final_json = vision_client.call_json(
            system_prompt=master_prompt,
            user_prompt=final_prompt,
            image_paths=[original_context_path, context_path, compare_path],
        )
        final_json = apply_local_acceptance_gate(final_json, report)
        write_json(out_path, final_json)
        return final_json

    final_acceptance_json = evaluate_final(
        params=final_params,
        image=final_image,
        report=final_report,
        score=final_score,
        context_path=final_context_path,
        compare_path=final_compare_path,
        out_path=region_dir / "final_acceptance.json",
    )

    strict_pass = report_strict_pass(final_report)
    hard_boundary_pass = bool(final_report.get("pass")) if isinstance(final_report, dict) else False
    final_visual_pass = final_acceptance_delivers(final_acceptance_json)
    revision_attempts: list[dict[str, Any]] = []
    revision_rounds: list[dict[str, Any]] = []
    revision_previews: list[dict[str, Any]] = []
    accepted = strict_pass and final_visual_pass
    if progress:
        progress(
            "region_initial_acceptance",
            {
                "accepted": accepted,
                "strict_pass": strict_pass,
                "hard_boundary_pass": hard_boundary_pass,
                "acceptance_level": final_acceptance_json.get("acceptance_level"),
                "final_decision": final_acceptance_json.get("final_decision"),
            },
        )

    if hard_boundary_pass and not accepted:
        write_json(region_dir / "final_acceptance_initial.json", final_acceptance_json)
        current_params = final_params
        current_image = final_image
        current_report = final_report
        current_score = final_score
        current_acceptance = final_acceptance_json
        seen_params: set[str] = {params_signature(current_params)}
        rank_patch = candidate_rank_json.get("suggested_patch")
        max_revision_rounds = 8
        for round_idx in range(1, max_revision_rounds + 1):
            basis_stage_gate = stage_gate_for_report(current_report) if isinstance(current_report, dict) else {}
            basis_blocking_stage, basis_stage_is_local = effective_blocking_stage(current_report, current_acceptance)
            basis_stage_source = (
                "local_report"
                if basis_stage_is_local
                else "vision_acceptance"
                if basis_blocking_stage
                else "none"
            )
            basis_stage_severity = stage_issue_severity(current_report, basis_blocking_stage)
            basis_stage_policy = stage_policy_summary(str(basis_blocking_stage) if basis_blocking_stage else None)
            if progress:
                progress(
                    "revision_round_started",
                    {
                        "round": round_idx,
                        "basis_candidate_id": current_params.candidate_id,
                        "basis_acceptance_level": current_acceptance.get("acceptance_level"),
                        "basis_final_decision": current_acceptance.get("final_decision"),
                        "basis_blocking_stage": basis_blocking_stage,
                        "basis_stage_source": basis_stage_source,
                        "basis_stage_severity": round(float(basis_stage_severity), 3),
                    },
                )
            round_patches = revision_patches_for_round(
                current_params,
                current_acceptance,
                current_report,
                rank_patch=rank_patch if round_idx == 1 and isinstance(rank_patch, dict) else None,
            )
            model_records: list[dict[str, Any]] = []
            if round_idx == 1:
                model_records.extend(
                    model_patch_records(current_params, candidate_rank_json, source="candidate_rank")
                )
            model_records.extend(
                model_patch_records(
                    current_params,
                    current_acceptance,
                    source=f"final_acceptance_basis_round_{round_idx - 1}",
                )
            )
            model_conflicts: list[dict[str, Any]] = []
            allowed_model_patches: list[dict[str, Any]] = []
            patch_source_lookup: dict[str, list[dict[str, Any]]] = {}
            for record in model_records:
                patch = record.get("patch")
                if not isinstance(patch, dict):
                    continue
                policy_audit = patch_policy_audit(
                    str(basis_blocking_stage) if basis_blocking_stage else None,
                    patch,
                )
                record["patch_policy"] = policy_audit
                patch_source_lookup.setdefault(patch_signature(patch), []).append(record)
                if policy_audit["allowed"]:
                    allowed_model_patches.append(patch)
                else:
                    model_conflicts.append(record)
            round_patches.extend(allowed_model_patches)
            filtered_patches: list[dict[str, Any]] = []
            rejected_local_patches: list[dict[str, Any]] = []
            for patch in dedupe_patches(round_patches, 24):
                policy_audit = patch_policy_audit(
                    str(basis_blocking_stage) if basis_blocking_stage else None,
                    patch,
                )
                if policy_audit["allowed"]:
                    filtered_patches.append(patch)
                else:
                    rejected_local_patches.append(
                        {
                            "patch": patch,
                            "patch_policy": policy_audit,
                        }
                    )
            round_patches = dedupe_patches(filtered_patches, 12)
            shape_reset_params = text_shape_reset_candidates(
                current_params,
                font_style_reference,
                plan,
                current_report,
                limit=48,
            )
            round_record: dict[str, Any] = {
                "round": round_idx,
                "basis_candidate_id": current_params.candidate_id,
                "basis_acceptance_level": current_acceptance.get("acceptance_level"),
                "basis_final_decision": current_acceptance.get("final_decision"),
                "patch_count": len(round_patches),
                "shape_reset_count": len(shape_reset_params),
                "basis_blocking_stage": basis_blocking_stage,
                "basis_stage_source": basis_stage_source,
                "basis_stage_severity": round(float(basis_stage_severity), 3),
                "stage_policy": basis_stage_policy,
                "model_suggestions": model_records,
                "model_conflicts": model_conflicts,
                "rejected_local_patches": rejected_local_patches,
            }
            if progress:
                progress(
                    "revision_round_candidates",
                    {
                        "round": round_idx,
                        "patch_count": len(round_patches),
                        "shape_reset_count": len(shape_reset_params),
                        "basis_blocking_stage": basis_blocking_stage,
                        "basis_stage_source": basis_stage_source,
                        "basis_stage_severity": round(float(basis_stage_severity), 3),
                        "stage_policy": basis_stage_policy,
                    },
                )
            if not round_patches and not shape_reset_params:
                round_record["stop_reason"] = "no_revision_candidates"
                revision_rounds.append(round_record)
                break

            round_candidates: list[
                tuple[float, float, CandidateParams, Image.Image, dict[str, Any], dict[str, Any]]
            ] = []

            candidate_jobs: list[tuple[str, int, dict[str, Any] | None, CandidateParams, dict[str, Any]]] = []
            for shape_idx, shape_params in enumerate(shape_reset_params, start=1):
                candidate_jobs.append(("shape_reset", shape_idx, None, shape_params, {"applied": False, "reason": "none", "changes": {}}))
            for patch_idx, patch in enumerate(round_patches, start=1):
                raw_patched_params = apply_suggested_patch(current_params, patch)
                patched_params = constrained_revision_params(
                    raw_patched_params,
                    current_params,
                    current_acceptance,
                    current_report,
                    round_idx=round_idx,
                )
                audit = constraint_audit(
                    raw_patched_params,
                    patched_params,
                    current_report,
                    current_acceptance,
                )
                candidate_jobs.append(("patch", patch_idx, patch, patched_params, audit))

            for candidate_origin, candidate_idx, patch, patched_params, patch_constraint_audit in candidate_jobs:
                patched_params = mutate_params(
                    patched_params,
                    candidate_id=(
                        f"{current_params.candidate_id}_s{round_idx:02d}_{candidate_idx:02d}"
                        if candidate_origin == "shape_reset"
                        else f"{current_params.candidate_id}_i{round_idx:02d}_{candidate_idx:02d}"
                    ),
                )
                signature = params_signature(patched_params)
                if signature in seen_params:
                    continue
                seen_params.add(signature)
                patched_image = render_candidate(original, plan, patched_params)
                patched_report = candidate_report(
                    original,
                    patched_image,
                    plan,
                    patched_params,
                    font_style_reference,
                )
                patched_score = region_candidate_score(
                    original,
                    patched_image,
                    plan,
                    patched_report,
                )
                patched_selection_score = revision_selection_score(
                    patched_score,
                    patched_params,
                    current_params,
                    current_acceptance,
                    current_report,
                    patched_report,
                )
                patched_strict = report_strict_pass(patched_report)
                patched_stage_gate = patched_report.get("stage_gate") or {}
                patched_blocking_stage = patched_stage_gate.get("blocking_stage")
                progresses_past_text_shape = (
                    report_blocks_text_shape(current_report)
                    and patched_report.get("pass")
                    and patched_blocking_stage != "text_shape"
                )
                current_blocking_stage = basis_blocking_stage
                current_stage_improvement = 0.0
                current_stage_severity_before = 0.0
                current_stage_severity_after = 0.0
                if basis_stage_is_local and current_blocking_stage and current_blocking_stage != "text_shape":
                    current_stage_severity_before = stage_issue_severity(
                        current_report,
                        str(current_blocking_stage),
                    )
                    current_stage_severity_after = stage_issue_severity(
                        patched_report,
                        str(current_blocking_stage),
                    )
                    current_stage_improvement = current_stage_severity_before - current_stage_severity_after
                improves_current_stage = (
                    bool(current_blocking_stage)
                    and current_blocking_stage != "text_shape"
                    and patched_report.get("pass")
                    and patched_blocking_stage == current_blocking_stage
                    and (patched_score < current_score - 1.0 or current_stage_improvement > 6.0)
                )
                attempt_record = {
                    "index": len(revision_attempts) + 1,
                    "round": round_idx,
                    "origin": candidate_origin,
                    "round_candidate": candidate_idx,
                    "basis_candidate_id": current_params.candidate_id,
                    "params": asdict(patched_params),
                    "strict_pass": patched_strict,
                    "stage_pass": report_stage_pass(patched_report),
                    "blocking_stage": patched_blocking_stage,
                    "progresses_past_text_shape": progresses_past_text_shape,
                    "improves_current_stage": improves_current_stage,
                    "current_blocking_stage": current_blocking_stage,
                    "current_stage_severity_before": round(float(current_stage_severity_before), 3),
                    "current_stage_severity_after": round(float(current_stage_severity_after), 3),
                    "current_stage_improvement": round(float(current_stage_improvement), 3),
                    "score": round(float(patched_score), 3),
                    "selection_score": round(float(patched_selection_score), 3),
                }
                if patch is not None:
                    attempt_record["patch"] = patch
                    attempt_record["patch_policy"] = patch_policy_audit(
                        str(current_blocking_stage) if current_blocking_stage else None,
                        patch,
                    )
                    suggestion_records = patch_source_lookup.get(patch_signature(patch), [])
                    if suggestion_records:
                        attempt_record["model_suggestions"] = suggestion_records
                    if patch_constraint_audit.get("applied"):
                        patch_constraint_audit["alternative_candidate_id"] = patched_params.candidate_id
                    attempt_record["constraint"] = patch_constraint_audit
                else:
                    attempt_record["patch_policy"] = {
                        **stage_policy_summary(str(current_blocking_stage) if current_blocking_stage else None),
                        "families": ["shape_reset"],
                        "primary_families": ["shape_reset"],
                        "allowed": current_blocking_stage == "text_shape",
                        "rejection_reason": None if current_blocking_stage == "text_shape" else "shape reset is only generated for text_shape",
                    }
                if not patched_strict:
                    attempt_record["strict_gate"] = patched_report.get("strict_gate")
                if not report_stage_pass(patched_report):
                    attempt_record["stage_gate"] = patched_report.get("stage_gate")
                revision_attempts.append(attempt_record)
                if patched_strict or progresses_past_text_shape or improves_current_stage:
                    round_candidates.append(
                        (
                            patched_selection_score,
                            patched_score,
                            patched_params,
                            patched_image,
                            patched_report,
                            attempt_record,
                        )
                    )

            if not round_candidates:
                round_record["stop_reason"] = "no_selectable_revision_candidate"
                revision_rounds.append(round_record)
                break

            round_candidates.sort(key=lambda item: item[0])
            selected_tuple = round_candidates[0]
            selected_reason = "lowest_selection_score"
            if report_blocks_text_shape(current_report):
                progressed_candidates = [
                    item
                    for item in round_candidates
                    if item[5].get("progresses_past_text_shape")
                ]
                if progressed_candidates:
                    progressed_candidates.sort(key=lambda item: item[0])
                    selected_tuple = progressed_candidates[0]
                    selected_reason = "progresses_past_text_shape"
            current_blocking_stage = basis_blocking_stage
            if basis_stage_is_local and current_blocking_stage and current_blocking_stage != "text_shape":
                improving_stage_candidates = [
                    item
                    for item in round_candidates
                    if float(item[5].get("current_stage_improvement") or 0.0) > 0.0
                ]
                if improving_stage_candidates:
                    best_improvement = max(
                        float(item[5].get("current_stage_improvement") or 0.0)
                        for item in improving_stage_candidates
                    )
                    minimum_improvement = max(6.0, best_improvement * 0.70)
                    near_best = [
                        item
                        for item in improving_stage_candidates
                        if float(item[5].get("current_stage_improvement") or 0.0) >= minimum_improvement
                    ]
                    if near_best:
                        near_best.sort(key=lambda item: (item[0], item[1]))
                        selected_tuple = near_best[0]
                        selected_reason = f"current_stage_severity_improved:{current_blocking_stage}"
                    else:
                        improving_stage_candidates.sort(
                            key=lambda item: (
                                -float(item[5].get("current_stage_improvement") or 0.0),
                                item[0],
                                item[1],
                            )
                        )
                        selected_tuple = improving_stage_candidates[0]
                        selected_reason = f"current_stage_severity_small_improvement:{current_blocking_stage}"
                else:
                    round_record["stop_reason"] = f"no_{current_blocking_stage}_severity_improvement"
                    round_record["attempt_count"] = len(revision_attempts)
                    revision_rounds.append(round_record)
                    break
            if (
                selected_reason == "lowest_selection_score"
                and not report_blocks_text_shape(current_report)
                and report_needs_wider_gray_strokes(current_report)
            ):
                stroke_candidates = [
                    item
                    for item in round_candidates
                    if float(item[2].stroke_opacity) > float(current_params.stroke_opacity)
                ]
                if stroke_candidates:
                    stroke_candidates.sort(key=lambda item: item[0])
                    if stroke_candidates[0][0] <= selected_tuple[0] + 360.0:
                        selected_tuple = stroke_candidates[0]
                        selected_reason = "stroke_body_recovery_priority"
            patched_selection_score, patched_score, patched_params, patched_image, patched_report, attempt_record = selected_tuple
            attempt_record["selected_for_visual"] = True
            attempt_record["selected_reason"] = selected_reason
            patched_context_path = region_dir / f"vision_final_context_iter{round_idx:02d}.png"
            patched_compare_path = region_dir / f"vision_final_compare_iter{round_idx:02d}.png"
            patched_acceptance = evaluate_final(
                params=patched_params,
                image=patched_image,
                report=patched_report,
                score=patched_score,
                context_path=patched_context_path,
                compare_path=patched_compare_path,
                out_path=region_dir / f"final_acceptance_iter{round_idx:02d}.json",
            )
            attempt_record["final_acceptance"] = patched_acceptance
            round_delivered = final_acceptance_delivers(patched_acceptance)
            if progress:
                progress(
                    "revision_round_finished",
                    {
                        "round": round_idx,
                        "candidate_id": patched_params.candidate_id,
                        "score": round(float(patched_score), 3),
                        "selection_score": round(float(patched_selection_score), 3),
                        "selected_reason": selected_reason,
                        "current_blocking_stage": attempt_record.get("current_blocking_stage"),
                        "current_stage_severity_before": attempt_record.get("current_stage_severity_before"),
                        "current_stage_severity_after": attempt_record.get("current_stage_severity_after"),
                        "current_stage_improvement": attempt_record.get("current_stage_improvement"),
                        "accepted": round_delivered,
                        "acceptance_level": patched_acceptance.get("acceptance_level"),
                        "final_decision": patched_acceptance.get("final_decision"),
                        "blocking_stage": (patched_report.get("stage_gate") or {}).get("blocking_stage"),
                    },
                )
            round_record.update(
                {
                    "selected_candidate_id": patched_params.candidate_id,
                    "selected_attempt_index": attempt_record["index"],
                    "selected_score": round(float(patched_score), 3),
                    "selected_selection_score": round(float(patched_selection_score), 3),
                    "selected_reason": selected_reason,
                    "current_blocking_stage": attempt_record.get("current_blocking_stage"),
                    "current_stage_severity_before": attempt_record.get("current_stage_severity_before"),
                    "current_stage_severity_after": attempt_record.get("current_stage_severity_after"),
                    "current_stage_improvement": attempt_record.get("current_stage_improvement"),
                    "accepted": round_delivered,
                    "acceptance_level": patched_acceptance.get("acceptance_level"),
                    "final_decision": patched_acceptance.get("final_decision"),
                    "blocking_stage": (patched_report.get("stage_gate") or {}).get("blocking_stage"),
                }
            )
            revision_rounds.append(round_record)
            revision_previews.append(
                {
                    "round": round_idx,
                    "kind": "revision_selected",
                    "candidate_id": patched_params.candidate_id,
                    "label": params_label(patched_params),
                    "score": round(float(patched_score), 3),
                    "path": str(patched_compare_path),
                    "selected_reason": selected_reason,
                    **candidate_trace_summary(patched_report),
                    "metrics": (patched_report.get("strict_visual_metrics") or {}).get("bands", {}),
                }
            )

            current_params = patched_params
            current_image = patched_image
            current_report = patched_report
            current_score = patched_score
            current_acceptance = patched_acceptance
            final_params = current_params
            final_image = current_image
            final_report = current_report
            final_score = current_score
            final_context_path = patched_context_path
            final_compare_path = patched_compare_path
            final_acceptance_json = current_acceptance
            write_json(region_dir / "final_acceptance.json", final_acceptance_json)
            if round_delivered:
                accepted = True
                break

    if report_strict_pass(final_report) and not accepted:
        for alt_idx, alt_params in enumerate(
            final_font_revision_candidates(final_params, font_style_reference, plan, final_report),
            start=len(revision_attempts) + 1,
        ):
            alt_params = mutate_params(
                alt_params,
                candidate_id=f"{final_params.candidate_id}_f{alt_idx:02d}",
            )
            alt_image = render_candidate(original, plan, alt_params)
            alt_report = candidate_report(
                original,
                alt_image,
                plan,
                alt_params,
                font_style_reference,
            )
            alt_score = region_candidate_score(original, alt_image, plan, alt_report)
            alt_strict = report_strict_pass(alt_report)
            attempt_record = {
                "index": alt_idx,
                "font_revision": True,
                "params": asdict(alt_params),
                "strict_pass": alt_strict,
                "score": round(float(alt_score), 3),
            }
            if not alt_strict:
                attempt_record["strict_gate"] = alt_report.get("strict_gate")
                revision_attempts.append(attempt_record)
                continue

            alt_context_path = region_dir / f"vision_final_context_f{alt_idx:02d}.png"
            alt_compare_path = region_dir / f"vision_final_compare_f{alt_idx:02d}.png"
            alt_acceptance = evaluate_final(
                params=alt_params,
                image=alt_image,
                report=alt_report,
                score=alt_score,
                context_path=alt_context_path,
                compare_path=alt_compare_path,
                out_path=region_dir / f"final_acceptance_f{alt_idx:02d}.json",
            )
            attempt_record["final_acceptance"] = alt_acceptance
            revision_attempts.append(attempt_record)
            if final_acceptance_delivers(alt_acceptance):
                final_params = alt_params
                final_image = alt_image
                final_report = alt_report
                final_score = alt_score
                final_context_path = alt_context_path
                final_compare_path = alt_compare_path
                final_acceptance_json = alt_acceptance
                write_json(region_dir / "final_acceptance.json", final_acceptance_json)
                accepted = True
                break

    final_stage_gate = final_report.get("stage_gate") if isinstance(final_report, dict) else {}
    if not isinstance(final_stage_gate, dict):
        final_stage_gate = stage_gate_for_report(final_report)
    next_round_plan = None
    if not accepted:
        blocking_stage = final_stage_gate.get("blocking_stage") or acceptance_blocking_stage(final_acceptance_json)
        next_round_plan = {
            "blocking_stage": blocking_stage,
            "stage_severity": round(float(stage_issue_severity(final_report, blocking_stage)), 3),
            "stage_source": (
                "local_report"
                if final_stage_gate.get("blocking_stage")
                else "vision_acceptance"
                if blocking_stage
                else "none"
            ),
            "reference_profile_dynamic_ink": (
                (final_report.get("reference_profile") or {}).get("dynamic_ink")
                if isinstance(final_report.get("reference_profile"), dict)
                else {}
            ),
            "actions": [],
        }
        actions = next_round_plan["actions"]
        if blocking_stage == "text_shape":
            actions.append("Search shape reset candidates first: font family, size, slot alignment, stroke body, local shear.")
        elif blocking_stage == "ink_gray_balance":
            actions.append("Generate lower-core candidates using reference_profile opacity floor, core_ink_gain, and core_darken_strength limits.")
            actions.append(f"Do not clamp opacity above {opacity_floor_for_excess_black(final_report):.2f} unless text_shape regresses.")
        elif blocking_stage == "photo_texture":
            actions.append("After shape and ink pass, tune blur, edge breakup, photo noise, and JPEG texture only.")
        elif blocking_stage == "background_cleanup":
            actions.append("Regenerate inpaint/background texture candidates before judging text darkness.")
        else:
            actions.append("No local blocking stage remains; retry final visual acceptance with saved final candidate context.")

    return final_params, {
        "enabled": True,
        "accepted": accepted,
        "accepted_reason": (
            "strict_and_visual_pass"
            if accepted
            else "not_accepted"
        ),
        "candidate_rank": candidate_rank_json,
        "final_acceptance": final_acceptance_json,
        "next_round_plan": next_round_plan,
        "revision_attempts": revision_attempts,
        "revision_rounds": revision_rounds,
        "artifacts": {
            "original_context": str(original_context_path),
            "candidate_sheet": str(vision_sheet_path),
            "final_context": str(final_context_path),
            "final_compare": str(final_compare_path),
            "revision_previews": revision_previews,
        },
    }


def compare_region_preview(
    original: Image.Image,
    candidate: Image.Image,
    roi: tuple[int, int, int, int],
    *,
    scale: int = 5,
) -> Image.Image:
    x1, y1, x2, y2 = roi
    pad = max(8, round(max(x2 - x1, y2 - y1) * 0.35))
    preview_box = clamp_box((x1 - pad, y1 - pad, x2 + pad, y2 + pad), original.size)
    old = original.crop(preview_box)
    new = candidate.crop(preview_box)
    w, h = old.size
    label_h = 24
    sheet = Image.new("RGB", (w * scale * 2, h * scale + label_h), (248, 248, 248))
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 5), "original", fill=(20, 20, 20))
    draw.text((w * scale + 6, 5), "candidate", fill=(20, 20, 20))
    sheet.paste(old.resize((w * scale, h * scale), Image.Resampling.NEAREST), (0, label_h))
    sheet.paste(new.resize((w * scale, h * scale), Image.Resampling.NEAREST), (w * scale, label_h))
    return sheet


def candidate_trace_summary(report: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    stage_gate = report.get("stage_gate") if isinstance(report.get("stage_gate"), dict) else stage_gate_for_report(report)
    blocking_stage = stage_gate.get("blocking_stage") if isinstance(stage_gate, dict) else None
    background_metrics = report.get("background_texture_metrics")
    background_issues = stage_issues(report, "background_cleanup")
    return {
        "blocking_stage": blocking_stage,
        "stage_pass": bool(stage_gate.get("pass")) if isinstance(stage_gate, dict) else None,
        "stage_severity": round(float(stage_issue_severity(report, blocking_stage)), 3),
        "background": {
            "issues": [
                str(issue.get("type") or "")
                for issue in background_issues
                if isinstance(issue, dict)
            ],
            "patch_mean_delta": (
                background_metrics.get("patch_mean_delta")
                if isinstance(background_metrics, dict)
                else None
            ),
            "patch_variance_ratio": (
                background_metrics.get("patch_variance_ratio")
                if isinstance(background_metrics, dict)
                else None
            ),
            "residual_energy_ratio": (
                background_metrics.get("residual_energy_ratio")
                if isinstance(background_metrics, dict)
                else None
            ),
            "white_ghost_probe": (
                background_metrics.get("white_ghost_probe")
                if isinstance(background_metrics, dict)
                else None
            ),
            "shadow_ghost_ratio": (
                background_metrics.get("shadow_ghost_ratio")
                if isinstance(background_metrics, dict)
                else None
            ),
        },
    }


def process_region(
    original: Image.Image,
    roi: tuple[int, int, int, int],
    *,
    source_text: str,
    target_text: str,
    run_dir: Path,
    region_id: str,
    vision_client: VisionClient,
    prompts: tuple[str, str, str],
    max_candidates: int = 120,
    vision_candidate_limit: int = 8,
    progress: ProgressCallback | None = None,
) -> tuple[Image.Image, Image.Image, list[dict[str, Any]], dict[str, Any], bool]:
    plan = build_region_plan(
        original,
        roi,
        source_text=source_text,
        target_text=target_text,
    )
    font_candidates = find_font_candidates(font_source="recommended")
    font_candidates, rejected_fonts = filter_fonts_by_required_text(
        font_candidates,
        f"{source_text}{target_text}",
    )
    source_count = len(text_chars(source_text))
    target_count = len(text_chars(target_text))
    initial_size = initial_font_size(plan)
    max_font_size = max_font_size_for_plan(plan)
    style_plan = plan
    if source_count and target_count and target_count < source_count and plan.slot_boxes:
        style_plan = RenderPlan(
            target_text=plan.target_text,
            source_text=plan.source_text,
            search_roi=plan.search_roi,
            target_roi=plan.target_roi,
            slot_boxes=plan.slot_boxes,
            protected_boxes=plan.protected_boxes,
            source_reference_box=slots_roi(plan.slot_boxes, original.size) or plan.source_reference_box,
            style_reference_box=plan.style_reference_box,
            style_reference_text=plan.style_reference_text,
            draw_mode=plan.draw_mode,
            text_angle_degrees=plan.text_angle_degrees,
        )
    font_style_reference = build_font_style_reference(
        original,
        style_plan,
        font_candidates,
        min_size=12,
        max_size=max(18, min(72, max_font_size)),
        prefer_serif_categories=True,
    )
    font_candidates = rank_fonts_by_style_reference(font_candidates, font_style_reference)
    font_name, font_path = font_candidates[0]
    centered = plan.draw_mode == "center"
    centered_longer = centered and bool(source_count and target_count > source_count)
    current = CandidateParams(
        candidate_id="current",
        font_name=font_name,
        font_path=font_path,
        font_size=initial_size,
        opacity=0.83 if centered_longer else 0.92 if centered else 1.0,
        blur=0.35 if centered_longer else 0.35 if centered else 0.18,
        stroke_opacity=0.0,
        ink_gain=0.0 if centered else 0.04,
        alpha_contrast=0.0,
        core_ink_gain=0.0 if centered else 0.48,
        core_darken_strength=0.0 if centered else 0.34,
        core_darken_threshold=130,
        core_darken_target_gray=28,
        text_dy=0,
        char_offsets=default_char_offsets(target_text),
        mask_threshold=155 if centered_longer else 165,
        mask_dilate_iterations=2,
        inpaint_radius=2 if centered_longer else 3,
        photo_warp=0.08 if not centered else 0.04,
        edge_breakup=0.010 if not centered else 0.006,
        photo_noise=0.018 if not centered else 0.012,
        jpeg_quality=94,
    )
    params_list = generate_candidates(
        current,
        font_candidates=font_candidates,
        font_style_reference=font_style_reference,
        font_pool_size=min(8, len(font_candidates)),
        iteration=0,
        limit=max_candidates,
    )
    if source_count and target_count > source_count:
        params_list = [params for params in params_list if params.font_size <= max_font_size]
    if centered:
        best_sizes = {
            item["font_path"]: int(item["font_size"])
            for item in font_style_reference.get("ranked_fonts", [])
            if item.get("font_path") and item.get("font_size")
        }
        for extra_font_name, extra_font_path in font_candidates[: min(5, len(font_candidates))]:
            base_size = best_sizes.get(extra_font_path, current.font_size)
            centered_grid = (
                (
                    (0.52, 0.75, 0.0, 0.0, 0.00, 0.00),
                    (0.56, 0.75, 0.0, 0.0, 0.00, 0.00),
                    (0.56, 0.85, 0.0, 0.0, 0.00, 0.00),
                    (0.60, 0.75, 0.0, 0.0, 0.00, 0.00),
                    (0.64, 0.75, 0.0, 0.0, 0.00, 0.00),
                    (0.66, 0.60, 0.0, 0.0, 0.00, 0.00),
                    (0.68, 0.60, 0.0, 0.0, 0.01, 0.00),
                    (0.70, 0.55, 0.0, 0.0, 0.01, 0.00),
                    (0.72, 0.55, 0.0, 0.0, 0.00, 0.00),
                    (0.66, 0.55, 0.0, 0.0, 0.02, 0.00),
                    (0.68, 0.50, 0.0, 0.0, 0.00, 0.00),
                )
                if centered_longer
                else (
                    (0.55, 0.70, 0.0, 0.0, 0.0, 0.0),
                    (0.65, 0.70, 0.0, 0.0, 0.0, 0.0),
                    (0.75, 0.70, 0.0, 0.0, 0.0, 0.0),
                    (0.85, 0.70, 0.0, 0.0, 0.0, 0.0),
                    (0.55, 0.50, 0.0, 0.0, 0.0, 0.0),
                    (0.65, 0.50, 0.0, 0.0, 0.0, 0.0),
                    (0.75, 0.50, 0.0, 0.0, 0.0, 0.0),
                )
            )
            for size_delta in (-1, 0, 1, 2):
                for opacity, blur, stroke_opacity, alpha_contrast, core_ink_gain, core_darken_strength in centered_grid:
                    params_list.append(
                        mutate_params(
                            current,
                            font_name=extra_font_name,
                            font_path=extra_font_path,
                            font_size=min(max_font_size, base_size + size_delta),
                            opacity=opacity,
                            blur=blur,
                            stroke_opacity=stroke_opacity,
                            ink_gain=0.0,
                            alpha_contrast=alpha_contrast,
                            core_ink_gain=core_ink_gain,
                            core_darken_strength=core_darken_strength,
                            char_offsets=(),
                        )
                    )
        params_list = dedupe_params(params_list, max_candidates + 140)

    if centered and old_region_lt55_pixels(original, plan.target_roi) < 16:
        soft_params = [
            params for params in params_list
            if params.core_darken_strength <= 0.05 and params.core_ink_gain <= 0.20
        ]
        if soft_params:
            params_list = soft_params
    shorter_replacement = bool(source_count and target_count and target_count < source_count)
    if shorter_replacement and plan.draw_mode in {"auto", "line_chars"}:
        best_sizes = {
            item["font_path"]: int(item["font_size"])
            for item in font_style_reference.get("ranked_fonts", [])
            if item.get("font_path") and item.get("font_size")
        }
        left_aligned_offsets = (
            ((0, 0), (0, 0)),
            ((0, 0), (-1, 0)),
            ((0, 0), (1, 0)),
            ((-1, 0), (0, 0)),
            ((1, 0), (0, 0)),
        )
        left_aligned_text_dy = (0, 1)
        left_aligned_grid = (
            (0.58, 0.65, 0.00, 0.00, 0.00, 0.04, 0.00, 185, 2, 1),
            (0.58, 0.65, 0.00, 0.00, 0.00, 0.04, 0.00, 185, 3, 1),
            (0.58, 0.65, 0.00, 0.00, 0.00, 0.04, 0.00, 195, 3, 1),
            (0.60, 0.62, 0.00, 0.00, 0.00, 0.04, 0.00, 185, 2, 1),
            (0.60, 0.62, 0.00, 0.00, 0.00, 0.04, 0.00, 185, 3, 1),
            (0.62, 0.60, 0.00, 0.00, 0.00, 0.04, 0.00, 185, 2, 1),
            (0.62, 0.60, 0.00, 0.00, 0.00, 0.04, 0.00, 185, 3, 1),
            (0.62, 0.60, 0.00, 0.00, 0.00, 0.02, 0.00, 175, 2, 1),
            (0.62, 0.60, 0.00, 0.00, 0.00, 0.04, 0.00, 195, 3, 2),
            (0.64, 0.60, 0.00, 0.00, 0.00, 0.04, 0.00, 195, 3, 2),
            (0.64, 0.55, 0.00, 0.00, 0.00, 0.04, 0.00, 195, 3, 2),
            (0.64, 0.56, 0.00, 0.00, 0.00, 0.04, 0.00, 205, 3, 2),
            (0.66, 0.52, 0.00, 0.00, 0.00, 0.06, 0.00, 205, 3, 2),
            (0.68, 0.52, 0.00, 0.00, 0.00, 0.06, 0.00, 185, 2, 2),
            (0.68, 0.50, 0.00, 0.00, 0.00, 0.08, 0.02, 195, 3, 2),
            (0.72, 0.42, 0.00, 0.01, 0.00, 0.14, 0.06, 175, 2, 1),
            (0.76, 0.38, 0.00, 0.02, 0.00, 0.18, 0.10, 175, 2, 1),
            (0.80, 0.34, 0.00, 0.02, 0.00, 0.22, 0.14, 175, 2, 2),
            (0.84, 0.30, 0.00, 0.03, 0.00, 0.26, 0.18, 175, 2, 2),
            (0.88, 0.28, 0.00, 0.03, 0.00, 0.30, 0.22, 165, 2, 2),
            (0.92, 0.24, 0.00, 0.04, 0.00, 0.34, 0.26, 165, 2, 2),
            (0.96, 0.20, 0.00, 0.04, 0.00, 0.38, 0.30, 165, 2, 3),
        )
        for extra_font_name, extra_font_path in font_candidates[: min(5, len(font_candidates))]:
            base_size = best_sizes.get(extra_font_path, current.font_size)
            for size_delta in (0, 1, 2, -1):
                for offsets in left_aligned_offsets:
                    for text_dy in left_aligned_text_dy:
                        for (
                            opacity,
                            blur,
                            stroke_opacity,
                            ink_gain,
                            alpha_contrast,
                            core_ink_gain,
                            core_darken_strength,
                            mask_threshold,
                            mask_dilate_iterations,
                            inpaint_radius,
                        ) in left_aligned_grid:
                            params_list.append(
                                mutate_params(
                                    current,
                                    font_name=extra_font_name,
                                    font_path=extra_font_path,
                                    font_size=base_size + size_delta,
                                    opacity=opacity,
                                    blur=blur,
                                    stroke_opacity=stroke_opacity,
                                    ink_gain=ink_gain,
                                    alpha_contrast=alpha_contrast,
                                    core_ink_gain=core_ink_gain,
                                    core_darken_strength=core_darken_strength,
                                    text_dy=text_dy,
                                    char_offsets=offsets,
                                    mask_threshold=mask_threshold,
                                    mask_dilate_iterations=mask_dilate_iterations,
                                    inpaint_radius=inpaint_radius,
                                )
                            )
        params_list = dedupe_params(params_list, max_candidates + 160)
    rendered: list[tuple[CandidateParams, Image.Image, dict[str, Any], float]] = []
    if progress:
        progress(
            "region_candidates_started",
            {
                "region_id": region_id,
                "source_text": source_text,
                "target_text": target_text,
                "candidate_count": len(params_list),
            },
        )
    for params in params_list:
        candidate = render_candidate(original, plan, params)
        report = candidate_report(original, candidate, plan, params, font_style_reference)
        score = region_candidate_score(original, candidate, plan, report)
        if not report_strict_pass(report):
            score += 10000.0
        rendered.append((params, candidate, report, score))

    if not rendered:
        raise RuntimeError("no candidate could be rendered")

    rendered.sort(key=lambda item: item[3])
    if progress:
        progress(
            "region_candidates_finished",
            {
                "region_id": region_id,
                "rendered": len(rendered),
                "best_score": round(float(rendered[0][3]), 3) if rendered else None,
            },
        )
    region_dir = run_dir / "regions" / re.sub(r"[^A-Za-z0-9_.-]+", "_", region_id or "region")
    chosen_params, vision_summary = run_region_vision_checks(
        original=original,
        rendered=rendered,
        plan=plan,
        region_dir=region_dir,
        vision_client=vision_client,
        prompts=prompts,
        candidate_limit=vision_candidate_limit,
        font_style_reference=font_style_reference,
        progress=progress,
    )
    if chosen_params is not None:
        best_params = chosen_params
        best_image = render_candidate(original, plan, best_params)
        best_report = candidate_report(
            original,
            best_image,
            plan,
            best_params,
            font_style_reference,
        )
        best_score = region_candidate_score(original, best_image, plan, best_report)
    else:
        best_params, best_image, best_report, best_score = rendered[0]
    preview_items: list[dict[str, Any]] = []
    revision_previews = (
        ((vision_summary.get("artifacts") or {}).get("revision_previews") or [])
        if isinstance(vision_summary, dict)
        else []
    )
    for preview in revision_previews[-5:]:
        path = Path(str(preview.get("path") or ""))
        if not path.exists():
            continue
        try:
            preview_image = Image.open(path).convert("RGB")
        except Exception:
            continue
        metrics = preview.get("metrics") if isinstance(preview.get("metrics"), dict) else {}
        preview_items.append(
            {
                "index": len(preview_items) + 1,
                "kind": str(preview.get("kind") or "revision_selected"),
                "candidate_id": str(preview.get("candidate_id") or ""),
                "label": f"iter {preview.get('round')} {preview.get('label') or ''}",
                "score": preview.get("score"),
                "blocking_stage": preview.get("blocking_stage"),
                "stage_severity": preview.get("stage_severity"),
                "selection_reason": preview.get("selected_reason"),
                "background": preview.get("background") if isinstance(preview.get("background"), dict) else {},
                "dataUrl": image_to_data_url(preview_image),
                "metrics": {
                    "lt55_delta": metrics.get("lt55_delta"),
                    "band_55_70_delta": metrics.get("band_55_70_delta"),
                    "band_70_90_delta": metrics.get("band_70_90_delta"),
                    "band_120_165_delta": metrics.get("band_120_165_delta"),
                },
            }
        )
    for params, candidate, report, score in rendered:
        if len(preview_items) >= 5:
            break
        preview = compare_region_preview(original, candidate, roi)
        bands = report.get("strict_visual_metrics", {}).get("bands", {})
        cleanup = report.get("extra_source_slot_cleanup_metrics") or {}
        cleanup_items = cleanup.get("per_box") if isinstance(cleanup, dict) else None
        cleanup_first = cleanup_items[0] if cleanup_items else {}
        preview_items.append(
            {
                "index": len(preview_items) + 1,
                "kind": "initial_local_rank",
                "candidate_id": params.candidate_id,
                "label": (
                    f"{params.font_name} {params.font_size}px "
                    f"blur {params.blur:.2f} core {params.core_ink_gain:.2f} "
                    f"dark {params.core_darken_strength:.2f}"
                ),
                "score": round(float(score), 3),
                **candidate_trace_summary(report),
                "dataUrl": image_to_data_url(preview),
                "metrics": {
                    "lt55_delta": bands.get("lt55_delta"),
                    "band_55_70_delta": bands.get("band_55_70_delta"),
                    "band_70_90_delta": bands.get("band_70_90_delta"),
                    "band_120_165_delta": bands.get("band_120_165_delta"),
                    "extra_slot_lt150_ratio": cleanup_first.get("new_lt150_ratio"),
                    "extra_slot_column_deviation": cleanup_first.get("max_column_mean_deviation"),
                },
            }
        )

    accepted = bool(vision_summary.get("accepted"))
    applied_image = best_image if accepted else original.copy()
    selected_candidate_path = region_dir / "selected_candidate.png"
    selected_compare_path = region_dir / "selected_candidate_compare.png"
    best_image.save(selected_candidate_path)
    compare_region_preview(original, best_image, roi).save(selected_compare_path)
    best_trace = candidate_trace_summary(best_report)
    vision_next_plan = vision_summary.get("next_round_plan") if isinstance(vision_summary, dict) else None
    visual_final_stage = (
        vision_next_plan.get("blocking_stage")
        if isinstance(vision_next_plan, dict)
        else None
    )
    revision_round_records = (
        vision_summary.get("revision_rounds")
        if isinstance(vision_summary.get("revision_rounds"), list)
        else []
    )
    last_round = revision_round_records[-1] if revision_round_records else {}
    return (
        applied_image,
        best_image,
        preview_items,
        {
            "plan": {
                "search_roi": list(plan.search_roi),
                "target_roi": list(plan.target_roi),
                "slot_boxes": [asdict(item) for item in plan.slot_boxes],
                "protected_boxes": [list(item) for item in plan.protected_boxes],
                "draw_mode": plan.draw_mode,
                "text_angle_degrees": round(float(plan.text_angle_degrees), 3),
            },
            "params": asdict(best_params),
            "score": round(float(best_score), 3),
            "hard_check": best_report,
            "vision": vision_summary,
            "trace": {
                "accepted": accepted,
                "final_is_rejected_candidate": not accepted,
                "final_candidate_id": best_params.candidate_id,
                "final_blocking_stage": best_trace.get("blocking_stage") or visual_final_stage,
                "final_stage_severity": best_trace.get("stage_severity"),
                "revision_round_count": len(revision_round_records),
                "last_round_stop_reason": last_round.get("stop_reason") if isinstance(last_round, dict) else None,
                "last_round_selected_reason": last_round.get("selected_reason") if isinstance(last_round, dict) else None,
                "next_round_plan": vision_next_plan,
            },
            "accepted": accepted,
            "applied": accepted,
            "artifacts": {
                "selected_candidate": str(selected_candidate_path),
                "selected_compare": str(selected_compare_path),
                "display_image_is_candidate": not accepted,
            },
            "rejected_fonts": rejected_fonts,
        },
        accepted,
    )


def append_web_job_event(job_id: str, event: str, record: dict[str, Any]) -> None:
    with WEB_JOB_LOCK:
        job = WEB_JOBS.get(job_id)
        if not job:
            return
        events = job.setdefault("events", [])
        events.append({"event": event, **record})
        if len(events) > MAX_WEB_JOB_EVENTS:
            del events[: len(events) - MAX_WEB_JOB_EVENTS]
        job["updated_at"] = time.time()


def run_web_job(job_id: str, payload: dict[str, Any]) -> None:
    def emit_progress(event: str, record: dict[str, Any]) -> None:
        append_web_job_event(job_id, event, record)

    try:
        result = process_payload(payload, progress=emit_progress)
        with WEB_JOB_LOCK:
            job = WEB_JOBS.get(job_id)
            if job:
                job["result"] = result
                job["done"] = True
                job["updated_at"] = time.time()
    except Exception as exc:
        error = str(exc)
        append_web_job_event(
            job_id,
            "job_failed",
            {
                "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "error": error,
            },
        )
        with WEB_JOB_LOCK:
            job = WEB_JOBS.get(job_id)
            if job:
                job["error"] = error
                job["done"] = True
                job["updated_at"] = time.time()


def create_web_job(payload: dict[str, Any]) -> str:
    job_id = uuid.uuid4().hex
    now = time.time()
    with WEB_JOB_LOCK:
        WEB_JOBS[job_id] = {
            "job_id": job_id,
            "created_at": now,
            "updated_at": now,
            "done": False,
            "error": None,
            "events": [],
            "result": None,
        }
    thread = threading.Thread(target=run_web_job, args=(job_id, payload), daemon=True)
    thread.start()
    return job_id


def web_job_status(job_id: str) -> dict[str, Any] | None:
    with WEB_JOB_LOCK:
        job = WEB_JOBS.get(job_id)
        if not job:
            return None
        return {
            "ok": True,
            "jobId": job_id,
            "done": bool(job.get("done")),
            "error": job.get("error"),
            "events": list(job.get("events") or []),
            "result": job.get("result"),
        }


def process_payload(payload: dict[str, Any], progress: ProgressCallback | None = None) -> dict[str, Any]:
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.jsonl"

    def emit(event: str, fields: dict[str, Any] | None = None) -> None:
        record = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "event": event,
            **(fields or {}),
        }
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if progress:
            progress(event, record)

    write_json(run_dir / "request.json", request_audit_payload(payload))
    emit("run_started", {"run_dir": str(run_dir)})
    prompts = load_web_prompts()
    vision_client = VisionClient(ENV_PATH)
    results: list[dict[str, Any]] = []
    for image_item in payload.get("images", []):
        image_id = str(image_item.get("id") or "")
        filename = str(image_item.get("filename") or "image.png")
        try:
            instruction_details = parse_instruction_details(str(image_item.get("instruction") or ""))
            source_text = instruction_details["source_text"]
            target_text = instruction_details["target_text"]
            if not target_text:
                raise ValueError("missing replacement instruction")
            emit(
                "image_started",
                {
                    "image_id": image_id,
                    "filename": filename,
                    "source_text": source_text,
                    "target_text": target_text,
                },
            )
            image = image_from_data_url(str(image_item.get("dataUrl") or ""))
            original_image = image.copy()
            orientation_summary: dict[str, Any] = {
                "applied": False,
                "orientation": "none",
                "attempts": [],
            }
            candidates: list[dict[str, Any]] = []
            region_results: list[dict[str, Any]] = []
            image_accepted = True
            display_image: Image.Image | None = None
            regions = list(image_item.get("regions", []))
            if not regions:
                image, regions, orientation_summary = auto_orient_for_instruction(
                    image,
                    instruction=str(image_item.get("instruction") or ""),
                    source_text=source_text,
                    target_text=target_text,
                )
                emit(
                    "auto_roi_finished",
                    {
                        "image_id": image_id,
                        "orientation": orientation_summary.get("orientation"),
                        "direction_score": (orientation_summary.get("selected_attempt") or {}).get("direction_score"),
                        "selected_score": orientation_summary.get("selected_score"),
                        "attempt_count": len(orientation_summary.get("attempts") or []),
                        "region_count": len(regions),
                    },
                )
                original_image = image.copy()
            display_image = image.copy()
            for region in regions:
                rect = region.get("rect") or {}
                x = int(round(float(rect.get("x", 0))))
                y = int(round(float(rect.get("y", 0))))
                w = int(round(float(rect.get("w", 0))))
                h = int(round(float(rect.get("h", 0))))
                if w < 2 or h < 2:
                    continue
                roi = clamp_box((x, y, x + w, y + h), image.size)
                region_id = str(region.get("id") or f"region_{len(region_results) + 1}")
                region_source_text = str(region.get("sourceText") or source_text)
                region_target_text = str(region.get("targetText") or target_text)
                emit(
                    "region_started",
                    {
                        "image_id": image_id,
                        "region_id": region_id,
                        "roi": list(roi),
                        "source_text": region_source_text,
                        "target_text": region_target_text,
                    },
                )

                def region_progress(event: str, fields: dict[str, Any]) -> None:
                    emit(event, {"image_id": image_id, **(fields or {})})

                image, region_display_image, region_candidates, summary, accepted = process_region(
                    image,
                    roi,
                    source_text=region_source_text,
                    target_text=region_target_text,
                    run_dir=run_dir,
                    region_id=region_id,
                    vision_client=vision_client,
                    prompts=prompts,
                    max_candidates=int(payload.get("maxCandidates") or 120),
                    vision_candidate_limit=int(payload.get("visionCandidateLimit") or 8),
                    progress=region_progress,
                )
                display_image = image.copy() if accepted else region_display_image.copy()
                emit(
                    "region_finished",
                    {
                        "image_id": image_id,
                        "region_id": region_id,
                        "accepted": accepted,
                        "revision_rounds": len((summary.get("vision") or {}).get("revision_rounds", [])),
                        "blocking_stage": (summary.get("trace") or {}).get("final_blocking_stage"),
                        "stage_severity": (summary.get("trace") or {}).get("final_stage_severity"),
                        "stop_reason": (summary.get("trace") or {}).get("last_round_stop_reason"),
                    },
                )
                image_accepted = image_accepted and accepted
                for candidate in region_candidates:
                    candidate["regionId"] = region_id
                candidates.extend(region_candidates)
                region_results.append(
                    {
                        "id": region_id,
                        "roi": list(roi),
                        "sourceText": region_source_text,
                        "targetText": region_target_text,
                        "auto": bool(region.get("auto")),
                        "accepted": accepted,
                        "summary": summary,
                    }
                )
            if not region_results:
                raise ValueError("no valid rectangles")

            safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(filename).stem)[:80] or image_id or "image"
            original_path = run_dir / f"{safe_stem}_original.png"
            final_path = run_dir / f"{safe_stem}_final.png"
            applied_path = run_dir / f"{safe_stem}_applied.png"
            original_image.save(original_path)
            image.save(applied_path)
            result_image = display_image or image
            result_image.save(final_path)
            results.append(
                {
                    "id": image_id,
                    "ok": True,
                    "accepted": image_accepted,
                    "applied": image_accepted,
                    "filename": filename,
                    "sourceDataUrl": image_to_data_url(original_image),
                    "resultDataUrl": image_to_data_url(result_image),
                    "candidates": candidates[:5],
                    "orientation": orientation_summary,
                    "regions": region_results,
                    "artifacts": {
                        "original": str(original_path),
                        "final": str(final_path),
                        "applied": str(applied_path),
                        "final_is_rejected_candidate": not image_accepted,
                    },
                }
            )
            emit("image_finished", {"image_id": image_id, "accepted": image_accepted})
        except Exception as exc:
            emit("image_failed", {"image_id": image_id, "error": str(exc)})
            results.append(
                {
                    "id": image_id,
                    "ok": False,
                    "filename": filename,
                    "error": str(exc),
                }
            )
    response = {"ok": True, "runDir": str(run_dir), "images": results}
    write_json(run_dir / "result.json", result_audit_payload(response))
    emit("run_finished", {"ok": True})
    return response


class RoiWebHandler(BaseHTTPRequestHandler):
    server_version = "RoiImageEditWeb/1.0"

    def do_HEAD(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = "/index.html" if parsed.path == "/" else parsed.path
        if path.startswith("/"):
            path = path[1:]
        file_path = (WEB_DIR / unquote(path)).resolve()
        if not str(file_path).startswith(str(WEB_DIR.resolve())) or not file_path.exists() or not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(file_path.stat().st_size))
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/api/process/status":
            job_id = parse_qs(parsed.query).get("job_id", [""])[0]
            if not job_id:
                self.write_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "missing job_id"})
                return
            status = web_job_status(job_id)
            if status is None:
                self.write_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "job not found"})
                return
            self.write_json(HTTPStatus.OK, status)
            return
        if path == "/":
            path = "/index.html"
        if path.startswith("/"):
            path = path[1:]
        file_path = (WEB_DIR / path).resolve()
        if not str(file_path).startswith(str(WEB_DIR.resolve())) or not file_path.exists() or not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        data = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path not in {"/api/process", "/api/process/start"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if path == "/api/process/start":
                job_id = create_web_job(payload)
                self.write_json(HTTPStatus.ACCEPTED, {"ok": True, "jobId": job_id})
                return
            self.write_json(HTTPStatus.OK, process_payload(payload))
        except Exception as exc:
            self.write_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"ok": False, "error": str(exc)},
            )

    def write_json(self, status: HTTPStatus, value: dict[str, Any]) -> None:
        data = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[roi-web] {self.address_string()} - {fmt % args}")


def serve(host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), RoiWebHandler)
    print(f"Serving ROI image edit web UI on http://{host}:{port}", flush=True)
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
