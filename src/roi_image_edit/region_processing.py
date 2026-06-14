from __future__ import annotations

import base64
import io
import json
import re
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageOps

from roi_image_edit.iterative_pipeline import (
    STRICT_ACCEPTANCE_APPENDIX,
    CandidateParams,
    RenderPlan,
    VisionClient,
    apply_suggested_patch,
    build_font_style_reference,
    clamp_box,
    dedupe_params,
    default_char_offsets,
    find_best_candidate_from_model,
    filter_fonts_by_required_text,
    find_font_candidates,
    generate_candidates,
    make_contact_sheet,
    mutate_params,
    params_label,
    rank_fonts_by_style_reference,
    render_candidate,
    report_strict_pass,
    write_json,
)
from roi_image_edit.prompt_assets import load_prompt, missing_prompt_names
from roi_image_edit.auto_roi_artifacts import auto_roi_evidence_payload, save_auto_roi_overlay
from roi_image_edit.failure_artifacts import failed_image_result
from roi_image_edit.image_classification import classify_region_roi_policy, with_region_roi_policy

from roi_image_edit.local_validation import (
    apply_local_acceptance_gate,
    candidate_report,
    compact_hard_reports,
    constraint_audit,
    old_region_lt55_pixels,
    opacity_floor_for_excess_black,
    region_candidate_score,
    region_context_box,
    report_has_excess_black_core,
    report_needs_wider_gray_strokes,
    report_stage_pass,
    save_region_compare,
    save_region_context,
    stage_issue_severity,
    stage_issues,
)

from roi_image_edit.forced_model_seeds import forced_model_seed_audit
from roi_image_edit.model_suggestions import (
    combined_model_suggestion_patch,
    filter_model_patch_records,
    model_stage_response_contract,
    model_suggestion_filter_report,
)
from roi_image_edit.pre_candidate_gates import pre_candidate_gate_report

from roi_image_edit.revision_selector import (
    constrained_revision_params,
    final_acceptance_delivers,
    revision_selection_score,
)
from roi_image_edit.revision_solver import (
    InkGrayCandidateGrid,
    controlled_escape_candidate_grid,
    final_font_revision_candidates,
    ink_gray_candidate_grid,
    layered_candidate_search_report,
    photo_texture_candidate_grid,
    report_blocks_text_shape,
    text_shape_reset_candidate_grid,
)
from roi_image_edit.stage_patchers import (
    acceptance_blocking_stage,
    dispatch_revision_patches,
    effective_blocking_stage,
    model_patch_records,
    params_signature,
    patch_signature,
)
from roi_image_edit.run_artifacts import (
    attach_stage_context_to_rank_report,
    delivery_artifact_manifest,
    model_stage_context,
    normalize_vision_candidate_limit,
    progress_record,
    request_audit_payload,
    result_audit_payload,
    revision_round_continuation_contract,
    stage_progress_fields,
    vision_candidate_request_payload,
)
from roi_image_edit.vision_targets import (
    non_regression_guard_report,
    vision_target_alignment,
    vision_target_alignment_complete,
    vision_target_from_acceptance,
    vision_target_recipe_patches,
    vision_target_recipe_report,
)
from roi_image_edit.stage_profiles import resolve_stage_profile
from roi_image_edit.stages import stage_gate_for_report
from roi_image_edit.roi_locator import (
    auto_orient_for_instruction,
    build_region_plan,
    initial_font_size,
    max_font_size_for_plan,
    parse_instruction_details,
    slots_roi,
    text_chars,
)
from roi_image_edit.stage_policy import (
    STAGE_ORDER,
    optimization_policy_audit,
    selected_optimization_step,
    stage_optimization_summary,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "output" / "web"
ENV_PATH = ROOT / ".env"
ProgressCallback = Callable[[str, dict[str, Any]], None]


def longer_replacement_soft_scan_candidates(
    current: CandidateParams,
    *,
    font_candidates: list[tuple[str, str]],
    font_style_reference: dict[str, Any],
    max_font_size: int,
) -> list[CandidateParams]:
    best_sizes = {
        item["font_path"]: int(item["font_size"])
        for item in font_style_reference.get("ranked_fonts", [])
        if item.get("font_path") and item.get("font_size")
    }
    soft_grid = (
        (0.64, 0.32, 0.00, 0.40, 0.00, 0.00),
        (0.64, 0.36, 0.00, 0.40, 0.00, 0.00),
        (0.64, 0.28, 0.00, 0.40, 0.00, 0.00),
        (0.66, 0.36, 0.00, 0.30, 0.00, 0.00),
        (0.62, 0.36, 0.00, 0.20, 0.00, 0.00),
        (0.66, 0.44, 0.00, 0.25, 0.00, 0.00),
        (0.64, 0.44, 0.00, 0.30, 0.00, 0.00),
        (0.66, 0.48, 0.00, 0.22, 0.00, 0.00),
        (0.60, 0.55, 0.00, 0.00, 0.00, 0.00),
        (0.62, 0.55, 0.00, 0.00, 0.00, 0.00),
        (0.64, 0.55, 0.00, 0.00, 0.00, 0.00),
        (0.66, 0.55, 0.00, 0.00, 0.00, 0.00),
        (0.60, 0.60, 0.00, 0.00, 0.00, 0.00),
        (0.62, 0.60, 0.00, 0.00, 0.00, 0.00),
        (0.70, 0.55, 0.00, 0.00, 0.00, 0.00),
        (0.78, 0.55, 0.00, 0.00, 0.00, 0.00),
        (0.78, 0.75, 0.00, 0.00, 0.00, 0.00),
        (0.82, 0.50, 0.00, 0.00, 0.00, 0.00),
        (0.86, 0.50, 0.00, 0.00, 0.00, 0.00),
        (0.90, 0.42, 0.00, 0.00, 0.00, 0.00),
        (0.90, 0.42, 0.01, 0.00, 0.04, 0.00),
    )
    candidates: list[CandidateParams] = []
    for font_name, font_path in font_candidates[: min(4, len(font_candidates))]:
        base_size = best_sizes.get(font_path, current.font_size)
        for size_delta in (-1, -2, 0):
            for opacity, blur, ink_gain, alpha_contrast, core_ink_gain, core_darken_strength in soft_grid:
                candidates.append(
                    mutate_params(
                        current,
                        font_name=font_name,
                        font_path=font_path,
                        font_size=max(8, min(max_font_size, base_size + size_delta)),
                        opacity=opacity,
                        blur=blur,
                        stroke_opacity=0.0,
                        ink_gain=ink_gain,
                        alpha_contrast=alpha_contrast,
                        core_ink_gain=core_ink_gain,
                        core_darken_strength=core_darken_strength,
                        char_offsets=current.char_offsets,
                    )
                )
    return candidates


def report_stage_status_pass(report: dict[str, Any] | None, stage_id: str | None) -> bool:
    if not isinstance(report, dict) or not stage_id:
        return False
    stage_gate = report.get("stage_gate")
    if not isinstance(stage_gate, dict):
        stage_gate = stage_gate_for_report(report)
    stage_status = stage_gate.get("stage_status")
    if not isinstance(stage_status, dict):
        return bool(stage_gate.get("pass") and not stage_gate.get("blocking_stage"))
    stage_report = stage_status.get(stage_id)
    return isinstance(stage_report, dict) and bool(stage_report.get("pass"))


def progresses_past_blocking_stage(
    report: dict[str, Any] | None,
    current_blocking_stage: str | None,
    next_blocking_stage: str | None,
) -> bool:
    if not current_blocking_stage or current_blocking_stage == "text_shape":
        return False
    if not report_stage_status_pass(report, current_blocking_stage):
        return False
    return next_blocking_stage != current_blocking_stage


def text_shape_ink_guard_selectable(
    before_report: dict[str, Any] | None,
    after_report: dict[str, Any] | None,
) -> dict[str, Any]:
    text_before = float(stage_issue_severity(before_report, "text_shape"))
    text_after = float(stage_issue_severity(after_report, "text_shape"))
    ink_before = float(stage_issue_severity(before_report, "ink_gray_balance"))
    ink_after = float(stage_issue_severity(after_report, "ink_gray_balance"))
    text_delta = text_after - text_before
    ink_improvement = ink_before - ink_after
    text_not_regressed = text_delta <= 1.0
    ink_improved = ink_improvement > 6.0
    enabled = bool(report_blocks_text_shape(before_report) and report_has_excess_black_core(before_report))
    return {
        "enabled": enabled,
        "guard_stage": "ink_gray_balance",
        "primary_stage": "text_shape",
        "text_shape_severity_before": round(text_before, 3),
        "text_shape_severity_after": round(text_after, 3),
        "text_shape_severity_delta": round(text_delta, 3),
        "text_shape_not_regressed": text_not_regressed,
        "ink_gray_severity_before": round(ink_before, 3),
        "ink_gray_severity_after": round(ink_after, 3),
        "ink_gray_improvement": round(ink_improvement, 3),
        "ink_gray_improved": ink_improved,
        "selectable": bool(enabled and text_not_regressed and ink_improved),
        "selection_rule": "text_shape candidates may not worsen ink; ink guard candidates may proceed only when text_shape does not regress and ink severity drops",
    }


def load_processing_prompts() -> tuple[str, str, str]:
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


def processing_prompt_context(plan: RenderPlan, stage_context: dict[str, Any] | None = None) -> str:
    target_chars = [ch for ch in plan.target_text if not ch.isspace()]
    protected_texts = [text for text in plan.protected_texts if text]
    stage_context_text = (
        "\n当前阶段上下文 JSON：\n"
        f"{json.dumps(stage_context, ensure_ascii=False, indent=2)}\n"
        if stage_context
        else ""
    )
    return (
        "\n\n动态任务补充：\n"
        f"- 字段 key field_key: {plan.field_key or ''}\n"
        f"- 本次字段标签 field_label_text: {plan.field_label_text or ''}\n"
        f"- 本次字段分隔符 field_separator_text: {plan.field_separator_text or ''}\n"
        f"- 本次受保护文本 protected_texts: {protected_texts}\n"
        f"- 旧文字 source_text: {plan.source_text or ''}\n"
        f"- 新文字 target_text: {plan.target_text}\n"
        f"- 本次真实目标字符序列: {target_chars}\n"
        "- prompt 模板不得假设固定字段、固定标签或固定字数；本次必须按 field context、source_text 和 target_text 的真实内容逐个判断。\n"
        f"- 用户画出的 search_roi: {list(plan.search_roi)}\n"
        f"- 本地脚本选择的 target_roi: {list(plan.target_roi)}\n"
        f"- 本地旧槽位 slot_boxes: {[asdict(slot) for slot in plan.slot_boxes]}\n"
        f"- 本地估计的局部文字倾角 text_angle_degrees: {round(float(plan.text_angle_degrees), 3)}\n"
        f"- 必须保持不变的 protected_boxes: {[list(box) for box in plan.protected_boxes]}\n"
        "- field_label_text、field_separator_text、protected_texts 和 protected_boxes 是本次实际任务上下文；如果为空，不能自行补全成某个固定字段标签。\n"
        "- 必须按阶段验收：先看字体/字号/字槽/基线/笔画粗细/局部倾斜姿态，再看黑度和灰边，再看照片质感和背景修补。\n"
        "- 如果 text_shape 阶段有 hard-blocking issues，不能用降黑、加模糊、加噪声或背景解释来判定通过。\n"
        "- 如果 text_shape.pass=true 但存在 deferred_issues/deferred_to_stage，说明这些形态诊断受黑灰影响；本轮可以按 deferred_to_stage 继续修复，但不能让 hard text shape 指标回退。\n"
        "- 必须检查 target_roi 是否覆盖完整旧文字，而不是覆盖字段标签、字段分隔符或 protected text 碎片。\n"
        "- 如果 source_text 和 target_text 字数不同，也只能改 source_text 所在字段值区域以及必要空白，不能改旧值前后的未修改文字。\n"
        "- 如果旧字擦除后的底色明显发白、过于平滑、像涂抹补丁，必须 pass=false。\n"
        "- 如果旧文字没有被完整清除、仍有任何旧字残留，必须 pass=false。\n"
        "- 如果新文字不在旧文字原位置，而是偏到标签、冒号或其他字段位置，必须 pass=false。\n"
        "- 如果 hard_check_report 中 font_style_gate.pass=false 或 strict_gate.pass=false，必须 pass=false。\n"
        "- 如果 hard_check_report 或当前阶段上下文给出 blocking_stage，视觉建议必须只围绕该阶段允许的参数族；禁止建议 blocked_patch_keys。\n"
        f"{stage_context_text}"
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


def prior_stage_regression_report(
    before_report: dict[str, Any] | None,
    after_report: dict[str, Any] | None,
    current_stage: str | None,
) -> dict[str, Any]:
    if not current_stage or current_stage not in STAGE_ORDER:
        return {
            "current_stage": current_stage,
            "prior_stage_ids": [],
            "pass": True,
            "regressions": [],
            "stage_severity": {},
        }
    prior_stage_ids = list(STAGE_ORDER[: STAGE_ORDER.index(current_stage)])
    stage_severity: dict[str, dict[str, Any]] = {}
    regressions: list[dict[str, Any]] = []
    for stage_id in prior_stage_ids:
        before_issues = stage_issues(before_report, stage_id)
        after_issues = stage_issues(after_report, stage_id)
        before_severity = float(stage_issue_severity(before_report, stage_id))
        after_severity = float(stage_issue_severity(after_report, stage_id))
        before_pass = not before_issues
        after_pass = not after_issues
        record = {
            "before": round(before_severity, 3),
            "after": round(after_severity, 3),
            "delta": round(after_severity - before_severity, 3),
            "before_pass": before_pass,
            "after_pass": after_pass,
        }
        stage_severity[stage_id] = record
        if before_pass and (not after_pass or after_severity > before_severity + 1.0):
            regressions.append(
                {
                    "stage_id": stage_id,
                    "before_severity": record["before"],
                    "after_severity": record["after"],
                    "delta": record["delta"],
                    "after_issue_types": [
                        str(issue.get("type") or "")
                        for issue in after_issues
                        if isinstance(issue, dict)
                    ],
                }
            )
    return {
        "current_stage": current_stage,
        "prior_stage_ids": prior_stage_ids,
        "pass": not regressions,
        "regressions": regressions,
        "stage_severity": stage_severity,
    }


def build_candidate_rejection_table(
    revision_attempts: list[dict[str, Any]],
    current_blocking_stage: str | None,
) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for attempt in revision_attempts:
        if not isinstance(attempt, dict):
            continue
        rejection_reason = "selectable"
        strict_pass = bool(attempt.get("strict_pass"))
        stage_pass = bool(attempt.get("stage_pass"))
        prior_regression = attempt.get("prior_stage_regression")
        prior_pass = bool(prior_regression.get("pass")) if isinstance(prior_regression, dict) else True
        if not strict_pass:
            rejection_reason = "strict_gate_failed"
        elif not stage_pass:
            rejection_reason = "stage_gate_failed" if attempt.get("blocking_stage") else "stage_gate_blocked"
        elif not prior_pass:
            rejection_reason = "prior_stage_regression"
        elif not (
            attempt.get("progresses_past_text_shape")
            or attempt.get("progresses_past_current_stage")
            or attempt.get("improves_current_stage")
            or (attempt.get("ink_guard") or {}).get("selectable")
        ):
            rejection_reason = "no_selectable_progress"

        entry = {
            "candidate_id": (attempt.get("params") or {}).get("candidate_id", ""),
            "origin": attempt.get("origin", "unknown"),
            "primary_stage": attempt.get("stage_id") or attempt.get("current_blocking_stage") or current_blocking_stage,
            "optimization_step": attempt.get("optimization_step", ""),
            "strict_pass": strict_pass,
            "stage_pass": stage_pass,
            "blocking_stage": attempt.get("blocking_stage"),
            "current_stage_severity_before": attempt.get("current_stage_severity_before", 0),
            "current_stage_severity_after": attempt.get("current_stage_severity_after", 0),
            "prior_stage_regression": not prior_pass,
            "vision_target_alignment": attempt.get("vision_target_alignment"),
            "vision_target_completion": attempt.get("vision_target_completion"),
            "non_regression_guard": attempt.get("non_regression_guard"),
            "vision_target_recipe": attempt.get("vision_target_recipe"),
            "selectable": rejection_reason == "selectable",
            "rejection_reason": rejection_reason,
        }
        table.append(entry)
    return table


def select_vision_rendered_candidates(
    rendered: list[tuple[CandidateParams, Image.Image, dict[str, Any], float]],
    limit: int,
) -> list[tuple[CandidateParams, Image.Image, dict[str, Any], float]]:
    effective_limit = max(1, int(limit or 1))
    stage_passed = [
        item for item in rendered
        if report_strict_pass(item[2]) and report_stage_pass(item[2])
    ]
    if stage_passed:
        if any(report_is_longer_replacement(item[2]) for item in stage_passed):
            selected: list[tuple[CandidateParams, Image.Image, dict[str, Any], float]] = []
            seen: set[str] = set()

            def add(items: list[tuple[CandidateParams, Image.Image, dict[str, Any], float]], count: int | None = None) -> None:
                for params, image, report, score in items:
                    if params.candidate_id in seen:
                        continue
                    selected.append((params, image, report, score))
                    seen.add(params.candidate_id)
                    if len(selected) >= effective_limit:
                        return
                    if count is not None and sum(1 for item in selected if item[0].candidate_id in seen) >= count:
                        return

            add(stage_passed[: min(3, effective_limit)])
            mid_blur_alpha = [
                item for item in stage_passed
                if 0.40 <= float(item[0].blur) <= 0.50 and float(item[0].alpha_contrast) >= 0.20
            ]
            add(mid_blur_alpha[:3])
            add(stage_passed)
            return selected[:effective_limit]
        return stage_passed[:effective_limit]

    strict_passed = [item for item in rendered if report_strict_pass(item[2])]
    if strict_passed:
        return strict_passed[:effective_limit]
    return rendered[:effective_limit]


def report_is_longer_replacement(report: dict[str, Any]) -> bool:
    roi_plan = report.get("roi_plan")
    if not isinstance(roi_plan, dict):
        return False
    try:
        source_count = int(roi_plan.get("source_slot_count") or 0)
        target_count = int(roi_plan.get("target_slot_count") or 0)
    except (TypeError, ValueError):
        return False
    return bool(source_count and target_count > source_count)


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
    max_revision_rounds: int = 8,
    pipeline_profile: str = "photo_scan",
    progress: ProgressCallback | None = None,
) -> tuple[CandidateParams | None, dict[str, Any]]:
    if not rendered:
        return None, {"enabled": True, "error": "no candidates for vision review"}

    master_prompt, candidate_prompt_template, final_prompt_template = prompts
    region_dir.mkdir(parents=True, exist_ok=True)
    context_box = region_context_box(plan.search_roi, original.size)
    original_context_path = region_dir / "vision_original_context.png"
    save_region_context(original, context_box, original_context_path)

    effective_candidate_limit = normalize_vision_candidate_limit(candidate_limit, len(rendered))
    vision_rendered = select_vision_rendered_candidates(rendered, effective_candidate_limit)
    vision_sheet_path = region_dir / "vision_candidate_sheet.png"
    sheet_items = [
        (
            params_label(params),
            compare_region_preview(original, candidate, plan.search_roi, scale=3),
        )
        for params, candidate, _report, _score in vision_rendered
    ]
    make_contact_sheet(sheet_items, vision_sheet_path, scale=1, cols=1)

    hard_reports = vision_candidate_request_payload(
        compact_hard_reports(vision_rendered, plan),
        pipeline_profile=pipeline_profile,
        requested_vision_candidate_limit=int(candidate_limit or 0),
        total_candidate_count=len(rendered),
    )
    vision_request_path = region_dir / "vision_candidate_request.json"
    write_json(vision_request_path, hard_reports)
    if progress:
        progress(
            "vision_candidate_request_ready",
            {
                "candidate_count": hard_reports.get("candidate_count"),
                "vision_candidate_limit": hard_reports.get("vision_candidate_limit"),
                "requested_vision_candidate_limit": hard_reports.get("requested_vision_candidate_limit"),
                "total_candidate_count": hard_reports.get("total_candidate_count"),
                "candidate_count_within_limit": hard_reports.get("candidate_count_within_limit"),
                "stage_context_complete": hard_reports.get("stage_context_complete"),
                "request_path": str(vision_request_path),
            },
        )
    prompt = candidate_prompt_template.replace(
        "{hard_check_report}",
        json.dumps(hard_reports, ensure_ascii=False, indent=2),
    )
    prompt += processing_prompt_context(
        plan,
        {
            "pipeline_profile": pipeline_profile,
            "stage_context_by_candidate": hard_reports.get("stage_context_by_candidate"),
        },
    )
    prompt += STRICT_ACCEPTANCE_APPENDIX
    candidate_rank_json = vision_client.call_json(
        system_prompt=master_prompt,
        user_prompt=prompt,
        image_paths=[original_context_path, vision_sheet_path],
        prompt_name="candidate_rank_prompt.txt",
        audit_path=region_dir / "visual_eval_candidate_rank_prompt_audit.json",
    )
    candidate_rank_json["local_stage_context"] = {
        "pipeline_profile": pipeline_profile,
        "stage_context_by_candidate": hard_reports.get("stage_context_by_candidate"),
        "stage_filter_contract": hard_reports.get("stage_filter_contract"),
    }
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
                "field_key": plan.field_key,
                "field_label_text": plan.field_label_text,
                "field_separator_text": plan.field_separator_text,
                "protected_texts": list(plan.protected_texts),
                "pipeline_profile": pipeline_profile,
                "stage_context": model_stage_context(report, pipeline_profile),
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
        final_prompt += processing_prompt_context(
            plan,
            model_stage_context(report, pipeline_profile),
        )
        final_prompt += STRICT_ACCEPTANCE_APPENDIX
        final_json = vision_client.call_json(
            system_prompt=master_prompt,
            user_prompt=final_prompt,
            image_paths=[original_context_path, context_path, compare_path],
            prompt_name="final_acceptance_prompt.txt",
            audit_path=out_path.with_name(f"{out_path.stem}_prompt_audit.json"),
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
                **stage_progress_fields(final_report),
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
        current_shape_parent_candidate_id = current_params.candidate_id
        rank_patch = candidate_rank_json.get("suggested_patch")
        max_revision_rounds = max(1, int(max_revision_rounds))
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
            basis_stage_optimization_policy = stage_optimization_summary(
                str(basis_blocking_stage) if basis_blocking_stage else None
            )
            previous_vision_targets = [
                round_record.get("vision_target")
                for round_record in revision_rounds
                if isinstance(round_record, dict)
            ]
            vision_target = vision_target_from_acceptance(
                current_report,
                current_acceptance,
                prior_targets=previous_vision_targets,
                round_index=round_idx,
                basis_candidate_id=current_params.candidate_id,
            )
            vision_recipe_patches = vision_target_recipe_patches(vision_target)
            vision_recipe_report = vision_target_recipe_report(vision_target)
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
                        "vision_disagreement": bool(vision_target.get("active")),
                        "vision_target": vision_target,
                        "stage_optimization_policy": basis_stage_optimization_policy,
                        "selected_optimization_step": basis_stage_optimization_policy.get("optimization_step"),
                        **stage_progress_fields(current_report),
                    },
                )
            model_records: list[dict[str, Any]] = []
            model_stage_response_contracts: list[dict[str, Any]] = []
            if round_idx == 1:
                model_records.extend(
                    model_patch_records(current_params, candidate_rank_json, source="candidate_rank")
                )
                model_stage_response_contracts.append(
                    {
                        "source": "candidate_rank",
                        **model_stage_response_contract(
                            candidate_rank_json,
                            str(basis_blocking_stage) if basis_blocking_stage else None,
                        ),
                    }
                )
            current_acceptance_source = f"final_acceptance_basis_round_{round_idx - 1}"
            model_records.extend(
                model_patch_records(
                    current_params,
                    current_acceptance,
                    source=current_acceptance_source,
                )
            )
            model_stage_response_contracts.append(
                {
                    "source": current_acceptance_source,
                    **model_stage_response_contract(
                        current_acceptance,
                        str(basis_blocking_stage) if basis_blocking_stage else None,
                    ),
                }
            )
            model_filter = filter_model_patch_records(
                model_records,
                str(basis_blocking_stage) if basis_blocking_stage else None,
            )
            model_records = [
                record
                for record in model_filter.get("records", [])
                if isinstance(record, dict)
            ]
            model_conflicts = [
                record
                for record in model_filter.get("rejected_records", [])
                if isinstance(record, dict)
            ]
            allowed_model_patches = [
                patch
                for patch in model_filter.get("allowed_patches", [])
                if isinstance(patch, dict)
            ]
            model_combo_report = combined_model_suggestion_patch(
                model_filter,
                source=current_acceptance_source,
            )
            model_combo_patch = (
                model_combo_report.get("patch")
                if model_combo_report.get("enabled") and isinstance(model_combo_report.get("patch"), dict)
                else None
            )
            if model_combo_patch and patch_signature(model_combo_patch) not in {
                patch_signature(patch) for patch in allowed_model_patches
            }:
                allowed_model_patches.append(model_combo_patch)
            patch_source_lookup = {
                str(signature): [
                    record
                    for record in records
                    if isinstance(record, dict)
                ]
                for signature, records in (model_filter.get("patch_source_lookup") or {}).items()
            }
            if model_combo_patch:
                patch_source_lookup.setdefault(patch_signature(model_combo_patch), []).append(
                    {
                        "source": current_acceptance_source,
                        "kind": "combined_parameter_suggestions",
                        "patch": model_combo_patch,
                        "combines_records": model_combo_report.get("record_count"),
                        "optimization_policy": model_combo_report.get("optimization_policy"),
                    }
                )
            vision_recipe_lookup = {
                patch_signature(patch): {
                    "source": "vision_target_recipe",
                    "vision_target": vision_target,
                    "combo_recipe": vision_target.get("combo_recipe"),
                    "patch": patch,
                }
                for patch in vision_recipe_patches
                if isinstance(patch, dict)
            }
            patch_dispatch_report = dispatch_revision_patches(
                current_params,
                current_acceptance,
                current_report,
                rank_patch=rank_patch if round_idx == 1 and isinstance(rank_patch, dict) else None,
                extra_patches=[*allowed_model_patches, *vision_recipe_patches],
            )
            round_patches = [
                patch
                for patch in patch_dispatch_report.get("patches", [])
                if isinstance(patch, dict)
            ]
            local_stage_filter_report = patch_dispatch_report.get("stage_filter_report") or {}
            rejected_local_patches = [
                patch
                for patch in local_stage_filter_report.get("rejected_patches", [])
                if isinstance(patch, dict)
            ]
            forced_seed_report = forced_model_seed_audit(
                model_records,
                allowed_model_patches,
                seen_params,
                local_stage_filter_report,
            )
            forced_seed_entries = forced_seed_report.get("forced_seeds", [])
            shape_candidate_grid = text_shape_reset_candidate_grid(
                current_params,
                font_style_reference,
                plan,
                current_report,
                limit=48,
            )
            shape_reset_params = shape_candidate_grid.candidates
            ink_candidate_grid = ink_gray_candidate_grid(
                current_params,
                current_report,
                limit=16,
                parent_shape_candidate_id=current_shape_parent_candidate_id,
            )
            ink_gray_params = ink_candidate_grid.candidates
            if report_blocks_text_shape(current_report):
                ink_guard_candidate_grid = ink_gray_candidate_grid(
                    current_params,
                    current_report,
                    limit=8,
                    parent_shape_candidate_id=current_shape_parent_candidate_id,
                    allow_text_shape_guard=True,
                )
            else:
                ink_guard_candidate_grid = InkGrayCandidateGrid(
                    candidates=[],
                    report={
                        "enabled": False,
                        "reason": "current_stage_is_not_text_shape",
                        "stage_id": "ink_gray_balance",
                        "guards_stage": "text_shape",
                        "guard_mode": "text_shape_excess_black_core",
                        "candidate_count": 0,
                    },
                )
            ink_guard_params = ink_guard_candidate_grid.candidates
            photo_candidate_grid = photo_texture_candidate_grid(
                current_params,
                current_report,
                limit=6,
            )
            photo_texture_params = photo_candidate_grid.candidates
            hard_boundary_passed = bool(
                (current_report or {}).get("pass")
                and (current_report or {}).get("stage_gate", {}).get("blocking_stage") != "hard_boundary"
            )
            prior_regression = prior_stage_regression_report(
                current_report, current_report, basis_blocking_stage
            ) if basis_blocking_stage else {"pass": True}
            escape_grid = controlled_escape_candidate_grid(
                current_params,
                current_report,
                basis_blocking_stage,
                hard_boundary_passed=hard_boundary_passed,
                prior_stage_pass=prior_regression.get("pass", True),
            )
            escape_candidates = escape_grid.get("candidates", []) if isinstance(escape_grid.get("candidates"), list) else []
            layered_search_report = layered_candidate_search_report(
                shape_candidate_grid.report,
                ink_candidate_grid.report,
                ink_guard_candidate_grid.report,
                photo_candidate_grid.report,
            )
            round_record: dict[str, Any] = {
                "round": round_idx,
                "basis_candidate_id": current_params.candidate_id,
                "basis_acceptance_level": current_acceptance.get("acceptance_level"),
                "basis_final_decision": current_acceptance.get("final_decision"),
                "patch_count": len(round_patches),
                "shape_reset_count": len(shape_reset_params),
                "ink_gray_count": len(ink_gray_params),
                "ink_guard_count": len(ink_guard_params),
                "photo_texture_count": len(photo_texture_params),
                "controlled_escape_count": len(escape_candidates),
                "micro_tuning_count": (
                    len(ink_gray_params)
                    if (ink_candidate_grid.report.get("axes", {}).get("near_threshold_micro_tuning", {}).get("enabled"))
                    else 0
                ),
                "basis_blocking_stage": basis_blocking_stage,
                "basis_stage_source": basis_stage_source,
                "basis_stage_severity": round(float(basis_stage_severity), 3),
                "vision_disagreement": bool(vision_target.get("active")),
                "vision_target": vision_target,
                "vision_target_recipe": vision_recipe_report,
                "non_regression_guard": {
                    "enabled": bool(vision_target.get("active")),
                    "axes": vision_target.get("axis_keys") or [],
                    "vision_target_stage": vision_target.get("stage"),
                    "reason": "round-level guard; per-candidate guard is recorded on attempts",
                },
                "stage_optimization_policy": basis_stage_optimization_policy,
                "stage_evidence": stage_progress_fields(current_report),
                "stage_patcher_dispatch": {
                    key: value
                    for key, value in patch_dispatch_report.items()
                    if key not in {"patches", "stage_filter_report"}
                },
                "shape_candidate_grid": shape_candidate_grid.report,
                "ink_gray_candidate_grid": ink_candidate_grid.report,
                "ink_guard_candidate_grid": ink_guard_candidate_grid.report,
                "photo_texture_candidate_grid": photo_candidate_grid.report,
                "layered_candidate_search": layered_search_report,
                "cross_stage_guard_report": {
                    "enabled": bool(ink_guard_params),
                    "guard_mode": "text_shape_excess_black_core",
                    "primary_stage": "text_shape",
                    "guard_stage": "ink_gray_balance",
                    "candidate_count": len(ink_guard_params),
                    "reason": ink_guard_candidate_grid.report.get("reason"),
                    "selection_rule": "shape remains primary; ink guard can continue only if text_shape does not regress and ink severity improves",
                },
                "stage_filter_report": local_stage_filter_report,
                "model_stage_response_contracts": model_stage_response_contracts,
                "model_suggestions": model_records,
                "model_suggestion_filter": model_suggestion_filter_report(model_filter),
                "model_suggestion_combo": model_combo_report,
                "model_suggestion_attempts": model_filter.get("attempt_records") or [],
                "model_conflicts": model_conflicts,
                "rejected_local_patches": rejected_local_patches,
                "controlled_escape_grid": {
                    "enabled": escape_grid.get("enabled"),
                    "reason": escape_grid.get("reason"),
                    "controlled_escape": escape_grid.get("controlled_escape"),
                    "primary_stage": escape_grid.get("primary_stage"),
                    "secondary_stage": escape_grid.get("secondary_stage"),
                    "candidate_count": escape_grid.get("candidate_count"),
                    "escape_limit": escape_grid.get("escape_limit"),
                    "cross_stage_cartesian_disabled": escape_grid.get("cross_stage_cartesian_disabled"),
                },
                "forced_model_seed_count": forced_seed_report.get("forced_seed_count", 0),
                "forced_model_seed_audit": forced_seed_report.get("audited_suggestions", []),
            }
            continuation_contract = revision_round_continuation_contract(
                round_record,
                max_revision_rounds=max_revision_rounds,
            )
            round_record["revision_continuation_contract"] = continuation_contract
            if progress:
                progress(
                    "revision_round_candidates",
                    {
                        "round": round_idx,
                        "patch_count": len(round_patches),
                        "shape_reset_count": len(shape_reset_params),
                        "ink_gray_count": len(ink_gray_params),
                        "ink_guard_count": len(ink_guard_params),
                        "photo_texture_count": len(photo_texture_params),
                        "micro_tuning_count": (
                            len(ink_gray_params)
                            if (ink_candidate_grid.report.get("axes", {}).get("near_threshold_micro_tuning", {}).get("enabled"))
                            else 0
                        ),
                        "controlled_escape_count": len(escape_candidates),
                        "forced_model_seed_count": forced_seed_report.get("forced_seed_count", 0),
                        "basis_blocking_stage": basis_blocking_stage,
                        "basis_stage_source": basis_stage_source,
                        "basis_stage_severity": round(float(basis_stage_severity), 3),
                        "vision_disagreement": bool(vision_target.get("active")),
                        "vision_target": vision_target,
                        "vision_target_recipe": vision_recipe_report,
                        "model_suggestion_combo": model_combo_report,
                        "stage_optimization_policy": basis_stage_optimization_policy,
                        "selected_optimization_step": basis_stage_optimization_policy.get("optimization_step"),
                        "layered_candidate_search": layered_search_report,
                        "revision_continuation_contract": continuation_contract,
                        **stage_progress_fields(current_report),
                    },
                )
            if not continuation_contract["continuation_allowed"]:
                round_record["stop_reason"] = "no_stage_specific_candidate_direction"
                revision_rounds.append(round_record)
                break
            if (
                not round_patches
                and not shape_reset_params
                and not ink_gray_params
                and not ink_guard_params
                and not photo_texture_params
            ):
                round_record["stop_reason"] = "no_revision_candidates"
                revision_rounds.append(round_record)
                break

            round_candidates: list[
                tuple[float, float, CandidateParams, Image.Image, dict[str, Any], dict[str, Any]]
            ] = []

            candidate_jobs: list[tuple[str, int, dict[str, Any] | None, CandidateParams, dict[str, Any]]] = []
            for shape_idx, shape_params in enumerate(shape_reset_params, start=1):
                candidate_jobs.append(("shape_reset", shape_idx, None, shape_params, {"applied": False, "reason": "none", "changes": {}}))
            for ink_idx, ink_params in enumerate(ink_gray_params, start=1):
                candidate_jobs.append(
                    (
                        "ink_gray_grid",
                        ink_idx,
                        None,
                        ink_params,
                        {
                            "applied": False,
                            "reason": "ink_gray_grid",
                            "changes": {},
                            "parent_candidate_id": current_params.candidate_id,
                        },
                    )
                )
            for guard_idx, guard_params in enumerate(ink_guard_params, start=1):
                candidate_jobs.append(
                    (
                        "ink_guard_grid",
                        guard_idx,
                        None,
                        guard_params,
                        {
                            "applied": False,
                            "reason": "text_shape_excess_black_core_ink_guard",
                            "changes": {},
                            "parent_candidate_id": current_params.candidate_id,
                        },
                    )
                )
            for photo_idx, photo_params in enumerate(photo_texture_params, start=1):
                candidate_jobs.append(
                    (
                        "photo_texture_grid",
                        photo_idx,
                        None,
                        photo_params,
                        {
                            "applied": False,
                            "reason": "photo_texture_grid",
                            "changes": {},
                            "parent_candidate_id": current_params.candidate_id,
                        },
                    )
                )
            for escape_idx, escape_params in enumerate(escape_candidates, start=1):
                candidate_jobs.append(
                    (
                        "controlled_escape",
                        escape_idx,
                        None,
                        escape_params,
                        {
                            "applied": False,
                            "reason": "controlled_cross_stage_escape",
                            "changes": {},
                            "parent_candidate_id": current_params.candidate_id,
                            "escape_context": {
                                "primary_stage": escape_grid.get("primary_stage"),
                                "secondary_stage": escape_grid.get("secondary_stage"),
                                "controlled_escape": True,
                            },
                        },
                    )
                )
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
            for seed_idx, seed_entry in enumerate(forced_seed_entries, start=1):
                seed_patch = seed_entry.get("converted_patch")
                if not isinstance(seed_patch, dict):
                    continue
                raw_seed_params = apply_suggested_patch(current_params, seed_patch)
                constrained_seed_params = constrained_revision_params(
                    raw_seed_params,
                    current_params,
                    current_acceptance,
                    current_report,
                    round_idx=round_idx,
                )
                seed_audit = constraint_audit(
                    raw_seed_params,
                    constrained_seed_params,
                    current_report,
                    current_acceptance,
                )
                if seed_audit.get("applied"):
                    seed_audit["alternative_candidate_id"] = ""
                forced_seed_context = {
                    "source": seed_entry.get("source"),
                    "kind": seed_entry.get("kind"),
                    "raw_suggestion": seed_entry.get("raw_suggestion"),
                    "converted_patch": seed_patch,
                    "constrained_patch": dict(seed_audit.get("changes") or {}),
                    "constraint_reason": seed_audit.get("reason"),
                }
                candidate_jobs.append(("forced_model_seed", seed_idx, seed_patch, constrained_seed_params, {**seed_audit, "forced_seed_context": forced_seed_context}))

            for candidate_origin, candidate_idx, patch, patched_params, patch_constraint_audit in candidate_jobs:
                suffix = "i"
                if candidate_origin == "shape_reset":
                    suffix = "s"
                elif candidate_origin == "ink_gray_grid":
                    suffix = "g"
                elif candidate_origin == "photo_texture_grid":
                    suffix = "p"
                patched_params = mutate_params(
                    patched_params,
                    candidate_id=f"{current_params.candidate_id}_{suffix}{round_idx:02d}_{candidate_idx:02d}",
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
                    pipeline_profile=pipeline_profile,
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
                    vision_target=vision_target,
                )
                target_alignment = vision_target_alignment(
                    vision_target,
                    patched_params,
                    current_params,
                )
                target_guard = non_regression_guard_report(
                    vision_target,
                    target_alignment,
                )
                target_completion = vision_target_alignment_complete(
                    vision_target,
                    target_alignment,
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
                progresses_past_current_stage = progresses_past_blocking_stage(
                    patched_report,
                    str(current_blocking_stage) if current_blocking_stage else None,
                    str(patched_blocking_stage) if patched_blocking_stage else None,
                )
                current_stage_improvement = 0.0
                current_stage_severity_before = 0.0
                current_stage_severity_after = 0.0
                ink_guard_selection = text_shape_ink_guard_selectable(
                    current_report,
                    patched_report,
                ) if candidate_origin == "ink_guard_grid" else {
                    "enabled": False,
                    "selectable": False,
                }
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
                prior_regression = prior_stage_regression_report(
                    current_report,
                    patched_report,
                    str(current_blocking_stage) if current_blocking_stage else None,
                )
                attempt_record = {
                    "index": len(revision_attempts) + 1,
                    "round": round_idx,
                    "origin": candidate_origin,
                    "round_candidate": candidate_idx,
                    "basis_candidate_id": current_params.candidate_id,
                    "stage_id": current_blocking_stage,
                    "params": asdict(patched_params),
                    "strict_pass": patched_strict,
                    "stage_pass": report_stage_pass(patched_report),
                    "blocking_stage": patched_blocking_stage,
                    "progresses_past_text_shape": progresses_past_text_shape,
                    "progresses_past_current_stage": progresses_past_current_stage,
                    "improves_current_stage": improves_current_stage,
                    "current_blocking_stage": current_blocking_stage,
                    "current_stage_severity_before": round(float(current_stage_severity_before), 3),
                    "current_stage_severity_after": round(float(current_stage_severity_after), 3),
                    "current_stage_improvement": round(float(current_stage_improvement), 3),
                    "prior_stage_regression": prior_regression,
                    "score": round(float(patched_score), 3),
                    "selection_score": round(float(patched_selection_score), 3),
                    "vision_target_alignment": target_alignment,
                    "vision_target_completion": target_completion,
                    "non_regression_guard": target_guard,
                }
                if patch is not None:
                    optimization_policy = optimization_policy_audit(
                        str(current_blocking_stage) if current_blocking_stage else None,
                        patch,
                    )
                    attempt_record["patch"] = patch
                    attempt_record["optimization_policy"] = optimization_policy
                    attempt_record["optimization_steps"] = optimization_policy.get("optimization_steps")
                    attempt_record["optimization_step"] = selected_optimization_step(optimization_policy)
                    suggestion_records = patch_source_lookup.get(patch_signature(patch), [])
                    if suggestion_records:
                        attempt_record["model_suggestions"] = suggestion_records
                    recipe_record = vision_recipe_lookup.get(patch_signature(patch))
                    if recipe_record:
                        attempt_record["vision_target_recipe"] = recipe_record
                    if patch_constraint_audit.get("applied"):
                        patch_constraint_audit["alternative_candidate_id"] = patched_params.candidate_id
                    attempt_record["constraint"] = patch_constraint_audit
                elif candidate_origin == "shape_reset":
                    optimization_policy = {
                        **stage_optimization_summary(str(current_blocking_stage) if current_blocking_stage else None),
                        "optimization_steps": ["shape_reset"],
                        "primary_optimization_steps": ["shape_reset"],
                        "optimization_step": "shape_reset",
                        "allowed": current_blocking_stage == "text_shape",
                        "rejection_reason": None if current_blocking_stage == "text_shape" else "shape reset is only generated for text_shape",
                    }
                    attempt_record["optimization_policy"] = optimization_policy
                    attempt_record["optimization_steps"] = optimization_policy.get("optimization_steps")
                    attempt_record["optimization_step"] = "shape_reset"
                elif candidate_origin == "ink_gray_grid":
                    optimization_policy = {
                        **stage_optimization_summary(str(current_blocking_stage) if current_blocking_stage else None),
                        "optimization_steps": ["ink_gray_balance"],
                        "primary_optimization_steps": ["ink_gray_balance"],
                        "optimization_step": "ink_gray_balance",
                        "allowed": current_blocking_stage == "ink_gray_balance",
                        "rejection_reason": (
                            None
                            if current_blocking_stage == "ink_gray_balance"
                            else "ink-gray grid is only generated for ink_gray_balance"
                        ),
                    }
                    attempt_record["optimization_policy"] = optimization_policy
                    attempt_record["optimization_steps"] = optimization_policy.get("optimization_steps")
                    attempt_record["optimization_step"] = "ink_gray_balance"
                    attempt_record["parent_shape_candidate_id"] = current_params.candidate_id
                elif candidate_origin == "ink_guard_grid":
                    optimization_policy = {
                        **stage_optimization_summary(str(current_blocking_stage) if current_blocking_stage else None),
                        "optimization_steps": ["ink_gray_balance_guard"],
                        "primary_optimization_steps": ["text_shape"],
                        "optimization_step": "ink_guard",
                        "allowed": current_blocking_stage == "text_shape",
                        "guarded_stage": "ink_gray_balance",
                        "guard_mode": "text_shape_excess_black_core",
                        "rejection_reason": (
                            None
                            if current_blocking_stage == "text_shape"
                            else "ink guard is only generated while text_shape is the primary blocking stage"
                        ),
                    }
                    attempt_record["optimization_policy"] = optimization_policy
                    attempt_record["optimization_steps"] = optimization_policy.get("optimization_steps")
                    attempt_record["optimization_step"] = "ink_guard"
                    attempt_record["guarded_stage"] = "ink_gray_balance"
                    attempt_record["ink_guard"] = ink_guard_selection
                    attempt_record["parent_shape_candidate_id"] = current_params.candidate_id
                elif candidate_origin == "photo_texture_grid":
                    optimization_policy = {
                        **stage_optimization_summary(str(current_blocking_stage) if current_blocking_stage else None),
                        "optimization_steps": ["photo_texture"],
                        "primary_optimization_steps": ["photo_texture"],
                        "optimization_step": "photo_texture",
                        "allowed": current_blocking_stage == "photo_texture",
                        "rejection_reason": (
                            None
                            if current_blocking_stage == "photo_texture"
                            else "photo texture grid is only generated for photo_texture"
                        ),
                    }
                    attempt_record["optimization_policy"] = optimization_policy
                    attempt_record["optimization_steps"] = optimization_policy.get("optimization_steps")
                    attempt_record["optimization_step"] = "photo_texture"
                elif candidate_origin == "controlled_escape":
                    patch_audit = patch_constraint_audit or {}
                    escape_context = patch_audit.get("escape_context") if isinstance(patch_audit, dict) else {}
                    optimization_policy = {
                        **stage_optimization_summary(str(current_blocking_stage) if current_blocking_stage else None),
                        "optimization_steps": ["controlled_escape"],
                        "primary_optimization_steps": ["controlled_escape"],
                        "optimization_step": "controlled_escape",
                        "allowed": True,
                        "controlled_escape": True,
                        "primary_stage": escape_context.get("primary_stage"),
                        "secondary_stage": escape_context.get("secondary_stage"),
                        "rejection_reason": None,
                    }
                    attempt_record["optimization_policy"] = optimization_policy
                    attempt_record["optimization_steps"] = optimization_policy.get("optimization_steps")
                    attempt_record["optimization_step"] = "controlled_escape"
                    attempt_record["controlled_escape"] = True
                    attempt_record["primary_stage"] = escape_context.get("primary_stage")
                    attempt_record["secondary_stage"] = escape_context.get("secondary_stage")
                elif candidate_origin == "forced_model_seed":
                    patch_audit = patch_constraint_audit or {}
                    forced_seed_context = patch_audit.get("forced_seed_context") if isinstance(patch_audit, dict) else {}
                    optimization_policy = {
                        **stage_optimization_summary(str(current_blocking_stage) if current_blocking_stage else None),
                        "optimization_steps": ["model_suggestion_seed"],
                        "primary_optimization_steps": ["model_suggestion_seed"],
                        "optimization_step": "model_suggestion",
                        "allowed": True,
                        "rejection_reason": None,
                    }
                    attempt_record["optimization_policy"] = optimization_policy
                    attempt_record["optimization_steps"] = optimization_policy.get("optimization_steps")
                    attempt_record["optimization_step"] = "model_suggestion"
                    attempt_record["forced_model_seed"] = {
                        "source": forced_seed_context.get("source"),
                        "kind": forced_seed_context.get("kind"),
                        "raw_suggestion": forced_seed_context.get("raw_suggestion"),
                        "converted_patch": forced_seed_context.get("converted_patch"),
                        "constrained_patch": forced_seed_context.get("constrained_patch"),
                        "constraint_reason": forced_seed_context.get("constraint_reason"),
                        "rendered": True,
                        "selectable": patched_strict and report_stage_pass(patched_report) and prior_regression.get("pass"),
                        "rejection_reason": (
                            None
                            if (patched_strict and report_stage_pass(patched_report) and prior_regression.get("pass"))
                            else (
                                "strict_gate_failed" if not patched_strict
                                else "stage_gate_failed" if not report_stage_pass(patched_report)
                                else "prior_stage_regression" if not prior_regression.get("pass")
                                else "unknown"
                            )
                        ),
                    }
                else:
                    optimization_policy = {
                        **stage_optimization_summary(str(current_blocking_stage) if current_blocking_stage else None),
                        "optimization_steps": [],
                        "primary_optimization_steps": [],
                        "optimization_step": candidate_origin,
                        "allowed": False,
                        "rejection_reason": f"unknown revision candidate origin: {candidate_origin}",
                    }
                    attempt_record["optimization_policy"] = optimization_policy
                    attempt_record["optimization_steps"] = []
                    attempt_record["optimization_step"] = candidate_origin
                if not patched_strict:
                    attempt_record["strict_gate"] = patched_report.get("strict_gate")
                if not report_stage_pass(patched_report):
                    attempt_record["stage_gate"] = patched_report.get("stage_gate")
                revision_attempts.append(attempt_record)
                if (
                    prior_regression.get("pass")
                    and (
                        patched_strict
                        or progresses_past_text_shape
                        or progresses_past_current_stage
                        or improves_current_stage
                        or ink_guard_selection.get("selectable")
                    )
                ):
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
                rejection_table = build_candidate_rejection_table(
                    revision_attempts,
                    str(basis_blocking_stage) if basis_blocking_stage else None,
                )
                round_record["candidate_rejection_table"] = rejection_table
                round_record["candidate_rejection_count"] = len(rejection_table)
                round_record["micro_tuning_count"] = len(ink_gray_params) if ink_candidate_grid.report.get("axes", {}).get("near_threshold_micro_tuning", {}).get("enabled") else 0
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
                elif report_has_excess_black_core(current_report):
                    ink_guard_candidates = [
                        item
                        for item in round_candidates
                        if (
                            item[5].get("origin") == "ink_guard_grid"
                            and (item[5].get("ink_guard") or {}).get("selectable")
                        )
                    ]
                    if ink_guard_candidates:
                        ink_guard_candidates.sort(
                            key=lambda item: (
                                -float((item[5].get("ink_guard") or {}).get("ink_gray_improvement") or 0.0),
                                item[0],
                                item[1],
                            )
                        )
                        selected_tuple = ink_guard_candidates[0]
                        selected_reason = "text_shape_ink_guard_reduces_excess_black_core"
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
            selected_continuation_contract = revision_round_continuation_contract(
                {
                    **round_record,
                    "selected_reason": selected_reason,
                    "selected_optimization_step": attempt_record.get("optimization_step"),
                },
                max_revision_rounds=max_revision_rounds,
            )
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
                        "vision_disagreement": bool(vision_target.get("active")),
                        "vision_target": vision_target,
                        "vision_target_alignment": attempt_record.get("vision_target_alignment"),
                        "vision_target_completion": attempt_record.get("vision_target_completion"),
                        "non_regression_guard": attempt_record.get("non_regression_guard"),
                        "stage_id": attempt_record.get("stage_id"),
                        "selected_optimization_step": attempt_record.get("optimization_step"),
                        "optimization_steps": attempt_record.get("optimization_steps"),
                        "current_blocking_stage": attempt_record.get("current_blocking_stage"),
                        "current_stage_severity_before": attempt_record.get("current_stage_severity_before"),
                        "current_stage_severity_after": attempt_record.get("current_stage_severity_after"),
                        "current_stage_improvement": attempt_record.get("current_stage_improvement"),
                        "accepted": round_delivered,
                        "acceptance_level": patched_acceptance.get("acceptance_level"),
                        "final_decision": patched_acceptance.get("final_decision"),
                        "revision_continuation_contract": selected_continuation_contract,
                        "blocking_stage": (patched_report.get("stage_gate") or {}).get("blocking_stage"),
                        **stage_progress_fields(patched_report),
                    },
                )
            round_record.update(
                {
                    "selected_candidate_id": patched_params.candidate_id,
                    "selected_attempt_index": attempt_record["index"],
                    "selected_score": round(float(patched_score), 3),
                    "selected_selection_score": round(float(patched_selection_score), 3),
                    "selected_reason": selected_reason,
                    "vision_disagreement": bool(vision_target.get("active")),
                    "vision_target": vision_target,
                    "vision_target_alignment": attempt_record.get("vision_target_alignment"),
                    "vision_target_completion": attempt_record.get("vision_target_completion"),
                    "non_regression_guard": attempt_record.get("non_regression_guard"),
                    "stage_id": attempt_record.get("stage_id"),
                    "selected_optimization_step": attempt_record.get("optimization_step"),
                    "optimization_steps": attempt_record.get("optimization_steps"),
                    "current_blocking_stage": attempt_record.get("current_blocking_stage"),
                    "current_stage_severity_before": attempt_record.get("current_stage_severity_before"),
                    "current_stage_severity_after": attempt_record.get("current_stage_severity_after"),
                    "current_stage_improvement": attempt_record.get("current_stage_improvement"),
                    "accepted": round_delivered,
                    "acceptance_level": patched_acceptance.get("acceptance_level"),
                    "final_decision": patched_acceptance.get("final_decision"),
                    "revision_continuation_contract": selected_continuation_contract,
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
                    "stage_id": attempt_record.get("stage_id"),
                    "optimization_step": attempt_record.get("optimization_step"),
                    "optimization_steps": attempt_record.get("optimization_steps"),
                    "patcher_source": attempt_record.get("origin"),
                    "round_candidate": attempt_record.get("round_candidate"),
                    "optimization_policy": attempt_record.get("optimization_policy"),
                    "model_suggestions": attempt_record.get("model_suggestions"),
                    "vision_target": vision_target,
                    "vision_target_alignment": attempt_record.get("vision_target_alignment"),
                    "vision_target_completion": attempt_record.get("vision_target_completion"),
                    "non_regression_guard": attempt_record.get("non_regression_guard"),
                    "patch": attempt_record.get("patch"),
                    **candidate_trace_summary(patched_report),
                    "metrics": (patched_report.get("strict_visual_metrics") or {}).get("bands", {}),
                }
            )

            current_params = patched_params
            current_image = patched_image
            current_report = patched_report
            current_score = patched_score
            current_acceptance = patched_acceptance
            if (
                attempt_record.get("origin") == "shape_reset"
                and (patched_report.get("stage_gate") or {}).get("blocking_stage") != "text_shape"
            ):
                current_shape_parent_candidate_id = patched_params.candidate_id
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
        final_font_candidates = final_font_revision_candidates(
            final_params,
            font_style_reference,
            plan,
            final_report,
        )
        if progress and final_font_candidates:
            progress(
                "finalist_revision_started",
                {
                    "candidate_count": len(final_font_candidates),
                    "basis_candidate_id": final_params.candidate_id,
                    "basis_blocking_stage": (final_report.get("stage_gate") or {}).get("blocking_stage"),
                },
            )
        base_alt_index = len(revision_attempts) + 1
        for finalist_index, alt_params in enumerate(final_font_candidates, start=1):
            alt_idx = base_alt_index + finalist_index - 1
            alt_params = mutate_params(
                alt_params,
                candidate_id=f"{final_params.candidate_id}_f{alt_idx:02d}",
            )
            if progress:
                progress(
                    "finalist_revision_candidate_started",
                    {
                        "index": finalist_index,
                        "total": len(final_font_candidates),
                        "candidate_id": alt_params.candidate_id,
                        "font_name": alt_params.font_name,
                        "font_size": alt_params.font_size,
                    },
                )
            alt_image = render_candidate(original, plan, alt_params)
            alt_report = candidate_report(
                original,
                alt_image,
                plan,
                alt_params,
                font_style_reference,
                pipeline_profile=pipeline_profile,
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
                if progress:
                    progress(
                        "finalist_revision_candidate_finished",
                        {
                            "index": finalist_index,
                            "total": len(final_font_candidates),
                            "candidate_id": alt_params.candidate_id,
                            "strict_pass": False,
                            "accepted": False,
                            "blocking_stage": (alt_report.get("stage_gate") or {}).get("blocking_stage"),
                            "score": round(float(alt_score), 3),
                        },
                    )
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
            alt_delivered = final_acceptance_delivers(alt_acceptance)
            if progress:
                progress(
                    "finalist_revision_candidate_finished",
                    {
                        "index": finalist_index,
                        "total": len(final_font_candidates),
                        "candidate_id": alt_params.candidate_id,
                        "strict_pass": True,
                        "accepted": alt_delivered,
                        "acceptance_level": alt_acceptance.get("acceptance_level"),
                        "final_decision": alt_acceptance.get("final_decision"),
                        "blocking_stage": alt_acceptance.get("blocking_stage")
                        or (alt_report.get("stage_gate") or {}).get("blocking_stage"),
                        "score": round(float(alt_score), 3),
                    },
                )
            if alt_delivered:
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
        if progress and final_font_candidates:
            progress(
                "finalist_revision_finished",
                {
                    "candidate_count": len(final_font_candidates),
                    "accepted": accepted,
                    "final_candidate_id": final_params.candidate_id,
                    "blocking_stage": (final_report.get("stage_gate") or {}).get("blocking_stage"),
                },
            )

    final_stage_gate = final_report.get("stage_gate") if isinstance(final_report, dict) else {}
    if not isinstance(final_stage_gate, dict):
        final_stage_gate = stage_gate_for_report(final_report)
    final_vision_target = vision_target_from_acceptance(
        final_report,
        final_acceptance_json,
        prior_targets=[
            round_record.get("vision_target")
            for round_record in revision_rounds
            if isinstance(round_record, dict)
        ],
        round_index=len(revision_rounds) + 1,
        basis_candidate_id=final_params.candidate_id,
    )
    final_target_alignment = vision_target_alignment(
        final_vision_target,
        final_params,
        final_params,
    )
    final_non_regression_guard = non_regression_guard_report(
        final_vision_target,
        final_target_alignment,
    )
    final_target_completion = vision_target_alignment_complete(
        final_vision_target,
        final_target_alignment,
    )
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
            "vision_disagreement": bool(final_vision_target.get("active")),
            "vision_target": final_vision_target,
            "vision_target_completion": final_target_completion,
            "non_regression_guard": final_non_regression_guard,
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
        "vision_disagreement": bool(final_vision_target.get("active")),
        "vision_target": final_vision_target,
        "vision_target_completion": final_target_completion,
        "non_regression_guard": final_non_regression_guard,
        "next_round_plan": next_round_plan,
        "revision_attempts": revision_attempts,
        "revision_rounds": revision_rounds,
        "artifacts": {
            "original_context": str(original_context_path),
            "candidate_sheet": str(vision_sheet_path),
            "final_context": str(final_context_path),
            "final_compare": str(final_compare_path),
            "vision_prompt_audits": sorted(str(path) for path in region_dir.glob("*prompt_audit.json")),
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
    shape_change = report.get("shape_change_report")
    return {
        **stage_progress_fields(report),
        "blocking_stage": blocking_stage,
        "stage_pass": bool(stage_gate.get("pass")) if isinstance(stage_gate, dict) else None,
        "stage_severity": round(float(stage_issue_severity(report, blocking_stage)), 3),
        "placement_strategy": report.get("placement_strategy"),
        "placement_strategy_reason": report.get("placement_strategy_reason"),
        "shape_change_large": (
            bool(shape_change.get("shape_change_large"))
            if isinstance(shape_change, dict)
            else None
        ),
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


def save_stage_candidate_evidence(
    original: Image.Image,
    rendered: list[tuple[CandidateParams, Image.Image, dict[str, Any], float]],
    plan: RenderPlan,
    region_dir: Path,
    *,
    pipeline_profile: str,
) -> dict[str, Any]:
    evidence_dir = region_dir / "stage_evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    stage_records: dict[str, Any] = {}
    tracked_stages = ("text_shape", "ink_gray_balance", "photo_texture", "background_cleanup")
    for stage_id in tracked_stages:
        stage_candidates = [
            item
            for item in rendered
            if ((item[2].get("stage_gate") or {}).get("blocking_stage") == stage_id)
        ]
        if not stage_candidates:
            stage_records[stage_id] = {
                "available": False,
                "reason": "no_candidate_blocked_at_stage",
            }
            continue
        params, candidate, report, score = min(stage_candidates, key=lambda item: item[3])
        compare_path = evidence_dir / f"{stage_id}_top_compare.png"
        report_path = evidence_dir / f"{stage_id}_top_report.json"
        compare_region_preview(original, candidate, plan.search_roi, scale=4).save(compare_path)
        record = {
            "available": True,
            "stage_id": stage_id,
            "candidate_id": params.candidate_id,
            "label": params_label(params),
            "score": round(float(score), 3),
            "compare_path": str(compare_path),
            "report_path": str(report_path),
            "stage_context": model_stage_context(report, pipeline_profile),
            "trace": candidate_trace_summary(report),
            "params": asdict(params),
            "strict_gate": report.get("strict_gate"),
            "shape_change_report": report.get("shape_change_report"),
            "row_baseline_metrics": report.get("row_baseline_metrics"),
            "placement_strategy_report": report.get("placement_strategy_report"),
            "ink_gray_metrics": report.get("char_gray_band_metrics"),
            "photo_texture_metrics": report.get("photo_texture_metrics"),
            "background_texture_metrics": report.get("background_texture_metrics"),
            "extra_source_slot_cleanup_metrics": report.get("extra_source_slot_cleanup_metrics"),
        }
        write_json(report_path, record)
        stage_records[stage_id] = record
    summary_path = evidence_dir / "summary.json"
    summary = {
        "pipeline_profile": pipeline_profile,
        "stage_order": list(STAGE_ORDER),
        "stages": stage_records,
    }
    write_json(summary_path, summary)
    return {
        "directory": str(evidence_dir),
        "summary": str(summary_path),
        "stages": stage_records,
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
    max_revision_rounds: int = 8,
    pipeline_profile: str = "photo_scan",
    progress: ProgressCallback | None = None,
    field_context: dict[str, Any] | None = None,
    classification: dict[str, Any] | None = None,
) -> tuple[Image.Image, Image.Image, list[dict[str, Any]], dict[str, Any], bool]:
    region_dir = run_dir / "regions" / re.sub(r"[^A-Za-z0-9_.-]+", "_", region_id or "region")
    region_dir.mkdir(parents=True, exist_ok=True)
    plan = build_region_plan(
        original,
        roi,
        source_text=source_text,
        target_text=target_text,
        field_key=str((field_context or {}).get("field_key") or (field_context or {}).get("field") or "") or None,
        field_label_text=str((field_context or {}).get("field_label_text") or "") or None,
        field_separator_text=str((field_context or {}).get("field_separator_text") or "") or None,
        protected_texts=tuple(str(item) for item in ((field_context or {}).get("protected_texts") or []) if str(item)),
    )
    image_classification = classification or {}
    region_roi_policy = classify_region_roi_policy(
        image_classification=image_classification,
        search_roi=plan.search_roi,
        edit_roi=plan.target_roi,
        source_text=source_text,
    )
    region_classification = with_region_roi_policy(
        image_classification,
        roi_policy=region_roi_policy,
        internal_profile=pipeline_profile,
    )
    pipeline_profile = str(region_classification.get("internal_profile") or pipeline_profile)
    length_report = (
        plan.slot_quality_report.get("length_change_report")
        if isinstance(plan.slot_quality_report, dict)
        else {}
    )
    if not isinstance(length_report, dict):
        length_report = {}
    roi_plan = {
        "search_roi": list(plan.search_roi),
        "edit_roi": list(plan.target_roi),
        "expanded_edit_roi": length_report.get("expanded_edit_roi"),
        "target_roi_before_length_policy": length_report.get("target_roi_before_length_policy"),
        "target_roi_after_length_policy": length_report.get("target_roi_after_length_policy") or list(plan.target_roi),
        "expansion_report": length_report.get("expansion_report"),
        "roi_policy": region_roi_policy,
        "source_slot_count": len(plan.slot_boxes),
        "target_slot_count": len(text_chars(target_text)),
    }
    source_count = len(text_chars(source_text))
    target_count = len(text_chars(target_text))
    raw_slot_report = plan.slot_quality_report if isinstance(plan.slot_quality_report, dict) else {}
    slot_report = {
        **raw_slot_report,
        "classification": region_classification,
        "class_key": region_classification.get("class_key"),
        "roi_policy": region_classification.get("roi_policy"),
        "internal_profile": region_classification.get("internal_profile"),
        "profile_source": region_classification.get("profile_source"),
        "roi_plan": roi_plan,
        "expanded_edit_roi": roi_plan.get("expanded_edit_roi"),
    }
    plan = replace(plan, slot_quality_report=slot_report)
    pre_candidate_report = pre_candidate_gate_report(
        candidate_count=0,
        regions=[{"id": region_id, "roi": list(roi)}],
        slot_quality_report=slot_report,
    )
    slot_quality_report_path = region_dir / "slot_quality_report.json"
    pre_candidate_gate_report_path = region_dir / "pre_candidate_gate_report.json"
    roi_plan_report_path = region_dir / "roi_plan_report.json"
    write_json(slot_quality_report_path, slot_report)
    write_json(pre_candidate_gate_report_path, pre_candidate_report)
    write_json(
        roi_plan_report_path,
        {
            **roi_plan,
            "classification": region_classification,
            "class_key": region_classification.get("class_key"),
            "internal_profile": region_classification.get("internal_profile"),
            "profile_source": region_classification.get("profile_source"),
        },
    )
    if source_count and not slot_report.get("pass", False):
        rejected_compare_path = region_dir / "slot_quality_rejected_compare.png"
        compare_region_preview(original, original, roi).save(rejected_compare_path)
        if progress:
            progress(
                "pre_candidate_gate_failed",
                {
                    "region_id": region_id,
                    "pipeline_profile": pipeline_profile,
                    "classification": region_classification,
                    "class_key": region_classification.get("class_key"),
                    "roi_policy": region_classification.get("roi_policy"),
                    "internal_profile": region_classification.get("internal_profile"),
                    "profile_source": region_classification.get("profile_source"),
                    "roi_plan": roi_plan,
                    "candidate_count": 0,
                    "failed_gate": pre_candidate_report["failed_gate"],
                    "pre_candidate_gate_report": pre_candidate_report,
                },
            )
            progress(
                "slot_quality_failed",
                {
                    "region_id": region_id,
                    "pipeline_profile": pipeline_profile,
                    "classification": region_classification,
                    "class_key": region_classification.get("class_key"),
                    "roi_policy": region_classification.get("roi_policy"),
                    "internal_profile": region_classification.get("internal_profile"),
                    "profile_source": region_classification.get("profile_source"),
                    "roi_plan": roi_plan,
                    "blocking_stage": "hard_boundary",
                    "slot_quality_report": slot_report,
                    "pre_candidate_gate_report": pre_candidate_report,
                },
            )
        summary = {
            "plan": {
                "search_roi": list(plan.search_roi),
                "target_roi": list(plan.target_roi),
                "slot_boxes": [asdict(item) for item in plan.slot_boxes],
                "protected_boxes": [list(item) for item in plan.protected_boxes],
                "field_key": plan.field_key,
                "field_label_text": plan.field_label_text,
                "field_separator_text": plan.field_separator_text,
                "protected_texts": list(plan.protected_texts),
                "draw_mode": plan.draw_mode,
                "pipeline_profile": pipeline_profile,
                "classification": region_classification,
                "class_key": region_classification.get("class_key"),
                "roi_policy": region_classification.get("roi_policy"),
                "internal_profile": region_classification.get("internal_profile"),
                "profile_source": region_classification.get("profile_source"),
                "roi_plan": roi_plan,
                "expanded_edit_roi": roi_plan.get("expanded_edit_roi"),
                "placement_strategy": plan.placement_strategy,
                "placement_strategy_reason": plan.placement_strategy_reason,
                "slot_quality_report": slot_report,
                "text_angle_degrees": round(float(plan.text_angle_degrees), 3),
            },
            "score": None,
            "hard_check": {
                "pass": False,
                "pipeline_profile": pipeline_profile,
                "stage_gate": {
                    "profile": pipeline_profile,
                    "blocking_stage": "hard_boundary",
                    "pass": False,
                    "stage_status": {
                        "hard_boundary": {
                            "pass": False,
                            "reason": "slot_quality_failed",
                            "issues": slot_report.get("issues") or [],
                        }
                    },
                },
                "slot_quality_report": slot_report,
                "pre_candidate_gate_report": pre_candidate_report,
            },
            "vision": {
                "enabled": False,
                "accepted": False,
                "reason": "slot_quality_failed_before_candidate_generation",
            },
            "trace": {
                "accepted": False,
                "final_is_rejected_candidate": True,
                "final_candidate_id": None,
                "final_blocking_stage": "hard_boundary",
                "final_stage_severity": None,
                "revision_round_count": 0,
                "last_round_stop_reason": "slot_quality_failed",
                "next_round_plan": {
                    "blocking_stage": "hard_boundary",
                    "stage_source": "slot_quality_gate",
                    "actions": [
                        "重新选择只覆盖旧值文字和必要空白的 ROI；不能包含字段标签或右侧未修改文字。"
                    ],
                },
                "pre_candidate_gate_report": pre_candidate_report,
            },
            "accepted": False,
            "applied": False,
            "artifacts": {
                "selected_candidate": None,
                "selected_compare": str(rejected_compare_path),
                "slot_quality_report": str(slot_quality_report_path),
                "pre_candidate_gate_report": str(pre_candidate_gate_report_path),
                "display_image_is_candidate": True,
                "roi_plan_report": str(roi_plan_report_path),
            },
            "rejected_fonts": [],
        }
        return original.copy(), original.copy(), [], summary, False
    font_candidates = find_font_candidates(font_source="recommended")
    font_candidates, rejected_fonts = filter_fonts_by_required_text(
        font_candidates,
        f"{source_text}{target_text}",
    )
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
            placement_strategy=plan.placement_strategy,
            placement_strategy_reason=plan.placement_strategy_reason,
            slot_quality_report=plan.slot_quality_report,
            field_key=plan.field_key,
            field_label_text=plan.field_label_text,
            field_separator_text=plan.field_separator_text,
            protected_texts=plan.protected_texts,
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
        params_list.extend(
            longer_replacement_soft_scan_candidates(
                current,
                font_candidates=font_candidates,
                font_style_reference=font_style_reference,
                max_font_size=max_font_size,
            )
        )
        params_list = dedupe_params(params_list, max_candidates + 80)
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
            (0.50, 0.58, 0.00, 0.00, 0.00, 0.00, 0.00, 185, 2, 1),
            (0.50, 0.62, 0.00, 0.00, 0.00, 0.00, 0.00, 185, 2, 1),
            (0.52, 0.58, 0.00, 0.00, 0.00, 0.00, 0.00, 185, 2, 1),
            (0.50, 0.78, 0.00, 0.00, 0.00, 0.00, 0.00, 185, 2, 1),
            (0.52, 0.74, 0.00, 0.00, 0.00, 0.00, 0.00, 185, 2, 1),
            (0.54, 0.70, 0.00, 0.00, 0.00, 0.00, 0.00, 185, 2, 1),
            (0.56, 0.68, 0.00, 0.00, 0.00, 0.00, 0.00, 185, 2, 1),
            (0.56, 0.72, 0.00, 0.00, 0.00, 0.00, 0.00, 195, 3, 1),
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
                "classification": region_classification,
                "class_key": region_classification.get("class_key"),
                "roi_policy": region_classification.get("roi_policy"),
                "internal_profile": region_classification.get("internal_profile"),
                "profile_source": region_classification.get("profile_source"),
                "pipeline_profile": pipeline_profile,
                "roi_plan": roi_plan,
                "candidate_count": len(params_list),
            },
        )
    for params in params_list:
        candidate = render_candidate(original, plan, params)
        report = candidate_report(
            original,
            candidate,
            plan,
            params,
            font_style_reference,
            pipeline_profile=pipeline_profile,
        )
        report["classification"] = region_classification
        report["class_key"] = region_classification.get("class_key")
        report["roi_policy"] = region_classification.get("roi_policy")
        report["internal_profile"] = region_classification.get("internal_profile")
        report["profile_source"] = region_classification.get("profile_source")
        report["roi_plan"] = roi_plan
        score = region_candidate_score(original, candidate, plan, report)
        if not report_strict_pass(report):
            score += 10000.0
        rendered.append((params, candidate, report, score))

    if not rendered:
        raise RuntimeError("no candidate could be rendered")

    rendered.sort(key=lambda item: item[3])
    stage_evidence = save_stage_candidate_evidence(
        original,
        rendered,
        plan,
        region_dir,
        pipeline_profile=pipeline_profile,
    )
    if progress:
        progress(
            "region_candidates_finished",
            {
                "region_id": region_id,
                "rendered": len(rendered),
                "best_score": round(float(rendered[0][3]), 3) if rendered else None,
                "classification": region_classification,
                "class_key": region_classification.get("class_key"),
                "roi_policy": region_classification.get("roi_policy"),
                "internal_profile": region_classification.get("internal_profile"),
                "profile_source": region_classification.get("profile_source"),
                "roi_plan": roi_plan,
                "stage_evidence": {
                    "summary": stage_evidence.get("summary"),
                    "available_stages": [
                        stage_id
                        for stage_id, record in (stage_evidence.get("stages") or {}).items()
                        if isinstance(record, dict) and record.get("available")
                    ],
                },
                **stage_progress_fields(rendered[0][2] if rendered else None),
            },
        )
    def vision_progress(event: str, fields: dict[str, Any]) -> None:
        if progress:
            progress(event, {"region_id": region_id, **(fields or {})})

    chosen_params, vision_summary = run_region_vision_checks(
        original=original,
        rendered=rendered,
        plan=plan,
        region_dir=region_dir,
        vision_client=vision_client,
        prompts=prompts,
        candidate_limit=vision_candidate_limit,
        font_style_reference=font_style_reference,
        max_revision_rounds=max_revision_rounds,
        pipeline_profile=pipeline_profile,
        progress=vision_progress,
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
            pipeline_profile=pipeline_profile,
        )
        best_report["classification"] = region_classification
        best_report["class_key"] = region_classification.get("class_key")
        best_report["roi_policy"] = region_classification.get("roi_policy")
        best_report["internal_profile"] = region_classification.get("internal_profile")
        best_report["profile_source"] = region_classification.get("profile_source")
        best_report["roi_plan"] = roi_plan
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
                "classification": region_classification,
                "class_key": region_classification.get("class_key"),
                "roi_policy": region_classification.get("roi_policy"),
                "internal_profile": region_classification.get("internal_profile"),
                "selection_reason": preview.get("selected_reason"),
                "patcher_source": preview.get("patcher_source"),
                "optimization_policy": preview.get("optimization_policy")
                if isinstance(preview.get("optimization_policy"), dict)
                else {},
                "model_suggestions": preview.get("model_suggestions")
                if isinstance(preview.get("model_suggestions"), list)
                else [],
                "patch": preview.get("patch") if isinstance(preview.get("patch"), dict) else {},
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
                "patcher_source": "initial_local_rank",
                "optimization_policy": stage_optimization_summary(
                    str((report.get("stage_gate") or {}).get("blocking_stage") or "")
                    or None
                ),
                "model_suggestions": [],
                "patch": {},
                "classification": region_classification,
                "class_key": region_classification.get("class_key"),
                "roi_policy": region_classification.get("roi_policy"),
                "internal_profile": region_classification.get("internal_profile"),
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
    vision_target = (
        vision_summary.get("vision_target")
        if isinstance(vision_summary.get("vision_target"), dict)
        else None
    )
    non_regression_guard = (
        vision_summary.get("non_regression_guard")
        if isinstance(vision_summary.get("non_regression_guard"), dict)
        else None
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
                "field_key": plan.field_key,
                "field_label_text": plan.field_label_text,
                "field_separator_text": plan.field_separator_text,
                "protected_texts": list(plan.protected_texts),
                "draw_mode": plan.draw_mode,
                "pipeline_profile": pipeline_profile,
                "classification": region_classification,
                "class_key": region_classification.get("class_key"),
                "roi_policy": region_classification.get("roi_policy"),
                "internal_profile": region_classification.get("internal_profile"),
                "profile_source": region_classification.get("profile_source"),
                "roi_plan": roi_plan,
                "expanded_edit_roi": roi_plan.get("expanded_edit_roi"),
                "stage_status": (best_report.get("stage_gate") or {}).get("stage_status"),
                "placement_strategy": plan.placement_strategy,
                "placement_strategy_reason": plan.placement_strategy_reason,
                "slot_quality_report": plan.slot_quality_report,
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
                "vision_disagreement": bool(vision_summary.get("vision_disagreement")),
                "vision_target": vision_target,
                "non_regression_guard": non_regression_guard,
                "next_round_plan": vision_next_plan,
            },
            "accepted": accepted,
            "applied": accepted,
            "artifacts": {
                "selected_candidate": str(selected_candidate_path),
                "selected_compare": str(selected_compare_path),
                "slot_quality_report": str(slot_quality_report_path),
                "pre_candidate_gate_report": str(pre_candidate_gate_report_path),
                "roi_plan_report": str(roi_plan_report_path),
                "stage_evidence": stage_evidence,
                "display_image_is_candidate": not accepted,
            },
            "rejected_fonts": rejected_fonts,
        },
        accepted,
    )
