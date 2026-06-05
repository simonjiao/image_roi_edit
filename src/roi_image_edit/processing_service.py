from __future__ import annotations

import base64
import io
import json
import re
import time
from dataclasses import asdict
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

from roi_image_edit.local_validation import (
    apply_local_acceptance_gate,
    candidate_report,
    compact_hard_reports,
    constraint_audit,
    old_region_lt55_pixels,
    opacity_floor_for_excess_black,
    region_candidate_score,
    region_context_box,
    report_needs_wider_gray_strokes,
    report_stage_pass,
    save_region_compare,
    save_region_context,
    stage_issue_severity,
    stage_issues,
)

from roi_image_edit.model_suggestions import (
    filter_model_patch_records,
    model_stage_response_contract,
    model_suggestion_filter_report,
)

from roi_image_edit.revision_solver import (
    constrained_revision_params,
    final_acceptance_delivers,
    final_font_revision_candidates,
    report_blocks_text_shape,
    revision_selection_score,
    text_shape_reset_candidates,
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
    model_stage_context,
    request_audit_payload,
    result_audit_payload,
    stage_progress_fields,
    vision_candidate_request_payload,
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
    stage_context_text = (
        "\n当前阶段上下文 JSON：\n"
        f"{json.dumps(stage_context, ensure_ascii=False, indent=2)}\n"
        if stage_context
        else ""
    )
    return (
        "\n\n动态任务补充：\n"
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

    effective_candidate_limit = max(1, int(candidate_limit or 0))
    vision_rendered = rendered[:effective_candidate_limit]
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
            model_records.extend(
                model_patch_records(
                    current_params,
                    current_acceptance,
                    source=f"final_acceptance_basis_round_{round_idx - 1}",
                )
            )
            model_stage_response_contracts.append(
                {
                    "source": f"final_acceptance_basis_round_{round_idx - 1}",
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
            patch_source_lookup = {
                str(signature): [
                    record
                    for record in records
                    if isinstance(record, dict)
                ]
                for signature, records in (model_filter.get("patch_source_lookup") or {}).items()
            }
            patch_dispatch_report = dispatch_revision_patches(
                current_params,
                current_acceptance,
                current_report,
                rank_patch=rank_patch if round_idx == 1 and isinstance(rank_patch, dict) else None,
                extra_patches=allowed_model_patches,
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
                "stage_optimization_policy": basis_stage_optimization_policy,
                "stage_evidence": stage_progress_fields(current_report),
                "stage_patcher_dispatch": {
                    key: value
                    for key, value in patch_dispatch_report.items()
                    if key not in {"patches", "stage_filter_report"}
                },
                "stage_filter_report": local_stage_filter_report,
                "model_stage_response_contracts": model_stage_response_contracts,
                "model_suggestions": model_records,
                "model_suggestion_filter": model_suggestion_filter_report(model_filter),
                "model_suggestion_attempts": model_filter.get("attempt_records") or [],
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
                        "stage_optimization_policy": basis_stage_optimization_policy,
                        "selected_optimization_step": basis_stage_optimization_policy.get("optimization_step"),
                        **stage_progress_fields(current_report),
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
                    "stage_id": current_blocking_stage,
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
                    if patch_constraint_audit.get("applied"):
                        patch_constraint_audit["alternative_candidate_id"] = patched_params.candidate_id
                    attempt_record["constraint"] = patch_constraint_audit
                else:
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
) -> tuple[Image.Image, Image.Image, list[dict[str, Any]], dict[str, Any], bool]:
    region_dir = run_dir / "regions" / re.sub(r"[^A-Za-z0-9_.-]+", "_", region_id or "region")
    plan = build_region_plan(
        original,
        roi,
        source_text=source_text,
        target_text=target_text,
    )
    source_count = len(text_chars(source_text))
    target_count = len(text_chars(target_text))
    slot_report = plan.slot_quality_report or {}
    if source_count and not slot_report.get("pass", False):
        region_dir.mkdir(parents=True, exist_ok=True)
        rejected_compare_path = region_dir / "slot_quality_rejected_compare.png"
        compare_region_preview(original, original, roi).save(rejected_compare_path)
        if progress:
            progress(
                "slot_quality_failed",
                {
                    "region_id": region_id,
                    "pipeline_profile": pipeline_profile,
                    "blocking_stage": "hard_boundary",
                    "slot_quality_report": slot_report,
                },
            )
        summary = {
            "plan": {
                "search_roi": list(plan.search_roi),
                "target_roi": list(plan.target_roi),
                "slot_boxes": [asdict(item) for item in plan.slot_boxes],
                "protected_boxes": [list(item) for item in plan.protected_boxes],
                "draw_mode": plan.draw_mode,
                "pipeline_profile": pipeline_profile,
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
            },
            "accepted": False,
            "applied": False,
            "artifacts": {
                "selected_candidate": None,
                "selected_compare": str(rejected_compare_path),
                "display_image_is_candidate": True,
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
        report = candidate_report(
            original,
            candidate,
            plan,
            params,
            font_style_reference,
            pipeline_profile=pipeline_profile,
        )
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
                "pipeline_profile": pipeline_profile,
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
                "next_round_plan": vision_next_plan,
            },
            "accepted": accepted,
            "applied": accepted,
            "artifacts": {
                "selected_candidate": str(selected_candidate_path),
                "selected_compare": str(selected_compare_path),
                "stage_evidence": stage_evidence,
                "display_image_is_candidate": not accepted,
            },
            "rejected_fonts": rejected_fonts,
        },
        accepted,
    )


def process_payload(payload: dict[str, Any], progress: ProgressCallback | None = None) -> dict[str, Any]:
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.jsonl"
    profile_resolution = resolve_stage_profile(
        str(payload.get("profile") or ""),
        str(payload.get("profileSuggestion") or ""),
    )
    pipeline_profile = str(profile_resolution["id"])

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
    emit(
        "run_started",
        {
            "run_dir": str(run_dir),
            "pipeline_profile": pipeline_profile,
            "profile_resolution": profile_resolution,
        },
    )
    prompts = load_processing_prompts()
    vision_client = VisionClient(ENV_PATH)
    results: list[dict[str, Any]] = []
    for image_item in payload.get("images", []):
        image_id = str(image_item.get("id") or "")
        filename = str(image_item.get("filename") or "image.png")
        instruction_details: dict[str, Any] | None = None
        failure_image: Image.Image | None = None
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
                    "instruction_details": instruction_details,
                    "source_text": source_text,
                    "target_text": target_text,
                    "pipeline_profile": pipeline_profile,
                },
            )
            image = image_from_data_url(str(image_item.get("dataUrl") or ""))
            failure_image = image.copy()
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
                        "pipeline_profile": pipeline_profile,
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
                    max_revision_rounds=int(payload.get("maxRevisionRounds") or 8),
                    pipeline_profile=pipeline_profile,
                    progress=region_progress,
                )
                display_image = image.copy() if accepted else region_display_image.copy()
                emit(
                    "region_finished",
                    {
                        "image_id": image_id,
                        "region_id": region_id,
                        "accepted": accepted,
                        "pipeline_profile": pipeline_profile,
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
            auto_roi_evidence = auto_roi_evidence_payload(regions)
            auto_roi_overlay_path: Path | None = None
            if auto_roi_evidence["region_count"]:
                auto_roi_overlay_path = run_dir / f"{safe_stem}_auto_roi_overlay.png"
                save_auto_roi_overlay(original_image, regions, auto_roi_overlay_path)
            auto_roi_stage_evidence = {
                **auto_roi_evidence,
                "overlay_path": str(auto_roi_overlay_path) if auto_roi_overlay_path else None,
            }
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
                    "instructionDetails": instruction_details,
                    "sourceDataUrl": image_to_data_url(original_image),
                    "resultDataUrl": image_to_data_url(result_image),
                    "candidates": candidates[:5],
                    "orientation": orientation_summary,
                    "autoRoiEvidence": auto_roi_evidence,
                    "stage_evidence": {
                        "auto_roi": auto_roi_stage_evidence,
                    },
                    "regions": region_results,
                    "artifacts": {
                        "original": str(original_path),
                        "final": str(final_path),
                        "applied": str(applied_path),
                        "auto_roi_overlay": str(auto_roi_overlay_path) if auto_roi_overlay_path else None,
                        "final_is_rejected_candidate": not image_accepted,
                    },
                }
            )
            emit("image_finished", {"image_id": image_id, "accepted": image_accepted})
        except Exception as exc:
            emit("image_failed", {"image_id": image_id, "error": str(exc)})
            results.append(
                failed_image_result(
                    run_dir=run_dir,
                    filename=filename,
                    image_id=image_id,
                    error=str(exc),
                    image=failure_image,
                    instruction_details=instruction_details,
                )
            )
    response = {
        "ok": True,
        "runDir": str(run_dir),
        "profile": pipeline_profile,
        "profileResolution": profile_resolution,
        "images": results,
    }
    write_json(run_dir / "result.json", result_audit_payload(response))
    emit("run_finished", {"ok": True})
    return response
