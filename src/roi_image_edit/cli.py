from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from roi_image_edit.environment import (
    environment_report,
    install_recommended_fonts,
    print_report,
)
from roi_image_edit.iterative_pipeline import (
    DEFAULT_ENV,
    DEFAULT_MAX_ITERATIONS,
    run_pipeline,
)
from roi_image_edit.processing_service import image_to_data_url, process_payload
from roi_image_edit.stage_profiles import stage_profile_choices


def parse_rect(value: str) -> dict[str, int]:
    parts = [part.strip() for part in str(value or "").split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("rect must be x,y,w,h")
    try:
        x, y, w, h = [int(round(float(part))) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("rect must contain numbers: x,y,w,h") from exc
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("rect width and height must be positive")
    return {"x": x, "y": y, "w": w, "h": h}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="roi-image-edit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check-env", help="Check dependencies, packaged prompts, API config, and recommended fonts.")
    check.add_argument("--env", type=Path, default=DEFAULT_ENV)
    check.add_argument("--metadata", type=Path, default=None)
    check.add_argument("--json", action="store_true", dest="as_json")

    install = subparsers.add_parser("install-fonts", help="Install recommended CJK fonts where possible.")
    install.add_argument("--json", action="store_true", dest="as_json")

    run = subparsers.add_parser("run", help="Run the iterative ROI replacement pipeline.")
    run.add_argument("--metadata", type=Path, required=True)
    run.add_argument("--env", type=Path, default=DEFAULT_ENV)
    run.add_argument("--output-dir", type=Path, default=Path("output"))
    run.add_argument("--logs-dir", type=Path, default=Path("logs"))
    run.add_argument("--text", default=None)
    run.add_argument("--source-text", default=None)
    run.add_argument("--roi", default=None, help="Manual crop-local target ROI x1,y1,x2,y2.")
    run.add_argument("--font-path", default=None)
    run.add_argument("--font-source", choices=("recommended", "scan"), default="recommended")
    run.add_argument("--font-scan-dirs", action="append", default=None)
    run.add_argument("--max-scanned-fonts", type=int, default=500)
    run.add_argument("--font-candidate-pool-size", type=int, default=4)
    run.add_argument("--vision-candidate-limit", type=int, default=48)
    run.add_argument("--font-size", type=int, default=0)
    run.add_argument("--opacity", type=float, default=0.86)
    run.add_argument("--blur", type=float, default=0.5)
    run.add_argument("--stroke-opacity", type=float, default=0.0)
    run.add_argument("--ink-gain", type=float, default=0.0)
    run.add_argument("--alpha-contrast", type=float, default=0.0)
    run.add_argument("--core-ink-gain", type=float, default=0.0)
    run.add_argument("--core-darken-strength", type=float, default=0.0)
    run.add_argument("--core-darken-threshold", type=int, default=120)
    run.add_argument("--core-darken-target-gray", type=int, default=28)
    run.add_argument("--mask-threshold", type=int, default=165)
    run.add_argument("--mask-dilate-iterations", type=int, default=2)
    run.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    run.add_argument("--max-candidates", type=int, default=28)
    run.add_argument("--vision", choices=("on", "auto", "off"), default="auto")
    run.add_argument("--acceptance-mode", choices=("strict", "normal"), default="strict")
    run.add_argument("--profile", choices=stage_profile_choices(), default="photo_scan")
    run.add_argument("--max-dark-pixel-ratio", type=float, default=1.12)
    run.add_argument("--min-dark-pixel-ratio", type=float, default=0.88)
    run.add_argument("--max-core-mean-gray-delta", type=float, default=18.0)
    run.add_argument("--max-edge-mean-gray-delta", type=float, default=16.0)
    run.add_argument("--max-core-lighten-delta", type=float, default=2.0)
    run.add_argument("--max-edge-lighten-delta", type=float, default=4.0)
    run.add_argument("--max-char-center-dx", type=float, default=2.0)
    run.add_argument("--max-char-center-distance-delta", type=float, default=2.0)
    run.add_argument("--max-font-style-score-ratio", type=float, default=1.25)
    run.add_argument("--max-model-local-score-delta", type=float, default=3.0)
    run.add_argument("--max-model-font-style-ratio-delta", type=float, default=0.05)
    run.add_argument("--blur-score-weight", type=float, default=160.0)
    run.add_argument("--blur-score-free-margin", type=float, default=0.15)
    run.add_argument("--font-ranking", choices=("style", "document"), default="style")
    run.add_argument("--font-style-min-size", type=int, default=16)
    run.add_argument("--font-style-max-size", type=int, default=28)
    run.add_argument("--style-reference-text", default=None)
    run.add_argument("--prefer-serif-fonts", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--enforce-font-similarity", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--use-tuning-prompt", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--sheet-scale", type=int, default=8)
    run.add_argument("--sheet-cols", type=int, default=4)
    run.add_argument("--compare-scale", type=int, default=10)
    run.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)

    process = subparsers.add_parser("process", help="Process one image from an instruction, with optional auto ROI.")
    process.add_argument("--image", type=Path, required=True, help="Input image path.")
    process.add_argument("--instruction", required=True, help="Replacement instruction, e.g. 字段 旧文字调整为新文字.")
    process.add_argument(
        "--rect",
        action="append",
        type=parse_rect,
        default=None,
        help="Manual ROI as x,y,w,h. Repeat for multiple regions. Omit to auto-select ROI from instruction.",
    )
    process.add_argument("--max-candidates", type=int, default=130)
    process.add_argument("--vision-candidate-limit", type=int, default=8)
    process.add_argument("--max-revision-rounds", type=int, default=12)
    process.add_argument("--profile", choices=stage_profile_choices(), default="photo_scan")
    process.add_argument("--output", type=Path, default=None, help="Optional path to copy the final image.")
    process.add_argument("--json", action="store_true", dest="as_json")

    return parser


def build_process_summary(
    response: dict,
    image_result: dict,
    final_path: Path | None,
) -> dict:
    artifacts = image_result.get("artifacts") or {}
    return {
        "ok": image_result.get("ok"),
        "accepted": image_result.get("accepted", False),
        "applied": image_result.get("applied", False),
        "error": image_result.get("error"),
        "run_dir": response.get("runDir"),
        "final_image": str(final_path) if final_path else None,
        "applied_image": artifacts.get("applied"),
        "final_is_rejected_candidate": artifacts.get("final_is_rejected_candidate", False),
        "orientation": image_result.get("orientation"),
        "instruction_details": image_result.get("instructionDetails"),
        "regions": [
            {
                "id": region.get("id"),
                "roi": region.get("roi"),
                "auto": region.get("auto", False),
                "source_text": region.get("sourceText"),
                "target_text": region.get("targetText"),
                "accepted": region.get("accepted"),
                "target_roi": (region.get("summary") or {}).get("plan", {}).get("target_roi"),
                "params": (region.get("summary") or {}).get("params"),
                "vision": (region.get("summary") or {}).get("vision", {}),
                "next_round_plan": ((region.get("summary") or {}).get("vision") or {}).get("next_round_plan"),
            }
            for region in image_result.get("regions", [])
        ],
    }


def _format_progress_list(value: object) -> str:
    if isinstance(value, (list, tuple)):
        items = [str(item) for item in value if str(item)]
        return ",".join(items) if items else "none"
    if value is None:
        return "none"
    return str(value)


def progress_selected_optimization_step(record: dict) -> object:
    direct = record.get("selected_optimization_step")
    if direct:
        return direct
    policy = record.get("stage_optimization_policy")
    if isinstance(policy, dict):
        return policy.get("optimization_step")
    return None


def progress_blocking_stage(record: dict) -> object:
    return (
        record.get("blocking_stage")
        or record.get("current_blocking_stage")
        or record.get("basis_blocking_stage")
        or record.get("stage_id")
    )


def format_process_progress_line(event: str, record: dict) -> str | None:
    if event in {"revision_round_started", "revision_round_candidates", "revision_round_finished"}:
        pieces = [
            "[progress]",
            event,
            f"round={record.get('round')}",
            f"profile={record.get('pipeline_profile')}",
            f"blocking_stage={progress_blocking_stage(record)}",
            f"reason={record.get('blocking_stage_reason') or record.get('selected_reason') or record.get('stop_reason')}",
            f"allowed_params={_format_progress_list(record.get('allowed_patch_keys'))}",
            f"blocked_params={_format_progress_list(record.get('blocked_patch_keys'))}",
            f"selected_optimization_step={progress_selected_optimization_step(record)}",
        ]
        if event == "revision_round_candidates":
            pieces.extend(
                [
                    f"patches={record.get('patch_count')}",
                    f"shape_resets={record.get('shape_reset_count')}",
                    f"basis_severity={record.get('basis_stage_severity')}",
                ]
            )
        if event == "revision_round_finished":
            pieces.extend(
                [
                    f"accepted={record.get('accepted')}",
                    f"decision={record.get('final_decision')}",
                    f"score={record.get('score')}",
                ]
            )
        return " ".join(pieces)

    if event in {
        "run_started",
        "image_started",
        "auto_roi_finished",
        "region_started",
        "slot_quality_failed",
        "region_candidates_finished",
        "region_initial_acceptance",
        "finalist_revision_started",
        "finalist_revision_finished",
        "region_finished",
        "image_finished",
    }:
        detail = " ".join(
            f"{key}={value}"
            for key, value in record.items()
            if key not in {"time", "event"}
        )
        return f"[progress] {event} {detail}".rstrip()

    if event == "finalist_revision_candidate_started":
        return (
            "[progress] "
            f"finalist {record.get('index')}/{record.get('total')} "
            f"start candidate={record.get('candidate_id')} "
            f"font={record.get('font_name')} size={record.get('font_size')}"
        )

    if event == "finalist_revision_candidate_finished":
        return (
            "[progress] "
            f"finalist {record.get('index')}/{record.get('total')} "
            f"accepted={record.get('accepted')} "
            f"decision={record.get('final_decision')} "
            f"blocking_stage={record.get('blocking_stage')} "
            f"score={record.get('score')}"
        )
    return None


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "check-env":
        report = environment_report(args.env, args.metadata)
        print_report(report, as_json=args.as_json)
        return

    if args.command == "install-fonts":
        result = install_recommended_fonts()
        if args.as_json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print_report({"dependencies": [], "fonts": result["fonts"], "prompts": [], "api": {}}, as_json=False)
        return

    if args.command == "run":
        summary = run_pipeline(args)
        print(
            json.dumps(
                {
                    "run_dir": summary["run_dir"],
                    "final_crop": summary["final_crop"],
                    "final_full": summary["final_full"],
                    "final_crop_hard_check": summary["final_crop_hard_check"],
                    "final_full_hard_check": summary["final_full_hard_check"],
                    "final_acceptance": summary["final_acceptance"],
                    "progress": summary["progress"],
                    "vision_errors": summary["vision_errors"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "process":
        input_image = Image.open(args.image).convert("RGB")
        regions = [
            {"id": f"region_{idx}", "rect": rect}
            for idx, rect in enumerate(args.rect or [], start=1)
        ]
        payload = {
            "profile": args.profile,
            "maxCandidates": args.max_candidates,
            "visionCandidateLimit": args.vision_candidate_limit,
            "maxRevisionRounds": args.max_revision_rounds,
            "images": [
                {
                    "id": "cli_image",
                    "filename": args.image.name,
                    "instruction": args.instruction,
                    "dataUrl": image_to_data_url(input_image),
                    "regions": regions,
                }
            ],
        }
        def print_progress(event: str, record: dict) -> None:
            if args.as_json:
                return
            line = format_process_progress_line(event, record)
            if line:
                print(line, flush=True)

        response = process_payload(payload, progress=print_progress)
        image_result = response["images"][0]
        final_path = None
        artifacts = image_result.get("artifacts") or {}
        if artifacts.get("final"):
            final_path = Path(artifacts["final"])
        if args.output and final_path:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            Image.open(final_path).save(args.output)
            final_path = args.output

        summary = build_process_summary(response, image_result, final_path)
        if args.as_json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            if not summary["ok"]:
                print(f"处理失败: {summary['error']}")
                print(f"run_dir: {summary['run_dir']}")
            else:
                print(f"accepted: {summary['accepted']}")
                print(f"applied: {summary['applied']}")
                print(f"run_dir: {summary['run_dir']}")
                print(f"final_image: {summary['final_image']}")
                if summary["final_is_rejected_candidate"]:
                    print("final_image_status: rejected_candidate_not_applied")
                    print(f"applied_image: {summary['applied_image']}")
                for region in summary["regions"]:
                    print(
                        "region "
                        f"{region['id']}: roi={region['roi']} auto={region['auto']} "
                        f"accepted={region['accepted']} target_roi={region['target_roi']}"
                    )
                    vision = region.get("vision") or {}
                    rounds = vision.get("revision_rounds") or []
                    if rounds:
                        last_round = rounds[-1]
                        print(
                            "  last_round: "
                            f"stage={last_round.get('current_blocking_stage') or last_round.get('blocking_stage')} "
                            f"reason={last_round.get('selected_reason') or last_round.get('stop_reason')} "
                            f"severity={last_round.get('current_stage_severity_before')}->"
                            f"{last_round.get('current_stage_severity_after')} "
                            f"delta={last_round.get('current_stage_improvement')}"
                        )
                    if region.get("next_round_plan"):
                        plan = region["next_round_plan"]
                        print(
                            "  next_round_plan: "
                            f"stage={plan.get('blocking_stage')} severity={plan.get('stage_severity')} "
                            f"actions={'; '.join(plan.get('actions') or [])}"
                        )
        if not summary["ok"]:
            raise SystemExit(1)
        if not summary["accepted"]:
            raise SystemExit(2)
        return

    parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
