from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from PIL import Image

from roi_image_edit.iterative_pipeline import write_json


def safe_image_stem(filename: str, image_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(filename).stem)[:80] or image_id or "image"


def failed_image_result(
    *,
    run_dir: Path,
    filename: str,
    image_id: str,
    error: str,
    image: Image.Image | None,
    instruction_details: dict[str, Any] | None,
    pre_candidate_gate_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    safe_stem = safe_image_stem(filename, image_id)
    rejected_input_path: Path | None = None
    if image is not None:
        rejected_input_path = run_dir / f"{safe_stem}_rejected_input.png"
        image.save(rejected_input_path)
    report = {
        "image_id": image_id,
        "filename": filename,
        "error": error,
        "instruction_details": instruction_details,
        "accepted": False,
        "applied": False,
        "candidate_count": 0,
        "failure_stage": "pre_candidate_generation",
        "reason": "image_processing_failed_before_candidate_generation",
        "pre_candidate_gate_report": pre_candidate_gate_report,
    }
    report_path = run_dir / f"{safe_stem}_failure_report.json"
    write_json(report_path, report)
    return {
        "id": image_id,
        "ok": False,
        "accepted": False,
        "applied": False,
        "filename": filename,
        "error": error,
        "instructionDetails": instruction_details,
        "candidates": [],
        "regions": [],
        "stage_evidence": {
            "failure": {
                **report,
                "report_path": str(report_path),
                "rejected_input": str(rejected_input_path) if rejected_input_path else None,
            },
        },
        "artifacts": {
            "rejected_input": str(rejected_input_path) if rejected_input_path else None,
            "failure_report": str(report_path),
            "final_is_rejected_candidate": True,
        },
    }
