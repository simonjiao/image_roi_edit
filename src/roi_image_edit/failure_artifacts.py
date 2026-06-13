from __future__ import annotations

import base64
from io import BytesIO
import re
from pathlib import Path
from typing import Any

from PIL import Image

from roi_image_edit.iterative_pipeline import write_json


def safe_image_stem(filename: str, image_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(filename).stem)[:80] or image_id or "image"


def rejected_image_data_url(image: Image.Image | None) -> str | None:
    if image is None:
        return None
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def failed_image_result(
    *,
    run_dir: Path,
    filename: str,
    image_id: str,
    error: str,
    image: Image.Image | None,
    instruction_details: dict[str, Any] | None,
    classification: dict[str, Any] | None = None,
    pre_candidate_gate_report: dict[str, Any] | None = None,
    orientation_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    safe_stem = safe_image_stem(filename, image_id)
    rejected_input_path: Path | None = None
    if image is not None:
        rejected_input_path = run_dir / f"{safe_stem}_rejected_input.png"
        image.save(rejected_input_path)
    image_data_url = rejected_image_data_url(image)
    orientation_report_path: Path | None = None
    if orientation_summary is not None:
        orientation_report_path = run_dir / f"{safe_stem}_auto_orientation_report.json"
        write_json(orientation_report_path, orientation_summary)
    classification_report_path: Path | None = None
    if classification is not None:
        classification_report_path = run_dir / f"{safe_stem}_classification_report.json"
        write_json(classification_report_path, classification)
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
        "orientation_summary": orientation_summary,
        "classification": classification,
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
        "sourceDataUrl": image_data_url,
        "resultDataUrl": image_data_url,
        "instructionDetails": instruction_details,
        "classification": classification,
        "class_key": (classification or {}).get("class_key") if isinstance(classification, dict) else None,
        "roi_policy": (classification or {}).get("roi_policy") if isinstance(classification, dict) else None,
        "internal_profile": (classification or {}).get("internal_profile") if isinstance(classification, dict) else None,
        "profile_source": (classification or {}).get("profile_source") if isinstance(classification, dict) else None,
        "candidates": [],
        "regions": [],
        "stage_evidence": {
            "failure": {
                **report,
                "report_path": str(report_path),
                "rejected_input": str(rejected_input_path) if rejected_input_path else None,
                "orientation_report": str(orientation_report_path) if orientation_report_path else None,
                "classification_report": str(classification_report_path) if classification_report_path else None,
            },
        },
        "artifacts": {
            "rejected_input": str(rejected_input_path) if rejected_input_path else None,
            "failure_report": str(report_path),
            "classification_report": str(classification_report_path) if classification_report_path else None,
            "auto_orientation_report": str(orientation_report_path) if orientation_report_path else None,
            "final_is_rejected_candidate": True,
        },
    }
