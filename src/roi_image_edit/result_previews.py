from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from roi_image_edit.region_processing import image_to_data_url


def rejected_region_preview_candidate(
    *,
    region_id: str,
    summary: dict[str, Any],
    pipeline_profile: str,
) -> dict[str, Any] | None:
    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
    compare_path_value = artifacts.get("selected_compare")
    if not compare_path_value:
        return None
    compare_path = Path(str(compare_path_value))
    if not compare_path.exists():
        return None
    trace = summary.get("trace") if isinstance(summary.get("trace"), dict) else {}
    plan = summary.get("plan") if isinstance(summary.get("plan"), dict) else {}
    try:
        preview = Image.open(compare_path).convert("RGB")
    except OSError:
        return None
    return {
        "index": 1,
        "candidate_id": f"{region_id}_pre_candidate_rejected",
        "regionId": region_id,
        "kind": "rejected_region_preview",
        "label": "拒绝任务对比图",
        "dataUrl": image_to_data_url(preview),
        "pipeline_profile": pipeline_profile,
        "blocking_stage": trace.get("final_blocking_stage") or "hard_boundary",
        "stage_severity": trace.get("final_stage_severity"),
        "score": None,
        "patcher_source": "rejected_region_artifact",
        "placement_strategy": plan.get("placement_strategy"),
        "selection_reason": trace.get("last_round_stop_reason") or "pre_candidate_gate_failed",
    }
