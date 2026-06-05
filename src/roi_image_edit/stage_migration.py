from __future__ import annotations

from copy import deepcopy
from typing import Any

from roi_image_edit.run_artifacts import model_stage_context
from roi_image_edit.stage_patchers import stage_patcher_registry_report
from roi_image_edit.stage_policy import STAGE_ORDER
from roi_image_edit.stages import stage_gate_for_report, stage_spec


STAGE_FAILURE_FIXTURES: dict[str, dict[str, Any]] = {
    "hard_boundary": {
        "pass": False,
        "pipeline_profile": "photo_scan",
        "issues": [{"type": "roi_outside"}],
    },
    "text_shape": {
        "pass": True,
        "pipeline_profile": "photo_scan",
        "font_style_gate": {"issues": [{"type": "font_style_mismatch"}]},
    },
    "ink_gray_balance": {
        "pass": True,
        "pipeline_profile": "photo_scan",
        "local_ink_balance_issues": [{"type": "core_mean_gray_too_light"}],
    },
    "photo_texture": {
        "pass": True,
        "pipeline_profile": "photo_scan",
        "local_photo_texture_issues": [{"type": "photo_texture_too_clean"}],
    },
    "background_cleanup": {
        "pass": True,
        "pipeline_profile": "photo_scan",
        "local_background_texture_issues": [{"type": "background_fill_too_smooth"}],
    },
}
STAGE_PASS_FIXTURE: dict[str, Any] = {
    "pass": True,
    "pipeline_profile": "photo_scan",
    "strict_gate": {"issues": []},
    "local_ink_balance_issues": [],
    "local_photo_texture_issues": [],
    "local_background_texture_issues": [],
}


def stage_migration_contract_report() -> dict[str, Any]:
    patchers = stage_patcher_registry_report()
    stages: dict[str, Any] = {}
    for stage_id in STAGE_ORDER:
        spec = stage_spec(stage_id)
        failure_report = deepcopy(STAGE_FAILURE_FIXTURES[stage_id])
        pass_report = deepcopy(STAGE_PASS_FIXTURE)
        failure_gate = stage_gate_for_report(failure_report, "photo_scan")
        pass_gate = stage_gate_for_report(pass_report, "photo_scan")
        stages[stage_id] = {
            "stage_id": stage_id,
            "detector": spec.detect.__name__,
            "detector_module": spec.detect.__module__,
            "patcher": patchers.get(stage_id),
            "allowed_patch_keys": sorted(spec.allowed_patch_keys),
            "blocked_patch_keys": sorted(spec.blocked_patch_keys),
            "failure_case": {
                "report": failure_report,
                "blocking_stage": failure_gate.get("blocking_stage"),
                "stage_evidence": model_stage_context(failure_report, "photo_scan"),
            },
            "pass_case": {
                "report": pass_report,
                "blocking_stage": pass_gate.get("blocking_stage"),
                "stage_evidence": model_stage_context(pass_report, "photo_scan"),
            },
        }
    return {
        "stage_order": list(STAGE_ORDER),
        "contract": (
            "Each migrated stage must expose a detector, optional stage patcher, "
            "allowed/blocked patch keys, a blocking failure case, a pass case, "
            "and model-facing stage evidence."
        ),
        "stages": stages,
    }
