from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from roi_image_edit.prompt_contracts import PROMPT_INPUT_CONTRACTS, PROMPT_OUTPUT_FIELD_HANDLING


VISION_PROMPT_AUDIT_SCHEMA_VERSION = 1


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _write_text_artifact(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _unresolved_placeholders(prompt_name: str, user_prompt: str) -> list[str]:
    contract = PROMPT_INPUT_CONTRACTS.get(prompt_name, {})
    placeholders = [f"{{{name}}}" for name in contract.get("formatted_payloads", ())]
    placeholders.extend(("{hard_check_report}", "{current_params}", "{final_params}"))
    return sorted({item for item in placeholders if item in user_prompt})


def prompt_input_presence(prompt_name: str, user_prompt: str) -> dict[str, Any]:
    contract = PROMPT_INPUT_CONTRACTS.get(prompt_name, {})
    groups: dict[str, dict[str, bool]] = {}
    missing: list[str] = []
    for group_name, fields in contract.items():
        group: dict[str, bool] = {}
        for field in fields:
            if group_name == "formatted_payloads":
                present = f"{{{field}}}" not in user_prompt
            else:
                present = str(field) in user_prompt
            group[str(field)] = present
            if not present:
                missing.append(f"{group_name}.{field}")
        groups[group_name] = group
    return {
        "groups": groups,
        "missing_fields": missing,
        "complete": not missing,
    }


def image_audit_items(image_paths: list[Path]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in image_paths:
        path = Path(path)
        try:
            size_bytes = path.stat().st_size
        except OSError:
            size_bytes = None
        items.append(
            {
                "path": str(path),
                "name": path.name,
                "exists": path.exists(),
                "size_bytes": size_bytes,
                "sha256": _sha256_file(path),
            }
        )
    return items


def response_audit(response_json: dict[str, Any] | None = None, error: str | None = None) -> dict[str, Any]:
    if response_json is None:
        return {
            "received": False,
            "json_object": False,
            "top_level_keys": [],
            "sha256": None,
            "error": error,
        }
    canonical = json.dumps(response_json, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "received": True,
        "json_object": isinstance(response_json, dict),
        "top_level_keys": sorted(str(key) for key in response_json.keys()),
        "sha256": _sha256_text(canonical),
        "error": error,
    }


def write_vision_prompt_audit(
    audit_path: Path,
    *,
    prompt_name: str,
    system_prompt: str,
    user_prompt: str,
    image_paths: list[Path],
    model: str | None = None,
    response_json: dict[str, Any] | None = None,
    error: str | None = None,
    fallback_used: bool = False,
    save_prompt_text: bool = True,
) -> dict[str, Any]:
    audit_path = Path(audit_path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    stem = audit_path.with_suffix("")
    system_prompt_path = stem.with_name(f"{stem.name}_system_prompt.txt")
    user_prompt_path = stem.with_name(f"{stem.name}_user_prompt.txt")
    if save_prompt_text:
        _write_text_artifact(system_prompt_path, system_prompt)
        _write_text_artifact(user_prompt_path, user_prompt)

    audit = {
        "schema_version": VISION_PROMPT_AUDIT_SCHEMA_VERSION,
        "created_at_unix": time.time(),
        "prompt_name": prompt_name,
        "model": model,
        "request": {
            "system_prompt_sha256": _sha256_text(system_prompt),
            "system_prompt_chars": len(system_prompt),
            "user_prompt_sha256": _sha256_text(user_prompt),
            "user_prompt_chars": len(user_prompt),
            "unresolved_placeholders": _unresolved_placeholders(prompt_name, user_prompt),
            "input_presence": prompt_input_presence(prompt_name, user_prompt),
            "output_field_handling": PROMPT_OUTPUT_FIELD_HANDLING.get(prompt_name, {}),
            "image_count": len(image_paths),
            "images": image_audit_items(image_paths),
            "prompt_text_artifacts": {
                "system_prompt": str(system_prompt_path) if save_prompt_text else None,
                "user_prompt": str(user_prompt_path) if save_prompt_text else None,
            },
        },
        "transport": {
            "response_format_json_requested": True,
            "temperature_zero_requested": True,
            "fallback_without_response_format_or_temperature": bool(fallback_used),
        },
        "response": response_audit(response_json, error),
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit
