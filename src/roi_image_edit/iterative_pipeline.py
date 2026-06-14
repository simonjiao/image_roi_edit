from __future__ import annotations

import argparse
import base64
import copy
import datetime as dt
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from roi_image_edit.environment import (
    environment_report,
    font_category_for_path,
    font_category_penalty,
    font_missing_text_chars,
    load_dotenv,
    PREFERRED_STYLE_CATEGORIES,
    resolve_recommended_fonts,
    resolve_scanned_fonts,
)
from roi_image_edit.prompt_assets import load_prompt, require_prompts
from roi_image_edit.run_artifacts import (
    attach_stage_context_to_rank_report,
    model_stage_context,
    normalize_vision_candidate_limit,
    vision_candidate_request_payload,
)
from roi_image_edit.stage_profiles import stage_profile, stage_profile_choices
from roi_image_edit.stages import stage_gate_for_report
from roi_image_edit.vision_audit import write_vision_prompt_audit


DEFAULT_ENV = Path(".env")
DEFAULT_MAX_ITERATIONS = 8
PREFERRED_CORE_MEAN_GRAY_DELTA = -1.4


def attach_report_stage_context(report: dict[str, Any], pipeline_profile: str) -> dict[str, Any]:
    report["pipeline_profile"] = pipeline_profile
    report["stage_gate"] = stage_gate_for_report(report, pipeline_profile)
    report["stage_context"] = model_stage_context(report, pipeline_profile)
    return report


@dataclass(frozen=True)
class TextRun:
    x1: int
    y1: int
    x2: int
    y2: int
    area: int


@dataclass(frozen=True)
class CandidateParams:
    candidate_id: str
    font_name: str
    font_path: str
    font_size: int
    opacity: float
    blur: float
    stroke_opacity: float = 0.0
    ink_gain: float = 0.0
    alpha_contrast: float = 0.0
    core_ink_gain: float = 0.0
    core_darken_strength: float = 0.0
    core_darken_threshold: int = 120
    core_darken_target_gray: int = 28
    text_dx: int = 0
    text_dy: int = 0
    char_offsets: tuple[tuple[int, int], ...] = ()
    mask_threshold: int = 165
    mask_dilate_iterations: int = 2
    inpaint_radius: int = 3
    photo_warp: float = 0.0
    edge_breakup: float = 0.0
    photo_noise: float = 0.0
    jpeg_quality: int = 0


@dataclass(frozen=True)
class RenderPlan:
    target_text: str
    source_text: str | None
    search_roi: tuple[int, int, int, int]
    target_roi: tuple[int, int, int, int]
    slot_boxes: tuple[TextRun, ...]
    protected_boxes: tuple[tuple[int, int, int, int], ...]
    source_reference_box: tuple[int, int, int, int] | None
    style_reference_box: tuple[int, int, int, int] | None
    style_reference_text: str | None
    draw_mode: str
    text_angle_degrees: float = 0.0
    placement_strategy: str = "top_left_anchor"
    placement_strategy_reason: str = "default"
    slot_quality_report: dict[str, Any] | None = None
    field_key: str | None = None
    field_label_text: str | None = None
    field_separator_text: str | None = None
    protected_texts: tuple[str, ...] = ()


def intersect_boxes(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def parse_box(value: str | list[int] | tuple[int, ...] | dict[str, int]) -> tuple[int, int, int, int]:
    if isinstance(value, str):
        parts = [int(x.strip()) for x in value.split(",")]
        if len(parts) != 4:
            raise ValueError(f"box must have 4 integers: {value}")
        return parts[0], parts[1], parts[2], parts[3]
    if isinstance(value, dict):
        return (
            int(value["x"]),
            int(value["y"]),
            int(value["x"]) + int(value["width"]),
            int(value["y"]) + int(value["height"]),
        )
    parts = [int(x) for x in value]
    if len(parts) != 4:
        raise ValueError(f"box must have 4 integers: {value}")
    return parts[0], parts[1], parts[2], parts[3]


def add_offset(box: tuple[int, int, int, int], dx: int, dy: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return x1 + dx, y1 + dy, x2 + dx, y2 + dy


def clamp_box(
    box: tuple[int, int, int, int], size: tuple[int, int]
) -> tuple[int, int, int, int]:
    w, h = size
    x1, y1, x2, y2 = box
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class ProgressTracker:
    def __init__(self, run_dir: Path, *, enabled: bool = True) -> None:
        self.run_dir = run_dir
        self.enabled = enabled
        self.progress_path = run_dir / "progress.jsonl"
        self.started_at = time.time()

    def emit(self, event: str, message: str, **payload: Any) -> None:
        record = {
            "time": dt.datetime.now().isoformat(timespec="seconds"),
            "elapsed_seconds": round(time.time() - self.started_at, 3),
            "event": event,
            "message": message,
            **payload,
        }
        with self.progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self.enabled:
            details = " ".join(
                f"{key}={value}" for key, value in payload.items()
                if isinstance(value, (str, int, float, bool))
            )
            suffix = f" {details}" if details else ""
            print(f"[roi-progress] {event}: {message}{suffix}", file=sys.stderr, flush=True)


def image_to_data_url(path: Path) -> str:
    mime = "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def normalize_base_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    return base_url


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def extract_chat_response_text(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first, dict) else {}
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, dict):
                        value = item.get("text")
                        if isinstance(value, str):
                            parts.append(value)
                    elif isinstance(item, str):
                        parts.append(item)
                joined = "\n".join(parts).strip()
                if joined:
                    return joined
            for key in ("reasoning_content", "text"):
                value = message.get(key)
                if isinstance(value, str) and value.strip():
                    return value
        text = first.get("text") if isinstance(first, dict) else None
        if isinstance(text, str) and text.strip():
            return text

    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    choice_keys = []
    message_keys = []
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        choice_keys = sorted(choices[0].keys())
        message = choices[0].get("message")
        if isinstance(message, dict):
            message_keys = sorted(message.keys())
    raise RuntimeError(
        "vision API response missing text content "
        f"(top_keys={sorted(data.keys())}, choice_keys={choice_keys}, message_keys={message_keys})"
    )


class VisionClient:
    def __init__(self, env_path: Path, timeout: int = 120) -> None:
        env = {**load_dotenv(env_path), **os.environ}
        self.api_key = env.get("OPENAI_API_KEY", "")
        self.base_url = normalize_base_url(env.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
        self.model = env.get("OPENAI_JUDGE_MODEL") or env.get("OPENAI_MODEL") or "gpt-5.5"
        self.timeout = timeout
        cache_flag = str(env.get("ROI_IMAGE_EDIT_VISION_CACHE", "1")).strip().lower()
        self.cache_enabled = cache_flag not in {"0", "false", "no", "off"}
        self.cache_dir = Path(env.get("ROI_IMAGE_EDIT_VISION_CACHE_DIR") or ".cache/roi_image_edit/vision")
        if not self.api_key:
            raise RuntimeError(f"OPENAI_API_KEY not found in {env_path}")

    def call_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[Path],
        prompt_name: str | None = None,
        audit_path: Path | None = None,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for path in image_paths:
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(path)}})

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        audit_prompt_name = prompt_name or "unknown_prompt"
        cache_key = self._cache_key(
            prompt_name=audit_prompt_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_paths=image_paths,
        )

        def write_audit(
            *,
            response_json: dict[str, Any] | None = None,
            error: str | None = None,
            fallback_used: bool = False,
            cache_hit: bool = False,
            elapsed_seconds: float | None = None,
        ) -> None:
            if audit_path is None:
                return
            write_vision_prompt_audit(
                audit_path,
                prompt_name=audit_prompt_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_paths=image_paths,
                model=self.model,
                response_json=response_json,
                error=error,
                fallback_used=fallback_used,
                cache_hit=cache_hit,
                cache_key=cache_key,
                elapsed_seconds=elapsed_seconds,
            )

        cached = self._read_cache(cache_key)
        if cached is not None:
            write_audit(response_json=cached, cache_hit=True, elapsed_seconds=0.0)
            return copy.deepcopy(cached)

        started_at = time.time()
        try:
            result = self._post_with_retry(payload)
            self._write_cache(cache_key, result)
            write_audit(response_json=result, elapsed_seconds=round(time.time() - started_at, 3))
            return result
        except RuntimeError as exc:
            if "response_format" not in str(exc) and "temperature" not in str(exc):
                write_audit(error=str(exc), elapsed_seconds=round(time.time() - started_at, 3))
                raise
            fallback = copy.deepcopy(payload)
            fallback.pop("response_format", None)
            fallback.pop("temperature", None)
            try:
                result = self._post_with_retry(fallback)
                self._write_cache(cache_key, result)
                write_audit(response_json=result, fallback_used=True, elapsed_seconds=round(time.time() - started_at, 3))
                return result
            except RuntimeError as fallback_exc:
                write_audit(
                    error=str(fallback_exc),
                    fallback_used=True,
                    elapsed_seconds=round(time.time() - started_at, 3),
                )
                raise

    def _cache_key(
        self,
        *,
        prompt_name: str,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[Path],
    ) -> str:
        image_items: list[dict[str, Any]] = []
        for path in image_paths:
            digest = hashlib.sha256()
            with Path(path).open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            image_items.append({"name": Path(path).name, "sha256": digest.hexdigest()})
        request_identity = {
            "schema_version": 1,
            "base_url": getattr(self, "base_url", ""),
            "model": getattr(self, "model", ""),
            "prompt_name": prompt_name,
            "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
            "user_prompt_sha256": hashlib.sha256(user_prompt.encode("utf-8")).hexdigest(),
            "images": image_items,
        }
        canonical = json.dumps(request_identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _cache_path(self, cache_key: str) -> Path:
        return Path(getattr(self, "cache_dir", Path(".cache/roi_image_edit/vision"))) / f"{cache_key}.json"

    def _read_cache(self, cache_key: str) -> dict[str, Any] | None:
        if not bool(getattr(self, "cache_enabled", False)):
            return None
        cache_path = self._cache_path(cache_key)
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        response = payload.get("response") if isinstance(payload, dict) else None
        return copy.deepcopy(response) if isinstance(response, dict) else None

    def _write_cache(self, cache_key: str, response: dict[str, Any]) -> None:
        if not bool(getattr(self, "cache_enabled", False)):
            return
        cache_path = self._cache_path(cache_key)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "created_at_unix": time.time(),
            "model": self.model,
            "base_url": self.base_url,
            "cache_key": cache_key,
            "response": response,
        }
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _post_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        retry_markers = (
            "HTTP 502",
            "HTTP 503",
            "HTTP 504",
            "temporarily unavailable",
            "connection failed",
            "timed out",
        )
        last_error: RuntimeError | None = None
        for attempt in range(1, 4):
            try:
                return self._post(payload)
            except RuntimeError as exc:
                last_error = exc
                if attempt >= 3 or not any(marker in str(exc) for marker in retry_markers):
                    raise
                time.sleep(1.5 * attempt)
        if last_error is not None:
            raise last_error
        raise RuntimeError("vision API retry loop ended without a response")

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"vision API HTTP {exc.code}: {detail[:1200]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"vision API connection failed: {exc}") from exc

        data = json.loads(body)
        text = extract_chat_response_text(data)
        return extract_json_object(text)


def find_font_candidates(
    manual_font: str | None = None,
    *,
    font_source: str = "recommended",
    scan_dirs: list[str] | tuple[str, ...] | None = None,
    max_scanned_fonts: int = 500,
) -> list[tuple[str, str]]:
    if font_source == "scan":
        return resolve_scanned_fonts(
            manual_font,
            scan_dirs=scan_dirs,
            include_recommended=True,
            max_fonts=max_scanned_fonts,
        )
    return resolve_recommended_fonts(manual_font)


def filter_fonts_by_required_text(
    font_candidates: list[tuple[str, str]],
    required_text: str,
) -> tuple[list[tuple[str, str]], list[dict[str, Any]]]:
    kept: list[tuple[str, str]] = []
    rejected: list[dict[str, Any]] = []
    for name, raw_path in font_candidates:
        missing = font_missing_text_chars(Path(raw_path), required_text)
        if missing:
            rejected.append(
                {
                    "name": name,
                    "path": raw_path,
                    "missing_chars": missing,
                }
            )
            continue
        kept.append((name, raw_path))
    if not kept:
        raise FileNotFoundError(f"No font can render required text: {required_text}")
    return kept, rejected


def gray_array(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def dark_runs(
    img: Image.Image,
    roi: tuple[int, int, int, int],
    threshold: int = 165,
    min_area: int = 4,
) -> list[TextRun]:
    x1, y1, x2, y2 = roi
    gray = gray_array(img)
    mask = gray[y1:y2, x1:x2] < threshold
    cols = np.count_nonzero(mask, axis=0)
    runs: list[TextRun] = []
    in_run = False
    start = 0
    for idx, count in enumerate(cols):
        if count > 0 and not in_run:
            start = idx
            in_run = True
        is_end = in_run and (count == 0 or idx == len(cols) - 1)
        if is_end:
            end = idx if count == 0 else idx + 1
            sub = mask[:, start:end]
            ys, xs = np.where(sub)
            if len(xs) >= min_area:
                runs.append(
                    TextRun(
                        x1=int(x1 + start + xs.min()),
                        y1=int(y1 + ys.min()),
                        x2=int(x1 + start + xs.max() + 1),
                        y2=int(y1 + ys.max() + 1),
                        area=int(len(xs)),
                    )
                )
            in_run = False
    return runs


def is_mostly_cjk(text: str) -> bool:
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return False
    cjk = [ch for ch in chars if "\u4e00" <= ch <= "\u9fff"]
    return len(cjk) / len(chars) >= 0.7


def text_chars(text: str | None) -> list[str]:
    return [ch for ch in (text or "") if not ch.isspace()]


def changed_text_slot_indices(plan: RenderPlan) -> set[int] | None:
    source_chars = text_chars(plan.source_text)
    target_chars = text_chars(plan.target_text)
    if not source_chars or not target_chars or len(source_chars) != len(target_chars):
        return None
    return {idx for idx, (source, target) in enumerate(zip(source_chars, target_chars)) if source != target}


def unchanged_text_slot_mask(plan: RenderPlan, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    mask = np.zeros((h, w), dtype=np.uint8)
    changed_indices = changed_text_slot_indices(plan)
    if changed_indices is None or not plan.slot_boxes:
        return mask

    source_chars = text_chars(plan.source_text)
    target_chars = text_chars(plan.target_text)
    if len(source_chars) != len(target_chars):
        return mask

    tx1, ty1, tx2, ty2 = plan.target_roi
    slots = tuple(sorted(plan.slot_boxes, key=lambda item: item.x1)[: len(target_chars)])
    for idx, slot in enumerate(slots):
        if idx in changed_indices:
            continue
        slot_w = max(1, slot.x2 - slot.x1)
        slot_h = max(1, slot.y2 - slot.y1)
        pad_x = max(2, int(round(slot_w * 0.08)))
        pad_y = max(2, int(round(slot_h * 0.08)))
        x1 = max(tx1, slot.x1 - pad_x)
        y1 = max(ty1, slot.y1 - pad_y)
        x2 = min(tx2, slot.x2 + pad_x)
        y2 = min(ty2, slot.y2 + pad_y)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
    return mask


def split_run_by_projection(
    img: Image.Image,
    run: TextRun,
    *,
    parts: int,
    threshold: int,
) -> tuple[TextRun, ...]:
    if parts <= 1 or run.x2 - run.x1 < parts:
        return (run,)

    gray = gray_array(img)
    mask = gray[run.y1 : run.y2, run.x1 : run.x2] < threshold
    width = run.x2 - run.x1
    split_points: list[int] = []
    if parts == 2:
        left = max(1, int(width * 0.35))
        right = min(width - 1, int(width * 0.70))
        cols = np.count_nonzero(mask, axis=0)
        if right > left:
            local = cols[left:right]
            split_points.append(left + int(np.argmin(local)))
        else:
            split_points.append(width // 2)
    else:
        split_points.extend(round(width * i / parts) for i in range(1, parts))

    bounds = [0, *split_points, width]
    split_runs: list[TextRun] = []
    for start, end in zip(bounds, bounds[1:]):
        if end <= start:
            continue
        sub = mask[:, start:end]
        ys, xs = np.where(sub)
        if len(xs) == 0:
            split_runs.append(
                TextRun(
                    x1=int(run.x1 + start),
                    y1=run.y1,
                    x2=int(run.x1 + end),
                    y2=run.y2,
                    area=0,
                )
            )
        else:
            split_runs.append(
                TextRun(
                    x1=int(run.x1 + start + xs.min()),
                    y1=int(run.y1 + ys.min()),
                    x2=int(run.x1 + start + xs.max() + 1),
                    y2=int(run.y1 + ys.max() + 1),
                    area=int(len(xs)),
                )
            )
    return tuple(split_runs) or (run,)


def infer_target_roi(
    img: Image.Image,
    *,
    search_roi: tuple[int, int, int, int],
    protected_box: tuple[int, int, int, int] | None,
    target_text: str,
    threshold: int,
) -> tuple[tuple[int, int, int, int], tuple[TextRun, ...], list[TextRun]]:
    runs = dark_runs(img, search_roi, threshold=threshold)
    selected = runs
    if protected_box and runs:
        protected_end = protected_box[2]
        trailing = [run for run in runs if run.x2 > protected_end]
        if trailing:
            selected = trailing
            if len(trailing) > 1:
                gaps = [trailing[i + 1].x1 - trailing[i].x2 for i in range(len(trailing) - 1)]
                max_gap = max(gaps)
                max_gap_idx = gaps.index(max_gap)
                if max_gap >= 4:
                    selected = trailing[max_gap_idx + 1 :]
            target_chars = len([ch for ch in target_text if not ch.isspace()])
            if is_mostly_cjk(target_text) and target_chars and len(selected) > target_chars:
                selected = selected[-target_chars:]

    target_chars = len([ch for ch in target_text if not ch.isspace()])
    if is_mostly_cjk(target_text) and target_chars > 1 and len(selected) == 1:
        selected = list(
            split_run_by_projection(
                img,
                selected[0],
                parts=target_chars,
                threshold=threshold,
            )
        )

    if not selected:
        return search_roi, tuple(), runs

    x1 = min(run.x1 for run in selected)
    y1 = min(run.y1 for run in selected)
    x2 = max(run.x2 for run in selected)
    y2 = max(run.y2 for run in selected)
    pad_x = 3
    pad_y = 3
    target_roi = clamp_box((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), img.size)
    return target_roi, tuple(selected), runs


def build_old_text_mask(
    arr: np.ndarray,
    roi: tuple[int, int, int, int],
    *,
    threshold: int = 165,
    dilate_iterations: int = 2,
) -> np.ndarray:
    x1, y1, x2, y2 = roi
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    roi_gray = gray[y1:y2, x1:x2]
    mask_roi = (roi_gray < threshold).astype(np.uint8) * 255

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_roi, 8)
    clean = np.zeros_like(mask_roi)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 4:
            clean[labels == i] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.dilate(clean, kernel, iterations=dilate_iterations)

    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = clean
    return mask


def build_slot_text_mask(
    arr: np.ndarray,
    plan: RenderPlan,
    *,
    threshold: int,
    dilate_iterations: int,
) -> np.ndarray:
    h, w = arr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if not plan.slot_boxes:
        return mask

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    source_chars = text_chars(plan.source_text)
    target_chars = text_chars(plan.target_text)
    changed_indices = changed_text_slot_indices(plan)
    slot_count = min(len(plan.slot_boxes), len(source_chars) or len(plan.slot_boxes))
    slots = tuple(sorted(plan.slot_boxes, key=lambda item: item.x1)[:slot_count])
    tx1, ty1, tx2, ty2 = plan.target_roi
    for slot_index, slot in enumerate(slots):
        if changed_indices is not None and slot_index not in changed_indices:
            continue
        slot_w = max(1, slot.x2 - slot.x1)
        slot_h = max(1, slot.y2 - slot.y1)
        pad_x = max(5, int(round(slot_w * 0.35)))
        pad_y = max(3, int(round(slot_h * 0.24)))
        x1 = max(tx1, slot.x1 - pad_x)
        y1 = max(ty1, slot.y1 - pad_y)
        x2 = min(tx2, slot.x2 + pad_x)
        y2 = min(ty2, slot.y2 + pad_y)
        if target_chars and slot_index >= len(target_chars):
            x2 = tx2
        if x2 <= x1 or y2 <= y1:
            continue
        if target_chars and slot_index >= len(target_chars):
            local = np.full((y2 - y1, x2 - x1), 255, dtype=np.uint8)
        else:
            local_threshold = min(int(threshold), 160)
            local = (gray[y1:y2, x1:x2] < local_threshold).astype(np.uint8) * 255
        num, labels, stats, _ = cv2.connectedComponentsWithStats(local, 8)
        clean = np.zeros_like(local)
        for idx in range(1, num):
            area = stats[idx, cv2.CC_STAT_AREA]
            if area >= 3:
                clean[labels == idx] = 255
        mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], clean)

    if np.any(mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=dilate_iterations)
        outside = np.ones((h, w), dtype=bool)
        outside[ty1:ty2, tx1:tx2] = False
        mask[outside] = 0
    return mask


def build_source_slot_cleanup_mask(
    arr: np.ndarray,
    plan: RenderPlan,
    *,
    threshold: int = 165,
    dilate_iterations: int = 1,
) -> np.ndarray:
    h, w = arr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    source_chars = text_chars(plan.source_text)
    if not source_chars or not plan.slot_boxes:
        return mask

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    cleanup_threshold = max(1, min(254, int(threshold)))
    slots = tuple(sorted(plan.slot_boxes, key=lambda item: item.x1)[: len(source_chars)])
    tx1, ty1, tx2, ty2 = plan.target_roi
    for slot in slots:
        x1 = max(tx1, slot.x1)
        y1 = max(ty1, slot.y1)
        x2 = min(tx2, slot.x2)
        y2 = min(ty2, slot.y2)
        if x2 <= x1 or y2 <= y1:
            continue
        local = (gray[y1:y2, x1:x2] < cleanup_threshold).astype(np.uint8) * 255
        num, labels, stats, _ = cv2.connectedComponentsWithStats(local, 8)
        clean = np.zeros_like(local)
        for idx in range(1, num):
            area = stats[idx, cv2.CC_STAT_AREA]
            if area >= 2:
                clean[labels == idx] = 255
        mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], clean)

    if np.any(mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=max(1, int(dilate_iterations)))
        outside = np.ones((h, w), dtype=bool)
        outside[ty1:ty2, tx1:tx2] = False
        mask[outside] = 0
    return mask


def extra_source_slot_cleanup_boxes(plan: RenderPlan) -> tuple[tuple[int, int, int, int], ...]:
    source_chars = [ch for ch in (plan.source_text or "") if not ch.isspace()]
    target_chars = [ch for ch in plan.target_text if not ch.isspace()]
    if (
        not source_chars
        or not target_chars
        or len(target_chars) >= len(source_chars)
        or not plan.slot_boxes
    ):
        return ()

    slots = tuple(sorted(plan.slot_boxes, key=lambda item: item.x1)[: len(source_chars)])
    if len(slots) <= len(target_chars):
        return ()

    tx1, ty1, tx2, ty2 = plan.target_roi
    first_extra = slots[len(target_chars)]
    previous_target = slots[len(target_chars) - 1] if target_chars else first_extra
    slot_heights = [max(1, slot.y2 - slot.y1) for slot in slots]
    pad_y = max(3, int(round(float(np.median(slot_heights)) * 0.18)))
    left = max(tx1, first_extra.x1, previous_target.x2 + 1)
    right = tx2
    top = max(ty1, first_extra.y1 - pad_y)
    bottom = ty2
    if right - left < 4 or bottom - top < 4:
        return ()
    return ((left, top, right, bottom),)


def build_extra_source_slot_cleanup_mask(
    plan: RenderPlan,
    size: tuple[int, int],
) -> np.ndarray:
    w, h = size
    mask = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in extra_source_slot_cleanup_boxes(plan):
        x1, y1, x2, y2 = clamp_box((x1, y1, x2, y2), size)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
    return mask


def build_trailing_value_cleanup_mask(
    plan: RenderPlan,
    alpha_layer: Image.Image,
    size: tuple[int, int],
) -> np.ndarray:
    source_chars = text_chars(plan.source_text)
    target_chars = text_chars(plan.target_text)
    w, h = size
    mask = np.zeros((h, w), dtype=np.uint8)
    if (
        not source_chars
        or not target_chars
        or len(target_chars) >= len(source_chars)
        or is_mostly_cjk(plan.target_text)
    ):
        return mask

    # For photographed numeric/date values, a full trailing rectangle leaves a
    # visible block. The old glyph mask already removes the actual source text;
    # keep this hook empty until a non-rectangular trailing mask is available.
    return mask

def restore_inpainted_texture(
    original_arr: np.ndarray,
    base_arr: np.ndarray,
    mask: np.ndarray,
    *,
    strength: float = 0.72,
) -> np.ndarray:
    if not np.any(mask):
        return base_arr

    soft_mask = cv2.GaussianBlur(mask, (0, 0), 0.8).astype(np.float32) / 255.0
    if float(soft_mask.max()) <= 0:
        return base_arr

    original_f = original_arr.astype(np.float32)
    base_f = base_arr.astype(np.float32)
    mask_bool = mask > 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 9))
    ring = cv2.dilate(mask, kernel, iterations=1) > 0
    original_gray = cv2.cvtColor(original_arr, cv2.COLOR_RGB2GRAY)
    bg_mask = ring & ~mask_bool & (original_gray >= 150)
    if int(np.count_nonzero(bg_mask)) >= 24 and int(np.count_nonzero(mask_bool)) >= 1:
        ref = original_f[bg_mask]
        dst = base_f[mask_bool]
        ref_mean = ref.mean(axis=0)
        ref_std = np.maximum(ref.std(axis=0), 1.0)
        dst_mean = dst.mean(axis=0)
        dst_std = np.maximum(dst.std(axis=0), 1.0)
        matched = (dst - dst_mean) * (ref_std / dst_std) + ref_mean
        base_f[mask_bool] = base_f[mask_bool] * 0.35 + matched * 0.65

    smooth = cv2.GaussianBlur(original_f, (0, 0), 1.2)
    residual = original_f - smooth
    residual_filled = cv2.inpaint(
        np.clip(residual + 128.0, 0, 255).astype(np.uint8),
        mask,
        3,
        cv2.INPAINT_TELEA,
    ).astype(np.float32) - 128.0
    restored = base_f + residual_filled * soft_mask[:, :, None] * strength
    return np.clip(restored, 0, 255).astype(np.uint8)


def suppress_inpaint_glow(
    original_arr: np.ndarray,
    base_arr: np.ndarray,
    mask: np.ndarray,
    *,
    strength: float = 0.72,
) -> np.ndarray:
    if not np.any(mask):
        return base_arr

    original_gray = cv2.cvtColor(original_arr, cv2.COLOR_RGB2GRAY)
    base_gray = cv2.cvtColor(base_arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mask_bool = mask > 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 11))
    ring = cv2.dilate(mask, kernel, iterations=1) > 0
    bg_mask = ring & ~mask_bool & (original_gray >= 145) & (original_gray <= 230)
    if int(np.count_nonzero(bg_mask)) < 24:
        return base_arr

    cap_gray = float(np.percentile(original_gray[bg_mask], 68))
    too_bright = mask_bool & (base_gray > cap_gray)
    if not np.any(too_bright):
        return base_arr

    adjusted = base_arr.astype(np.float32)
    delta = (base_gray[too_bright] - cap_gray) * max(0.0, min(1.0, strength))
    adjusted[too_bright] -= delta[:, None]
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def suppress_inpaint_shadow(
    original_arr: np.ndarray,
    base_arr: np.ndarray,
    mask: np.ndarray,
    *,
    strength: float = 0.75,
) -> np.ndarray:
    if not np.any(mask):
        return base_arr

    original_gray = cv2.cvtColor(original_arr, cv2.COLOR_RGB2GRAY)
    base_gray = cv2.cvtColor(base_arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mask_bool = mask > 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 11))
    ring = cv2.dilate(mask, kernel, iterations=1) > 0
    bg_mask = ring & ~mask_bool & (original_gray >= 145) & (original_gray <= 235)
    if int(np.count_nonzero(bg_mask)) < 24:
        return base_arr

    floor_gray = float(np.percentile(original_gray[bg_mask], 50))
    too_dark = mask_bool & (base_gray < floor_gray)
    if not np.any(too_dark):
        return base_arr

    adjusted = base_arr.astype(np.float32)
    delta = (floor_gray - base_gray[too_dark]) * max(0.0, min(1.0, strength))
    adjusted[too_dark] += delta[:, None]
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def repair_mask_from_row_background(
    original_arr: np.ndarray,
    base_arr: np.ndarray,
    mask: np.ndarray,
    roi: tuple[int, int, int, int],
    *,
    strength: float = 0.28,
) -> np.ndarray:
    if not np.any(mask):
        return base_arr

    h, w = mask.shape
    x1, y1, x2, y2 = roi
    pad_x = max(10, int(round((x2 - x1) * 0.25)))
    bx1 = max(0, x1 - pad_x)
    bx2 = min(w, x2 + pad_x)
    if bx2 <= bx1:
        return base_arr

    original_gray = cv2.cvtColor(original_arr, cv2.COLOR_RGB2GRAY)
    repaired = base_arr.astype(np.float32).copy()
    original_f = original_arr.astype(np.float32)
    fill_mask = mask > 0
    bg_mask = (~fill_mask) & (original_gray >= 125) & (original_gray <= 225)
    strength = max(0.0, min(1.0, float(strength)))

    for y in range(max(0, y1), min(h, y2)):
        fill_x = np.where(fill_mask[y, x1:x2])[0] + x1
        if len(fill_x) == 0:
            continue
        row_bg = np.where(bg_mask[y, bx1:bx2])[0] + bx1
        if len(row_bg) < 4:
            continue
        row_colors = original_f[y, row_bg]
        fallback = np.median(row_colors, axis=0)
        for x in fill_x:
            left_candidates = row_bg[row_bg < x]
            right_candidates = row_bg[row_bg > x]
            if len(left_candidates) and len(right_candidates):
                lx = int(left_candidates[-1])
                rx = int(right_candidates[0])
                denom = max(1, rx - lx)
                t = float(x - lx) / float(denom)
                fill_color = original_f[y, lx] * (1.0 - t) + original_f[y, rx] * t
            elif len(left_candidates):
                fill_color = original_f[y, int(left_candidates[-1])]
            elif len(right_candidates):
                fill_color = original_f[y, int(right_candidates[0])]
            else:
                fill_color = fallback
            repaired[y, x] = repaired[y, x] * (1.0 - strength) + fill_color * strength

    return np.clip(repaired, 0, 255).astype(np.uint8)


def repair_extra_source_slot_background(
    original_arr: np.ndarray,
    base_arr: np.ndarray,
    plan: RenderPlan,
) -> np.ndarray:
    cleanup_boxes = extra_source_slot_cleanup_boxes(plan)
    if not cleanup_boxes:
        return base_arr

    repaired = base_arr
    size = (original_arr.shape[1], original_arr.shape[0])
    for box in cleanup_boxes:
        mask = np.zeros(original_arr.shape[:2], dtype=np.uint8)
        x1, y1, x2, y2 = clamp_box(box, size)
        if x2 <= x1 or y2 <= y1:
            continue
        mask[y1:y2, x1:x2] = 255
        repaired = reconstruct_background_plane_texture(
            original_arr,
            repaired,
            mask,
            (x1, y1, x2, y2),
            pad=70,
            noise_strength=0.45,
            texture_blur=0.08,
        )
    return repaired


def reconstruct_background_plane_texture(
    original_arr: np.ndarray,
    fallback_arr: np.ndarray,
    mask: np.ndarray,
    roi: tuple[int, int, int, int],
    *,
    pad: int = 50,
    noise_strength: float = 0.65,
    texture_blur: float = 0.20,
) -> np.ndarray:
    if not np.any(mask):
        return fallback_arr

    h, w = mask.shape
    x1, y1, x2, y2 = roi
    bx1 = max(0, x1 - pad)
    by1 = max(0, y1 - pad)
    bx2 = min(w, x2 + pad)
    by2 = min(h, y2 + pad)
    if bx2 <= bx1 or by2 <= by1:
        return fallback_arr

    gray = cv2.cvtColor(original_arr, cv2.COLOR_RGB2GRAY)
    yy, xx = np.mgrid[by1:by2, bx1:bx2]
    local_mask = mask[by1:by2, bx1:bx2] > 0
    local_gray = gray[by1:by2, bx1:bx2]
    bg_mask = (~local_mask) & (local_gray >= 145) & (local_gray <= 215)

    grad_x = cv2.Sobel(local_gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(local_gray, cv2.CV_32F, 0, 1, ksize=3)
    bg_mask &= np.sqrt(grad_x * grad_x + grad_y * grad_y) < 55
    if int(np.count_nonzero(bg_mask)) < 40:
        return fallback_arr

    fill_y, fill_x = np.where(mask > 0)
    within = (fill_x >= bx1) & (fill_x < bx2) & (fill_y >= by1) & (fill_y < by2)
    fill_y = fill_y[within]
    fill_x = fill_x[within]
    if len(fill_x) == 0:
        return fallback_arr

    sample_x = xx[bg_mask].astype(np.float32)
    sample_y = yy[bg_mask].astype(np.float32)
    sample_features = np.column_stack(
        [np.ones_like(sample_x), sample_x - bx1, sample_y - by1]
    )
    fill_features = np.column_stack(
        [
            np.ones_like(fill_x, dtype=np.float32),
            fill_x.astype(np.float32) - bx1,
            fill_y.astype(np.float32) - by1,
        ]
    )

    channels: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    for channel in range(3):
        values = original_arr[by1:by2, bx1:bx2, channel][bg_mask].astype(np.float32)
        coef = np.linalg.lstsq(sample_features, values, rcond=None)[0]
        predicted_fill = fill_features @ coef
        predicted_samples = sample_features @ coef
        channels.append(predicted_fill)
        residuals.append(values - predicted_samples)

    fill_values = np.stack(channels, axis=1)
    residual_pool = np.stack(residuals, axis=1)
    seed = int((x1 * 73856093) ^ (y1 * 19349663) ^ (x2 * 83492791) ^ (y2 * 2654435761))
    rng = np.random.default_rng(seed & 0xFFFFFFFF)
    residual = residual_pool[rng.integers(0, len(residual_pool), size=len(fill_x))]
    low = np.percentile(residual_pool, 2, axis=0)
    high = np.percentile(residual_pool, 98, axis=0)
    residual = np.clip(residual, low, high)

    reconstructed = fallback_arr.astype(np.float32).copy()
    reconstructed[fill_y, fill_x] = fill_values + residual * noise_strength
    if texture_blur > 0:
        blurred = cv2.GaussianBlur(reconstructed, (0, 0), texture_blur)
        only_reconstructed = np.zeros(mask.shape, dtype=np.uint8)
        only_reconstructed[fill_y, fill_x] = 255
        soft = cv2.GaussianBlur(only_reconstructed, (0, 0), 0.35).astype(np.float32) / 255.0
        reconstructed = reconstructed * (1.0 - soft[:, :, None]) + blurred * soft[:, :, None]
    return np.clip(reconstructed, 0, 255).astype(np.uint8)


def retexture_inpaint_patch(
    original_arr: np.ndarray,
    base_arr: np.ndarray,
    mask: np.ndarray,
    roi: tuple[int, int, int, int],
    *,
    strength: float = 0.16,
    residual_scale: float = 0.08,
) -> np.ndarray:
    if not np.any(mask):
        return base_arr

    h, w = mask.shape
    x1, y1, x2, y2 = roi
    pad_x = max(18, int(round((x2 - x1) * 0.65)))
    pad_y = max(12, int(round((y2 - y1) * 0.85)))
    bx1 = max(0, x1 - pad_x)
    by1 = max(0, y1 - pad_y)
    bx2 = min(w, x2 + pad_x)
    by2 = min(h, y2 + pad_y)
    local_mask = mask[by1:by2, bx1:bx2] > 0
    if not np.any(local_mask):
        return base_arr

    original_gray = cv2.cvtColor(original_arr, cv2.COLOR_RGB2GRAY)
    local_gray = original_gray[by1:by2, bx1:bx2]
    bg_mask = (~local_mask) & (local_gray >= 120) & (local_gray <= 230)
    if int(np.count_nonzero(bg_mask)) < 40:
        return base_arr

    base_f = base_arr.astype(np.float32).copy()
    original_f = original_arr.astype(np.float32)
    fill_y, fill_x = np.where(mask > 0)
    within = (fill_x >= bx1) & (fill_x < bx2) & (fill_y >= by1) & (fill_y < by2)
    fill_y = fill_y[within]
    fill_x = fill_x[within]
    if len(fill_x) == 0:
        return base_arr

    yy, xx = np.mgrid[by1:by2, bx1:bx2]
    sample_x = xx[bg_mask].astype(np.float32)
    sample_y = yy[bg_mask].astype(np.float32)
    sample_features = np.column_stack(
        [np.ones_like(sample_x), sample_x - bx1, sample_y - by1]
    )
    fill_features = np.column_stack(
        [
            np.ones_like(fill_x, dtype=np.float32),
            fill_x.astype(np.float32) - bx1,
            fill_y.astype(np.float32) - by1,
        ]
    )

    predicted = []
    residuals = []
    for channel in range(3):
        samples = original_f[by1:by2, bx1:bx2, channel][bg_mask].astype(np.float32)
        coef = np.linalg.lstsq(sample_features, samples, rcond=None)[0]
        predicted.append(fill_features @ coef)
        residuals.append(samples - sample_features @ coef)
    predicted_fill = np.stack(predicted, axis=1)
    residual_pool = np.stack(residuals, axis=1)
    seed = int((x1 * 73856093) ^ (y1 * 19349663) ^ (x2 * 83492791) ^ (y2 * 2654435761))
    rng = np.random.default_rng(seed & 0xFFFFFFFF)
    residual = residual_pool[rng.integers(0, len(residual_pool), size=len(fill_x))]
    low = np.percentile(residual_pool, 3, axis=0)
    high = np.percentile(residual_pool, 97, axis=0)
    residual = np.clip(residual, low, high)
    texture_fill = predicted_fill + residual * max(0.0, min(1.0, float(residual_scale)))

    current_fill = base_f[fill_y, fill_x]
    base_f[fill_y, fill_x] = current_fill * (1.0 - strength) + texture_fill * strength
    return np.clip(base_f, 0, 255).astype(np.uint8)


def background_retexture_strength(params: CandidateParams, *, target_expanded: bool) -> float:
    base = 0.12 if target_expanded else 0.16
    texture_gain = max(0.0, float(params.photo_noise)) * 1.8 + max(0.0, float(params.edge_breakup)) * 3.0
    return max(0.08, min(0.48, base + texture_gain))


def roi_scan_texture_strength(params: CandidateParams, *, target_expanded: bool) -> float:
    base = 0.12 if target_expanded else 0.30
    texture_gain = max(0.0, float(params.photo_noise)) * 1.4 + max(0.0, float(params.edge_breakup)) * 2.0
    return max(0.08, min(0.62, base + texture_gain))


def old_text_mask_roi(plan: RenderPlan) -> tuple[int, int, int, int]:
    if plan.source_reference_box is not None:
        overlap = intersect_boxes(plan.source_reference_box, plan.target_roi)
        if overlap is not None:
            return overlap
    return plan.target_roi


def apply_roi_scan_texture(
    original_arr: np.ndarray,
    edited_arr: np.ndarray,
    roi: tuple[int, int, int, int],
    old_text_mask: np.ndarray,
    *,
    strength: float = 0.30,
) -> np.ndarray:
    x1, y1, x2, y2 = roi
    if x2 <= x1 or y2 <= y1:
        return edited_arr

    original_f = original_arr.astype(np.float32)
    edited_f = edited_arr.astype(np.float32)
    smooth = cv2.GaussianBlur(original_f, (0, 0), 1.1)
    residual = original_f - smooth
    fill_mask = old_text_mask.copy()
    if np.any(fill_mask):
        residual = cv2.inpaint(
            np.clip(residual + 128.0, 0, 255).astype(np.uint8),
            fill_mask,
            3,
            cv2.INPAINT_TELEA,
        ).astype(np.float32) - 128.0

    roi_residual = residual[y1:y2, x1:x2]
    edited_f[y1:y2, x1:x2] += roi_residual * strength
    return np.clip(edited_f, 0, 255).astype(np.uint8)


def feather_roi_boundary(
    original_arr: np.ndarray,
    edited_arr: np.ndarray,
    roi: tuple[int, int, int, int],
    alpha_layer: Image.Image | None = None,
    protected_mask: np.ndarray | None = None,
    *,
    width: int = 5,
) -> np.ndarray:
    x1, y1, x2, y2 = roi
    if x2 <= x1 or y2 <= y1 or width <= 0:
        return edited_arr
    h, w = edited_arr.shape[:2]
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return edited_arr

    roi_h = y2 - y1
    roi_w = x2 - x1
    yy, xx = np.indices((roi_h, roi_w), dtype=np.float32)
    distance = np.minimum.reduce(
        [
            xx,
            yy,
            roi_w - 1 - xx,
            roi_h - 1 - yy,
        ]
    )
    edge_weight = np.clip(distance / float(width), 0.0, 1.0)
    if alpha_layer is not None:
        alpha = np.array(alpha_layer, dtype=np.uint8)[y1:y2, x1:x2]
        edge_weight = np.where(alpha > 4, 1.0, edge_weight)
    if protected_mask is not None:
        mask_roi = protected_mask[y1:y2, x1:x2]
        edge_weight = np.where(mask_roi > 0, 1.0, edge_weight)
    weight = edge_weight[:, :, None]
    result = edited_arr.astype(np.float32).copy()
    result[y1:y2, x1:x2] = (
        result[y1:y2, x1:x2] * weight
        + original_arr[y1:y2, x1:x2].astype(np.float32) * (1.0 - weight)
    )
    return np.clip(result, 0, 255).astype(np.uint8)


def font_text_bbox(font: ImageFont.FreeTypeFont, text: str) -> tuple[int, int, int, int]:
    scratch = Image.new("L", (1, 1), 0)
    draw = ImageDraw.Draw(scratch)
    return draw.textbbox((0, 0), text, font=font)


def default_char_offsets(text: str) -> tuple[tuple[int, int], ...]:
    return tuple((0, 0) for _ in text)


def apply_fractional_stroke(alpha_layer: Image.Image, stroke_opacity: float) -> Image.Image:
    stroke_opacity = max(0.0, min(1.0, float(stroke_opacity)))
    if stroke_opacity <= 0:
        return alpha_layer
    expanded = alpha_layer.filter(ImageFilter.MaxFilter(3))
    base_arr = np.array(alpha_layer, dtype=np.float32)
    expanded_arr = np.array(expanded, dtype=np.float32) * stroke_opacity
    combined = np.maximum(base_arr, expanded_arr)
    return Image.fromarray(np.clip(combined, 0, 255).astype(np.uint8), mode="L")


def apply_ink_gain(alpha_layer: Image.Image, ink_gain: float) -> Image.Image:
    ink_gain = max(0.0, min(1.0, float(ink_gain)))
    if ink_gain <= 0:
        return alpha_layer
    arr = np.array(alpha_layer, dtype=np.float32)
    glyph = arr > 0
    arr[glyph] = arr[glyph] + (255.0 - arr[glyph]) * ink_gain
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")


def apply_alpha_contrast(alpha_layer: Image.Image, alpha_contrast: float) -> Image.Image:
    alpha_contrast = max(0.0, min(2.0, float(alpha_contrast)))
    if alpha_contrast <= 0:
        return alpha_layer
    arr = np.array(alpha_layer, dtype=np.float32)
    nonzero = arr > 0
    normalized = arr / 255.0
    factor = 1.0 + alpha_contrast
    adjusted = (normalized - 0.5) * factor + 0.5
    adjusted = np.clip(adjusted, 0.0, 1.0) * 255.0
    adjusted[~nonzero] = 0.0
    return Image.fromarray(adjusted.astype(np.uint8), mode="L")


def apply_core_ink_gain(alpha_layer: Image.Image, core_ink_gain: float) -> Image.Image:
    core_ink_gain = max(0.0, min(1.0, float(core_ink_gain)))
    if core_ink_gain <= 0:
        return alpha_layer
    arr = np.array(alpha_layer, dtype=np.float32)
    core_weight = np.clip((arr - 120.0) / 80.0, 0.0, 1.0)
    arr = arr + (255.0 - arr) * core_ink_gain * core_weight
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")


def apply_scan_edge_breakup(
    alpha_layer: Image.Image,
    seed_box: tuple[int, int, int, int],
    *,
    strength: float = 0.025,
    quant_step: int = 12,
    scanline_strength: float = 0.02,
) -> Image.Image:
    arr = np.array(alpha_layer, dtype=np.float32)
    if not np.any(arr > 0):
        return alpha_layer

    if quant_step > 1:
        arr = np.round(arr / float(quant_step)) * float(quant_step)

    edge = (arr > 6) & (arr < 135)
    if np.any(edge) and strength > 0:
        x1, y1, x2, y2 = seed_box
        seed = int((x1 * 73856093) ^ (y1 * 19349663) ^ (x2 * 83492791) ^ (y2 * 2654435761))
        rng = np.random.default_rng(seed & 0xFFFFFFFF)
        noise = rng.random(arr.shape)
        drop_prob = strength * np.clip((135.0 - arr) / 129.0, 0.0, 1.0)
        arr[edge & (noise < drop_prob)] *= 0.45
        thin = edge & (noise >= drop_prob) & (noise < drop_prob + strength * 0.5)
        arr[thin] *= 0.65

    if scanline_strength > 0:
        rows = np.arange(arr.shape[0], dtype=np.float32)[:, None]
        modulation = 1.0 - scanline_strength * ((rows % 2) == 0).astype(np.float32)
        arr = np.where(arr > 0, arr * modulation, arr)

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")


def apply_photo_alpha_warp(
    alpha_layer: Image.Image,
    seed_box: tuple[int, int, int, int],
    strength: float,
) -> Image.Image:
    strength = max(0.0, min(1.0, float(strength)))
    if strength <= 0:
        return alpha_layer

    arr = np.array(alpha_layer, dtype=np.uint8)
    if not np.any(arr > 0):
        return alpha_layer

    h, w = arr.shape
    x1, y1, x2, y2 = seed_box
    pad = 4
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return alpha_layer

    crop = arr[y1:y2, x1:x2]
    if not np.any(crop > 0):
        return alpha_layer

    yy, xx = np.indices(crop.shape, dtype=np.float32)
    seed = int((x1 * 73856093) ^ (y1 * 19349663) ^ (x2 * 83492791) ^ (y2 * 2654435761))
    phase_a = float((seed & 0xFF) / 255.0 * np.pi * 2.0)
    phase_b = float(((seed >> 8) & 0xFF) / 255.0 * np.pi * 2.0)
    amp = 0.85 * strength
    center_x = max(1.0, float(crop.shape[1] - 1) / 2.0)

    map_x = xx + np.sin((yy + phase_a) / 7.5) * amp * 0.28
    map_y = (
        yy
        + np.sin((xx + phase_b) / 9.0) * amp * 0.55
        + ((xx - center_x) / center_x) * amp * 0.20
    )
    warped = cv2.remap(
        crop,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    result = arr.copy()
    result[y1:y2, x1:x2] = warped
    return Image.fromarray(result, mode="L")


def apply_photo_alpha_resample(
    alpha_layer: Image.Image,
    seed_box: tuple[int, int, int, int],
    strength: float,
) -> Image.Image:
    strength = max(0.0, min(1.0, float(strength)))
    if strength < 0.03:
        return alpha_layer

    arr = np.array(alpha_layer, dtype=np.uint8)
    if not np.any(arr > 0):
        return alpha_layer

    h, w = arr.shape
    x1, y1, x2, y2 = seed_box
    pad = 3
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return alpha_layer

    crop = arr[y1:y2, x1:x2]
    if not np.any(crop > 0):
        return alpha_layer

    scale = max(0.78, 1.0 - 0.20 * strength)
    small_w = max(1, int(round(crop.shape[1] * scale)))
    small_h = max(1, int(round(crop.shape[0] * scale)))
    if small_w == crop.shape[1] and small_h == crop.shape[0]:
        return alpha_layer

    down = cv2.resize(crop, (small_w, small_h), interpolation=cv2.INTER_AREA)
    up = cv2.resize(down, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    up = cv2.GaussianBlur(up, (0, 0), 0.06 + 0.12 * strength)
    blend = min(0.45, 0.18 + 0.32 * strength)
    mixed = crop.astype(np.float32) * (1.0 - blend) + up.astype(np.float32) * blend
    result = arr.copy()
    result[y1:y2, x1:x2] = np.clip(mixed, 0, 255).astype(np.uint8)
    return Image.fromarray(result, mode="L")


def rgba_from_alpha(alpha_layer: Image.Image) -> Image.Image:
    rgba = Image.new("RGBA", alpha_layer.size, (0, 0, 0, 0))
    rgba.putalpha(alpha_layer)
    return rgba


def apply_core_darken(
    edited_arr: np.ndarray,
    alpha_layer: Image.Image,
    params: CandidateParams,
) -> np.ndarray:
    strength = max(0.0, min(1.0, float(params.core_darken_strength)))
    if strength <= 0:
        return edited_arr

    threshold = max(0, min(254, int(params.core_darken_threshold)))
    target_gray = max(0, min(120, int(params.core_darken_target_gray)))
    alpha = np.array(alpha_layer, dtype=np.float32)
    if not np.any(alpha > threshold):
        return edited_arr

    weight = np.clip((alpha - float(threshold)) / float(255 - threshold), 0.0, 1.0)
    weight = np.power(weight, 0.75) * strength
    if not np.any(weight > 0):
        return edited_arr

    arr = edited_arr.astype(np.float32)
    gray = arr[:, :, 0] * 0.299 + arr[:, :, 1] * 0.587 + arr[:, :, 2] * 0.114
    target = np.full_like(gray, float(target_gray))
    desired_gray = gray + (target - gray) * weight
    desired_gray = np.minimum(gray, desired_gray)
    scale = np.divide(
        desired_gray,
        np.maximum(gray, 1.0),
        out=np.ones_like(gray, dtype=np.float32),
        where=gray > 1.0,
    )
    arr *= scale[:, :, None]
    return np.clip(arr, 0, 255).astype(np.uint8)


def apply_photo_text_texture(
    edited_arr: np.ndarray,
    original_arr: np.ndarray,
    alpha_layer: Image.Image,
    roi: tuple[int, int, int, int],
    params: CandidateParams,
) -> np.ndarray:
    photo_noise = max(0.0, min(1.0, float(params.photo_noise)))
    jpeg_quality = int(params.jpeg_quality or 0)
    if photo_noise <= 0 and not (1 <= jpeg_quality <= 99):
        return edited_arr

    alpha = np.array(alpha_layer, dtype=np.uint8)
    if not np.any(alpha > 4):
        return edited_arr

    x1, y1, x2, y2 = roi
    h, w = alpha.shape
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return edited_arr

    result = edited_arr.astype(np.float32).copy()
    roi_alpha = alpha[y1:y2, x1:x2]
    text_mask = roi_alpha > 4
    if not np.any(text_mask):
        return edited_arr

    neighborhood = cv2.dilate(text_mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1) > 0
    roi_result = result[y1:y2, x1:x2]

    if 1 <= jpeg_quality <= 99:
        encoded_ok, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(np.clip(roi_result, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, max(35, min(99, jpeg_quality))],
        )
        if encoded_ok:
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if decoded is not None:
                decoded_rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB).astype(np.float32)
                weight = max(0.0, min(0.42, (100.0 - float(jpeg_quality)) / 85.0))
                roi_result[neighborhood] = roi_result[neighborhood] * (1.0 - weight) + decoded_rgb[neighborhood] * weight

    if photo_noise > 0:
        seed = int((x1 * 73856093) ^ (y1 * 19349663) ^ (x2 * 83492791) ^ (y2 * 2654435761))
        rng = np.random.default_rng(seed & 0xFFFFFFFF)
        random_noise = rng.normal(0.0, 34.0 * photo_noise, roi_result.shape).astype(np.float32)

        original_roi = original_arr[y1:y2, x1:x2].astype(np.float32)
        smooth = cv2.GaussianBlur(original_roi, (0, 0), 1.2)
        residual = original_roi - smooth
        alpha_weight = np.clip(roi_alpha.astype(np.float32) / 180.0, 0.0, 1.0)[:, :, None]
        texture = random_noise + residual * min(0.28, photo_noise * 2.2)
        roi_result[neighborhood] += texture[neighborhood] * (0.25 + alpha_weight[neighborhood] * 0.75)

    result[y1:y2, x1:x2] = roi_result
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_text_angle(alpha_layer: Image.Image, roi: tuple[int, int, int, int], angle_degrees: float) -> Image.Image:
    angle = max(-8.0, min(8.0, float(angle_degrees or 0.0)))
    if abs(angle) < 0.15:
        return alpha_layer
    x1, y1, x2, y2 = roi
    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    return alpha_layer.rotate(
        -angle,
        resample=Image.Resampling.BICUBIC,
        center=center,
    )


def text_rotation_box(plan: RenderPlan) -> tuple[int, int, int, int]:
    chars = [ch for ch in plan.target_text if not ch.isspace()]
    used_slots = target_char_slots_for_plan(plan)
    if chars and plan.draw_mode in {"auto", "per_char", "line_chars"} and is_mostly_cjk(plan.target_text) and used_slots:
        return (
            min(slot.x1 for slot in used_slots),
            min(slot.y1 for slot in used_slots),
            max(slot.x2 for slot in used_slots),
            max(slot.y2 for slot in used_slots),
        )
    return plan.target_roi


def target_char_slots_for_plan(plan: RenderPlan) -> tuple[TextRun, ...]:
    chars = text_chars(plan.target_text)
    if not chars or not plan.slot_boxes:
        return ()
    ordered_slots = tuple(sorted(plan.slot_boxes, key=lambda item: item.x1))
    if len(ordered_slots) >= len(chars):
        return ordered_slots[: len(chars)]
    source_chars = text_chars(plan.source_text)
    if (
        not source_chars
        or len(chars) <= len(source_chars)
        or plan.placement_strategy != "left_anchor_span"
        or plan.draw_mode != "line_chars"
    ):
        return ordered_slots

    widths = [max(1, slot.x2 - slot.x1) for slot in ordered_slots]
    heights = [max(1, slot.y2 - slot.y1) for slot in ordered_slots]
    areas = [max(1, slot.area) for slot in ordered_slots]
    gaps = [
        max(1, ordered_slots[idx + 1].x1 - ordered_slots[idx].x2)
        for idx in range(len(ordered_slots) - 1)
    ]
    slot_w = max(1, int(round(float(np.median(np.array(widths, dtype=np.float32))))))
    slot_h = max(1, int(round(float(np.median(np.array(heights, dtype=np.float32))))))
    slot_area = max(1, int(round(float(np.median(np.array(areas, dtype=np.float32))))))
    gap = max(1, int(round(float(np.median(np.array(gaps, dtype=np.float32)))))) if gaps else max(1, int(round(slot_w * 0.14)))
    center_y = slot_row_center_y(plan, len(source_chars)) or float(ordered_slots[-1].y1 + ordered_slots[-1].y2) / 2.0
    y1 = int(round(center_y - slot_h / 2.0))
    y2 = y1 + slot_h
    slots = list(ordered_slots)
    next_x1 = ordered_slots[-1].x2 + gap
    while len(slots) < len(chars):
        slots.append(TextRun(x1=next_x1, y1=y1, x2=next_x1 + slot_w, y2=y2, area=slot_area))
        next_x1 += slot_w + gap
    return tuple(slots)


def source_slot_for_target_index(plan: RenderPlan, index: int) -> TextRun | None:
    source_slots = tuple(sorted(plan.slot_boxes, key=lambda item: item.x1))
    if not source_slots:
        return None
    source_chars = text_chars(plan.source_text)
    if source_chars and index < len(source_chars):
        return source_slots[min(index, len(source_slots) - 1)]
    return source_slots[-1]


def slot_row_center_y(plan: RenderPlan, count: int) -> float | None:
    if count <= 0 or not plan.slot_boxes:
        return None
    ordered_slots = tuple(sorted(plan.slot_boxes, key=lambda item: item.x1))
    source_chars = [ch for ch in (plan.source_text or "") if not ch.isspace()]
    reference_count = len(source_chars) if source_chars else count
    reference_slots = ordered_slots[: min(len(ordered_slots), reference_count or count)]
    centers = [float(slot.y1 + slot.y2) / 2.0 for slot in reference_slots]
    if not centers:
        return None
    return float(np.median(np.array(centers, dtype=np.float32)))


def char_text_position(
    font: ImageFont.FreeTypeFont,
    ch: str,
    slot: TextRun,
    params: CandidateParams,
    idx: int,
    offsets: tuple[tuple[int, int], ...],
    *,
    row_center_y: float | None = None,
    center_in_slot: bool = False,
) -> tuple[float, float]:
    bbox = font_text_bbox(font, ch)
    cdx, cdy = offsets[idx] if idx < len(offsets) else (0, 0)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    if center_in_slot:
        x = (slot.x1 + slot.x2) / 2.0 - text_w / 2.0 - bbox[0] + params.text_dx + cdx
    else:
        x = slot.x1 - bbox[0] + params.text_dx + cdx
    if center_in_slot:
        y = (slot.y1 + slot.y2) / 2.0 - text_h / 2.0 - bbox[1] + params.text_dy + cdy
    elif row_center_y is None:
        y = slot.y1 - bbox[1] + params.text_dy + cdy
    else:
        y = row_center_y - text_h / 2.0 - bbox[1] + params.text_dy + cdy
    return x, y


def estimate_slot_edge_shear(
    gray: np.ndarray,
    slot: TextRun,
    *,
    threshold: int,
) -> float | None:
    h, w = gray.shape[:2]
    x1 = max(0, min(w, slot.x1))
    x2 = max(0, min(w, slot.x2))
    y1 = max(0, min(h, slot.y1))
    y2 = max(0, min(h, slot.y2))
    if x2 <= x1 or y2 <= y1:
        return None

    crop = gray[y1:y2, x1:x2]
    mask = crop < max(90, min(210, int(threshold)))
    if int(np.count_nonzero(mask)) < 18:
        return None

    rows: list[float] = []
    lefts: list[float] = []
    rights: list[float] = []
    centers: list[float] = []
    min_row_pixels = max(2, int(round((x2 - x1) * 0.05)))
    for row_idx in range(mask.shape[0]):
        xs = np.flatnonzero(mask[row_idx])
        if len(xs) < min_row_pixels:
            continue
        rows.append(float(row_idx))
        lefts.append(float(xs.min()))
        rights.append(float(xs.max()))
        centers.append(float(xs.mean()))

    if len(rows) < 6:
        return None

    y = np.array(rows, dtype=np.float32)
    y = y - float(y.mean())
    denom = float(np.dot(y, y))
    if denom < 1.0:
        return None

    slopes: list[float] = []
    for values, weight in (
        (lefts, 0.45),
        (rights, 0.45),
        (centers, 0.25),
    ):
        x = np.array(values, dtype=np.float32)
        x = x - float(x.mean())
        slope = float(np.dot(y, x) / denom)
        if math.isfinite(slope):
            slopes.append(slope * weight)
    if not slopes:
        return None

    shear = float(sum(slopes) / sum((0.45, 0.45, 0.25)[: len(slopes)]))
    if abs(shear) < 0.018:
        return None
    return max(-0.12, min(0.12, shear))


def reference_slot_shear(
    original: Image.Image | None,
    plan: RenderPlan,
    idx: int,
    params: CandidateParams,
) -> float:
    if original is None or not plan.slot_boxes:
        return 0.0
    source_chars = text_chars(plan.source_text)
    target_chars = text_chars(plan.target_text)
    if not source_chars or len(source_chars) != len(target_chars):
        return 0.0

    ordered_slots = tuple(sorted(plan.slot_boxes, key=lambda item: item.x1))
    if idx >= len(ordered_slots):
        return 0.0

    gray = gray_array(original)
    primary = estimate_slot_edge_shear(gray, ordered_slots[idx], threshold=params.mask_threshold)

    changed_indices = changed_text_slot_indices(plan) or set()
    neighbor_shears: list[float] = []
    for distance in range(1, len(ordered_slots)):
        for neighbor_idx in (idx - distance, idx + distance):
            if neighbor_idx < 0 or neighbor_idx >= len(ordered_slots):
                continue
            if neighbor_idx in changed_indices:
                continue
            value = estimate_slot_edge_shear(
                gray,
                ordered_slots[neighbor_idx],
                threshold=params.mask_threshold,
            )
            if value is not None:
                neighbor_shears.append(value)
        if neighbor_shears:
            break

    if primary is None and not neighbor_shears:
        return 0.0
    if primary is None:
        shear = neighbor_shears[0]
    elif neighbor_shears:
        shear = primary * 0.75 + neighbor_shears[0] * 0.25
    else:
        shear = primary
    # Keep the replacement close to the photographed slot pose, but cap the
    # inherited shear so a noisy source glyph cannot turn into a large rotation.
    return max(-0.10, min(0.10, shear * 0.84))


def char_pose_metrics(
    original: Image.Image | None,
    plan: RenderPlan,
    params: CandidateParams,
) -> dict[str, Any]:
    if original is None:
        return {"enabled": False, "reason": "original image not available"}
    source_chars = text_chars(plan.source_text)
    target_chars = text_chars(plan.target_text)
    if not source_chars or len(source_chars) != len(target_chars):
        return {"enabled": False, "reason": "source and target chars are not one-to-one"}
    if not plan.slot_boxes:
        return {"enabled": False, "reason": "missing per-character slots"}

    ordered_slots = tuple(sorted(plan.slot_boxes, key=lambda item: item.x1))
    gray = gray_array(original)
    changed_indices = changed_text_slot_indices(plan) or set()
    items: list[dict[str, Any]] = []
    for idx, target_char in enumerate(target_chars[: len(ordered_slots)]):
        slot = ordered_slots[idx]
        source_shear = estimate_slot_edge_shear(gray, slot, threshold=params.mask_threshold)
        neighbor_shear = None
        neighbor_index = None
        for distance in range(1, len(ordered_slots)):
            for candidate_index in (idx - distance, idx + distance):
                if candidate_index < 0 or candidate_index >= len(ordered_slots):
                    continue
                if candidate_index in changed_indices:
                    continue
                value = estimate_slot_edge_shear(
                    gray,
                    ordered_slots[candidate_index],
                    threshold=params.mask_threshold,
                )
                if value is not None:
                    neighbor_shear = value
                    neighbor_index = candidate_index
                    break
            if neighbor_shear is not None:
                break

        applied = reference_slot_shear(original, plan, idx, params) if idx in changed_indices else 0.0
        reference_shear = source_shear
        if source_shear is not None and neighbor_shear is not None:
            reference_shear = source_shear * 0.75 + neighbor_shear * 0.25
        elif source_shear is None:
            reference_shear = neighbor_shear
        items.append(
            {
                "index": idx,
                "source_char": source_chars[idx] if idx < len(source_chars) else None,
                "target_char": target_char,
                "changed": idx in changed_indices,
                "slot_box": [slot.x1, slot.y1, slot.x2, slot.y2],
                "source_slot_shear": None if source_shear is None else round(float(source_shear), 4),
                "neighbor_index": neighbor_index,
                "neighbor_shear": None if neighbor_shear is None else round(float(neighbor_shear), 4),
                "reference_shear": None if reference_shear is None else round(float(reference_shear), 4),
                "applied_shear": round(float(applied), 4),
            }
        )
    return {"enabled": True, "per_char": items}


def apply_char_slot_shear(
    alpha_layer: Image.Image,
    slot: TextRun,
    shear_x: float,
) -> Image.Image:
    if abs(shear_x) < 0.012:
        return alpha_layer

    arr = np.array(alpha_layer, dtype=np.uint8)
    if not np.any(arr > 0):
        return alpha_layer

    h, w = arr.shape
    pad = 6
    x1 = max(0, slot.x1 - pad)
    y1 = max(0, slot.y1 - pad)
    x2 = min(w, slot.x2 + pad)
    y2 = min(h, slot.y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return alpha_layer

    crop = arr[y1:y2, x1:x2]
    if not np.any(crop > 0):
        return alpha_layer

    yy, xx = np.indices(crop.shape, dtype=np.float32)
    center_y = float((slot.y1 + slot.y2) / 2.0 - y1)
    map_x = xx - float(shear_x) * (yy - center_y)
    map_y = yy
    warped = cv2.remap(
        crop,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    result = arr.copy()
    result[y1:y2, x1:x2] = np.maximum(result[y1:y2, x1:x2], warped)
    result[y1:y2, x1:x2] = np.where(crop > 0, warped, result[y1:y2, x1:x2])
    return Image.fromarray(result, mode="L")


def draw_replacement_layer(
    *,
    size: tuple[int, int],
    plan: RenderPlan,
    params: CandidateParams,
    original: Image.Image | None = None,
) -> Image.Image:
    font = ImageFont.truetype(params.font_path, params.font_size)
    alpha_layer = Image.new("L", size, 0)
    alpha = int(max(0.0, min(1.0, params.opacity)) * 255)
    offsets = params.char_offsets or default_char_offsets(plan.target_text)

    chars = text_chars(plan.target_text)
    changed_indices = changed_text_slot_indices(plan)
    target_slots = target_char_slots_for_plan(plan)
    use_slots = (
        plan.draw_mode == "per_char"
        or plan.draw_mode == "line_chars"
        or (plan.draw_mode == "auto" and is_mostly_cjk(plan.target_text) and len(plan.slot_boxes) >= len(chars))
    )
    row_center_y = slot_row_center_y(plan, len(chars)) if plan.draw_mode == "line_chars" else None
    center_in_slot = plan.placement_strategy == "center_primary"

    if use_slots and chars:
        ordered_slots = target_slots if target_slots else tuple(sorted(plan.slot_boxes, key=lambda item: item.x1))
        for idx, ch in enumerate(chars):
            if changed_indices is not None and idx not in changed_indices:
                continue
            slot = ordered_slots[min(idx, len(ordered_slots) - 1)]
            char_layer = Image.new("L", size, 0)
            draw = ImageDraw.Draw(char_layer)
            x, y = char_text_position(
                font,
                ch,
                slot,
                params,
                idx,
                offsets,
                row_center_y=row_center_y,
                center_in_slot=center_in_slot,
            )
            draw.text((x, y), ch, font=font, fill=alpha)
            shear_x = reference_slot_shear(original, plan, idx, params)
            char_layer = apply_char_slot_shear(char_layer, slot, shear_x)
            alpha_layer = Image.fromarray(
                np.maximum(np.array(alpha_layer, dtype=np.uint8), np.array(char_layer, dtype=np.uint8)),
                mode="L",
            )
    elif plan.draw_mode == "span_chars" and chars and plan.slot_boxes:
        draw = ImageDraw.Draw(alpha_layer)
        ordered_slots = tuple(sorted(plan.slot_boxes, key=lambda item: item.x1))
        span_x1 = min(slot.x1 for slot in ordered_slots)
        span_x2 = max(slot.x2 for slot in ordered_slots)
        x1, y1, _x2, y2 = plan.target_roi
        span_w = max(1.0, float(span_x2 - span_x1))
        cell_w = span_w / max(1, len(chars))
        for idx, ch in enumerate(chars):
            bbox = font_text_bbox(font, ch)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            cdx, cdy = offsets[idx] if idx < len(offsets) else (0, 0)
            cell_center_x = span_x1 + cell_w * (idx + 0.5)
            x = cell_center_x - text_w / 2.0 - bbox[0] + params.text_dx + cdx
            y = y1 + ((y2 - y1) - text_h) / 2.0 - bbox[1] + params.text_dy + cdy
            draw.text((x, y), ch, font=font, fill=alpha)
    elif plan.draw_mode == "center":
        draw = ImageDraw.Draw(alpha_layer)
        bbox = font_text_bbox(font, plan.target_text)
        x1, y1, x2, y2 = plan.target_roi
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = x1 + ((x2 - x1) - text_w) / 2.0 - bbox[0] + params.text_dx
        y = y1 + ((y2 - y1) - text_h) / 2.0 - bbox[1] + params.text_dy
        draw.text((x, y), plan.target_text, font=font, fill=alpha)
    else:
        draw = ImageDraw.Draw(alpha_layer)
        bbox = font_text_bbox(font, plan.target_text)
        x1, y1, _, _ = plan.target_roi
        x = x1 - bbox[0] + params.text_dx
        y = y1 - bbox[1] + params.text_dy
        draw.text((x, y), plan.target_text, font=font, fill=alpha)

    alpha_layer = apply_ink_gain(alpha_layer, params.ink_gain)
    alpha_layer = apply_fractional_stroke(alpha_layer, params.stroke_opacity)
    if params.blur > 0:
        alpha_layer = alpha_layer.filter(ImageFilter.GaussianBlur(params.blur))
    alpha_layer = apply_alpha_contrast(alpha_layer, params.alpha_contrast)
    alpha_layer = apply_core_ink_gain(alpha_layer, params.core_ink_gain)
    alpha_layer = apply_text_angle(alpha_layer, text_rotation_box(plan), plan.text_angle_degrees)
    alpha_layer = apply_photo_alpha_warp(alpha_layer, text_rotation_box(plan), params.photo_warp)
    alpha_layer = apply_photo_alpha_resample(
        alpha_layer,
        text_rotation_box(plan),
        max(0.0, params.edge_breakup - 0.05) * 4.0,
    )
    return rgba_from_alpha(alpha_layer)


def render_candidate(
    original: Image.Image,
    plan: RenderPlan,
    params: CandidateParams,
) -> Image.Image:
    original = original.convert("RGB")
    arr = np.array(original)
    h, w = arr.shape[:2]
    x1, y1, x2, y2 = plan.target_roi

    mask_roi = old_text_mask_roi(plan)
    source_chars = text_chars(plan.source_text)
    target_chars = text_chars(plan.target_text)
    if source_chars and target_chars and len(target_chars) <= len(source_chars) and plan.slot_boxes:
        mask = build_slot_text_mask(
            arr,
            plan,
            threshold=params.mask_threshold,
            dilate_iterations=params.mask_dilate_iterations,
        )
        if not np.any(mask):
            mask = build_old_text_mask(
                arr,
                mask_roi,
                threshold=params.mask_threshold,
                dilate_iterations=params.mask_dilate_iterations,
            )
    else:
        mask = build_old_text_mask(
            arr,
            mask_roi,
            threshold=params.mask_threshold,
            dilate_iterations=params.mask_dilate_iterations,
        )
    layer = draw_replacement_layer(size=(w, h), plan=plan, params=params, original=original)
    source_slot_cleanup_mask = build_source_slot_cleanup_mask(
        arr,
        plan,
        threshold=165,
        dilate_iterations=max(1, min(2, params.mask_dilate_iterations)),
    )
    if np.any(source_slot_cleanup_mask):
        mask = np.maximum(mask, source_slot_cleanup_mask)
    trailing_cleanup_mask = build_trailing_value_cleanup_mask(plan, layer, (w, h))
    if np.any(trailing_cleanup_mask):
        mask = np.maximum(mask, trailing_cleanup_mask)
    base_arr = cv2.inpaint(arr, mask, params.inpaint_radius, cv2.INPAINT_TELEA)
    target_expanded = plan.source_reference_box is not None and plan.source_reference_box != plan.target_roi
    if target_expanded:
        base_arr = reconstruct_background_plane_texture(arr, base_arr, mask, mask_roi)
    else:
        base_arr = restore_inpainted_texture(arr, base_arr, mask)
    base_arr = retexture_inpaint_patch(
        arr,
        base_arr,
        mask,
        mask_roi,
        strength=background_retexture_strength(params, target_expanded=target_expanded),
    )
    base_arr = suppress_inpaint_glow(arr, base_arr, mask)
    base_arr = suppress_inpaint_shadow(arr, base_arr, mask)
    base_arr = repair_mask_from_row_background(arr, base_arr, mask, mask_roi)
    base_arr = suppress_inpaint_shadow(arr, base_arr, mask, strength=0.45)
    trailing_bbox = mask_bbox(trailing_cleanup_mask > 0)
    if trailing_bbox is not None:
        base_arr = repair_mask_from_row_background(
            arr,
            base_arr,
            trailing_cleanup_mask,
            trailing_bbox,
            strength=0.72,
        )
        base_arr = retexture_inpaint_patch(
            arr,
            base_arr,
            trailing_cleanup_mask,
            trailing_bbox,
            strength=max(0.58, background_retexture_strength(params, target_expanded=target_expanded)),
            residual_scale=0.72,
        )
        base_arr = suppress_inpaint_glow(arr, base_arr, trailing_cleanup_mask, strength=0.35)
        base_arr = suppress_inpaint_shadow(arr, base_arr, trailing_cleanup_mask, strength=0.35)
    base_arr = repair_extra_source_slot_background(arr, base_arr, plan)
    base = Image.fromarray(base_arr).convert("RGBA")
    if target_expanded or params.edge_breakup > 0:
        layer = rgba_from_alpha(
            apply_scan_edge_breakup(
                layer.getchannel("A"),
                plan.target_roi,
                strength=max(0.025 if target_expanded else 0.0, params.edge_breakup),
                quant_step=10 if params.edge_breakup > 0 else 12,
                scanline_strength=0.018 if params.edge_breakup > 0 else 0.02,
            )
        )
    edited = Image.alpha_composite(base, layer).convert("RGB")
    edited_arr = np.array(edited)
    edited_arr = apply_core_darken(edited_arr, layer.getchannel("A"), params)
    edited_arr = apply_roi_scan_texture(
        arr,
        edited_arr,
        plan.target_roi,
        mask,
        strength=roi_scan_texture_strength(params, target_expanded=target_expanded),
    )
    edited_arr = apply_photo_text_texture(edited_arr, arr, layer.getchannel("A"), plan.target_roi, params)
    edited_arr = feather_roi_boundary(arr, edited_arr, plan.target_roi, layer.getchannel("A"), mask)
    preserve_mask = unchanged_text_slot_mask(plan, (w, h)) > 0
    if np.any(preserve_mask):
        edited_arr[preserve_mask] = arr[preserve_mask]

    outside = np.ones((h, w), dtype=bool)
    outside[y1:y2, x1:x2] = False
    edited_arr[outside] = arr[outside]
    return Image.fromarray(edited_arr)


def hard_check(
    original: Image.Image,
    candidate: Image.Image,
    roi: tuple[int, int, int, int],
    protected_boxes: tuple[tuple[int, int, int, int], ...] = (),
) -> dict[str, Any]:
    size_match = original.size == candidate.size
    report: dict[str, Any] = {
        "original_size": list(original.size),
        "candidate_size": list(candidate.size),
        "size_match": size_match,
    }
    if not size_match:
        report.update(
            {
                "outside_roi_changed_pixels": None,
                "border_changed_pixels": None,
                "protected_changed_pixels": None,
                "pass": False,
            }
        )
        return report

    orig = np.array(original.convert("RGB"))
    cand = np.array(candidate.convert("RGB"))
    h, w = orig.shape[:2]
    x1, y1, x2, y2 = roi
    diff = np.any(orig != cand, axis=2)

    outside = np.ones((h, w), dtype=bool)
    outside[y1:y2, x1:x2] = False

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    protected = np.zeros((h, w), dtype=bool)
    for box in protected_boxes:
        px1, py1, px2, py2 = clamp_box(box, original.size)
        protected[py1:py2, px1:px2] = True

    outside_changed = int(np.count_nonzero(diff & outside))
    border_changed = int(np.count_nonzero(diff & border))
    protected_changed = int(np.count_nonzero(diff & protected))
    report.update(
        {
            "roi": list(roi),
            "outside_roi_changed_pixels": outside_changed,
            "border_changed_pixels": border_changed,
            "protected_changed_pixels": protected_changed,
            "pass": outside_changed == 0 and border_changed == 0 and protected_changed == 0,
        }
    )
    return report


def dark_bbox(
    img: Image.Image,
    roi: tuple[int, int, int, int],
    threshold: int = 150,
) -> dict[str, Any] | None:
    x1, y1, x2, y2 = roi
    gray = gray_array(img)
    mask = gray[y1:y2, x1:x2] < threshold
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return {
        "x1": int(xs.min() + x1),
        "y1": int(ys.min() + y1),
        "x2": int(xs.max() + x1 + 1),
        "y2": int(ys.max() + y1 + 1),
        "width": int(xs.max() - xs.min() + 1),
        "height": int(ys.max() - ys.min() + 1),
        "cx": float(xs.mean() + x1),
        "cy": float(ys.mean() + y1),
        "dark_pixels": int(mask.sum()),
    }


def gray_stats(
    img: Image.Image,
    roi: tuple[int, int, int, int],
    threshold: int = 150,
) -> dict[str, Any] | None:
    x1, y1, x2, y2 = roi
    gray = gray_array(img)
    roi_gray = gray[y1:y2, x1:x2]
    mask = roi_gray < threshold
    if int(mask.sum()) == 0:
        return None
    return {
        "dark_pixels": int(mask.sum()),
        "mean_gray": float(roi_gray[mask].mean()),
        "min_gray": int(roi_gray[mask].min()),
        "max_gray": int(roi_gray[mask].max()),
    }


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def font_style_score(
    original: Image.Image,
    *,
    reference_box: tuple[int, int, int, int],
    reference_text: str,
    reference_kind: str,
    slot_boxes: tuple[TextRun, ...] = (),
    font_name: str,
    font_path: str,
    font_size: int,
    opacity: float = 0.86,
    blur: float = 0.35,
    stroke_opacity: float = 0.0,
    ink_gain: float = 0.0,
    alpha_contrast: float = 0.0,
    core_ink_gain: float = 0.0,
    threshold: int = 150,
) -> dict[str, Any] | None:
    x1, y1, x2, y2 = reference_box
    patch = original.crop(reference_box).convert("RGB")
    gray = cv2.cvtColor(np.array(patch), cv2.COLOR_RGB2GRAY)
    original_mask = gray < threshold
    original_bbox = mask_bbox(original_mask)
    if original_bbox is None:
        return None

    original_pixels = int(original_mask.sum())
    try:
        font = ImageFont.truetype(font_path, font_size)
    except OSError:
        return None

    layer = Image.new("L", patch.size, 0)
    draw = ImageDraw.Draw(layer)
    fill = int(255 * max(0.0, min(1.0, opacity)))
    chars = [ch for ch in reference_text if not ch.isspace()]
    use_slots = bool(slot_boxes) and len(slot_boxes) >= len(chars) and len(chars) > 0
    if use_slots:
        for idx, ch in enumerate(chars):
            slot = slot_boxes[min(idx, len(slot_boxes) - 1)]
            text_bbox = font_text_bbox(font, ch)
            draw_x = slot.x1 - x1 - text_bbox[0]
            draw_y = slot.y1 - y1 - text_bbox[1]
            draw.text((draw_x, draw_y), ch, font=font, fill=fill)
    else:
        text_bbox = font_text_bbox(font, reference_text)
        draw_x = original_bbox[0] - text_bbox[0]
        draw_y = original_bbox[1] - text_bbox[1]
        draw.text((draw_x, draw_y), reference_text, font=font, fill=fill)
    layer = apply_ink_gain(layer, ink_gain)
    layer = apply_fractional_stroke(layer, stroke_opacity)
    if blur > 0:
        layer = layer.filter(ImageFilter.GaussianBlur(blur))
    layer = apply_alpha_contrast(layer, alpha_contrast)
    layer = apply_core_ink_gain(layer, core_ink_gain)

    candidate_mask = np.array(layer) > 50
    candidate_bbox = mask_bbox(candidate_mask)
    if candidate_bbox is None:
        return None

    intersection = int(np.logical_and(original_mask, candidate_mask).sum())
    union = int(np.logical_or(original_mask, candidate_mask).sum())
    iou = float(intersection / union) if union else 0.0
    candidate_pixels = int(candidate_mask.sum())
    pixel_ratio = float(candidate_pixels / original_pixels) if original_pixels else None

    original_w = original_bbox[2] - original_bbox[0]
    original_h = original_bbox[3] - original_bbox[1]
    candidate_w = candidate_bbox[2] - candidate_bbox[0]
    candidate_h = candidate_bbox[3] - candidate_bbox[1]
    score = (
        (1.0 - iou) * 100.0
        + abs(candidate_h - original_h) * 4.0
        + abs(candidate_w - original_w) * 2.0
        + (abs((pixel_ratio or 0.0) - 1.0) * 20.0 if pixel_ratio is not None else 100.0)
    )
    return {
        "reference_kind": reference_kind,
        "font_name": font_name,
        "font_path": font_path,
        "font_size": font_size,
        "stroke_opacity": stroke_opacity,
        "ink_gain": ink_gain,
        "alpha_contrast": alpha_contrast,
        "core_ink_gain": core_ink_gain,
        "score": float(score),
        "iou": iou,
        "pixel_ratio": pixel_ratio,
        "original_bbox": list(add_offset(original_bbox, x1, y1)),
        "candidate_bbox": list(add_offset(candidate_bbox, x1, y1)),
        "original_pixels": original_pixels,
        "candidate_pixels": candidate_pixels,
    }


def text_runs_from_json(items: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None) -> tuple[TextRun, ...]:
    return tuple(TextRun(**item) for item in (items or ()))


def build_style_reference_specs(plan: RenderPlan) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    if plan.source_reference_box and plan.source_text:
        references.append(
            {
                "kind": "source_text_roi",
                "reference_text": plan.source_text,
                "reference_box": list(plan.source_reference_box),
                "slot_boxes": [asdict(item) for item in plan.slot_boxes],
                "weight": 0.78,
                "role": "primary",
            }
        )
    if plan.style_reference_box and plan.style_reference_text:
        references.append(
            {
                "kind": "protected_label",
                "reference_text": plan.style_reference_text,
                "reference_box": list(plan.style_reference_box),
                "slot_boxes": [],
                "weight": 0.22 if references else 1.0,
                "role": "secondary" if references else "primary",
            }
        )
    return references


def score_font_against_style_references(
    original: Image.Image,
    references: list[dict[str, Any]],
    *,
    font_name: str,
    font_path: str,
    font_size: int,
    opacity: float = 0.86,
    blur: float = 0.35,
    stroke_opacity: float = 0.0,
    ink_gain: float = 0.0,
    alpha_contrast: float = 0.0,
    core_ink_gain: float = 0.0,
) -> dict[str, Any] | None:
    reference_scores: list[dict[str, Any]] = []
    weighted_total = 0.0
    total_weight = 0.0
    for reference in references:
        score = font_style_score(
            original,
            reference_box=tuple(reference["reference_box"]),
            reference_text=str(reference["reference_text"]),
            reference_kind=str(reference["kind"]),
            slot_boxes=text_runs_from_json(reference.get("slot_boxes")),
            font_name=font_name,
            font_path=font_path,
            font_size=font_size,
            opacity=opacity,
            blur=blur,
            stroke_opacity=stroke_opacity,
            ink_gain=ink_gain,
            alpha_contrast=alpha_contrast,
            core_ink_gain=core_ink_gain,
        )
        if not score:
            continue
        weight = float(reference.get("weight", 1.0))
        score["weight"] = weight
        reference_scores.append(score)
        weighted_total += float(score["score"]) * weight
        total_weight += weight

    if not reference_scores or total_weight <= 0:
        return None

    category = font_category_for_path(font_path, font_name)
    category_penalty = font_category_penalty(category)
    raw_score = weighted_total / total_weight
    adjusted_score = raw_score * (1.0 + category_penalty)
    return {
        "font_name": font_name,
        "font_path": font_path,
        "font_size": font_size,
        "stroke_opacity": stroke_opacity,
        "ink_gain": ink_gain,
        "alpha_contrast": alpha_contrast,
        "core_ink_gain": core_ink_gain,
        "category": category,
        "category_penalty": category_penalty,
        "raw_score": float(raw_score),
        "score": float(adjusted_score),
        "reference_scores": reference_scores,
    }


def build_font_style_reference(
    original: Image.Image,
    plan: RenderPlan,
    font_candidates: list[tuple[str, str]],
    *,
    min_size: int = 16,
    max_size: int = 28,
    prefer_serif_categories: bool = True,
) -> dict[str, Any]:
    references = build_style_reference_specs(plan)
    if not references:
        return {
            "enabled": False,
            "reason": "style reference boxes or text are not configured",
            "best_available": None,
            "preferred_best_available": None,
            "by_path": {},
            "references": [],
        }

    by_path: dict[str, Any] = {}
    all_scores: list[dict[str, Any]] = []
    for font_name, font_path in font_candidates:
        best: dict[str, Any] | None = None
        for font_size in range(min_size, max_size + 1):
            score = score_font_against_style_references(
                original,
                references,
                font_name=font_name,
                font_path=font_path,
                font_size=font_size,
            )
            if score and (best is None or score["score"] < best["score"]):
                best = score
        if best:
            by_path[font_path] = best
            all_scores.append(best)

    best_available = min(all_scores, key=lambda item: item["score"]) if all_scores else None
    preferred_scores = [
        item for item in all_scores
        if item.get("category") in PREFERRED_STYLE_CATEGORIES
    ]
    preferred_best_available = (
        min(preferred_scores, key=lambda item: item["score"]) if preferred_scores else None
    )
    return {
        "enabled": best_available is not None,
        "reference_text": references[0]["reference_text"],
        "reference_box": references[0]["reference_box"],
        "primary_reference_kind": references[0]["kind"],
        "references": references,
        "best_available": best_available,
        "preferred_best_available": preferred_best_available,
        "preferred_categories": sorted(PREFERRED_STYLE_CATEGORIES),
        "prefer_serif_categories": prefer_serif_categories,
        "by_path": by_path,
        "ranked_fonts": sorted(all_scores, key=lambda item: item["score"]),
    }


def rank_fonts_by_style_reference(
    font_candidates: list[tuple[str, str]],
    font_style_reference: dict[str, Any],
) -> list[tuple[str, str]]:
    if not font_style_reference.get("enabled"):
        return font_candidates
    scores = {
        item["font_path"]: item["score"]
        for item in font_style_reference.get("ranked_fonts", [])
    }
    return sorted(font_candidates, key=lambda item: scores.get(item[1], float("inf")))


def font_style_gate(
    original: Image.Image,
    plan: RenderPlan,
    params: CandidateParams,
    font_style_reference: dict[str, Any],
    *,
    max_score_ratio: float,
) -> dict[str, Any]:
    if not font_style_reference.get("enabled"):
        return {"enabled": False, "pass": True, "issues": []}

    best_available = font_style_reference.get("best_available")
    if not best_available or not best_available.get("score"):
        return {"enabled": False, "pass": True, "issues": []}

    references = font_style_reference.get("references") or []
    candidate = score_font_against_style_references(
        original,
        references,
        font_name=params.font_name,
        font_path=params.font_path,
        font_size=params.font_size,
        opacity=params.opacity,
        blur=params.blur,
        stroke_opacity=params.stroke_opacity,
        ink_gain=params.ink_gain,
        alpha_contrast=params.alpha_contrast,
        core_ink_gain=params.core_ink_gain,
    )
    font_best = font_style_reference.get("by_path", {}).get(params.font_path)
    if not candidate or not font_best:
        return {
            "enabled": True,
            "pass": False,
            "issues": [{"type": "font_style_score_missing", "font": params.font_name}],
            "best_available": best_available,
            "candidate": candidate,
            "font_best": font_best,
        }

    best_score = float(best_available["score"])
    candidate_ratio = float(candidate["score"] / best_score)
    font_best_ratio = float(font_best["score"] / best_score)
    issues: list[dict[str, Any]] = []
    if font_best_ratio > max_score_ratio:
        issues.append(
            {
                "type": "font_family_style_score_ratio",
                "font": params.font_name,
                "actual": font_best_ratio,
                "limit": max_score_ratio,
                "font_best_score": font_best["score"],
                "best_available_score": best_score,
                "best_available_font": best_available["font_name"],
            }
        )
    if candidate_ratio > max_score_ratio * 1.12:
        issues.append(
            {
                "type": "font_render_style_score_ratio",
                "font": params.font_name,
                "actual": candidate_ratio,
                "limit": max_score_ratio * 1.12,
                "candidate_score": candidate["score"],
                "best_available_score": best_score,
                "best_available_font": best_available["font_name"],
            }
        )
    preferred_best = font_style_reference.get("preferred_best_available")
    preferred_categories = set(font_style_reference.get("preferred_categories") or [])
    if (
        font_style_reference.get("prefer_serif_categories", True)
        and preferred_best
        and candidate.get("category") not in preferred_categories
        and candidate.get("category") != "manual"
    ):
        issues.append(
            {
                "type": "font_category_not_preferred",
                "font": params.font_name,
                "category": candidate.get("category"),
                "preferred_categories": sorted(preferred_categories),
                "preferred_best_font": preferred_best.get("font_name"),
                "preferred_best_score": preferred_best.get("score"),
            }
        )

    return {
        "enabled": True,
        "pass": not issues,
        "issues": issues,
        "max_score_ratio": max_score_ratio,
        "best_available": best_available,
        "preferred_best_available": preferred_best,
        "font_best": {
            **font_best,
            "score_ratio_to_best": font_best_ratio,
        },
        "candidate": {
            **candidate,
            "score_ratio_to_best": candidate_ratio,
        },
    }


STRICT_ACCEPTANCE_APPENDIX = """

附加严格验收规则：
1. 不要因为硬校验通过就自动给 pass=true。硬校验只代表像素边界合规。
2. 如果新文字肉眼明显更黑、更重、更粗、灰边面积更大，必须 pass=false，acceptance_level=marginal 或 fail。
2a. 如果新文字边缘比旧文字明显更软、更糊、失焦感更强，也必须 pass=false，acceptance_level=marginal 或 fail。
3. 如果字体结构与原图字体仍明显不同，必须 pass=false，acceptance_level=marginal 或 fail。
4. 如果 strict_visual_metrics 中任一 threshold 的 dark_pixel_ratio 超过 max_dark_pixel_ratio 或低于 min_dark_pixel_ratio，必须 pass=false。
5. 如果核心笔画平均灰度或灰边平均灰度与旧字差距超过阈值，必须 pass=false；如果核心笔画明显偏浅，必须 pass=false。
6. 如果 char_alignment_metrics 中字距或单字中心偏离旧字 slot 超过阈值，必须 pass=false。
7. 如果 font_style_gate 中 pass=false，必须 pass=false；尤其要尊重 source_text_roi 对原旧文字 ROI 的字体结构评分。
8. 如果文字说明中出现“字体结构仍有差异”“字体风格仍略有差异”“不应直接判定通过”等结论，即使 visual_findings.font_similarity 写成 ok，也必须 pass=false。
9. 必须先验收 text_shape 阶段：字体、字号、字槽、基线和严重局部姿态存在 hard-blocking issues 时，不能用黑度、模糊、噪声或背景质感解释为通过；若本地已把受黑灰影响的笔画体量诊断标记为 deferred_issues，则应进入 deferred_to_stage 继续修复，但不能让 hard text_shape 指标回退。
10. 只有字体、黑度、笔画重量、灰边、位置、背景都无明显差异时，才允许 acceptance_level=pass。
"""


def strict_visual_metrics(
    original: Image.Image,
    candidate: Image.Image,
    roi: tuple[int, int, int, int],
    *,
    thresholds: tuple[int, ...] = (120, 140, 150, 160, 165),
) -> dict[str, Any]:
    result: dict[str, Any] = {"thresholds": {}}
    old_gray = gray_array(original)
    new_gray = gray_array(candidate)
    x1, y1, x2, y2 = roi
    old_roi = old_gray[y1:y2, x1:x2]
    new_roi = new_gray[y1:y2, x1:x2]
    old_lt55 = int((old_roi < 55).sum())
    new_lt55 = int((new_roi < 55).sum())
    old_lt70 = int((old_roi < 70).sum())
    new_lt70 = int((new_roi < 70).sum())
    old_lt90 = int((old_roi < 90).sum())
    new_lt90 = int((new_roi < 90).sum())
    old_lt120 = int((old_roi < 120).sum())
    new_lt120 = int((new_roi < 120).sum())
    old_55_70 = old_lt70 - old_lt55
    new_55_70 = new_lt70 - new_lt55
    old_70_90 = old_lt90 - old_lt70
    new_70_90 = new_lt90 - new_lt70
    old_90_120 = old_lt120 - old_lt90
    new_90_120 = new_lt120 - new_lt90
    old_70_120 = old_lt120 - old_lt70
    new_70_120 = new_lt120 - new_lt70
    old_120_165 = int(((old_roi >= 120) & (old_roi < 165)).sum())
    new_120_165 = int(((new_roi >= 120) & (new_roi < 165)).sum())
    old_lt165 = int((old_roi < 165).sum())
    new_lt165 = int((new_roi < 165).sum())
    result["bands"] = {
        "old_lt55_pixels": old_lt55,
        "new_lt55_pixels": new_lt55,
        "lt55_delta": new_lt55 - old_lt55,
        "old_lt70_pixels": old_lt70,
        "new_lt70_pixels": new_lt70,
        "lt70_delta": new_lt70 - old_lt70,
        "old_lt90_pixels": old_lt90,
        "new_lt90_pixels": new_lt90,
        "lt90_delta": new_lt90 - old_lt90,
        "old_lt120_pixels": old_lt120,
        "new_lt120_pixels": new_lt120,
        "lt120_delta": new_lt120 - old_lt120,
        "old_55_70_pixels": old_55_70,
        "new_55_70_pixels": new_55_70,
        "band_55_70_delta": new_55_70 - old_55_70,
        "old_70_90_pixels": old_70_90,
        "new_70_90_pixels": new_70_90,
        "band_70_90_delta": new_70_90 - old_70_90,
        "old_90_120_pixels": old_90_120,
        "new_90_120_pixels": new_90_120,
        "band_90_120_delta": new_90_120 - old_90_120,
        "old_70_120_pixels": old_70_120,
        "new_70_120_pixels": new_70_120,
        "band_70_120_delta": new_70_120 - old_70_120,
        "old_120_165_pixels": old_120_165,
        "new_120_165_pixels": new_120_165,
        "band_120_165_delta": new_120_165 - old_120_165,
        "old_lt165_pixels": old_lt165,
        "new_lt165_pixels": new_lt165,
        "old_lt55_share_of_lt165": float(old_lt55 / old_lt165) if old_lt165 else None,
        "new_lt55_share_of_lt165": float(new_lt55 / new_lt165) if new_lt165 else None,
        "old_lt70_share_of_lt165": float(old_lt70 / old_lt165) if old_lt165 else None,
        "new_lt70_share_of_lt165": float(new_lt70 / new_lt165) if new_lt165 else None,
        "old_lt90_share_of_lt165": float(old_lt90 / old_lt165) if old_lt165 else None,
        "new_lt90_share_of_lt165": float(new_lt90 / new_lt165) if new_lt165 else None,
        "old_120_165_share_of_lt165": float(old_120_165 / old_lt165) if old_lt165 else None,
        "new_120_165_share_of_lt165": float(new_120_165 / new_lt165) if new_lt165 else None,
    }
    for threshold in thresholds:
        old_mask = old_roi < threshold
        new_mask = new_roi < threshold
        old_pixels = int(old_mask.sum())
        new_pixels = int(new_mask.sum())
        old_mean = float(old_roi[old_mask].mean()) if old_pixels else None
        new_mean = float(new_roi[new_mask].mean()) if new_pixels else None
        ratio = float(new_pixels / old_pixels) if old_pixels else None
        result["thresholds"][str(threshold)] = {
            "old_dark_pixels": old_pixels,
            "new_dark_pixels": new_pixels,
            "dark_pixel_ratio": ratio,
            "old_mean_gray": old_mean,
            "new_mean_gray": new_mean,
            "mean_gray_delta": (new_mean - old_mean) if old_mean is not None and new_mean is not None else None,
        }
    return result


def gray_band_counts(gray: np.ndarray) -> dict[str, int]:
    lt35 = int((gray < 35).sum())
    lt40 = int((gray < 40).sum())
    lt45 = int((gray < 45).sum())
    lt55 = int((gray < 55).sum())
    lt70 = int((gray < 70).sum())
    lt90 = int((gray < 90).sum())
    lt120 = int((gray < 120).sum())
    lt165 = int((gray < 165).sum())
    return {
        "lt35": lt35,
        "lt40": lt40,
        "lt45": lt45,
        "lt55": lt55,
        "lt70": lt70,
        "lt90": lt90,
        "lt120": lt120,
        "lt165": lt165,
        "band_55_70": lt70 - lt55,
        "band_70_90": lt90 - lt70,
        "band_90_120": lt120 - lt90,
        "band_120_165": lt165 - lt120,
    }


def char_gray_band_metrics(
    original: Image.Image,
    candidate: Image.Image,
    plan: RenderPlan,
) -> dict[str, Any]:
    target_chars = [ch for ch in plan.target_text if not ch.isspace()]
    source_chars = [ch for ch in (plan.source_text or "") if not ch.isspace()]
    if not is_mostly_cjk(plan.source_text or plan.target_text):
        return {"enabled": False, "reason": "non-CJK span replacement uses ROI-level metrics"}
    if plan.draw_mode == "center":
        return {"enabled": False, "reason": "centered replacement has no per-character target slots"}
    target_slots = target_char_slots_for_plan(plan)
    if not target_slots or not target_chars:
        return {"enabled": False, "reason": "missing per-character slots"}

    old_gray = gray_array(original)
    new_gray = gray_array(candidate)
    items: list[dict[str, Any]] = []
    for idx, ch in enumerate(target_chars[: len(target_slots)]):
        slot = target_slots[idx]
        reference_slot = source_slot_for_target_index(plan, idx) or slot
        old_roi = old_gray[reference_slot.y1 : reference_slot.y2, reference_slot.x1 : reference_slot.x2]
        new_roi = new_gray[slot.y1 : slot.y2, slot.x1 : slot.x2]
        old_counts = gray_band_counts(old_roi)
        new_counts = gray_band_counts(new_roi)
        deltas = {key: new_counts[key] - old_counts[key] for key in old_counts}
        items.append(
            {
                "index": idx,
                "source_char": source_chars[idx] if idx < len(source_chars) else None,
                "target_char": ch,
                "slot_box": [slot.x1, slot.y1, slot.x2, slot.y2],
                "reference_slot_box": [
                    reference_slot.x1,
                    reference_slot.y1,
                    reference_slot.x2,
                    reference_slot.y2,
                ],
                "old": old_counts,
                "new": new_counts,
                "delta": deltas,
            }
        )
    return {"enabled": True, "per_char": items}


def extra_source_slot_cleanup_metrics(
    original: Image.Image,
    candidate: Image.Image,
    plan: RenderPlan,
    params: CandidateParams | None = None,
) -> dict[str, Any]:
    boxes = extra_source_slot_cleanup_boxes(plan)
    if not boxes:
        return {"enabled": False, "reason": "replacement is not shorter than source text"}

    old_gray = gray_array(original)
    new_gray = gray_array(candidate)
    size = original.size
    ignore_text = np.zeros(old_gray.shape, dtype=bool)
    if params is not None:
        alpha = np.array(draw_replacement_layer(size=size, plan=plan, params=params).getchannel("A"))
        ignore_text = alpha > 18

    items: list[dict[str, Any]] = []
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = clamp_box(box, size)
        if x2 <= x1 or y2 <= y1:
            continue
        valid = ~ignore_text[y1:y2, x1:x2]
        area = int(np.count_nonzero(valid))
        if area <= 0:
            continue
        old_roi = old_gray[y1:y2, x1:x2][valid]
        new_roi = new_gray[y1:y2, x1:x2][valid]

        column_means: list[float] = []
        local_new = new_gray[y1:y2, x1:x2]
        for col in range(x2 - x1):
            keep = valid[:, col]
            if int(np.count_nonzero(keep)) < 3:
                continue
            column_means.append(float(local_new[:, col][keep].mean()))
        if column_means:
            col_arr = np.array(column_means, dtype=np.float32)
            median = float(np.median(col_arr))
            max_column_mean_deviation = float(np.max(np.abs(col_arr - median)))
            p95_column_mean_deviation = float(np.percentile(np.abs(col_arr - median), 95))
        else:
            max_column_mean_deviation = 0.0
            p95_column_mean_deviation = 0.0

        old_lt120 = int(np.count_nonzero(old_roi < 120))
        old_lt150 = int(np.count_nonzero(old_roi < 150))
        old_lt165 = int(np.count_nonzero(old_roi < 165))
        new_lt120 = int(np.count_nonzero(new_roi < 120))
        new_lt150 = int(np.count_nonzero(new_roi < 150))
        new_lt165 = int(np.count_nonzero(new_roi < 165))
        items.append(
            {
                "index": index,
                "box": [x1, y1, x2, y2],
                "area": area,
                "old_lt120_pixels": old_lt120,
                "old_lt150_pixels": old_lt150,
                "old_lt165_pixels": old_lt165,
                "new_lt120_pixels": new_lt120,
                "new_lt150_pixels": new_lt150,
                "new_lt165_pixels": new_lt165,
                "new_lt120_ratio": float(new_lt120 / area),
                "new_lt150_ratio": float(new_lt150 / area),
                "new_lt165_ratio": float(new_lt165 / area),
                "lt150_retention_ratio": float(new_lt150 / old_lt150) if old_lt150 else None,
                "mean_gray": float(new_roi.mean()),
                "max_column_mean_deviation": max_column_mean_deviation,
                "p95_column_mean_deviation": p95_column_mean_deviation,
            }
        )

    return {"enabled": True, "boxes": [list(item) for item in boxes], "per_box": items}


def extra_source_slot_cleanup_issues(
    metrics: dict[str, Any],
    *,
    max_lt120_ratio: float = 0.012,
    max_lt150_ratio: float = 0.055,
    max_lt150_retention_ratio: float = 0.22,
    max_column_mean_deviation: float = 7.0,
) -> list[dict[str, Any]]:
    if not metrics.get("enabled"):
        return []

    issues: list[dict[str, Any]] = []
    for item in metrics.get("per_box", []):
        lt120_ratio = item.get("new_lt120_ratio")
        lt150_ratio = item.get("new_lt150_ratio")
        retention = item.get("lt150_retention_ratio")
        column_deviation = item.get("max_column_mean_deviation")
        if lt120_ratio is not None and float(lt120_ratio) > max_lt120_ratio:
            issues.append(
                {
                    "type": "extra_source_slot_dark_core_residue",
                    "box": item.get("box"),
                    "actual": lt120_ratio,
                    "limit": max_lt120_ratio,
                    "new_lt120_pixels": item.get("new_lt120_pixels"),
                }
            )
        if lt150_ratio is not None and float(lt150_ratio) > max_lt150_ratio:
            issues.append(
                {
                    "type": "extra_source_slot_dark_residue",
                    "box": item.get("box"),
                    "actual": lt150_ratio,
                    "limit": max_lt150_ratio,
                    "new_lt150_pixels": item.get("new_lt150_pixels"),
                }
            )
        if retention is not None and float(retention) > max_lt150_retention_ratio:
            issues.append(
                {
                    "type": "extra_source_slot_dark_retention",
                    "box": item.get("box"),
                    "actual": retention,
                    "limit": max_lt150_retention_ratio,
                    "old_lt150_pixels": item.get("old_lt150_pixels"),
                    "new_lt150_pixels": item.get("new_lt150_pixels"),
                }
            )
        if column_deviation is not None and float(column_deviation) > max_column_mean_deviation:
            issues.append(
                {
                    "type": "extra_source_slot_vertical_residue",
                    "box": item.get("box"),
                    "actual": column_deviation,
                    "limit": max_column_mean_deviation,
                }
            )
    return issues


def strict_acceptance_gate(
    acceptance: dict[str, Any] | None,
    metrics: dict[str, Any],
    *,
    max_dark_pixel_ratio: float,
    min_dark_pixel_ratio: float,
    max_core_mean_gray_delta: float,
    max_edge_mean_gray_delta: float,
    max_core_lighten_delta: float,
    max_edge_lighten_delta: float,
    max_char_center_dx: float,
    max_char_center_distance_delta: float,
    font_style_report: dict[str, Any] | None = None,
    char_alignment_report: dict[str, Any] | None = None,
    enforce_font_similarity: bool = True,
) -> dict[str, Any]:
    issues = strict_gate_issues(
        metrics,
        max_dark_pixel_ratio=max_dark_pixel_ratio,
        min_dark_pixel_ratio=min_dark_pixel_ratio,
        max_core_mean_gray_delta=max_core_mean_gray_delta,
        max_edge_mean_gray_delta=max_edge_mean_gray_delta,
        max_core_lighten_delta=max_core_lighten_delta,
        max_edge_lighten_delta=max_edge_lighten_delta,
    )
    char_issues = char_alignment_issues(
        char_alignment_report or {},
        max_char_center_dx=max_char_center_dx,
        max_char_center_distance_delta=max_char_center_distance_delta,
    )
    font_issues = list((font_style_report or {}).get("issues", []))
    model_font_issues = model_font_similarity_issues(acceptance) if enforce_font_similarity else []

    gated = copy.deepcopy(acceptance) if acceptance else {}
    gated["strict_visual_metrics"] = metrics
    if font_style_report is not None:
        gated["font_style_gate"] = font_style_report
    if char_alignment_report is not None:
        gated["char_alignment_metrics"] = char_alignment_report
    gated["strict_gate"] = {
        "max_dark_pixel_ratio": max_dark_pixel_ratio,
        "min_dark_pixel_ratio": min_dark_pixel_ratio,
        "max_core_mean_gray_delta": max_core_mean_gray_delta,
        "max_edge_mean_gray_delta": max_edge_mean_gray_delta,
        "max_core_lighten_delta": max_core_lighten_delta,
        "max_edge_lighten_delta": max_edge_lighten_delta,
        "max_char_center_dx": max_char_center_dx,
        "max_char_center_distance_delta": max_char_center_distance_delta,
        "issues": issues + char_issues + font_issues + model_font_issues,
        "pass": not issues and not char_issues and not font_issues and not model_font_issues,
    }
    if issues or char_issues or font_issues or model_font_issues:
        gated["pass"] = False
        gated["acceptance_level"] = "marginal"
        gated["final_decision"] = "revise"
        reason = gated.get("reason", "")
        reasons = []
        if issues:
            reasons.append("新文字深色像素、核心黑度或灰边均值未贴近旧字")
        if char_issues:
            reasons.append("新文字单字中心或字间距未贴近旧字")
        if font_issues:
            reasons.append("字体风格参考分数未通过")
        if model_font_issues:
            reasons.append("视觉模型文本中存在字体差异结论")
        gate_reason = "严格验收未通过：" + "；".join(reasons) + "。"
        gated["reason"] = f"{reason} {gate_reason}".strip()
        must_fix = gated.get("must_fix")
        if not isinstance(must_fix, list):
            must_fix = []
        if issues:
            must_fix.append("Reduce text darkness/stroke coverage or use a closer font, then rerun acceptance.")
        if char_issues:
            must_fix.append("Adjust per-character offsets so each new character tracks the old character slots.")
        if font_issues or model_font_issues:
            must_fix.append("Choose a font with a closer old-text ROI style score before accepting.")
        gated["must_fix"] = must_fix
    else:
        gated.setdefault("pass", True)
        gated.setdefault("acceptance_level", "pass")
        gated.setdefault("final_decision", "accept")
    return gated


def model_font_similarity_issues(acceptance: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not acceptance:
        return []
    findings = acceptance.get("visual_findings")
    font_similarity = None
    if isinstance(findings, dict):
        font_similarity = findings.get("font_similarity")

    issues: list[dict[str, Any]] = []
    if font_similarity and str(font_similarity).lower() not in {"ok", "pass"}:
        issues.append(
            {
                "type": "model_font_similarity_field",
                "actual": font_similarity,
                "limit": "ok",
            }
        )

    text_parts = [
        str(acceptance.get("reason", "")),
        json.dumps(acceptance.get("must_fix", []), ensure_ascii=False),
        json.dumps(acceptance.get("optional_tuning", []), ensure_ascii=False),
    ]
    text = "\n".join(text_parts)
    patterns = [
        "字体结构.*差异",
        "字体风格.*差异",
        "字体.*仍.*不同",
        "字体.*不太相似",
        "slightly_off",
        "wrong_style",
        "不应直接判定.*通过",
        "不能.*通过",
    ]
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            issues.append(
                {
                    "type": "model_font_similarity_text",
                    "pattern": pattern,
                }
            )
            break
    return issues


def strict_gate_issues(
    metrics: dict[str, Any],
    *,
    max_dark_pixel_ratio: float,
    min_dark_pixel_ratio: float,
    max_core_mean_gray_delta: float,
    max_edge_mean_gray_delta: float,
    max_core_lighten_delta: float,
    max_edge_lighten_delta: float,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for threshold, values in metrics.get("thresholds", {}).items():
        threshold_int = int(threshold)
        ratio = values.get("dark_pixel_ratio")
        if ratio is not None and ratio > max_dark_pixel_ratio:
            issues.append(
                {
                    "type": "dark_pixel_ratio_too_high",
                    "threshold": threshold_int,
                    "actual": ratio,
                    "limit": max_dark_pixel_ratio,
                    "old_dark_pixels": values.get("old_dark_pixels"),
                    "new_dark_pixels": values.get("new_dark_pixels"),
                }
            )
        if ratio is not None and ratio < min_dark_pixel_ratio:
            issues.append(
                {
                    "type": "dark_pixel_ratio_too_low",
                    "threshold": threshold_int,
                    "actual": ratio,
                    "limit": min_dark_pixel_ratio,
                    "old_dark_pixels": values.get("old_dark_pixels"),
                    "new_dark_pixels": values.get("new_dark_pixels"),
                }
            )
        mean_delta = values.get("mean_gray_delta")
        if mean_delta is None:
            continue
        delta = float(mean_delta)
        if threshold_int == 120 and delta > max_core_lighten_delta:
            issues.append(
                {
                    "type": "core_mean_gray_too_light",
                    "threshold": threshold_int,
                    "actual": mean_delta,
                    "limit": max_core_lighten_delta,
                    "old_mean_gray": values.get("old_mean_gray"),
                    "new_mean_gray": values.get("new_mean_gray"),
                }
            )
        if threshold_int == 165 and delta > max_edge_lighten_delta:
            issues.append(
                {
                    "type": "edge_mean_gray_too_light",
                    "threshold": threshold_int,
                    "actual": mean_delta,
                    "limit": max_edge_lighten_delta,
                    "old_mean_gray": values.get("old_mean_gray"),
                    "new_mean_gray": values.get("new_mean_gray"),
                }
            )
        if threshold_int == 120 and abs(delta) > max_core_mean_gray_delta:
            issues.append(
                {
                    "type": "core_mean_gray_delta",
                    "threshold": threshold_int,
                    "actual": mean_delta,
                    "limit": max_core_mean_gray_delta,
                    "direction": "too_light" if delta > 0 else "too_dark",
                    "old_mean_gray": values.get("old_mean_gray"),
                    "new_mean_gray": values.get("new_mean_gray"),
                }
            )
        if threshold_int == 165 and abs(delta) > max_edge_mean_gray_delta:
            issues.append(
                {
                    "type": "edge_mean_gray_delta",
                    "threshold": threshold_int,
                    "actual": mean_delta,
                    "limit": max_edge_mean_gray_delta,
                    "direction": "too_light" if delta > 0 else "too_dark",
                    "old_mean_gray": values.get("old_mean_gray"),
                    "new_mean_gray": values.get("new_mean_gray"),
                }
            )
    return issues


def replacement_char_bboxes(
    size: tuple[int, int],
    plan: RenderPlan,
    params: CandidateParams,
    *,
    alpha_threshold: int = 50,
) -> list[tuple[int, int, int, int] | None]:
    chars = text_chars(plan.target_text)
    if plan.draw_mode == "center":
        return []
    ordered_slots = target_char_slots_for_plan(plan)
    if not chars or len(ordered_slots) < len(chars):
        return []

    font = ImageFont.truetype(params.font_path, params.font_size)
    boxes: list[tuple[int, int, int, int] | None] = []
    offsets = params.char_offsets or default_char_offsets(plan.target_text)
    row_center_y = slot_row_center_y(plan, len(chars)) if plan.draw_mode == "line_chars" else None
    center_in_slot = plan.placement_strategy == "center_primary"
    changed_indices = changed_text_slot_indices(plan)
    for idx, ch in enumerate(chars):
        if changed_indices is not None and idx not in changed_indices:
            boxes.append(None)
            continue
        layer = Image.new("L", size, 0)
        draw = ImageDraw.Draw(layer)
        slot = ordered_slots[min(idx, len(ordered_slots) - 1)]
        x, y = char_text_position(
            font,
            ch,
            slot,
            params,
            idx,
            offsets,
            row_center_y=row_center_y,
            center_in_slot=center_in_slot,
        )
        draw.text((x, y), ch, font=font, fill=int(max(0.0, min(1.0, params.opacity)) * 255))
        layer = apply_ink_gain(layer, params.ink_gain)
        layer = apply_fractional_stroke(layer, params.stroke_opacity)
        if params.blur > 0:
            layer = layer.filter(ImageFilter.GaussianBlur(params.blur))
        layer = apply_alpha_contrast(layer, params.alpha_contrast)
        layer = apply_core_ink_gain(layer, params.core_ink_gain)
        boxes.append(mask_bbox(np.array(layer) > alpha_threshold))
    return boxes


def char_alignment_metrics(
    size: tuple[int, int],
    plan: RenderPlan,
    params: CandidateParams,
) -> dict[str, Any]:
    chars = text_chars(plan.target_text)
    source_chars = text_chars(plan.source_text)
    if plan.draw_mode == "center":
        return {"enabled": False, "reason": "centered replacement has no per-character target slots"}
    target_slots = target_char_slots_for_plan(plan)
    candidate_boxes = replacement_char_bboxes(size, plan, params)
    if not candidate_boxes or len(target_slots) < len(chars):
        return {"enabled": False, "reason": "missing per-character slots"}

    per_char: list[dict[str, Any]] = []
    candidate_centers: list[tuple[float, float] | None] = []
    slot_centers: list[tuple[float, float]] = []
    ordered_slots = target_slots
    for idx, (ch, candidate_box) in enumerate(zip(chars, candidate_boxes)):
        slot = ordered_slots[idx]
        slot_box = (slot.x1, slot.y1, slot.x2, slot.y2)
        slot_cx = (slot.x1 + slot.x2) / 2.0
        slot_cy = (slot.y1 + slot.y2) / 2.0
        slot_centers.append((slot_cx, slot_cy))
        if candidate_box is None:
            candidate_centers.append(None)
            per_char.append({"index": idx, "char": ch, "candidate_box": None, "slot_box": list(slot_box)})
            continue

        x1, y1, x2, y2 = candidate_box
        candidate_cx = (x1 + x2) / 2.0
        candidate_cy = (y1 + y2) / 2.0
        candidate_centers.append((candidate_cx, candidate_cy))
        per_char.append(
            {
                "index": idx,
                "char": ch,
                "slot_box": list(slot_box),
                "candidate_box": [int(x1), int(y1), int(x2), int(y2)],
                "center_dx": candidate_cx - slot_cx,
                "center_dy": candidate_cy - slot_cy,
                "left_dx": x1 - slot.x1,
                "height_delta": (y2 - y1) - (slot.y2 - slot.y1),
                "width_delta": (x2 - x1) - (slot.x2 - slot.x1),
            }
        )

    metrics: dict[str, Any] = {"enabled": True, "per_char": per_char}
    present_candidate_centers = [center for center in candidate_centers if center is not None]
    if len(present_candidate_centers) >= 2:
        candidate_center_ys = [center[1] for center in present_candidate_centers]
        slot_center_ys = [center[1] for center in slot_centers[: len(present_candidate_centers)]]
        metrics.update(
            {
                "candidate_center_y_range": max(candidate_center_ys) - min(candidate_center_ys),
                "slot_center_y_range": max(slot_center_ys) - min(slot_center_ys),
            }
        )
    if len(candidate_centers) >= 2 and candidate_centers[0] and candidate_centers[1]:
        candidate_distance = candidate_centers[1][0] - candidate_centers[0][0]
        slot_distance = slot_centers[1][0] - slot_centers[0][0]
        metrics.update(
            {
                "candidate_center_distance": candidate_distance,
                "slot_center_distance": slot_distance,
                "center_distance_delta": candidate_distance - slot_distance,
            }
        )
    return metrics


def char_alignment_issues(
    metrics: dict[str, Any],
    *,
    max_char_center_dx: float,
    max_char_center_distance_delta: float,
    max_char_center_dy: float | None = None,
    max_replacement_center_y_range: float | None = None,
) -> list[dict[str, Any]]:
    if not metrics.get("enabled"):
        return []

    issues: list[dict[str, Any]] = []
    distance_delta = metrics.get("center_distance_delta")
    if distance_delta is not None and abs(float(distance_delta)) > max_char_center_distance_delta:
        issues.append(
            {
                "type": "char_center_distance_delta",
                "actual": distance_delta,
                "limit": max_char_center_distance_delta,
                "candidate_center_distance": metrics.get("candidate_center_distance"),
                "slot_center_distance": metrics.get("slot_center_distance"),
            }
        )
    center_y_range = metrics.get("candidate_center_y_range")
    if (
        max_replacement_center_y_range is not None
        and center_y_range is not None
        and float(center_y_range) > max_replacement_center_y_range
    ):
        issues.append(
            {
                "type": "replacement_center_y_range",
                "actual": center_y_range,
                "limit": max_replacement_center_y_range,
                "slot_center_y_range": metrics.get("slot_center_y_range"),
            }
        )

    for item in metrics.get("per_char", []):
        center_dx = item.get("center_dx")
        if center_dx is not None and abs(float(center_dx)) > max_char_center_dx:
            issues.append(
                {
                    "type": "char_center_dx",
                    "index": item.get("index"),
                    "char": item.get("char"),
                    "actual": center_dx,
                    "limit": max_char_center_dx,
                    "slot_box": item.get("slot_box"),
                    "candidate_box": item.get("candidate_box"),
                }
            )
        candidate_box = item.get("candidate_box")
        slot_box = item.get("slot_box")
        if isinstance(candidate_box, list) and len(candidate_box) == 4 and isinstance(slot_box, list) and len(slot_box) == 4:
            slot_h = max(1.0, float(slot_box[3] - slot_box[1]))
            candidate_h = max(0.0, float(candidate_box[3] - candidate_box[1]))
            min_height = max(1.0, slot_h * 0.88)
            max_height = slot_h * 1.22
            height_tolerance_px = max(0.35, min(0.75, slot_h * 0.03))
            if candidate_h + height_tolerance_px < min_height:
                issues.append(
                    {
                        "type": "char_height_too_small",
                        "index": item.get("index"),
                        "char": item.get("char"),
                        "actual": round(candidate_h, 3),
                        "limit": round(min_height, 3),
                        "tolerance_px": round(height_tolerance_px, 3),
                        "slot_height": round(slot_h, 3),
                        "slot_box": slot_box,
                        "candidate_box": candidate_box,
                    }
                )
            elif candidate_h - height_tolerance_px > max_height:
                issues.append(
                    {
                        "type": "char_height_too_large",
                        "index": item.get("index"),
                        "char": item.get("char"),
                        "actual": round(candidate_h, 3),
                        "limit": round(max_height, 3),
                        "tolerance_px": round(height_tolerance_px, 3),
                        "slot_height": round(slot_h, 3),
                        "slot_box": slot_box,
                        "candidate_box": candidate_box,
                    }
                )
        center_dy = item.get("center_dy")
        if (
            max_char_center_dy is not None
            and center_dy is not None
            and abs(float(center_dy)) > max_char_center_dy
        ):
            issues.append(
                {
                    "type": "char_center_dy",
                    "index": item.get("index"),
                    "char": item.get("char"),
                    "actual": center_dy,
                    "limit": max_char_center_dy,
                    "slot_box": item.get("slot_box"),
                    "candidate_box": item.get("candidate_box"),
                }
            )
    return issues


def char_alignment_gate(
    size: tuple[int, int],
    plan: RenderPlan,
    params: CandidateParams,
    *,
    max_char_center_dx: float,
    max_char_center_distance_delta: float,
    max_char_center_dy: float | None = None,
    max_replacement_center_y_range: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics = char_alignment_metrics(size, plan, params)
    issues = char_alignment_issues(
        metrics,
        max_char_center_dx=max_char_center_dx,
        max_char_center_distance_delta=max_char_center_distance_delta,
        max_char_center_dy=max_char_center_dy,
        max_replacement_center_y_range=max_replacement_center_y_range,
    )
    return metrics, issues


def make_contact_sheet(
    items: list[tuple[str, Image.Image]],
    out_path: Path,
    *,
    scale: int = 8,
    cols: int = 4,
) -> None:
    if not items:
        raise ValueError("items must not be empty")
    w, h = items[0][1].size
    label_h = 34
    rows = math.ceil(len(items) / cols)
    sheet = Image.new("RGB", (cols * w * scale, rows * (h * scale + label_h)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for idx, (label, img) in enumerate(items):
        row = idx // cols
        col = idx % cols
        x = col * w * scale
        y = row * (h * scale + label_h)
        draw.text((x + 4, y + 4), label[:70], fill=(0, 0, 0))
        enlarged = img.resize((w * scale, h * scale), Image.Resampling.NEAREST)
        sheet.paste(enlarged, (x, y + label_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def make_compare_image(
    original: Image.Image,
    candidate: Image.Image,
    out_path: Path,
    *,
    scale: int = 10,
) -> None:
    w, h = original.size
    label_h = 26
    sheet = Image.new("RGB", (2 * w * scale, h * scale + label_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 4), "original", fill=(0, 0, 0))
    draw.text((w * scale + 4, 4), "candidate", fill=(0, 0, 0))
    sheet.paste(original.resize((w * scale, h * scale), Image.Resampling.NEAREST), (0, label_h))
    sheet.paste(candidate.resize((w * scale, h * scale), Image.Resampling.NEAREST), (w * scale, label_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def params_label(params: CandidateParams) -> str:
    return (
        f"{params.candidate_id} {params.font_name} s{params.font_size} "
        f"op{params.opacity:.2f} bl{params.blur:.2f} st{params.stroke_opacity:.2f} "
        f"ink{params.ink_gain:.2f} ac{params.alpha_contrast:.2f} "
        f"core{params.core_ink_gain:.2f} cd{params.core_darken_strength:.2f} "
        f"pw{params.photo_warp:.2f} eb{params.edge_breakup:.2f} pn{params.photo_noise:.2f} q{params.jpeg_quality} "
        f"dx{params.text_dx} dy{params.text_dy}"
    )


def replace_candidate_id(params: CandidateParams, candidate_id: str) -> CandidateParams:
    data = asdict(params)
    data["candidate_id"] = candidate_id
    data["char_offsets"] = tuple(tuple(x) for x in data["char_offsets"])
    return CandidateParams(**data)


def dedupe_params(params_list: list[CandidateParams], limit: int) -> list[CandidateParams]:
    seen: set[tuple[Any, ...]] = set()
    result: list[CandidateParams] = []
    for params in params_list:
        key = (
            params.font_path,
            params.font_size,
            round(params.opacity, 3),
            round(params.blur, 3),
            round(params.stroke_opacity, 3),
            round(params.ink_gain, 3),
            round(params.alpha_contrast, 3),
            round(params.core_ink_gain, 3),
            round(params.core_darken_strength, 3),
            params.core_darken_threshold,
            params.core_darken_target_gray,
            params.text_dx,
            params.text_dy,
            params.char_offsets,
            params.mask_threshold,
            params.mask_dilate_iterations,
            round(params.photo_warp, 3),
            round(params.edge_breakup, 3),
            round(params.photo_noise, 3),
            params.jpeg_quality,
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(replace_candidate_id(params, f"c{len(result):03d}"))
        if len(result) >= limit:
            break
    return result


def mutate_params(params: CandidateParams, **updates: Any) -> CandidateParams:
    data = asdict(params)
    data.update(updates)
    data["opacity"] = round(float(data["opacity"]), 3)
    data["blur"] = round(max(0.0, float(data["blur"])), 3)
    data["stroke_opacity"] = round(max(0.0, min(1.0, float(data.get("stroke_opacity", 0.0)))), 3)
    data["ink_gain"] = round(max(0.0, min(1.0, float(data.get("ink_gain", 0.0)))), 3)
    data["alpha_contrast"] = round(max(0.0, min(2.0, float(data.get("alpha_contrast", 0.0)))), 3)
    data["core_ink_gain"] = round(max(0.0, min(1.0, float(data.get("core_ink_gain", 0.0)))), 3)
    data["core_darken_strength"] = round(max(0.0, min(1.0, float(data.get("core_darken_strength", 0.0)))), 3)
    data["core_darken_threshold"] = max(0, min(254, int(data.get("core_darken_threshold", 120))))
    data["core_darken_target_gray"] = max(0, min(120, int(data.get("core_darken_target_gray", 28))))
    data["photo_warp"] = round(max(0.0, min(1.0, float(data.get("photo_warp", 0.0)))), 3)
    data["edge_breakup"] = round(max(0.0, min(0.2, float(data.get("edge_breakup", 0.0)))), 3)
    data["photo_noise"] = round(max(0.0, min(0.35, float(data.get("photo_noise", 0.0)))), 3)
    data["jpeg_quality"] = max(0, min(99, int(data.get("jpeg_quality", 0))))
    data["font_size"] = max(6, int(data["font_size"]))
    data["char_offsets"] = tuple(tuple(x) for x in data["char_offsets"])
    return CandidateParams(**data)


def cjk_spacing_offset_seeds(current: CandidateParams) -> tuple[tuple[tuple[int, int], ...], ...]:
    if len(current.char_offsets or ()) != 2:
        return ()
    return (
        ((0, 0), (0, 0)),
        ((-1, 0), (1, 0)),
        ((0, 0), (-1, 0)),
        ((1, 0), (-1, 0)),
        ((3, 0), (-2, 1)),
        ((4, 0), (-2, 1)),
        ((5, 0), (-1, 0)),
        ((5, 0), (-1, 1)),
        ((6, 0), (0, 1)),
        ((4, 0), (-1, 1)),
        ((5, 0), (0, 1)),
        ((3, 0), (-1, 1)),
        ((6, 0), (-1, 1)),
    )


def generate_candidates(
    current: CandidateParams,
    *,
    font_candidates: list[tuple[str, str]],
    font_style_reference: dict[str, Any] | None = None,
    font_pool_size: int = 4,
    iteration: int,
    limit: int,
) -> list[CandidateParams]:
    params_list: list[CandidateParams] = [current]
    active_fonts = font_candidates[: max(1, font_pool_size)]
    if iteration == 0:
        best_sizes = {
            item["font_path"]: int(item["font_size"])
            for item in (font_style_reference or {}).get("ranked_fonts", [])
            if item.get("font_path") and item.get("font_size")
        }
        priority_grid = (
            (1.00, 0.22, 0.00, 0.08, 0.00, 0.52, (1,), (((3, 0), (-1, 1)), ((3, 0), (-2, 1)))),
            (1.00, 0.20, 0.00, 0.08, 0.00, 0.50, (1,), (((3, 0), (-2, 1)), ((3, 0), (-1, 1)))),
            (1.00, 0.22, 0.00, 0.05, 0.00, 0.52, (1,), (((3, 0), (-2, 1)), ((3, 0), (-1, 1)))),
            (1.00, 0.24, 0.00, 0.05, 0.00, 0.16, (1,), (((5, 0), (-1, 0)), ((4, 0), (-2, 1)), ((3, 0), (-2, 1)))),
            (1.00, 0.25, 0.00, 0.06, 0.00, 0.36, (0, 1), (((3, 0), (-1, 1)), ((3, 0), (-2, 1)), ((4, 0), (-2, 1)))),
            (1.00, 0.25, 0.00, 0.05, 0.00, 0.36, (0, 1), (((3, 0), (-1, 1)), ((3, 0), (-2, 1)), ((4, 0), (-2, 1)))),
            (1.00, 0.22, 0.00, 0.05, 0.00, 0.04, (1, 2), (((3, 0), (-2, 1)), ((3, -1), (-2, 1)), ((4, 0), (-2, 1)))),
            (1.00, 0.28, 0.00, 0.05, 0.02, 0.12, (1, 2), (((5, 0), (-1, 0)), ((4, 0), (-2, 1)), ((3, 0), (-2, 1)))),
            (1.00, 0.28, 0.00, 0.00, 0.04, 0.16, (1, 2), (((3, -1), (-2, 1)), ((3, 0), (-2, 1)), ((4, 0), (-2, 1)))),
            (1.00, 0.25, 0.00, 0.00, 0.04, 0.12, (1, 2), (((3, -1), (-2, 1)), ((3, 0), (-2, 1)), ((4, 0), (-2, 1)))),
            (1.00, 0.22, 0.00, 0.00, 0.02, 0.10, (1, 2), (((3, -1), (-2, 1)), ((3, 0), (-2, 1)), ((4, 0), (-2, 1)))),
            (1.00, 0.22, 0.00, 0.06, 0.00, 0.00, (1, 2), (((3, -1), (-2, 1)), ((3, 0), (-2, 1)), ((4, 0), (-2, 1)))),
            (1.00, 0.22, 0.00, 0.04, 0.00, 0.00, (1, 2), (((3, -1), (-2, 1)), ((3, 0), (-2, 1)), ((4, 0), (-2, 1)))),
            (1.00, 0.25, 0.00, 0.08, 0.01, 0.00, (1, 2), (((3, -1), (-2, 1)), ((3, 0), (-2, 1)), ((5, 0), (-1, 0)))),
            (1.00, 0.20, 0.00, 0.03, 0.00, 0.00, (1, 2), (((3, 0), (-2, 1)), ((4, 0), (-2, 1)), ((5, 0), (-1, 0)))),
            (1.00, 0.20, 0.00, 0.02, 0.00, 0.00, (1, 2), (((3, 0), (-2, 1)), ((4, 0), (-2, 1)), ((5, 0), (-1, 0)))),
            (1.00, 0.25, 0.00, 0.02, 0.02, 0.00, (1, 2), (((3, 0), (-2, 1)), ((4, 0), (-2, 1)), ((5, 0), (-1, 0)))),
            (1.00, 0.25, 0.00, 0.03, 0.02, 0.00, (1, 2), (((3, 0), (-2, 1)), ((4, 0), (-2, 1)), ((5, 0), (-1, 0)))),
            (0.82, 0.20, 0.00, 0.06, 0.00, 0.00, (1, 2), (((3, 0), (-2, 1)), ((4, 0), (-2, 1)), ((5, 0), (-1, 0)))),
            (0.82, 0.20, 0.00, 0.04, 0.00, 0.00, (1, 2), (((3, 0), (-2, 1)), ((4, 0), (-2, 1)), ((5, 0), (-1, 0)))),
            (0.82, 0.20, 0.00, 0.08, 0.00, 0.00, (1, 2), (((3, 0), (-2, 1)), ((4, 0), (-2, 1)), ((5, 0), (-1, 0)))),
            (1.00, 0.25, 0.00, 0.00, 0.00, 0.00, (1, 2), (((3, 0), (-2, 1)), ((4, 0), (-2, 1)), ((5, 0), (-1, 0)))),
            (1.00, 0.25, 0.00, 0.02, 0.00, 0.00, (1, 2), (((3, 0), (-2, 1)), ((4, 0), (-2, 1)), ((5, 0), (-1, 0)))),
            (1.00, 0.25, 0.00, 0.04, 0.00, 0.00, (1, 2), (((3, 0), (-2, 1)), ((4, 0), (-2, 1)), ((5, 0), (-1, 0)))),
            (0.76, 0.15, 0.10, 0.35, 0.00, 0.00, (1, 2), (((3, 0), (-2, 1)), ((4, 0), (-2, 1)), ((5, 0), (-1, 0)))),
            (0.78, 0.15, 0.08, 0.25, 0.00, 0.00, (1, 2), (((3, 0), (-2, 1)), ((4, 0), (-2, 1)), ((5, 0), (-1, 0)))),
            (0.80, 0.20, 0.06, 0.20, 0.00, 0.00, (1, 2), (((3, 0), (-2, 1)), ((4, 0), (-2, 1)), ((5, 0), (-1, 0)))),
            (0.72, 0.10, 0.18, 0.45, 0.00, 0.00, (1, 2), (((3, 0), (-2, 1)), ((4, 0), (-2, 1)), ((5, 0), (-1, 0)))),
            (0.76, 0.15, 0.14, 0.35, 0.00, 0.00, (1, 2), (((3, 0), (-2, 1)), ((4, 0), (-2, 1)), ((5, 0), (-1, 0)))),
            (0.82, 0.20, 0.00, 0.00, 0.00, 0.00, (1, 2), (((3, 0), (-2, 1)), ((4, 0), (-2, 1)), ((5, 0), (-1, 0)))),
            (0.78, 0.15, 0.00, 0.00, 0.00, 0.00, (2, 3), (((3, 0), (-2, 1)), ((4, 0), (-2, 1)))),
            (0.86, 0.25, 0.00, 0.00, 0.00, 0.00, (1, 2), (((5, 0), (-1, 0)), ((6, 0), (0, 0)), ((5, 0), (-1, 1)))),
            (0.86, 0.20, 0.00, 0.00, 0.00, 0.00, (1, 2), (((3, 0), (-2, 1)), ((4, 0), (-2, 1)))),
        )
        core_darken_grid = (
            (1.00, 0.22, 0.00, 0.05, 0.00, 0.60, 0.35, 130, 26, (1,), (((3, 0), (-2, 1)), ((3, 0), (-1, 1)))),
            (1.00, 0.22, 0.00, 0.05, 0.00, 0.52, 0.45, 140, 24, (1,), (((3, 0), (-2, 1)), ((3, 0), (-1, 1)))),
            (1.00, 0.18, 0.00, 0.04, 0.00, 0.48, 0.34, 130, 28, (1,), (((3, 0), (-2, 1)), ((3, 0), (-1, 1)))),
        )
        for font_name, font_path in active_fonts:
            for (
                opacity,
                blur,
                stroke_opacity,
                ink_gain,
                alpha_contrast,
                core_ink_gain,
                core_darken_strength,
                core_darken_threshold,
                core_darken_target_gray,
                size_deltas,
                offsets_list,
            ) in core_darken_grid:
                base_size = best_sizes.get(font_path, current.font_size)
                for size_delta in size_deltas:
                    for offsets in offsets_list:
                        params_list.append(
                            mutate_params(
                                current,
                                font_name=font_name,
                                font_path=font_path,
                                font_size=base_size + size_delta,
                                opacity=opacity,
                                blur=blur,
                                stroke_opacity=stroke_opacity,
                                ink_gain=ink_gain,
                                alpha_contrast=alpha_contrast,
                                core_ink_gain=core_ink_gain,
                                core_darken_strength=core_darken_strength,
                                core_darken_threshold=core_darken_threshold,
                                core_darken_target_gray=core_darken_target_gray,
                                char_offsets=offsets,
                            )
                        )
        for font_name, font_path in active_fonts:
            for (
                opacity,
                blur,
                stroke_opacity,
                ink_gain,
                alpha_contrast,
                core_ink_gain,
                size_deltas,
                offsets_list,
            ) in priority_grid:
                base_size = best_sizes.get(font_path, current.font_size)
                for size_delta in size_deltas:
                    for offsets in offsets_list:
                        params_list.append(
                            mutate_params(
                                current,
                                font_name=font_name,
                                font_path=font_path,
                                font_size=base_size + size_delta,
                                opacity=opacity,
                                blur=blur,
                                stroke_opacity=stroke_opacity,
                                ink_gain=ink_gain,
                                alpha_contrast=alpha_contrast,
                                core_ink_gain=core_ink_gain,
                                char_offsets=offsets,
                            )
                        )
        tuning_grid = (
            (current.opacity, current.blur),
            (0.86, 0.25),
            (0.88, 0.25),
            (0.90, 0.25),
            (0.82, 0.20),
            (0.78, 0.15),
            (0.78, 0.10),
            (1.00, 0.25),
            (0.98, 0.25),
            (0.96, 0.15),
            (1.00, 0.15),
            (0.92, 0.15),
            (0.92, 0.00),
            (0.96, 0.30),
            (0.88, 0.35),
            (0.86, 0.50),
        )
        for opacity, blur in (
            (0.82, 0.20),
            (0.78, 0.15),
            (0.86, 0.25),
            (0.86, 0.20),
            (0.88, 0.25),
            (0.90, 0.25),
            (1.00, 0.25),
        ):
            for size_delta in (0, 1):
                for offsets in cjk_spacing_offset_seeds(current):
                    for font_name, font_path in active_fonts:
                        base_size = best_sizes.get(font_path, current.font_size)
                        params_list.append(
                            mutate_params(
                                current,
                                font_name=font_name,
                                font_path=font_path,
                                font_size=base_size + size_delta,
                                opacity=opacity,
                                blur=blur,
                                char_offsets=offsets,
                            )
                        )
        for opacity, blur in tuning_grid:
            for font_name, font_path in active_fonts:
                base_size = best_sizes.get(font_path, current.font_size)
                for size_delta in (0, 1, -1, 2):
                    params_list.append(
                        mutate_params(
                            current,
                            font_name=font_name,
                            font_path=font_path,
                            font_size=base_size + size_delta,
                            opacity=opacity,
                            blur=blur,
                        )
                    )
    else:
        params_list.extend(
            [
                mutate_params(current, font_size=current.font_size - 1),
                mutate_params(current, font_size=current.font_size + 1),
                mutate_params(current, opacity=max(0.2, current.opacity - 0.04)),
                mutate_params(current, opacity=min(1.0, current.opacity + 0.04)),
                mutate_params(current, blur=max(0.0, current.blur - 0.15)),
                mutate_params(current, blur=current.blur + 0.15),
                mutate_params(current, stroke_opacity=max(0.0, current.stroke_opacity - 0.10)),
                mutate_params(current, stroke_opacity=min(1.0, current.stroke_opacity + 0.10)),
                mutate_params(current, ink_gain=max(0.0, current.ink_gain - 0.10)),
                mutate_params(current, ink_gain=min(1.0, current.ink_gain + 0.10)),
                mutate_params(current, alpha_contrast=max(0.0, current.alpha_contrast - 0.15)),
                mutate_params(current, alpha_contrast=min(2.0, current.alpha_contrast + 0.15)),
                mutate_params(current, core_ink_gain=max(0.0, current.core_ink_gain - 0.08)),
                mutate_params(current, core_ink_gain=min(1.0, current.core_ink_gain + 0.08)),
                mutate_params(current, core_darken_strength=max(0.0, current.core_darken_strength - 0.10)),
                mutate_params(current, core_darken_strength=min(1.0, current.core_darken_strength + 0.10)),
                mutate_params(
                    current,
                    ink_gain=min(1.0, current.ink_gain + 0.03),
                    core_ink_gain=min(1.0, current.core_ink_gain + 0.16),
                    core_darken_strength=min(1.0, current.core_darken_strength + 0.12),
                    blur=max(0.0, current.blur - 0.02),
                ),
                mutate_params(
                    current,
                    ink_gain=min(1.0, current.ink_gain + 0.03),
                    core_ink_gain=min(1.0, current.core_ink_gain + 0.24),
                    core_darken_strength=min(1.0, current.core_darken_strength + 0.18),
                    blur=max(0.0, current.blur - 0.04),
                ),
                mutate_params(
                    current,
                    opacity=min(1.0, current.opacity + 0.04),
                    blur=current.blur + 0.10,
                ),
                mutate_params(
                    current,
                    opacity=min(1.0, current.opacity + 0.04),
                    blur=current.blur + 0.15,
                ),
                mutate_params(current, text_dx=current.text_dx - 1),
                mutate_params(current, text_dx=current.text_dx + 1),
                mutate_params(current, text_dy=current.text_dy - 1),
                mutate_params(current, text_dy=current.text_dy + 1),
            ]
        )

    if iteration == 0 and len(params_list) > 1:
        dy_base_count = min(len(params_list), max(4, (limit // 5) + 2))
        dy_variants: list[CandidateParams] = []
        for base in params_list[:dy_base_count]:
            for dy in (-2, -1, 1):
                dy_variants.append(mutate_params(base, text_dy=base.text_dy + dy))
        params_list = params_list[:dy_base_count] + dy_variants + params_list[dy_base_count:]

    offsets = list(current.char_offsets)
    for idx in range(min(2, len(offsets))):
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            new_offsets = list(offsets)
            ox, oy = new_offsets[idx]
            new_offsets[idx] = (ox + dx, oy + dy)
            params_list.append(mutate_params(current, char_offsets=tuple(new_offsets)))
    return dedupe_params(params_list, limit)


def local_score(
    original: Image.Image,
    candidate: Image.Image,
    plan: RenderPlan,
    report: dict[str, Any],
    *,
    blur_score_weight: float = 160.0,
    blur_score_free_margin: float = 0.15,
) -> float:
    if not report.get("pass"):
        return 1_000_000.0
    strict_gate = report.get("strict_gate")
    if isinstance(strict_gate, dict) and not strict_gate.get("pass", True):
        return 100_000.0 + len(strict_gate.get("issues", [])) * 1_000.0
    source_count = len([ch for ch in (plan.source_text or "") if not ch.isspace()])
    target_count = len([ch for ch in plan.target_text if not ch.isspace()])
    shorter_replacement = bool(source_count and target_count and target_count < source_count)
    count_ratio = float(target_count / source_count) if source_count and target_count else 1.0
    old_bbox = dark_bbox(original, plan.target_roi) or {}
    new_bbox = dark_bbox(candidate, plan.target_roi) or {}
    old_gray = gray_stats(original, plan.target_roi) or {}
    new_gray = gray_stats(candidate, plan.target_roi) or {}
    score = 0.0
    bbox_weights = (("height", 6.0), ("width", 1.2), ("cx", 2.0), ("cy", 2.0))
    if shorter_replacement:
        bbox_weights = (("height", 8.0), ("cy", 3.0))
    for key, weight in bbox_weights:
        if key in old_bbox and key in new_bbox:
            score += abs(float(old_bbox[key]) - float(new_bbox[key])) * weight
    if "dark_pixels" in old_gray and "dark_pixels" in new_gray:
        target_dark_pixels = float(old_gray["dark_pixels"]) * (count_ratio if shorter_replacement else 1.0)
        score += abs(float(new_gray["dark_pixels"]) - target_dark_pixels) * 0.08
    if "mean_gray" in old_gray and "mean_gray" in new_gray:
        score += abs(old_gray["mean_gray"] - new_gray["mean_gray"]) * 0.25
    strict_metrics = report.get("strict_visual_metrics", {}).get("thresholds", {})
    core = strict_metrics.get("120", {})
    edge = strict_metrics.get("165", {})
    if core.get("mean_gray_delta") is not None:
        # A very small darker core reads clearer and closer to low-resolution
        # printed text than exact mean-gray equality, provided the hard gates
        # still keep the dark-pixel area bounded.
        preferred_core_delta = 0.0 if shorter_replacement else PREFERRED_CORE_MEAN_GRAY_DELTA
        score += abs(float(core["mean_gray_delta"]) - preferred_core_delta) * 8.0
    if edge.get("mean_gray_delta") is not None:
        score += abs(float(edge["mean_gray_delta"])) * 0.8
    bands = report.get("strict_visual_metrics", {}).get("bands", {})
    if isinstance(bands, dict):
        if shorter_replacement:
            for prefix, weight in (
                ("lt55", 1.2),
                ("lt70", 0.8),
                ("lt90", 0.45),
                ("lt120", 0.25),
                ("120_165", 0.12),
            ):
                old_key = f"old_{prefix}_pixels" if prefix.startswith("lt") else f"old_{prefix}_pixels"
                new_key = f"new_{prefix}_pixels" if prefix.startswith("lt") else f"new_{prefix}_pixels"
                old_value = bands.get(old_key)
                new_value = bands.get(new_key)
                if old_value is None or new_value is None:
                    continue
                target_value = float(old_value) * count_ratio
                score += abs(float(new_value) - target_value) * weight
            new_lt55_share = bands.get("new_lt55_share_of_lt165")
            old_lt55_share = bands.get("old_lt55_share_of_lt165")
            if new_lt55_share is not None and old_lt55_share is not None:
                score += max(0.0, float(new_lt55_share) - float(old_lt55_share) - 0.03) * 1600.0
            return score
        black_core_delta = bands.get("lt70_delta")
        darkest_core_delta = bands.get("lt55_delta")
        deep_gray_delta = bands.get("band_55_70_delta")
        mid_gray_delta = bands.get("band_70_90_delta")
        dark_gray_delta = bands.get("band_70_120_delta")
        gray_band_delta = bands.get("band_120_165_delta")
        core_delta = bands.get("lt90_delta")
        if black_core_delta is not None:
            # <70 alone can be misleading because 55-70 deep gray can masquerade
            # as ink. Prefer a larger dark core only when the <55 true-black
            # layer also improves.
            score += abs(float(black_core_delta) - 62.0) * 0.7
        if darkest_core_delta is not None:
            score += abs(float(darkest_core_delta) - 46.0) * 2.0
            score += max(0.0, 28.0 - float(darkest_core_delta)) * 9.0
            score += max(0.0, float(darkest_core_delta) - 68.0) * 9.0
        if deep_gray_delta is not None:
            score += abs(float(deep_gray_delta) - 6.0) * 1.5
            score += max(0.0, float(deep_gray_delta) - 18.0) * 8.0
        if mid_gray_delta is not None:
            # 70-90 pixels read as gray stroke interiors at this scale. The
            # current target text has more strokes than the source, so matching
            # this band exactly still looks gray; a negative delta is expected
            # once the centers are properly black.
            score += abs(float(mid_gray_delta) + 46.0) * 0.8
            score += max(0.0, float(mid_gray_delta) - 2.0) * 9.0
            score += max(0.0, -64.0 - float(mid_gray_delta)) * 2.0
        if dark_gray_delta is not None:
            score += abs(float(dark_gray_delta) + 70.0) * 0.45
        if gray_band_delta is not None:
            # Penalize excess semi-transparent gray edge more than modest
            # under-coverage; too much 120-165 gray reads as a hazy outline.
            score += abs(float(gray_band_delta) + 32.0) * 0.6
            score += max(0.0, float(gray_band_delta)) * 4.0
            score += max(0.0, -70.0 - float(gray_band_delta)) * 2.0
        if core_delta is not None:
            score += abs(float(core_delta) - 40.0) * 0.55
    char_bands = report.get("char_gray_band_metrics")
    if isinstance(char_bands, dict) and char_bands.get("enabled"):
        for item in char_bands.get("per_char", []):
            delta = item.get("delta") or {}
            old = item.get("old") or {}
            lt55_delta = delta.get("lt55")
            lt40_delta = delta.get("lt40")
            band_55_70_delta = delta.get("band_55_70")
            band_70_90_delta = delta.get("band_70_90")
            band_90_120_delta = delta.get("band_90_120")
            lt165_delta = delta.get("lt165")
            # For complex replacement glyphs, exact old-character black-pixel
            # equality leaves the new glyph gray. Use old slots as a floor, then
            # penalize gray interior growth.
            if lt40_delta is not None:
                score += max(0.0, -float(lt40_delta)) * 5.0
            if lt55_delta is not None:
                glyph_area_delta = float(lt165_delta or 0.0)
                target_lt55_delta = 32.0 + min(8.0, max(0.0, glyph_area_delta) * 0.18)
                score += max(0.0, 24.0 - float(lt55_delta)) * 20.0
                score += abs(float(lt55_delta) - target_lt55_delta) * 1.6
                score += max(0.0, float(lt55_delta) - 64.0) * 7.0
            if band_55_70_delta is not None:
                score += max(0.0, float(band_55_70_delta) - 10.0) * 5.0
            if band_70_90_delta is not None:
                score += max(0.0, float(band_70_90_delta) - 2.0) * 8.0
            if band_90_120_delta is not None:
                score += max(0.0, float(band_90_120_delta) - 6.0) * 5.0
            if old.get("lt165") and lt165_delta is not None:
                score += max(0.0, -float(lt165_delta) - float(old["lt165"]) * 0.22) * 2.0
    for threshold_name, values in strict_metrics.items():
        ratio = values.get("dark_pixel_ratio")
        if ratio is not None:
            ratio_value = float(ratio)
            if str(threshold_name) == "165":
                # The high threshold tracks the soft gray edge around scanned text.
                # Under-coverage here reads as thin text even when the core is dark.
                score += abs(ratio_value - 1.0) * 90.0
                if ratio_value < 1.0:
                    score += (1.0 - ratio_value) * 260.0
            else:
                score += abs(ratio_value - 1.0) * 20.0
    alignment = report.get("char_alignment_metrics")
    if isinstance(alignment, dict) and alignment.get("enabled"):
        distance_delta = alignment.get("center_distance_delta")
        if distance_delta is not None:
            score += abs(float(distance_delta)) * 8.0
        for item in alignment.get("per_char", []):
            center_dx = item.get("center_dx")
            center_dy = item.get("center_dy")
            if center_dx is not None:
                score += abs(float(center_dx)) * 8.0
            if center_dy is not None:
                score += abs(float(center_dy)) * 1.5
    font_style = report.get("font_style_gate")
    if isinstance(font_style, dict) and font_style.get("enabled"):
        ratio = (font_style.get("candidate") or {}).get("score_ratio_to_best")
        if ratio is not None:
            score += max(0.0, float(ratio) - 1.0) * 180.0
        if not font_style.get("pass", True):
            score += 500.0
    params = report.get("params")
    if isinstance(params, dict) and params.get("blur") is not None:
        score += max(0.0, float(params["blur"]) - blur_score_free_margin) * blur_score_weight
        score += max(0.0, float(params.get("ink_gain", 0.0))) * 10.0
        score -= min(max(0.0, float(params.get("core_ink_gain", 0.0))), 0.08) * 20.0
    return score


def font_style_score_ratio_from_report(report: dict[str, Any] | None) -> float | None:
    if not isinstance(report, dict):
        return None
    font_style = report.get("font_style_gate")
    if not isinstance(font_style, dict):
        return None
    ratio = (font_style.get("candidate") or {}).get("score_ratio_to_best")
    if ratio is None:
        return None
    try:
        return float(ratio)
    except (TypeError, ValueError):
        return None


def report_strict_pass(report: dict[str, Any]) -> bool:
    return bool(report.get("pass")) and bool(report.get("strict_gate", {}).get("pass", True))


def find_best_candidate_from_model(
    model_json: dict[str, Any],
    candidates: list[tuple[CandidateParams, Image.Image, dict[str, Any]]],
) -> CandidateParams | None:
    raw = str(model_json.get("best_candidate", ""))
    match = re.search(r"c\d{3}", raw)
    if match:
        wanted = match.group(0)
        for params, _, _ in candidates:
            if params.candidate_id == wanted:
                return params
    for params, _, _ in candidates:
        if params.candidate_id in raw or params_label(params) in raw:
            return params
    return None


def bounded_int(value: Any, low: int, high: int) -> int:
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return 0
    return max(low, min(high, number))


def bounded_float(value: Any, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(low, min(high, number))


def apply_suggested_patch(params: CandidateParams, patch: dict[str, Any] | None) -> CandidateParams:
    if not patch:
        return params
    font_size = params.font_size + bounded_int(patch.get("font_size_delta", 0), -1, 1)
    opacity = max(0.2, min(1.0, params.opacity + bounded_float(patch.get("opacity_delta", 0), -0.06, 0.06)))
    blur = max(0.0, min(2.0, params.blur + bounded_float(patch.get("blur_delta", 0), -0.2, 0.2)))
    stroke_opacity = max(0.0, min(1.0, params.stroke_opacity + bounded_float(patch.get("stroke_opacity_delta", 0), -0.15, 0.15)))
    ink_gain = max(0.0, min(1.0, params.ink_gain + bounded_float(patch.get("ink_gain_delta", 0), -0.15, 0.15)))
    alpha_contrast = max(0.0, min(2.0, params.alpha_contrast + bounded_float(patch.get("alpha_contrast_delta", 0), -0.25, 0.25)))
    core_ink_gain = max(0.0, min(1.0, params.core_ink_gain + bounded_float(patch.get("core_ink_gain_delta", 0), -0.15, 0.15)))
    core_darken_strength = max(0.0, min(1.0, params.core_darken_strength + bounded_float(patch.get("core_darken_strength_delta", 0), -0.15, 0.15)))
    core_darken_threshold = max(0, min(254, params.core_darken_threshold + bounded_int(patch.get("core_darken_threshold_delta", 0), -20, 20)))
    core_darken_target_gray = max(0, min(120, params.core_darken_target_gray + bounded_int(patch.get("core_darken_target_gray_delta", 0), -20, 20)))
    photo_warp = max(0.0, min(1.0, params.photo_warp + bounded_float(patch.get("photo_warp_delta", 0), -0.18, 0.18)))
    edge_breakup = max(0.0, min(0.2, params.edge_breakup + bounded_float(patch.get("edge_breakup_delta", 0), -0.04, 0.04)))
    photo_noise = max(0.0, min(0.35, params.photo_noise + bounded_float(patch.get("photo_noise_delta", 0), -0.08, 0.08)))
    jpeg_quality = max(0, min(99, params.jpeg_quality + bounded_int(patch.get("jpeg_quality_delta", 0), -12, 12)))
    text_dx = params.text_dx + bounded_int(patch.get("text_dx_delta", 0), -1, 1)
    text_dy = params.text_dy + bounded_int(patch.get("text_dy_delta", 0), -1, 1)
    mask_threshold = max(80, min(230, params.mask_threshold + bounded_int(patch.get("mask_threshold_delta", 0), -20, 20)))
    mask_dilate_iterations = max(
        0,
        min(5, params.mask_dilate_iterations + bounded_int(patch.get("mask_dilate_iterations_delta", 0), -1, 1)),
    )
    inpaint_radius = max(1, min(8, params.inpaint_radius + bounded_int(patch.get("inpaint_radius_delta", 0), -1, 1)))
    offsets = list(params.char_offsets)
    char_offsets_delta = patch.get("char_offsets_delta")
    if isinstance(char_offsets_delta, list):
        while len(offsets) < len(char_offsets_delta):
            offsets.append((0, 0))
        for idx, delta in enumerate(char_offsets_delta):
            if idx >= len(offsets):
                break
            if isinstance(delta, dict):
                dx = bounded_int(delta.get("dx", 0), -2, 2)
                dy = bounded_int(delta.get("dy", 0), -2, 2)
            elif isinstance(delta, (list, tuple)) and len(delta) >= 2:
                dx = bounded_int(delta[0], -2, 2)
                dy = bounded_int(delta[1], -2, 2)
            else:
                continue
            offsets[idx] = (offsets[idx][0] + dx, offsets[idx][1] + dy)
    return mutate_params(
        params,
        font_size=font_size,
        opacity=opacity,
        blur=blur,
        stroke_opacity=stroke_opacity,
        ink_gain=ink_gain,
        alpha_contrast=alpha_contrast,
        core_ink_gain=core_ink_gain,
        core_darken_strength=core_darken_strength,
        core_darken_threshold=core_darken_threshold,
        core_darken_target_gray=core_darken_target_gray,
        photo_warp=photo_warp,
        edge_breakup=edge_breakup,
        photo_noise=photo_noise,
        jpeg_quality=jpeg_quality,
        text_dx=text_dx,
        text_dy=text_dy,
        char_offsets=tuple(offsets),
        mask_threshold=mask_threshold,
        mask_dilate_iterations=mask_dilate_iterations,
        inpaint_radius=inpaint_radius,
    )


def paste_crop(full_image: Image.Image, crop: Image.Image, split_box: tuple[int, int, int, int]) -> Image.Image:
    result = full_image.copy().convert("RGB")
    x1, y1, _, _ = split_box
    result.paste(crop.convert("RGB"), (x1, y1))
    return result


def vision_task_context(plan: RenderPlan) -> str:
    source_chars = text_chars(plan.source_text)
    target_chars = text_chars(plan.target_text)
    protected_texts = [text for text in plan.protected_texts if text]
    return (
        "\n\n动态任务上下文：\n"
        f"- field_key: {plan.field_key or ''}\n"
        f"- field_label_text: {plan.field_label_text or ''}\n"
        f"- field_separator_text: {plan.field_separator_text or ''}\n"
        f"- protected_texts: {protected_texts}\n"
        f"- source_text: {plan.source_text or ''}\n"
        f"- target_text: {plan.target_text}\n"
        f"- source_chars: {source_chars}\n"
        f"- target_chars: {target_chars}\n"
        f"- search_roi: {list(plan.search_roi)}\n"
        f"- target_roi: {list(plan.target_roi)}\n"
        f"- slot_boxes: {[asdict(slot) for slot in plan.slot_boxes]}\n"
        f"- protected_boxes: {[list(box) for box in plan.protected_boxes]}\n"
        f"- text_angle_degrees: {round(float(plan.text_angle_degrees), 3)}\n"
        "- source_text 为空时，按 target_roi 内旧文字暗色组件作为旧槽位参照；不要补全成固定字段值。\n"
        "- source_text 和 target_text 字数不同时，只能使用旧文字区域及合理空白，不能覆盖前后不应修改文字。\n"
        "- field_label_text、field_separator_text、protected_texts 和 protected_boxes 表示本次任务实际识别到或由指令提供的受保护上下文；这些内容必须保持不变。\n"
    )


def build_task_from_metadata(
    metadata_path: Path,
    *,
    target_text: str | None,
    source_text: str | None,
    manual_roi: str | None,
    mask_threshold: int,
    style_reference_text: str | None,
) -> tuple[Image.Image, Image.Image, tuple[int, int, int, int] | None, RenderPlan, dict[str, Any]]:
    metadata = read_json(metadata_path)
    fixture_dir = metadata_path.parent
    full_path = fixture_dir / metadata["image"]
    full_image = Image.open(full_path).convert("RGB")
    split_box = parse_box(metadata["split_rectangles"][0]) if metadata.get("split_rectangles") else None
    crop = full_image.crop(split_box) if split_box else full_image.copy()

    edit = metadata.get("edit", {})
    text = target_text or edit.get("text")
    if not text:
        raise ValueError("target text is required via --text or metadata edit.text")
    old_text = source_text or edit.get("source_text")
    search_roi = parse_box(manual_roi or edit.get("roi"))
    protected_box = parse_box(edit["ref_text_box"]) if edit.get("ref_text_box") else None
    if manual_roi:
        target_roi = parse_box(manual_roi)
        slot_boxes = tuple(dark_runs(crop, target_roi, threshold=mask_threshold))
        all_runs = list(slot_boxes)
    else:
        target_roi, slot_boxes, all_runs = infer_target_roi(
            crop,
            search_roi=search_roi,
            protected_box=protected_box,
            target_text=text,
            threshold=mask_threshold,
        )

    protected_boxes: tuple[tuple[int, int, int, int], ...] = ()
    if protected_box:
        protected_boxes = (protected_box,)

    plan = RenderPlan(
        target_text=text,
        source_text=old_text,
        search_roi=search_roi,
        target_roi=target_roi,
        slot_boxes=slot_boxes,
        protected_boxes=protected_boxes,
        source_reference_box=target_roi,
        style_reference_box=protected_box,
        style_reference_text=style_reference_text or edit.get("style_reference_text") or None,
        draw_mode="auto",
        field_key=edit.get("field_key") or edit.get("field"),
        field_label_text=edit.get("field_label_text"),
        field_separator_text=edit.get("field_separator_text"),
        protected_texts=tuple(str(item) for item in (edit.get("protected_texts") or []) if str(item)),
    )
    context = {
        "metadata_path": str(metadata_path),
        "source_image": str(full_path),
        "split_box": list(split_box) if split_box else None,
        "search_roi": list(search_roi),
        "target_roi": list(target_roi),
        "source_text": plan.source_text,
        "source_reference_box": list(plan.source_reference_box) if plan.source_reference_box else None,
        "protected_boxes": [list(x) for x in protected_boxes],
        "style_reference_box": list(protected_box) if protected_box else None,
        "style_reference_text": plan.style_reference_text,
        "all_dark_runs": [asdict(x) for x in all_runs],
        "selected_slot_boxes": [asdict(x) for x in slot_boxes],
        "metadata": metadata,
    }
    return full_image, crop, split_box, plan, context


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    progress = ProgressTracker(run_dir, enabled=args.progress)
    pipeline_profile = stage_profile(str(getattr(args, "profile", None) or "photo_scan")).id
    progress.emit(
        "run_started",
        "created run directory",
        run_dir=str(run_dir),
        pipeline_profile=pipeline_profile,
    )

    full_image, crop, split_box, plan, context = build_task_from_metadata(
        args.metadata,
        target_text=args.text,
        source_text=args.source_text,
        manual_roi=args.roi,
        mask_threshold=args.mask_threshold,
        style_reference_text=args.style_reference_text,
    )
    crop.save(run_dir / "original_crop.png")
    if split_box:
        full_image.save(run_dir / "original_full.png")
    progress.emit(
        "task_loaded",
        "loaded source image and inferred ROI",
        target_roi=list(plan.target_roi),
        target_text=plan.target_text,
    )

    font_candidates = find_font_candidates(
        args.font_path,
        font_source=args.font_source,
        scan_dirs=args.font_scan_dirs,
        max_scanned_fonts=args.max_scanned_fonts,
    )
    required_font_text = "".join(
        part or ""
        for part in (plan.target_text, plan.source_text, plan.style_reference_text)
    )
    font_candidates, rejected_font_candidates = filter_fonts_by_required_text(
        font_candidates,
        required_font_text,
    )
    font_style_reference = build_font_style_reference(
        crop,
        plan,
        font_candidates,
        min_size=args.font_style_min_size,
        max_size=args.font_style_max_size,
        prefer_serif_categories=args.prefer_serif_fonts,
    )
    if args.font_ranking == "style":
        font_candidates = rank_fonts_by_style_reference(font_candidates, font_style_reference)
    progress.emit(
        "fonts_resolved",
        "resolved usable font candidates",
        font_count=len(font_candidates),
        rejected_font_count=len(rejected_font_candidates),
        first_font=font_candidates[0][0] if font_candidates else None,
        font_source=args.font_source,
        font_ranking=args.font_ranking,
        font_candidate_pool_size=args.font_candidate_pool_size,
    )
    if font_style_reference.get("enabled"):
        best_style = font_style_reference.get("best_available") or {}
        progress.emit(
            "font_style_reference_ready",
            "computed old-text ROI font style reference",
            reference_text=font_style_reference.get("reference_text"),
            primary_reference=font_style_reference.get("primary_reference_kind"),
            best_font=best_style.get("font_name"),
            best_score=round(float(best_style.get("score", 0.0)), 3),
        )
    initial_font_name, initial_font_path = font_candidates[0]
    metadata_font_size = int(context["metadata"].get("edit", {}).get("font_size", args.font_size))
    initial_size = args.font_size or metadata_font_size
    current = CandidateParams(
        candidate_id="current",
        font_name=initial_font_name,
        font_path=initial_font_path,
        font_size=initial_size,
        opacity=args.opacity,
        blur=args.blur,
        stroke_opacity=args.stroke_opacity,
        ink_gain=args.ink_gain,
        alpha_contrast=args.alpha_contrast,
        core_ink_gain=args.core_ink_gain,
        core_darken_strength=args.core_darken_strength,
        core_darken_threshold=args.core_darken_threshold,
        core_darken_target_gray=args.core_darken_target_gray,
        char_offsets=default_char_offsets(plan.target_text),
        mask_threshold=args.mask_threshold,
        mask_dilate_iterations=args.mask_dilate_iterations,
    )
    metadata_max_iterations = int(context["metadata"].get("judge", {}).get("max_iterations", DEFAULT_MAX_ITERATIONS))
    max_iterations = args.max_iterations if args.max_iterations is not None else DEFAULT_MAX_ITERATIONS
    progress.emit(
        "iteration_limit_set",
        "configured iteration limit",
        max_iterations=max_iterations,
        metadata_max_iterations=metadata_max_iterations,
    )

    require_prompts()

    vision_client: VisionClient | None = None
    vision_errors: list[str] = []
    if args.vision != "off":
        try:
            vision_client = VisionClient(args.env)
            progress.emit(
                "vision_ready",
                "vision client initialized",
                model=vision_client.model,
                base_url=vision_client.base_url,
            )
        except Exception as exc:
            if args.vision == "on":
                raise
            vision_errors.append(str(exc))
            progress.emit("vision_unavailable", "vision client unavailable", error=str(exc))
    else:
        progress.emit("vision_disabled", "running without vision model")

    master_prompt = load_prompt("master_prompt.txt")
    candidate_prompt_template = load_prompt("candidate_rank_prompt.txt")
    tuning_prompt_template = load_prompt("tuning_prompt.txt")
    final_prompt_template = load_prompt("final_acceptance_prompt.txt")

    history: list[dict[str, Any]] = []
    final_crop: Image.Image | None = None
    final_params = current

    for iteration in range(max_iterations):
        iter_dir = run_dir / f"iteration_{iteration:02d}"
        candidates_dir = iter_dir / "candidates"
        candidates_dir.mkdir(parents=True, exist_ok=True)
        progress.emit(
            "iteration_started",
            "starting iteration",
            iteration=iteration + 1,
            max_iterations=max_iterations,
        )

        candidate_params = generate_candidates(
            current,
            font_candidates=font_candidates,
            font_style_reference=font_style_reference,
            font_pool_size=args.font_candidate_pool_size,
            iteration=iteration,
            limit=args.max_candidates,
        )
        progress.emit(
            "candidates_generated",
            "generated candidate parameter set",
            iteration=iteration + 1,
            candidate_count=len(candidate_params),
        )
        rendered: list[tuple[CandidateParams, Image.Image, dict[str, Any]]] = []
        sheet_items: list[tuple[str, Image.Image]] = []
        hard_reports: dict[str, Any] = {}
        for params in candidate_params:
            image = render_candidate(crop, plan, params)
            report = hard_check(crop, image, plan.target_roi, plan.protected_boxes)
            metrics = {
                "old_dark_bbox": dark_bbox(crop, plan.target_roi),
                "candidate_dark_bbox": dark_bbox(image, plan.target_roi),
                "old_gray_stats": gray_stats(crop, plan.target_roi),
                "candidate_gray_stats": gray_stats(image, plan.target_roi),
            }
            strict_metrics = strict_visual_metrics(crop, image, plan.target_roi)
            strict_issues = strict_gate_issues(
                strict_metrics,
                max_dark_pixel_ratio=args.max_dark_pixel_ratio,
                min_dark_pixel_ratio=args.min_dark_pixel_ratio,
                max_core_mean_gray_delta=args.max_core_mean_gray_delta,
                max_edge_mean_gray_delta=args.max_edge_mean_gray_delta,
                max_core_lighten_delta=args.max_core_lighten_delta,
                max_edge_lighten_delta=args.max_edge_lighten_delta,
            )
            alignment_metrics, alignment_issues = char_alignment_gate(
                crop.size,
                plan,
                params,
                max_char_center_dx=args.max_char_center_dx,
                max_char_center_distance_delta=args.max_char_center_distance_delta,
            )
            font_style_report = font_style_gate(
                crop,
                plan,
                params,
                font_style_reference,
                max_score_ratio=args.max_font_style_score_ratio,
            )
            report["params"] = asdict(params)
            report["metrics"] = metrics
            report["strict_visual_metrics"] = strict_metrics
            report["char_gray_band_metrics"] = char_gray_band_metrics(crop, image, plan)
            report["char_alignment_metrics"] = alignment_metrics
            report["font_style_gate"] = font_style_report
            report["strict_gate"] = {
                "max_dark_pixel_ratio": args.max_dark_pixel_ratio,
                "min_dark_pixel_ratio": args.min_dark_pixel_ratio,
                "max_core_mean_gray_delta": args.max_core_mean_gray_delta,
                "max_edge_mean_gray_delta": args.max_edge_mean_gray_delta,
                "max_core_lighten_delta": args.max_core_lighten_delta,
                "max_edge_lighten_delta": args.max_edge_lighten_delta,
                "max_char_center_dx": args.max_char_center_dx,
                "max_char_center_distance_delta": args.max_char_center_distance_delta,
                "max_font_style_score_ratio": args.max_font_style_score_ratio,
                "pass": not strict_issues
                and not alignment_issues
                and bool(font_style_report.get("pass", True)),
                "issues": strict_issues + alignment_issues + list(font_style_report.get("issues", [])),
            }
            attach_report_stage_context(report, pipeline_profile)
            image_path = candidates_dir / f"{params.candidate_id}.png"
            image.save(image_path)
            rendered.append((params, image, report))
            sheet_items.append((params_label(params), image))
            hard_reports[params.candidate_id] = {"params": asdict(params), "hard_check": report}

        hard_report_path = iter_dir / "hard_check_report.json"
        write_json(
            hard_report_path,
            attach_stage_context_to_rank_report(
                {"candidates": hard_reports},
                pipeline_profile=pipeline_profile,
            ),
        )
        contact_sheet_path = iter_dir / "contact_sheet.png"
        make_contact_sheet(sheet_items, contact_sheet_path, scale=args.sheet_scale, cols=args.sheet_cols)
        progress.emit(
            "candidates_rendered",
            "rendered candidates and wrote hard-check report",
            iteration=iteration + 1,
            contact_sheet=str(contact_sheet_path),
        )

        scored_rendered = sorted(
            rendered,
            key=lambda item: local_score(
                crop,
                item[1],
                plan,
                item[2],
                blur_score_weight=args.blur_score_weight,
                blur_score_free_margin=args.blur_score_free_margin,
            ),
        )
        local_best = scored_rendered[0]
        local_best_score = local_score(
            crop,
            local_best[1],
            plan,
            local_best[2],
            blur_score_weight=args.blur_score_weight,
            blur_score_free_margin=args.blur_score_free_margin,
        )
        local_best_font_style_ratio = font_style_score_ratio_from_report(local_best[2])
        chosen_params = local_best[0]
        progress.emit(
            "local_best_selected",
            "selected local metric fallback candidate",
            iteration=iteration + 1,
            candidate_id=chosen_params.candidate_id,
            font=chosen_params.font_name,
        )
        candidate_rank_json: dict[str, Any] | None = None
        tuning_json: dict[str, Any] | None = None
        vision_contact_sheet_path = contact_sheet_path
        vision_hard_reports = hard_reports
        vision_candidate_count = len(rendered)

        if vision_client:
            requested_vision_limit = int(getattr(args, "vision_candidate_limit", 0) or 0)
            vision_limit = normalize_vision_candidate_limit(
                requested_vision_limit,
                len(scored_rendered),
            )
            if vision_limit < len(scored_rendered):
                vision_rendered = scored_rendered[:vision_limit]
                vision_hard_reports = {
                    item[0].candidate_id: hard_reports[item[0].candidate_id]
                    for item in vision_rendered
                }
                vision_contact_sheet_path = iter_dir / "vision_contact_sheet.png"
                make_contact_sheet(
                    [(params_label(params), image) for params, image, _report in vision_rendered],
                    vision_contact_sheet_path,
                    scale=args.sheet_scale,
                    cols=args.sheet_cols,
                )
                vision_candidate_count = len(vision_rendered)
                progress.emit(
                    "vision_candidate_subset_ready",
                    "prepared locally ranked subset for vision candidate review",
                    iteration=iteration + 1,
                    candidate_count=vision_candidate_count,
                    total_candidate_count=len(rendered),
                    requested_vision_candidate_limit=requested_vision_limit,
                    vision_candidate_limit=vision_limit,
                    contact_sheet=str(vision_contact_sheet_path),
                )
            vision_hard_payload = vision_candidate_request_payload(
                {"candidates": vision_hard_reports},
                pipeline_profile=pipeline_profile,
                requested_vision_candidate_limit=requested_vision_limit,
                total_candidate_count=len(rendered),
            )
            write_json(iter_dir / "vision_candidate_request.json", vision_hard_payload)
            prompt = candidate_prompt_template.replace(
                "{hard_check_report}",
                json.dumps(vision_hard_payload, ensure_ascii=False, indent=2),
            )
            prompt += vision_task_context(plan)
            if args.acceptance_mode == "strict":
                prompt += STRICT_ACCEPTANCE_APPENDIX
            progress.emit(
                "vision_candidate_rank_started",
                "requesting candidate ranking",
                iteration=iteration + 1,
                candidate_count=vision_candidate_count,
            )
            try:
                candidate_rank_json = vision_client.call_json(
                    system_prompt=master_prompt,
                    user_prompt=prompt,
                    image_paths=[run_dir / "original_crop.png", vision_contact_sheet_path],
                    prompt_name="candidate_rank_prompt.txt",
                    audit_path=iter_dir / "visual_eval_candidate_rank_prompt_audit.json",
                )
                candidate_rank_json["local_stage_context"] = {
                    "pipeline_profile": pipeline_profile,
                    "stage_context_by_candidate": vision_hard_payload.get("stage_context_by_candidate"),
                    "stage_filter_contract": vision_hard_payload.get("stage_filter_contract"),
                }
                write_json(iter_dir / "visual_eval_candidate_rank.json", candidate_rank_json)
                model_best = find_best_candidate_from_model(candidate_rank_json, rendered)
                if model_best:
                    model_tuple = next(
                        (item for item in rendered if item[0].candidate_id == model_best.candidate_id),
                        None,
                    )
                    model_report = model_tuple[2] if model_tuple else None
                    model_score = (
                        local_score(
                            crop,
                            model_tuple[1],
                            plan,
                            model_report,
                            blur_score_weight=args.blur_score_weight,
                            blur_score_free_margin=args.blur_score_free_margin,
                        )
                        if model_tuple and model_report
                        else float("inf")
                    )
                    model_score_allowed = (
                        args.acceptance_mode != "strict"
                        or model_score <= local_best_score + args.max_model_local_score_delta
                    )
                    model_font_style_ratio = font_style_score_ratio_from_report(model_report)
                    model_font_style_allowed = (
                        args.acceptance_mode != "strict"
                        or local_best_font_style_ratio is None
                        or model_font_style_ratio is None
                        or model_font_style_ratio
                        <= local_best_font_style_ratio + args.max_model_font_style_ratio_delta
                    )
                    if (
                        args.acceptance_mode != "strict"
                        or (
                            model_report
                            and report_strict_pass(model_report)
                            and model_score_allowed
                            and model_font_style_allowed
                        )
                    ):
                        chosen_params = model_best
                    else:
                        progress.emit(
                            "vision_candidate_rejected",
                            "rejected model-selected candidate outside strict metric window",
                            iteration=iteration + 1,
                            candidate_id=model_best.candidate_id,
                            font=model_best.font_name,
                            model_score=round(float(model_score), 3) if model_score != float("inf") else "inf",
                            local_best_score=round(float(local_best_score), 3),
                            model_font_style_ratio=(
                                round(model_font_style_ratio, 4)
                                if model_font_style_ratio is not None
                                else None
                            ),
                            local_best_font_style_ratio=(
                                round(local_best_font_style_ratio, 4)
                                if local_best_font_style_ratio is not None
                                else None
                            ),
                            max_model_font_style_ratio_delta=args.max_model_font_style_ratio_delta,
                        )
                rank_patched_params = apply_suggested_patch(
                    chosen_params,
                    candidate_rank_json.get("suggested_patch"),
                )
                if rank_patched_params != chosen_params:
                    rank_patched_image = render_candidate(crop, plan, rank_patched_params)
                    rank_patched_report = hard_check(
                        crop,
                        rank_patched_image,
                        plan.target_roi,
                        plan.protected_boxes,
                    )
                    rank_patched_metrics = strict_visual_metrics(
                        crop,
                        rank_patched_image,
                        plan.target_roi,
                    )
                    rank_patched_issues = strict_gate_issues(
                        rank_patched_metrics,
                        max_dark_pixel_ratio=args.max_dark_pixel_ratio,
                        min_dark_pixel_ratio=args.min_dark_pixel_ratio,
                        max_core_mean_gray_delta=args.max_core_mean_gray_delta,
                        max_edge_mean_gray_delta=args.max_edge_mean_gray_delta,
                        max_core_lighten_delta=args.max_core_lighten_delta,
                        max_edge_lighten_delta=args.max_edge_lighten_delta,
                    )
                    rank_patched_alignment, rank_patched_alignment_issues = char_alignment_gate(
                        crop.size,
                        plan,
                        rank_patched_params,
                        max_char_center_dx=args.max_char_center_dx,
                        max_char_center_distance_delta=args.max_char_center_distance_delta,
                    )
                    rank_patched_font_style = font_style_gate(
                        crop,
                        plan,
                        rank_patched_params,
                        font_style_reference,
                        max_score_ratio=args.max_font_style_score_ratio,
                    )
                    rank_patched_report["strict_visual_metrics"] = rank_patched_metrics
                    rank_patched_report["char_gray_band_metrics"] = char_gray_band_metrics(
                        crop,
                        rank_patched_image,
                        plan,
                    )
                    rank_patched_report["char_alignment_metrics"] = rank_patched_alignment
                    rank_patched_report["font_style_gate"] = rank_patched_font_style
                    rank_patched_report["strict_gate"] = {
                        "max_dark_pixel_ratio": args.max_dark_pixel_ratio,
                        "min_dark_pixel_ratio": args.min_dark_pixel_ratio,
                        "max_core_mean_gray_delta": args.max_core_mean_gray_delta,
                        "max_edge_mean_gray_delta": args.max_edge_mean_gray_delta,
                        "max_core_lighten_delta": args.max_core_lighten_delta,
                        "max_edge_lighten_delta": args.max_edge_lighten_delta,
                        "max_char_center_dx": args.max_char_center_dx,
                        "max_char_center_distance_delta": args.max_char_center_distance_delta,
                        "max_font_style_score_ratio": args.max_font_style_score_ratio,
                        "pass": not rank_patched_issues
                        and not rank_patched_alignment_issues
                        and bool(rank_patched_font_style.get("pass", True)),
                        "issues": rank_patched_issues
                        + rank_patched_alignment_issues
                        + list(rank_patched_font_style.get("issues", [])),
                    }
                    attach_report_stage_context(rank_patched_report, pipeline_profile)
                    chosen_tuple = next(
                        (item for item in rendered if item[0].candidate_id == chosen_params.candidate_id),
                        None,
                    )
                    chosen_score = (
                        local_score(
                            crop,
                            chosen_tuple[1],
                            plan,
                            chosen_tuple[2],
                            blur_score_weight=args.blur_score_weight,
                            blur_score_free_margin=args.blur_score_free_margin,
                        )
                        if chosen_tuple
                        else local_best_score
                    )
                    rank_patched_score = local_score(
                        crop,
                        rank_patched_image,
                        plan,
                        rank_patched_report,
                        blur_score_weight=args.blur_score_weight,
                        blur_score_free_margin=args.blur_score_free_margin,
                    )
                    rank_patch_worse_by_metrics = rank_patched_score > chosen_score + 0.1
                    if args.acceptance_mode == "strict" and (
                        not report_strict_pass(rank_patched_report) or rank_patch_worse_by_metrics
                    ):
                        rank_patched_image.save(iter_dir / "rejected_candidate_rank_patch.png")
                        write_json(
                            iter_dir / "rejected_candidate_rank_patch_report.json",
                            {
                                "params": asdict(rank_patched_params),
                                "hard_check": rank_patched_report,
                                "reason": (
                                    "candidate-rank patch would fail strict gate"
                                    if not report_strict_pass(rank_patched_report)
                                    else "candidate-rank patch worsens local true-black/gray-band metrics"
                                ),
                                "chosen_local_score": chosen_score,
                                "patched_local_score": rank_patched_score,
                            },
                        )
                        progress.emit(
                            "candidate_rank_patch_rejected",
                            (
                                "rejected candidate-rank patch that would fail strict gate"
                                if not report_strict_pass(rank_patched_report)
                                else "rejected candidate-rank patch that worsened local metrics"
                            ),
                            iteration=iteration + 1,
                            font=rank_patched_params.font_name,
                            font_size=rank_patched_params.font_size,
                            opacity=rank_patched_params.opacity,
                            blur=rank_patched_params.blur,
                            chosen_score=round(float(chosen_score), 3),
                            patched_score=round(float(rank_patched_score), 3),
                        )
                    else:
                        chosen_params = rank_patched_params
                progress.emit(
                    "vision_candidate_rank_finished",
                    "candidate ranking returned",
                    iteration=iteration + 1,
                    best_candidate=str(candidate_rank_json.get("best_candidate")),
                    pass_value=bool(candidate_rank_json.get("pass")),
                )
            except Exception as exc:
                if args.vision == "on":
                    raise
                error = f"iteration {iteration} candidate rank: {exc}"
                vision_errors.append(error)
                write_json(iter_dir / "visual_eval_candidate_rank_error.json", {"error": error})
                progress.emit(
                    "vision_candidate_rank_error",
                    "candidate ranking failed",
                    iteration=iteration + 1,
                    error=str(exc),
                )

        current_image = render_candidate(crop, plan, chosen_params)
        current_report = hard_check(crop, current_image, plan.target_roi, plan.protected_boxes)
        current_strict_metrics = strict_visual_metrics(crop, current_image, plan.target_roi)
        current_strict_issues = strict_gate_issues(
            current_strict_metrics,
            max_dark_pixel_ratio=args.max_dark_pixel_ratio,
            min_dark_pixel_ratio=args.min_dark_pixel_ratio,
            max_core_mean_gray_delta=args.max_core_mean_gray_delta,
            max_edge_mean_gray_delta=args.max_edge_mean_gray_delta,
            max_core_lighten_delta=args.max_core_lighten_delta,
            max_edge_lighten_delta=args.max_edge_lighten_delta,
        )
        current_alignment_metrics, current_alignment_issues = char_alignment_gate(
            crop.size,
            plan,
            chosen_params,
            max_char_center_dx=args.max_char_center_dx,
            max_char_center_distance_delta=args.max_char_center_distance_delta,
        )
        current_font_style_report = font_style_gate(
            crop,
            plan,
            chosen_params,
            font_style_reference,
            max_score_ratio=args.max_font_style_score_ratio,
        )
        current_report["strict_visual_metrics"] = current_strict_metrics
        current_report["char_gray_band_metrics"] = char_gray_band_metrics(crop, current_image, plan)
        current_report["char_alignment_metrics"] = current_alignment_metrics
        current_report["font_style_gate"] = current_font_style_report
        current_report["strict_gate"] = {
            "max_dark_pixel_ratio": args.max_dark_pixel_ratio,
            "min_dark_pixel_ratio": args.min_dark_pixel_ratio,
            "max_core_mean_gray_delta": args.max_core_mean_gray_delta,
            "max_edge_mean_gray_delta": args.max_edge_mean_gray_delta,
            "max_core_lighten_delta": args.max_core_lighten_delta,
            "max_edge_lighten_delta": args.max_edge_lighten_delta,
            "max_char_center_dx": args.max_char_center_dx,
            "max_char_center_distance_delta": args.max_char_center_distance_delta,
            "max_font_style_score_ratio": args.max_font_style_score_ratio,
            "pass": not current_strict_issues
            and not current_alignment_issues
            and bool(current_font_style_report.get("pass", True)),
            "issues": current_strict_issues + current_alignment_issues + list(current_font_style_report.get("issues", [])),
        }
        attach_report_stage_context(current_report, pipeline_profile)
        current_image_path = iter_dir / "current_after_patch.png"
        compare_path = iter_dir / "compare_current.png"
        current_image.save(current_image_path)
        make_compare_image(crop, current_image, compare_path, scale=args.compare_scale)

        stop_iteration = False
        if vision_client and args.use_tuning_prompt:
            prompt = (
                tuning_prompt_template.replace(
                    "{current_params}",
                    json.dumps(asdict(chosen_params), ensure_ascii=False, indent=2),
                ).replace(
                    "{hard_check_report}",
                    json.dumps(current_report, ensure_ascii=False, indent=2),
                )
            )
            prompt += vision_task_context(plan)
            if args.acceptance_mode == "strict":
                prompt += STRICT_ACCEPTANCE_APPENDIX
            progress.emit(
                "vision_tuning_started",
                "requesting tuning advice",
                iteration=iteration + 1,
            )
            try:
                tuning_json = vision_client.call_json(
                    system_prompt=master_prompt,
                    user_prompt=prompt,
                    image_paths=[run_dir / "original_crop.png", current_image_path, compare_path],
                    prompt_name="tuning_prompt.txt",
                    audit_path=iter_dir / "visual_eval_tuning_prompt_audit.json",
                )
                write_json(iter_dir / "visual_eval_tuning.json", tuning_json)
                tuning_wants_stop = bool(tuning_json.get("stop_iteration")) or bool(tuning_json.get("pass"))
                stop_iteration = tuning_wants_stop and (
                    args.acceptance_mode != "strict"
                    or (
                        not current_strict_issues
                        and not current_alignment_issues
                        and bool(current_font_style_report.get("pass", True))
                    )
                )
                if not stop_iteration:
                    pre_patch_pass = (
                        bool(current_report.get("pass"))
                        and not current_strict_issues
                        and not current_alignment_issues
                        and bool(current_font_style_report.get("pass", True))
                    )
                    patched_params = apply_suggested_patch(chosen_params, tuning_json.get("suggested_patch"))
                    patched_image = render_candidate(crop, plan, patched_params)
                    patched_report = hard_check(crop, patched_image, plan.target_roi, plan.protected_boxes)
                    patched_strict_metrics = strict_visual_metrics(crop, patched_image, plan.target_roi)
                    patched_strict_issues = strict_gate_issues(
                        patched_strict_metrics,
                        max_dark_pixel_ratio=args.max_dark_pixel_ratio,
                        min_dark_pixel_ratio=args.min_dark_pixel_ratio,
                        max_core_mean_gray_delta=args.max_core_mean_gray_delta,
                        max_edge_mean_gray_delta=args.max_edge_mean_gray_delta,
                        max_core_lighten_delta=args.max_core_lighten_delta,
                        max_edge_lighten_delta=args.max_edge_lighten_delta,
                    )
                    patched_alignment_metrics, patched_alignment_issues = char_alignment_gate(
                        crop.size,
                        plan,
                        patched_params,
                        max_char_center_dx=args.max_char_center_dx,
                        max_char_center_distance_delta=args.max_char_center_distance_delta,
                    )
                    patched_font_style_report = font_style_gate(
                        crop,
                        plan,
                        patched_params,
                        font_style_reference,
                        max_score_ratio=args.max_font_style_score_ratio,
                    )
                    patched_report["strict_visual_metrics"] = patched_strict_metrics
                    patched_report["char_gray_band_metrics"] = char_gray_band_metrics(
                        crop,
                        patched_image,
                        plan,
                    )
                    patched_report["char_alignment_metrics"] = patched_alignment_metrics
                    patched_report["font_style_gate"] = patched_font_style_report
                    patched_report["strict_gate"] = {
                        "max_dark_pixel_ratio": args.max_dark_pixel_ratio,
                        "min_dark_pixel_ratio": args.min_dark_pixel_ratio,
                        "max_core_mean_gray_delta": args.max_core_mean_gray_delta,
                        "max_edge_mean_gray_delta": args.max_edge_mean_gray_delta,
                        "max_core_lighten_delta": args.max_core_lighten_delta,
                        "max_edge_lighten_delta": args.max_edge_lighten_delta,
                        "max_char_center_dx": args.max_char_center_dx,
                        "max_char_center_distance_delta": args.max_char_center_distance_delta,
                        "max_font_style_score_ratio": args.max_font_style_score_ratio,
                        "pass": not patched_strict_issues
                        and not patched_alignment_issues
                        and bool(patched_font_style_report.get("pass", True)),
                        "issues": patched_strict_issues
                        + patched_alignment_issues
                        + list(patched_font_style_report.get("issues", [])),
                    }
                    attach_report_stage_context(patched_report, pipeline_profile)
                    patched_pass = (
                        bool(patched_report.get("pass"))
                        and not patched_strict_issues
                        and not patched_alignment_issues
                        and bool(patched_font_style_report.get("pass", True))
                    )
                    current_metric_score = local_score(
                        crop,
                        current_image,
                        plan,
                        current_report,
                        blur_score_weight=args.blur_score_weight,
                        blur_score_free_margin=args.blur_score_free_margin,
                    )
                    patched_metric_score = local_score(
                        crop,
                        patched_image,
                        plan,
                        patched_report,
                        blur_score_weight=args.blur_score_weight,
                        blur_score_free_margin=args.blur_score_free_margin,
                    )
                    patch_worse_by_metrics = patched_metric_score > current_metric_score + 0.1
                    if pre_patch_pass and (not patched_pass or patch_worse_by_metrics):
                        patched_image.save(iter_dir / "rejected_tuning_patch.png")
                        write_json(
                            iter_dir / "rejected_tuning_patch_report.json",
                            {
                                "params": asdict(patched_params),
                                "hard_check": patched_report,
                                "reason": (
                                    "tuning patch would fail strict gate"
                                    if not patched_pass
                                    else "tuning patch worsens local true-black/gray-band metrics"
                                ),
                                "current_local_score": current_metric_score,
                                "patched_local_score": patched_metric_score,
                            },
                        )
                        progress.emit(
                            "tuning_patch_rejected",
                            (
                                "rejected tuning patch that would fail strict gate"
                                if not patched_pass
                                else "rejected tuning patch that worsened local metrics"
                            ),
                            iteration=iteration + 1,
                            font=patched_params.font_name,
                            font_size=patched_params.font_size,
                            opacity=patched_params.opacity,
                            blur=patched_params.blur,
                            current_score=round(float(current_metric_score), 3),
                            patched_score=round(float(patched_metric_score), 3),
                        )
                    else:
                        chosen_params = patched_params
                        current_image = patched_image
                        current_report = patched_report
                        current_strict_metrics = patched_strict_metrics
                        current_strict_issues = patched_strict_issues
                        current_alignment_metrics = patched_alignment_metrics
                        current_alignment_issues = patched_alignment_issues
                        current_font_style_report = patched_font_style_report
                        current_image.save(iter_dir / "current_after_tuning_patch.png")
                progress.emit(
                    "vision_tuning_finished",
                    "tuning advice returned",
                    iteration=iteration + 1,
                    pass_value=bool(tuning_json.get("pass")),
                    stop_iteration=stop_iteration,
                    strict_gate_pass=not current_strict_issues
                    and not current_alignment_issues
                    and bool(current_font_style_report.get("pass", True)),
                )
            except Exception as exc:
                if args.vision == "on":
                    raise
                error = f"iteration {iteration} tuning: {exc}"
                vision_errors.append(error)
                write_json(iter_dir / "visual_eval_tuning_error.json", {"error": error})
                progress.emit(
                    "vision_tuning_error",
                    "tuning request failed",
                    iteration=iteration + 1,
                    error=str(exc),
                )

        history.append(
            {
                "iteration": iteration,
                "chosen_params": asdict(chosen_params),
                "chosen_hard_check": current_report,
                "candidate_rank": candidate_rank_json,
                "tuning": tuning_json,
                "contact_sheet": str(contact_sheet_path),
                "vision_contact_sheet": str(vision_contact_sheet_path),
                "vision_candidate_count": vision_candidate_count,
                "current_image": str(current_image_path),
            }
        )
        final_crop = current_image
        final_params = replace_candidate_id(chosen_params, "final")
        current = final_params
        progress.emit(
            "iteration_finished",
            "finished iteration",
            iteration=iteration + 1,
            font=final_params.font_name,
            font_size=final_params.font_size,
            opacity=final_params.opacity,
            blur=final_params.blur,
            strict_gate_pass=not current_strict_issues
            and not current_alignment_issues
            and bool(current_font_style_report.get("pass", True)),
        )
        if stop_iteration:
            progress.emit(
                "iteration_stop_requested",
                "stopping before max iterations",
                iteration=iteration + 1,
            )
            break

    if final_crop is None:
        final_crop = render_candidate(crop, plan, final_params)

    progress.emit("final_render_started", "writing final images")
    final_crop_path = run_dir / "final_crop.png"
    final_crop.save(final_crop_path)
    final_crop_report = hard_check(crop, final_crop, plan.target_roi, plan.protected_boxes)
    final_strict_metrics = strict_visual_metrics(crop, final_crop, plan.target_roi)
    final_strict_issues = strict_gate_issues(
        final_strict_metrics,
        max_dark_pixel_ratio=args.max_dark_pixel_ratio,
        min_dark_pixel_ratio=args.min_dark_pixel_ratio,
        max_core_mean_gray_delta=args.max_core_mean_gray_delta,
        max_edge_mean_gray_delta=args.max_edge_mean_gray_delta,
        max_core_lighten_delta=args.max_core_lighten_delta,
        max_edge_lighten_delta=args.max_edge_lighten_delta,
    )
    final_alignment_metrics, final_alignment_issues = char_alignment_gate(
        crop.size,
        plan,
        final_params,
        max_char_center_dx=args.max_char_center_dx,
        max_char_center_distance_delta=args.max_char_center_distance_delta,
    )
    final_font_style_report = font_style_gate(
        crop,
        plan,
        final_params,
        font_style_reference,
        max_score_ratio=args.max_font_style_score_ratio,
    )
    final_crop_report["strict_visual_metrics"] = final_strict_metrics
    final_char_gray_metrics = char_gray_band_metrics(crop, final_crop, plan)
    final_crop_report["char_gray_band_metrics"] = final_char_gray_metrics
    final_crop_report["char_alignment_metrics"] = final_alignment_metrics
    final_crop_report["font_style_gate"] = final_font_style_report
    final_crop_report["strict_gate"] = {
        "max_dark_pixel_ratio": args.max_dark_pixel_ratio,
        "min_dark_pixel_ratio": args.min_dark_pixel_ratio,
        "max_core_mean_gray_delta": args.max_core_mean_gray_delta,
        "max_edge_mean_gray_delta": args.max_edge_mean_gray_delta,
        "max_core_lighten_delta": args.max_core_lighten_delta,
        "max_edge_lighten_delta": args.max_edge_lighten_delta,
        "max_char_center_dx": args.max_char_center_dx,
        "max_char_center_distance_delta": args.max_char_center_distance_delta,
        "max_font_style_score_ratio": args.max_font_style_score_ratio,
        "pass": not final_strict_issues
        and not final_alignment_issues
        and bool(final_font_style_report.get("pass", True)),
        "issues": final_strict_issues + final_alignment_issues + list(final_font_style_report.get("issues", [])),
    }
    attach_report_stage_context(final_crop_report, pipeline_profile)
    make_compare_image(crop, final_crop, run_dir / "compare_final_crop.png", scale=args.compare_scale)

    final_full_path = None
    final_full_report = None
    full_roi = None
    if split_box:
        final_full = paste_crop(full_image, final_crop, split_box)
        final_full_path = run_dir / "final_full.png"
        final_full.save(final_full_path)
        sx1, sy1, _, _ = split_box
        full_roi = add_offset(plan.target_roi, sx1, sy1)
        full_protected = tuple(add_offset(box, sx1, sy1) for box in plan.protected_boxes)
        final_full_report = hard_check(full_image, final_full, full_roi, full_protected)
    progress.emit(
        "final_hard_check_finished",
        "final hard checks completed",
        crop_pass=bool(final_crop_report.get("pass")),
        full_pass=bool(final_full_report.get("pass")) if final_full_report else None,
    )

    final_acceptance_json: dict[str, Any] | None = None
    if vision_client:
        hard_payload = {
            "pipeline_profile": pipeline_profile,
            "stage_context": model_stage_context(final_crop_report, pipeline_profile),
            "crop": final_crop_report,
            "full": final_full_report,
            "strict_visual_metrics": final_strict_metrics,
            "char_gray_band_metrics": final_char_gray_metrics,
            "char_alignment_metrics": final_alignment_metrics,
            "font_style_gate": final_font_style_report,
            "max_dark_pixel_ratio": args.max_dark_pixel_ratio,
            "min_dark_pixel_ratio": args.min_dark_pixel_ratio,
            "max_core_mean_gray_delta": args.max_core_mean_gray_delta,
            "max_edge_mean_gray_delta": args.max_edge_mean_gray_delta,
            "max_core_lighten_delta": args.max_core_lighten_delta,
            "max_edge_lighten_delta": args.max_edge_lighten_delta,
            "max_char_center_dx": args.max_char_center_dx,
            "max_char_center_distance_delta": args.max_char_center_distance_delta,
            "max_font_style_score_ratio": args.max_font_style_score_ratio,
        }
        prompt = (
            final_prompt_template.replace(
                "{final_params}",
                json.dumps(asdict(final_params), ensure_ascii=False, indent=2),
            ).replace(
                "{hard_check_report}",
                json.dumps(hard_payload, ensure_ascii=False, indent=2),
            )
        )
        prompt += vision_task_context(plan)
        if args.acceptance_mode == "strict":
            prompt += STRICT_ACCEPTANCE_APPENDIX
        try:
            progress.emit("final_acceptance_started", "requesting final visual acceptance")
            final_acceptance_json = vision_client.call_json(
                system_prompt=master_prompt,
                user_prompt=prompt,
                image_paths=[run_dir / "original_crop.png", final_crop_path, run_dir / "compare_final_crop.png"],
                prompt_name="final_acceptance_prompt.txt",
                audit_path=run_dir / "final_acceptance_prompt_audit.json",
            )
            write_json(run_dir / "final_acceptance.json", final_acceptance_json)
            progress.emit(
                "final_acceptance_finished",
                "final visual acceptance returned",
                pass_value=bool(final_acceptance_json.get("pass")),
                acceptance_level=str(final_acceptance_json.get("acceptance_level")),
            )
        except Exception as exc:
            if args.vision == "on":
                raise
            error = f"final acceptance: {exc}"
            vision_errors.append(error)
            write_json(run_dir / "final_acceptance_error.json", {"error": error})
            progress.emit("final_acceptance_error", "final visual acceptance failed", error=str(exc))
    if args.acceptance_mode == "strict":
        final_acceptance_json = strict_acceptance_gate(
            final_acceptance_json,
            final_strict_metrics,
            max_dark_pixel_ratio=args.max_dark_pixel_ratio,
            min_dark_pixel_ratio=args.min_dark_pixel_ratio,
            max_core_mean_gray_delta=args.max_core_mean_gray_delta,
            max_edge_mean_gray_delta=args.max_edge_mean_gray_delta,
            max_core_lighten_delta=args.max_core_lighten_delta,
            max_edge_lighten_delta=args.max_edge_lighten_delta,
            max_char_center_dx=args.max_char_center_dx,
            max_char_center_distance_delta=args.max_char_center_distance_delta,
            font_style_report=final_font_style_report,
            char_alignment_report=final_alignment_metrics,
            enforce_font_similarity=args.enforce_font_similarity,
        )
        write_json(run_dir / "final_acceptance_strict.json", final_acceptance_json)
        progress.emit(
            "strict_acceptance_gate_finished",
            "strict acceptance gate completed",
            pass_value=bool(final_acceptance_json.get("pass")),
            final_decision=str(final_acceptance_json.get("final_decision")),
        )

    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "context": context,
        "font_candidates": [{"name": name, "path": path} for name, path in font_candidates],
        "rejected_font_candidates": rejected_font_candidates,
        "required_font_text": required_font_text,
        "font_style_reference": font_style_reference,
        "font_source": args.font_source,
        "font_scan_dirs": args.font_scan_dirs,
        "max_scanned_fonts": args.max_scanned_fonts,
        "font_candidate_pool_size": args.font_candidate_pool_size,
        "pipeline_profile": pipeline_profile,
        "environment": environment_report(args.env, args.metadata),
        "final_params": asdict(final_params),
        "final_crop": str(final_crop_path),
        "final_full": str(final_full_path) if final_full_path else None,
        "final_crop_hard_check": final_crop_report,
        "final_full_hard_check": final_full_report,
        "final_strict_visual_metrics": final_strict_metrics,
        "final_char_alignment_metrics": final_alignment_metrics,
        "final_font_style_gate": final_font_style_report,
        "full_roi": list(full_roi) if full_roi else None,
        "vision_mode": args.vision,
        "acceptance_mode": args.acceptance_mode,
        "max_dark_pixel_ratio": args.max_dark_pixel_ratio,
        "min_dark_pixel_ratio": args.min_dark_pixel_ratio,
        "max_core_mean_gray_delta": args.max_core_mean_gray_delta,
        "max_edge_mean_gray_delta": args.max_edge_mean_gray_delta,
        "max_core_lighten_delta": args.max_core_lighten_delta,
        "max_edge_lighten_delta": args.max_edge_lighten_delta,
        "max_char_center_dx": args.max_char_center_dx,
        "max_char_center_distance_delta": args.max_char_center_distance_delta,
        "max_font_style_score_ratio": args.max_font_style_score_ratio,
        "max_model_local_score_delta": args.max_model_local_score_delta,
        "max_model_font_style_ratio_delta": args.max_model_font_style_ratio_delta,
        "blur_score_weight": args.blur_score_weight,
        "blur_score_free_margin": args.blur_score_free_margin,
        "max_iterations": max_iterations,
        "metadata_max_iterations": metadata_max_iterations,
        "progress": str(progress.progress_path),
        "vision_errors": vision_errors,
        "vision_prompt_audits": sorted(str(path) for path in run_dir.rglob("*prompt_audit.json")),
        "final_acceptance": final_acceptance_json,
        "history": history,
    }
    write_json(run_dir / "summary.json", summary)
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.logs_dir / f"{run_id}_summary.json", summary)
    progress.emit(
        "run_finished",
        "wrote summary",
        summary=str(run_dir / "summary.json"),
        final_crop=str(final_crop_path),
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Iterative ROI text replacement with packaged prompts.")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument("--text", default=None)
    parser.add_argument("--source-text", default=None)
    parser.add_argument("--roi", default=None, help="Manual crop-local target ROI x1,y1,x2,y2.")
    parser.add_argument("--font-path", default=None)
    parser.add_argument("--font-source", choices=("recommended", "scan"), default="recommended")
    parser.add_argument("--font-scan-dirs", action="append", default=None)
    parser.add_argument("--max-scanned-fonts", type=int, default=500)
    parser.add_argument("--font-candidate-pool-size", type=int, default=4)
    parser.add_argument("--vision-candidate-limit", type=int, default=48)
    parser.add_argument("--font-size", type=int, default=0)
    parser.add_argument("--opacity", type=float, default=0.86)
    parser.add_argument("--blur", type=float, default=0.5)
    parser.add_argument("--stroke-opacity", type=float, default=0.0)
    parser.add_argument("--ink-gain", type=float, default=0.0)
    parser.add_argument("--alpha-contrast", type=float, default=0.0)
    parser.add_argument("--core-ink-gain", type=float, default=0.0)
    parser.add_argument("--core-darken-strength", type=float, default=0.0)
    parser.add_argument("--core-darken-threshold", type=int, default=120)
    parser.add_argument("--core-darken-target-gray", type=int, default=28)
    parser.add_argument("--mask-threshold", type=int, default=165)
    parser.add_argument("--mask-dilate-iterations", type=int, default=2)
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--max-candidates", type=int, default=28)
    parser.add_argument("--vision", choices=("on", "auto", "off"), default="auto")
    parser.add_argument("--acceptance-mode", choices=("strict", "normal"), default="strict")
    parser.add_argument("--profile", choices=stage_profile_choices(), default="photo_scan")
    parser.add_argument("--max-dark-pixel-ratio", type=float, default=1.12)
    parser.add_argument("--min-dark-pixel-ratio", type=float, default=0.88)
    parser.add_argument("--max-core-mean-gray-delta", type=float, default=18.0)
    parser.add_argument("--max-edge-mean-gray-delta", type=float, default=16.0)
    parser.add_argument("--max-core-lighten-delta", type=float, default=2.0)
    parser.add_argument("--max-edge-lighten-delta", type=float, default=4.0)
    parser.add_argument("--max-char-center-dx", type=float, default=2.0)
    parser.add_argument("--max-char-center-distance-delta", type=float, default=2.0)
    parser.add_argument("--max-font-style-score-ratio", type=float, default=1.25)
    parser.add_argument("--max-model-local-score-delta", type=float, default=3.0)
    parser.add_argument("--max-model-font-style-ratio-delta", type=float, default=0.05)
    parser.add_argument("--blur-score-weight", type=float, default=160.0)
    parser.add_argument("--blur-score-free-margin", type=float, default=0.15)
    parser.add_argument("--font-ranking", choices=("style", "document"), default="style")
    parser.add_argument("--font-style-min-size", type=int, default=16)
    parser.add_argument("--font-style-max-size", type=int, default=28)
    parser.add_argument("--style-reference-text", default=None)
    parser.add_argument("--prefer-serif-fonts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enforce-font-similarity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-tuning-prompt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sheet-scale", type=int, default=8)
    parser.add_argument("--sheet-cols", type=int, default=4)
    parser.add_argument("--compare-scale", type=int, default=10)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = run_pipeline(args)
    print(json.dumps(
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
    ))


if __name__ == "__main__":
    main()
