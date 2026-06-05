from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from roi_image_edit.processing_service import process_payload


ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT / "web"
MAX_WEB_JOB_EVENTS = 800
WEB_JOB_LOCK = threading.Lock()
WEB_JOBS: dict[str, dict[str, Any]] = {}


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
