#!/usr/bin/env python3
"""
mpc_api.py
==========
Local API that triggers Module 4 MPC runs (single day or a date-range season)
and reports progress, for an interactive frontend to poll. Same shape as
Module3's forecast_api.py (stdlib ThreadingHTTPServer, CORS-open, JSON over
GET) with one deliberate difference: Module 4 runs take minutes to hours, not
seconds, so jobs are launched as background subprocesses (never blocking a
request) and progress is reported via a status endpoint the frontend polls.

Does NOT change any pipeline logic - every run is triggered exactly the way
you'd already trigger it from the command line (run_daily_cli.py /
run_simulation.py), just launched as a subprocess instead of typed by hand.

Endpoints:
  GET /health
  GET /api/run/daily?date=YYYY-MM-DD&fast_test=1
  GET /api/run/season?start=YYYY-MM-DD&end=YYYY-MM-DD&season=yala&duration=135
                      &forecast=data/module4_forecasts_mar_aug_2025.csv
                      &output=my_run&fast_test=1
  GET /api/status?job=<id>
  GET /api/stop?job=<id>

Only one job runs at a time (NSGA-II is CPU-heavy; concurrent runs would just
contend for cores and, for the daily job, clobber the same overwritten
outputs/ files) - a second run request while one is active gets HTTP 409.

/api/stop kills the job's subprocess and deletes its output folder ONLY if
that folder was newly created by this job. If the job reused a folder that
already existed before it started (e.g. resuming a season run), nothing is
deleted - the already-completed days stay intact and resumable, exactly as
run_simulation.py's own resume design expects.

Usage:
    python mpc_api.py
"""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

BASE_DIR = Path(__file__).resolve().parent   # Module4/mpc-pipeline/
LOG_DIR = BASE_DIR / "_api_logs"
LOG_DIR.mkdir(exist_ok=True)

_lock = threading.Lock()
_jobs = {}            # job_id -> job dict
_active_job_id = None


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _active_job():
    if _active_job_id is None:
        return None
    job = _jobs.get(_active_job_id)
    if job is None:
        return None
    if job["proc"].poll() is None:
        return job
    return None   # process has exited; no longer "active"


def _start_job(kind, cmd, output_dir, meta=None):
    global _active_job_id
    with _lock:
        running = _active_job()
        if running is not None:
            return None, running["id"]

        job_id = uuid.uuid4().hex[:10]
        log_path = LOG_DIR / f"{job_id}.log"
        log_handle = open(log_path, "w", encoding="utf-8")
        # Captured BEFORE the subprocess can create it, so /api/stop knows
        # whether this job owns the whole folder (safe to delete on stop) or
        # is reusing/resuming one that already had data in it (never delete).
        pre_existing_output = Path(output_dir).is_dir()
        # PYTHONUNBUFFERED: logging_utils.progress() only flushes when stdout
        # is a real TTY (module4/logging_utils.py IS_TTY check) - it's a file
        # here, so without this every print() sits in Python's default ~8KB
        # buffer and the log (and /api/status's log_tail) stays empty for the
        # whole run, only appearing all at once when the process exits. Fast
        # runs finish before that's noticeable; a full pop=200/gen=300 run
        # takes minutes, so it looked like "no output" - it wasn't stuck, the
        # output just hadn't been flushed to the file yet.
        child_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(
            cmd, cwd=str(BASE_DIR), stdout=log_handle, stderr=subprocess.STDOUT,
            env=child_env,
        )
        job = {
            "id": job_id,
            "kind": kind,
            "cmd": [str(c) for c in cmd],
            "output_dir": str(output_dir),
            "pre_existing_output": pre_existing_output,
            "log_path": str(log_path),
            "proc": proc,
            "log_handle": log_handle,
            "status": "running",
            "started_at": _now(),
            "finished_at": None,
            "error": None,
            "cleaned_up": None,
            **(meta or {}),
        }
        _jobs[job_id] = job
        _active_job_id = job_id
        return job, None


def _kill_tree(proc):
    """
    Kill proc AND its descendants. proc.terminate()/kill() alone only signals
    the immediate child - on Windows, `.venv\\Scripts\\python.exe` launches as
    a stub that re-execs the real interpreter as a further child, which
    survives a plain terminate() untouched (confirmed: it did, twice, while
    testing this). taskkill /T kills the whole tree in one call.
    """
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True, timeout=10,
            )
            return
        except (OSError, subprocess.TimeoutExpired):
            pass   # fall through to the plain terminate/kill below
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def _stop_job(job):
    global _active_job_id
    with _lock:
        if job["status"] != "running":
            return _job_status(job)   # already finished on its own; nothing to stop

        proc = job["proc"]
        _kill_tree(proc)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass

        try:
            job["log_handle"].close()
        except OSError:
            pass

        job["status"] = "stopped"
        job["finished_at"] = _now()

        if not job["pre_existing_output"]:
            # This job's output folder didn't exist before it started, so
            # everything in it belongs to this incomplete run only.
            time.sleep(0.2)   # let the OS release any just-closed file handles
            shutil.rmtree(job["output_dir"], ignore_errors=True)
            job["cleaned_up"] = True
        else:
            job["cleaned_up"] = False

        if _active_job_id == job["id"]:
            _active_job_id = None

        return _job_status(job)


_SEASON_LINE_RE = re.compile(r"day (\d+)/(\d+)\s+feasible=(\w+)")


def _job_status(job):
    proc = job["proc"]
    ret = proc.poll()
    if ret is not None and job["status"] == "running":
        job["status"] = "done" if ret == 0 else "error"
        job["finished_at"] = _now()
        try:
            job["log_handle"].close()
        except OSError:
            pass
        if ret != 0:
            tail = _tail(job["log_path"], 20)
            job["error"] = tail or f"process exited with code {ret}"

    out = {
        "id": job["id"],
        "kind": job["kind"],
        "status": job["status"],
        "started_at": job["started_at"],
        "finished_at": job["finished_at"],
        "output_dir": job["output_dir"],
        "error": job["error"],
        "cleaned_up": job["cleaned_up"],
        "log_tail": _tail(job["log_path"], 60),
    }

    if job["kind"] == "season":
        log_path = Path(job["output_dir"]) / "run_progress.log"
        if log_path.exists():
            try:
                lines = [l for l in log_path.read_text(
                    encoding="utf-8", errors="ignore").splitlines() if l.strip()]
            except OSError:
                lines = []
            if lines:
                out["last_line"] = lines[-1]
                m = _SEASON_LINE_RE.search(lines[-1])
                if m:
                    out["day"] = int(m.group(1))
                    out["total_days"] = int(m.group(2))
                    out["last_feasible"] = m.group(3) == "True"

    if job["kind"] == "daily" and job["status"] == "done":
        status_path = Path(job["output_dir"]) / "last_run_status.json"
        if status_path.exists():
            try:
                out["result"] = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass

    return out


def _tail(path, n):
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
        return "\n".join(lines[-n:])
    except OSError:
        return None


class MPCHandler(BaseHTTPRequestHandler):
    def _send(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass   # keep the console quiet; per-job output goes to _api_logs/<job_id>.log

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        def p(name, default=""):
            return (params.get(name, [default])[0] or default).strip()

        if parsed.path == "/health":
            self._send({"status": "ok"})
            return

        if parsed.path == "/api/run/daily":
            date_value = p("date")
            if not date_value:
                self._send({"error": "date is required (YYYY-MM-DD)"}, status=400)
                return
            fast_test = p("fast_test").lower() in {"1", "true", "yes"}
            cmd = [sys.executable, "run_daily_cli.py", "--date", date_value]
            if fast_test:
                cmd.append("--fast-test")
            job, blocked_by = _start_job("daily", cmd, output_dir="outputs",
                                         meta={"date": date_value, "fast_test": fast_test})
            if job is None:
                self._send({"error": "a job is already running", "job_id": blocked_by}, status=409)
                return
            self._send({"job_id": job["id"], "status": "running"})
            return

        if parsed.path == "/api/run/season":
            start = p("start")
            end = p("end")
            if not start or not end:
                self._send({"error": "start and end are required (YYYY-MM-DD)"}, status=400)
                return
            season = p("season", "yala")
            duration = p("duration", "135")
            forecast = p("forecast", "data/module4_forecasts_mar_aug_2025.csv")
            output = p("output") or f"outputs_web_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            fast_test = p("fast_test").lower() in {"1", "true", "yes"}

            cmd = [sys.executable, "run_simulation.py",
                   "--start", start, "--end", end,
                   "--season", season, "--duration", duration,
                   "--forecast", forecast, "--output", output,
                   "--consolidated-output"]
            if fast_test:
                cmd.append("--fast-test")

            job, blocked_by = _start_job("season", cmd, output_dir=output, meta={
                "start": start, "end": end, "season": season, "duration": duration,
                "forecast": forecast, "fast_test": fast_test,
            })
            if job is None:
                self._send({"error": "a job is already running", "job_id": blocked_by}, status=409)
                return
            self._send({"job_id": job["id"], "status": "running"})
            return

        if parsed.path == "/api/status":
            job_id = p("job")
            job = _jobs.get(job_id)
            if job is None:
                self._send({"error": f"unknown job_id {job_id!r}"}, status=404)
                return
            self._send(_job_status(job))
            return

        if parsed.path == "/api/stop":
            job_id = p("job")
            job = _jobs.get(job_id)
            if job is None:
                self._send({"error": f"unknown job_id {job_id!r}"}, status=404)
                return
            self._send(_stop_job(job))
            return

        self.send_error(404)


def main():
    port = int(os.environ.get("MPC_API_PORT", "8001"))
    server = ThreadingHTTPServer(("127.0.0.1", port), MPCHandler)
    print(f"Module 4 MPC API listening on http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
