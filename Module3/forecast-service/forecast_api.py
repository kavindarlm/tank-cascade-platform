#!/usr/bin/env python3
"""Minimal local API that runs the forecast batch script and serves its CSV as JSON."""
import csv
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path
from urllib.parse import parse_qs, urlparse

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _log(message, level="INFO"):
    print(f"[forecast_api][{level}] {message}", flush=True)


def _safe_float(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, numeric)


def _list_history():
    """Return metadata for every saved forecast CSV, newest first."""
    files = []
    for path in sorted(OUTPUT_DIR.glob("forecast_*.csv"), reverse=True):
        date_str = path.stem.replace("forecast_", "")
        try:
            stat = path.stat()
        except OSError:
            continue

        size_kb = stat.st_size / 1024
        size_label = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb / 1024:.1f} MB"

        row_count = 0
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                row_count = sum(1 for _ in csv.DictReader(handle))
        except OSError:
            pass

        files.append({
            "date": date_str,
            "filename": path.name,
            "generated_at": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
            "size_label": size_label,
            "row_count": row_count,
        })
    return files


def _load_tank_names():
    tank_names = {}
    tanks_path = BASE_DIR / "data" / "tanks.csv"
    if not tanks_path.exists():
        return tank_names

    with tanks_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            tank_id = str(row.get("tank_id", "")).strip()
            if tank_id:
                tank_names[tank_id] = (row.get("tank_name") or f"Tank {tank_id}").strip()
    return tank_names


def _load_csv_rows(csv_path):
    if not csv_path.exists():
        return []

    tank_names = _load_tank_names()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            risk = (row.get("primary_risk") or row.get("classifier_risk") or "normal").strip().lower()
            if not risk:
                risk = "normal"

            tank_id = str(row.get("tank_id", "")).strip()
            rows.append({
                "tank_id": tank_id,
                "tank_name": tank_names.get(tank_id, f"Tank {tank_id}"),
                "date": row.get("date", ""),
                "storage": row.get("storage", ""),
                "storage_pct": row.get("storage_pct", ""),
                "storage_source": row.get("storage_source", ""),
                "primary_risk": risk,
                "classifier_risk": (row.get("classifier_risk") or "").strip(),
                "agreement": row.get("agreement", ""),
                "prob_drought": row.get("prob_drought", ""),
                "prob_normal": row.get("prob_normal", ""),
                "prob_overflow": row.get("prob_overflow", ""),
                "confidence": row.get("confidence", ""),
                "days_gap": row.get("days_gap", ""),
                "storage_pct_value": _safe_float(row.get("storage_pct", "")),
                "confidence_value": _safe_float(row.get("confidence", "")),
                "prob_drought_value": _safe_float(row.get("prob_drought", "")),
                "prob_normal_value": _safe_float(row.get("prob_normal", "")),
                "prob_overflow_value": _safe_float(row.get("prob_overflow", "")),
            })

    return rows


def _run_forecast(date_str, force=False):
    output_path = OUTPUT_DIR / f"forecast_{date_str}.csv"
    script_path = BASE_DIR / "batch_forecast.py"

    _log(f"Starting forecast run for date={date_str}, force={force}, output={output_path}")

    if output_path.exists() and not force:
        # Always regenerate for the requested date so the UI reflects the
        # latest forecast run instead of a stale previously generated file.
        pass
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(script_path), "--date", date_str, "--output", str(output_path)]
    _log(f"Executing forecast command: {' '.join(cmd)}")

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=1800,
        )
    except FileNotFoundError as exc:
        _log(f"Forecast subprocess could not start: {exc}", level="ERROR")
        raise RuntimeError(f"Forecast process could not start: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        _log(f"Forecast subprocess timed out after {exc.timeout} seconds", level="ERROR")
        raise RuntimeError("The forecast request timed out. Please try again.") from exc

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    combined_output = stderr or stdout or "No forecast output captured."

    if completed.returncode != 0:
        _log(f"Forecast command failed with exit code {completed.returncode}: {combined_output[:1500]}", level="ERROR")
        if output_path.exists():
            _log(f"Using fallback existing CSV output for date={date_str} because the refresh failed.")
            return output_path, "fallback"
        raise RuntimeError(combined_output[:500] or "Forecast run failed")

    _log(f"Forecast command finished successfully. Output: {output_path}")
    if stdout:
        _log(f"Forecast stdout: {stdout[:1000]}")
    if stderr:
        _log(f"Forecast stderr: {stderr[:1000]}")

    return output_path, "generated"


class ForecastHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(REPO_ROOT), **kwargs)

    def _send(self, payload, status=200, content_type="application/json"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        date_value = (params.get("date", [""])[0] or "").strip() or "2026-07-29"
        force = params.get("force", [""])[0].lower() in {"1", "true", "yes"}
        download = params.get("download", [""])[0].lower() in {"1", "true", "yes"}
        view = params.get("view", [""])[0].lower() in {"1", "true", "yes"}

        if parsed.path == "/health":
            self._send({"status": "ok"})
            return

        if parsed.path == "/api/forecast/history":
            _log("Received forecast history request")
            try:
                files = _list_history()
                self._send({"count": len(files), "files": files})
            except Exception as exc:
                error_message = str(exc) or "Failed to list saved forecasts."
                _log(f"Forecast history request failed: {error_message}", level="ERROR")
                self._send({"error": error_message}, status=500)
            return

        if parsed.path == "/api/forecast":
            _log(f"Received forecast request: date={date_value}, force={force}, view={view}, download={download}")
            try:
                if view:
                    csv_path = OUTPUT_DIR / f"forecast_{date_value}.csv"
                    if not csv_path.exists():
                        raise RuntimeError(f"No saved forecast found for {date_value}.")
                    source = "cached"
                else:
                    csv_path, source = _run_forecast(date_value, force=force)

                rows = _load_csv_rows(csv_path)
                _log(f"Forecast CSV loaded successfully for date={date_value}: {len(rows)} rows from {csv_path}")

                if download:
                    fieldnames = [
                        "tank_id",
                        "date",
                        "storage",
                        "storage_pct",
                        "storage_source",
                        "t+1",
                        "t+2",
                        "t+3",
                        "t+4",
                        "t+5",
                        "t+6",
                        "t+7",
                        "primary_risk",
                        "classifier_risk",
                        "agreement",
                        "prob_drought",
                        "prob_normal",
                        "prob_overflow",
                        "drought_duration_days",
                        "overflow_duration_days",
                        "confidence",
                        "days_gap",
                    ]
                    stream = StringIO()
                    writer = csv.DictWriter(stream, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in rows:
                        writer.writerow({k: row.get(k, "") for k in fieldnames})
                    self._send(stream.getvalue().encode("utf-8"), content_type="text/csv")
                    _log(f"Forecast CSV download served for date={date_value}")
                    return

                self._send({
                    "date": date_value,
                    "source": source,
                    "count": len(rows),
                    "rows": rows,
                })
            except Exception as exc:
                error_message = str(exc) or "Forecast request failed unexpectedly."
                _log(f"Forecast request failed for date={date_value}: {error_message}", level="ERROR")
                _log(traceback.format_exc(), level="ERROR")
                self._send({"error": error_message, "error_type": "forecast_error"}, status=500)
            return

        if parsed.path == "/":
            self.path = "/frontend/index.html"
        elif parsed.path in {"/index.html", "/storage.html", "/connectivity.html", "/forecasting.html"}:
            self.path = f"/frontend{parsed.path}"

        return super().do_GET()


def main():
    port = int(os.environ.get("FORECAST_API_PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), ForecastHandler)
    print(f"Forecast API listening on http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
