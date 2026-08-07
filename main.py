#!/usr/bin/env python3
"""Common entry point for the Tank Cascade Platform.

Run this from the repo root instead of cd-ing into each module's service
folder. It reuses Module 3's forecast API (forecast_api.py) and Module 4's
MPC API (mpc_api.py) as-is - both modules resolve their own paths from their
own file location, not from the current working directory, so importing
them here does not change their behavior.

PlatformHandler combines both APIs (plus the shared frontend/ static files)
behind a single port, so the whole platform starts with one `python main.py`
instead of running forecast_api.py and mpc_api.py separately on their own
ports.
"""
import os
import sys
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "Module3" / "forecast-service"))
sys.path.insert(0, str(REPO_ROOT / "Module4" / "mpc-pipeline"))

import forecast_api
import mpc_api

# Module 4's routes; everything else (Module 3's /health, /api/forecast*,
# and static frontend files) is handled by forecast_api.ForecastHandler.
_MPC_PATHS = {"/api/run/daily", "/api/run/season", "/api/status", "/api/tanks/geo", "/api/stop"}


class PlatformHandler(forecast_api.ForecastHandler):
    def do_GET(self):
        if urlparse(self.path).path in _MPC_PATHS:
            # mpc_api.MPCHandler.do_GET only touches self.path/self._send/
            # self.send_error, all of which this handler already provides,
            # so its route logic can run unmodified against this instance.
            return mpc_api.MPCHandler.do_GET(self)
        return super().do_GET()

    def log_message(self, fmt, *args):
        # /api/status is polled every second or two for the life of an MPC
        # run (minutes to hours) - logging every poll would drown out
        # everything else on the console.
        if urlparse(self.path).path == "/api/status":
            return
        super().log_message(fmt, *args)


def main():
    # module4.config's DATA_DIR ("data/") is read relative to the process's
    # cwd, not this file's location - /api/tanks/geo calls into it directly
    # (no subprocess, unlike the run/daily and run/season jobs, which already
    # pin their own cwd). Matching mpc_api.py's original standalone cwd
    # (`cd Module4/mpc-pipeline && python mpc_api.py`) here keeps that
    # in-process call working without touching module4/config.py.
    os.chdir(REPO_ROOT / "Module4" / "mpc-pipeline")

    port = int(os.environ.get("FORECAST_API_PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), PlatformHandler)
    print("Tank Cascade Platform")
    print(f"  Module 3 forecast API + Module 4 MPC API + frontend all on http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
