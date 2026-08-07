"""
run_daily_cli.py
=================
CLI wrapper around mpc_loop.run_mpc()/save_all() for a single live-Module-3
day. Exists so mpc_api.py can launch a daily run as an isolated subprocess
(the same approach run_simulation.py already uses for the season job),
without adding argparse to main.py itself.

Does NOT change any pipeline logic - this only builds a Config and calls the
existing, unmodified run_mpc()/save_all(), exactly like the one-off snippet
already documented in README.md.

Usage:
    python run_daily_cli.py --date 2026-07-29
    python run_daily_cli.py --date 2026-07-29 --fast-test
"""
import argparse
import json
import os
import sys

from module4.config import Config
from module4.mpc_loop import run_mpc, save_all


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", required=True, help="RUN_DATE, YYYY-MM-DD")
    ap.add_argument("--fast-test", action="store_true",
                    help="shrink NSGA-II (pop=20, gen=10, K=5) for a quick run - "
                         "not for real results")
    args = ap.parse_args()

    overrides = {"POP_SIZE": 20, "N_GEN": 10, "K": 5} if args.fast_test else {}
    cfg = Config(MODULE3_LIVE_FORECAST=True, RUN_DATE=args.date, **overrides).validate()

    results = run_mpc(cfg, season_length=cfg.SEASON_LENGTH, verbose=True)
    paths = save_all(results, cfg)

    wl = results["weights_log"][0] if results["weights_log"] else {}
    dec = results["decisions"][0] if results["decisions"] else None
    feasible = bool(wl.get("feasible", False))

    total_release_m3 = 0.0
    top_releases = []
    n_tanks = 0
    if dec is not None:
        n_tanks = len(dec["tank_ids"])
        total_release_m3 = round(float(sum(dec["R_today"])), 2)
        pairs = sorted(zip(dec["tank_ids"], dec["R_today"]), key=lambda x: -x[1])[:5]
        top_releases = [{"tank_id": t, "release_m3": round(float(r), 2)} for t, r in pairs]

    summary = {
        "date": args.date,
        "fast_test": args.fast_test,
        "feasible": feasible,
        "output_dir": cfg.OUTPUT_DIR,
        "paths": paths,
        "p_drought": wl.get("p_drought"),
        "p_overflow": wl.get("p_overflow"),
        "weights": {
            "shortage": wl.get("w_shortage"),
            "overflow": wl.get("w_overflow"),
            "equity": wl.get("w_equity"),
            "loss": wl.get("w_loss"),
        },
        "c3_tier": wl.get("c3_tier"),
        "reserve_deficit_m3": wl.get("reserve_deficit_m3"),
        "n_tanks": n_tanks,
        "total_release_m3": total_release_m3,
        "top_releases": top_releases,
    }
    status_path = os.path.join(cfg.OUTPUT_DIR, "last_run_status.json")
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
