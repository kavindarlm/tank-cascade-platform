"""
main.py
======
Entry point for Module 4 - MO-MPC tank cascade decision support.

Usage:
    python main.py

This runs the full receding-horizon MPC loop over one irrigation season using
the input CSVs in data/. It expects all required data files to be present (the
data_loader raises a clear error if any are missing).

Mind Mappers - Group 16, University of Moratuwa, 2026.
"""

from module4.config import Config
from module4.mpc_loop import run_mpc, save_all


def main():
    print("=" * 66)
    print("  Module 4 - Multi-Objective Model Predictive Control")
    print("  Integrated Digital Management of Tank Cascade Systems")
    print("  Mind Mappers - Group 16  |  University of Moratuwa  |  2026")
    print("=" * 66)

    cfg = Config().validate()

    # Run a short season by default; increase for a full run.
    results = run_mpc(cfg, season_length=cfg.SEASON_LENGTH, verbose=True)

    paths = save_all(results, cfg)

    print("\n" + "=" * 66)
    print("  Season simulation complete.")
    print(f"  Days with a feasible decision: "
          f"{sum(1 for w in results['weights_log'] if w.get('feasible'))}")
    print(f"  Outputs written:")
    for name, p in paths.items():
        print(f"    - {name}: {p}")
    print("=" * 66)


if __name__ == "__main__":
    main()
