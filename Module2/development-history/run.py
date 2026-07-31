#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DEFAULT_NOTEBOOKS = [
    "01_phase1_graph.ipynb",
    "02_phase2_gnn.ipynb",
    "03_phase3_outputs.ipynb",
    "04_phase4_runoff_coefficient.ipynb",
]


def run_notebook(notebook_name: str, timeout: int) -> None:
    notebook_path = NOTEBOOKS_DIR / notebook_name
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    print(f"\n===== Running {notebook_name} =====")
    command = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        f"--ExecutePreprocessor.timeout={timeout}",
        str(notebook_path),
    ]

    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    print(f"===== Completed {notebook_name} =====")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the cascade_gnn notebooks sequentially")
    parser.add_argument(
        "--notebooks",
        nargs="+",
        default=DEFAULT_NOTEBOOKS,
        help="Notebook names to run. Defaults to all project notebooks.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout in seconds for each notebook execution (default: 3600).",
    )
    args = parser.parse_args()

    notebooks = args.notebooks
    if notebooks == ["all"]:
        notebooks = DEFAULT_NOTEBOOKS

    for notebook_name in notebooks:
        run_notebook(notebook_name, args.timeout)

    print("\nAll requested notebooks completed successfully.")


if __name__ == "__main__":
    main()
