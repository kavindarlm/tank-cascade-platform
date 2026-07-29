#!/usr/bin/env python3
"""Launch the project notebooks with Jupyter from the repository root."""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path


def find_project_root() -> Path:
    return Path(__file__).resolve().parent


def discover_notebooks(project_root: Path) -> list[Path]:
    notebooks_dir = project_root / "notebooks"
    if not notebooks_dir.exists():
        return []

    notebooks = sorted(
        path for path in notebooks_dir.rglob("*.ipynb") if path.is_file() and "lib" not in path.parts
    )
    return notebooks


def build_server_command(project_root: Path, args: argparse.Namespace) -> list[str]:
    if importlib.util.find_spec("jupyterlab") is not None:
        command = [sys.executable, "-m", "jupyterlab"]
    elif importlib.util.find_spec("notebook") is not None:
        command = [sys.executable, "-m", "notebook"]
    else:
        raise RuntimeError(
            "Jupyter is not installed in this environment. Install it with 'pip install -r requirements.txt'."
        )

    command.extend([
        "--ip",
        args.host,
        "--port",
        str(args.port),
        "--ServerApp.root_dir",
        str(project_root),
        "--ServerApp.token=",
        "--ServerApp.password=",
    ])

    if args.no_browser:
        command.append("--no-browser")
    elif args.browser:
        command.append("--browser")

    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the project notebooks with Jupyter")
    parser.add_argument("--host", default="127.0.0.1", help="Hostname for the Jupyter server")
    parser.add_argument("--port", type=int, default=8888, help="Port for the Jupyter server")
    parser.add_argument("--open", help="Optional notebook path relative to the project root")
    parser.add_argument("--browser", action="store_true", help="Allow Jupyter to open a browser")
    parser.add_argument("--no-browser", action="store_true", help="Disable opening a browser automatically")
    parser.add_argument("--dry-run", action="store_true", help="Print the launch command without starting the server")
    return parser.parse_args()


def main() -> int:
    project_root = find_project_root()
    notebooks = discover_notebooks(project_root)

    args = parse_args()

    if not notebooks:
        print("No notebooks were found under the notebooks/ directory.")
        return 1

    print(f"Project root: {project_root}")
    print("Discovered notebooks:")
    for notebook in notebooks:
        print(f"- {notebook.relative_to(project_root).as_posix()}")

    if args.open:
        print(f"Requested notebook: {args.open}")

    command = build_server_command(project_root, args)
    print("Launch command:")
    print(" ".join(command))

    if args.dry_run:
        return 0

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(project_root))

    print(f"Starting Jupyter server on http://{args.host}:{args.port}")
    subprocess.Popen(command, cwd=str(project_root), env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
