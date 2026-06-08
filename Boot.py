"""Convenience launcher for common project commands.

Examples:
    python Boot.py app
    python Boot.py train -- 10 --unfreeze
    python Boot.py diagnostics
    python Boot.py augment
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VENV_ACTIVATE = Path.home() / "unet-env" / "bin" / "activate"
COMMANDS = {
    "app": ["streamlit", "run", "app.py"],
    "train": [sys.executable, "train.py"],
    "diagnostics": [sys.executable, "diag_scr.py"],
    "augment": [sys.executable, "augment_data.py"],
}


def _with_venv(command: list[str]) -> list[str]:
    """Run via the legacy ~/unet-env activation script when it exists."""
    if not VENV_ACTIVATE.exists():
        return command

    quoted_command = " ".join(subprocess.list2cmdline([part]) for part in command)
    return [
        "bash",
        "-lc",
        f"source {subprocess.list2cmdline([str(VENV_ACTIVATE)])} && {quoted_command}",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Launcher for ML_Nvidiav0.2 workflows")
    parser.add_argument("command", choices=COMMANDS, help="Workflow to run")
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Extra arguments for the selected command. Use '--' before them if needed.",
    )
    parsed = parser.parse_args()

    extra_args = parsed.args[1:] if parsed.args[:1] == ["--"] else parsed.args
    command = COMMANDS[parsed.command] + extra_args
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(BASE_DIR))

    completed = subprocess.run(_with_venv(command), cwd=BASE_DIR, env=env, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
