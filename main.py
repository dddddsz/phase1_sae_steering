"""Unified entry point for Phase 1 SAE steering experiments."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> int:
    kaggle_working = Path("/kaggle/working")
    default_output_dir = (
        kaggle_working / "phase1_sae_steering" / "outputs" / "conditional_ablation"
        if kaggle_working.is_dir()
        else PROJECT_ROOT / "outputs" / "conditional_ablation"
    )

    runner = (
        PROJECT_ROOT
        / "experiments"
        / "conditional_ablation"
        / "run_conditional_ablation.py"
    )
    cmd = [
        sys.executable,
        str(runner),
        "--exp",
        "AB",
        "--output-dir",
        str(default_output_dir),
        *sys.argv[1:],
    ]
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
