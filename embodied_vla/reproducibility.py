from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch


def runtime_metadata(*, device: str | torch.device | None = None) -> dict[str, Any]:
    """Capture the execution context needed to interpret an experiment."""

    return {
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "command": [str(argument) for argument in sys.argv],
        "working_directory": str(Path.cwd().resolve()),
        "git_commit": _git_commit(),
        "python": sys.version,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "processor": platform.processor() or platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "packages": {
            "numpy": np.__version__,
            "torch": torch.__version__,
            "mujoco": mujoco.__version__,
        },
        "torch": {
            "device": str(device) if device is not None else None,
            "intraop_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "cuda_available": torch.cuda.is_available(),
        },
    }


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None
