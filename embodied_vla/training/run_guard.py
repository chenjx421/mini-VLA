from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path


@contextmanager
def claim_run_directory(
    output_dir: Path,
    *,
    resume: bool = False,
) -> Iterator[None]:
    """Exclusively claim a new or explicitly resumed experiment directory.

    Refusing to append to an existing run protects metrics and checkpoints from
    accidental duplicate trainers. A killed process leaves its lock behind. An
    explicit resume may replace that lock only after its recorded PID is gone.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [path.name for path in output_dir.iterdir()]
    if existing and not resume:
        entries = ", ".join(sorted(existing)[:5])
        raise FileExistsError(
            f"experiment directory is not empty: {output_dir} ({entries}); "
            "choose a new --output-dir"
        )

    lock_path = output_dir / ".run.lock"
    if resume and lock_path.exists():
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        pid = int(payload["pid"])
        if _pid_is_running(pid):
            raise RuntimeError(
                f"cannot resume active experiment {output_dir}: PID {pid} is running"
            )
        lock_path.unlink()

    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"experiment directory is already claimed: {output_dir}") from error

    try:
        payload = {
            "pid": os.getpid(),
            "started_at": datetime.now(UTC).isoformat(),
        }
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True)
        yield
    except BaseException:
        raise
    else:
        lock_path.unlink()


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        process_query_limited_information = 0x1000
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(
            process_query_limited_information,
            False,
            pid,
        )
        if not handle:
            return False
        kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
