from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class SupervisorState:
    command: list[str]
    started_at: str
    deadline_at: str
    hours: float
    status: str
    pid: int | None = None
    returncode: int | None = None


def _atomic_json(path: Path, state: SupervisorState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(asdict(state), indent=2, sort_keys=True))
    os.replace(temporary, path)


def supervise(command: list[str], hours: float, state_path: Path) -> int:
    """Run a resumable experiment command with a hard wall-clock limit.

    The child owns algorithm-level checkpoints. This supervisor records lifecycle state,
    forwards termination, and never reports a timed-out or failed child as completed.
    """
    if not command:
        raise ValueError("experiment command is required")
    if hours <= 0:
        raise ValueError("hours must be positive")
    start_wall = datetime.now(timezone.utc)
    deadline_wall = start_wall.timestamp() + hours * 3600
    state = SupervisorState(
        command=command,
        started_at=start_wall.isoformat(),
        deadline_at=datetime.fromtimestamp(deadline_wall, timezone.utc).isoformat(),
        hours=hours,
        status="starting",
    )
    _atomic_json(state_path, state)
    process = subprocess.Popen(command, start_new_session=True)
    state.pid = process.pid
    state.status = "running"
    _atomic_json(state_path, state)

    stopping = False

    def stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)

    old_term = signal.signal(signal.SIGTERM, stop)
    old_int = signal.signal(signal.SIGINT, stop)
    try:
        while process.poll() is None:
            if stopping:
                break
            if time.time() >= deadline_wall:
                state.status = "deadline_reached"
                _atomic_json(state_path, state)
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(min(30, max(0.1, deadline_wall - time.time())))
        returncode = process.wait()
    finally:
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)
    state.returncode = returncode
    if state.status != "deadline_reached":
        state.status = "completed" if returncode == 0 else ("terminated" if stopping else "failed")
    _atomic_json(state_path, state)
    return returncode

