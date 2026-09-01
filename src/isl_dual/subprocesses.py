from __future__ import annotations

import os
import signal
import subprocess
import threading
from collections.abc import Sequence


def run_process_group(command: Sequence[str], *, timeout: int, cwd: str | None = None, env: dict[str, str] | None = None, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run a subprocess in its own group and guarantee descendant cleanup on timeout."""
    process = subprocess.Popen(
        list(command), cwd=cwd, env=env, text=True,
        stdin=subprocess.PIPE if input_text is not None else None,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
    )
    old_handlers: dict[signal.Signals, object] = {}
    if threading.current_thread() is threading.main_thread():
        def _interrupt(signum: int, _frame: object) -> None:
            if signum == signal.SIGINT:
                raise KeyboardInterrupt
            raise SystemExit(128 + signum)
        for handled_signal in (signal.SIGTERM, signal.SIGINT):
            old_handlers[handled_signal] = signal.getsignal(handled_signal)
            signal.signal(handled_signal, _interrupt)
    try:
        try:
            stdout, stderr = process.communicate(input=input_text, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                stdout, stderr = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
            raise subprocess.TimeoutExpired(command, timeout, output=stdout, stderr=stderr) from error
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise
    finally:
        for handled_signal, old_handler in old_handlers.items():
            signal.signal(handled_signal, old_handler)
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
