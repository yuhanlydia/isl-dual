from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from .leakage import SecretBundle, assert_deployment_input, assert_forward_input
from .models import AcquisitionTask, DeploymentTask, Graph
from .subprocesses import run_process_group


class CodexExecutionError(RuntimeError):
    pass


class CodexInfrastructureError(CodexExecutionError):
    """Remote/service failure that must not be interpreted as task reward 0."""


class _TransientCodexError(CodexInfrastructureError):
    pass


class CodexExecutor:
    """Execute every plan in an isolated directory and ephemeral Codex session.

    Workspace state remains isolated per rollout.  Only package-download state and
    the fact that an identical Python requirements set has already been installed
    are shared, which removes repeated dependency setup without sharing task files
    or agent-produced artifacts.
    """

    _python_install_lock = threading.RLock()
    _installed_python_requirements: set[str] = set()
    _transient_markers = (
        "429",
        "rate limit",
        "rate_limit",
        "too many requests",
        "temporarily unavailable",
        "service unavailable",
        "overloaded",
        "connection reset",
        "connection refused",
        "connection error",
        "network error",
        "try again",
    )

    def __init__(
        self,
        timeout_seconds: int = 900,
        model: str | None = None,
        dependency_cache: Path | None = None,
        max_retries: int = 2,
    ):
        self.timeout_seconds = timeout_seconds
        self.model = model
        self.max_retries = max(0, int(max_retries))
        self.dependency_cache = Path(
            dependency_cache
            or os.environ.get(
                "ISL_DUAL_DEPENDENCY_CACHE",
                str(Path(tempfile.gettempdir()) / "isl-dual-dependency-cache"),
            )
        )
        self.dependency_cache.mkdir(parents=True, exist_ok=True)
        self._pip_cache = self.dependency_cache / "pip"
        self._npm_cache = self.dependency_cache / "npm"
        self._pip_cache.mkdir(parents=True, exist_ok=True)
        self._npm_cache.mkdir(parents=True, exist_ok=True)

    def execute(
        self,
        task: AcquisitionTask | DeploymentTask,
        graph: Graph,
        plan: tuple[str, ...],
    ) -> dict[str, Any]:
        """Run one rollout, retrying only infrastructure-like Codex failures.

        Every retry recreates the temporary workspace from the original task source,
        so a partially failed attempt can never leak filesystem state into the next
        attempt. Scientific/model failures returned by the native verifier are not
        retried here. If all retries are exhausted, raise CodexInfrastructureError so
        MCTS/campaign orchestration can checkpoint and stop rather than writing a
        scientifically false zero reward.
        """
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._execute_once(task, graph, plan)
            except _TransientCodexError as error:
                last_error = error
                if attempt >= self.max_retries:
                    break
                time.sleep(min(8.0, 2.0 * (2 ** attempt)))
        raise CodexInfrastructureError(
            f"transient Codex execution failed after {self.max_retries + 1} attempts: "
            f"{last_error}"
        ) from last_error

    def _execute_once(
        self,
        task: AcquisitionTask | DeploymentTask,
        graph: Graph,
        plan: tuple[str, ...],
    ) -> dict[str, Any]:
        selected_nodes = [graph.node_map()[node_id] for node_id in plan]
        procedural_plan = [{
            "name": node.name,
            "preconditions": list(node.preconditions),
            "input_requirements": list(node.inputs),
            "action": node.action,
            "expected_outputs": list(node.outputs),
            "validator": node.validator,
            "required": node.required,
        } for node in selected_nodes]
        payload = {"task": task.x, "procedural_plan": procedural_plan}
        if isinstance(task, AcquisitionTask):
            assert_forward_input(payload, SecretBundle(expert_artifact=task.expert_artifact))
        else:
            assert set(payload) == {"task", "procedural_plan"}
            assert_deployment_input(payload, SecretBundle())
        prompt = (
            "ISOLATED WORKSPACE MAPPING:\nThe current working directory is this rollout's virtual /root/task. "
            "Resolve every task reference under /root/task relative to the current working directory; "
            "do not access a literal host /root/task path.\n\nTASK:\n" + task.x + "\n\nPROCEDURAL PLAN:\n" +
            json.dumps(procedural_plan, indent=2, ensure_ascii=False) +
            "\n\nFollow the plan as procedural guidance. Use tools and observable environment "
            "feedback as normally allowed. Do not assume access to any expert solution. "
            "Complete the task in the current workspace."
        )
        with tempfile.TemporaryDirectory(prefix="isl-dual-rollout-") as temp:
            workspace = Path(temp) / "workspace"
            tool_bin = Path(temp) / "bin"
            tool_bin.mkdir()
            (tool_bin / "python").symlink_to("/usr/bin/python3")
            if task.workspace_source:
                shutil.copytree(
                    task.workspace_source,
                    workspace,
                    ignore=shutil.ignore_patterns("Dockerfile"),
                )
            else:
                workspace.mkdir()
            self._prepare_dependencies(workspace)
            output_file = Path(temp) / "last-message.txt"
            command = [
                "codex", "exec", "--ephemeral", "--skip-git-repo-check",
                "--dangerously-bypass-approvals-and-sandbox",
                "-C", str(workspace), "-o", str(output_file),
            ]
            if self.model:
                command.extend(["--model", self.model])
            command.append("-")
            process_env = os.environ.copy()
            process_env["PATH"] = str(tool_bin) + os.pathsep + process_env.get("PATH", "")
            process_env["PIP_NO_CACHE_DIR"] = "1"
            process_env["npm_config_cache"] = str(self._npm_cache)
            try:
                completed = run_process_group(
                    command,
                    timeout=self.timeout_seconds,
                    env=process_env,
                    input_text=prompt,
                )
            except subprocess.TimeoutExpired as error:
                raise _TransientCodexError(
                    f"ephemeral Codex execution timed out after {self.timeout_seconds}s"
                ) from error
            if completed.returncode != 0:
                stderr = completed.stderr[-4000:]
                if self._is_transient_failure(stderr):
                    raise _TransientCodexError(stderr)
                raise CodexExecutionError(stderr)
            files = [
                p
                for p in workspace.rglob("*")
                if p.is_file()
                and not any(
                    part in {
                        "node_modules", ".npm-cache", ".poetry_env", ".venv",
                        ".git", ".pytest_cache", "__pycache__",
                    }
                    for part in p.relative_to(workspace).parts
                )
            ]
            return {
                "workspace": {
                    str(p.relative_to(workspace)): self._read_artifact(p)
                    for p in files
                },
                "modes": {
                    str(p.relative_to(workspace)): p.stat().st_mode & 0o777
                    for p in files
                },
                "message": output_file.read_text(errors="replace")
                if output_file.exists()
                else "",
            }

    @classmethod
    def _is_transient_failure(cls, message: str) -> bool:
        lowered = message.lower()
        return any(marker in lowered for marker in cls._transient_markers)

    @staticmethod
    def _read_artifact(path: Path) -> Any:
        raw = path.read_bytes()
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return {
                "encoding": "base64",
                "data": base64.b64encode(raw).decode("ascii"),
            }

    def _prepare_dependencies(self, workspace: Path) -> None:
        requirements = workspace / "requirements.txt"
        if requirements.exists():
            digest = hashlib.sha256(requirements.read_bytes()).hexdigest()
            # pip installs into the process environment, so installing an
            # identical requirements set more than once per runner process is
            # pure repeated work. Serialize the first install to avoid three
            # concurrent trees racing the same environment.
            with self._python_install_lock:
                if digest not in self._installed_python_requirements:
                    env = os.environ.copy()
                    env["PIP_CACHE_DIR"] = str(self._pip_cache)
                    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
                    subprocess.run(
                        [
                            "python3", "-m", "pip", "install",
                            "-r", str(requirements),
                        ],
                        cwd=workspace,
                        env=env,
                        text=True,
                        capture_output=True,
                        timeout=self.timeout_seconds,
                        check=True,
                    )
                    self._installed_python_requirements.add(digest)

        if (workspace / "package-lock.json").exists():
            # node_modules remains rollout-local. Only npm's content-addressed
            # download cache is shared, preserving filesystem isolation while
            # avoiding repeated network/package fetches.
            env = os.environ.copy()
            env["npm_config_cache"] = str(self._npm_cache)
            subprocess.run(
                [
                    "npm", "ci", "--ignore-scripts", "--prefer-offline",
                    "--no-audit", "--no-fund",
                ],
                cwd=workspace,
                env=env,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=True,
            )
