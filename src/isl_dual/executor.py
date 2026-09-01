from __future__ import annotations

import json
import base64
import shutil
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Any

from .leakage import SecretBundle, assert_forward_input
from .models import AcquisitionTask, DeploymentTask, Graph
from .subprocesses import run_process_group


class CodexExecutionError(RuntimeError):
    pass


class CodexExecutor:
    """Execute every plan in an isolated directory and ephemeral Codex session."""

    def __init__(self, timeout_seconds: int = 900, model: str | None = None):
        self.timeout_seconds = timeout_seconds
        self.model = model

    def execute(
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
        prompt = (
            "TASK:\n" + task.x + "\n\nPROCEDURAL PLAN:\n" +
            json.dumps(procedural_plan, indent=2, ensure_ascii=False) +
            "\n\nFollow the plan as procedural guidance. Use tools and observable environment "
            "feedback as normally allowed. Do not assume access to any expert solution. "
            "Complete the task in the current workspace."
        )
        with tempfile.TemporaryDirectory(prefix="isl-dual-rollout-") as temp:
            workspace = Path(temp) / "workspace"
            tool_bin = Path(temp) / "bin"; tool_bin.mkdir(); (tool_bin / "python").symlink_to("/usr/bin/python3")
            if task.workspace_source:
                shutil.copytree(task.workspace_source, workspace, ignore=shutil.ignore_patterns("Dockerfile"))
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
            try:
                completed = run_process_group(command, timeout=self.timeout_seconds, env=process_env, input_text=prompt)
            except subprocess.TimeoutExpired as error:
                raise CodexExecutionError(f"ephemeral Codex execution timed out after {self.timeout_seconds}s") from error
            if completed.returncode != 0:
                raise CodexExecutionError(completed.stderr[-4000:])
            files = [p for p in workspace.rglob("*") if p.is_file() and not any(part in {"node_modules", ".npm-cache", ".poetry_env", ".git", ".pytest_cache", "__pycache__"} for part in p.relative_to(workspace).parts)]
            return {
                "workspace": {str(p.relative_to(workspace)): self._read_artifact(p) for p in files},
                "modes": {str(p.relative_to(workspace)): p.stat().st_mode & 0o777 for p in files},
                "message": output_file.read_text(errors="replace") if output_file.exists() else "",
            }

    @staticmethod
    def _read_artifact(path: Path) -> Any:
        raw = path.read_bytes()
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return {"encoding": "base64", "data": base64.b64encode(raw).decode("ascii")}

    def _prepare_dependencies(self, workspace: Path) -> None:
        requirements = workspace / "requirements.txt"
        if requirements.exists():
            subprocess.run(["python3", "-m", "pip", "install", "-r", str(requirements)], cwd=workspace, text=True, capture_output=True, timeout=self.timeout_seconds, check=True)
        if (workspace / "package-lock.json").exists():
            subprocess.run(["npm", "ci", "--ignore-scripts"], cwd=workspace, text=True, capture_output=True, timeout=self.timeout_seconds, check=True)
