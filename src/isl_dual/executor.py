from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .leakage import SecretBundle, assert_forward_input
from .models import AcquisitionTask, DeploymentTask, Graph


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
        actions = [graph.node_map()[node_id].action for node_id in plan]
        payload = {"task": task.x, "procedural_plan": actions}
        if isinstance(task, AcquisitionTask):
            assert_forward_input(payload, SecretBundle(expert_artifact=task.expert_artifact))
        prompt = (
            "TASK:\n" + task.x + "\n\nPROCEDURAL PLAN:\n" +
            "\n".join(f"{i}. {action}" for i, action in enumerate(actions, 1)) +
            "\n\nFollow the plan as procedural guidance. Use tools and observable environment "
            "feedback as normally allowed. Do not assume access to any expert solution. "
            "Complete the task in the current workspace."
        )
        with tempfile.TemporaryDirectory(prefix="isl-dual-rollout-") as temp:
            workspace = Path(temp) / "workspace"
            if task.workspace_source:
                shutil.copytree(task.workspace_source, workspace)
            else:
                workspace.mkdir()
            output_file = Path(temp) / "last-message.txt"
            command = [
                "codex", "exec", "--ephemeral", "--skip-git-repo-check",
                "--dangerously-bypass-approvals-and-sandbox",
                "-C", str(workspace), "-o", str(output_file),
            ]
            if self.model:
                command.extend(["--model", self.model])
            command.append(prompt)
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            if completed.returncode != 0:
                raise CodexExecutionError(completed.stderr[-4000:])
            return {
                "workspace": {str(p.relative_to(workspace)): p.read_text(errors="replace")
                              for p in workspace.rglob("*") if p.is_file()},
                "message": output_file.read_text(errors="replace") if output_file.exists() else "",
            }

