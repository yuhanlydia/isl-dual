from __future__ import annotations

import json
import base64
import os
import shutil
import subprocess
import tempfile
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
import tomli

from .models import AcquisitionTask, DeploymentTask

EPHEMERAL_PARTS = {"node_modules", ".npm-cache", ".poetry_env", "__pycache__", ".pytest_cache"}


def _observable_path(path: str) -> bool:
    return not (set(Path(path).parts) & EPHEMERAL_PARTS) and "$PROJECT_ROOT" not in path and '"' not in path


def snapshot(root: Path) -> dict[str, Any]:
    files = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.name == "Dockerfile":
            continue
        raw = path.read_bytes()
        try:
            value: Any = raw.decode("utf-8")
        except UnicodeDecodeError:
            value = {"encoding": "base64", "data": base64.b64encode(raw).decode("ascii")}
        files[str(path.relative_to(root))] = value
    modes = {str(path.relative_to(root)): path.stat().st_mode & 0o777 for path in root.rglob("*") if path.is_file() and path.name != "Dockerfile"}
    return {"files": files, "modes": modes}


@dataclass(frozen=True)
class HostNativeVerifier:
    tests_dir: Path
    base_environment: Path
    project_relative: str = "."
    timeout_seconds: int = 600

    def __call__(self, output: Any) -> float:
        return self.evaluate(output)[0]

    def evaluate(self, output: Any) -> tuple[float, str | None]:
        if isinstance(output, dict) and "workspace" in output:
            files, modes = output["workspace"], output.get("modes", {})
        elif isinstance(output, dict) and "files" in output:
            files, modes = output["files"], output.get("modes", {})
        else:
            files, modes = output, {}
        if not isinstance(files, dict):
            raise TypeError("host verifier expects a workspace snapshot")
        with tempfile.TemporaryDirectory(prefix="isl-dual-verify-") as temp:
            root, logs = Path(temp) / "task", Path(temp) / "logs"
            tool_bin = Path(temp) / "bin"; tool_bin.mkdir(); (tool_bin / "python").symlink_to("/usr/bin/python3")
            if isinstance(output, dict) and "delta" in output:
                shutil.copytree(self.base_environment, root, ignore=shutil.ignore_patterns("Dockerfile"))
                files = output.get("_replay_delta", output["delta"])
                modes = output.get("_replay_modes", output.get("modes", {}))
                for relative in output.get("_replay_deleted", output.get("deleted", [])):
                    target = root / relative
                    if target.exists(): target.unlink()
            else:
                root.mkdir()
            logs.mkdir()
            for relative, content in files.items():
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(content, dict) and content.get("encoding") == "base64":
                    target.write_bytes(base64.b64decode(content["data"]))
                else:
                    target.write_text(str(content))
                if relative in modes:
                    target.chmod(int(modes[relative]))
            self._prepare_dependencies(root)
            env = os.environ.copy()
            env.update(PROJECT_ROOT=str((root / self.project_relative).resolve()), TASK_ROOT=str(root), HARBOR_LOG_DIR=str(logs), PYTHON_BIN="python3", PATH=str(tool_bin) + os.pathsep + env["PATH"])
            completed = subprocess.run(["bash", str(self.tests_dir / "test.sh")], env=env, text=True, capture_output=True, timeout=self.timeout_seconds)
            reward_path = logs / "reward.txt"
            if not reward_path.exists():
                return 0.0, "native verifier did not produce reward.txt: " + completed.stderr[-1000:]
            reward = max(0.0, min(1.0, float(reward_path.read_text().strip())))
            failures: list[str] = []
            for report_name in ("outcome_report.json", "process_report.json"):
                report_path = logs / report_name
                if not report_path.exists():
                    continue
                report = json.loads(report_path.read_text())
                for section in ("public", "hidden"):
                    for item in (report.get(section) or {}).get("results", []):
                        if not item.get("passed", False):
                            failures.append(f"{report_name}:{item.get('name', '?')}: {item.get('error') or item.get('detail') or ''}")
            summary = "\n".join(failures[:20]) or (None if reward >= 1.0 else completed.stderr[-2000:] or "verifier reward below 1 without detailed failures")
            return reward, summary

    def _prepare_dependencies(self, root: Path) -> None:
        requirements = root / "requirements.txt"
        if requirements.exists():
            subprocess.run(["python3", "-m", "pip", "install", "-r", str(requirements)], cwd=root, text=True, capture_output=True, timeout=self.timeout_seconds, check=True)
        if (root / "package-lock.json").exists():
            subprocess.run(["npm", "ci", "--ignore-scripts"], cwd=root, text=True, capture_output=True, timeout=self.timeout_seconds, check=True)


@dataclass(frozen=True)
class FamilyBundle:
    family_id: str
    acquisition: list[AcquisitionTask]
    deployment: list[DeploymentTask]


def _project_relative(task_dir: Path) -> str:
    task_toml = tomli.loads((task_dir / "task.toml").read_text())
    configured = str(task_toml.get("verifier", {}).get("env", {}).get("PROJECT_ROOT", "/root/task"))
    prefix = "/root/task"
    if configured == prefix:
        return "."
    if configured.startswith(prefix + "/"):
        return configured[len(prefix) + 1:]
    raise ValueError(f"unsupported PROJECT_ROOT outside /root/task: {configured}")


def _solution_project_relative(task_dir: Path) -> str:
    text = (task_dir / "solution" / "solve.sh").read_text()
    match = re.search(r'PROJECT_ROOT="\$\{PROJECT_ROOT:-(/root/task(?:/[^}]*)?)\}"', text)
    if not match or match.group(1) == "/root/task":
        return "."
    return match.group(1)[len("/root/task/"):]


def _materialize_expert_artifact(task_dir: Path, project_relative: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="isl-dual-outcome-") as temp:
        root = Path(temp) / "task"
        tool_bin = Path(temp) / "bin"; tool_bin.mkdir(); (tool_bin / "python").symlink_to("/usr/bin/python3")
        shutil.copytree(task_dir / "environment", root, ignore=shutil.ignore_patterns("Dockerfile"))
        before = snapshot(root)
        env = os.environ.copy(); env["PROJECT_ROOT"] = str((root / project_relative).resolve()); env["TASK_ROOT"] = str(root); env["PYTHON_BIN"] = "python3"; env["PATH"] = str(tool_bin) + os.pathsep + env["PATH"]
        completed = subprocess.run(["bash", str(task_dir / "solution" / "solve.sh")], env=env, cwd=root, text=True, capture_output=True, timeout=600)
        if completed.returncode != 0:
            raise RuntimeError(f"expert artifact materialization failed for {task_dir.name}: {completed.stderr[-2000:]}")
        after = snapshot(root)
        # Outcome representation intentionally contains no command log or solution script.
        replay_changed = {path: content for path, content in after["files"].items() if before["files"].get(path) != content or before["modes"].get(path) != after["modes"].get(path)}
        changed = {path: content for path, content in replay_changed.items() if _observable_path(path)}
        deleted = sorted(set(before["files"]) - set(after["files"]))
        return {
            "delta": changed,
            "modes": {path: after["modes"][path] for path in changed},
            "deleted": [path for path in deleted if _observable_path(path)],
            "_replay_delta": replay_changed,
            "_replay_modes": {path: after["modes"][path] for path in replay_changed},
            "_replay_deleted": deleted,
        }


def load_family(benchmark_root: Path, family_id: str, materialize_artifacts: bool = True, artifact_cache: Path | None = None) -> FamilyBundle:
    task_root = benchmark_root / "benchmark" / "tasks"
    records: list[tuple[int, Path, dict[str, Any]]] = []
    for task_dir in task_root.iterdir():
        spec_path = task_dir / "task-spec.yaml"
        if not spec_path.exists():
            continue
        spec = yaml.safe_load(spec_path.read_text())
        if spec.get("family_id") == family_id:
            records.append((int(spec["task_index"]), task_dir, spec))
    records.sort()
    if [index for index, _, _ in records] != [1, 2, 3, 4, 5, 6]:
        raise ValueError(f"family {family_id} does not contain exactly T1-T6")
    acquisition: list[AcquisitionTask] = []
    deployment: list[DeploymentTask] = []
    for index, task_dir, spec in records:
        instruction = (task_dir / "instruction.md").read_text()
        project_relative = _project_relative(task_dir)
        solution_relative = _solution_project_relative(task_dir)
        verifier = HostNativeVerifier(task_dir / "tests", task_dir / "environment", project_relative)
        workspace = str((task_dir / "environment").resolve())
        if index <= 3:
            cache_path = artifact_cache / f"{spec['task_id']}.json" if artifact_cache else None
            if cache_path and cache_path.exists():
                artifact = json.loads(cache_path.read_text())
            else:
                artifact = _materialize_expert_artifact(task_dir, solution_relative) if materialize_artifacts else {}
                if cache_path:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = cache_path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True))
                    os.replace(temporary, cache_path)
            acquisition.append(AcquisitionTask(spec["task_id"], instruction, artifact, verifier, workspace))
        else:
            deployment.append(DeploymentTask(spec["task_id"], instruction, verifier, workspace))
    return FamilyBundle(family_id, acquisition, deployment)


def audit_no_curated_access(benchmark_root: Path, payload: str) -> None:
    skills_root = benchmark_root / "benchmark" / "skills"
    for skill_file in skills_root.glob("*/SKILL.md"):
        text = skill_file.read_text()
        if text and text in payload:
            raise AssertionError(f"curated skill leaked from {skill_file}")
