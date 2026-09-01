from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .models import AcquisitionTask, DeploymentTask


def snapshot(root: Path) -> dict[str, Any]:
    files = {str(path.relative_to(root)): path.read_text(errors="replace") for path in root.rglob("*") if path.is_file() and path.name != "Dockerfile"}
    modes = {str(path.relative_to(root)): path.stat().st_mode & 0o777 for path in root.rglob("*") if path.is_file() and path.name != "Dockerfile"}
    return {"files": files, "modes": modes}


@dataclass(frozen=True)
class HostNativeVerifier:
    tests_dir: Path
    base_environment: Path
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
            if isinstance(output, dict) and "delta" in output:
                shutil.copytree(self.base_environment, root, ignore=shutil.ignore_patterns("Dockerfile"))
                files, modes = output["delta"], output.get("modes", {})
                for relative in output.get("deleted", []):
                    target = root / relative
                    if target.exists(): target.unlink()
            else:
                root.mkdir()
            logs.mkdir()
            for relative, content in files.items():
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(str(content))
                if relative in modes:
                    target.chmod(int(modes[relative]))
            self._prepare_dependencies(root)
            env = os.environ.copy()
            env.update(PROJECT_ROOT=str(root), HARBOR_LOG_DIR=str(logs), PYTHON_BIN="python3")
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


def _materialize_expert_artifact(task_dir: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="isl-dual-outcome-") as temp:
        root = Path(temp) / "task"
        shutil.copytree(task_dir / "environment", root, ignore=shutil.ignore_patterns("Dockerfile"))
        before = snapshot(root)
        env = os.environ.copy(); env["PROJECT_ROOT"] = str(root); env["PYTHON_BIN"] = "python3"
        completed = subprocess.run(["bash", str(task_dir / "solution" / "solve.sh")], env=env, cwd=root, text=True, capture_output=True, timeout=600)
        if completed.returncode != 0:
            raise RuntimeError(f"expert artifact materialization failed for {task_dir.name}: {completed.stderr[-2000:]}")
        after = snapshot(root)
        # Outcome representation intentionally contains no command log or solution script.
        changed = {path: content for path, content in after["files"].items() if before["files"].get(path) != content or before["modes"].get(path) != after["modes"].get(path)}
        deleted = sorted(set(before["files"]) - set(after["files"]))
        return {"delta": changed, "modes": {path: after["modes"][path] for path in changed}, "deleted": deleted}


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
        verifier = HostNativeVerifier(task_dir / "tests", task_dir / "environment")
        workspace = str((task_dir / "environment").resolve())
        if index <= 3:
            cache_path = artifact_cache / f"{spec['task_id']}.json" if artifact_cache else None
            if cache_path and cache_path.exists():
                artifact = json.loads(cache_path.read_text())
            else:
                artifact = _materialize_expert_artifact(task_dir) if materialize_artifacts else {}
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
