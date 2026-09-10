from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


_SUMMARY_RE = re.compile(r"^Overall:\s*(\d+)\s*/\s*(\d+)\s*\(([0-9.]+)%\)\s*$", re.MULTILINE)
_NUMERIC_SUFFIX_RE = re.compile(r"-(\d+)$")


@dataclass(frozen=True)
class BenchmarkRun:
    command: tuple[str, ...]
    passed: int | None
    total: int | None
    score: float | None
    stdout: str
    stderr: str
    dry_run: bool = False


def parse_skilllearn_summary(stdout: str) -> tuple[int, int, float]:
    matches = list(_SUMMARY_RE.finditer(stdout))
    if not matches:
        raise ValueError("SkillLearnBench output does not contain an Overall summary")
    match = matches[-1]
    passed = int(match.group(1))
    total = int(match.group(2))
    if total <= 0:
        raise ValueError("SkillLearnBench Overall summary has zero trials")
    return passed, total, passed / total


def _numeric_instance_key(path: Path) -> tuple[int, str]:
    match = _NUMERIC_SUFFIX_RE.search(path.name)
    if match is None:
        return (10**9, path.name)
    return (int(match.group(1)), path.name)


def discover_skilllearn_instances(root: Path, task: str) -> list[str]:
    task_root = Path(root) / "tasks" / task
    if not task_root.is_dir():
        raise FileNotFoundError(f"SkillLearnBench task directory is missing: {task_root}")
    candidates = [
        child
        for child in task_root.iterdir()
        if child.is_dir() and (child / "instruction.md").is_file()
    ]
    candidates.sort(key=_numeric_instance_key)
    if not candidates:
        raise ValueError(f"SkillLearnBench task has no query instances: {task}")
    return [f"{task}/{child.name}" for child in candidates]


class SkillLearnBenchAdapter:
    def __init__(self, root: Path, *, python_executable: str | None = None) -> None:
        self.root = Path(root).resolve()
        self.python_executable = python_executable or sys.executable

    def preflight(self, tasks: Sequence[str]) -> None:
        script = self.root / "evaluate_skills.py"
        if not script.is_file():
            raise FileNotFoundError(f"evaluate_skills.py is missing under {self.root}")
        if not (self.root / "tasks").is_dir():
            raise FileNotFoundError(f"tasks directory is missing under {self.root}")
        for task in tasks:
            discover_skilllearn_instances(self.root, task)

    def build_command(
        self,
        task_instances: Sequence[str],
        *,
        skill_path: Path | None,
        agent: str,
        model: str,
        repeats: int = 1,
        max_workers: int = 3,
        max_steps: int = 100,
        trials_dir: Path,
        dry_run: bool = False,
        skip_metrics: bool = True,
    ) -> list[str]:
        if not task_instances:
            raise ValueError("SkillLearnBench evaluation requires at least one task instance")
        command = [
            self.python_executable,
            "-m",
            "isl_dual.skilllearn_bridge",
            "--benchmark-root",
            str(self.root),
            "--trials-dir",
            str(Path(trials_dir).resolve()),
            "--agent",
            agent,
            "--model",
            model,
            "--repeats",
            str(repeats),
            "--max-workers",
            str(max_workers),
            "--max-steps",
            str(max_steps),
            "--skill-path",
            "none" if skill_path is None else str(Path(skill_path).resolve()),
        ]
        if skip_metrics:
            command.append("--skip-metrics")
        if dry_run:
            command.append("--dry-run")
        command.extend(str(item) for item in task_instances)
        return command

    def evaluate(
        self,
        task_instances: Sequence[str],
        *,
        skill_path: Path | None,
        agent: str,
        model: str,
        repeats: int = 1,
        max_workers: int = 3,
        max_steps: int = 100,
        trials_dir: Path,
        dry_run: bool = False,
        skip_metrics: bool = True,
        timeout_seconds: int | None = None,
    ) -> BenchmarkRun:
        command = self.build_command(
            task_instances,
            skill_path=skill_path,
            agent=agent,
            model=model,
            repeats=repeats,
            max_workers=max_workers,
            max_steps=max_steps,
            trials_dir=trials_dir,
            dry_run=dry_run,
            skip_metrics=skip_metrics,
        )
        completed = subprocess.run(
            command,
            cwd=self.root,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "SkillLearnBench evaluation failed with exit code "
                f"{completed.returncode}: {completed.stderr[-4000:]}"
            )
        if dry_run:
            return BenchmarkRun(tuple(command), None, None, None, completed.stdout, completed.stderr, True)
        passed, total, score = parse_skilllearn_summary(completed.stdout)
        return BenchmarkRun(tuple(command), passed, total, score, completed.stdout, completed.stderr, False)


class SWESkillsBenchAdapter:
    def __init__(self, root: Path, *, python_executable: str | None = None) -> None:
        self.root = Path(root).resolve()
        self.python_executable = python_executable or sys.executable

    def preflight(self) -> None:
        for name in ("run_all_skills.py", "run_all_skills_eval.py"):
            path = self.root / name
            if not path.is_file():
                raise FileNotFoundError(f"{name} is missing under {self.root}")

    def build_headroom_commands(
        self,
        skill_ids: Sequence[str],
        *,
        dry_run: bool = False,
        resume: bool = True,
    ) -> list[list[str]]:
        self.preflight()
        if not skill_ids:
            raise ValueError("SWE-Skills-Bench headroom gate requires skill IDs")
        selected = ",".join(skill_ids)
        commands: list[list[str]] = []
        for use_skill in (True, False):
            command = [
                self.python_executable,
                str(self.root / "run_all_skills.py"),
                "--use-skill" if use_skill else "--no-use-skill",
                "--only",
                selected,
            ]
            if resume:
                command.append("--resume")
            if dry_run:
                command.append("--dry-run")
            commands.append(command)
        for use_skill in (True, False):
            command = [
                self.python_executable,
                str(self.root / "run_all_skills_eval.py"),
                "--use-skill" if use_skill else "--no-use-skill",
                "--use-agent",
                "--only",
                selected,
            ]
            if resume:
                command.append("--resume")
            if dry_run:
                command.append("--dry-run")
            commands.append(command)
        return commands

    def run_headroom(
        self,
        skill_ids: Sequence[str],
        *,
        dry_run: bool = False,
        resume: bool = True,
        timeout_seconds: int | None = None,
    ) -> list[BenchmarkRun]:
        runs: list[BenchmarkRun] = []
        for command in self.build_headroom_commands(skill_ids, dry_run=dry_run, resume=resume):
            completed = subprocess.run(
                command,
                cwd=self.root,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "SWE-Skills-Bench command failed with exit code "
                    f"{completed.returncode}: {completed.stderr[-4000:]}"
                )
            runs.append(
                BenchmarkRun(
                    tuple(command), None, None, None,
                    completed.stdout, completed.stderr, dry_run,
                )
            )
        return runs
