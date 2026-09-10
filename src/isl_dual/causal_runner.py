from __future__ import annotations

import hashlib
import json
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Sequence

from .benchmark_adapters import BenchmarkRun, SkillLearnBenchAdapter, discover_skilllearn_instances
from .causal_pruning import PruningResult, causal_prune
from .skill_modules import SkillPackage, load_skill_package, render_skill_package


DEFAULT_TASKS = (
    "weighted-gdp-calculation",
    "financial-analysis",
    "github-repo-analytics",
)
DEFAULT_SEED_CONFIG = "b1-one-shot-claude-sonnet-4-6"
DEFAULT_HUMAN_CONFIG = "human_authored"


@dataclass(frozen=True)
class CMSConfig:
    agent: str
    model: str
    selection_instances: int = 2
    repeats: int = 1
    max_workers: int = 3
    max_steps: int = 100
    skip_metrics: bool = True
    tolerance: float = 0.0
    min_modules: int = 1
    random_seed: int = 20260910


@dataclass(frozen=True)
class CMSTaskResult:
    task: str
    selection_instances: tuple[str, ...]
    heldout_instances: tuple[str, ...]
    seed_module_ids: tuple[str, ...]
    cms_module_ids: tuple[str, ...]
    cms_module_titles: tuple[str, ...]
    selection_seed: float
    selection_cms: float
    heldout_scores: dict[str, float | None]
    seed_bytes: int
    cms_bytes: int
    knockout_effects: tuple[dict[str, Any], ...]


def resolve_skill_task_path(root: Path, config: str, task: str) -> Path:
    root = Path(root)
    candidates = (
        root / "skills" / config / task,
        root / "output" / "skill_generation_results" / config / task,
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"SkillLearnBench skill task directory not found for config={config!r}, task={task!r}; "
        f"checked: {', '.join(str(path) for path in candidates)}"
    )


def _tree_digest(root: Path | None) -> str:
    if root is None:
        return "none"
    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in Path(root).rglob("*") if path.is_file())


def _benchmark_identity(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
    )
    if completed.returncode == 0 and completed.stdout.strip():
        return completed.stdout.strip()
    marker = Path(root) / "evaluate_skills.py"
    return hashlib.sha256(marker.read_bytes() if marker.is_file() else str(root).encode()).hexdigest()


def _cache_key(
    adapter: Any,
    task_instances: Sequence[str],
    skill_path: Path | None,
    config: CMSConfig,
) -> str:
    payload = {
        "benchmark": _benchmark_identity(Path(adapter.root)),
        "instances": list(task_instances),
        "skill_digest": _tree_digest(skill_path),
        "agent": config.agent,
        "model": config.model,
        "repeats": config.repeats,
        "max_workers": config.max_workers,
        "max_steps": config.max_steps,
        "skip_metrics": config.skip_metrics,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _benchmark_run_to_dict(run: BenchmarkRun) -> dict[str, Any]:
    return {
        "command": list(run.command),
        "passed": run.passed,
        "total": run.total,
        "score": run.score,
        "stdout": run.stdout,
        "stderr": run.stderr,
        "dry_run": run.dry_run,
    }


def _benchmark_run_from_dict(value: dict[str, Any]) -> BenchmarkRun:
    return BenchmarkRun(
        command=tuple(str(item) for item in value["command"]),
        passed=value.get("passed"),
        total=value.get("total"),
        score=value.get("score"),
        stdout=str(value.get("stdout", "")),
        stderr=str(value.get("stderr", "")),
        dry_run=bool(value.get("dry_run", False)),
    )


def _evaluate_cached(
    adapter: Any,
    task_instances: Sequence[str],
    *,
    skill_path: Path | None,
    output: Path,
    config: CMSConfig,
) -> BenchmarkRun:
    key = _cache_key(adapter, task_instances, skill_path, config)
    cache_path = Path(output) / "cache" / "evaluations" / f"{key}.json"
    if cache_path.is_file():
        return _benchmark_run_from_dict(json.loads(cache_path.read_text()))

    run = adapter.evaluate(
        list(task_instances),
        skill_path=skill_path,
        agent=config.agent,
        model=config.model,
        repeats=config.repeats,
        max_workers=config.max_workers,
        max_steps=config.max_steps,
        trials_dir=Path(output) / "trials" / key,
        dry_run=False,
        skip_metrics=config.skip_metrics,
    )
    if run.score is None:
        raise RuntimeError("benchmark returned no scalar score for a live CMS evaluation")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(_benchmark_run_to_dict(run), indent=2, sort_keys=True))
    temporary.replace(cache_path)
    return run


def _materialize_subset(
    package: SkillPackage,
    retained_ids: frozenset[str] | set[str],
    *,
    output: Path,
    task: str,
    label: str,
) -> Path:
    retained = frozenset(retained_ids)
    token = hashlib.sha256("\n".join(sorted(retained)).encode()).hexdigest()[:12]
    variant_root = Path(output) / "skills" / task / f"{label}-{token}"
    destination = variant_root / task
    render_skill_package(package, retained, destination)
    return variant_root


def _score_or_fail(run: BenchmarkRun) -> float:
    if run.score is None:
        raise RuntimeError("live benchmark evaluation did not return a score")
    return float(run.score)


def run_cms_task(
    adapter: Any,
    *,
    task: str,
    seed_task_path: Path,
    human_task_path: Path | None,
    output: Path,
    config: CMSConfig,
) -> CMSTaskResult:
    output = Path(output)
    instances = discover_skilllearn_instances(Path(adapter.root), task)
    if config.selection_instances < 1 or config.selection_instances >= len(instances):
        raise ValueError(
            f"selection_instances must be in [1, {len(instances) - 1}] for task {task}"
        )
    selection = tuple(instances[: config.selection_instances])
    heldout = tuple(instances[config.selection_instances :])

    seed_package = load_skill_package(Path(seed_task_path))
    seed_ids = tuple(module.id for module in seed_package.modules)
    if not seed_ids:
        raise ValueError(f"seed skill for {task} has no removable H2 modules")
    titles = {module.id: module.title for module in seed_package.modules}

    def selection_score(retained: frozenset[str]) -> float:
        variant = _materialize_subset(
            seed_package,
            retained,
            output=output,
            task=task,
            label="selection",
        )
        return _score_or_fail(
            _evaluate_cached(
                adapter,
                selection,
                skill_path=variant,
                output=output,
                config=config,
            )
        )

    pruning: PruningResult = causal_prune(
        seed_ids,
        selection_score,
        tolerance=config.tolerance,
        min_modules=config.min_modules,
    )
    cms_ids = tuple(pruning.retained_ids)

    seed_root = _materialize_subset(
        seed_package, frozenset(seed_ids), output=output, task=task, label="seed"
    )
    cms_root = _materialize_subset(
        seed_package, frozenset(cms_ids), output=output, task=task, label="cms"
    )

    rng = random.Random(config.random_seed + int(hashlib.sha256(task.encode()).hexdigest()[:8], 16))
    if len(cms_ids) >= len(seed_ids):
        random_ids = seed_ids
    else:
        sampled = set(rng.sample(list(seed_ids), len(cms_ids)))
        random_ids = tuple(module_id for module_id in seed_ids if module_id in sampled)
    random_root = _materialize_subset(
        seed_package, frozenset(random_ids), output=output, task=task, label="random"
    )

    human_root: Path | None = None
    if human_task_path is not None:
        human_package = load_skill_package(Path(human_task_path))
        human_ids = frozenset(module.id for module in human_package.modules)
        human_root = _materialize_subset(
            human_package, human_ids, output=output, task=task, label="human"
        )

    heldout_paths: dict[str, Path | None] = {
        "no_skill": None,
        "seed": seed_root,
        "random_prune": random_root,
        "cms": cms_root,
        "human_authored": human_root,
    }
    heldout_scores: dict[str, float | None] = {}
    for name, path in heldout_paths.items():
        if name == "human_authored" and path is None:
            heldout_scores[name] = None
            continue
        heldout_scores[name] = _score_or_fail(
            _evaluate_cached(
                adapter,
                heldout,
                skill_path=path,
                output=output,
                config=config,
            )
        )

    result = CMSTaskResult(
        task=task,
        selection_instances=selection,
        heldout_instances=heldout,
        seed_module_ids=seed_ids,
        cms_module_ids=cms_ids,
        cms_module_titles=tuple(titles[module_id] for module_id in cms_ids),
        selection_seed=pruning.full_score,
        selection_cms=pruning.final_score,
        heldout_scores=heldout_scores,
        seed_bytes=_tree_bytes(seed_root / task),
        cms_bytes=_tree_bytes(cms_root / task),
        knockout_effects=tuple(asdict(effect) for effect in pruning.effects),
    )
    result_path = output / "tasks" / task / "result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = result_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(asdict(result), indent=2, sort_keys=True))
    temporary.replace(result_path)
    return result


def summarize_pilot(results: Sequence[CMSTaskResult]) -> dict[str, Any]:
    if not results:
        raise ValueError("pilot summary requires task results")
    cms_lifts: list[float] = []
    headroom_lifts: list[float] = []
    cms_wins = 0
    headroom_positive = 0
    compressed_tasks = 0

    for result in results:
        no_skill = result.heldout_scores.get("no_skill")
        seed = result.heldout_scores.get("seed")
        cms = result.heldout_scores.get("cms")
        human = result.heldout_scores.get("human_authored")
        if no_skill is None or seed is None or cms is None:
            raise ValueError(f"task {result.task} is missing required held-out scores")
        cms_lift = float(cms) - float(seed)
        cms_lifts.append(cms_lift)
        if cms_lift > 1e-12:
            cms_wins += 1
        best_skill = max(float(seed), float(human) if human is not None else float(seed))
        headroom_lift = best_skill - float(no_skill)
        headroom_lifts.append(headroom_lift)
        if headroom_lift > 1e-12:
            headroom_positive += 1
        if len(result.cms_module_ids) < len(result.seed_module_ids) and result.cms_bytes < result.seed_bytes:
            compressed_tasks += 1

    required_wins = min(2, len(results))
    aggregate_cms_lift = fmean(cms_lifts)
    aggregate_headroom_lift = fmean(headroom_lifts)
    headroom_pass = headroom_positive >= required_wins and aggregate_headroom_lift > 0
    method_pass = (
        cms_wins >= required_wins
        and compressed_tasks >= required_wins
        and aggregate_cms_lift > 0
    )
    return {
        "tasks": len(results),
        "required_wins": required_wins,
        "headroom_positive_tasks": headroom_positive,
        "headroom_pass": headroom_pass,
        "aggregate_headroom_lift": aggregate_headroom_lift,
        "cms_wins": cms_wins,
        "compressed_tasks": compressed_tasks,
        "aggregate_cms_lift": aggregate_cms_lift,
        "method_pass": method_pass,
        "go": bool(headroom_pass and method_pass),
    }
