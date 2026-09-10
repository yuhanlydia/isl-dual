from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Sequence

from .benchmark_adapters import (
    BenchmarkRun,
    SWESkillsBenchAdapter,
    SkillLearnBenchAdapter,
    discover_skilllearn_instances,
)
from .causal_pruning import PruningResult, causal_prune
from .skill_modules import SkillPackage, load_skill_package, render_skill_package


DEFAULT_TASKS = (
    "weighted-gdp-calculation",
    "financial-analysis",
    "github-repo-analytics",
)
DEFAULT_SEED_CONFIG = "b1-one-shot-claude-sonnet-4-6"
DEFAULT_HUMAN_CONFIG = "human_authored"
DEFAULT_SWE_SKILLS = (
    "risk-metrics-calculation",
    "gitlab-ci-patterns",
    "tdd-workflow",
)


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


def _resolve_skill_config_root(root: Path, config: str) -> Path:
    root = Path(root)
    candidates = (
        root / "skills" / config,
        root / "output" / "skill_generation_results" / config,
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"SkillLearnBench skill config directory not found for {config!r}; "
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


def plan_pilot(
    skilllearn_root: Path,
    *,
    tasks: Sequence[str] = DEFAULT_TASKS,
    seed_config: str = DEFAULT_SEED_CONFIG,
    human_config: str = DEFAULT_HUMAN_CONFIG,
    selection_instances: int = 2,
    agent: str = "claude-code",
    model: str = "claude-sonnet-4-6",
    max_workers: int = 3,
    max_steps: int = 100,
    output: Path = Path("runs/cms-v1-pilot"),
) -> dict[str, Any]:
    root = Path(skilllearn_root).resolve()
    output = Path(output).resolve()
    adapter = SkillLearnBenchAdapter(root)
    adapter.preflight(tasks)
    seed_root = _resolve_skill_config_root(root, seed_config)
    human_root = _resolve_skill_config_root(root, human_config)
    rows: list[dict[str, Any]] = []
    for task in tasks:
        instances = discover_skilllearn_instances(root, task)
        if selection_instances < 1 or selection_instances >= len(instances):
            raise ValueError(
                f"selection_instances must be in [1, {len(instances) - 1}] for {task}"
            )
        selection = instances[:selection_instances]
        heldout = instances[selection_instances:]
        package = load_skill_package(resolve_skill_task_path(root, seed_config, task))
        module_count = len(package.modules)
        if module_count == 0:
            raise ValueError(f"seed skill for {task} has no removable H2 modules")
        no_skill_command = adapter.build_command(
            heldout,
            skill_path=None,
            agent=agent,
            model=model,
            max_workers=max_workers,
            max_steps=max_steps,
            trials_dir=output / "plan" / task / "no_skill",
            dry_run=True,
        )
        seed_command = adapter.build_command(
            heldout,
            skill_path=seed_root,
            agent=agent,
            model=model,
            max_workers=max_workers,
            max_steps=max_steps,
            trials_dir=output / "plan" / task / "seed",
            dry_run=True,
        )
        rows.append(
            {
                "task": task,
                "selection_instances": selection,
                "heldout_instances": heldout,
                "seed_skill": str(resolve_skill_task_path(root, seed_config, task)),
                "human_skill": str(resolve_skill_task_path(root, human_config, task)),
                "seed_modules": module_count,
                "max_selection_skill_evaluations": 1 + 2 * module_count,
                "heldout_conditions": 5,
                "example_no_skill_command": no_skill_command,
                "example_seed_command": seed_command,
            }
        )
    return {
        "benchmark_root": str(root),
        "output": str(output),
        "agent": agent,
        "model": model,
        "selection_instances_per_task": selection_instances,
        "tasks": rows,
        "note": "dry-run planning only; no Docker/model/API calls are executed",
    }


def run_pilot(
    skilllearn_root: Path,
    *,
    output: Path,
    tasks: Sequence[str] = DEFAULT_TASKS,
    seed_config: str = DEFAULT_SEED_CONFIG,
    human_config: str = DEFAULT_HUMAN_CONFIG,
    config: CMSConfig,
) -> dict[str, Any]:
    root = Path(skilllearn_root).resolve()
    output = Path(output).resolve()
    adapter = SkillLearnBenchAdapter(root)
    adapter.preflight(tasks)
    results: list[CMSTaskResult] = []
    for task in tasks:
        result = run_cms_task(
            adapter,
            task=task,
            seed_task_path=resolve_skill_task_path(root, seed_config, task),
            human_task_path=resolve_skill_task_path(root, human_config, task),
            output=output,
            config=config,
        )
        results.append(result)
    summary = summarize_pilot(results)
    payload = {
        "method": "Causal Minimal Skill (CMS)",
        "skilllearn_root": str(root),
        "tasks": [asdict(result) for result in results],
        "summary": summary,
        "config": asdict(config),
        "seed_config": seed_config,
        "human_config": human_config,
    }
    output.mkdir(parents=True, exist_ok=True)
    pilot_path = output / "pilot.json"
    temporary = pilot_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(pilot_path)
    return payload


def run_skilllearn_headroom(
    skilllearn_root: Path,
    *,
    output: Path,
    tasks: Sequence[str],
    seed_config: str,
    human_config: str,
    config: CMSConfig,
) -> dict[str, Any]:
    root = Path(skilllearn_root).resolve()
    output = Path(output).resolve()
    adapter = SkillLearnBenchAdapter(root)
    adapter.preflight(tasks)
    seed_root = _resolve_skill_config_root(root, seed_config)
    human_root = _resolve_skill_config_root(root, human_config)
    rows: dict[str, dict[str, float]] = {}
    for task in tasks:
        instances = discover_skilllearn_instances(root, task)
        scores: dict[str, float] = {}
        for name, skill_path in (
            ("no_skill", None),
            ("seed", seed_root),
            ("human_authored", human_root),
        ):
            scores[name] = _score_or_fail(
                _evaluate_cached(
                    adapter,
                    instances,
                    skill_path=skill_path,
                    output=output,
                    config=config,
                )
            )
        rows[task] = scores
    lifts = [max(row["seed"], row["human_authored"]) - row["no_skill"] for row in rows.values()]
    payload = {
        "tasks": rows,
        "positive_tasks": sum(lift > 1e-12 for lift in lifts),
        "aggregate_headroom_lift": fmean(lifts),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "headroom.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _add_skilllearn_common(parser: argparse.ArgumentParser, *, output_required: bool) -> None:
    parser.add_argument("--skilllearn-root", type=Path, required=True)
    if output_required:
        parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--seed-config", default=DEFAULT_SEED_CONFIG)
    parser.add_argument("--human-config", default=DEFAULT_HUMAN_CONFIG)
    parser.add_argument("--agent", default="claude-code")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--selection-instances", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--min-modules", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=20260910)
    parser.set_defaults(skip_metrics=True)
    parser.add_argument(
        "--with-metrics",
        action="store_false",
        dest="skip_metrics",
        help="also run SkillLearnBench LLM-as-judge metrics; not needed for the primary pass/fail gate",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="isl-causal-skill",
        description="Causal Minimal Skill: execution-grounded module knockout and conservative skill compression.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="validate SkillLearnBench layout and seed skills")
    _add_skilllearn_common(preflight, output_required=False)

    pilot = subparsers.add_parser("pilot", help="run the canonical 3-task GO/STOP pilot")
    _add_skilllearn_common(pilot, output_required=True)
    pilot.add_argument("--dry-run", action="store_true")

    prune = subparsers.add_parser("prune", help="run CMS on one or more selected tasks")
    _add_skilllearn_common(prune, output_required=True)
    prune.add_argument("--dry-run", action="store_true")

    headroom = subparsers.add_parser("skilllearn-headroom", help="measure no-skill vs seed vs human headroom")
    _add_skilllearn_common(headroom, output_required=True)
    headroom.add_argument("--dry-run", action="store_true")

    swe = subparsers.add_parser("swe-headroom", help="run the optional known-positive SWE-Skills-Bench protocol gate")
    swe.add_argument("--swe-root", type=Path, required=True)
    swe.add_argument("--skills", nargs="+", default=list(DEFAULT_SWE_SKILLS))
    swe.add_argument("--dry-run", action="store_true")
    swe.add_argument("--no-resume", action="store_false", dest="resume", default=True)
    return parser


def _config_from_args(args: argparse.Namespace) -> CMSConfig:
    return CMSConfig(
        agent=args.agent,
        model=args.model,
        selection_instances=args.selection_instances,
        repeats=args.repeats,
        max_workers=args.max_workers,
        max_steps=args.max_steps,
        skip_metrics=args.skip_metrics,
        tolerance=args.tolerance,
        min_modules=args.min_modules,
        random_seed=args.random_seed,
    )


def _preflight(args: argparse.Namespace) -> None:
    root = args.skilllearn_root.resolve()
    adapter = SkillLearnBenchAdapter(root)
    adapter.preflight(args.tasks)
    for task in args.tasks:
        load_skill_package(resolve_skill_task_path(root, args.seed_config, task))
        load_skill_package(resolve_skill_task_path(root, args.human_config, task))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        _preflight(args)
        print("preflight: PASS")
        print("Note: live execution will additionally require Docker and the selected upstream agent API key.")
        return 0

    if args.command in {"pilot", "prune", "skilllearn-headroom"}:
        _preflight(args)
        if args.dry_run:
            plan = plan_pilot(
                args.skilllearn_root,
                tasks=args.tasks,
                seed_config=args.seed_config,
                human_config=args.human_config,
                selection_instances=args.selection_instances,
                agent=args.agent,
                model=args.model,
                max_workers=args.max_workers,
                max_steps=args.max_steps,
                output=args.output,
            )
            print("NO MODEL/API CALLS — CMS dry-run plan")
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0

        config = _config_from_args(args)
        if args.command == "skilllearn-headroom":
            payload = run_skilllearn_headroom(
                args.skilllearn_root,
                output=args.output,
                tasks=args.tasks,
                seed_config=args.seed_config,
                human_config=args.human_config,
                config=config,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0

        payload = run_pilot(
            args.skilllearn_root,
            output=args.output,
            tasks=args.tasks,
            seed_config=args.seed_config,
            human_config=args.human_config,
            config=config,
        )
        decision = "GO" if payload["summary"]["go"] else "STOP"
        print(f"CMS decision: {decision}")
        print(json.dumps(payload["summary"], indent=2, sort_keys=True))
        print(f"report: {Path(args.output).resolve() / 'pilot.json'}")
        return 0

    if args.command == "swe-headroom":
        adapter = SWESkillsBenchAdapter(args.swe_root)
        if args.dry_run:
            print("NO MODEL/API CALLS — SWE-Skills-Bench headroom commands")
            for command in adapter.build_headroom_commands(args.skills, dry_run=True, resume=args.resume):
                print(" ".join(command))
            return 0
        runs = adapter.run_headroom(args.skills, dry_run=False, resume=args.resume)
        print(f"completed {len(runs)} upstream SWE-Skills-Bench commands")
        print("Run the upstream scripts/compare_pass_rate.py --all report to inspect per-skill treatment effects.")
        return 0

    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
