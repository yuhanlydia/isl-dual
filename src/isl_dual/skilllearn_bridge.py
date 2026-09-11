from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


def _load_upstream(root: Path):
    root = root.resolve()
    if not (root / "evaluate_skills.py").is_file():
        raise FileNotFoundError(f"evaluate_skills.py is missing under {root}")
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module("evaluate_skills")
    finally:
        if sys.path and sys.path[0] == str(root):
            sys.path.pop(0)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Thin bridge to SkillLearnBench.hyper_eval without its unconditional Anthropic-only CLI guard."
    )
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--trials-dir", type=Path, required=True)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--skill-path", required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--build-workers", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("tasks", nargs="+")
    args = parser.parse_args(argv)

    root = args.benchmark_root.resolve()
    upstream = _load_upstream(root)
    upstream._load_dotenv()

    skill_path = None if args.skill_path.lower() == "none" else Path(args.skill_path).resolve()
    skill_paths = [skill_path]
    task_root = root / "tasks"

    if not args.dry_run:
        upstream._require_docker()
        upstream._validate_api_keys(
            agent_id=args.agent,
            task_ids=list(args.tasks),
            task_root=task_root,
            need_agent_keys=True,
            need_judge_key=not args.skip_metrics,
        )
        upstream._ensure_skill_paths(skill_paths, list(args.tasks))

    return int(
        upstream.hyper_eval(
            list(args.tasks),
            task_root=task_root,
            agent_id=args.agent,
            model=args.model,
            skill_paths=skill_paths,
            repeats=max(1, args.repeats),
            max_steps=max(1, args.max_steps),
            max_workers=max(1, args.max_workers),
            build_workers=max(1, args.build_workers),
            remove_images=False,
            record=True,
            dry_run=args.dry_run,
            config_path=None,
            trials_dir=args.trials_dir.resolve(),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
