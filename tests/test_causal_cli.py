from __future__ import annotations

from pathlib import Path

from isl_dual.causal_runner import (
    DEFAULT_TASKS,
    build_parser,
    main,
    plan_pilot,
)


def _package(root: Path, config: str, task: str) -> None:
    skill = root / "skills" / config / task / "workflow"
    skill.mkdir(parents=True, exist_ok=True)
    (skill / "SKILL.md").write_text(
        """---\nname: workflow\ndescription: workflow\n---\n# Workflow\n\n## Inspect\nInspect inputs.\n\n## Execute\nExecute procedure.\n\n## Verify\nVerify result.\n"""
    )


def _root(root: Path) -> Path:
    (root / "evaluate_skills.py").parent.mkdir(parents=True, exist_ok=True)
    (root / "evaluate_skills.py").write_text("# upstream")
    for task in DEFAULT_TASKS:
        for index in range(1, 6):
            query = root / "tasks" / task / f"{task}-{index}"
            query.mkdir(parents=True)
            (query / "instruction.md").write_text("do it")
        _package(root, "b1-one-shot-claude-sonnet-4-6", task)
        _package(root, "human_authored", task)
    return root


def test_pilot_parser_has_canonical_defaults() -> None:
    args = build_parser().parse_args([
        "pilot",
        "--skilllearn-root", "/bench",
        "--output", "/run",
    ])

    assert tuple(args.tasks) == DEFAULT_TASKS
    assert args.seed_config == "b1-one-shot-claude-sonnet-4-6"
    assert args.agent == "claude-code"
    assert args.model == "claude-sonnet-4-6"
    assert args.selection_instances == 2
    assert args.max_workers == 3
    assert args.skip_metrics is True
    assert args.dry_run is False


def test_plan_pilot_reports_disjoint_splits_and_evaluation_upper_bound(tmp_path: Path) -> None:
    root = _root(tmp_path / "bench")

    plan = plan_pilot(
        root,
        tasks=DEFAULT_TASKS,
        seed_config="b1-one-shot-claude-sonnet-4-6",
        human_config="human_authored",
        selection_instances=2,
        agent="claude-code",
        model="claude-sonnet-4-6",
        max_workers=3,
        max_steps=100,
        output=tmp_path / "run",
    )

    assert len(plan["tasks"]) == 3
    for item in plan["tasks"]:
        assert len(item["selection_instances"]) == 2
        assert len(item["heldout_instances"]) == 3
        assert set(item["selection_instances"]).isdisjoint(item["heldout_instances"])
        assert item["seed_modules"] == 3
        assert item["max_selection_skill_evaluations"] == 7  # full + 3 knockouts + <=3 greedy
        assert item["heldout_conditions"] == 5
        assert item["example_no_skill_command"]
        assert item["example_seed_command"]


def test_pilot_dry_run_makes_no_external_benchmark_calls(tmp_path: Path, monkeypatch, capsys) -> None:
    root = _root(tmp_path / "bench")

    def forbidden_run(*args, **kwargs):
        raise AssertionError("dry-run must not execute benchmark subprocesses")

    monkeypatch.setattr("isl_dual.benchmark_adapters.subprocess.run", forbidden_run)
    code = main([
        "pilot",
        "--skilllearn-root", str(root),
        "--output", str(tmp_path / "run"),
        "--dry-run",
    ])

    assert code == 0
    output = capsys.readouterr().out
    assert "NO MODEL/API CALLS" in output
    assert "weighted-gdp-calculation" in output
    assert "max_selection_skill_evaluations" in output
    assert not (tmp_path / "run" / "pilot.json").exists()


def test_preflight_accepts_valid_fixture_without_docker_or_api(tmp_path: Path, capsys) -> None:
    root = _root(tmp_path / "bench")

    code = main([
        "preflight",
        "--skilllearn-root", str(root),
    ])

    assert code == 0
    assert "preflight: PASS" in capsys.readouterr().out
