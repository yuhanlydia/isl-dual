from __future__ import annotations

from pathlib import Path

import pytest

from isl_dual.benchmark_adapters import (
    SkillLearnBenchAdapter,
    SWESkillsBenchAdapter,
    discover_skilllearn_instances,
    parse_skilllearn_summary,
)


def _skilllearn_root(root: Path) -> Path:
    (root / "tasks" / "demo" / "demo-10").mkdir(parents=True)
    (root / "tasks" / "demo" / "demo-2").mkdir(parents=True)
    (root / "tasks" / "demo" / "demo-1").mkdir(parents=True)
    for name in ("demo-10", "demo-2", "demo-1"):
        (root / "tasks" / "demo" / name / "instruction.md").write_text("do it")
    (root / "evaluate_skills.py").write_text("# upstream")
    return root


def test_discover_skilllearn_instances_sorts_by_numeric_suffix(tmp_path: Path) -> None:
    root = _skilllearn_root(tmp_path)

    instances = discover_skilllearn_instances(root, "demo")

    assert instances == ["demo/demo-1", "demo/demo-2", "demo/demo-10"]


def test_skilllearn_adapter_builds_bridge_command_with_isolated_trials_dir(tmp_path: Path) -> None:
    root = _skilllearn_root(tmp_path / "bench")
    skill_root = tmp_path / "variant"
    (skill_root / "demo").mkdir(parents=True)
    adapter = SkillLearnBenchAdapter(root, python_executable="python-test")

    command = adapter.build_command(
        ["demo/demo-1", "demo/demo-2"],
        skill_path=skill_root,
        agent="codex",
        model="gpt-test",
        repeats=2,
        max_workers=3,
        max_steps=40,
        trials_dir=tmp_path / "trials",
        dry_run=True,
    )

    assert command[:4] == ["python-test", "-m", "isl_dual.skilllearn_bridge", "--benchmark-root"]
    assert str(root.resolve()) in command
    assert "--skill-path" in command
    assert str(skill_root.resolve()) in command
    assert "--trials-dir" in command
    assert str((tmp_path / "trials").resolve()) in command
    assert command[command.index("--agent") + 1] == "codex"
    assert command[command.index("--model") + 1] == "gpt-test"
    assert command[command.index("--repeats") + 1] == "2"
    assert command[command.index("--max-workers") + 1] == "3"
    assert command[command.index("--max-steps") + 1] == "40"
    assert "--skip-metrics" in command
    assert "--dry-run" in command
    assert command[-2:] == ["demo/demo-1", "demo/demo-2"]


def test_skilllearn_no_skill_uses_none_sentinel(tmp_path: Path) -> None:
    adapter = SkillLearnBenchAdapter(_skilllearn_root(tmp_path))

    command = adapter.build_command(
        ["demo/demo-1"],
        skill_path=None,
        agent="claude-code",
        model="claude-test",
        trials_dir=tmp_path / "trials",
    )

    assert command[command.index("--skill-path") + 1] == "none"


def test_skilllearn_preflight_fails_on_missing_layout(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="evaluate_skills.py"):
        SkillLearnBenchAdapter(tmp_path).preflight(["demo"])


def test_parse_skilllearn_summary_reads_overall_pass_rate() -> None:
    passed, total, score = parse_skilllearn_summary(
        "noise\n=== Summary ===\n  cfg: 3/5 (60.00%)\n\nOverall: 3/5 (60.00%)\n"
    )

    assert (passed, total) == (3, 5)
    assert score == pytest.approx(0.6)


def test_parse_skilllearn_summary_rejects_missing_summary() -> None:
    with pytest.raises(ValueError, match="Overall"):
        parse_skilllearn_summary("no summary here")


def _swe_root(root: Path) -> Path:
    for name in ("run_all_skills.py", "run_all_skills_eval.py"):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("# upstream")
    return root


def test_swe_headroom_builds_official_paired_commands(tmp_path: Path) -> None:
    root = _swe_root(tmp_path)
    adapter = SWESkillsBenchAdapter(root, python_executable="python-test")

    commands = adapter.build_headroom_commands(
        ["risk-metrics-calculation", "gitlab-ci-patterns", "tdd-workflow"],
        dry_run=True,
        resume=True,
    )

    assert len(commands) == 4
    joined = [" ".join(command) for command in commands]
    assert any("run_all_skills.py --use-skill" in command for command in joined)
    assert any("run_all_skills.py --no-use-skill" in command for command in joined)
    assert any("run_all_skills_eval.py --use-skill --use-agent" in command for command in joined)
    assert any("run_all_skills_eval.py --no-use-skill --use-agent" in command for command in joined)
    assert all("risk-metrics-calculation,gitlab-ci-patterns,tdd-workflow" in command for command in joined)
    assert all("--dry-run" in command for command in joined)
    assert all("--resume" in command for command in joined)
