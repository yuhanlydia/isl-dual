from __future__ import annotations

from pathlib import Path

import pytest

from isl_dual.benchmark_adapters import BenchmarkRun
from isl_dual.causal_runner import (
    CMSConfig,
    CMSTaskResult,
    resolve_skill_task_path,
    run_cms_task,
    summarize_pilot,
)


def _write_package(root: Path, config: str, task: str, *, human: bool = False) -> Path:
    task_root = root / "skills" / config / task
    if human:
        skill = task_root / "oracle-skill"
        skill.mkdir(parents=True, exist_ok=True)
        (skill / "SKILL.md").write_text(
            "---\nname: oracle-skill\ndescription: oracle\n---\n# Oracle\n\n## Procedure\nUse the complete correct workflow.\n"
        )
    else:
        for name, body in (
            ("helpful-skill", "Use the reusable helpful procedure."),
            ("harmful-skill", "Add an unnecessary harmful detour."),
        ):
            skill = task_root / name
            skill.mkdir(parents=True, exist_ok=True)
            (skill / "SKILL.md").write_text(
                f"---\nname: {name}\ndescription: {name}\n---\n# {name}\n\n## Procedure\n{body}\n"
            )
    return task_root


def _benchmark_root(root: Path, task: str = "demo", instances: int = 5) -> Path:
    (root / "evaluate_skills.py").parent.mkdir(parents=True, exist_ok=True)
    (root / "evaluate_skills.py").write_text("# fake upstream")
    for index in range(1, instances + 1):
        query = root / "tasks" / task / f"{task}-{index}"
        query.mkdir(parents=True)
        (query / "instruction.md").write_text(f"instance {index}")
    _write_package(root, "seed", task)
    _write_package(root, "human_authored", task, human=True)
    return root


class FakeAdapter:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.calls = 0

    def evaluate(
        self,
        task_instances,
        *,
        skill_path,
        agent,
        model,
        repeats=1,
        max_workers=3,
        max_steps=100,
        trials_dir,
        dry_run=False,
        skip_metrics=True,
        timeout_seconds=None,
    ):
        self.calls += 1
        if skill_path is None:
            score = 0.4
        else:
            text = "\n".join(
                path.read_text()
                for path in (Path(skill_path) / "demo").rglob("SKILL.md")
            )
            if "name: oracle-skill" in text:
                score = 0.9
            else:
                score = 0.5
                if "name: helpful-skill" in text:
                    score += 0.3
                if "name: harmful-skill" in text:
                    score -= 0.2
        total = len(task_instances) * repeats
        passed = round(score * total)
        return BenchmarkRun(
            command=("fake",),
            passed=passed,
            total=total,
            score=score,
            stdout=f"Overall: {passed}/{total} ({score:.2%})",
            stderr="",
            dry_run=False,
        )


def test_resolve_skill_task_path_uses_committed_skill_tree(tmp_path: Path) -> None:
    root = _benchmark_root(tmp_path)

    resolved = resolve_skill_task_path(root, "seed", "demo")

    assert resolved == root / "skills" / "seed" / "demo"


def test_run_cms_task_prunes_harmful_native_skill_and_improves_heldout(tmp_path: Path) -> None:
    root = _benchmark_root(tmp_path / "bench")
    adapter = FakeAdapter(root)
    result = run_cms_task(
        adapter,
        task="demo",
        seed_task_path=resolve_skill_task_path(root, "seed", "demo"),
        human_task_path=resolve_skill_task_path(root, "human_authored", "demo"),
        output=tmp_path / "run",
        config=CMSConfig(agent="fake", model="fake", selection_instances=2, random_seed=7),
    )

    assert result.selection_instances == ("demo/demo-1", "demo/demo-2")
    assert result.heldout_instances == ("demo/demo-3", "demo/demo-4", "demo/demo-5")
    assert len(result.seed_module_ids) == 2
    assert len(result.cms_module_ids) == 1
    assert result.cms_module_titles == ("helpful-skill",)
    assert result.selection_seed == pytest.approx(0.6)
    assert result.selection_cms == pytest.approx(0.8)
    assert result.heldout_scores["seed"] == pytest.approx(0.6)
    assert result.heldout_scores["cms"] == pytest.approx(0.8)
    assert result.heldout_scores["no_skill"] == pytest.approx(0.4)
    assert result.heldout_scores["human_authored"] == pytest.approx(0.9)
    assert result.cms_bytes < result.seed_bytes
    cms_task_roots = list((tmp_path / "run" / "skills" / "demo").glob("cms-*/demo"))
    assert len(cms_task_roots) == 1
    assert (cms_task_roots[0] / "helpful-skill" / "SKILL.md").is_file()
    assert not (cms_task_roots[0] / "harmful-skill").exists()
    assert (tmp_path / "run" / "tasks" / "demo" / "result.json").is_file()


def test_run_cms_task_reuses_disk_evaluation_cache(tmp_path: Path) -> None:
    root = _benchmark_root(tmp_path / "bench")
    config = CMSConfig(agent="fake", model="fake", selection_instances=2, random_seed=7)
    first_adapter = FakeAdapter(root)
    run_cms_task(
        first_adapter,
        task="demo",
        seed_task_path=resolve_skill_task_path(root, "seed", "demo"),
        human_task_path=resolve_skill_task_path(root, "human_authored", "demo"),
        output=tmp_path / "run",
        config=config,
    )
    assert first_adapter.calls > 0

    second_adapter = FakeAdapter(root)
    run_cms_task(
        second_adapter,
        task="demo",
        seed_task_path=resolve_skill_task_path(root, "seed", "demo"),
        human_task_path=resolve_skill_task_path(root, "human_authored", "demo"),
        output=tmp_path / "run",
        config=config,
    )

    assert second_adapter.calls == 0


def _result(task: str, *, no_skill: float, seed: float, cms: float, human: float, seed_n: int, cms_n: int) -> CMSTaskResult:
    return CMSTaskResult(
        task=task,
        selection_instances=(f"{task}/{task}-1", f"{task}/{task}-2"),
        heldout_instances=(f"{task}/{task}-3",),
        seed_module_ids=tuple(f"s{i}" for i in range(seed_n)),
        cms_module_ids=tuple(f"s{i}" for i in range(cms_n)),
        cms_module_titles=tuple(f"module-{i}" for i in range(cms_n)),
        selection_seed=seed,
        selection_cms=cms,
        heldout_scores={
            "no_skill": no_skill,
            "seed": seed,
            "random_prune": seed,
            "cms": cms,
            "human_authored": human,
        },
        seed_bytes=1000,
        cms_bytes=600,
        knockout_effects=(),
    )


def test_summarize_pilot_requires_headroom_two_cms_wins_and_compression() -> None:
    summary = summarize_pilot([
        _result("a", no_skill=0.4, seed=0.5, cms=0.7, human=0.8, seed_n=4, cms_n=2),
        _result("b", no_skill=0.5, seed=0.55, cms=0.65, human=0.75, seed_n=4, cms_n=2),
        _result("c", no_skill=0.6, seed=0.6, cms=0.58, human=0.6, seed_n=4, cms_n=4),
    ])

    assert summary["headroom_pass"] is True
    assert summary["cms_wins"] == 2
    assert summary["compressed_tasks"] == 2
    assert summary["aggregate_cms_lift"] > 0
    assert summary["go"] is True


def test_summarize_pilot_stops_when_cms_has_no_aggregate_advantage() -> None:
    summary = summarize_pilot([
        _result("a", no_skill=0.4, seed=0.6, cms=0.5, human=0.8, seed_n=4, cms_n=2),
        _result("b", no_skill=0.4, seed=0.6, cms=0.6, human=0.8, seed_n=4, cms_n=2),
        _result("c", no_skill=0.4, seed=0.6, cms=0.61, human=0.8, seed_n=4, cms_n=2),
    ])

    assert summary["go"] is False
    assert summary["aggregate_cms_lift"] < 0
