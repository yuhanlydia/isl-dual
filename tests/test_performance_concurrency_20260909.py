from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import isl_dual.baselines as baselines_module
import isl_dual.executor as executor_module
import isl_dual.mcts as mcts_module
import isl_dual.pipeline as pipeline_module
from isl_dual.baselines import Baseline
from isl_dual.config import PilotConfig
from isl_dual.executor import CodexExecutor
from isl_dual.mcts import EvidenceJournal
from isl_dual.models import AcquisitionTask, Graph, MCTSResult, Node, Rollout


def _graph(identifier: str) -> Graph:
    node = Node(
        "n1",
        "do task",
        (),
        ("task",),
        "complete the task",
        ("done",),
        "verify",
        True,
    )
    return Graph(identifier, (node,))


def _task(identifier: str) -> AcquisitionTask:
    return AcquisitionTask(
        identifier,
        f"task {identifier}",
        {"delta": {}},
        lambda output: 1.0,
    )


class _SlowExecutor:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def execute(self, task, graph, plan):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.03)
        with self.lock:
            self.active -= 1
        return {"ok": True}


def test_forward_loop_parallelizes_independent_graph_task_trees(monkeypatch: pytest.MonkeyPatch) -> None:
    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_mcts(graph, task, executor, **kwargs):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return MCTSResult([Rollout(("n1",), 1.0)])

    monkeypatch.setattr(pipeline_module, "mcts", fake_mcts)
    graphs = [_graph(f"g{i}") for i in range(4)]
    tasks = [_task(f"t{i}") for i in range(3)]
    config = PilotConfig(candidate_graphs=4, mcts_budget=1, forward_workers=3)

    forward, stability, evidence = pipeline_module._forward_loop(
        graphs,
        tasks,
        object(),
        config,
        0,
    )

    assert max_active >= 2
    assert set(forward) == {graph.id for graph in graphs}
    assert all(score == 1.0 for score in forward.values())
    assert all(value == 0.0 for value in stability.values())
    assert len(evidence) == len(graphs) * len(tasks)


def test_evidence_journal_is_safe_under_concurrent_tree_completion(tmp_path: Path) -> None:
    journal = EvidenceJournal(tmp_path / "evidence.json")

    def record(index: int) -> None:
        journal.record(
            f"tree:{index}",
            graph_id=f"g{index % 4}",
            task_id=f"t{index % 3}",
            phase="round1",
            rollout_id=index,
            plan=("n1",),
            reward=1.0,
            failure=None,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(record, index) for index in range(64)]
        for future in futures:
            future.result()

    saved = json.loads((tmp_path / "evidence.json").read_text())
    assert len(saved) == 64
    assert set(saved) == {f"tree:{index}" for index in range(64)}


def test_dependency_setup_reuses_python_install_and_shared_npm_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, str] | None]] = []

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, **kwargs):
        calls.append((list(command), kwargs.get("env")))
        return Completed()

    monkeypatch.setattr(executor_module.subprocess, "run", fake_run)
    executor = CodexExecutor(dependency_cache=tmp_path / "dependency-cache")

    workspaces = []
    for name in ("a", "b"):
        workspace = tmp_path / name
        workspace.mkdir()
        (workspace / "requirements.txt").write_text("PyYAML==6.0.3\n")
        (workspace / "package-lock.json").write_text("{}")
        workspaces.append(workspace)

    executor._prepare_dependencies(workspaces[0])
    executor._prepare_dependencies(workspaces[1])

    pip_calls = [call for call in calls if call[0][:3] == ["python3", "-m", "pip"]]
    npm_calls = [call for call in calls if call[0][:2] == ["npm", "ci"]]

    assert len(pip_calls) == 1
    assert len(npm_calls) == 2
    npm_caches = {call[1]["npm_config_cache"] for call in npm_calls if call[1] is not None}
    assert len(npm_caches) == 1
    assert "--prefer-offline" in npm_calls[0][0]
    assert "--no-audit" in npm_calls[0][0]
    assert "--no-fund" in npm_calls[0][0]


def test_greedy_baseline_parallelizes_candidate_task_evaluations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graphs = [_graph(f"g{i}") for i in range(4)]
    tasks = [_task(f"t{i}") for i in range(3)]
    executor = _SlowExecutor()

    monkeypatch.setattr(
        baselines_module,
        "_candidate_graphs",
        lambda tasks, proposer, config: graphs,
    )
    monkeypatch.setattr(
        baselines_module,
        "_static",
        lambda graphs, tasks, critic, config: {graph.id: 0.0 for graph in graphs},
    )

    selected = baselines_module.select_dag_baseline(
        Baseline.GREEDY_FORWARD,
        tasks,
        object(),
        object(),
        executor,
        PilotConfig(candidate_graphs=4, forward_workers=3),
    )

    assert executor.max_active >= 2
    assert selected.forward_scores


def test_transient_codex_rate_limit_is_retried_not_scored_as_model_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    class Completed:
        def __init__(self, returncode: int, stderr: str = "") -> None:
            self.returncode = returncode
            self.stdout = ""
            self.stderr = stderr

    def fake_run_process_group(command, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return Completed(1, "HTTP 429 rate limit; please try again")
        return Completed(0)

    monkeypatch.setattr(executor_module, "run_process_group", fake_run_process_group)
    monkeypatch.setattr(executor_module.time, "sleep", lambda _: None)

    executor = CodexExecutor(
        dependency_cache=tmp_path / "dependency-cache",
        max_retries=2,
    )
    result = executor.execute(_task("t-retry"), _graph("g-retry"), ("n1",))

    assert attempts == 2
    assert result["workspace"] == {}


def test_persistent_codex_infrastructure_failure_is_not_converted_to_zero_reward() -> None:
    infrastructure_error = getattr(executor_module, "CodexInfrastructureError")

    class BrokenExecutor:
        def execute(self, task, graph, plan):
            raise infrastructure_error("persistent 429")

    with pytest.raises(infrastructure_error):
        mcts_module.mcts(
            _graph("g-infra"),
            _task("t-infra"),
            BrokenExecutor(),
            budget=1,
        )
