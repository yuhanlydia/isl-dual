from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import isl_dual.executor as executor_module
import isl_dual.pipeline as pipeline_module
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
