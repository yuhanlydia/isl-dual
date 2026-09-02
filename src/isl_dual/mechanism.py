"""Corrected six-family mechanism pilot and its diagnostic ablations.

This module is deliberately separate from the exploratory 30-family campaign.
It writes only under the caller-provided namespace (normally ``runs/v1-corrected``)
and resumes completed family/ablation units from atomic JSON manifests.
"""

from __future__ import annotations

import json
import math
import os
import random
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import yaml

from .baselines import _candidate_graphs, deterministic_plan
from .cache import CachedCritic, CachedExecutor, CachedJSONClient, CachedProposer, JSONCache
from .codex_components import CodexCritic, CodexJSON, CodexProposer, graph_from_dict, graph_to_dict
from .compile import compile_graph_to_skill
from .config import PilotConfig
from .experiment import _evaluate_graph, _skill_graph, _with_checkpointed_verifiers, run_family
from .executor import CodexExecutor
from .graph import validate_graph
from .mcts import STOP, dependencies_satisfied, legal_actions, mcts, requirements_satisfied
from .models import AcquisitionTask, Graph, Node, OrGroup
from .pipeline import train_inverse_skill
from .scoring import top2_mean
from .skillevol_host import FamilyBundle, load_family


MECHANISM_FAMILIES = ("E1-LS1", "E2-LS1", "E3-LS1", "E4-LS1", "E5-LS1", "E6-LS1")
SEARCH_BUDGETS = (4, 8, 16)


def _atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str))
    os.replace(temporary, path)


def _components(output: Path, model: str | None):
    cache = JSONCache(output / "cache")
    client = CodexJSON(model=model)
    return (
        CachedProposer(CodexProposer(client), cache),
        CachedCritic(CodexCritic(client), cache),
        CachedExecutor(CodexExecutor(model=model), cache),
    )


def _within_family_permutation(bundle: FamilyBundle) -> FamilyBundle:
    tasks = bundle.acquisition
    if len(tasks) != 3:
        raise ValueError("within-family permutation requires three acquisition tasks")
    shuffled = [replace(task, expert_artifact=tasks[(index + 1) % 3].expert_artifact) for index, task in enumerate(tasks)]
    return FamilyBundle(bundle.family_id, shuffled, bundle.deployment)


def _train_subset(bundle: FamilyBundle, n: int, output: Path, model: str | None, seed: int) -> dict[str, Any]:
    subset = bundle.acquisition[:n]
    config = PilotConfig(acquisition_tasks=n, seed=seed)
    proposer, critic, executor = _components(output, model)
    # Mutator is imported lazily to keep this module's top-level import graph small.
    from .cache import CachedMutator
    from .codex_components import CodexMutator
    mutator = CachedMutator(CodexMutator(critic.inner.client), JSONCache(output / "cache"))
    result = train_inverse_skill(subset, proposer, critic, executor, mutator, config=config)
    scores = _evaluate_graph(_skill_graph("learning-curve", result.skill), ("skill", "verify"), bundle.deployment, executor)
    return {"n_acquisition": n, "winner": result.graph.id, "posterior_entropy": _entropy(result.posterior), "deployment_scores": scores, "mean_reward": statistics.fmean(scores.values())}


def _entropy(values: dict[str, float]) -> float:
    return -sum(value * math.log(value) for value in values.values() if value > 0.0)


def _random_plan(graph: Graph, seed: int, max_len: int = 12) -> tuple[str, ...]:
    rng = random.Random(seed)
    plan: list[str] = []
    while len(plan) < max_len:
        actions = [action for action in legal_actions(graph, tuple(plan), max_len) if action != STOP]
        if requirements_satisfied(graph, set(plan)) and (not actions or rng.random() < 0.4):
            return tuple(plan)
        required = [action for action in actions if graph.node_map()[action].required]
        if required:
            plan.append(rng.choice(required))
        elif actions:
            plan.append(rng.choice(actions))
        else:
            break
    if requirements_satisfied(graph, set(plan)):
        return tuple(plan)
    raise RuntimeError("random sampler could not complete a legal plan")


def _search_ablation(bundle: FamilyBundle, output: Path, model: str | None, seed: int) -> dict[str, Any]:
    proposer, critic, executor = _components(output, model)
    config = PilotConfig(seed=seed)
    graphs = _candidate_graphs(bundle.acquisition, proposer, config)
    deployment_scores: dict[str, float] = {}
    for graph in graphs:
        deployment_scores[graph.id] = statistics.fmean(_evaluate_graph(_skill_graph("candidate", compile_graph_to_skill(graph)), ("skill", "verify"), bundle.deployment, executor).values())
    methods: dict[str, dict[str, dict[str, float]]] = {}
    for budget in SEARCH_BUDGETS:
        methods[str(budget)] = {"random": {}, "greedy": {}, "mcts": {}}
        for graph_index, graph in enumerate(graphs):
            random_rewards: list[float] = []
            greedy_rewards: list[float] = []
            mcts_rewards: list[float] = []
            plan = deterministic_plan(graph)
            for task_index, task in enumerate(bundle.acquisition):
                for rollout_id in range(budget):
                    random_rewards.append(float(task.verifier(executor.execute(task, graph, _random_plan(graph, seed + graph_index * 1000 + task_index * 100 + rollout_id)))) )
                    greedy_rewards.append(float(task.verifier(executor.execute(task, graph, plan))))
                mcts_rewards.extend(mcts(graph, task, executor, budget=budget, c_uct=config.c_uct, max_plan_length=config.max_plan_length, p_stop=config.p_stop, seed=seed + graph_index * 101 + task_index).rewards)
            methods[str(budget)]["random"][graph.id] = max(random_rewards) if random_rewards else 0.0
            methods[str(budget)]["greedy"][graph.id] = max(greedy_rewards) if greedy_rewards else 0.0
            methods[str(budget)]["mcts"][graph.id] = top2_mean(mcts_rewards) if mcts_rewards else 0.0
    return {"budgets": methods, "candidate_deployment_scores": deployment_scores, "candidate_graphs": {graph.id: graph_to_dict(graph) for graph in graphs}}


def _corruptions(graph: Graph) -> dict[str, Graph]:
    result: dict[str, Graph] = {}
    optional = [node for node in graph.nodes if not node.required]
    if optional:
        removed = {optional[0].id}
        candidate = replace(graph, id=graph.id + "-shortcut", nodes=tuple(node for node in graph.nodes if node.id not in removed), edges=tuple(edge for edge in graph.edges if not (set(edge) & removed)), or_groups=tuple(group for group in graph.or_groups if not (set(group.members) & removed)))
        try:
            validate_graph(candidate); result["C1_shortcut"] = candidate
        except Exception:
            pass
    specific = Node("spurious_specific", "Acquisition-specific shortcut", (), (), "Use only the exact filename and constants from one acquisition example", ("spurious_done",), "The copied case-specific shortcut was used", False)
    candidate = replace(graph, id=graph.id + "-over-specific", nodes=graph.nodes + (specific,))
    try:
        validate_graph(candidate); result["C2_over_specific"] = candidate
    except Exception:
        pass
    redundant = Node("redundant", "Perform an unnecessary extra formatting pass", (), (), "Repeat a completed operation without changing the artifact", ("redundant_done",), "The redundant pass completed", False)
    candidate = replace(graph, id=graph.id + "-redundant", nodes=graph.nodes + (redundant,))
    try:
        validate_graph(candidate); result["C4_redundant"] = candidate
    except Exception:
        pass
    if graph.edges:
        edge = graph.edges[-1]
        candidate = replace(graph, id=graph.id + "-order", edges=tuple(item for item in graph.edges if item != edge))
        try:
            validate_graph(candidate); result["C3_order_violation"] = candidate
        except Exception:
            pass
    return result


def _spurious_ablation(bundle: FamilyBundle, primary: dict[str, Any], output: Path, model: str | None, seed: int) -> dict[str, Any]:
    graph = graph_from_dict(primary["graph"], primary["graph"]["id"])
    proposer, critic, executor = _components(output, model)
    scores: dict[str, Any] = {}
    for name, candidate in {"clean": graph, **_corruptions(graph)}.items():
        artifact = critic.score(candidate, bundle.acquisition).artifact_score
        forward = []
        for index, task in enumerate(bundle.acquisition):
            forward.append(top2_mean(mcts(candidate, task, executor, budget=8, c_uct=1.4, max_plan_length=12, p_stop=0.4, seed=seed + index).rewards))
        deployment = _evaluate_graph(_skill_graph("spurious", compile_graph_to_skill(candidate)), ("skill", "verify"), bundle.deployment, executor)
        scores[name] = {"static_artifact_score": artifact, "forward_score": statistics.fmean(forward), "deployment_score": statistics.fmean(deployment.values())}
    return scores


def run_mechanism_pilot(benchmark_root: Path, output: Path, model: str | None = None, families: Iterable[str] = MECHANISM_FAMILIES) -> dict[str, Any]:
    """Run/resume the six-family corrected mechanism pilot and diagnostics."""
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "mechanism.json"
    state = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"families": {}, "config": {"families": list(families), "model": model}}
    for family_id in families:
        family_out = output / "families" / family_id
        entry = state["families"].setdefault(family_id, {})
        result_path = family_out / "result.json"
        if not result_path.exists():
            try:
                result = run_family(benchmark_root, family_id, family_out, model, PilotConfig())
                entry["primary"] = "completed"
                _atomic(manifest_path, state)
            except Exception as error:
                entry.update(primary="failed", error=f"{type(error).__name__}: {error}")
                _atomic(manifest_path, state)
                continue
        else:
            result = json.loads(result_path.read_text())
        bundle = _with_checkpointed_verifiers(load_family(benchmark_root, family_id, artifact_cache=family_out / "artifacts"), JSONCache(family_out / "cache"))
        within_out = output / "ablations" / "within_family_shuffle" / family_id
        if not (within_out / "result.json").exists():
            try:
                run_family(benchmark_root, family_id, within_out, model, PilotConfig(), bundle_override=_within_family_permutation(bundle))
                entry["within_family_shuffle"] = "completed"
                _atomic(manifest_path, state)
            except Exception as error:
                entry.update(within_family_shuffle="failed", within_family_shuffle_error=f"{type(error).__name__}: {error}")
                _atomic(manifest_path, state)
        curve_dir = output / "ablations" / "learning_curve" / family_id
        curve_path = curve_dir / "result.json"
        if not curve_path.exists():
            curve_dir.mkdir(parents=True, exist_ok=True)
            curve = [_train_subset(bundle, n, curve_dir / f"n-{n}", model, PilotConfig().seed + n) for n in (1, 2, 3)]
            _atomic(curve_path, curve); entry["learning_curve"] = "completed"; _atomic(manifest_path, state)
        search_path = output / "ablations" / "search" / family_id / "result.json"
        if not search_path.exists():
            search_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic(search_path, _search_ablation(bundle, output / "ablations" / "search" / family_id, model, PilotConfig().seed))
            entry["search"] = "completed"; _atomic(manifest_path, state)
        spurious_path = output / "ablations" / "spurious" / family_id / "result.json"
        if not spurious_path.exists():
            spurious_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic(spurious_path, _spurious_ablation(bundle, result, output / "ablations" / "spurious" / family_id, model, PilotConfig().seed))
            entry["spurious"] = "completed"; _atomic(manifest_path, state)
    return state


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run the corrected six-family ISL-Dual mechanism pilot")
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.4")
    args = parser.parse_args()
    print(json.dumps(run_mechanism_pilot(args.benchmark_root, args.output, args.model), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
