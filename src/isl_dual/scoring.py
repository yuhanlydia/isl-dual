from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Iterable, Mapping

from .models import Graph, MCTSResult, Utility


def softmax(log_weights: Mapping[str, float]) -> dict[str, float]:
    if not log_weights:
        return {}
    pivot = max(log_weights.values())
    values = {key: math.exp(value - pivot) for key, value in log_weights.items()}
    total = sum(values.values())
    return {key: value / total for key, value in values.items()}


def complexity(graph: Graph) -> float:
    return (len(graph.nodes) + 0.5 * len(graph.edges)) / 18.0


def top2_mean(rewards: Iterable[float]) -> float:
    ordered = sorted(rewards, reverse=True)
    if not ordered:
        return 0.0
    return sum(ordered[:2]) / min(2, len(ordered))


def summarize_forward(task_scores: list[float]) -> tuple[float, float]:
    return statistics.fmean(task_scores), statistics.pstdev(task_scores)


def _utility(with_values: list[float], without_values: list[float]) -> Utility:
    delta = None
    if len(with_values) >= 2 and len(without_values) >= 2:
        delta = statistics.fmean(with_values) - statistics.fmean(without_values)
    return Utility(delta=delta, n_with=len(with_values), n_without=len(without_values))


def estimate_utilities(
    graphs: Mapping[str, Graph], evidence: Mapping[tuple[str, str], MCTSResult]
) -> tuple[dict[tuple[str, str], Utility], dict[tuple[str, tuple[str, str]], Utility]]:
    node_utilities: dict[tuple[str, str], Utility] = {}
    edge_utilities: dict[tuple[str, tuple[str, str]], Utility] = {}
    for graph_id, graph in graphs.items():
        rollouts = [r for (gid, _), result in evidence.items() if gid == graph_id for r in result.rollouts]
        for node in graph.nodes:
            with_values = [r.reward for r in rollouts if node.id in r.plan]
            without_values = [r.reward for r in rollouts if node.id not in r.plan]
            node_utilities[(graph_id, node.id)] = _utility(with_values, without_values)
        for edge in graph.edges:
            src, dst = edge
            with_values = [
                r.reward for r in rollouts
                if src in r.plan and dst in r.plan and r.plan.index(src) < r.plan.index(dst)
            ]
            without_values = [r.reward for r in rollouts if not (
                src in r.plan and dst in r.plan and r.plan.index(src) < r.plan.index(dst)
            )]
            edge_utilities[(graph_id, edge)] = _utility(with_values, without_values)
    return node_utilities, edge_utilities
