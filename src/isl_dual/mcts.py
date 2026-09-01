from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .models import AcquisitionTask, Executor, Graph, MCTSResult, Rollout

STOP = "__STOP__"


@dataclass
class TreeState:
    prefix: tuple[str, ...]
    visits: int = 0
    action_visits: dict[str, int] = field(default_factory=dict)
    action_values: dict[str, float] = field(default_factory=dict)
    children: dict[str, "TreeState"] = field(default_factory=dict)


def legal_actions(graph: Graph, prefix: tuple[str, ...], max_len: int) -> list[str]:
    used = set(prefix)
    actions = [
        node.id
        for node in graph.nodes
        if node.id not in used and graph.parents(node.id) <= used
    ]
    required = {node.id for node in graph.nodes if node.required}
    if required <= used:
        actions.append(STOP)
    if len(prefix) >= max_len:
        return [STOP] if required <= used else []
    return actions


def _uct(state: TreeState, action: str, c_uct: float) -> float:
    q = state.action_values.get(action, 0.0)
    n_action = state.action_visits.get(action, 0)
    return q + c_uct * math.sqrt(math.log(state.visits + 1) / (n_action + 1))


def _complete_plan(
    graph: Graph,
    prefix: tuple[str, ...],
    rng: random.Random,
    max_len: int,
    p_stop: float,
) -> tuple[str, ...]:
    plan = list(prefix)
    required = {node.id for node in graph.nodes if node.required}
    while len(plan) < max_len:
        actions = [a for a in legal_actions(graph, tuple(plan), max_len) if a != STOP]
        required_available = [a for a in actions if a in required]
        if required_available:
            plan.append(rng.choice(required_available))
            continue
        if required <= set(plan) and (not actions or rng.random() < p_stop):
            break
        if not actions:
            break
        plan.append(rng.choice(actions))
    return tuple(plan)


def mcts(
    graph: Graph,
    task: AcquisitionTask,
    executor: Executor,
    budget: int = 8,
    c_uct: float = 1.4,
    max_plan_length: int = 12,
    p_stop: float = 0.4,
    seed: int = 0,
) -> MCTSResult:
    rng = random.Random(seed)
    root = TreeState(prefix=())
    rollouts: list[Rollout] = []

    for _ in range(budget):
        state = root
        path: list[tuple[TreeState, str]] = []
        while True:
            actions = legal_actions(graph, state.prefix, max_plan_length)
            if not actions or actions == [STOP]:
                break
            unvisited = [a for a in actions if a not in state.children and a != STOP]
            if unvisited:
                action = rng.choice(unvisited)
                child = TreeState(prefix=state.prefix + (action,))
                state.children[action] = child
                path.append((state, action))
                state = child
                break
            action = max(actions, key=lambda a: _uct(state, a, c_uct))
            if action == STOP:
                break
            path.append((state, action))
            state = state.children[action]

        plan = _complete_plan(graph, state.prefix, rng, max_plan_length, p_stop)
        output = executor.execute(task, graph, plan)
        reward = max(0.0, min(1.0, float(task.verifier(output))))
        rollouts.append(Rollout(plan=plan, reward=reward, output=output))

        for parent, action in path:
            parent.visits += 1
            count = parent.action_visits.get(action, 0) + 1
            old = parent.action_values.get(action, 0.0)
            parent.action_visits[action] = count
            parent.action_values[action] = old + (reward - old) / count

    return MCTSResult(rollouts=rollouts)
