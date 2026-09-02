from __future__ import annotations

import statistics
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .codex_components import CodexJSON, observable_artifact
from .compile import compile_graph_to_skill
from .config import PilotConfig
from .graph import validate_dedupe
from .mcts import STOP, legal_actions, mcts, requirements_satisfied
from .models import AcquisitionTask, Critic, Executor, Graph, Proposer
from .pipeline import PROPOSAL_MODES
from .scoring import complexity, softmax, top2_mean


class Baseline(str, Enum):
    NO_SKILL = "B0_no_skill"
    DIRECT_TEXT = "B1_direct_text_skill"
    ONE_SHOT_DAG = "B2_one_shot_dag"
    STATIC_CRITIC = "B3_static_critic"
    GREEDY_FORWARD = "B4_greedy_forward"
    MCTS_FORWARD = "B5_mcts_forward"
    ISL_DUAL = "B6_isl_dual"
    ORACLE_SOLUTION = "B7_oracle_solution_procedure"
    # Backward-compatible symbol; SkillEvolBench's solve.sh is an oracle
    # solution procedure, not an observed agent trajectory.
    FULL_TRAJECTORY = ORACLE_SOLUTION
    CURATED_SKILL = "B8_curated_skill"


@dataclass(frozen=True)
class SelectedSkill:
    baseline: Baseline
    skill: str | None
    graph: Graph | None
    posterior: dict[str, float]
    forward_scores: dict[str, float]


DIRECT_SKILL_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["skill"],
    "properties": {"skill": {"type": "string"}},
}


def direct_text_skill(tasks: list[AcquisitionTask], client: CodexJSON) -> SelectedSkill:
    import json
    observed = [{"task": task.x, "successful_final_artifact": observable_artifact(task.expert_artifact)} for task in tasks]
    prompt = "Write a portable SKILL.md containing only reusable procedural knowledge inferred from these outcome-only examples. Do not infer chain-of-thought and do not copy case-specific answers, filenames, constants, or output content. Include Preconditions, Procedure, Failure checks, and Stop condition.\n" + json.dumps(observed)
    value = client.call(prompt, DIRECT_SKILL_SCHEMA)
    return SelectedSkill(Baseline.DIRECT_TEXT, str(value["skill"]), None, {}, {})


def full_trajectory_skill(
    tasks: list[AcquisitionTask], trajectories: list[str], client: CodexJSON,
) -> SelectedSkill:
    """B7 upper-information oracle-solution baseline.

    The benchmark's ``solution/solve.sh`` is executable oracle procedure
    information, not a recorded agent execution trajectory.
    """
    import json
    if len(tasks) != len(trajectories) or not all(item.strip() for item in trajectories):
        raise ValueError("B7 requires one non-empty expert trajectory per acquisition task")
    observed = [
        {"task": task.x, "oracle_solution_procedure": trajectory}
        for task, trajectory in zip(tasks, trajectories, strict=True)
    ]
    prompt = (
        "Write a portable SKILL.md containing reusable procedural knowledge from the "
        "explicit oracle solution procedures below. Abstract away task-specific answers, "
        "filenames, constants, and output content. Include Preconditions, Procedure, Failure "
        "checks, and Stop condition.\n" + json.dumps(observed)
    )
    value = client.call(prompt, DIRECT_SKILL_SCHEMA)
    return SelectedSkill(Baseline.FULL_TRAJECTORY, str(value["skill"]), None, {}, {})


def _candidate_graphs(tasks: list[AcquisitionTask], proposer: Proposer, config: PilotConfig) -> list[Graph]:
    graphs: list[Graph] = []
    attempts = 0
    while len(graphs) < config.candidate_graphs and attempts < 4:
        count = 2 + attempts
        graphs.extend(graph for mode in PROPOSAL_MODES for graph in proposer.propose(tasks, mode, count))
        graphs = validate_dedupe(graphs, config.max_graph_nodes)
        attempts += 1
    if len(graphs) < config.candidate_graphs:
        raise RuntimeError(f"expected {config.candidate_graphs} valid distinct candidates, got {len(graphs)}")
    return graphs[:config.candidate_graphs]


def _static(graphs: list[Graph], tasks: list[AcquisitionTask], critic: Critic, config: PilotConfig) -> dict[str, float]:
    return {graph.id: config.beta_artifact * critic.score(graph, tasks).artifact_score - config.complexity_penalty * complexity(graph) for graph in graphs}


def deterministic_plan(graph: Graph) -> tuple[str, ...]:
    plan: list[str] = []
    max_len = 12
    while True:
        actions = legal_actions(graph, tuple(plan), max_len)
        non_stop = sorted(action for action in actions if action != STOP)
        required = [action for action in non_stop if graph.node_map()[action].required]
        if required:
            plan.append(required[0])
            continue
        if requirements_satisfied(graph, set(plan)):
            return tuple(plan)
        if non_stop:
            plan.append(non_stop[0])
            continue
        raise RuntimeError("required nodes cannot be topologically completed")


def select_dag_baseline(
    baseline: Baseline, tasks: list[AcquisitionTask], proposer: Proposer,
    critic: Critic, executor: Executor, config: PilotConfig | None = None,
) -> SelectedSkill:
    config = config or PilotConfig()
    if baseline == Baseline.ONE_SHOT_DAG:
        graph = proposer.propose(tasks, "minimal", 1)[0]
        return SelectedSkill(baseline, compile_graph_to_skill(graph), graph, {graph.id: 1.0}, {})
    if baseline not in {Baseline.STATIC_CRITIC, Baseline.GREEDY_FORWARD, Baseline.MCTS_FORWARD}:
        raise ValueError(f"unsupported DAG selection baseline: {baseline}")
    graphs = _candidate_graphs(tasks, proposer, config)
    weights = _static(graphs, tasks, critic, config)
    forward: dict[str, float] = {}
    if baseline != Baseline.STATIC_CRITIC:
        for graph_index, graph in enumerate(graphs):
            task_scores = []
            for task_index, task in enumerate(tasks):
                if baseline == Baseline.GREEDY_FORWARD:
                    output = executor.execute(task, graph, deterministic_plan(graph))
                    score = float(task.verifier(output))
                else:
                    result = mcts(graph, task, executor, config.mcts_budget, config.c_uct, config.max_plan_length, config.p_stop, config.seed + graph_index * 101 + task_index)
                    score = top2_mean(result.rewards)
                task_scores.append(score)
            forward[graph.id] = statistics.fmean(task_scores)
            stability = statistics.pstdev(task_scores)
            weights[graph.id] += config.beta_forward * forward[graph.id] - config.stability_penalty * stability
    posterior = softmax(weights)
    winner = max(graphs, key=lambda graph: posterior[graph.id])
    return SelectedSkill(baseline, compile_graph_to_skill(winner), winner, posterior, forward)


def upper_information_skill(baseline: Baseline, skill_text: str) -> SelectedSkill:
    if baseline not in {Baseline.ORACLE_SOLUTION, Baseline.CURATED_SKILL}:
        raise ValueError("upper-information helper is only for B7/B8")
    if not skill_text.strip():
        raise ValueError("upper-information skill must be explicitly supplied")
    return SelectedSkill(baseline, skill_text, None, {}, {})
