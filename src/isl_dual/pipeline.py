from __future__ import annotations

import math
from collections.abc import Mapping

from .compile import compile_graph_to_skill, operational_pruning
from .config import PilotConfig
from .graph import validate_dedupe, validate_graph
from .leakage import SecretBundle, assert_inverse_input
from .mcts import EvidenceJournal, mcts
from .models import AcquisitionTask, Critic, Executor, Graph, MCTSResult, Mutator, Proposer, TrainingResult
from .scoring import complexity, estimate_utilities, softmax, summarize_forward, top2_mean

PROPOSAL_MODES = ("minimal", "verification_first", "failure_aware", "transfer_first")


def _mutation_prior(
    q1: Mapping[str, float],
    pool: list[Graph],
    mutation_probability: float = 0.3,
) -> dict[str, float]:
    """Allocate q1 mass over retained parents and their retained mutants.

    A mutated parent keeps (1-mu) of its mass and its retained children share
    mu of that mass.  If pool truncation removes a child, the mutation mass is
    divided among the children actually retained, preserving total support
    mass rather than silently losing probability.
    """
    if not 0.0 <= mutation_probability <= 1.0:
        raise ValueError("mutation_probability must be in [0, 1]")
    pool_ids = {graph.id for graph in pool}
    children: dict[str, list[Graph]] = {}
    for graph in pool:
        parent_id = str(graph.metadata.get("parent_id", graph.id))
        if parent_id != graph.id:
            children.setdefault(parent_id, []).append(graph)

    priors: dict[str, float] = {}
    for graph in pool:
        parent_id = str(graph.metadata.get("parent_id", graph.id))
        parent_mass = float(q1.get(parent_id, q1.get(graph.id, 0.0)))
        retained_children = children.get(parent_id, [])
        if parent_id == graph.id:
            if retained_children:
                priors[graph.id] = (1.0 - mutation_probability) * parent_mass
            else:
                priors[graph.id] = parent_mass
        elif retained_children:
            priors[graph.id] = mutation_probability * parent_mass / len(retained_children)
        else:
            priors[graph.id] = parent_mass

    # A malformed/truncated pool must not create a zero-mass posterior.  Keep
    # any q1 support whose parent survived but had no retained child intact.
    for graph_id, mass in q1.items():
        if graph_id in pool_ids and graph_id not in priors:
            priors[graph_id] = float(mass)
    total = sum(priors.values())
    if total <= 0.0:
        raise ValueError("mutation prior has no positive mass")
    return {graph_id: mass / total for graph_id, mass in priors.items()}


def _forward_loop(
    graphs: list[Graph],
    tasks: list[AcquisitionTask],
    executor: Executor,
    config: PilotConfig,
    seed_offset: int,
    journal: EvidenceJournal | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[tuple[str, str], MCTSResult]]:
    forward: dict[str, float] = {}
    stability: dict[str, float] = {}
    evidence: dict[tuple[str, str], MCTSResult] = {}
    for graph_index, graph in enumerate(graphs):
        scores: list[float] = []
        for task_index, task in enumerate(tasks):
            result = mcts(
                graph, task, executor,
                budget=config.mcts_budget,
                c_uct=config.c_uct,
                max_plan_length=config.max_plan_length,
                p_stop=config.p_stop,
                seed=config.seed + seed_offset + graph_index * 101 + task_index,
                journal=journal,
                journal_prefix=f"{'round1' if seed_offset == 0 else 'round2'}:{graph.id}:{task.id}",
            )
            evidence[(graph.id, task.id)] = result
            scores.append(top2_mean(result.rewards))
        forward[graph.id], stability[graph.id] = summarize_forward(scores)
    return forward, stability, evidence


def train_inverse_skill(
    acquisition_tasks: list[AcquisitionTask],
    proposer: Proposer,
    critic: Critic,
    executor: Executor,
    mutator: Mutator,
    config: PilotConfig | None = None,
    evidence_journal: EvidenceJournal | None = None,
) -> TrainingResult:
    config = config or PilotConfig()
    if config.outer_loops != 2:
        raise ValueError("this implementation fixes T_outer=2 for the pilot")
    if len(acquisition_tasks) != config.acquisition_tasks:
        raise ValueError(f"pilot requires exactly {config.acquisition_tasks} acquisition tasks")
    inverse_payload = [(task.x, task.expert_artifact) for task in acquisition_tasks]
    assert_inverse_input(inverse_payload, SecretBundle())

    proposed: list[Graph] = []
    attempts = 0
    while len(proposed) < config.candidate_graphs and attempts < 4:
        per_mode_count = 2 + attempts
        for mode in PROPOSAL_MODES:
            proposed.extend(proposer.propose(acquisition_tasks, mode, per_mode_count))
        proposed = validate_dedupe(proposed, config.max_graph_nodes)
        attempts += 1
    if len(proposed) < config.candidate_graphs:
        raise RuntimeError("proposer failed to produce 8 valid, distinct DAGs")
    graphs = proposed[:config.candidate_graphs]

    log_weights: dict[str, float] = {}
    artifact_scores: dict[str, float] = {}
    for graph in graphs:
        artifact = critic.score(graph, acquisition_tasks).artifact_score
        artifact_scores[graph.id] = artifact
        log_weights[graph.id] = (
            config.beta_artifact * artifact
            - config.complexity_penalty * complexity(graph)
        )
    q0 = softmax(log_weights)

    forward1, stability1, evidence1 = _forward_loop(graphs, acquisition_tasks, executor, config, 0, evidence_journal)
    for graph in graphs:
        log_weights[graph.id] += (
            config.beta_forward * forward1[graph.id]
            - config.stability_penalty * stability1[graph.id]
        )
    q1 = softmax(log_weights)
    graph_map = {graph.id: graph for graph in graphs}
    node_utils, edge_utils = estimate_utilities(graph_map, evidence1)

    top_ids = sorted(q1, key=q1.get, reverse=True)[:config.graphs_mutated]
    mutant_groups: list[list[Graph]] = []
    for graph_id in top_ids:
        generated = mutator.mutate(
            graph_map[graph_id], evidence1, node_utils, edge_utils,
            config.mutants_per_graph,
        )
        mutant_groups.append(generated)
    # Round-robin avoids the K_max truncation retaining all mutants from only
    # the first posterior parent.
    mutants = [
        group[index]
        for index in range(config.mutants_per_graph)
        for group in mutant_groups
        if index < len(group)
    ]
    pool = validate_dedupe(graphs + mutants, config.max_graph_nodes)[:config.max_pool]

    # Mutants inherit their parent's evidence weight; this is explicit metadata, not artifact re-critique.
    mutation_priors = _mutation_prior(q1, pool, config.mutation_probability)
    forward2, stability2, evidence2 = _forward_loop(pool, acquisition_tasks, executor, config, 100_000, evidence_journal)
    second_weights = {
        graph.id: math.log(mutation_priors[graph.id])
        + config.beta_forward * forward2[graph.id]
        - config.stability_penalty * stability2[graph.id]
        for graph in pool
    }
    q2 = softmax(second_weights)
    winner = max(pool, key=lambda graph: q2[graph.id])

    all_evidence: dict[tuple[str, str], MCTSResult] = {}
    for key in set(evidence1) | set(evidence2):
        combined_rollouts = []
        if key in evidence1:
            combined_rollouts.extend(evidence1[key].rollouts)
        if key in evidence2:
            combined_rollouts.extend(evidence2[key].rollouts)
        all_evidence[key] = MCTSResult(rollouts=combined_rollouts)
    pool_map = {graph.id: graph for graph in pool}
    final_node_utils, _ = estimate_utilities(pool_map, all_evidence)
    winner = operational_pruning(winner, final_node_utils, config.utility_threshold)
    validate_graph(winner, config.max_graph_nodes)
    skill = compile_graph_to_skill(winner)
    _leakage_audit_skill(skill, acquisition_tasks)
    return TrainingResult(
        skill=skill,
        graph=winner,
        posterior=q2,
        q0=q0,
        q1=q1,
        forward_scores=forward2,
        artifact_scores=artifact_scores,
        candidates={graph.id: graph for graph in pool},
        evidence=all_evidence,
    )


def _leakage_audit_skill(skill: str, tasks: list[AcquisitionTask]) -> None:
    for task in tasks:
        artifact = task.expert_artifact
        if isinstance(artifact, str) and artifact and artifact in skill:
            raise AssertionError(f"acquisition artifact leaked into compiled skill: {task.id}")
        if isinstance(artifact, dict):
            files = artifact.get("delta", artifact.get("files", {}))
            if isinstance(files, dict):
                for path, content in files.items():
                    if str(path).split("/")[-1] in skill:
                        raise AssertionError(f"acquisition filename leaked into compiled skill: {task.id}:{path}")
                    if isinstance(content, str):
                        for line in content.splitlines():
                            stripped = line.strip()
                            if len(stripped) >= 80 and stripped in skill:
                                raise AssertionError(f"acquisition content leaked into compiled skill: {task.id}:{path}")
