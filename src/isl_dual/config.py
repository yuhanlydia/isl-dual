from dataclasses import dataclass


@dataclass(frozen=True)
class PilotConfig:
    acquisition_tasks: int = 3
    deployment_tasks: int = 3
    candidate_graphs: int = 8
    max_graph_nodes: int = 12
    outer_loops: int = 2
    mcts_budget: int = 8
    c_uct: float = 1.4
    max_plan_length: int = 12
    p_stop: float = 0.4
    beta_artifact: float = 2.0
    beta_forward: float = 4.0
    stability_penalty: float = 0.5
    complexity_penalty: float = 0.1
    graphs_mutated: int = 2
    mutants_per_graph: int = 3
    mutation_probability: float = 0.3
    max_pool: int = 12
    utility_threshold: float = 0.05
    # MCTS remains sequential *within* one tree because later UCT decisions
    # depend on earlier rollout rewards.  Independent (graph, task) trees are
    # scientifically independent and can run concurrently without changing
    # K, rollout budget, seeds, or posterior math.
    forward_workers: int = 3
    seed: int = 20260901
