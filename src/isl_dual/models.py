from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Mapping, Protocol


@dataclass(frozen=True)
class Node:
    id: str
    name: str
    preconditions: tuple[str, ...]
    inputs: tuple[str, ...]
    action: str
    outputs: tuple[str, ...]
    validator: str
    required: bool = True


@dataclass(frozen=True)
class OrGroup:
    id: str
    members: tuple[str, ...]
    required: bool = False


@dataclass(frozen=True)
class Graph:
    id: str
    nodes: tuple[Node, ...]
    edges: tuple[tuple[str, str], ...] = ()
    or_groups: tuple[OrGroup, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def node_map(self) -> dict[str, Node]:
        return {node.id: node for node in self.nodes}

    def parents(self, node_id: str) -> set[str]:
        return {src for src, dst in self.edges if dst == node_id}

    def fingerprint(self) -> tuple[Any, ...]:
        nodes = tuple(sorted((n.name, n.action, n.required) for n in self.nodes))
        edges = tuple(sorted(self.edges))
        groups = tuple(sorted(tuple(sorted(g.members)) for g in self.or_groups))
        return nodes, edges, groups


@dataclass(frozen=True)
class AcquisitionTask:
    id: str
    x: str
    expert_artifact: Any
    verifier: Callable[[Any], float]
    workspace_source: str | None = None


@dataclass(frozen=True)
class DeploymentTask:
    id: str
    x: str
    verifier: Callable[[Any], float]
    workspace_source: str | None = None


@dataclass(frozen=True)
class CriticScore:
    sufficiency: float
    transfer: float
    consistency: float

    @property
    def artifact_score(self) -> float:
        return 0.4 * self.sufficiency + 0.4 * self.transfer + 0.2 * self.consistency


@dataclass(frozen=True)
class Rollout:
    plan: tuple[str, ...]
    reward: float
    output: Any = None
    failure: str | None = None


@dataclass
class MCTSResult:
    rollouts: list[Rollout]

    @property
    def rewards(self) -> list[float]:
        return [r.reward for r in self.rollouts]


@dataclass(frozen=True)
class Utility:
    delta: float | None
    n_with: int
    n_without: int


@dataclass
class TrainingResult:
    skill: str
    graph: Graph
    posterior: dict[str, float]
    q0: dict[str, float]
    q1: dict[str, float]
    forward_scores: dict[str, float]
    evidence: dict[tuple[str, str], MCTSResult]


class Proposer(Protocol):
    def propose(self, tasks: list[AcquisitionTask], mode: str, count: int) -> list[Graph]: ...


class Critic(Protocol):
    def score(self, graph: Graph, tasks: list[AcquisitionTask]) -> CriticScore: ...


class Executor(Protocol):
    def execute(self, task: AcquisitionTask | DeploymentTask, graph: Graph, plan: tuple[str, ...]) -> Any: ...


class Mutator(Protocol):
    def mutate(
        self,
        graph: Graph,
        evidence: Mapping[tuple[str, str], MCTSResult],
        node_utilities: Mapping[tuple[str, str], Utility],
        edge_utilities: Mapping[tuple[str, tuple[str, str]], Utility],
        count: int,
    ) -> list[Graph]: ...


def clone_graph(graph: Graph, **changes: Any) -> Graph:
    return replace(graph, **changes)

