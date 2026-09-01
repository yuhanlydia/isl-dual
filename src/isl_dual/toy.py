from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from .models import AcquisitionTask, CriticScore, Graph, MCTSResult, Node, Utility


def node(node_id: str, action: str, required: bool = True) -> Node:
    return Node(
        id=node_id, name=action.title(), preconditions=("A task is available",),
        inputs=("task",), action=action, outputs=(node_id + "_done",),
        validator=f"Confirm that {action} completed", required=required,
    )


class ToyProposer:
    def __init__(self) -> None:
        self.counter = 0

    def propose(self, tasks: list[AcquisitionTask], mode: str, count: int) -> list[Graph]:
        graphs = []
        for variant in range(count):
            self.counter += 1
            inspect = node("inspect", "inspect the observable input")
            solve = node("solve", "derive a general solution")
            strategy = mode.replace("_", " ")
            verify = node(
                "verify",
                f"verify the result against the task requirements using {strategy} strategy {variant + 1}",
                required=False,
            )
            distract = node(
                "guess", f"guess without checking using {strategy} strategy {variant + 1}",
                required=False,
            )
            optional = verify if (self.counter % 3) else distract
            graphs.append(Graph(
                id=f"{mode}-{self.counter}", nodes=(inspect, solve, optional),
                edges=(("inspect", "solve"),), metadata={"mode": mode},
            ))
        return graphs


class ToyCritic:
    def score(self, graph: Graph, tasks: list[AcquisitionTask]) -> CriticScore:
        # Deliberately makes a plausible but bad "guess" graph look strong.
        plausible = 0.95 if any(n.id == "guess" for n in graph.nodes) else 0.75
        return CriticScore(plausible, 0.8, 0.9)


class ToyExecutor:
    def execute(self, task: AcquisitionTask, graph: Graph, plan: tuple[str, ...]) -> dict[str, object]:
        return {"solved": "solve" in plan, "verified": "verify" in plan, "guessed": "guess" in plan}


class ToyMutator:
    def mutate(
        self, graph: Graph, evidence: Mapping[tuple[str, str], MCTSResult],
        node_utilities: Mapping[tuple[str, str], Utility],
        edge_utilities: Mapping[tuple[str, tuple[str, str]], Utility], count: int,
    ) -> list[Graph]:
        mutants = []
        for index in range(count):
            nodes = tuple(n for n in graph.nodes if n.id != "guess")
            if any(n.id == "verify" for n in nodes):
                nodes = tuple(
                    replace(n, action=n.action + f"; inspect failure evidence variant {index + 1}")
                    if n.id == "verify" else n
                    for n in nodes
                )
            else:
                nodes += (node("verify", "verify the result against the task requirements", required=False),)
            mutants.append(replace(
                graph, id=f"{graph.id}-m{index}", nodes=nodes,
                metadata={"parent_id": graph.id, "mutation": "CHANGE_BRANCH"},
            ))
        return mutants


def toy_tasks() -> list[AcquisitionTask]:
    tasks = []
    for index in range(3):
        tasks.append(AcquisitionTask(
            id=f"T{index + 1}", x=f"Solve generic task {index + 1}",
            expert_artifact=f"private-artifact-{index + 1}",
            verifier=lambda output: 1.0 if output["solved"] and output["verified"] and not output["guessed"] else 0.2,
        ))
    return tasks
