from __future__ import annotations

import random
from dataclasses import replace

from .graph import InvalidGraph, validate_graph
from .models import AcquisitionTask, Graph


def artifact_shuffle(families: list[list[AcquisitionTask]], seed: int) -> list[list[AcquisitionTask]]:
    """Pair each family's inputs with artifacts from a different family.

    Task inputs, verifiers, and workspaces stay fixed. Families must have the same number
    of acquisition tasks. A cyclic derangement is used so no family retains its artifacts.
    """
    if len(families) < 2:
        raise ValueError("artifact shuffle requires at least two families")
    width = len(families[0])
    if width == 0 or any(len(family) != width for family in families):
        raise ValueError("all families must have the same non-zero acquisition width")
    rng = random.Random(seed)
    shift = rng.randrange(1, len(families))
    shuffled = []
    for index, family in enumerate(families):
        donor = families[(index + shift) % len(families)]
        shuffled.append([replace(task, expert_artifact=donor[position].expert_artifact) for position, task in enumerate(family)])
    return shuffled


def edge_shuffle(graph: Graph, seed: int, attempts: int = 100) -> Graph:
    """Randomize only E while preserving nodes, edge count, and DAG validity."""
    rng = random.Random(seed)
    ids = [node.id for node in graph.nodes]
    original = set(graph.edges)
    for _ in range(attempts):
        order = ids[:]; rng.shuffle(order)
        possible = [(order[i], order[j]) for i in range(len(order)) for j in range(i + 1, len(order))]
        rng.shuffle(possible)
        edges = tuple(possible[:len(graph.edges)])
        if set(edges) == original:
            continue
        candidate = replace(graph, id=graph.id + "-edge-shuffle", edges=edges, metadata={**graph.metadata, "control": "edge_shuffle"})
        try:
            validate_graph(candidate)
        except InvalidGraph:
            continue
        return candidate
    raise RuntimeError("could not produce a valid distinct edge shuffle")

