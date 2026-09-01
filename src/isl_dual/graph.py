from __future__ import annotations

from collections import deque

from .models import Graph


class InvalidGraph(ValueError):
    pass


def validate_graph(graph: Graph, max_nodes: int = 12) -> None:
    if not 2 <= len(graph.nodes) <= max_nodes:
        raise InvalidGraph(f"node count must be in [2, {max_nodes}]")
    ids = [n.id for n in graph.nodes]
    if len(ids) != len(set(ids)):
        raise InvalidGraph("node ids must be unique")
    if any(not n.action.strip() for n in graph.nodes):
        raise InvalidGraph("actions must be non-empty")
    known = set(ids)
    if any(src not in known or dst not in known for src, dst in graph.edges):
        raise InvalidGraph("all edge endpoints must exist")
    if any(src == dst for src, dst in graph.edges):
        raise InvalidGraph("self edges are invalid")

    children = {node_id: [] for node_id in ids}
    indegree = {node_id: 0 for node_id in ids}
    for src, dst in set(graph.edges):
        children[src].append(dst)
        indegree[dst] += 1
    queue = deque(node_id for node_id, degree in indegree.items() if degree == 0)
    visited: set[str] = set()
    while queue:
        current = queue.popleft()
        visited.add(current)
        for child in children[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(visited) != len(ids):
        raise InvalidGraph("graph must be acyclic")

    node_map = graph.node_map()
    for src, dst in graph.edges:
        if node_map[dst].required and not node_map[src].required:
            raise InvalidGraph("mandatory node cannot depend on an optional node")

    seen_members: set[str] = set()
    for group in graph.or_groups:
        members = set(group.members)
        if len(members) < 2 or not members <= known:
            raise InvalidGraph("OR groups need at least two existing members")
        if seen_members & members:
            raise InvalidGraph("a node cannot occur in multiple OR groups")
        seen_members |= members
        if group.required and not any(node_map[m].required for m in members):
            raise InvalidGraph("required OR group must contain a required alternative")


def validate_dedupe(graphs: list[Graph], max_nodes: int = 12) -> list[Graph]:
    accepted: list[Graph] = []
    fingerprints: set[tuple[object, ...]] = set()
    for graph in graphs:
        try:
            validate_graph(graph, max_nodes=max_nodes)
        except InvalidGraph:
            continue
        fingerprint = graph.fingerprint()
        if fingerprint not in fingerprints:
            fingerprints.add(fingerprint)
            accepted.append(graph)
    return accepted

