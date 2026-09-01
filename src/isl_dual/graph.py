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
    if len(graph.edges) != len(set(graph.edges)):
        raise InvalidGraph("duplicate edges are invalid")

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
    seen_members: set[str] = set()
    for group in graph.or_groups:
        members = set(group.members)
        if len(members) < 2 or not members <= known:
            raise InvalidGraph("OR groups need at least two existing members")
        if seen_members & members:
            raise InvalidGraph("a node cannot occur in multiple OR groups")
        seen_members |= members

    member_to_group = {member: group for group in graph.or_groups for member in group.members}
    parents_by_child = {node_id: {src for src, dst in graph.edges if dst == node_id} for node_id in ids}
    for node in graph.nodes:
        if not node.required:
            continue
        for parent in parents_by_child[node.id]:
            if node_map[parent].required:
                continue
            group = member_to_group.get(parent)
            grouped_parents = parents_by_child[node.id] & set(group.members) if group else set()
            if len(grouped_parents) < 2:
                raise InvalidGraph("mandatory node cannot depend on an excluded optional node")


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


def topological_node_ids(graph: Graph) -> list[str]:
    """Stable topological ordering using original node order as the tie-breaker."""
    position = {node.id: index for index, node in enumerate(graph.nodes)}
    indegree = {node.id: 0 for node in graph.nodes}
    children = {node.id: [] for node in graph.nodes}
    for source, target in graph.edges:
        indegree[target] += 1
        children[source].append(target)
    available = sorted((node_id for node_id, degree in indegree.items() if degree == 0), key=position.get)
    ordered: list[str] = []
    while available:
        current = available.pop(0); ordered.append(current)
        for child in children[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                available.append(child); available.sort(key=position.get)
    if len(ordered) != len(graph.nodes):
        raise InvalidGraph("graph must be acyclic")
    return ordered
