from __future__ import annotations

from dataclasses import replace

from .models import Graph, Utility, clone_graph
from .graph import topological_node_ids


def operational_pruning(
    graph: Graph, utilities: dict[tuple[str, str], Utility], threshold: float = 0.05
) -> Graph:
    remove = {
        node.id
        for node in graph.nodes
        if not node.required
        and (utility := utilities.get((graph.id, node.id))) is not None
        and utility.delta is not None
        and utility.delta < -threshold
        and utility.n_with >= 2
        and utility.n_without >= 2
    }
    surviving_nodes = tuple(node for node in graph.nodes if node.id not in remove)
    surviving_groups = []
    force_required: set[str] = set()
    for group in graph.or_groups:
        members = tuple(member for member in group.members if member not in remove)
        if len(members) >= 2:
            surviving_groups.append(replace(group, members=members))
        elif len(members) == 1 and group.required:
            force_required.add(members[0])
    surviving_nodes = tuple(replace(node, required=True) if node.id in force_required else node for node in surviving_nodes)
    return clone_graph(
        graph,
        nodes=surviving_nodes,
        edges=tuple((a, b) for a, b in graph.edges if a not in remove and b not in remove),
        or_groups=tuple(surviving_groups),
    )


def compile_graph_to_skill(graph: Graph) -> str:
    node_map = graph.node_map()
    ordered_nodes = [node_map[node_id] for node_id in topological_node_ids(graph)]
    member_groups = {member: group.id for group in graph.or_groups for member in group.members}
    lines = ["# Skill", "", "## Preconditions"]
    preconditions = list(dict.fromkeys(p for n in graph.nodes for p in n.preconditions))
    lines.extend(f"- {item}" for item in preconditions or ["A concrete task input is available."])
    lines.extend(["", "## Procedure", ""])
    for index, node in enumerate(ordered_nodes, start=1):
        marker = " (optional)" if not node.required else ""
        if node.id in member_groups:
            marker += f" (OR branch: choose one from {member_groups[node.id]})"
        lines.append(f"{index}. {node.action}{marker}")
    lines.extend(["", "## Failure checks", ""])
    lines.extend(f"- {node.validator}" for node in graph.nodes if node.validator.strip())
    lines.extend(["", "## Stop condition", ""])
    required_outputs = [o for n in graph.nodes if n.required for o in n.outputs]
    if required_outputs:
        lines.append("Stop when the required outputs exist and all validators pass: " + ", ".join(required_outputs) + ".")
    else:
        lines.append("Stop when all required nodes and validators have completed successfully.")
    return "\n".join(lines) + "\n"
