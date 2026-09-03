from __future__ import annotations

import json
import hashlib
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .models import AcquisitionTask, CriticScore, Graph, MCTSResult, Mutator, Node, OrGroup, Utility
from .graph import InvalidGraph, validate_graph
from .subprocesses import run_process_group


def graph_from_dict(data: Mapping[str, Any], graph_id: str | None = None) -> Graph:
    nodes = tuple(Node(
        id=str(n["id"]), name=str(n["name"]),
        preconditions=tuple(map(str, n.get("preconditions", []))),
        inputs=tuple(map(str, n.get("inputs", []))), action=str(n["action"]),
        outputs=tuple(map(str, n.get("outputs", []))), validator=str(n.get("validator", "")),
        required=bool(n.get("required", True)),
    ) for n in data["nodes"])
    edges = tuple((str(e[0]), str(e[1])) for e in data.get("edges", []))
    groups = tuple(OrGroup(str(g["id"]), tuple(map(str, g["members"])), bool(g.get("required", False))) for g in data.get("or_groups", []))
    return Graph(id=graph_id or str(data["id"]), nodes=nodes, edges=edges, or_groups=groups, metadata=data.get("metadata", {}))


def graph_to_dict(graph: Graph) -> dict[str, Any]:
    return {
        "id": graph.id,
        "nodes": [{
            "id": n.id, "name": n.name, "preconditions": list(n.preconditions),
            "inputs": list(n.inputs), "action": n.action, "outputs": list(n.outputs),
            "validator": n.validator, "required": n.required,
        } for n in graph.nodes],
        "edges": [list(e) for e in graph.edges],
        "or_groups": [{"id": g.id, "members": list(g.members), "required": g.required} for g in graph.or_groups],
        "metadata": dict(graph.metadata),
    }


class CodexJSON:
    def __init__(self, model: str | None = None, timeout_seconds: int = 300, transient_attempts: int = 3):
        self.model, self.timeout_seconds, self.transient_attempts = model, timeout_seconds, transient_attempts

    def call(self, prompt: str, schema: dict[str, Any]) -> Any:
        errors: list[str] = []
        for attempt in range(1, self.transient_attempts + 1):
            with tempfile.TemporaryDirectory(prefix="isl-dual-json-") as temp:
                root = Path(temp)
                schema_path, output_path = root / "schema.json", root / "output.json"
                schema_path.write_text(json.dumps(schema))
                command = ["codex", "exec", "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only", "-C", str(root), "--output-schema", str(schema_path), "-o", str(output_path)]
                if self.model:
                    command.extend(["--model", self.model])
                command.append("-")
                try:
                    result = run_process_group(command, timeout=self.timeout_seconds, input_text=prompt + f"\nTransient call attempt: {attempt}.")
                except subprocess.TimeoutExpired:
                    errors.append(f"attempt {attempt}: timed out after {self.timeout_seconds}s")
                    continue
                if result.returncode != 0:
                    errors.append(f"attempt {attempt}: exit {result.returncode}: {result.stderr[-2000:]}")
                    continue
                try:
                    return json.loads(output_path.read_text())
                except (OSError, json.JSONDecodeError) as error:
                    errors.append(f"attempt {attempt}: invalid structured output: {error}")
        raise RuntimeError("Codex structured call failed after transient retries:\n" + "\n".join(errors))


GRAPH_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "required": ["id", "nodes", "edges", "or_groups", "metadata"],
    "properties": {
        "id": {"type": "string"},
        "nodes": {"type": "array", "minItems": 2, "maxItems": 12, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["id", "name", "preconditions", "inputs", "action", "outputs", "validator", "required"],
            "properties": {
                "id": {"type": "string"}, "name": {"type": "string"},
                "preconditions": {"type": "array", "items": {"type": "string"}},
                "inputs": {"type": "array", "items": {"type": "string"}},
                "action": {"type": "string"}, "outputs": {"type": "array", "items": {"type": "string"}},
                "validator": {"type": "string"}, "required": {"type": "boolean"},
            }}},
        "edges": {"type": "array", "items": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "string"}}},
        "or_groups": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["id", "members", "required"], "properties": {"id": {"type": "string"}, "members": {"type": "array", "minItems": 2, "items": {"type": "string"}}, "required": {"type": "boolean"}}}},
        "metadata": {"type": "object", "additionalProperties": False, "properties": {}},
    },
}
MUTATION_OPERATIONS = (
    "ADD_NODE", "REMOVE_OPTIONAL_NODE", "SPLIT_NODE", "MERGE_NODES",
    "ADD_EDGE", "REMOVE_EDGE", "CHANGE_BRANCH",
)
MUTANT_SCHEMA = json.loads(json.dumps(GRAPH_SCHEMA))
MUTANT_SCHEMA["properties"]["metadata"] = {
    "type": "object", "additionalProperties": False, "required": ["mutation"],
    "properties": {"mutation": {"type": "string", "enum": list(MUTATION_OPERATIONS)}},
}


def _mutation_schema(operation: str) -> dict[str, Any]:
    schema = json.loads(json.dumps(MUTANT_SCHEMA))
    schema["properties"]["metadata"]["properties"]["mutation"]["enum"] = [operation]
    return schema


def _feasible_mutation_operations(graph: Graph) -> list[str]:
    """Return structural operators that have at least one valid neighbor in principle."""
    node_count = len(graph.nodes)
    operations: list[str] = []
    if node_count < 12:
        operations.extend(["ADD_NODE", "SPLIT_NODE"])
    if node_count > 2:
        operations.append("MERGE_NODES")
    if node_count > 2 and any(not node.required for node in graph.nodes):
        operations.append("REMOVE_OPTIONAL_NODE")
    max_dag_edges = node_count * (node_count - 1) // 2
    if len(graph.edges) < max_dag_edges:
        operations.append("ADD_EDGE")
    if graph.edges:
        operations.append("REMOVE_EDGE")
    if graph.or_groups:
        operations.append("CHANGE_BRANCH")
    return operations


def observable_artifact(artifact: Any) -> Any:
    if isinstance(artifact, dict):
        return {key: observable_artifact(value) for key, value in artifact.items() if not str(key).startswith("_")}
    if isinstance(artifact, list):
        return [observable_artifact(value) for value in artifact]
    return artifact


def _artifact_text(tasks: list[AcquisitionTask]) -> str:
    return json.dumps([{"task_id": t.id, "task": t.x, "successful_final_artifact": observable_artifact(t.expert_artifact)} for t in tasks], ensure_ascii=False)


class CodexProposer:
    def __init__(self, client: CodexJSON): self.client = client

    def propose(self, tasks: list[AcquisitionTask], mode: str, count: int) -> list[Graph]:
        results = []
        attempts = 0
        last_error = ""
        while len(results) < count and attempts < count * 6:
            index = len(results)
            attempts += 1
            prompt = f"""You do not know the expert's actual execution history. Do not reconstruct chain-of-thought.
Infer one reusable {mode} procedural DAG that could explain the successful final artifacts and transfer to related unseen tasks.
Do not copy case-specific answers, filenames, constants, or output content unless genuinely reusable. This is variant {index + 1}, attempt {attempts}; make it structurally distinct.
Validity contract: use 2-12 unique nodes; non-empty actions; existing edge endpoints; no self-edge or cycle; a required node may not depend on a lone optional node; OR groups contain at least two existing non-overlapping members. Incoming edges from alternatives in one OR group form a single any-of dependency clause.
The previous rejected attempt failed validation with: {last_error or 'none'}.
Observed outcome-only data:\n{_artifact_text(tasks)}"""
            data = self.client.call(prompt, GRAPH_SCHEMA)
            digest = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:10]
            graph = graph_from_dict(data, f"{mode}-{index + 1}-{digest}")
            try:
                validate_graph(graph)
            except InvalidGraph as error:
                last_error = str(error)
                continue
            if graph.fingerprint() in {item.fingerprint() for item in results}:
                last_error = "semantic duplicate of an accepted graph"
                continue
            results.append(graph)
            last_error = ""
        if len(results) != count:
            raise RuntimeError(f"Codex proposer produced only {len(results)}/{count} valid graphs after {attempts} attempts: {last_error}")
        return results


class CodexCritic:
    SCHEMA = {"type": "object", "additionalProperties": False, "required": ["sufficiency", "transfer", "consistency"], "properties": {k: {"type": "number", "minimum": 0, "maximum": 1} for k in ("sufficiency", "transfer", "consistency")}}

    def __init__(self, client: CodexJSON): self.client = client

    def score(self, graph: Graph, tasks: list[AcquisitionTask]) -> CriticScore:
        prompt = "Score this candidate procedural DAG against all three outcome-only examples. Judge sufficiency, case-independent transfer, and consistency in [0,1]. Do not infer an expert trajectory.\nDAG:\n" + json.dumps(graph_to_dict(graph)) + "\nDATA:\n" + _artifact_text(tasks)
        value = self.client.call(prompt, self.SCHEMA)
        return CriticScore(float(value["sufficiency"]), float(value["transfer"]), float(value["consistency"]))


class CodexMutator:
    def __init__(self, client: CodexJSON): self.client = client

    def mutate(self, graph: Graph, evidence: Mapping[tuple[str, str], MCTSResult], node_utilities: Mapping[tuple[str, str], Utility], edge_utilities: Mapping[tuple[str, tuple[str, str]], Utility], count: int) -> list[Graph]:
        rollouts = [{"task": task, "plan": list(r.plan), "reward": r.reward, "failure": r.failure} for (gid, task), result in evidence.items() if gid == graph.id for r in result.rollouts]
        utilities = {node: {"delta": u.delta, "n_with": u.n_with, "n_without": u.n_without} for (gid, node), u in node_utilities.items() if gid == graph.id}
        edge_values = {str(edge): {"delta": u.delta, "n_with": u.n_with, "n_without": u.n_without} for (gid, edge), u in edge_utilities.items() if gid == graph.id}
        operations = _feasible_mutation_operations(graph)
        if not operations or count <= 0:
            return []
        results: list[Graph] = []
        attempts = 0
        max_attempts = max(count * 6, len(operations) * 2)
        last_error = ""
        while len(results) < count and attempts < max_attempts:
            index = len(results)
            operation = operations[attempts % len(operations)]
            attempts += 1
            prompt = (
                f"Revise the procedural graph using execution evidence. Make exactly one {operation} edit and no other structural edit. "
                "Add only reusable operations indicated by failures; weaken nodes that reduce reward; never introduce task-specific answer content. "
                "Preserve DAG validity: 2-12 nodes, existing endpoints, acyclic, non-empty actions, no required node depending on a lone optional node, and valid non-overlapping OR groups. "
                "Incoming edges from alternatives in one OR group form a single any-of dependency clause.\nGRAPH:\n"
                + json.dumps(graph_to_dict(graph))
                + "\nROLLOUTS:\n" + json.dumps(rollouts)
                + "\nNODE UTILITIES:\n" + json.dumps(utilities)
                + "\nEDGE UTILITIES:\n" + json.dumps(edge_values)
                + f"\nRequested mutation: {operation}. Produce distinct mutant {index + 1}, attempt {attempts}. Previous rejection: {last_error or 'none'}."
            )
            data = self.client.call(prompt, _mutation_schema(operation))
            declared = str((data.get("metadata") or {}).get("mutation", ""))
            if declared != operation:
                last_error = f"model declared {declared or 'none'} instead of requested {operation}"
                continue
            mutant = graph_from_dict(data, f"{graph.id}-m{index + 1}")
            mutant = Graph(mutant.id, mutant.nodes, mutant.edges, mutant.or_groups, {"parent_id": graph.id, "mutation": operation})
            try:
                validate_graph(mutant)
                validate_declared_mutation(graph, mutant, operation)
            except InvalidGraph as error:
                last_error = str(error)
                continue
            if mutant.fingerprint() == graph.fingerprint() or mutant.fingerprint() in {item.fingerprint() for item in results}:
                last_error = "mutation made no distinct semantic change"
                continue
            results.append(mutant)
            last_error = ""
        # Mutation is an optional evolution proposal, not a family-level validity
        # requirement. Returning a short list preserves the valid evidence already
        # collected and lets the second loop compare whatever valid neighbors exist.
        return results


def validate_declared_mutation(parent: Graph, mutant: Graph, operation: str) -> None:
    """Reject a model response whose actual structural edit is outside its declared operator."""
    old, new = parent.node_map(), mutant.node_map()
    old_ids, new_ids = set(old), set(new)
    shared_unchanged = all(old[node_id] == new[node_id] for node_id in old_ids & new_ids)
    old_edges, new_edges = set(parent.edges), set(mutant.edges)
    same_groups = parent.or_groups == mutant.or_groups
    if operation == "ADD_NODE":
        added = new_ids - old_ids
        valid = len(added) == 1 and not (old_ids - new_ids) and shared_unchanged and old_edges <= new_edges and all(set(edge) & added for edge in new_edges - old_edges) and same_groups
    elif operation == "REMOVE_OPTIONAL_NODE":
        removed = old_ids - new_ids
        expected_edges = {edge for edge in old_edges if not (set(edge) & removed)}
        expected_groups = tuple(group for group in parent.or_groups if not (set(group.members) & removed))
        valid = len(removed) == 1 and not (new_ids - old_ids) and not old[next(iter(removed))].required and shared_unchanged and new_edges == expected_edges and mutant.or_groups == expected_groups
    elif operation == "SPLIT_NODE":
        valid = len(new_ids) == len(old_ids) + 1
    elif operation == "MERGE_NODES":
        valid = len(new_ids) == len(old_ids) - 1
    elif operation == "ADD_EDGE":
        valid = old_ids == new_ids and shared_unchanged and same_groups and old_edges < new_edges and len(new_edges - old_edges) == 1
    elif operation == "REMOVE_EDGE":
        valid = old_ids == new_ids and shared_unchanged and same_groups and new_edges < old_edges and len(old_edges - new_edges) == 1
    elif operation == "CHANGE_BRANCH":
        valid = old_ids == new_ids and shared_unchanged and (old_edges != new_edges or not same_groups)
    else:
        valid = False
    if not valid:
        raise InvalidGraph(f"mutant does not implement exactly one declared {operation} operation")
