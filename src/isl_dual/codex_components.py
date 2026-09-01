from __future__ import annotations

import json
import hashlib
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .models import AcquisitionTask, CriticScore, Graph, MCTSResult, Mutator, Node, OrGroup, Utility
from .graph import InvalidGraph, validate_graph


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
                command.append(prompt + f"\nTransient call attempt: {attempt}.")
                try:
                    result = subprocess.run(command, text=True, capture_output=True, timeout=self.timeout_seconds)
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
MUTANT_SCHEMA = json.loads(json.dumps(GRAPH_SCHEMA))
MUTANT_SCHEMA["properties"]["metadata"] = {
    "type": "object", "additionalProperties": False, "required": ["mutation"],
    "properties": {"mutation": {"type": "string", "enum": ["ADD_NODE", "REMOVE_OPTIONAL_NODE", "SPLIT_NODE", "MERGE_NODES", "ADD_EDGE", "REMOVE_EDGE", "CHANGE_BRANCH"]}},
}


def _artifact_text(tasks: list[AcquisitionTask]) -> str:
    return json.dumps([{"task_id": t.id, "task": t.x, "successful_final_artifact": t.expert_artifact} for t in tasks], ensure_ascii=False)


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
Validity contract: use 2-12 unique nodes; non-empty actions; existing edge endpoints; no self-edge or cycle; a required node may not depend on an optional node; OR groups contain at least two existing non-overlapping members, and a required OR group contains a required alternative.
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
        results = []; attempts = 0; last_error = ""
        while len(results) < count and attempts < count * 6:
            index = len(results); attempts += 1
            prompt = "Revise the procedural graph using execution evidence. Prefer one minimal edit among ADD_NODE, REMOVE_OPTIONAL_NODE, SPLIT_NODE, MERGE_NODES, ADD_EDGE, REMOVE_EDGE, CHANGE_BRANCH. Add only reusable operations indicated by failures; weaken nodes that reduce reward; never introduce task-specific answer content. Preserve DAG validity: 2-12 nodes, existing endpoints, acyclic, non-empty actions, no required node depending on an optional node, and valid non-overlapping OR groups.\nGRAPH:\n" + json.dumps(graph_to_dict(graph)) + "\nROLLOUTS:\n" + json.dumps(rollouts) + "\nNODE UTILITIES:\n" + json.dumps(utilities) + f"\nProduce distinct mutant {index + 1}, attempt {attempts}. Previous rejection: {last_error or 'none'}."
            data = self.client.call(prompt, MUTANT_SCHEMA)
            mutant = graph_from_dict(data, f"{graph.id}-m{index + 1}")
            mutant = Graph(mutant.id, mutant.nodes, mutant.edges, mutant.or_groups, {"parent_id": graph.id, "mutation": str(data["metadata"]["mutation"])})
            try:
                validate_graph(mutant)
            except InvalidGraph as error:
                last_error = str(error); continue
            if mutant.fingerprint() == graph.fingerprint() or mutant.fingerprint() in {item.fingerprint() for item in results}:
                last_error = "mutation made no distinct semantic change"; continue
            results.append(mutant); last_error = ""
        if len(results) != count:
            raise RuntimeError(f"Codex mutator produced only {len(results)}/{count} valid mutants after {attempts} attempts: {last_error}")
        return results
