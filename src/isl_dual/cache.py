from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from .codex_components import graph_from_dict, graph_to_dict
from .models import AcquisitionTask, CriticScore, Graph, MCTSResult, Utility


class JSONCache:
    def __init__(self, root: Path): self.root = root

    def key(self, namespace: str, payload: Any) -> Path:
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
        return self.root / namespace / f"{digest}.json"

    def get(self, path: Path) -> Any | None:
        return json.loads(path.read_text()) if path.exists() else None

    def put(self, path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str))
        os.replace(temporary, path)


def _task_digest(tasks: list[AcquisitionTask]) -> list[dict[str, Any]]:
    return [{"id": t.id, "x": t.x, "artifact": t.expert_artifact} for t in tasks]


class CachedProposer:
    def __init__(self, inner: Any, cache: JSONCache): self.inner, self.cache = inner, cache
    def propose(self, tasks: list[AcquisitionTask], mode: str, count: int) -> list[Graph]:
        path = self.cache.key("proposer", {"tasks": _task_digest(tasks), "mode": mode, "count": count})
        value = self.cache.get(path)
        if value is None:
            graphs = self.inner.propose(tasks, mode, count)
            value = [graph_to_dict(g) for g in graphs]; self.cache.put(path, value)
        return [graph_from_dict(item) for item in value]


class CachedCritic:
    def __init__(self, inner: Any, cache: JSONCache): self.inner, self.cache = inner, cache
    def score(self, graph: Graph, tasks: list[AcquisitionTask]) -> CriticScore:
        path = self.cache.key("critic", {"graph": graph_to_dict(graph), "tasks": _task_digest(tasks)})
        value = self.cache.get(path)
        if value is None:
            score = self.inner.score(graph, tasks)
            value = {"sufficiency": score.sufficiency, "transfer": score.transfer, "consistency": score.consistency}; self.cache.put(path, value)
        return CriticScore(**value)


class CachedExecutor:
    def __init__(self, inner: Any, cache: JSONCache):
        self.inner, self.cache = inner, cache
        self._occurrences: dict[tuple[str, str, tuple[str, ...]], int] = defaultdict(int)
    def execute(self, task: Any, graph: Graph, plan: tuple[str, ...]) -> Any:
        identity = (task.id, graph.id, plan)
        occurrence = self._occurrences[identity]
        self._occurrences[identity] += 1
        path = self.cache.key("executor", {"task_id": task.id, "x": task.x, "graph": graph_to_dict(graph), "plan": plan, "occurrence": occurrence})
        value = self.cache.get(path)
        if value is None:
            value = self.inner.execute(task, graph, plan); self.cache.put(path, value)
        return value


class CachedMutator:
    def __init__(self, inner: Any, cache: JSONCache): self.inner, self.cache = inner, cache
    def mutate(self, graph: Graph, evidence: Mapping[tuple[str, str], MCTSResult], node_utilities: Mapping[tuple[str, str], Utility], edge_utilities: Mapping[tuple[str, tuple[str, str]], Utility], count: int) -> list[Graph]:
        evidence_digest = [{"key": key, "plans": [(r.plan, r.reward) for r in result.rollouts]} for key, result in evidence.items() if key[0] == graph.id]
        utility_digest = {str(key): (value.delta, value.n_with, value.n_without) for key, value in node_utilities.items() if key[0] == graph.id}
        path = self.cache.key("mutator", {"graph": graph_to_dict(graph), "evidence": evidence_digest, "utilities": utility_digest, "count": count})
        value = self.cache.get(path)
        if value is None:
            graphs = self.inner.mutate(graph, evidence, node_utilities, edge_utilities, count)
            value = [graph_to_dict(g) for g in graphs]; self.cache.put(path, value)
        return [graph_from_dict(item) for item in value]
