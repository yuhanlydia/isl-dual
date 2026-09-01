from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cache import CachedCritic, CachedExecutor, CachedMutator, CachedProposer, JSONCache
from .codex_components import CodexCritic, CodexJSON, CodexMutator, CodexProposer, graph_to_dict
from .executor import CodexExecutor
from .models import Graph, Node
from .pipeline import train_inverse_skill
from .skillevol_host import load_family


def run_family(benchmark_root: Path, family_id: str, output: Path, model: str | None = None) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    bundle = load_family(benchmark_root, family_id)
    artifact_rewards = {task.id: task.verifier(task.expert_artifact) for task in bundle.acquisition}
    if any(reward < 1.0 for reward in artifact_rewards.values()):
        raise RuntimeError(f"expert outcomes fail native verifier: {artifact_rewards}")
    cache = JSONCache(output / "cache")
    client = CodexJSON(model=model)
    proposer = CachedProposer(CodexProposer(client), cache)
    critic = CachedCritic(CodexCritic(client), cache)
    executor = CachedExecutor(CodexExecutor(model=model), cache)
    mutator = CachedMutator(CodexMutator(client), cache)
    result = train_inverse_skill(bundle.acquisition, proposer, critic, executor, mutator)
    (output / "SKILL.md").write_text(result.skill)
    skill_node = Node("skill", "Apply the following frozen procedural skill exactly as reusable guidance:\n\n" + result.skill, ("A deployment task is available",), ("task",), "Apply the following frozen procedural skill exactly as reusable guidance:\n\n" + result.skill, ("completed_task",), "Verify the task requirements", True)
    deployment_graph = Graph("frozen-deployment", (skill_node, Node("verify", "Verify the final artifact against all observable task requirements", (), (), "Verify the final artifact against all observable task requirements", ("verified",), "All observable checks pass", True)), (("skill", "verify"),))
    deployment_scores = {}
    for task in bundle.deployment:
        produced = executor.execute(task, deployment_graph, ("skill", "verify"))
        deployment_scores[task.id] = task.verifier(produced)
    summary = {"family_id": family_id, "artifact_rewards": artifact_rewards, "winner": result.graph.id, "graph": graph_to_dict(result.graph), "artifact_scores": result.artifact_scores, "q0": result.q0, "q1": result.q1, "q2": result.posterior, "forward_scores": result.forward_scores, "deployment_scores": deployment_scores}
    (output / "result.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model")
    args = parser.parse_args()
    print(json.dumps(run_family(args.benchmark_root, args.family, args.output, args.model), indent=2))


if __name__ == "__main__": main()
