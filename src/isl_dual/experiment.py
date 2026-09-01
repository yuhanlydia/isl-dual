from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from .cache import CachedCritic, CachedExecutor, CachedMutator, CachedProposer, JSONCache
from .codex_components import CodexCritic, CodexJSON, CodexMutator, CodexProposer, graph_to_dict
from .executor import CodexExecutor
from .models import Graph, Node
from .compile import compile_graph_to_skill
from .report import scientific_report
from .pipeline import train_inverse_skill
from .skillevol_host import audit_no_curated_access, load_family


def run_family(benchmark_root: Path, family_id: str, output: Path, model: str | None = None) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    benchmark_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=benchmark_root, text=True, capture_output=True, check=True).stdout.strip()
    bundle = load_family(benchmark_root, family_id, artifact_cache=output / "artifacts" / benchmark_commit)
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
    audit_no_curated_access(benchmark_root, result.skill)
    (output / "SKILL.md").write_text(result.skill)
    skill_node = Node("skill", "Apply the following frozen procedural skill exactly as reusable guidance:\n\n" + result.skill, ("A deployment task is available",), ("task",), "Apply the following frozen procedural skill exactly as reusable guidance:\n\n" + result.skill, ("completed_task",), "Verify the task requirements", True)
    deployment_graph = Graph("frozen-deployment", (skill_node, Node("verify", "Verify the final artifact against all observable task requirements", (), (), "Verify the final artifact against all observable task requirements", ("verified",), "All observable checks pass", True)), (("skill", "verify"),))
    deployment_scores = {}
    for task in bundle.deployment:
        produced = executor.execute(task, deployment_graph, ("skill", "verify"))
        deployment_scores[task.id] = task.verifier(produced)
    no_skill_node = Node("act", "Complete the task using only the task input and observable environment feedback", (), ("task",), "Complete the task using only the task input and observable environment feedback", ("completed_task",), "Check observable requirements", True)
    no_skill_graph = Graph("no-skill-deployment", (no_skill_node,))
    no_skill_scores = {}
    for task in bundle.deployment:
        produced = executor.execute(task, no_skill_graph, ("act",))
        no_skill_scores[task.id] = task.verifier(produced)
    # Candidate diagnostics are evaluation-only: no score is fed into q or graph mutation.
    candidate_deployment: dict[str, float] = {}
    for graph_id in result.artifact_scores:  # initial K have both A_k and F_k
        graph = result.candidates[graph_id]
        candidate_skill = compile_graph_to_skill(graph)
        candidate_node = Node("skill", "Frozen candidate skill", ("A deployment task is available",), ("task",), candidate_skill, ("completed_task",), "Verify observable requirements", True)
        candidate_graph = Graph("candidate-deployment-" + graph_id, (candidate_node,))
        rewards = []
        for task in bundle.deployment:
            produced = executor.execute(task, candidate_graph, ("skill",))
            rewards.append(task.verifier(produced))
        candidate_deployment[graph_id] = sum(rewards) / len(rewards)
    common_forward = {k: result.forward_scores[k] for k in result.artifact_scores}
    diagnostics = scientific_report(
        no_skill_reward=sum(no_skill_scores.values()) / len(no_skill_scores), isl_reward=sum(deployment_scores.values()) / len(deployment_scores),
        artifact_scores=result.artifact_scores, forward_scores=common_forward,
        deployment_scores=candidate_deployment, q0=result.q0,
        q2=result.posterior,
        tau_artifact=0.8, tau_transfer=0.5,
    )
    summary = {"family_id": family_id, "benchmark_commit": benchmark_commit, "artifact_rewards": artifact_rewards, "winner": result.graph.id, "graph": graph_to_dict(result.graph), "artifact_scores": result.artifact_scores, "q0": result.q0, "q1": result.q1, "q2": result.posterior, "forward_scores": result.forward_scores, "deployment_scores": deployment_scores, "no_skill_scores": no_skill_scores, "candidate_deployment_scores": candidate_deployment, "scientific_metrics": diagnostics}
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
