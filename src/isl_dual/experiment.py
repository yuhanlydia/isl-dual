from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from .cache import CachedCritic, CachedExecutor, CachedJSONClient, CachedMutator, CachedProposer, JSONCache
from .baselines import Baseline, SelectedSkill, direct_text_skill, full_trajectory_skill, select_dag_baseline, upper_information_skill
from .codex_components import CodexCritic, CodexJSON, CodexMutator, CodexProposer, graph_to_dict
from .executor import CodexExecutor
from .models import Graph, Node
from .compile import compile_graph_to_skill
from .report import go_gate, scientific_report
from .config import PilotConfig
from .pipeline import _leakage_audit_skill, train_inverse_skill
from .skillevol_host import audit_no_curated_access, load_family


def run_family(benchmark_root: Path, family_id: str, output: Path, model: str | None = None, config: PilotConfig | None = None) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    benchmark_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=benchmark_root, text=True, capture_output=True, check=True).stdout.strip()
    bundle = load_family(benchmark_root, family_id, artifact_cache=output / "artifacts" / benchmark_commit)
    artifact_rewards = {task.id: task.verifier(task.expert_artifact) for task in bundle.acquisition}
    if any(reward < 1.0 for reward in artifact_rewards.values()):
        raise RuntimeError(f"expert outcomes fail native verifier: {artifact_rewards}")
    cache = JSONCache(output / "cache")
    client = CodexJSON(model=model)
    baseline_client = CachedJSONClient(client, cache, "baseline_skill")
    proposer = CachedProposer(CodexProposer(client), cache)
    critic = CachedCritic(CodexCritic(client), cache)
    executor = CachedExecutor(CodexExecutor(model=model), cache)
    mutator = CachedMutator(CodexMutator(client), cache)
    result = train_inverse_skill(bundle.acquisition, proposer, critic, executor, mutator, config=config)
    audit_no_curated_access(benchmark_root, result.skill)
    (output / "SKILL.md").write_text(result.skill)
    skill_node = Node("skill", "Apply the following frozen procedural skill exactly as reusable guidance:\n\n" + result.skill, ("A deployment task is available",), ("task",), "Apply the following frozen procedural skill exactly as reusable guidance:\n\n" + result.skill, ("completed_task",), "Verify the task requirements", True)
    deployment_graph = Graph("frozen-deployment", (skill_node, Node("verify", "Verify the final artifact against all observable task requirements", (), (), "Verify the final artifact against all observable task requirements", ("verified",), "All observable checks pass", True)), (("skill", "verify"),))
    deployment_scores = _evaluate_graph(deployment_graph, ("skill", "verify"), bundle.deployment, executor)
    no_skill_node = Node("act", "Complete the task using only the task input and observable environment feedback", (), ("task",), "Complete the task using only the task input and observable environment feedback", ("completed_task",), "Check observable requirements", True)
    no_skill_graph = Graph("no-skill-deployment", (no_skill_node,))
    no_skill_scores = _evaluate_graph(no_skill_graph, ("act",), bundle.deployment, executor)

    # B1--B5 are outcome-only baselines. Each is selected using acquisition data,
    # frozen, and then evaluated once on deployment tasks without feedback.
    selected: dict[Baseline, SelectedSkill] = {
        Baseline.DIRECT_TEXT: direct_text_skill(bundle.acquisition, baseline_client),
    }
    for baseline in (Baseline.ONE_SHOT_DAG, Baseline.STATIC_CRITIC, Baseline.GREEDY_FORWARD, Baseline.MCTS_FORWARD):
        selected[baseline] = select_dag_baseline(baseline, bundle.acquisition, proposer, critic, executor, config)
    selected[Baseline.ISL_DUAL] = SelectedSkill(Baseline.ISL_DUAL, result.skill, result.graph, result.posterior, result.forward_scores)

    # Upper-information controls are explicitly sourced and never enter ISL training.
    task_dirs = _family_task_dirs(benchmark_root, family_id)
    trajectories = [(task_dirs[index] / "solution" / "solve.sh").read_text() for index in (1, 2, 3)]
    selected[Baseline.FULL_TRAJECTORY] = full_trajectory_skill(bundle.acquisition, trajectories, baseline_client)
    selected[Baseline.CURATED_SKILL] = upper_information_skill(Baseline.CURATED_SKILL, _curated_skill(benchmark_root, task_dirs[1]))

    baseline_task_scores: dict[str, dict[str, float]] = {Baseline.NO_SKILL.value: no_skill_scores}
    for baseline, selection in selected.items():
        assert selection.skill is not None
        if baseline not in {Baseline.FULL_TRAJECTORY, Baseline.CURATED_SKILL}:
            _leakage_audit_skill(selection.skill, bundle.acquisition)
        if baseline == Baseline.ISL_DUAL:
            baseline_task_scores[baseline.value] = deployment_scores
        else:
            graph = _skill_graph(baseline.value, selection.skill)
            baseline_task_scores[baseline.value] = _evaluate_graph(graph, ("skill", "verify"), bundle.deployment, executor)
    baseline_rewards = {name: sum(scores.values()) / len(scores) for name, scores in baseline_task_scores.items()}
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
    summary = {"family_id": family_id, "benchmark_commit": benchmark_commit, "artifact_rewards": artifact_rewards, "winner": result.graph.id, "graph": graph_to_dict(result.graph), "artifact_scores": result.artifact_scores, "q0": result.q0, "q1": result.q1, "q2": result.posterior, "forward_scores": result.forward_scores, "deployment_scores": deployment_scores, "no_skill_scores": no_skill_scores, "baseline_task_scores": baseline_task_scores, "baseline_rewards": baseline_rewards, "go_gate": go_gate({"B1": baseline_rewards[Baseline.DIRECT_TEXT.value], "B3": baseline_rewards[Baseline.STATIC_CRITIC.value], "B4": baseline_rewards[Baseline.GREEDY_FORWARD.value], "B6": baseline_rewards[Baseline.ISL_DUAL.value]}), "candidate_deployment_scores": candidate_deployment, "scientific_metrics": diagnostics}
    (output / "result.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _skill_graph(identifier: str, skill: str) -> Graph:
    skill_node = Node("skill", "Apply frozen procedural skill", ("A deployment task is available",), ("task",), "Apply the following frozen procedural skill exactly as reusable guidance:\n\n" + skill, ("completed_task",), "Follow the procedure", True)
    verify_node = Node("verify", "Verify final artifact", (), (), "Verify the final artifact against all observable task requirements", ("verified",), "All observable checks pass", True)
    return Graph("deployment-" + identifier, (skill_node, verify_node), (("skill", "verify"),))


def _evaluate_graph(graph: Graph, plan: tuple[str, ...], tasks: list[object], executor: CachedExecutor) -> dict[str, float]:
    scores: dict[str, float] = {}
    for task in tasks:
        produced = executor.execute(task, graph, plan)
        scores[task.id] = task.verifier(produced)
    return scores


def _family_task_dirs(benchmark_root: Path, family_id: str) -> dict[int, Path]:
    import yaml
    found: dict[int, Path] = {}
    for task_dir in (benchmark_root / "benchmark" / "tasks").iterdir():
        spec_path = task_dir / "task-spec.yaml"
        if spec_path.exists():
            spec = yaml.safe_load(spec_path.read_text())
            if spec.get("family_id") == family_id:
                found[int(spec["task_index"])] = task_dir
    if set(found) != set(range(1, 7)):
        raise ValueError(f"family {family_id} does not contain T1--T6")
    return found


def _curated_skill(benchmark_root: Path, task_dir: Path) -> str:
    import yaml
    spec = yaml.safe_load((task_dir / "task-spec.yaml").read_text())
    slug = str(spec["latent_skill_id"]).split(".", 1)[1]
    path = benchmark_root / "benchmark" / "skills" / slug / "SKILL.md"
    if not path.exists():
        raise FileNotFoundError(f"curated skill declared by benchmark is missing: {path}")
    return path.read_text()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model")
    args = parser.parse_args()
    print(json.dumps(run_family(args.benchmark_root, args.family, args.output, args.model), indent=2))


if __name__ == "__main__": main()
