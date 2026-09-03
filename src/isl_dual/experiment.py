from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import yaml

from .baselines import (
    Baseline,
    SelectedSkill,
    direct_text_skill,
    full_trajectory_skill,
    select_dag_baseline,
    selected_from_training,
    upper_information_skill,
)
from .cache import (
    CachedCritic,
    CachedExecutor,
    CachedJSONClient,
    CachedMutator,
    CachedProposer,
    CachedVerifier,
    JSONCache,
)
from .codex_components import CodexCritic, CodexJSON, CodexMutator, CodexProposer, graph_to_dict
from .compile import compile_graph_to_skill
from .config import PilotConfig
from .controls import edge_shuffle
from .executor import CodexExecutor
from .mcts import EvidenceJournal
from .models import AcquisitionTask, DeploymentTask, Graph, Node
from .pipeline import _leakage_audit_skill, train_inverse_skill
from .report import go_gate, scientific_report
from .skillevol_host import FamilyBundle, audit_no_curated_access, load_family


def run_family(
    benchmark_root: Path,
    family_id: str,
    output: Path,
    model: str | None = None,
    config: PilotConfig | None = None,
    bundle_override: FamilyBundle | None = None,
    require_passing_artifacts: bool = True,
    run_expensive_controls: bool = True,
) -> dict[str, object]:
    """Run one SkillEvolBench family.

    B3 and B5 are nested ablations of B6 and therefore reuse the exact q0/q1
    states and first-loop evidence from the full run. Cross-family artifact
    shuffle is optional because it requires another full ISL run.
    """
    config = config or PilotConfig()
    output.mkdir(parents=True, exist_ok=True)
    benchmark_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=benchmark_root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    cache = JSONCache(output / "cache")
    artifact_root = output / "artifacts" / benchmark_commit
    bundle = bundle_override or load_family(benchmark_root, family_id, artifact_cache=artifact_root)
    bundle = _with_checkpointed_verifiers(bundle, cache)

    artifact_rewards = _verified_artifact_rewards(
        bundle.acquisition, artifact_root / "native-verification.json"
    )
    if require_passing_artifacts and any(reward < 1.0 for reward in artifact_rewards.values()):
        raise RuntimeError(f"expert outcomes fail native verifier: {artifact_rewards}")

    client = CodexJSON(model=model)
    baseline_client = CachedJSONClient(client, cache, "baseline_skill")
    proposer = CachedProposer(CodexProposer(client), cache)
    critic = CachedCritic(CodexCritic(client), cache)
    executor = CachedExecutor(CodexExecutor(model=model), cache)
    mutator = CachedMutator(CodexMutator(client), cache)

    result = train_inverse_skill(
        bundle.acquisition,
        proposer,
        critic,
        executor,
        mutator,
        config=config,
        evidence_journal=EvidenceJournal(output / "evidence.json"),
    )
    audit_no_curated_access(benchmark_root, result.skill)
    (output / "SKILL.md").write_text(result.skill)

    deployment_graph = _skill_graph("B6_isl_dual", result.skill)
    deployment_scores = _evaluate_graph(
        deployment_graph, ("skill", "verify"), bundle.deployment, executor
    )
    no_skill_node = Node(
        "act",
        "Complete the task using only the task input and observable environment feedback",
        (),
        ("task",),
        "Complete the task using only the task input and observable environment feedback",
        ("completed_task",),
        "Check observable requirements",
        True,
    )
    no_skill_graph = Graph("no-skill-deployment", (no_skill_node,))
    no_skill_scores = _evaluate_graph(
        no_skill_graph, ("act",), bundle.deployment, executor
    )

    # Outcome-only baselines. B3/q0 and B5/q1 are exact nested states of B6,
    # avoiding a second stochastic copy of the same candidate pool and MCTS.
    selected: dict[Baseline, SelectedSkill] = {
        Baseline.DIRECT_TEXT: direct_text_skill(bundle.acquisition, baseline_client),
        Baseline.ONE_SHOT_DAG: select_dag_baseline(
            Baseline.ONE_SHOT_DAG,
            bundle.acquisition,
            proposer,
            critic,
            executor,
            config,
        ),
        Baseline.STATIC_CRITIC: selected_from_training(Baseline.STATIC_CRITIC, result),
        Baseline.GREEDY_FORWARD: select_dag_baseline(
            Baseline.GREEDY_FORWARD,
            bundle.acquisition,
            proposer,
            critic,
            executor,
            config,
        ),
        Baseline.MCTS_FORWARD: selected_from_training(Baseline.MCTS_FORWARD, result),
        Baseline.ISL_DUAL: SelectedSkill(
            Baseline.ISL_DUAL,
            result.skill,
            result.graph,
            result.posterior,
            result.forward_scores,
        ),
    }

    # Upper-information controls remain isolated from ISL.
    task_dirs = _family_task_dirs(benchmark_root, family_id)
    oracle_procedures = [
        (task_dirs[index] / "solution" / "solve.sh").read_text()
        for index in (1, 2, 3)
    ]
    selected[Baseline.ORACLE_SOLUTION] = full_trajectory_skill(
        bundle.acquisition, oracle_procedures, baseline_client
    )
    selected[Baseline.CURATED_SKILL] = upper_information_skill(
        Baseline.CURATED_SKILL, _curated_skill(benchmark_root, task_dirs[1])
    )

    baseline_task_scores: dict[str, dict[str, float]] = {
        Baseline.NO_SKILL.value: no_skill_scores
    }
    for baseline, selection in selected.items():
        assert selection.skill is not None
        if baseline not in {Baseline.ORACLE_SOLUTION, Baseline.CURATED_SKILL}:
            _leakage_audit_skill(selection.skill, bundle.acquisition)
        if baseline == Baseline.ISL_DUAL:
            baseline_task_scores[baseline.value] = deployment_scores
        else:
            graph = _skill_graph(baseline.value, selection.skill)
            baseline_task_scores[baseline.value] = _evaluate_graph(
                graph, ("skill", "verify"), bundle.deployment, executor
            )
    baseline_rewards = {
        name: sum(scores.values()) / len(scores)
        for name, scores in baseline_task_scores.items()
    }

    # Candidate diagnostics are evaluation-only; held-out scores never feed back.
    candidate_deployment: dict[str, float] = {}
    for graph_id in result.artifact_scores:
        graph = result.candidates[graph_id]
        candidate_skill = compile_graph_to_skill(graph)
        candidate_graph = _skill_graph("candidate-" + graph_id, candidate_skill)
        rewards = []
        for task in bundle.deployment:
            produced = executor.execute(task, candidate_graph, ("skill", "verify"))
            rewards.append(task.verifier(produced))
        candidate_deployment[graph_id] = sum(rewards) / len(rewards)

    common_forward = {
        key: result.forward_scores[key]
        for key in result.artifact_scores
        if key in result.forward_scores
    }
    diagnostics = scientific_report(
        no_skill_reward=sum(no_skill_scores.values()) / len(no_skill_scores),
        isl_reward=sum(deployment_scores.values()) / len(deployment_scores),
        artifact_scores=result.artifact_scores,
        forward_scores=common_forward,
        deployment_scores=candidate_deployment,
        q0=result.q0,
        q2=result.posterior,
        tau_artifact=0.8,
        tau_transfer=0.5,
    )

    edge_control = _edge_shuffle_control(
        result.graph, bundle.deployment, executor, config.seed
    )
    if run_expensive_controls:
        artifact_control = _artifact_shuffle_control(
            benchmark_root,
            family_id,
            bundle,
            output / "controls" / "artifact_shuffle",
            model,
            config,
            benchmark_commit,
        )
    else:
        artifact_control = {
            "status": "deferred",
            "reason": "run_expensive_controls=False; run after primary GO gates",
        }

    summary = {
        "family_id": family_id,
        "run_metadata": _run_metadata(output, benchmark_commit, model, config),
        "benchmark_commit": benchmark_commit,
        "artifact_rewards": artifact_rewards,
        "winner": result.graph.id,
        "graph": graph_to_dict(result.graph),
        "candidate_graphs": {
            graph_id: graph_to_dict(graph)
            for graph_id, graph in result.candidates.items()
        },
        "artifact_scores": result.artifact_scores,
        "q0": result.q0,
        "q1": result.q1,
        "q2": result.posterior,
        "forward_scores": result.forward_scores,
        "deployment_scores": deployment_scores,
        "no_skill_scores": no_skill_scores,
        "baseline_task_scores": baseline_task_scores,
        "baseline_rewards": baseline_rewards,
        "go_gate": go_gate(
            {
                "B1": baseline_rewards[Baseline.DIRECT_TEXT.value],
                "B3": baseline_rewards[Baseline.STATIC_CRITIC.value],
                "B4": baseline_rewards[Baseline.GREEDY_FORWARD.value],
                "B6": baseline_rewards[Baseline.ISL_DUAL.value],
            }
        ),
        "candidate_deployment_scores": candidate_deployment,
        "scientific_metrics": diagnostics,
        "causal_controls": {
            "edge_shuffle": edge_control,
            "artifact_shuffle": artifact_control,
        },
    }
    (output / "result.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _skill_graph(identifier: str, skill: str) -> Graph:
    skill_node = Node(
        "skill",
        "Apply frozen procedural skill",
        ("A deployment task is available",),
        ("task",),
        "Apply the following frozen procedural skill exactly as reusable guidance:\n\n"
        + skill,
        ("completed_task",),
        "Follow the procedure",
        True,
    )
    verify_node = Node(
        "verify",
        "Verify final artifact",
        (),
        (),
        "Verify the final artifact against all observable task requirements",
        ("verified",),
        "All observable checks pass",
        True,
    )
    return Graph(
        "deployment-" + identifier,
        (skill_node, verify_node),
        (("skill", "verify"),),
    )


def _evaluate_graph(
    graph: Graph,
    plan: tuple[str, ...],
    tasks: list[DeploymentTask],
    executor: CachedExecutor,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for task in tasks:
        produced = executor.execute(task, graph, plan)
        scores[task.id] = task.verifier(produced)
    return scores


def _family_task_dirs(benchmark_root: Path, family_id: str) -> dict[int, Path]:
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
    spec = yaml.safe_load((task_dir / "task-spec.yaml").read_text())
    slug = str(spec["latent_skill_id"]).split(".", 1)[1]
    path = benchmark_root / "benchmark" / "skills" / slug / "SKILL.md"
    if not path.exists():
        raise FileNotFoundError(f"curated skill declared by benchmark is missing: {path}")
    return path.read_text()


def _run_metadata(
    output: Path,
    benchmark_commit: str,
    model: str | None,
    config: PilotConfig,
) -> dict[str, object]:
    source_root = Path(__file__).resolve().parents[2]
    source = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        text=True,
        capture_output=True,
    )
    codex = subprocess.run(["codex", "--version"], text=True, capture_output=True)
    cache_digests = sorted(
        path.stem for path in (output / "cache").glob("*/*.json")
    )
    return {
        "adapter": "host-native",
        "official_harbor_score": False,
        "benchmark_commit": benchmark_commit,
        "native_verifier": f"unmodified SkillEvolBench tests/test.sh@{benchmark_commit}",
        "isl_dual_commit": source.stdout.strip() if source.returncode == 0 else None,
        "codex_model": model or "codex_default",
        "codex_cli": codex.stdout.strip() if codex.returncode == 0 else None,
        "pilot_config": asdict(config),
        "component_cache_digests": cache_digests,
    }


def _with_checkpointed_verifiers(
    bundle: FamilyBundle,
    cache: JSONCache,
) -> FamilyBundle:
    def wrap(
        task: AcquisitionTask | DeploymentTask,
    ) -> AcquisitionTask | DeploymentTask:
        verifier = task.verifier.inner if isinstance(task.verifier, CachedVerifier) else task.verifier
        return replace(task, verifier=CachedVerifier(task.id, verifier, cache))

    return FamilyBundle(
        bundle.family_id,
        [wrap(task) for task in bundle.acquisition],
        [wrap(task) for task in bundle.deployment],
    )


def _verified_artifact_rewards(
    tasks: list[AcquisitionTask],
    checkpoint: Path,
) -> dict[str, float]:
    """Persist both passing and failing expert-artifact preflight evidence."""
    digests = {
        task.id: hashlib.sha256(
            json.dumps(task.expert_artifact, sort_keys=True).encode()
        ).hexdigest()
        for task in tasks
    }
    if checkpoint.exists():
        value = json.loads(checkpoint.read_text())
        if value.get("artifact_digests") == digests and "rewards" in value:
            return {
                str(key): float(reward)
                for key, reward in value["rewards"].items()
            }

    rewards: dict[str, float] = {}
    failures: dict[str, str] = {}
    for task in tasks:
        evaluator = getattr(task.verifier, "evaluate", None)
        if evaluator is not None:
            reward, failure = evaluator(task.expert_artifact)
        else:
            reward, failure = task.verifier(task.expert_artifact), None
        reward = float(reward)
        rewards[task.id] = reward
        if reward < 1.0:
            failures[task.id] = (
                failure
                or "expert artifact received native reward below 1.0 without verifier diagnostics"
            )

    payload = {
        "artifact_digests": digests,
        "all_passed": all(reward >= 1.0 for reward in rewards.values()),
        "rewards": rewards,
        "failures": failures,
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(checkpoint)
    return rewards


def _edge_shuffle_control(
    graph: Graph,
    deployment_tasks: list[DeploymentTask],
    executor: CachedExecutor,
    seed: int,
) -> dict[str, object]:
    if not graph.edges:
        return {
            "status": "not_applicable",
            "reason": "selected graph has no edges to shuffle",
        }
    try:
        shuffled = edge_shuffle(graph, seed)
    except RuntimeError as error:
        return {"status": "not_applicable", "reason": str(error)}
    skill = compile_graph_to_skill(shuffled)
    scores = _evaluate_graph(
        _skill_graph("edge-shuffle", skill),
        ("skill", "verify"),
        deployment_tasks,
        executor,
    )
    return {
        "status": "completed",
        "graph": graph_to_dict(shuffled),
        "task_scores": scores,
        "mean_reward": sum(scores.values()) / len(scores),
    }


def _artifact_shuffle_control(
    benchmark_root: Path,
    family_id: str,
    target_bundle: FamilyBundle,
    output: Path,
    model: str | None,
    config: PilotConfig,
    benchmark_commit: str,
) -> dict[str, object]:
    """Run full ISL with target inputs paired to a deterministic donor family."""
    families = sorted(
        {
            str(yaml.safe_load(path.read_text())["family_id"])
            for path in (benchmark_root / "benchmark" / "tasks").glob("*/task-spec.yaml")
        }
    )
    donor_id = families[(families.index(family_id) + 1) % len(families)]
    donor = load_family(
        benchmark_root,
        donor_id,
        artifact_cache=output / "donor_artifacts" / benchmark_commit,
    )
    control_cache = JSONCache(output / "cache")
    donor = _with_checkpointed_verifiers(donor, control_cache)
    target_bundle = _with_checkpointed_verifiers(target_bundle, control_cache)
    donor_artifact_rewards = _verified_artifact_rewards(
        donor.acquisition,
        output / "donor_artifacts" / benchmark_commit / "native-verification.json",
    )
    if any(reward < 1.0 for reward in donor_artifact_rewards.values()):
        raise RuntimeError(
            f"artifact-shuffle donor outcomes fail native verifier: {donor_artifact_rewards}"
        )
    shuffled_tasks = [
        replace(task, expert_artifact=donor.acquisition[index].expert_artifact)
        for index, task in enumerate(target_bundle.acquisition)
    ]
    client = CodexJSON(model=model)
    result = train_inverse_skill(
        shuffled_tasks,
        CachedProposer(CodexProposer(client), control_cache),
        CachedCritic(CodexCritic(client), control_cache),
        CachedExecutor(CodexExecutor(model=model), control_cache),
        CachedMutator(CodexMutator(client), control_cache),
        config,
        evidence_journal=EvidenceJournal(output / "evidence.json"),
    )
    audit_no_curated_access(benchmark_root, result.skill)
    executor = CachedExecutor(CodexExecutor(model=model), control_cache)
    graph = _skill_graph("artifact-shuffle", result.skill)
    scores = _evaluate_graph(
        graph, ("skill", "verify"), target_bundle.deployment, executor
    )
    payload = {
        "status": "completed",
        "donor_family": donor_id,
        "donor_artifact_rewards": donor_artifact_rewards,
        "winner": result.graph.id,
        "task_scores": scores,
        "mean_reward": sum(scores.values()) / len(scores),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument(
        "--defer-expensive-controls",
        action="store_true",
        help="defer cross-family artifact shuffle until after primary GO gates",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_family(
                args.benchmark_root,
                args.family,
                args.output,
                args.model,
                run_expensive_controls=not args.defer_expensive_controls,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
