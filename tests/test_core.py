import pytest

from isl_dual.graph import InvalidGraph, validate_graph
from isl_dual.leakage import SecretBundle, assert_forward_input
from isl_dual.models import AcquisitionTask, Graph, OrGroup, Utility
from isl_dual.mcts import STOP, legal_actions
from isl_dual.mcts import TreeState, _uct, mcts
from isl_dual.metrics import entropy, spearman, spurious_skill_rejection_rate
from isl_dual.supervisor import supervise
from isl_dual.controls import artifact_shuffle, edge_shuffle
from isl_dual.cache import CachedExecutor, CachedJSONClient, JSONCache
from isl_dual.baselines import Baseline, deterministic_plan, full_trajectory_skill, upper_information_skill
from isl_dual.compile import compile_graph_to_skill, operational_pruning
from isl_dual.codex_components import validate_declared_mutation
from isl_dual.config import PilotConfig
from isl_dual.executor import CodexExecutor
from isl_dual.experiment import _edge_shuffle_control, _verified_artifact_rewards
from isl_dual.report import go_gate
from isl_dual.subprocesses import run_process_group
from isl_dual.pipeline import train_inverse_skill
from isl_dual.status import summarize
from isl_dual.toy import ToyCritic, ToyExecutor, ToyMutator, ToyProposer, node, toy_tasks


def test_cycle_is_rejected():
    graph = Graph("bad", (node("a", "a"), node("b", "b")), (("a", "b"), ("b", "a")))
    with pytest.raises(InvalidGraph):
        validate_graph(graph)


def test_duplicate_edge_is_rejected():
    graph = Graph("bad", (node("a", "a"), node("b", "b")), (("a", "b"), ("a", "b")))
    with pytest.raises(InvalidGraph):
        validate_graph(graph)


def test_required_cannot_depend_on_optional():
    graph = Graph("bad", (node("a", "a", False), node("b", "b", True)), (("a", "b"),))
    with pytest.raises(InvalidGraph):
        validate_graph(graph)


def test_forward_leakage_guard():
    with pytest.raises(AssertionError):
        assert_forward_input("prompt contains SECRET", SecretBundle(expert_artifact="SECRET"))
    with pytest.raises(AssertionError):
        assert_forward_input({"copied": {"answer": [1, 2, 3]}}, SecretBundle(expert_artifact={"answer": [1, 2, 3]}))


def test_dual_loop_runs_and_compiles_procedure():
    result = train_inverse_skill(toy_tasks(), ToyProposer(), ToyCritic(), ToyExecutor(), ToyMutator())
    assert "## Procedure" in result.skill
    assert all("private-artifact" not in line for line in result.skill.splitlines())
    assert len(result.q0) == 8
    assert 8 < len(result.posterior) <= 12
    assert abs(sum(result.posterior.values()) - 1.0) < 1e-9


def test_pilot_rejects_unimplemented_outer_loop_count():
    with pytest.raises(ValueError, match="T_outer=2"):
        train_inverse_skill(
            toy_tasks(), ToyProposer(), ToyCritic(), ToyExecutor(), ToyMutator(),
            PilotConfig(outer_loops=3),
        )


def test_scientific_metrics():
    assert entropy({"a": 0.5, "b": 0.5}) > 0
    assert spearman([1, 2, 3], [2, 4, 8]) == pytest.approx(1.0)
    assert spurious_skill_rejection_rate(
        {"g": 0.9}, {"g": 0.1}, {"g": 0.8}, {"g": 0.2}, 0.8, 0.2,
    ) == 1.0


def test_supervisor_records_completion(tmp_path):
    state = tmp_path / "supervisor.json"
    assert supervise(["/bin/true"], 0.001, state) == 0
    assert '"status": "completed"' in state.read_text()


def test_supervisor_enforces_wall_clock_deadline(tmp_path):
    state = tmp_path / "deadline.json"
    assert supervise(["/bin/sleep", "10"], 0.00001, state) != 0
    assert '"status": "deadline_reached"' in state.read_text()


def test_edge_shuffle_preserves_nodes_and_edge_count():
    graph = Graph("g", (node("a", "a"), node("b", "b"), node("c", "c")), (("a", "b"),))
    shuffled = edge_shuffle(graph, 4)
    assert shuffled.nodes == graph.nodes
    assert len(shuffled.edges) == len(graph.edges)
    assert shuffled.edges != graph.edges


def test_rollout_cache_keeps_duplicate_occurrences_distinct(tmp_path):
    class CountingExecutor:
        def __init__(self): self.calls = 0
        def execute(self, task, graph, plan):
            self.calls += 1
            return {"call": self.calls}
    task = toy_tasks()[0]
    graph = Graph("g", (node("a", "a"), node("b", "b")), (("a", "b"),))
    inner = CountingExecutor(); cached = CachedExecutor(inner, JSONCache(tmp_path))
    assert cached.execute(task, graph, ("a", "b"))["call"] == 1
    assert cached.execute(task, graph, ("a", "b"))["call"] == 2
    resumed_inner = CountingExecutor(); resumed = CachedExecutor(resumed_inner, JSONCache(tmp_path))
    assert resumed.execute(task, graph, ("a", "b"))["call"] == 1
    assert resumed.execute(task, graph, ("a", "b"))["call"] == 2
    assert resumed_inner.calls == 0


def test_or_group_selects_exactly_one_branch_before_stop():
    graph = Graph(
        "or", (node("a", "a"), node("b", "b", True), node("c", "c", False)),
        (("a", "b"), ("a", "c")), (OrGroup("branch", ("b", "c"), True),),
    )
    assert set(legal_actions(graph, ("a",), 12)) == {"b", "c"}
    assert legal_actions(graph, ("a", "b"), 12) == [STOP]


def test_or_branch_edges_are_alternative_dependencies_for_merge_node():
    graph = Graph(
        "diamond",
        (node("a", "start"), node("b", "branch b", False), node("c", "branch c", False), node("d", "merge")),
        (("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")),
        (OrGroup("branch", ("b", "c"), True),),
    )
    validate_graph(graph)
    assert "d" in legal_actions(graph, ("a", "b"), 12)
    assert "d" in legal_actions(graph, ("a", "c"), 12)
    assert "c" not in legal_actions(graph, ("a", "b"), 12)


def test_baseline_contracts_are_explicit():
    graph = Graph("g", (node("a", "a"), node("b", "b")), (("a", "b"),))
    assert deterministic_plan(graph) == ("a", "b")
    assert upper_information_skill(Baseline.CURATED_SKILL, "procedure").skill == "procedure"


def test_skill_compilation_is_topological_not_json_order():
    graph = Graph("g", (node("b", "second"), node("a", "first")), (("a", "b"),))
    skill = compile_graph_to_skill(graph)
    assert skill.index("first") < skill.index("second")


def test_binary_artifacts_are_losslessly_encoded(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"\x00\xff\x10")
    encoded = CodexExecutor._read_artifact(path)
    assert encoded["encoding"] == "base64"
    import base64
    assert base64.b64decode(encoded["data"]) == b"\x00\xff\x10"


def test_go_gate_matches_pilot_decision_rule():
    gate = go_gate({"B1": 0.2, "B3": 0.4, "B4": 0.6, "B6": 0.8})
    assert gate["go"] is True


def test_isolated_process_accepts_private_stdin():
    result = run_process_group(["/bin/sh", "-c", "read value; printf %s \"$value\""], timeout=2, input_text="private prompt\n")
    assert result.stdout == "private prompt"


def test_one_failed_rollout_becomes_zero_reward_not_family_abort():
    class FailingExecutor:
        def execute(self, task, graph, plan): raise RuntimeError("transient")
    graph = Graph("g", (node("a", "a"), node("b", "b")), (("a", "b"),))
    result = mcts(graph, toy_tasks()[0], FailingExecutor(), budget=1)
    assert result.rewards == [0.0]
    assert "RuntimeError" in result.rollouts[0].failure


def test_visited_stop_action_has_decaying_exploration_bonus():
    state = TreeState(prefix=("a",), visits=8)
    unseen = _uct(state, STOP, 1.4)
    state.action_visits[STOP] = 8
    state.action_values[STOP] = 0.25
    visited = _uct(state, STOP, 1.4)
    assert visited < unseen


def test_status_reports_latest_cache_activity(tmp_path):
    cache = tmp_path / "families" / "f" / "cache" / "executor"
    cache.mkdir(parents=True)
    (cache / "one.json").write_text("{}")
    summary = summarize(tmp_path)
    assert summary["cache_records"] == {"executor": 1}
    assert summary["latest_cache_activity"]["executor"].endswith("+00:00")


def test_non_graph_json_calls_are_checkpointed(tmp_path):
    class Client:
        model = "test"
        def __init__(self): self.calls = 0
        def call(self, prompt, schema):
            self.calls += 1
            return {"skill": "procedure"}
    inner = Client()
    client = CachedJSONClient(inner, JSONCache(tmp_path))
    assert client.call("prompt", {"type": "object"}) == {"skill": "procedure"}
    assert client.call("prompt", {"type": "object"}) == {"skill": "procedure"}
    assert inner.calls == 1


def test_full_trajectory_baseline_requires_explicit_trajectory_per_task():
    class Client:
        def call(self, prompt, schema): return {"skill": "# Skill\n\n## Procedure\n\n1. Diagnose."}
    with pytest.raises(ValueError):
        full_trajectory_skill(toy_tasks(), ["only one"], Client())


def test_edge_control_explicitly_reports_graph_without_edges():
    graph = Graph("g", (node("a", "a"), node("b", "b")))
    result = _edge_shuffle_control(graph, [], None, 7)
    assert result["status"] == "not_applicable"


def test_passing_expert_artifact_verification_is_checkpointed(tmp_path):
    calls = []
    def verifier(artifact):
        calls.append(artifact)
        return 1.0
    tasks = [AcquisitionTask("t1", "x", {"delta": {"a": "b"}}, verifier)]
    checkpoint = tmp_path / "native.json"
    assert _verified_artifact_rewards(tasks, checkpoint) == {"t1": 1.0}
    assert _verified_artifact_rewards(tasks, checkpoint) == {"t1": 1.0}
    assert len(calls) == 1


def test_declared_edge_mutation_rejects_hidden_node_edit():
    parent = Graph("g", (node("a", "a"), node("b", "b")))
    valid = Graph("m", parent.nodes, (("a", "b"),))
    validate_declared_mutation(parent, valid, "ADD_EDGE")
    changed = Graph("m2", (node("a", "changed"), parent.nodes[1]), (("a", "b"),))
    with pytest.raises(InvalidGraph):
        validate_declared_mutation(parent, changed, "ADD_EDGE")


def test_pruning_preserves_required_or_group_semantics():
    graph = Graph(
        "g", (node("a", "a"), node("b", "b", False), node("c", "c", False)),
        or_groups=(OrGroup("choice", ("b", "c"), True),),
    )
    pruned = operational_pruning(graph, {("g", "b"): Utility(-0.2, 2, 2)})
    assert {item.id for item in pruned.nodes} == {"a", "c"}
    assert pruned.node_map()["c"].required is True
    assert pruned.or_groups == ()
