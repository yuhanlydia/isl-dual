import pytest

from isl_dual.graph import InvalidGraph, validate_graph
from isl_dual.leakage import SecretBundle, assert_forward_input
from isl_dual.models import Graph, OrGroup
from isl_dual.mcts import STOP, legal_actions
from isl_dual.metrics import entropy, spearman, spurious_skill_rejection_rate
from isl_dual.supervisor import supervise
from isl_dual.controls import artifact_shuffle, edge_shuffle
from isl_dual.cache import CachedExecutor, JSONCache
from isl_dual.baselines import Baseline, deterministic_plan, upper_information_skill
from isl_dual.compile import compile_graph_to_skill
from isl_dual.executor import CodexExecutor
from isl_dual.report import go_gate
from isl_dual.subprocesses import run_process_group
from isl_dual.pipeline import train_inverse_skill
from isl_dual.toy import ToyCritic, ToyExecutor, ToyMutator, ToyProposer, node, toy_tasks


def test_cycle_is_rejected():
    graph = Graph("bad", (node("a", "a"), node("b", "b")), (("a", "b"), ("b", "a")))
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
