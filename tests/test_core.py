import pytest

from isl_dual.graph import InvalidGraph, validate_graph
from isl_dual.leakage import SecretBundle, assert_forward_input
from isl_dual.models import Graph
from isl_dual.metrics import entropy, spearman, spurious_skill_rejection_rate
from isl_dual.supervisor import supervise
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
