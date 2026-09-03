import json

from isl_dual.codex_components import CodexMutator, graph_to_dict
from isl_dual.config import PilotConfig
from isl_dual.executor import CodexExecutor
from isl_dual.experiment import _verified_artifact_rewards
from isl_dual.models import AcquisitionTask, Graph
from isl_dual.pipeline import train_inverse_skill
from isl_dual.skillevol_host import snapshot
from isl_dual.toy import ToyCritic, ToyExecutor, ToyMutator, ToyProposer, node, toy_tasks


def test_snapshot_omits_generated_dependency_trees(tmp_path):
    (tmp_path / "source.py").write_text("print('keep')")
    generated = tmp_path / "node_modules" / "pkg"
    generated.mkdir(parents=True)
    (generated / "huge.js").write_text("generated")
    venv = tmp_path / ".venv" / "lib"
    venv.mkdir(parents=True)
    (venv / "site.py").write_text("generated")

    value = snapshot(tmp_path)

    assert "source.py" in value["files"]
    assert not any("node_modules" in path for path in value["files"])
    assert not any(".venv" in path for path in value["files"])


def test_mutator_shortfall_is_nonfatal_and_calls_are_operation_conditioned():
    parent = Graph("parent", (node("a", "inspect"), node("b", "solve")), (("a", "b"),))

    class AlwaysInvalidClient:
        def __init__(self):
            self.schemas = []

        def call(self, prompt, schema):
            self.schemas.append(schema)
            value = graph_to_dict(parent)
            value["metadata"] = {"mutation": "CHANGE_BRANCH"}
            return value

    client = AlwaysInvalidClient()
    mutants = CodexMutator(client).mutate(parent, {}, {}, {}, count=3)

    assert mutants == []
    assert client.schemas
    assert all(
        len(schema["properties"]["metadata"]["properties"]["mutation"]["enum"]) == 1
        for schema in client.schemas
    )


def test_second_forward_round_reuses_parent_evidence():
    class CountingExecutor(ToyExecutor):
        def __init__(self):
            self.calls = 0

        def execute(self, task, graph, plan):
            self.calls += 1
            return super().execute(task, graph, plan)

    config = PilotConfig(mcts_budget=1)
    executor = CountingExecutor()
    result = train_inverse_skill(
        toy_tasks(), ToyProposer(), ToyCritic(), executor, ToyMutator(), config=config
    )
    mutant_count = len(result.candidates) - config.candidate_graphs
    expected = (
        config.candidate_graphs * config.acquisition_tasks * config.mcts_budget
        + mutant_count * config.acquisition_tasks * config.mcts_budget
    )
    assert executor.calls == expected


def test_failed_expert_artifact_preflight_is_checkpointed_with_diagnostics(tmp_path):
    class Verifier:
        def evaluate(self, output):
            return 0.25, "reference artifact failed hidden check"

        def __call__(self, output):
            return self.evaluate(output)[0]

    task = AcquisitionTask("t1", "task", {"delta": {"a.py": "x = 1"}}, Verifier())
    checkpoint = tmp_path / "native.json"

    rewards = _verified_artifact_rewards([task], checkpoint)

    assert rewards == {"t1": 0.25}
    payload = json.loads(checkpoint.read_text())
    assert payload["all_passed"] is False
    assert payload["failures"]["t1"] == "reference artifact failed hidden check"


def test_executor_dependency_installs_use_ephemeral_caches(tmp_path, monkeypatch):
    (tmp_path / "requirements.txt").write_text("example-package==1.0\n")
    (tmp_path / "package-lock.json").write_text("{}")
    calls = []

    class Completed:
        returncode = 0

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return Completed()

    monkeypatch.setattr("isl_dual.executor.subprocess.run", fake_run)
    CodexExecutor()._prepare_dependencies(tmp_path)

    pip_command, pip_kwargs = calls[0]
    npm_command, npm_kwargs = calls[1]
    assert "--no-cache-dir" in pip_command
    assert npm_command[:2] == ["npm", "ci"]
    assert npm_kwargs["env"]["npm_config_cache"].startswith(str(tmp_path))
