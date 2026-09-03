# ISL-Dual

ISL-Dual (**Inverse Skill Learning with an Inverse–Forward Dual Loop**) learns a
portable procedure from successful final artifacts without observing expert trajectories
or curated skills. It trains no model parameters: learning happens over a latent
procedural DAG, its posterior, and evidence from actual forward execution.

> Research pilot. The core engine and isolated Codex executor run end to end.
> SkillEvolBench is an external dependency and is not vendored here.

## Algorithm

For each family, the learner sees only three `(task input, successful final artifact)`
pairs. It proposes eight diverse DAGs, computes a static artifact posterior, and searches
execution plans with MCTS. Every rollout receives a clean workspace and a fresh
`codex exec --ephemeral` process. The native verifier runs only after execution and never
compares against the expert artifact. Forward evidence updates the posterior and drives
one restricted graph-mutation pass. After two outer loops the winner is pruned and
compiled to a frozen `SKILL.md` for T4–T6 deployment.

```text
(x, final artifact) -> DAG posterior -> MCTS plan search
                                         |
                              fresh Codex + workspace
                                         |
                              native verifier reward
                                         |
                         posterior update + mutation
                                         |
                               frozen SKILL.md
```

## Fixed pilot configuration

| Parameter | Value |
|---|---:|
| Acquisition / deployment tasks | 3 / 3 |
| Candidate DAGs | 8 |
| Maximum nodes / plan length | 12 / 12 |
| Outer loops | 2 |
| MCTS budget | 8 |
| UCT constant / rollout stop probability | 1.4 / 0.4 |
| Static / forward weights | 2.0 / 4.0 |
| Stability / complexity penalties | 0.5 / 0.1 |
| Graphs mutated / mutants each | 2 / 3 |
| Maximum graph pool | 12 |
| Utility pruning threshold | 0.05 |

## Install and verify

Python 3.10+ and an authenticated Codex CLI are required.

```bash
python3 -m pip install -e . --no-build-isolation
python3 -m pytest -q
isl-dual smoke --output runs/smoke
```

The deterministic smoke family validates the two-loop algorithm cheaply. A manual
integration check also exercises the actual ephemeral Codex subprocess because calling a
model from unit tests would be costly and flaky.

The requirement-by-requirement implementation map is in
[`docs/SPEC_AUDIT.md`](docs/SPEC_AUDIT.md).

## Components

- `models.py`: typed DAG, tasks, rollouts, utilities, and component protocols.
- `graph.py`: acyclicity, endpoint, node-count, optional-dependency, and OR-group checks.
- `mcts.py`: UCT selection, expansion, randomized legal completion, and mean backup.
- `executor.py`: isolated temporary workspace plus fresh ephemeral Codex per rollout.
- `pipeline.py`: proposal, q0, first forward loop, utility estimation, mutation, q2,
  pruning, compilation, and leakage audit.
- `metrics.py`: entropy, skill lift, Spearman correlation, and spurious-skill rejection.
- `baselines.py`: executable B1–B5 selectors plus explicit B7/B8 information contracts.
- `controls.py`: deterministic artifact-shuffle and DAG-edge-shuffle causal controls.
- `report.py`: candidate-level forward/static transfer correlations and posterior metrics.
- `cache.py`: model-scoped, occurrence-aware checkpoints that preserve fresh rollouts.
- `supervisor.py`: signal-aware wall-clock supervisor for a fixed ten-hour command.

## SkillEvolBench integration

Clone the official benchmark separately and pin its commit. ISL-Dual must adapt its 30
families without importing curated `benchmark/skills/*/SKILL.md` or expert solution
trajectories into model prompts. For every family:

1. Materialize successful T1–T3 final workspaces as outcome artifacts.
2. Supply proposer, critic, executor, mutator, and native verifier implementations.
3. Run acquisition learning and freeze the compiled skill.
4. Execute T4–T6 once with only task plus frozen skill.
5. Persist benchmark commit, model, verifier version, seed, and prompt hashes.

The benchmark uses containerized Harbor tasks. Docker/Harbor readiness is a hard
preflight requirement for official scores. Do not substitute host-side toy verifiers.

## Leakage contract

Runtime assertions enforce:

- proposer/critic: no expert trajectory or curated skill;
- forward executor: no expert artifact, trajectory, or curated skill;
- mutator: graph plus operational evidence only;
- deployment: no expert artifact, posterior update, or hidden reward.

All stages should use `codex exec --ephemeral`. Private artifacts, credentials, run logs,
and workspaces belong under ignored paths; `runs/` is excluded from Git.

## Experiments

Required baselines are B0 no skill, B1 direct outcome-to-text skill, B2 one-shot DAG,
B3 eight DAGs plus static critic, B4 greedy forward, B5 MCTS forward, and B6 full
ISL-Dual. B7 oracle solution procedure (the benchmark's executable `solve.sh`) and
B8 curated skill are upper-information controls. B7 is not an observed agent trajectory;
a true trajectory baseline is reserved for B9 when trajectory logs are available.
Artifact shuffle and edge shuffle are the two required causal controls.

Primary metrics are skill lift, acquisition-forward versus held-out Spearman correlation,
static-score correlation, spurious-skill rejection rate, and posterior entropy change.
The first go/no-go tests are `B4 > B3` and `B6 > B1`. If B3, B4, and B6 are effectively
equal, do not scale the run before revisiting the method.

The real family runner executes B0–B8 and writes per-task scores, family means, and the
GO-gate decision to `result.json`. B0 uses the same deployment executor with no procedural
skill. B7 explicitly treats each acquisition task's benchmark `solution/solve.sh` as its
oracle solution procedure; B8 explicitly reads the benchmark-declared curated `SKILL.md`.
Neither source is loaded by B1–B6 or fed into ISL. Non-graph skill-generation
calls are checkpointed just like proposer and critic calls. The runner also freezes and
deploys every initial candidate graph for candidate-level correlation metrics without
feeding held-out rewards back into learning.

After the main and upper-information baselines, each family runner executes both causal
controls in separate namespaces. Edge shuffle preserves the selected nodes and edge count,
randomizes only valid dependency edges, freezes the resulting skill, and evaluates it on
deployment. Artifact shuffle pairs the family's acquisition inputs with the next family's
successful outcomes (a deterministic cyclic derangement), reruns the complete dual loop,
and evaluates the frozen result on the original family's deployment tasks. A zero-edge
winner is recorded as `not_applicable` for edge shuffle rather than assigned a fabricated
control score.

## Long-run discipline

- Use a wall-clock deadline and atomic checkpoints at family/task/graph boundaries.
- Resume completed units rather than restarting paid rollouts.
- Handle SIGINT/SIGTERM by saving state before exit.
- Never tune on deployment tasks or feed hidden verifier results back to the agent.
- A ten-hour soak run is meaningful only after Docker, Harbor, artifacts, and native
  verifiers pass preflight.

After strict preflight succeeds, wrap the real runner as follows:

```bash
isl-dual --hours 10 --output runs/official-10h supervise -- \
  python3 path/to/skillevol_adapter.py --resume
```

The supervisor writes `supervisor.json`, forwards SIGINT/SIGTERM, stops at the deadline,
and preserves the child's exit status. The child runner remains responsible for
algorithm-level family/task/graph checkpoints.

Inspect a live run without printing prompts or artifact contents:

```bash
isl-dual-status runs/skillevol-10h
```

For the official 30-family queue, the host-native research adapter can be launched with:

```bash
isl-dual --hours 10 --output runs/skillevol-10h supervise -- \
  isl-dual-campaign --benchmark-root /path/to/SkillEvolBench \
  --output runs/skillevol-10h --model gpt-5.4
```

The campaign immediately advances after a completed family and atomically records failures
before continuing. Its verifier invokes each task's unmodified `tests/test.sh`; this adapter
must be reported as host-native and is not a substitute for an official Harbor score unless
the expert outcome for all three acquisition tasks first receives native reward `1.0`.
If all 30 primary families finish before the supervisor deadline, the campaign immediately
starts independent full-family replications with incremented fixed seeds. It continues
useful robustness work and checkpointing until the wall-clock supervisor stops it.

## Corrected mechanism pilot

The latest audited state and known failures are recorded in
[`docs/PILOT_STATUS.md`](docs/PILOT_STATUS.md). The pilot is not considered a
scientific success until its deployment results and prespecified GO gates are complete.

The exploratory namespace is retained as `runs/skillevol-10h`. Before scientific claims,
run the corrected mechanism pilot in a new namespace:

```bash
isl-dual-mechanism --benchmark-root /path/to/SkillEvolBench \
  --output runs/v1-corrected --model gpt-5.4
```

This six-family pilot uses one `LS1` family from each benchmark environment and records
the primary B0/B1/B3/B4/B5/B6/B8 results plus within-family artifact permutation, 1/2/3
artifact learning curves, equal-budget random/greedy/MCTS search, and controlled spurious
graph diagnostics. It is resumable through `runs/v1-corrected/mechanism.json` and never
reuses the exploratory namespace. MCTS STOP is terminal and explorable, B4 shares the
MCTS OR-group dependency semantics, and the second-round mutation prior conserves q1
probability mass with `mu=0.3`.

## License

MIT
