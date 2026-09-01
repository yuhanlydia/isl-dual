# ISL-Dual specification audit

This matrix maps the fixed pilot specification to executable code and persisted evidence.
It is an implementation audit, not a claim of an official SkillEvolBench score. The active
campaign uses the unmodified native `tests/test.sh` scripts through the host-native adapter
because nested Docker/Harbor cannot run in the current container (`unshare` is denied).

| Spec | Requirement | Implementation / evidence |
|---:|---|---|
| 1, 6 | Outcome-only acquisition; no trajectory or curated skill | `skillevol_host.py` materializes only final deltas; `codex_components.observable_artifact` removes private replay fields; proposer prompt prohibits trajectory reconstruction and case copying. |
| 2 | Typed latent `G=(V,E,O)` DAG and six node fields | `models.py` defines `Graph`, `Node`, and `OrGroup`; JSON schemas require every procedural field. |
| 3, 10–15 | Search legal execution plans, not graphs, with UCT | `mcts.py` implements plan-prefix states, legal actions, UCT `c=1.4`, expansion, seeded completion, STOP, execution, verifier evaluation, and running-mean backup. |
| 3, 11 | Required/optional nodes and OR branches | `legal_actions`, `requirements_satisfied`, and `dependencies_satisfied` enforce required nodes, exclusive OR choice, and any-of dependencies for converging branches. |
| 4 | Inverse–forward posterior/evolution loop | `pipeline.train_inverse_skill` runs proposal → static posterior → forward loop → mutation → second forward loop → freeze. |
| 5 | K=8, four modes, two candidates each | `PROPOSAL_MODES` and `PilotConfig`; invalid/duplicate candidates are replenished before K is fixed. |
| 7 | Automated DAG validity checks | `graph.validate_graph` checks node bounds, IDs, actions, endpoints, duplicate/self edges, cycles, optional dependencies, and OR membership; invalid proposals/mutants are rejected and regenerated. |
| 8 | Static critic and complexity posterior | `CodexCritic`, `scoring.complexity`, and `pipeline.py` implement the 0.4/0.4/0.2 artifact score, beta 2.0, and complexity penalty 0.1. |
| 9, 15–16 | Fresh forward executor and native verifier | `CodexExecutor` creates a temporary workspace and fresh `codex exec --ephemeral` for every uncached occurrence; `HostNativeVerifier` invokes the task's unmodified tests afterward. |
| 14 | Seeded rollout, required-first, p_stop=0.4, max 12 | `_complete_plan` and `PilotConfig`. |
| 17 | Budget 8, top-two mean, family mean/std | `_forward_loop`, `top2_mean`, and `summarize_forward`. |
| 18–19 | Forward posterior beta 4.0 and stability 0.5 | Both forward updates are explicit in `pipeline.py`; config fixes forward evidence above artifact judgment. |
| 20–21 | Node and edge operational utility, n>=2 | `scoring.estimate_utilities` returns `delta=None` until both comparison groups have at least two rollouts. |
| 22–25 | Seven mutations, top H=2, M=3, max pool 12 | `MUTANT_SCHEMA`, `validate_declared_mutation`, round-robin mutant pooling, and second forward loop enforce the fixed mutation contract without crossover. |
| 24 | Mutation receives operational evidence, not artifacts | `CodexMutator` receives plans, rewards/failures, node utilities, and edge utilities. Its API has no acquisition-task/artifact argument. |
| 26–28 | Argmax graph, operational pruning, SKILL.md compilation | `pipeline.py` selects q2 argmax; `compile.py` prunes supported harmful optional nodes while preserving OR semantics and emits procedural Markdown. |
| 29–30 | Frozen deployment without MCTS or learning feedback | `experiment.py` wraps the frozen skill in a deterministic two-node execution graph and evaluates T4–T6 once; deployment scores are written only after execution. |
| 31–32 | Full algorithm and MCTS pseudocode | `pipeline.py` and `mcts.py`; unit tests exercise the complete toy dual loop and search invariants. |
| 33 | Fixed pilot hyperparameters | `config.PilotConfig`; non-two outer-loop configurations are rejected rather than silently ignored. |
| 34 | B0–B8 | `baselines.py` implements selectors/contracts; `experiment.run_family` actually trains/selects, freezes, deploys, and records all nine baselines. B7 reads explicit acquisition solution scripts; B8 reads the declared curated skill only in its isolated upper-information branch. |
| 35 | Artifact- and edge-shuffle controls | `controls.py` defines the transformations; `experiment.py` executes both in separate cache namespaces and records donor provenance/control scores. |
| 36–38 | Lift, two correlations, SSR, entropy | `metrics.py` and `report.scientific_report`; candidate deployment rewards are evaluation-only and never returned to the learner. |
| 39 | Leakage guards and ephemeral stages | `leakage.py`, executor assertions, prompt construction, separate B7/B8 branch, and fresh structured Codex subprocesses. Prompts are sent over stdin, not command-line arguments. |
| 40 | GO gate | `report.go_gate` records B4>B3, B6>B1, and the flat B3/B4/B6 failure condition in each family result. |

## Persisted evidence

Each native verifier result is atomically journaled by task, verifier-script digest, and
output digest, so a deadline in the middle of a family still leaves durable reward/failure
evidence and deterministic resume does not rerun an already evaluated output. Each completed
family writes `SKILL.md` and `result.json`. The result includes the pinned
benchmark and ISL-Dual commits, model/CLI identity, adapter label, fixed configuration,
component-cache digests, expert-artifact native verification, q0/q1/q2, forward and
deployment scores, B0–B8, controls, scientific metrics, and the GO decision. `campaign.json`
is updated atomically at family boundaries; component calls and rollout occurrences are
individually cached. `supervisor.json` is authoritative for the wall-clock deadline.
