# Pilot status (2026-09-03)

## Scientific conclusion

There is **not yet a valid positive scientific result** because no repaired run has completed the frozen T4--T6 deployment comparisons needed for the B0--B8 GO gates and transfer-correlation analysis. The old `runs/skillevol-10h` and `runs/v1-corrected` namespaces remain exploratory evidence only and must not be used as the paper's main result.

The important update is that the known blockers now have engineering explanations and regression coverage; none of them is evidence that the inverse-skill hypothesis failed.

## Repaired implementation

The repair branch fixes the previous blockers and redundant experiment cost:

1. MCTS `STOP` remains terminal/explorable and B4 uses the same OR/dependency semantics as MCTS.
2. Mutation posterior mass remains conserved with `mu=0.3`.
3. The Codex mutator is conditioned on structurally feasible mutation operators. Failure to produce all requested mutants no longer aborts a family; the dual loop continues with the valid neighborhood actually found.
4. Round two reuses first-round evidence for unchanged parent DAGs and executes only retained mutants. Mutants are scored by forward improvement relative to their parent.
5. B3 and B5 are exact nested ablations of B6: B3 reuses q0 and B5 reuses q1/first-loop MCTS evidence instead of rerunning a stochastic copy.
6. Artifact snapshots exclude generated dependency/cache trees such as `node_modules`, `.venv`, and package caches. Pip/npm installation uses ephemeral no-cache paths.
7. Failed expert-artifact preflights are checkpointed with verifier diagnostics instead of being lost.
8. The mechanism runner is primary-first. Cross-family artifact shuffle, learning curves, search-budget sweeps, and spurious-DAG diagnostics are deferred until primary GO/NO-GO evidence exists.

## E2 artifact failure resolved

The earlier E2-LS1-T2 expert artifact reward of `0.0` was not reproduced on a clean runner. CI now pins official SkillEvolBench commit `9e3daa339987c3cfa624121e1be442593a53d43c` and runs the full host-native materialize/replay/verifier chain. E2-LS1-T1, T2, and T3 each receive native reward `1.0` with no verifier failure. This indicates that the old T2=0 observation came from the previous run environment/cache/disk state rather than an intrinsically invalid benchmark solution.

## Verification

The repaired PR merge snapshot passes the complete Python suite (`43 passed`) and the pinned E2 native-preflight integration job. These checks validate code and benchmark replay mechanics; they do **not** substitute for running the Codex-backed SkillEvolBench scientific experiment.

## Required next run

Use a fresh namespace, not an old cache. The staged commands and exact GO/NO-GO criteria are in [`NEXT_EXPERIMENT.md`](NEXT_EXPERIMENT.md).

Start with the two-family primary signal check, then the six-environment primary pilot. Run expensive diagnostics only after the primary mechanism has signal. Scale to all 30 families only if paired family means support outcome-only skill lift, forward-tested selection over static selection, and B6 improvement over B1/B3.

The local GPU requirement remains zero for the current Codex-backed experiment: the pipeline performs remote model inference plus CPU graph/MCTS/verifier orchestration. A separate 16 GB local-model reproducibility configuration is specified in `NEXT_EXPERIMENT.md` and should be attempted only after the main mechanism passes its gate.
