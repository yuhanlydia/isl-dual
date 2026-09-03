# Pilot status (2026-09-03)

The long-running processes were stopped on request. Existing run directories and
checkpointed evidence were preserved; no run data was deleted.

## Current conclusion

There is **not yet a valid positive scientific result**. The available records are
exploratory/acquisition-time evidence only. No corrected six-family pilot produced a
complete `result.json`, frozen deployment scores for T4--T6, or the required B0--B8
GO-gate and transfer-correlation report. Therefore the current run must not be used as
the paper's main result.

## Preserved namespaces

- `runs/skillevol-10h`: exploratory campaign. It contains partial family evidence but
  no complete campaign result.
- `runs/v1-corrected`: corrected mechanism-pilot checkpoint. It contains E1-LS1 and
  partial E3-LS1 rollout evidence, plus the atomic mechanism manifest; it has no
  completed pilot result.

The corrected code already includes the three required algorithmic safeguards:

1. MCTS `STOP` is a terminal, explorable action and executes exactly the current
   prefix.
2. The greedy baseline reuses MCTS dependency semantics, including OR groups.
3. Mutation children receive a mass-conserving transition prior (`mu=0.3`) instead of
   copying the parent's posterior log weight.

## Why the corrected pilot did not finish

The manifest records the following blocking observations:

- E1-LS1: the Codex mutator failed to produce three valid `CHANGE_BRANCH` mutants
  after its bounded retry budget.
- E2-LS1: the benchmark expert artifact for T2 received native reward `0.0`, so the
  family was correctly rejected by the artifact preflight.
- E3--E6: materialization failed with `OSError: [Errno 28] No space left on device`.

The disk-pressure issue was mitigated by cleaning only the disposable UV package
cache (about 8.6 GiB); run evidence and benchmark data were retained. A later resume
attempt was stopped before it could establish a complete result. GPU utilization was
zero during this work: the pipeline performs Codex/CPU orchestration, not parameter
training.

## Required next run

Before making claims or scaling to all 30 families:

1. Keep the corrected namespace and resume atomically, or start a clearly named
   `v1-corrected-rerun` namespace; never overwrite evidence.
2. Resolve the E1 mutation-schema failure and the E2 native-artifact preflight failure
   (or report those families as excluded with a prespecified rule).
3. Run all six mechanism families to completion, including deployment and ablations:
   within-family artifact permutation, 1/2/3-artifact curves, search-budget controls,
   and spurious-DAG diagnostics.
4. Report paired family differences, B0/B1/B3/B4/B5/B6/B8, forward-vs-deployment and
   static-vs-deployment Spearman correlations, posterior entropy, top-1 identification,
   skill lift, and curated-gap closure.

The minimum scientific gates remain `B1 > B0`, forward-tested selection above static
selection, and `B6 > B1`/`B6 > B3` on paired family means. Until those are measured,
the method is an implementation pilot rather than a validated result.
