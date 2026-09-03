# ISL-Dual next experiment plan (2026-09-03)

## Current scientific status

The existing run does **not** falsify ISL-Dual. No corrected run completed the required frozen T4--T6 deployment comparisons. The observed blockers were implementation/infrastructure failures: mutation-schema shortfall, stale/failed artifact preflight evidence, and disk exhaustion. A clean CI reproduction pinned to SkillEvolBench commit `9e3daa339987c3cfa624121e1be442593a53d43c` now verifies all E2-LS1 acquisition expert artifacts at native reward 1.0.

The next experiment must use a fresh namespace. Do not mix old `runs/skillevol-10h` or `runs/v1-corrected` caches with the repaired code.

## Primary hypothesis

The paper-level hypothesis remains:

\[
\text{final expert artifacts} \rightarrow q_0(G)
\rightarrow \text{forward execution} \rightarrow q_1(G)
\rightarrow \text{graph evolution} \rightarrow q_2(G)
\]

and forward operational evidence should identify more transferable procedures than artifact plausibility alone.

The decisive comparisons are:

- `B0`: no skill.
- `B1`: direct outcome-to-text skill.
- `B2`: one-shot DAG.
- `B3`: static q0 winner. This is now reused directly from the B6 run.
- `B4`: greedy forward execution on the shared candidate pool.
- `B5`: first-loop MCTS q1 winner. This is now reused directly from the B6 run.
- `B6`: full dual loop with mutation/evolution.
- `B7`: oracle solution-procedure upper-information control.
- `B8`: curated-skill upper-information control.

The primary GO gates are:

\[
B1>B0,
\qquad
\max(B4,B5)>B3,
\qquad
B6>B1,
\qquad
B6>B3.
\]

Evolution is specifically supported when `B6 > B5`. The strongest mechanism diagnostic is

\[
\rho(F_{\text{forward}},R_{\text{deployment}})
>
\rho(A_{\text{static}},R_{\text{deployment}}).
\]

## Stage 1: two-family signal check

Use the repaired branch/main code and a fresh namespace:

```bash
isl-dual-mechanism \
  --benchmark-root /path/to/SkillEvolBench \
  --output runs/v2-primary-2fam \
  --model gpt-5.4 \
  --families E1-LS1 E2-LS1
```

Fixed settings remain `K=8` candidate DAGs, three acquisition tasks, three frozen deployment tasks, MCTS budget 8, two outer loops, `beta_artifact=2`, `beta_forward=4`, `mu=0.3`, maximum pool 12. Do not tune these on T4--T6.

This stage intentionally defers cross-family artifact shuffle, 1/2/3-artifact curves, 4/8/16 search sweeps, and spurious-DAG corruptions. First obtain complete B0--B8 deployment results and candidate-level forward/deployment correlations.

Decision rule: continue to Stage 2 if the paired direction favors B6 over B1/B3 and forward evidence ranks held-out candidates better than static artifact scores. If both families show flat `B3 ~= B5 ~= B6`, stop scaling and inspect the skill representation rather than increasing rollout budget.

## Stage 2: six-environment mechanism pilot

If Stage 1 has signal, run one LS1 family from every SkillEvolBench environment:

```bash
isl-dual-mechanism \
  --benchmark-root /path/to/SkillEvolBench \
  --output runs/v2-primary-6fam \
  --model gpt-5.4
```

Report family-paired differences for B6-B0, B6-B1, B5-B3, B6-B5, and B6-B3; forward-vs-deployment and static-vs-deployment Spearman correlations; posterior entropy; top-1 skill identification; skill lift; and fraction of the curated-skill gap closed.

Do not scale to all 30 families unless the six-family mean supports the primary gates.

## Stage 3: diagnostics only after primary signal

Reuse the completed six-family namespace:

```bash
isl-dual-mechanism \
  --benchmark-root /path/to/SkillEvolBench \
  --output runs/v2-primary-6fam \
  --model gpt-5.4 \
  --diagnostics
```

The diagnostic stage runs cross-family artifact shuffle, within-family artifact permutation, 1/2/3 acquisition-artifact curves, equal-budget random/greedy/MCTS comparisons at budgets 4/8/16, and controlled spurious-DAG corruptions.

For the search study, compare held-out ranking correlation as well as acquisition reward. A search method that obtains higher acquisition reward but lower forward-to-deployment correlation is overfitting the acquisition cases and should not be preferred.

## Stage 4: full benchmark

Only after Stage 2/3 pass the mechanism gates, run all 30 families with three independent runs. Use family as the paired bootstrap unit and report 95% confidence intervals. The main claim is not that outcome-only supervision must beat the curated upper bound; it is that the inverse-forward loop improves over information-matched outcome-only baselines and closes a meaningful fraction of the gap to trajectory/procedure-rich supervision.

## Resource policy

### Current Codex experiment

The main pipeline trains no neural parameters and uses remote Codex inference. Local GPU use is therefore **0 GB VRAM**. The 16 GB constraint is automatically satisfied. Recommended host resources are 8--16 CPU cores, 16--32 GB RAM, and at least 30--50 GB free SSD space. Put `TMPDIR` on the large disk. Keep executor concurrency at 1 until a complete family succeeds; model JSON calls may later be parallelized conservatively.

The repaired code reduces the dominant remote-execution cost per primary family:

- B5 reuses the B6 first-loop q1/MCTS evidence: about 192 duplicate acquisition rollouts removed at K=8, 3 tasks, budget 8.
- Round two evaluates only retained mutants; unchanged parents reuse round-one evidence: another 192 parent rollouts removed.
- Cross-family artifact shuffle is deferred until diagnostics: roughly another full dual-loop control is removed from the primary critical path.
- Deterministic greedy search is evaluated once per graph/task rather than repeated to imitate a rollout budget.

This makes the primary run both cheaper and scientifically cleaner because nested ablations share the same stochastic candidate/evidence realization.

### Optional 16 GB local-model reproducibility experiment

Do this only after ISL-Dual passes the Codex mechanism gate. Use a 7--8B instruct/coder model quantized to 4-bit as the hard 16 GB configuration. For proposer/critic/mutator structured-output calls, use a local serving engine and batch 2--4 short requests when KV-cache headroom permits. Keep agentic executor/tool-use rollouts at batch 1 because each rollout owns a mutable workspace. Use 16k context first; increase only after measured VRAM headroom. A 24 GB card may use a larger batch or a larger 4-bit model, but no main-paper result should require more than the 16 GB configuration.

## Fresh-run requirements

Every official namespace must record the ISL-Dual commit, pinned SkillEvolBench commit, Codex CLI/model identity, artifact digests, verifier-script digests, q0/q1/q2, graph-attributed rollout evidence, deployment scores, and failure diagnostics. Never reuse a failed expert-artifact cache from an older code revision. A family with a fresh expert artifact reward below 1.0 must be stopped and diagnosed before any ISL comparison is interpreted.
