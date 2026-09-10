# Causal Minimal Skill Design

## Goal

Pivot the primary research direction from full ISL-Dual search to the empirically successful mechanism already observed in E2-LS1: **operational pruning of procedural knowledge**. The new method studies whether an initially useful but over-complete skill can be decomposed into modules and compressed to a smaller subset whose causal contribution is validated by execution.

The new primary claim is deliberately narrower than the legacy ISL-Dual claim:

> A skill should contain only procedural modules that causally improve execution. Removing neutral or harmful modules can improve held-out task performance while reducing skill size.

Legacy ISL-Dual code and results remain in the repository for provenance and ablation; they are no longer the default next experiment.

## Method

Call the method **Causal Minimal Skill (CMS)**.

Given a seed skill package `S`, deterministically decompose text skills into modules `M = {m_1, ..., m_K}`. A module is a top-level reusable procedural section (Markdown H2 by default); YAML frontmatter, skill description, scripts, references, and non-Markdown resources are preserved and are never deleted by the first pilot.

For a selection set of task instances `D_sel`, measure the full-skill score:

`R(S) = mean_{x in D_sel} reward(S, x)`.

For each removable module, construct a knockout skill `S \ {m_j}` and measure:

`Delta_j = R(S) - R(S \ {m_j})`.

Interpretation:

- `Delta_j > tau_keep`: module is causally helpful; keep it.
- `Delta_j < -tau_drop`: removing it improves execution; drop it.
- otherwise: module is empirically neutral. Drop neutral modules only when the reduced candidate does not decrease the selection score.

After the one-at-a-time screen, run a deterministic greedy compression pass. Starting from the full skill, consider modules from least useful to most useful; accept a deletion only when the candidate selection score is at least the current score minus `selection_tolerance`. Stop when no deletion is acceptable or `min_modules` is reached.

Primary objective:

`S* = argmax_{S' subseteq S} [ R_sel(S') - lambda_size * |S'| ]`

implemented through the conservative knockout/greedy approximation above rather than combinatorial exhaustive search.

The held-out instances are never used for pruning decisions.

## Why this is the new primary direction

The completed E2-LS1 result showed the only clear positive mechanism in the current repository: B6 reached 0.7889 while the unpruned `minimal-2` candidate corresponding to the winner had 0.7611 held-out performance. MCTS/static selection did not differentiate B3/B4/B5, so the search machinery is not the part currently supported by evidence. CMS isolates the supported mechanism instead of increasing search complexity.

## Benchmark strategy

### Gate A: SWE-Skills-Bench headroom/interface check

SWE-Skills-Bench is used only as a cheap external sanity gate because its published results already identify a small set of skills with positive treatment effect. It is not the main continual-learning benchmark because the released dataset is one benchmark task per skill (49 rows), with pass rate computed over verifier tests rather than reusable cross-instance learning splits.

Default positive skills:

- `risk-metrics-calculation`
- `gitlab-ci-patterns`
- `tdd-workflow`

The orchestration command runs the upstream benchmark's own paired `--use-skill` / `--no-use-skill` evaluation for these IDs and records the upstream comparison report. This gate asks only whether the local model/runtime can reproduce positive skill headroom.

Gate A passes when at least 2 of the 3 selected skills have positive local skill lift and their aggregate lift is positive. If Gate A fails, do not interpret CMS results from that runtime as evidence about skill quality; fix the execution/injection protocol or stop.

### Gate B: SkillLearnBench causal-pruning pilot

SkillLearnBench is the primary benchmark because it provides 20 skill-dependent tasks with 100 verified instances and accepts arbitrary generated skill directories through `evaluate_skills.py --skill-path`.

Default pilot tasks are chosen for clear reusable workflows and deterministic evaluation:

- `weighted-gdp-calculation` (6 instances)
- `financial-analysis` (6 instances)
- `github-repo-analytics` (5 instances)

The runner must discover the actual number of instances from `tasks/<task>/<task>-N/`; it must not hard-code counts.

For each task:

- selection instances: first `selection_instances=2` instances by numeric suffix;
- held-out instances: all remaining instances;
- seed skill: default `skills/b1-one-shot-claude-sonnet-4-6/<task>/`, with `skills/human_authored/<task>/` supported as an upper-bound diagnostic;
- execution: call the upstream `evaluate_skills.py` using the requested `agent`, `model`, `--skip-metrics`, and generated skill paths;
- no verifier source code is provided to the skill author/pruner. CMS only consumes scalar pass/fail outcomes produced by upstream evaluation.

Gate B passes when CMS beats the unpruned seed on held-out accuracy in at least 2/3 pilot tasks and the aggregate held-out lift is positive. A stronger paper-level target is positive held-out lift with fewer modules/tokens on most tasks.

## Baselines

The pilot reports:

- `B0 no_skill`: upstream no-skill condition.
- `B1 seed`: unmodified one-shot seed skill.
- `B2 random_prune`: same number of modules as CMS, randomly retained with a fixed seed; run only after CMS has produced a smaller skill.
- `B3 causal_minimal (ours)`: CMS knockout + conservative greedy deletion.
- `B4 human_authored`: upstream human skill, diagnostic upper bound.

SkillOpt and SkillRevise are required for a full paper but are not blockers for the first GO/STOP gate. The experiment plan records exact commands for adding their produced skill paths later.

## Skill package model

Create focused units:

- `skill_modules.py`: parse/render skill Markdown while preserving frontmatter and non-removable material.
- `benchmark_adapters.py`: validate external benchmark roots and construct upstream commands; no benchmark source is vendored.
- `causal_pruning.py`: pure pruning logic independent of subprocess execution.
- `causal_runner.py`: orchestration, caching, reports, CLI.

A `SkillModule` has:

- stable `id`
- `title`
- complete markdown `body`
- `removable: bool`
- source relative path.

A `SkillPackage` owns all files under one task's skill directory plus parsed modules. Rendering a subset writes a complete upstream-compatible skill directory, preserving scripts/resources byte-for-byte.

## Evaluation cache

Every external evaluation is expensive. Cache by a stable key containing:

- benchmark root git commit when available;
- task instance IDs;
- agent and model;
- seed package digest;
- retained module IDs;
- repeats;
- max steps.

A completed cache entry stores command, return code, parsed pass counts, raw stdout/stderr tail, and source report paths. Incomplete/failed calls are not treated as zero reward.

## External-run contract

For SkillLearnBench, the adapter invokes upstream `evaluate_skills.py` rather than reimplementing Docker/agent logic. Generated CMS skill configurations live under the ISL-Dual output root and are passed through `--skill-path`.

The adapter must support `--dry-run` and print exact upstream commands without execution.

The runner must fail fast when:

- benchmark root is missing required scripts;
- task directories or seed skills are missing;
- Docker is unavailable for live runs;
- the selected agent's required credentials are missing according to upstream preflight;
- an upstream evaluation exits non-zero;
- expected result artifacts cannot be parsed.

Infrastructure failure is never converted into model reward 0.

## CLI

Add console script:

`isl-causal-skill = isl_dual.causal_runner:main`

Subcommands:

### `preflight`

Validate benchmark layout and selected task/seed availability.

### `swe-headroom`

Run or dry-run the three known-positive SWE-Skills-Bench paired conditions using the upstream scripts.

### `skilllearn-headroom`

Evaluate `none`, seed, and human-authored on all instances of the selected SkillLearnBench tasks before optimization.

### `prune`

Run CMS on selection instances, materialize final skills, then evaluate `none`, seed, random-prune, CMS, and human-authored on held-out instances.

### `pilot`

Canonical one-command sequence: preflight -> SkillLearnBench headroom -> CMS prune -> held-out comparison -> GO/STOP report. SWE headroom remains a separate optional protocol check because it uses a different upstream harness/provider.

## Default pilot command

```bash
isl-causal-skill pilot \
  --skilllearn-root /path/to/SkillLearnBench \
  --output runs/cms-v1-pilot \
  --tasks weighted-gdp-calculation financial-analysis github-repo-analytics \
  --seed-config b1-one-shot-claude-sonnet-4-6 \
  --agent codex \
  --model gpt-5.6-luna \
  --selection-instances 2 \
  --max-workers 3 \
  --skip-metrics
```

The user may substitute an upstream-supported model. `--dry-run` must be run first on a new machine.

## Scientific stop rule

Do not scale to all 20 tasks until the pilot is complete.

STOP this direction if either condition holds after a protocol-valid run:

1. seed/human skills show no positive headroom on the selected SkillLearnBench tasks; or
2. CMS fails to beat the unpruned seed on at least 2/3 tasks and aggregate held-out lift is non-positive.

GO to full evaluation only if CMS produces a reproducible held-out advantage while retaining fewer modules/tokens.

## Legacy compatibility

Do not delete old ISL-Dual CLIs, results, or data structures. Update README/agent handoff so CMS is the recommended next experiment and label the existing `isl-dual-mechanism` campaign as the completed legacy exploratory line.
