# Causal Minimal Skill Design

## Goal

Pivot the primary research direction from full ISL-Dual search to the empirically supported mechanism observed in E2-LS1: **operational pruning of procedural knowledge**. The method asks whether an over-complete native skill set can be reduced to a smaller subset whose members have execution-grounded causal value and whose frozen held-out transfer is better.

Primary claim under test:

> A reusable skill set should contain only native procedural skills that causally improve execution. Removing neutral or harmful skills can improve held-out performance while reducing procedural complexity.

Legacy ISL-Dual code/results stay in the repository for provenance and future ablation; they are not the default next experiment.

## Method: Causal Minimal Skill (CMS)

Given a seed SkillLearnBench task skill set:

```text
<task>/
  skill-a/SKILL.md
  skill-b/SKILL.md
  skill-c/SKILL.md
```

CMS treats each independently registered native skill directory as one causal unit:

```text
S = {s_1, ..., s_K}
```

The v1 pilot does **not** flatten or rewrite the retained skills. Removing `s_j` deletes its whole directory; retaining it preserves its `SKILL.md` and bundled resources byte-for-byte.

For selection instances `D_sel`:

```text
R(S) = mean_{x in D_sel} reward(S, x)
Delta_j = R(S) - R(S \ {s_j})
```

A positive `Delta_j` means that removing the skill hurts execution; a negative value means its removal improves execution. Neutral skills may be removed only if the current reduced set keeps the selection score within a fixed tolerance.

After the leave-one-skill-out screen, CMS runs a deterministic conservative greedy pass. Starting from the full set, it considers skills from least useful to most useful and accepts a deletion only when:

```text
R(current \ {s_j}) >= R(current) - selection_tolerance
```

and `min_modules` is respected.

Held-out instances never enter these decisions.

The implementation also supports Markdown-H2 section parsing for a future granularity ablation, but **native skill-directory granularity is the v1 default and the paper's first causal unit**.

## Why this is the supported branch

The completed E2-LS1 exploratory result reached `B6=0.7889`, while the corresponding unpruned `minimal-2` candidate was `0.7611`. Static/greedy/MCTS selectors did not separate B3/B4/B5. The positive evidence therefore points to pruning/compact procedural guidance rather than larger search. CMS isolates that mechanism.

## Benchmarks

### Main: SkillLearnBench

SkillLearnBench is the primary CMS benchmark because it provides skill-dependent tasks with multiple verified instances and accepts arbitrary skill-set directories through its upstream evaluator.

Canonical pilot:

```text
weighted-gdp-calculation
financial-analysis
github-repo-analytics
```

The committed `b1-one-shot-claude-sonnet-4-6` seed for each of these tasks currently contains exactly three native subskills, so the first screen is small and interpretable.

Per task:

- dynamically discover `tasks/<task>/<task>-N/`;
- first 2 numeric instances: selection;
- all remaining instances: held-out;
- seed: `skills/b1-one-shot-claude-sonnet-4-6/<task>/`;
- human diagnostic: `skills/human_authored/<task>/`;
- evaluation: upstream SkillLearnBench Docker/agent/verifier through `hyper_eval()`;
- CMS only consumes scalar upstream pass/fail outcomes; verifier source is never provided to pruning.

### Optional: SWE-Skills-Bench headroom gate

SWE-Skills-Bench is not the main continual-learning benchmark because its released rows are skill-level tasks rather than repeated learning instances. It is supported only as a known-positive skill-interface sanity check on:

```text
risk-metrics-calculation
gitlab-ci-patterns
tdd-workflow
```

using the upstream paired skill/no-skill commands.

## Baselines

The first SkillLearnBench pilot reports:

- `B0 no_skill` — upstream no-skill condition.
- `B1 seed` — committed unpruned one-shot native skill set.
- `B2 random_prune` — same number of retained native skills as CMS, fixed random seed.
- `B3 causal_minimal` — CMS knockout + conservative greedy deletion.
- `B4 human_authored` — upstream human-authored skill set, diagnostic upper bound.

SkillOpt and SkillRevise are required for a full paper if the pilot passes, but are not blockers for the initial GO/STOP experiment.

The first pilot intentionally starts from an existing one-shot seed and therefore does **not** claim outcome-only induction. If CMS succeeds, the next stage restores outcome-only seed induction as a separate experimental factor.

## Components

- `skill_modules.py` — native skill-directory parser/renderer; optional section granularity.
- `causal_pruning.py` — pure leave-one-out treatment effects and conservative greedy reduction.
- `benchmark_adapters.py` — thin external benchmark command/result adapters.
- `skilllearn_bridge.py` — calls upstream SkillLearnBench `hyper_eval()` while bypassing only its unrelated unconditional Anthropic top-level CLI guard.
- `causal_runner.py` — split discipline, content-addressed evaluation cache, controls, report, and CLI.

## Evaluation cache

Every external execution is cached using a stable key containing:

- benchmark git commit (or fallback evaluator digest);
- exact query instance IDs;
- agent/model;
- full rendered skill-set content digest;
- repeats;
- max workers/steps;
- metric mode.

A completed cache record stores command, pass count, total count, scalar score, and stdout/stderr. Non-zero infrastructure failures raise and are never converted to reward zero.

## Runtime contract

No local GPU is required. The main benchmark runs Docker agents against remote APIs.

Protocol-faithful default:

```text
agent = claude-code
model = claude-sonnet-4-6
```

The canonical default requires `ANTHROPIC_API_KEY`; `github-repo-analytics` also requires `GH_TOKEN`. Codex can be selected explicitly when `OPENAI_API_KEY` is available. Host ChatGPT/Codex login state is not assumed to propagate inside benchmark Docker.

SkillLearnBench's current top-level CLI checks an Anthropic key before parsing the selected agent. CMS therefore imports the upstream evaluator and calls `hyper_eval()` through a thin bridge. Live bridge execution still invokes upstream Docker readiness, per-agent/per-task credential validation, skill-path validation, agent execution, and verifier logic.

## CLI

Console script:

```text
isl-causal-skill = isl_dual.causal_runner:main
```

Subcommands:

- `preflight` — file/layout/skill checks only; no Docker or API.
- `pilot` — canonical three-task CMS GO/STOP experiment; supports mandatory `--dry-run`.
- `prune` — same CMS engine for an explicit task list.
- `skilllearn-headroom` — no-skill vs seed vs human on all instances.
- `swe-headroom` — optional known-positive SWE-Skills-Bench paired protocol gate.

Canonical sequence:

```bash
isl-causal-skill preflight \
  --skilllearn-root /path/to/SkillLearnBench

isl-causal-skill pilot \
  --skilllearn-root /path/to/SkillLearnBench \
  --output runs/cms-v1-pilot \
  --dry-run

isl-causal-skill pilot \
  --skilllearn-root /path/to/SkillLearnBench \
  --output runs/cms-v1-pilot
```

## Scientific stop rule

Do not scale beyond the three-task pilot until it finishes.

Headroom must satisfy:

```text
max(seed, human) > no_skill in >= 2/3 tasks
mean headroom lift > 0
```

CMS must satisfy:

```text
CMS > seed on held-out in >= 2/3 tasks
mean(CMS - seed) > 0
CMS retains fewer native skills AND fewer bytes in >= 2/3 tasks
```

If either gate fails after a protocol-valid run, **STOP this direction**. Do not rescue it by increasing MCTS budget or adding search complexity.

If both pass, expand to paper-level evaluation: broader SkillLearnBench coverage, repeated runs/models, SkillOpt/SkillRevise, outcome-only seed induction, and native-skill treatment-effect analysis. Section-level pruning is then a granularity ablation.

## Legacy compatibility

Do not delete the legacy ISL-Dual CLIs, code, or `runs/v2-primary-2fam-luna` evidence. They provide the provenance for this pivot but are no longer the canonical experiment.
