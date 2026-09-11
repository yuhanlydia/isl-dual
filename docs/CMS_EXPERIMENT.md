# Causal Minimal Skill (CMS) — Canonical Experiment

This document is the canonical experiment plan after the completed exploratory ISL-Dual v2 run. The next scientific question is no longer whether a larger inverse/MCTS framework works in general. It is whether **execution-grounded removal of unnecessary native skills produces a smaller skill set that transfers better**.

## 1. Method

SkillLearnBench already represents a task's procedural knowledge as a directory containing multiple independently registered native skills. CMS therefore uses each immediate native skill directory containing a `SKILL.md` as one causal module:

```text
S = {s_1, ..., s_K}
```

For example, the committed `weighted-gdp-calculation` one-shot seed contains three native skills: `excel-index-match`, `excel-openpyxl-formulas`, and `excel-weighted-mean`.

CMS measures each native skill by execution knockout on selection instances:

```text
Delta_j = R(S) - R(S \ {s_j})
```

Then it greedily removes the least useful skills, accepting a deletion only when the current selection score does not decrease beyond the configured tolerance. Removal deletes the whole native skill directory, including its `SKILL.md` and bundled resources. Retained directories are copied byte-for-byte. The final reduced skill set is frozen and evaluated only on held-out instances.

Selection and held-out instances are disjoint. Held-out results never feed back into pruning.

A finer Markdown-H2 section parser is implemented only for a future granularity ablation; it is **not** the default v1 experiment.

Primary comparison:

```text
B0  no_skill
B1  unpruned seed skill set
B2  random prune to the same number of native skills as CMS
B3  causal minimal skill set (CMS, ours)
B4  human-authored skill set (diagnostic upper bound)
```

The first pilot starts from SkillLearnBench's committed one-shot seed skill set rather than claiming outcome-only induction. This intentionally isolates the pruning mechanism that was positive in E2-LS1. If CMS passes the GO gate, outcome-only seed induction can be reintroduced as the next experiment without confounding whether pruning itself works.

## 2. Main benchmark: SkillLearnBench

Clone the official repository:

```bash
git clone https://github.com/cxcscmu/SkillLearnBench.git /path/to/SkillLearnBench
```

Use the repository's own installation instructions and Docker runtime. CMS does not vendor or reimplement its verifier.

Default pilot tasks:

```text
weighted-gdp-calculation
financial-analysis
github-repo-analytics
```

All three committed one-shot seed configurations currently contain three native subskills, making the first causal screen small and directly interpretable.

For each task, CMS discovers the available query instances dynamically. The first two numeric instances are selection instances; every remaining instance is frozen held-out evaluation.

Default seed:

```text
skills/b1-one-shot-claude-sonnet-4-6/<task>/
```

Human diagnostic:

```text
skills/human_authored/<task>/
```

## 3. Runtime and credentials

No local GPU is required. The benchmark uses Docker and remote model APIs.

The protocol-faithful default is SkillLearnBench's native evaluation setup:

```text
agent = claude-code
model = claude-sonnet-4-6
```

Set the required credentials in `/path/to/SkillLearnBench/.env` or export them in the shell. The upstream `.env.example` requires:

```bash
ANTHROPIC_API_KEY=<real key>
OPENAI_API_KEY=<real key>   # only needed when using Codex/OpenAI or LLM judge metrics
GH_TOKEN=<real token>       # required by github-repo-analytics
```

For the default pilot with `--skip-metrics` (the default), `OPENAI_API_KEY` is not needed unless the selected agent is Codex.

Important: a local ChatGPT/Codex CLI login is **not assumed** to authenticate SkillLearnBench's Docker agent. To run Codex through the upstream benchmark, provide the API credential required by its `codex` agent configuration:

```bash
isl-causal-skill pilot ... --agent codex --model <upstream-supported-model>
```

The CMS bridge bypasses one upstream CLI bug that unconditionally checks an Anthropic key before parsing `--agent`, but live execution still calls SkillLearnBench's own Docker check and its own per-agent/per-task API-key validation.

## 4. Update ISL-Dual and verify

```bash
cd /path/to/isl-dual
git pull
python3 -m pip install -e . --no-build-isolation
python3 -m pytest -q
```

Then validate only filesystem/layout assumptions; this does not call Docker or a model API:

```bash
isl-causal-skill preflight \
  --skilllearn-root /path/to/SkillLearnBench
```

## 5. Mandatory dry run

Before any live model calls:

```bash
isl-causal-skill pilot \
  --skilllearn-root /path/to/SkillLearnBench \
  --output runs/cms-v1-pilot \
  --dry-run
```

The dry run prints, for each task:

- exact selection instances;
- exact held-out instances;
- seed/human skill locations;
- native skill count;
- maximum number of selection skill-set evaluations (`1 + 2K` upper bound for full + one-at-a-time knockouts + greedy pass);
- five held-out conditions;
- example upstream SkillLearnBench commands.

It must print `NO MODEL/API CALLS` and must not create a scientific `pilot.json`.

## 6. Live pilot

After preflight and dry-run succeed:

```bash
isl-causal-skill pilot \
  --skilllearn-root /path/to/SkillLearnBench \
  --output runs/cms-v1-pilot
```

The default live configuration is:

```text
selection_instances = 2
causal module = one native skill directory containing SKILL.md
causal knockout = leave-one-native-skill-out
selection tolerance = 0.0
minimum retained native skills = 1
repeats = 1
max_workers = 3
max_steps = 100
metrics = skipped for the primary pass/fail gate
random seed = 20260910
```

Do not tune these settings on held-out results.

Because the three canonical seeds currently have `K=3`, the v1 screen needs at most seven distinct selection skill-set evaluations per task before cache reuse (`1 + 3` knockouts + at most `3` greedy candidates), followed by five held-out conditions. Exact executed subsets can be fewer because the content-addressed cache deduplicates repeated variants.

Safe resume: rerun the exact same command and output namespace. Every upstream evaluation is cached by benchmark identity, task instances, model/agent, seed/variant contents, repeats, and step budget. A completed cached evaluation is reused. Failed infrastructure calls are not cached as reward zero.

Outputs:

```text
runs/cms-v1-pilot/
  pilot.json
  tasks/<task>/result.json
  skills/<task>/<variant>/<task>/<retained-native-skill>/...
  cache/evaluations/<digest>.json
  trials/<digest>/...
```

## 7. GO / STOP rule

The CLI prints an explicit decision after all three tasks finish.

### Headroom gate

At least 2 of 3 tasks must show positive skill headroom:

```text
max(seed, human_authored) > no_skill
```

and aggregate headroom lift must be positive.

If this fails under a protocol-valid run, stop this benchmark/runtime path instead of tuning CMS.

### CMS method gate

CMS must satisfy all three:

```text
CMS > seed on held-out performance in >= 2/3 tasks
mean(CMS - seed) > 0 across the pilot
CMS retains fewer native skills and fewer bytes than seed in >= 2/3 tasks
```

If these fail, **STOP this research direction**. Do not expand to all 20 tasks and do not add MCTS/search complexity.

If they pass, **GO** to a full paper-level evaluation: all suitable SkillLearnBench tasks, multiple seeds/repeats/models, SkillOpt and SkillRevise baselines, outcome-only seed induction, and native-skill treatment-effect analyses. Markdown-section granularity can then be added as an ablation.

## 8. Cheap headroom-only check

To test only whether skills matter for this runtime before running knockouts:

```bash
isl-causal-skill skilllearn-headroom \
  --skilllearn-root /path/to/SkillLearnBench \
  --output runs/cms-v1-headroom
```

This evaluates no-skill, committed one-shot seed, and human-authored skills on all instances of the three pilot tasks.

## 9. Optional external protocol gate: SWE-Skills-Bench

SWE-Skills-Bench is optional and is **not** the main CMS learning benchmark. It is useful because prior work identified known-positive skills and therefore provides a quick skill-interface sanity check.

Clone the upstream repository:

```bash
git clone https://github.com/GeniusHTX/SWE-Skills-Bench.git /path/to/SWE-Skills-Bench
```

Dry-run the three known-positive families selected for the gate:

```bash
isl-causal-skill swe-headroom \
  --swe-root /path/to/SWE-Skills-Bench \
  --skills risk-metrics-calculation gitlab-ci-patterns tdd-workflow \
  --dry-run
```

Then run live with the credentials/runtime required by that upstream repository:

```bash
isl-causal-skill swe-headroom \
  --swe-root /path/to/SWE-Skills-Bench \
  --skills risk-metrics-calculation gitlab-ci-patterns tdd-workflow
```

CMS does not parse this optional gate into the main GO/STOP score. Use the upstream pass-rate comparison report for the treatment-effect check.

## 10. What not to run now

Do not launch a six-family or 30-family legacy ISL-Dual sweep. Do not increase MCTS budget. Do not add more candidate DAGs.

The completed v2 experiment remains useful evidence that motivated this pivot, but the next experiment is CMS on SkillLearnBench. The legacy code remains available for reproducibility and future ablation only.
