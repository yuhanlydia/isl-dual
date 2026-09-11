# Agent Start — Current Canonical Experiment

The current primary research path is **Causal Minimal Skill (CMS)**. The earlier `isl-dual-mechanism` SkillEvolBench campaign is complete and is now a legacy exploratory line. Do not start a new 6/30-family MCTS sweep unless explicitly requested.

The detailed method, benchmark rationale, credentials, outputs, and GO/STOP rule are in `docs/CMS_EXPERIMENT.md`.

## 1. Update and verify this repository

```bash
cd /path/to/isl-dual
git pull
python3 -m pip install -e . --no-build-isolation
python3 -m pytest -q
```

The new shell entrypoint must exist:

```bash
isl-causal-skill --help
```

## 2. Main benchmark

Use the official SkillLearnBench repository:

```bash
git clone https://github.com/cxcscmu/SkillLearnBench.git /path/to/SkillLearnBench
```

Follow its installation instructions and make sure Docker works. No local GPU is required; evaluation uses remote model APIs inside the upstream Docker harness.

The canonical pilot uses exactly these tasks:

```text
weighted-gdp-calculation
financial-analysis
github-repo-analytics
```

Each committed one-shot seed currently contains three independently registered native skill directories. CMS treats those native skills—not Markdown headings inside them—as the causal units in v1.

Default seed skill set:

```text
b1-one-shot-claude-sonnet-4-6
```

Default agent/runtime:

```text
claude-code / claude-sonnet-4-6
```

Credentials are inherited from SkillLearnBench. For the default pilot, provide a valid `ANTHROPIC_API_KEY`; `github-repo-analytics` additionally needs `GH_TOKEN`. If running the upstream Codex agent instead, provide the API credential it requires (`OPENAI_API_KEY`). Do not assume a host ChatGPT/Codex login is available inside SkillLearnBench Docker.

## 3. Preflight — no API calls

```bash
isl-causal-skill preflight \
  --skilllearn-root /path/to/SkillLearnBench
```

This checks benchmark layout, query instances, and seed/human skill packages only.

## 4. Mandatory dry run — no API calls

```bash
isl-causal-skill pilot \
  --skilllearn-root /path/to/SkillLearnBench \
  --output runs/cms-v1-pilot \
  --dry-run
```

Confirm that the output explicitly says:

```text
NO MODEL/API CALLS
```

For each task it must show:

- first two numeric instances as selection instances;
- all remaining instances as held-out;
- disjoint selection/held-out sets;
- native seed-skill count;
- maximum selection-evaluation count;
- five held-out conditions: no-skill, seed, random-prune, CMS, human-authored.

For the current canonical seeds, `K=3`, so the maximum selection-set evaluations are `1 + 2K = 7` per task before cache deduplication.

## 5. Run the canonical pilot

```bash
isl-causal-skill pilot \
  --skilllearn-root /path/to/SkillLearnBench \
  --output runs/cms-v1-pilot
```

Do not tune on held-out results. Defaults are fixed:

```text
selection instances = 2
causal module = one native skill directory containing SKILL.md
leave-one-native-skill-out causal knockout
greedy deletion tolerance = 0.0
min retained native skills = 1
repeats = 1
max workers = 3
max steps = 100
primary metrics = upstream pass/fail only
random seed = 20260910
```

A finer Markdown-H2 section mode exists only for future granularity ablation and is not used by the canonical pilot.

The run is resumable. Re-run the exact command and same output namespace after an interruption. Completed benchmark evaluations are content-addressed and reused from `cache/evaluations/`; infrastructure failures are never stored as scientific zero rewards.

## 6. Decision rule

Do **not** scale unless the CLI prints `CMS decision: GO`.

GO requires:

```text
positive skill headroom in >= 2/3 tasks
aggregate skill headroom > 0
CMS > seed on held-out in >= 2/3 tasks
mean(CMS - seed) > 0
CMS retains fewer native skills AND fewer bytes in >= 2/3 tasks
```

If the result is `STOP`, stop this research direction. Do not rescue it by increasing MCTS budget, candidate count, or search depth.

If the result is `GO`, the next paper-level stage is: expand benchmark coverage, add repeated runs/models, compare against SkillOpt/SkillRevise, restore outcome-only seed induction, and analyze native-skill treatment effects. Section-level pruning can be added later as an ablation.

## 7. Optional headroom checks

SkillLearnBench headroom only:

```bash
isl-causal-skill skilllearn-headroom \
  --skilllearn-root /path/to/SkillLearnBench \
  --output runs/cms-v1-headroom
```

Optional known-positive SWE-Skills-Bench protocol gate:

```bash
isl-causal-skill swe-headroom \
  --swe-root /path/to/SWE-Skills-Bench \
  --skills risk-metrics-calculation gitlab-ci-patterns tdd-workflow \
  --dry-run
```

See `docs/CMS_EXPERIMENT.md` before running that optional gate live.

## 8. Legacy experiment provenance

The completed SkillEvolBench v2 results remain under `runs/v2-primary-2fam-luna/`. They should not be deleted. They motivated the CMS pivot: E2-LS1 showed the only clear positive lift after operational pruning, while the MCTS/static selection baselines were flat.

Legacy CLIs (`isl-dual-mechanism`, `isl-dual-family`, etc.) remain available for reproducibility, but are not the current default task.
