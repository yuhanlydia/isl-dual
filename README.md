# ISL-Dual / Causal Minimal Skill

This repository now contains two research lines:

1. **Causal Minimal Skill (CMS)** — the current primary experiment.
2. **ISL-Dual** — the completed exploratory inverse-skill / DAG / MCTS line retained for provenance and ablation.

The latest SkillEvolBench v2 experiment found one clear positive mechanism: the final operationally pruned E2-LS1 skill reached `0.7889`, while its corresponding unpruned candidate was `0.7611`; the static/greedy/MCTS selection baselines were otherwise flat. The next experiment therefore isolates the supported mechanism instead of scaling the unsupported search machinery.

## Current method: Causal Minimal Skill

CMS treats a procedural skill as reusable modules rather than one indivisible prompt. For a seed skill

```text
S = {m1, ..., mK}
```

it measures each module by an execution knockout on **selection instances only**:

```text
Delta_j = R(S) - R(S \ {m_j})
```

Then it greedily removes the least useful modules, accepting a deletion only when the current selection score does not decrease beyond a fixed tolerance. The resulting compact skill is frozen and evaluated on held-out instances that were never used for pruning.

```text
seed SKILL.md
    |
    v
H2 procedural modules
    |
    +--> full skill execution
    +--> leave-one-module-out execution
    |          |
    |          v
    |      causal effects
    |          |
    v          v
conservative greedy pruning
    |
    v
minimal frozen skill
    |
    v
held-out evaluation
```

Primary comparison:

| ID | Condition |
|---|---|
| B0 | no skill |
| B1 | unpruned seed skill |
| B2 | random prune to the same module count as CMS |
| **B3** | **Causal Minimal Skill (ours)** |
| B4 | human-authored skill, diagnostic upper bound |

The first pilot deliberately starts from SkillLearnBench's committed one-shot seed skill. It does **not** claim outcome-only induction yet. This isolates whether causal minimality itself works. If CMS passes the GO gate, outcome-only seed induction is reintroduced as a second-stage experiment.

## Main benchmark: SkillLearnBench

Use the official external repository; it is not vendored here:

```bash
git clone https://github.com/cxcscmu/SkillLearnBench.git /path/to/SkillLearnBench
```

Canonical pilot tasks:

```text
weighted-gdp-calculation
financial-analysis
github-repo-analytics
```

CMS dynamically discovers every query instance. By default:

```text
first 2 numeric instances -> selection / causal pruning
all remaining instances   -> frozen held-out evaluation
```

Default seed:

```text
skills/b1-one-shot-claude-sonnet-4-6/<task>/
```

Default protocol-faithful agent/runtime:

```text
claude-code / claude-sonnet-4-6
```

No local GPU is required. SkillLearnBench uses Docker and remote model APIs. The default pilot requires `ANTHROPIC_API_KEY`; `github-repo-analytics` additionally requires `GH_TOKEN`. A Codex run can be selected explicitly with `--agent codex --model ...` when the upstream-required `OPENAI_API_KEY` is available. A host ChatGPT/Codex login is not assumed to propagate into the benchmark Docker container.

## Install

```bash
python3 -m pip install -e . --no-build-isolation
python3 -m pytest -q
```

The canonical CLI is:

```bash
isl-causal-skill --help
```

## Run the next experiment

Filesystem preflight; no Docker or model call:

```bash
isl-causal-skill preflight \
  --skilllearn-root /path/to/SkillLearnBench
```

Mandatory dry-run; no Docker or model call:

```bash
isl-causal-skill pilot \
  --skilllearn-root /path/to/SkillLearnBench \
  --output runs/cms-v1-pilot \
  --dry-run
```

Live pilot:

```bash
isl-causal-skill pilot \
  --skilllearn-root /path/to/SkillLearnBench \
  --output runs/cms-v1-pilot
```

Safe resume uses the **same command and same output namespace**. External evaluations are content-addressed by benchmark identity, exact task instances, model/agent, skill contents, repeats, and execution budget. Completed calls are reused; infrastructure failures are never cached as scientific zero rewards.

Detailed instructions are in:

- [`docs/AGENT_START.md`](docs/AGENT_START.md) — canonical handoff for an execution agent.
- [`docs/CMS_EXPERIMENT.md`](docs/CMS_EXPERIMENT.md) — method, runtime, credentials, outputs, optional SWE-Skills-Bench gate, and GO/STOP rule.
- [`docs/superpowers/specs/2026-09-10-causal-minimal-skill-design.md`](docs/superpowers/specs/2026-09-10-causal-minimal-skill-design.md) — design specification.

## Fixed CMS pilot settings

| Parameter | Default |
|---|---:|
| Pilot tasks | 3 |
| Selection instances / task | 2 |
| Held-out instances | all remaining |
| Removable unit | Markdown H2 section |
| Knockout | leave-one-module-out |
| Greedy tolerance | 0.0 |
| Minimum modules | 1 |
| Repeats | 1 |
| Max workers | 3 |
| Max agent steps | 100 |
| Primary metric | upstream pass/fail |
| Random seed | 20260910 |

Held-out scores do not feed back into pruning and must not be used to tune these settings.

## GO / STOP

Do not scale to all SkillLearnBench tasks unless the pilot prints:

```text
CMS decision: GO
```

GO requires all of the following:

```text
positive skill headroom in >= 2/3 tasks
aggregate skill headroom > 0
CMS > seed on held-out in >= 2/3 tasks
mean(CMS - seed) > 0
CMS has fewer modules AND fewer bytes than seed in >= 2/3 tasks
```

If the pilot prints `STOP`, stop this research direction. Do not rescue it by increasing MCTS budget, candidate count, or search depth.

If it prints `GO`, the paper-level stage is: broader SkillLearnBench coverage, repeated runs/models, SkillOpt and SkillRevise baselines, outcome-only seed induction, and module-level treatment-effect analyses.

## Optional known-positive skill-interface gate

SWE-Skills-Bench is supported only as an optional protocol/headroom check, not as the main continual-learning benchmark. After cloning its official repository, the selected known-positive skills are:

```text
risk-metrics-calculation
gitlab-ci-patterns
tdd-workflow
```

Dry-run:

```bash
isl-causal-skill swe-headroom \
  --swe-root /path/to/SWE-Skills-Bench \
  --skills risk-metrics-calculation gitlab-ci-patterns tdd-workflow \
  --dry-run
```

See `docs/CMS_EXPERIMENT.md` before running it live.

## CMS implementation

- `skill_modules.py` — parse/render modular `SKILL.md` packages while preserving YAML frontmatter and non-Markdown resources.
- `causal_pruning.py` — deterministic leave-one-module-out effects plus conservative greedy compression.
- `benchmark_adapters.py` — thin wrappers around external SkillLearnBench/SWE-Skills-Bench execution.
- `skilllearn_bridge.py` — calls SkillLearnBench's own `hyper_eval()` while bypassing its unrelated unconditional Anthropic-only top-level CLI guard; live per-agent/per-task key validation is still upstream-authoritative.
- `causal_runner.py` — selection/held-out splitting, content-addressed evaluation cache, random-prune control, held-out evaluation, GO/STOP report, and CLI.

Generated results live under the chosen output namespace, e.g.:

```text
runs/cms-v1-pilot/
  pilot.json
  tasks/<task>/result.json
  skills/<task>/<variant>/...
  cache/evaluations/<digest>.json
  trials/<digest>/...
```

## Legacy ISL-Dual

The original ISL-Dual line learns a portable procedure from successful final artifacts without observing expert trajectories or curated skills. It proposes latent procedural DAGs, scores a static artifact posterior, performs execution search with MCTS, updates the posterior, mutates graphs, prunes the winner, and freezes a `SKILL.md` for deployment.

That code remains available through the legacy CLIs:

```text
isl-dual
isl-dual-family
isl-dual-campaign
isl-dual-status
isl-dual-mechanism
```

The completed v2 evidence is retained under `runs/v2-primary-2fam-luna/`. Do not delete it. It is the empirical provenance for the CMS pivot.

Historical implementation and mechanism notes remain in:

- `docs/SPEC_AUDIT.md`
- `docs/PILOT_STATUS.md`
- `docs/NEXT_EXPERIMENT.md`

Do not start a new six-family or 30-family legacy campaign unless explicitly requested.

## License

MIT
