# Agent Start — Current ISL-Dual Experiment

This file is the canonical handoff for the current experiment. Follow it instead of old exploratory namespaces or older README examples.

## 1. Update and verify code

```bash
git pull
python3 -m pip install -e . --no-build-isolation
python3 -m pytest -q
```

Do not reuse the obsolete exploratory namespaces `runs/skillevol-10h` or `runs/v1-corrected` for scientific claims.

### If `v2-primary-2fam` is already running

The performance patch is not hot-loaded into an existing Python process. Stop the current process cleanly, preserve its output directory, pull the new code, and restart **the same command with the same model and the same output namespace**. Do not delete `cache/`, `evidence.json`, artifacts, or partial run state.

The deterministic per-tree seeds and occurrence-aware executor cache let the restarted process replay already completed work from cache until it reaches the previous frontier, then continue with new remote calls. Existing scientific evidence is therefore retained rather than discarded.

In particular, if the current server is using `runs/v2-primary-2fam-luna`, keep that exact output path and the same model identity when resuming. Do not silently switch it to the example `gpt-5.4` configuration mid-run.

## 2. Required external environment

The current primary experiment uses remote Codex inference and requires no local GPU.

Before running:

```bash
codex login status
```

Codex must already be authenticated.

SkillEvolBench is an external repository. Use the official benchmark and pin it to:

```text
9e3daa339987c3cfa624121e1be442593a53d43c
```

Example:

```bash
git -C /path/to/SkillEvolBench fetch --all
git -C /path/to/SkillEvolBench checkout 9e3daa339987c3cfa624121e1be442593a53d43c
```

Keep at least 30–50 GB of free SSD space and put temporary/package-cache data on the large disk when possible:

```bash
export TMPDIR=/path/to/large-disk/tmp
export ISL_DUAL_DEPENDENCY_CACHE=/path/to/large-disk/isl-dual-dependency-cache
mkdir -p "$TMPDIR" "$ISL_DUAL_DEPENDENCY_CACHE"
```

Every rollout still owns a fresh task workspace. The shared dependency cache contains only package-download state; task files and agent-produced artifacts are never shared across rollouts.

## 3. Performance behavior after the 2026-09-09 patch

The scientific configuration is unchanged: `K=8`, MCTS budget `8`, two outer loops, identical seeds, verifier semantics, posterior equations, and fresh workspaces.

The runtime changes are orchestration-only:

- default `forward_workers=3`;
- for one acquisition task, up to three independent graph-MCTS trees run concurrently;
- the eight UCT rollouts **inside each MCTS tree remain strictly sequential**;
- different acquisition tasks run in separate waves so incompatible dependency environments cannot race;
- identical Python requirements are installed once per runner process;
- npm uses a shared isolated download cache with rollout-local `node_modules`;
- B4 greedy graph evaluations use the same safe graph-level concurrency;
- transient Codex `429`/timeout/network failures are retried on fresh workspaces;
- if infrastructure remains unavailable after retries, the run stops/checkpoints instead of recording a false scientific reward of zero;
- cache and evidence-journal writes are thread-safe.

This should reduce the dominant forward-loop wall clock toward roughly one third of the former serial implementation when Codex service concurrency is available, with additional savings from dependency reuse. Actual speed depends on remote Codex latency/rate limits and task dependency installation cost.

## 4. Run Stage 1 only

For a new run, start with exactly two families and a fresh namespace:

```bash
isl-dual-mechanism \
  --benchmark-root /path/to/SkillEvolBench \
  --output runs/v2-primary-2fam \
  --model gpt-5.4 \
  --families E1-LS1 E2-LS1
```

For an existing v2 run, resume its **original** command/output/model instead of starting this example namespace.

Do **not** add `--diagnostics` yet.

Fixed scientific settings are deliberately not tuned on T4–T6:

- candidate DAGs `K=8`
- acquisition/deployment tasks `3/3`
- MCTS budget `8`
- outer loops `2`
- `beta_artifact=2`
- `beta_forward=4`
- mutation probability `mu=0.3`
- maximum graph pool `12`

The primary run must finish frozen T4–T6 deployment and report B0–B8 before any expensive diagnostic sweep.

## 5. Scientific decision gate

The primary hypotheses are:

```text
B1 > B0
max(B4, B5) > B3
B6 > B1
B6 > B3
```

Evolution is specifically supported if:

```text
B6 > B5
```

The key mechanism diagnostic is whether candidate forward-execution scores rank held-out deployment performance better than static artifact scores:

```text
Spearman(forward, deployment) > Spearman(static, deployment)
```

If both families are flat around `B3 ~= B5 ~= B6`, do not increase rollout budget or scale to 30 families. Inspect the procedural representation and forward signal first.

## 6. Only after Stage 1 has signal

Run the six-environment primary pilot:

```bash
isl-dual-mechanism \
  --benchmark-root /path/to/SkillEvolBench \
  --output runs/v2-primary-6fam \
  --model gpt-5.4
```

Only after the six-family primary result has signal, reuse that same namespace for diagnostics:

```bash
isl-dual-mechanism \
  --benchmark-root /path/to/SkillEvolBench \
  --output runs/v2-primary-6fam \
  --model gpt-5.4 \
  --diagnostics
```

The detailed rationale, metrics, 30-family scaling rule, and optional 16 GB local-model reproducibility plan are in `docs/NEXT_EXPERIMENT.md`.
