# Agent Start — Current ISL-Dual Experiment

This file is the canonical handoff for the next run. Follow it instead of old exploratory namespaces or older README examples.

## 1. Update and verify code

```bash
git pull
python3 -m pip install -e . --no-build-isolation
python3 -m pytest -q
```

The current repaired main commit is expected to descend from:

```text
19eaa3d66ab9316f893f5e39f1a7fa3c8ea2e4f1
```

Do not reuse results or caches from `runs/skillevol-10h` or `runs/v1-corrected`.

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

Keep at least 30–50 GB of free SSD space and preferably point `TMPDIR` to the large disk. The repaired executor excludes generated dependency trees from artifacts and uses ephemeral package caches, but every rollout still owns a fresh workspace.

## 3. Run Stage 1 only

Start with exactly two families and a fresh namespace:

```bash
isl-dual-mechanism \
  --benchmark-root /path/to/SkillEvolBench \
  --output runs/v2-primary-2fam \
  --model gpt-5.4 \
  --families E1-LS1 E2-LS1
```

Do **not** add `--diagnostics` yet.

Fixed settings are deliberately not tuned on T4–T6:

- candidate DAGs `K=8`
- acquisition/deployment tasks `3/3`
- MCTS budget `8`
- outer loops `2`
- `beta_artifact=2`
- `beta_forward=4`
- mutation probability `mu=0.3`
- maximum graph pool `12`

The primary run must finish frozen T4–T6 deployment and report B0–B8 before any expensive diagnostic sweep.

## 4. Scientific decision gate

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

## 5. Only after Stage 1 has signal

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
