# Causal Minimal Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a directly runnable Causal Minimal Skill (CMS) research path that validates skill headroom, causally prunes modular skill text using selection-instance execution, and evaluates compact skills on held-out SkillLearnBench instances.

**Architecture:** Keep legacy ISL-Dual unchanged. Add a deterministic skill parser/renderer, pure causal-pruning policy, thin adapters around upstream benchmark CLIs, and one orchestration CLI. Upstream Docker/agent/verifier behavior remains authoritative; this repository only creates skill variants, schedules paired evaluations, caches results, and reports GO/STOP gates.

**Tech Stack:** Python 3.10+, stdlib (`argparse`, `csv`, `dataclasses`, `hashlib`, `json`, `pathlib`, `subprocess`, `shutil`), existing PyYAML/TOML dependencies, pytest, external SkillLearnBench and SWE-Skills-Bench repositories.

**Spec:** `docs/superpowers/specs/2026-09-10-causal-minimal-skill-design.md`

## Global Constraints

- Preserve all existing ISL-Dual CLIs, results, and legacy behavior.
- Do not vendor benchmark code or benchmark task data.
- Do not expose verifier source text to the pruning method.
- Selection instances and held-out instances must be disjoint.
- Infrastructure failure must raise/record failure; never map it to reward 0.
- Default SkillLearnBench pilot tasks: `weighted-gdp-calculation`, `financial-analysis`, `github-repo-analytics`.
- Default selection count: first 2 numerically sorted instances.
- Default seed config: `b1-one-shot-claude-sonnet-4-6`.
- Preserve YAML frontmatter, scripts, references, and non-Markdown files when pruning.
- Pilot STOP rule: no scaling unless CMS beats seed on >=2/3 tasks and aggregate held-out lift is positive.

---

### Task 1: Skill package parser and renderer

**Files:**
- Create: `src/isl_dual/skill_modules.py`
- Test: `tests/test_causal_skill_modules.py`

**Interfaces:**
- Produces: `SkillModule`, `SkillMarkdown`, `SkillPackage`, `load_skill_package(path: Path) -> SkillPackage`, `render_skill_package(package, retained_ids, destination) -> Path`.
- Consumes: filesystem only.

- [ ] **Step 1: Write failing tests** covering frontmatter preservation, H2 module extraction, stable module IDs, retention order, and byte-for-byte preservation of non-Markdown resources.
- [ ] **Step 2: Run `python -m pytest tests/test_causal_skill_modules.py -q` and confirm RED** because `isl_dual.skill_modules` does not exist.
- [ ] **Step 3: Implement minimal parser/renderer.** Frontmatter is the initial `--- ... ---` block and is non-removable. Module boundaries start at Markdown H2 (`## `); content before the first H2 is non-removable prefix. Files named `SKILL.md` are modularized; other files are copied unchanged in v1.
- [ ] **Step 4: Run the focused test and full pytest suite; confirm GREEN.**
- [ ] **Step 5: Commit.**

### Task 2: Pure causal pruning policy

**Files:**
- Create: `src/isl_dual/causal_pruning.py`
- Test: `tests/test_causal_pruning.py`

**Interfaces:**
- Consumes: module IDs and callback `score(retained_ids: frozenset[str]) -> float`.
- Produces: `KnockoutEffect`, `PruningResult`, `causal_prune(module_ids, score, *, keep_threshold=0.0, tolerance=0.0, min_modules=1) -> PruningResult`.

- [ ] **Step 1: Write failing tests** for helpful, harmful, neutral modules; conservative greedy removal; deterministic tie order; and `min_modules` protection.
- [ ] **Step 2: Run focused tests and confirm RED.**
- [ ] **Step 3: Implement full-score + leave-one-out effects, then greedy deletion sorted by ascending effect and module ID.** Accept a deletion when candidate score >= current score - tolerance. Return every evaluated subset and final score for auditability.
- [ ] **Step 4: Run focused + full tests; confirm GREEN.**
- [ ] **Step 5: Commit.**

### Task 3: External benchmark adapters

**Files:**
- Create: `src/isl_dual/benchmark_adapters.py`
- Test: `tests/test_benchmark_adapters.py`

**Interfaces:**
- Produces: `SkillLearnBenchAdapter`, `SWESkillsBenchAdapter`, `BenchmarkRun`, `discover_skilllearn_instances(root, task) -> list[str]`.
- SkillLearnBench adapter methods: `preflight(...)`, `evaluate(task_instances, skill_path|None, *, agent, model, max_workers, max_steps, skip_metrics, dry_run) -> BenchmarkRun`.
- SWE adapter method: `paired_headroom(skill_ids, *, dry_run) -> list[list[str]]` and live execution wrapper.

- [ ] **Step 1: Write failing tests** using temporary fake benchmark roots; assert instance numeric sorting, exact command construction, missing-layout failure, dry-run no execution, and report parsing from a compact JSON result cache produced by wrapper tests.
- [ ] **Step 2: Run focused tests and confirm RED.**
- [ ] **Step 3: Implement thin subprocess adapters.** For SkillLearnBench, call its own `evaluate_skills.py` with explicit query instance IDs and `--skill-path`; force isolated `--trials-dir`/output namespace if upstream supports it, otherwise parse stdout summary and upstream report files. For SWE, invoke `run_all_skills.py`/`run_all_skills_eval.py` with `--only` and paired use/no-use-skill conditions.
- [ ] **Step 4: Run focused + full tests; confirm GREEN.**
- [ ] **Step 5: Commit.**

### Task 4: Cached CMS task runner

**Files:**
- Create: `src/isl_dual/causal_runner.py`
- Test: `tests/test_causal_runner.py`

**Interfaces:**
- Produces: `CMSConfig`, `CMSTaskResult`, `run_cms_task(...)`, cache-key helpers, and CLI parser.
- Consumes: `SkillLearnBenchAdapter`, skill module renderer, causal pruning policy.

- [ ] **Step 1: Write failing tests** with a fake adapter that assigns known scores to retained module sets. Verify disjoint split, seed/full evaluation, knockout caching, final held-out evaluation, random-prune size matching, and GO/STOP aggregation.
- [ ] **Step 2: Run focused tests and confirm RED.**
- [ ] **Step 3: Implement stable cache under `<output>/cache/evaluations/`, skill variants under `<output>/skills/<task>/`, and task JSON under `<output>/tasks/<task>/result.json`.** Cache key includes benchmark commit/layout digest when available, task instances, model/agent, seed digest, retained IDs, repeats/max-steps.
- [ ] **Step 4: Implement canonical pilot orchestration for the three default tasks.** Selection uses first two instances; held-out uses the rest. Evaluate B0 no-skill, B1 seed, B2 random-prune, B3 CMS, B4 human-authored.
- [ ] **Step 5: Run focused + full tests; confirm GREEN.**
- [ ] **Step 6: Commit.**

### Task 5: CLI entrypoint, preflight, and experiment handoff

**Files:**
- Modify: `pyproject.toml`
- Create: `docs/CMS_EXPERIMENT.md`
- Modify: `docs/AGENT_START.md`
- Modify: `README.md`
- Test: `tests/test_causal_cli.py`

**Interfaces:**
- Produces console script `isl-causal-skill` with subcommands `preflight`, `swe-headroom`, `skilllearn-headroom`, `prune`, `pilot`.

- [ ] **Step 1: Write failing CLI parser tests** for canonical defaults, `--dry-run`, task overrides, seed path/config, and STOP-gate defaults.
- [ ] **Step 2: Confirm RED.**
- [ ] **Step 3: Register console script and implement CLI.** `pilot --dry-run` must print all planned instance splits, skill paths, variant count upper bound, and exact upstream commands without invoking Docker/model APIs.
- [ ] **Step 4: Document machine setup and exact commands.** Include external clone commands, Docker/API requirements inherited from upstream, dry-run first, live pilot command, output locations, and GO/STOP interpretation.
- [ ] **Step 5: Run focused + full tests; confirm GREEN.**
- [ ] **Step 6: Commit.**

### Task 6: Verification, review, and merge

**Files:**
- No planned production-file changes unless verification exposes defects.

- [ ] **Step 1: Run `python -m pytest -q` in CI.**
- [ ] **Step 2: Run a repository-only dry-run smoke test in CI using synthetic benchmark fixtures; no external API/model call.**
- [ ] **Step 3: Review diff for leakage, accidental deletion of legacy code, unsafe shell construction, and cache-key omissions.**
- [ ] **Step 4: Update PR body with exact method and experiment semantics.**
- [ ] **Step 5: Merge only after all CI checks are green.**
- [ ] **Step 6: Verify main CI after merge and report the final commit plus canonical agent command.**

## Execution decision

The user explicitly requested direct repository implementation after approving the design, so execution proceeds inline in this session on branch `causal-minskill-20260910` rather than waiting for another choice.
