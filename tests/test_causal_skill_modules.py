from __future__ import annotations

from pathlib import Path

from isl_dual.skill_modules import load_skill_package, render_skill_package


def _write_skill(root: Path) -> Path:
    task = root / "task"
    skill = task / "workflow"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        """---
name: workflow
description: Focused workflow skill.
metadata:
  version: \"1.0\"
---

# Workflow

Intro text that must always survive pruning.

## Inspect inputs
Read the inputs and identify constraints.

### Detail
Keep nested detail with its H2 parent.

## Execute procedure
Perform the required operation.

## Verify output
Check the final result against the contract.
"""
    )
    refs = skill / "references"
    refs.mkdir()
    (refs / "formula.txt").write_bytes(b"alpha\x00beta\n")
    return task


def _write_multi_skill(root: Path) -> Path:
    task = root / "task"
    for name, body in (
        ("helpful-skill", "Use the reusable helpful procedure."),
        ("harmful-skill", "Take an unnecessary harmful detour."),
        ("neutral-skill", "Optional neutral context."),
    ):
        skill = task / name
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {name}\n---\n\n# {name}\n\n## Procedure\n{body}\n"
        )
        (skill / "resource.txt").write_text(f"resource:{name}\n")
    return task


def test_load_skill_package_preserves_frontmatter_and_extracts_h2_modules(tmp_path: Path) -> None:
    task = _write_skill(tmp_path)

    package = load_skill_package(task, granularity="section")

    assert package.granularity == "section"
    assert len(package.markdown_files) == 1
    markdown = package.markdown_files[0]
    assert markdown.relative_path == Path("workflow/SKILL.md")
    assert markdown.frontmatter.startswith("---\nname: workflow")
    assert "# Workflow" in markdown.prefix
    assert "Intro text" in markdown.prefix
    assert [module.title for module in markdown.modules] == [
        "Inspect inputs",
        "Execute procedure",
        "Verify output",
    ]
    assert "### Detail" in markdown.modules[0].body
    assert all(module.removable for module in markdown.modules)
    assert len({module.id for module in markdown.modules}) == 3


def test_module_ids_are_stable_for_identical_skill_content(tmp_path: Path) -> None:
    first = _write_skill(tmp_path / "a")
    second = _write_skill(tmp_path / "b")

    ids_a = [module.id for module in load_skill_package(first, granularity="section").modules]
    ids_b = [module.id for module in load_skill_package(second, granularity="section").modules]

    assert ids_a == ids_b


def test_render_skill_package_keeps_selected_modules_in_original_order_and_resources_byte_exact(tmp_path: Path) -> None:
    task = _write_skill(tmp_path / "source")
    package = load_skill_package(task, granularity="section")
    inspect, execute, verify = package.modules
    destination = tmp_path / "rendered"

    render_skill_package(package, {inspect.id, verify.id}, destination)

    rendered = (destination / "workflow" / "SKILL.md").read_text()
    assert rendered.startswith("---\nname: workflow")
    assert "Intro text that must always survive pruning." in rendered
    assert "## Inspect inputs" in rendered
    assert "## Verify output" in rendered
    assert "## Execute procedure" not in rendered
    assert rendered.index("## Inspect inputs") < rendered.index("## Verify output")
    assert (destination / "workflow" / "references" / "formula.txt").read_bytes() == b"alpha\x00beta\n"


def test_default_granularity_treats_each_native_skill_directory_as_one_module(tmp_path: Path) -> None:
    task = _write_multi_skill(tmp_path)

    package = load_skill_package(task)

    assert package.granularity == "skill"
    assert [module.title for module in package.modules] == [
        "harmful-skill",
        "helpful-skill",
        "neutral-skill",
    ]
    assert [module.relative_path for module in package.modules] == [
        Path("harmful-skill/SKILL.md"),
        Path("helpful-skill/SKILL.md"),
        Path("neutral-skill/SKILL.md"),
    ]
    assert all(module.body.startswith("---\nname:") for module in package.modules)


def test_skill_granularity_removes_entire_unretained_skill_directory(tmp_path: Path) -> None:
    task = _write_multi_skill(tmp_path / "source")
    package = load_skill_package(task)
    helpful = next(module for module in package.modules if module.title == "helpful-skill")
    destination = tmp_path / "rendered"

    render_skill_package(package, {helpful.id}, destination)

    assert (destination / "helpful-skill" / "SKILL.md").is_file()
    assert (destination / "helpful-skill" / "resource.txt").read_text() == "resource:helpful-skill\n"
    assert not (destination / "harmful-skill").exists()
    assert not (destination / "neutral-skill").exists()


def test_render_rejects_unknown_module_id(tmp_path: Path) -> None:
    task = _write_skill(tmp_path / "source")
    package = load_skill_package(task, granularity="section")

    try:
        render_skill_package(package, {"does-not-exist"}, tmp_path / "out")
    except ValueError as error:
        assert "unknown module" in str(error).lower()
    else:
        raise AssertionError("unknown module id should fail")
