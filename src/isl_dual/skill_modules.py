from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


Granularity = Literal["section", "skill"]


@dataclass(frozen=True)
class SkillModule:
    id: str
    title: str
    body: str
    removable: bool
    relative_path: Path


@dataclass(frozen=True)
class SkillMarkdown:
    relative_path: Path
    frontmatter: str
    prefix: str
    modules: tuple[SkillModule, ...]


@dataclass(frozen=True)
class SkillPackage:
    root: Path
    markdown_files: tuple[SkillMarkdown, ...]
    granularity: Granularity = "section"

    @property
    def modules(self) -> tuple[SkillModule, ...]:
        return tuple(module for markdown in self.markdown_files for module in markdown.modules)


def _split_frontmatter(text: str) -> tuple[str, str]:
    if not text.startswith("---\n"):
        return "", text
    lines = text.splitlines(keepends=True)
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return "".join(lines[: index + 1]), "".join(lines[index + 1 :])
    return "", text


def _module_id(relative_path: Path, title: str, body: str) -> str:
    payload = "\n".join((relative_path.as_posix(), title.strip(), body.strip()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _parse_markdown_sections(relative_path: Path, text: str) -> SkillMarkdown:
    frontmatter, remainder = _split_frontmatter(text)
    lines = remainder.splitlines(keepends=True)
    starts = [index for index, line in enumerate(lines) if line.startswith("## ")]
    if not starts:
        return SkillMarkdown(relative_path, frontmatter, remainder, ())

    prefix = "".join(lines[: starts[0]])
    modules: list[SkillModule] = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        body = "".join(lines[start:end])
        title = lines[start][3:].strip()
        modules.append(
            SkillModule(
                id=_module_id(relative_path, title, body),
                title=title,
                body=body,
                removable=True,
                relative_path=relative_path,
            )
        )
    return SkillMarkdown(relative_path, frontmatter, prefix, tuple(modules))


def _parse_native_skill(relative_path: Path, text: str) -> SkillMarkdown:
    frontmatter, remainder = _split_frontmatter(text)
    title = relative_path.parent.name or relative_path.stem
    module = SkillModule(
        id=_module_id(relative_path, title, text),
        title=title,
        body=text,
        removable=True,
        relative_path=relative_path,
    )
    return SkillMarkdown(relative_path, frontmatter, remainder, (module,))


def load_skill_package(
    root: Path,
    *,
    granularity: Granularity = "section",
) -> SkillPackage:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"skill package root does not exist: {root}")
    if granularity not in {"section", "skill"}:
        raise ValueError(f"unsupported skill granularity: {granularity}")

    parser = _parse_markdown_sections if granularity == "section" else _parse_native_skill
    markdown_files = tuple(
        parser(path.relative_to(root), path.read_text())
        for path in sorted(root.rglob("SKILL.md"))
        if path.is_file()
    )
    if not markdown_files:
        raise ValueError(f"skill package contains no SKILL.md: {root}")
    return SkillPackage(root=root, markdown_files=markdown_files, granularity=granularity)


def render_skill_package(
    package: SkillPackage,
    retained_ids: set[str] | frozenset[str],
    destination: Path,
) -> Path:
    retained = set(retained_ids)
    known = {module.id for module in package.modules}
    unknown = retained - known
    if unknown:
        raise ValueError(f"unknown module id(s): {sorted(unknown)}")

    destination = Path(destination)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(package.root, destination)

    if package.granularity == "skill":
        for module in package.modules:
            if module.id in retained:
                continue
            skill_dir = destination / module.relative_path.parent
            if skill_dir == destination:
                raise ValueError("skill-directory pruning requires SKILL.md below the task root")
            if skill_dir.exists():
                shutil.rmtree(skill_dir)
        return destination

    for markdown in package.markdown_files:
        body = markdown.frontmatter + markdown.prefix
        body += "".join(module.body for module in markdown.modules if module.id in retained)
        target = destination / markdown.relative_path
        target.write_text(body)
    return destination
