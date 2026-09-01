from __future__ import annotations

import argparse
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .experiment import run_family


def _atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(temporary, path)


def discover_families(benchmark_root: Path) -> list[str]:
    families = set()
    for spec_path in (benchmark_root / "benchmark" / "tasks").glob("*/task-spec.yaml"):
        families.add(str(yaml.safe_load(spec_path.read_text())["family_id"]))
    return sorted(families)


def run_campaign(benchmark_root: Path, output: Path, model: str | None = None) -> None:
    state_path = output / "campaign.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "benchmark_root": str(benchmark_root.resolve()), "model": model,
        "completed": [], "attempts": [], "started_at": datetime.now(timezone.utc).isoformat(),
    }
    families = discover_families(benchmark_root)
    if len(families) != 30:
        raise RuntimeError(f"expected 30 SkillEvolBench families, found {len(families)}")
    for family_id in families:
        if family_id in state["completed"]:
            continue
        attempt = {"family": family_id, "started_at": datetime.now(timezone.utc).isoformat(), "status": "running"}
        state["attempts"].append(attempt); _atomic(state_path, state)
        try:
            run_family(benchmark_root, family_id, output / "families" / family_id, model)
        except Exception as error:
            attempt.update(status="failed", finished_at=datetime.now(timezone.utc).isoformat(), error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()[-8000:])
            _atomic(state_path, state)
            continue
        attempt.update(status="completed", finished_at=datetime.now(timezone.utc).isoformat())
        state["completed"].append(family_id); _atomic(state_path, state)
    state["finished_at"] = datetime.now(timezone.utc).isoformat()
    state["status"] = "completed"
    _atomic(state_path, state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all real SkillEvolBench families sequentially with checkpoints")
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model")
    args = parser.parse_args()
    run_campaign(args.benchmark_root, args.output, args.model)


if __name__ == "__main__": main()
