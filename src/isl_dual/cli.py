from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import train_inverse_skill
from .toy import ToyCritic, ToyExecutor, ToyMutator, ToyProposer, toy_tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="ISL-Dual pilot runner")
    parser.add_argument("command", choices=["smoke"])
    parser.add_argument("--output", type=Path, default=Path("runs/smoke"))
    args = parser.parse_args()
    result = train_inverse_skill(toy_tasks(), ToyProposer(), ToyCritic(), ToyExecutor(), ToyMutator())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "SKILL.md").write_text(result.skill)
    (args.output / "result.json").write_text(json.dumps({
        "winner": result.graph.id,
        "q0": result.q0,
        "q1": result.q1,
        "q2": result.posterior,
        "forward_scores": result.forward_scores,
    }, indent=2, sort_keys=True))
    print(f"winner={result.graph.id}")
    print(f"skill={args.output / 'SKILL.md'}")


if __name__ == "__main__":
    main()

