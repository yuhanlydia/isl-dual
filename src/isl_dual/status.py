from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def summarize(run_root: Path) -> dict[str, object]:
    supervisor_path, campaign_path = run_root / "supervisor.json", run_root / "campaign.json"
    supervisor = json.loads(supervisor_path.read_text()) if supervisor_path.exists() else {}
    campaign = json.loads(campaign_path.read_text()) if campaign_path.exists() else {}
    attempts = campaign.get("attempts", [])
    cache_counts = Counter(path.parent.name for path in run_root.glob("**/cache/*/*.json"))
    return {
        "supervisor_status": supervisor.get("status", "missing"),
        "deadline_at": supervisor.get("deadline_at"),
        "campaign_status": campaign.get("status", "primary"),
        "completed_families": len(campaign.get("completed", [])),
        "attempt_statuses": dict(Counter(str(attempt.get("status", "unknown")) for attempt in attempts)),
        "current_units": [{"family": attempt.get("family"), "replication": attempt.get("replication")} for attempt in attempts if attempt.get("status") == "running"],
        "cache_records": dict(sorted(cache_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Show a compact ISL-Dual long-run status")
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.run_root), indent=2, sort_keys=True))


if __name__ == "__main__": main()
