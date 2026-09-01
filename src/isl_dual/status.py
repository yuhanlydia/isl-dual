from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def summarize(run_root: Path) -> dict[str, object]:
    supervisor_path, campaign_path = run_root / "supervisor.json", run_root / "campaign.json"
    supervisor = json.loads(supervisor_path.read_text()) if supervisor_path.exists() else {}
    campaign = json.loads(campaign_path.read_text()) if campaign_path.exists() else {}
    attempts = campaign.get("attempts", [])
    cache_files = list(run_root.glob("**/cache/*/*.json"))
    cache_counts = Counter(path.parent.name for path in cache_files)
    latest_cache_activity = {}
    for component in cache_counts:
        modified = max(path.stat().st_mtime for path in cache_files if path.parent.name == component)
        latest_cache_activity[component] = datetime.fromtimestamp(modified, timezone.utc).isoformat()
    campaign_pid = supervisor.get("pid")
    campaign_live = isinstance(campaign_pid, int) and Path(f"/proc/{campaign_pid}").exists()
    watchdog_pid = None
    if campaign_live:
        try:
            status_lines = Path(f"/proc/{campaign_pid}/status").read_text().splitlines()
            process_status = dict(line.split(":", 1) for line in status_lines if ":" in line)
            candidate = int(process_status["PPid"].strip())
            if candidate > 1 and Path(f"/proc/{candidate}").exists():
                watchdog_pid = candidate
        except (OSError, KeyError, ValueError):
            pass
    verified_artifacts = 0
    for path in run_root.glob("**/artifacts/*/native-verification.json"):
        try:
            verified_artifacts += json.loads(path.read_text()).get("all_passed") is True
        except (OSError, json.JSONDecodeError):
            pass
    return {
        "supervisor_status": supervisor.get("status", "missing"),
        "campaign_pid": campaign_pid,
        "campaign_process_live": campaign_live,
        "watchdog_pid": watchdog_pid,
        "watchdog_process_live": watchdog_pid is not None,
        "deadline_at": supervisor.get("deadline_at"),
        "campaign_status": campaign.get("status", "primary"),
        "completed_families": len(campaign.get("completed", [])),
        "attempt_statuses": dict(Counter(str(attempt.get("status", "unknown")) for attempt in attempts)),
        "current_units": [{"family": attempt.get("family"), "replication": attempt.get("replication")} for attempt in attempts if attempt.get("status") == "running"],
        "cache_records": dict(sorted(cache_counts.items())),
        "latest_cache_activity": dict(sorted(latest_cache_activity.items())),
        "verified_artifact_families": verified_artifacts,
        "completed_result_files": sum(1 for _ in run_root.glob("**/result.json")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Show a compact ISL-Dual long-run status")
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.run_root), indent=2, sort_keys=True))


if __name__ == "__main__": main()
