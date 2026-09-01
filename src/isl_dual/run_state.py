from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class RunState:
    run_id: str
    stage: str = "created"
    completed_units: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    updated_at: str = ""

    @classmethod
    def load(cls, path: Path, run_id: str) -> "RunState":
        return cls(**json.loads(path.read_text())) if path.exists() else cls(run_id=run_id)

    def save(self, path: Path) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))
        os.replace(temporary, path)
