"""Loading finished sessions (steps.jsonl + metrics.json + config_used.json)."""

from __future__ import annotations

import json
from pathlib import Path


def load_session(session_dir: str | Path) -> dict:
    d = Path(session_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"session directory not found: {d}")
    steps = []
    with open(d / "steps.jsonl", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                steps.append(json.loads(line))
    metrics = json.loads((d / "metrics.json").read_text(encoding="utf-8")) if (d / "metrics.json").exists() else {}
    config = json.loads((d / "config_used.json").read_text(encoding="utf-8")) if (d / "config_used.json").exists() else {}
    return {"dir": d, "steps": steps, "metrics": metrics, "config": config}
