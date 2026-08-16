"""Lightweight monitoring: latency tracking, cost tracking, and benchmark regression detection."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class RunLog:
    """Append-only JSON-lines log of pipeline runs for regression detection."""

    def __init__(self, log_path: str = "outputs/run_log.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, metrics: dict[str, Any], run_version: str) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_version": run_version,
            **metrics,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        logger.info("Appended run log entry (version %s)", run_version)

    def load_history(self) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        entries = []
        for line in self.log_path.read_text().strip().split("\n"):
            if line:
                entries.append(json.loads(line))
        return entries

    def check_regression(
        self,
        current_metrics: dict[str, Any],
        baseline_n: int = 3,
        latency_threshold: float = 1.5,
        cost_threshold: float = 2.0,
    ) -> dict[str, Any]:
        """Compare current metrics against the last N runs.

        Flags a regression if average latency or cost exceeds the historical
        average by more than the given thresholds (as a multiplier).
        """
        history = self.load_history()
        if len(history) < baseline_n:
            return {"checked": False, "reason": f"fewer than {baseline_n} historical runs"}

        recent = history[-baseline_n:]
        flags = {}

        for metric in ("avg_sleep_latency", "avg_trad_latency"):
            hist_values = [r.get(metric) for r in recent if r.get(metric) is not None]
            if not hist_values:
                continue
            hist_avg = sum(hist_values) / len(hist_values)
            current = current_metrics.get(metric)
            if current is not None and hist_avg > 0 and current / hist_avg > latency_threshold:
                flags[metric] = {
                    "current": current,
                    "historical_avg": round(hist_avg, 4),
                    "ratio": round(current / hist_avg, 2),
                    "regression": True,
                }

        for metric in ("total_sleep_cost", "total_trad_cost"):
            hist_values = [r.get(metric) for r in recent if r.get(metric) is not None]
            if not hist_values:
                continue
            hist_avg = sum(hist_values) / len(hist_values)
            current = current_metrics.get(metric)
            if current is not None and hist_avg > 0 and current / hist_avg > cost_threshold:
                flags[metric] = {
                    "current": current,
                    "historical_avg": round(hist_avg, 6),
                    "ratio": round(current / hist_avg, 2),
                    "regression": True,
                }

        return {
            "checked": True,
            "baseline_runs": baseline_n,
            "regressions_found": len(flags),
            "details": flags,
        }
