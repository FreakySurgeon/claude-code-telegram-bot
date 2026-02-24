"""Structured metrics logging for Claude runs."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

METRICS_FILE = Path(__file__).parent.parent.parent / "logs" / "metrics.jsonl"


def write_metric(
    *,
    source: str,
    run_type: str,
    model: str | None,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float | None,
    num_turns: int,
    duration_s: float,
    duration_api_ms: int,
    status: str,
    session_id: str | None,
) -> None:
    """Append a metric line to the JSONL log."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "type": run_type,
        "model": model or "default",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
        "num_turns": num_turns,
        "duration_s": round(duration_s, 1),
        "duration_api_ms": duration_api_ms,
        "status": status,
        "session_id": session_id,
    }
    try:
        METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with METRICS_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        logger.warning("Failed to write metric", exc_info=True)
